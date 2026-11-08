"""Deterministic source qualification and atomic adoption, separate from QA grants.

Offline trusted inputs remain required for independent requalification. They are
not a declaration that the online runtime can already use cloud object storage.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import zlib
from collections import Counter
from datetime import UTC
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert

from citequest.source_refs_v2 import PaperSourceReference
from citequest.updates_v2.models import SourceDenial
from citequest.v2 import models as core
from citequest.v2.domain import DomainError
from citequest.v2.services import digest, normalize_identifier
from citequest.v2.storage import CAS, StoredBlob
from citequest.v2.storage_config import app_storage

from . import models as m
from .application_contract import (
    ADOPTION,
    CHAIN_PATH,
    EVALUATION,
    POLICY,
    RIGHTS_PATH,
    SNAPSHOT,
    ApplicationFreezeInput,
    original_inventory,
    profile,
    read_pinned,
)
from .application_members import RECORD_FIELDS, ApplicationMember, membership_records, record_sha, records_sha
from .landing import Landing
from .native_codec import NativeRepresentationDraft, NativeTextLocator
from .native_identity import CHAIN_SHA, NATIVE_PROFILE_SHA, POLICY_SHA, RIGHTS_SHA
from .native_index import NativeRepresentationPointer, read_for_index
from .native_text import audit_text
from .resolution import Resolver
from .standard_seal import REVISION, SOURCE_MANIFEST


def implementation(root: Path) -> dict:
    """Bind the actual qualification implementation and library versions."""
    paths = sorted(
        {
            *root.glob("src/citequest/data_v2/*.py"),
            *root.glob("src/citequest/v2/*.py"),
            root / "src/citequest/updates_v2/models.py",
            root / "src/citequest/source_refs_v2.py",
            root / "scripts/s05_data/freeze_application.py",
            root / "migrations/app/versions/s05_data_0003_application_freeze.py",
            root / "migrations/app/env.py",
        }
    )
    files = [
        {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths
    ]
    return {
        "files": files,
        "files_sha256": digest(files),
        "runtime": {
            name: importlib.metadata.version(name) for name in ("sqlalchemy", "alembic", "psycopg", "pydantic")
        },
    }


def current_denials(
    landing: Landing, store: CAS, row: m.SourceRow, source: core.SourceRecord, doi: str | None
) -> list[dict]:
    app_storage().assert_store(store)
    if row.security_domain not in {"public", landing.catalog.actor.security_domain}:
        raise DomainError("source_scope_not_found", 404)
    lineages = [
        ("litsearch_s2orc", row.native_id, REVISION),
        ("litsearch", row.native_id, REVISION),
        ("acl_anthology", source.native_id, source.source_revision),
    ]
    if doi:
        lineages.append(("crossref", normalize_identifier("doi", doi), None))
    # Read the same monotone shared table as UpdateReader. This boundary needs
    # no XML/HTTP parser dependencies and never infers a whole-Work denial.
    found = []
    for namespace, native_id, revision in lineages:
        for denial in landing.db.scalars(
            select(SourceDenial).where(
                SourceDenial.security_domain == row.security_domain,
                SourceDenial.source_namespace == namespace,
                SourceDenial.native_id == native_id,
                SourceDenial.active.is_(True),
            )
        ):
            if denial.source_revision == "*" or denial.source_revision == revision:
                found.append(
                    {
                        "source_namespace": namespace,
                        "native_id": native_id,
                        "source_revision": denial.source_revision,
                        "reason": denial.reason,
                        "observation_id": str(denial.observation_id),
                    }
                )
    return found


def _snapshot(landing: Landing, *, lock: bool = False) -> m.SourceSnapshot:
    snapshot = landing.snapshot(SNAPSHOT)
    if lock:
        landing.catalog.lock_owner()
        snapshot = landing.db.get(m.SourceSnapshot, SNAPSHOT, with_for_update=True, populate_existing=True)
    # The internal publisher may reuse a Session for retry. Refresh its current
    # persisted capability before consulting the shared Catalog guard.
    landing.db.get(core.Owner, landing.catalog.actor.owner_id, populate_existing=True)
    landing.catalog.require_publisher()
    if (
        snapshot.importer_owner_id != landing.catalog.actor.owner_id
        or snapshot.security_domain != "public"
        or snapshot.source_namespace != "litsearch_s2orc"
        or snapshot.source_revision != REVISION
        or snapshot.source_manifest_sha256 != SOURCE_MANIFEST
        or snapshot.representation_profile != "litsearch-s2orc-native-text/1"
        or snapshot.purpose != "application"
        or snapshot.status == "disabled"
        or snapshot.policy_sha256 != POLICY
        or snapshot.evaluation_sha256 != EVALUATION
        or landing.policy_sha256 != POLICY
        or landing.evaluation_sha256 != EVALUATION
    ):
        raise DomainError("application_snapshot_not_adoptable", 409)
    return snapshot


def _blob(landing: Landing, store: CAS, domain: str, sha: str, maximum: int) -> bytes:
    blob = landing.db.get(core.Blob, (domain, sha), populate_existing=True)
    if blob is None or not 0 < blob.size_bytes <= maximum:
        raise DomainError("application_blob_missing_or_oversize", 422)
    return store.read(StoredBlob(domain, sha, blob.size_bytes))


def source_record_sha(source: core.SourceRecord) -> str:
    facts = {
        key: getattr(source, key)
        for key in (
            "source",
            "native_id",
            "source_revision",
            "raw_sha256",
            "raw_security_domain",
            "security_domain",
            "rights_status",
            "license",
            "source_url",
        )
    }
    facts.update(work_id=str(source.work_id), retrieved_at=source.retrieved_at.astimezone(UTC).isoformat())
    return digest(facts)


def qualify_member(
    landing: Landing, store: CAS, root: Path, snapshot: m.SourceSnapshot, candidate: dict, rights: dict
) -> dict:
    """Replay actual evidence; a prior candidate label is never sufficient."""
    native = str(candidate["native_id"])
    row = landing.db.scalar(
        select(m.SourceRow).where(m.SourceRow.snapshot_id == snapshot.id, m.SourceRow.native_id == native)
    )
    if row is None:
        raise DomainError("application_candidate_row_missing", 422)
    resolved = Resolver(landing).latest(row.id)
    if resolved is None:
        raise DomainError("application_candidate_resolution_missing", 422)
    projection = read_for_index(landing, store, candidate["representation_id"])
    # Both materialized and pinned producer bytes must describe the same body.
    representation = landing.db.get(m.NativeRepresentation, candidate["representation_id"])
    raw_draft = _blob(landing, store, representation.storage_security_domain, representation.draft_sha256, 16 * 1024**2)
    producer = read_pinned(
        root, candidate["representation_path"], candidate["representation_json_sha256"], 16 * 1024**2
    )
    if raw_draft != producer:
        raise DomainError("application_native_producer_CAS_mismatch", 422)
    draft = NativeRepresentationDraft.model_validate_json(raw_draft)
    technical = audit_text(draft.literal_text, draft.original_annotation_strings)
    if (
        draft.annotation_errors
        or draft.profile_sha256 != NATIVE_PROFILE_SHA
        or technical["technical_status"] != "source_fulltext_candidate"
    ):
        raise DomainError("application_native_technical_rejected:" + ",".join(technical.get("reasons", [])), 422)
    # Exercise every decoded source annotation against its original literal,
    # including less common annotation families, not only paragraph examples.
    # validate_identity already ran at strict decoding. Construct the bounded
    # batch directly; read_many validates the full draft before/after all reads.
    # Calling locator() for every span would repeatedly decode all annotations.
    locators = [
        NativeTextLocator(
            source_snapshot_sha256=draft.source_pointer.source_snapshot_sha256,
            source_pointer_sha256=draft.source_pointer_sha256,
            source_text_sha256=draft.source_text_sha256,
            char_start=start,
            char_end=end,
        )
        for spans in draft.annotation_spans.values()
        for start, end in spans
    ]
    recovered = draft.read_many(locators)
    if any(not literal.strip() for literal in recovered):
        raise DomainError("application_locator_empty_literal", 422)
    version = landing.catalog.version(projection.work_version_id)
    work = landing.catalog.work(projection.work_id)
    source = landing.db.get(core.SourceRecord, UUID(resolved.proof["source_record_id"]), populate_existing=True)
    attribution = rights["attribution"]
    if (
        rights["rank"] != candidate["rank"]
        or rights["native_id"] != candidate["native_id"]
        or rights["rights_state"] != "verified_project_text_indexing_attribution"
        or rights["license_id"] != "CC-BY-4.0"
        or rights["copyright_policy_sha256"] != POLICY_SHA
        or rights["publisher"] != "Association for Computational Linguistics"
        or int(rights["official_year"]) < 2016
        or rights["attribution"] != candidate["attribution"]
        or rights["source_text_sha256"] != draft.source_text_sha256
        or source is None
        or source.work_id != work.id
        or version.work_id != work.id
        or source.raw_sha256 != candidate["PDF_sha256"]
        or version.source_sha256 != source.raw_sha256
        or source.rights_status != "verified"
        or source.license != "CC-BY-4.0"
        or not work.public
        or work.canonical_redirect_work_id is not None
    ):
        raise DomainError("application_rights_or_current_work_conflict", 409)
    pdf = _blob(landing, store, source.raw_security_domain, source.raw_sha256, 50 * 1024**2)
    if (
        hashlib.sha1(pdf).hexdigest() != rights["native_SHA1"]
        or f"{zlib.crc32(pdf):08x}" != rights["official_metadata_CRC32"]
        or not pdf.startswith(b"%PDF-")
    ):
        raise DomainError("application_original_PDF_identity_mismatch", 422)
    # Receipts are fixed local provenance, never a new upstream HTTP request.
    pdf_path = root / rights["pdf_path"]
    if read_pinned(root, rights["pdf_path"], source.raw_sha256, 50 * 1024**2) != pdf:
        raise DomainError("application_original_PDF_CAS_mismatch", 422)
    receipts = []
    for path in sorted(pdf_path.parent.glob("HTTP-*.json")):
        raw = path.read_bytes()
        receipt = json.loads(raw)
        if (
            receipt.get("http_status") == 200
            and receipt.get("received_body_bytes") == len(pdf)
            and receipt.get("usage_complete_for_application_body_accounting") is True
            and receipt.get("SHA256") == source.raw_sha256
            and receipt.get("bytes") == len(pdf)
            and receipt.get("MIME", "").split(";")[0].strip().lower() == "application/pdf"
            and receipt.get("CRC32") == rights["official_metadata_CRC32"]
            and receipt.get("url") == source.source_url
            and source.source_url.startswith("https://aclanthology.org/")
        ):
            receipts.append({"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()})
    if not receipts:
        raise DomainError("application_anonymous_official_download_provenance_missing", 422)
    strong = [("litsearch_corpus_id", native)]
    for identifier in candidate["work_input_draft"]["identifiers"]:
        normalized = normalize_identifier(identifier["namespace"], identifier["raw"])
        actual = landing.db.get(
            core.Identifier, ("public", identifier["namespace"], normalized), populate_existing=True
        )
        if actual is None or actual.work_id != work.id:
            raise DomainError("application_strong_identifier_mapping_conflict", 409)
        if (identifier["namespace"], normalized) not in strong:
            strong.append((identifier["namespace"], normalized))
    doi = attribution.get("doi")
    if current_denials(landing, store, row, source, doi):
        raise DomainError("application_current_source_denial", 409)
    event = landing.db.scalar(
        select(core.Outbox).where(core.Outbox.idempotency_key == f"NativeRepresentation:{representation.id}:1:stage")
    )
    if event is None or event.payload_sha256 != digest(event.payload):
        raise DomainError("application_native_outbox_proof_missing", 422)
    pointer = NativeRepresentationPointer.model_validate_json(json.dumps(event.payload["data"]))
    expected_pointer = NativeRepresentationPointer(
        representation_id=representation.id,
        source_row_id=row.id,
        snapshot_id=snapshot.id,
        security_domain=representation.security_domain,
        index_projection_id=representation.index_projection_id,
        source_text_sha256=representation.source_text_sha256,
        profile=representation.profile,
        profile_sha256=representation.profile_sha256,
        storage_security_domain=representation.storage_security_domain,
        draft_sha256=representation.draft_sha256,
        source_reference=projection.source_reference,
    )
    if pointer != expected_pointer:
        raise DomainError("application_native_outbox_PaperRef_mismatch", 422)
    return {
        "rank": candidate["rank"],
        "native_id": native,
        "source_row_id": str(row.id),
        "resolution_id": str(resolved.id),
        "resolution_revision": resolved.revision,
        "resolution_proof_sha256": resolved.proof_sha256,
        "work_id": str(work.id),
        "work_version_id": str(version.id),
        "source_record_id": str(source.id),
        "representation_id": representation.id,
        "draft_sha256": representation.draft_sha256,
        "source_text_sha256": draft.source_text_sha256,
        "source_pointer_sha256": row.source_pointer_sha256,
        "codec_profile_sha256": draft.profile_sha256,
        "annotation_values_sha256": draft.annotation_values_sha256,
        "source_reference": projection.source_reference.model_dump(mode="json"),
        "raw_PDF_sha256": source.raw_sha256,
        "raw_PDF_security_domain": source.raw_security_domain,
        "native_blob_security_domain": representation.storage_security_domain,
        "source_record_sha256": source_record_sha(source),
        "strong_identities": [[namespace, value] for namespace, value in sorted(strong)],
        "publisher_identity": [source.source, source.native_id],
        "doi": doi,
        "license_id": rights["license_id"],
        "license_url": rights["license_url"],
        "attribution": attribution,
        "rights_row_sha256": digest(rights),
        "source_chain_row_sha256": digest(candidate),
        "anonymous_download_receipts": receipts,
        "annotation_locator_count": len(locators),
        "locator_literals_sha256": digest(recovered),
        "characters": len(draft.literal_text),
        "utf8_bytes": len(draft.original_utf8()),
        "technical_audit_sha256": digest(technical),
        "source_text_coverage": "entire_native_literal_text",
        "coordinate_system": "unicode_codepoint",
        "original_PDF_complete_extraction_claimed": False,
        "third_party_figure_rights_claimed": False,
    }


def qualify_application(landing: Landing, store: CAS, root: Path) -> dict:
    """Read-only evaluation of every original position and every source candidate."""
    root = root.resolve(strict=True)
    app_storage(root).assert_store(store)
    contract, profile_sha = profile(root)
    snapshot = _snapshot(landing)
    chain = original_inventory(read_pinned(root, CHAIN_PATH, CHAIN_SHA, 32 * 1024**2))
    rights = original_inventory(read_pinned(root, RIGHTS_PATH, RIGHTS_SHA, 32 * 1024**2))
    policy_path = (
        "data/s05/acl-source/98d89b457f355afa93ba6659b8132ea5ac46ed5d/copyright-policy-" + POLICY_SHA + ".html"
    )
    read_pinned(root, policy_path, POLICY_SHA, 1024**2)
    candidates = [row for row in chain if row.get("component_candidate") is True]
    if len(candidates) != contract.pinned_component_candidates:
        raise DomainError("application_pinned_candidate_inventory_changed", 422)
    known = {str(row["native_id"]) for row in candidates}
    actual = set(landing.db.scalars(select(m.SourceRow.native_id).where(m.SourceRow.snapshot_id == SNAPSHOT)))
    if actual - known:
        raise DomainError("application_rows_outside_original_candidate_inventory", 409)
    members, outcomes, seen = [], [], {}
    assessed = 0
    for candidate, original_rights in zip(chain, rights, strict=True):
        if candidate["native_id"] != original_rights["native_id"]:
            raise DomainError("application_original_rights_order_changed", 422)
        outcome = {
            "rank": candidate["rank"],
            "native_id": str(candidate["native_id"]),
            "original_source_chain_row_sha256": digest(candidate),
            "original_rights_row_sha256": digest(original_rights),
            "original_reason": candidate.get("reason"),
            "original_rights_state": original_rights["rights_state"],
            "original_codec_state": candidate.get("codec_state"),
            "component_candidate": candidate.get("component_candidate") is True,
            "qualified": False,
            "reasons": [],
        }
        if outcome["component_candidate"]:
            assessed += 1
            try:
                member = qualify_member(landing, store, root, snapshot, candidate, original_rights)
                unique = [
                    ("work", member["work_id"]),
                    ("pdf", member["raw_PDF_sha256"]),
                    ("publisher", digest(member["publisher_identity"])),
                ]
                unique += [(namespace, value) for namespace, value in member["strong_identities"]]
                duplicates = sorted({seen[key] for key in unique if key in seen})
                if duplicates:
                    outcome["reasons"] = ["duplicate_qualified_strong_identity"]
                    outcome["duplicate_prior_ranks"] = duplicates
                else:
                    seen.update(dict.fromkeys(unique, candidate["rank"]))
                    members.append(member)
                    outcome.update(qualified=True, member_sha256=digest(member))
            except (DomainError, ValueError, OSError, KeyError, TypeError) as exc:
                outcome["reasons"] = [
                    exc.code if isinstance(exc, DomainError) else type(exc).__name__ + ":" + str(exc)[:250]
                ]
        else:
            outcome["reasons"] = ["original_source_candidate_excluded:" + str(candidate.get("reason"))]
        outcomes.append(outcome)
    files = landing.db.scalar(
        select(func.count()).select_from(core.File).where(core.File.owner_id == snapshot.importer_owner_id)
    )
    if files:
        raise DomainError("application_importer_has_synthetic_or_unexpected_File", 409)
    coverage = {
        "original_positions": len(outcomes),
        "component_candidates_assessed": assessed,
        "qualified": len(members),
        "excluded": len(outcomes) - len(members),
        "candidate_rejected": assessed - len(members),
        "license_counts": dict(Counter(m["license_id"] for m in members)),
        "original_rights_counts": dict(Counter(o["original_rights_state"] for o in outcomes)),
        "final_reason_counts": dict(Counter(reason for o in outcomes for reason in o["reasons"])),
        "fallback_audited": 0,
        "source_text_coverage": "entire_native_literal_text",
        "original_PDF_complete_extraction_claimed": False,
        "third_party_figure_rights_claimed": False,
        "importer_File_count": files,
        "freeze_grants_qa_or_stage_closure": False,
    }
    result = {
        "schema_version": "cq.application-qualification/1",
        "qualification_implementation": implementation(root),
        "snapshot_id": str(snapshot.id),
        "source_manifest_sha256": SOURCE_MANIFEST,
        "source_revision": REVISION,
        "profile_sha256": profile_sha,
        "original_policy_sha256": POLICY,
        "evaluation_sha256": EVALUATION,
        "stop_rule_adoption_sha256": ADOPTION,
        "source_chain_manifest_sha256": CHAIN_SHA,
        "rights_manifest_sha256": RIGHTS_SHA,
        "copyright_policy_sha256": POLICY_SHA,
        "codec_profile_sha256": NATIVE_PROFILE_SHA,
        "members": members,
        "membership_sha256": digest(members),
        "outcomes": outcomes,
        "outcomes_sha256": digest(outcomes),
        "coverage": coverage,
        "coverage_sha256": digest(coverage),
        "eligible_to_freeze": contract.minimum_qualified <= len(members) <= contract.maximum_qualified,
        "all_original_positions_preserved": len(outcomes) == 3000,
        "all_source_candidates_assessed": assessed == len(candidates),
    }
    return result


def freeze_application(landing: Landing, store: CAS, root: Path, request: ApplicationFreezeInput) -> dict:
    """Caller owns one transaction; requalification always precedes receipt replay."""
    app_storage(root).assert_store(store)
    snapshot = _snapshot(landing, lock=True)
    # Current ACL/denial changes must serialize with adoption, including writes
    # by other server instances. This initial, offline adoption holds a global
    # write barrier during requalification. It is not an online update path.
    landing.db.execute(
        text(
            "LOCK TABLE cq_app.works, cq_app.work_access, cq_app.source_update_denials, "
            "cq_app.source_records, cq_app.source_identities, cq_app.work_identifiers, "
            "cq_app.work_versions, cq_app.blobs "
            "IN SHARE MODE"
        )
    )
    actual = qualify_application(landing, store, root)
    if (
        actual["profile_sha256"] != request.expected_profile_sha256
        or digest(actual) != request.expected_qualification_sha256
    ):
        raise DomainError("application_qualification_precondition_changed", 409)
    if (
        not actual["eligible_to_freeze"]
        or not actual["all_original_positions_preserved"]
        or not actual["all_source_candidates_assessed"]
    ):
        raise DomainError("application_minimum_or_complete_audit_not_met", 409)
    records = membership_records(snapshot.id, actual["members"])
    audit_raw = json.dumps(actual, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    audit_sha = hashlib.sha256(audit_raw).hexdigest()
    seal = compact_seal(actual, snapshot.security_domain, len(audit_raw), records)
    seal_sha = digest(seal)
    payload = {
        "operation": "freeze_application_source",
        "snapshot_id": str(snapshot.id),
        "request": request.model_dump(mode="json"),
    }

    def effect() -> dict:
        audit = store.put(snapshot.security_domain, io.BytesIO(audit_raw), max_bytes=32 * 1024**2)
        if audit.sha256 != audit_sha:
            raise DomainError("application_audit_CAS_identity_conflict", 409)
        existing_blob = landing.db.get(core.Blob, (audit.security_domain, audit.sha256), populate_existing=True)
        if existing_blob is None:
            landing.db.add(
                core.Blob(security_domain=audit.security_domain, sha256=audit.sha256, size_bytes=audit.size_bytes)
            )
        elif existing_blob.size_bytes != audit.size_bytes:
            raise DomainError("application_audit_blob_record_conflict", 409)
        if snapshot.status == "landing":
            landing.db.execute(
                insert(ApplicationMember)
                .values(records)
                .on_conflict_do_nothing(index_elements=["snapshot_id", "source_row_id"])
            )
        stored = list(
            landing.db.scalars(
                select(ApplicationMember)
                .where(ApplicationMember.snapshot_id == snapshot.id)
                .order_by(ApplicationMember.rank)
            )
        )
        if len(stored) != len(records) or any(
            any(getattr(value, name) != expected[name] for name in expected)
            for value, expected in zip(stored, records, strict=True)
        ):
            raise DomainError("application_materialized_membership_conflict", 409)
        if snapshot.status == "staged":
            if not snapshot.profile_adopted or snapshot.seal != seal or snapshot.seal_sha256 != seal_sha:
                raise DomainError("application_snapshot_seal_conflict", 409)
        elif snapshot.status == "landing" and not snapshot.profile_adopted and snapshot.seal is None:
            snapshot.status, snapshot.profile_adopted = "staged", True
            snapshot.seal, snapshot.seal_sha256 = seal, seal_sha
        else:
            raise DomainError("application_snapshot_not_adoptable", 409)
        landing.db.flush()
        envelope = {
            "schema_version": "cq.outbox.payload/1",
            "entity_type": "SourceSnapshot",
            "entity_revision": 1,
            "entity_id": str(snapshot.id),
            "security_domain": snapshot.security_domain,
            "operation": "stage",
            "entity_schema": "cq.application-source-seal/2",
            "data": {"status": "staged", "profile_adopted": True, "seal_sha256": seal_sha, "seal": seal},
        }
        key = f"SourceSnapshot:{snapshot.id}:1:stage"
        landing.db.execute(
            insert(core.Outbox)
            .values(
                idempotency_key=key,
                entity_id=str(snapshot.id),
                entity_revision=1,
                operation="stage",
                entity_type="SourceSnapshot",
                security_domain=snapshot.security_domain,
                schema_version="cq.outbox.payload/1",
                payload=envelope,
                payload_sha256=digest(envelope),
            )
            .on_conflict_do_nothing(index_elements=["idempotency_key"])
        )
        event = landing.db.scalar(select(core.Outbox).where(core.Outbox.idempotency_key == key))
        if event is None or event.payload_sha256 != digest(envelope):
            raise DomainError("application_seal_outbox_conflict", 409)
        return {
            "snapshot_id": str(snapshot.id),
            "seal_sha256": seal_sha,
            "qualified": len(actual["members"]),
            "membership_sha256": actual["membership_sha256"],
            "profile_adopted": True,
            "index_published": False,
            "QA_granted": False,
            "S05_or_S09_gate_passed": False,
        }

    return landing.catalog.operation(request.client_operation_id, payload, effect)


def compact_seal(actual: dict, domain: str, audit_size: int, records: list[dict]) -> dict:
    """Constant-size serving authority; full 3000-position evidence is in CAS."""
    seal = {
        name: actual[name]
        for name in (
            "snapshot_id",
            "source_manifest_sha256",
            "source_revision",
            "profile_sha256",
            "original_policy_sha256",
            "evaluation_sha256",
            "stop_rule_adoption_sha256",
            "source_chain_manifest_sha256",
            "rights_manifest_sha256",
            "copyright_policy_sha256",
            "codec_profile_sha256",
            "membership_sha256",
            "outcomes_sha256",
            "coverage",
            "coverage_sha256",
            "eligible_to_freeze",
            "all_original_positions_preserved",
            "all_source_candidates_assessed",
        )
    }
    seal.update(
        schema_version="cq.application-source-seal/2",
        qualification_sha256=digest(actual),
        membership_records_sha256=records_sha(records),
        membership_table="cq_app.application_corpus_members",
        qualification_implementation_sha256=digest(actual["qualification_implementation"]),
        qualification_runtime=actual["qualification_implementation"]["runtime"],
        audit_blob={
            "security_domain": domain,
            "sha256": digest(actual),
            "size_bytes": audit_size,
            "schema_version": "cq.application-qualification/1",
        },
        membership_immutable=True,
        runtime_cloud_ready_claimed=False,
        index_published=False,
        QA_granted=False,
        S05_or_S09_gate_passed=False,
    )
    return seal


def read_application_audit(landing: Landing, store: CAS, snapshot: m.SourceSnapshot) -> dict:
    app_storage().assert_store(store)
    seal = snapshot.seal
    if not isinstance(seal, dict) or digest(seal) != snapshot.seal_sha256:
        raise DomainError("application_adopted_seal_invalid", 409)
    pointer = seal["audit_blob"]
    raw = _blob(landing, store, pointer["security_domain"], pointer["sha256"], 32 * 1024**2)
    if len(raw) != pointer["size_bytes"] or hashlib.sha256(raw).hexdigest() != seal["qualification_sha256"]:
        raise DomainError("application_qualification_audit_blob_invalid", 409)
    actual = json.loads(raw)
    records = membership_records(snapshot.id, actual["members"])
    if (
        digest(actual) != seal["qualification_sha256"]
        or digest(actual["members"]) != seal["membership_sha256"]
        or digest(actual["outcomes"]) != seal["outcomes_sha256"]
        or records_sha(records) != seal["membership_records_sha256"]
    ):
        raise DomainError("application_qualification_audit_membership_invalid", 409)
    return actual


def sealed_resolution(
    landing: Landing, snapshot: m.SourceSnapshot, row: m.SourceRow, representation_id: str
) -> m.SourceResolution:
    """Use exactly the adopted PaperRef; current ACL/denials still override it."""
    seal = snapshot.seal
    if (
        not isinstance(seal, dict)
        or seal.get("schema_version") != "cq.application-source-seal/2"
        or digest(seal) != snapshot.seal_sha256
        or seal.get("membership_table") != "cq_app.application_corpus_members"
    ):
        raise DomainError("application_adopted_seal_invalid", 409)
    pin = landing.db.get(ApplicationMember, (snapshot.id, row.id), populate_existing=True)
    if pin is None or pin.representation_id != representation_id:
        raise DomainError("native_representation_not_in_frozen_application", 404)
    if record_sha({name: getattr(pin, name) for name in RECORD_FIELDS}) != pin.record_sha256:
        raise DomainError("application_indexed_membership_hash_invalid", 409)
    resolved = landing.db.get(m.SourceResolution, pin.resolution_id, populate_existing=True)
    if (
        resolved is None
        or resolved.source_row_id != row.id
        or resolved.revision != pin.resolution_revision
        or resolved.proof_sha256 != pin.resolution_proof_sha256
        or resolved.work_id != pin.work_id
        or resolved.work_version_id != pin.work_version_id
        or resolved.representation_id != representation_id
    ):
        raise DomainError("application_sealed_resolution_pin_conflict", 409)
    source = landing.db.get(core.SourceRecord, pin.source_record_id, populate_existing=True)
    if source is None or source_record_sha(source) != pin.source_record_sha256:
        raise DomainError("application_sealed_SourceRecord_conflict", 409)
    expected = PaperSourceReference.model_validate_json(json.dumps(pin.source_reference))
    actual = PaperSourceReference.model_validate_json(json.dumps(resolved.proof["source_reference"]))
    if actual != expected:
        raise DomainError("application_sealed_PaperRef_conflict", 409)
    return resolved
