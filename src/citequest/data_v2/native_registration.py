"""Trusted source-chain registration; paper identity never implies application adoption."""

from __future__ import annotations

import hashlib
import io
import json
import unicodedata
import zlib
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid5

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from citequest.source_refs_v2 import PaperSourceReference
from citequest.v2 import models as core
from citequest.v2.domain import DomainError, SourceInput, WorkInput
from citequest.v2.presentation import work as present_work
from citequest.v2.services import digest
from citequest.v2.storage import CAS
from citequest.v2.storage_config import app_storage

from . import models as m
from .canonical import CanonicalResolver
from .landing import Landing
from .native_codec import NativeRepresentationDraft
from .native_identity import CHAIN_SHA, NATIVE_PROFILE_SHA, POLICY_SHA, RIGHTS_SHA
from .native_index import NativeRepresentationPointer
from .resolution import Resolver
from .standard_seal import SOURCE_MANIFEST


class NativePaperRegistrar:
    """Internal pinned-reader boundary. No HTTP/MCP endpoint accepts this proof.

    Reads only the already-audited manifest. An uploaded JSON, self-reported
    licence or LLM-proposed Work cannot invoke article registration through it.
    """

    def __init__(self, landing: Landing, store: CAS, workspace: Path) -> None:
        self.landing, self.catalog, self.db, self.store = landing, landing.catalog, landing.db, store
        self.root = workspace.resolve()
        app_storage(self.root).assert_store(store)
        base = self.root / "reports/stages/v2/S05/rights-preparation/pdf-rights-20261104T222257393069Z"
        chain = self.read(base / "source-chain-reconciliation/outcomes.jsonl", CHAIN_SHA, 32 * 1024**2)
        rights = self.read(base / "rights-outcomes.jsonl", RIGHTS_SHA, 32 * 1024**2)
        self.candidates = {
            row["rank"]: row for row in map(json.loads, chain.splitlines()) if row["component_candidate"]
        }
        self.rights = {row["rank"]: row for row in map(json.loads, rights.splitlines())}
        policy = (
            self.root
            / "data/s05/acl-source/98d89b457f355afa93ba6659b8132ea5ac46ed5d"
            / ("copyright-policy-" + POLICY_SHA + ".html")
        )
        self.read(policy, POLICY_SHA, 1024**2)
        if len(self.candidates) != 692 or len(self.rights) != 3000:
            raise DomainError("source_chain_component_inventory_changed", 422)

    def read(self, path: Path, expected: str, maximum: int) -> bytes:
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise DomainError("source_chain_path_escape", 422)
        with path.open("rb") as stream:
            raw = stream.read(maximum + 1)
        if len(raw) > maximum or hashlib.sha256(raw).hexdigest() != expected:
            raise DomainError("source_chain_bytes_changed", 422)
        return raw

    def register(self, snapshot_id: UUID, rank: int) -> dict:
        if rank not in self.candidates:
            raise DomainError("not_an_audited_unique_source_candidate", 422)
        candidate, rights = self.candidates[rank], self.rights[rank]
        snapshot = self.landing.snapshot(snapshot_id, writing=True)
        if (
            snapshot.security_domain != "public"
            or snapshot.source_namespace != "litsearch_s2orc"
            or snapshot.purpose != "application"
            or snapshot.representation_profile != "litsearch-s2orc-native-text/1"
            or snapshot.source_manifest_sha256 != SOURCE_MANIFEST
            or snapshot.profile_adopted
            or snapshot.importer_owner_id != self.catalog.actor.owner_id
        ):
            raise DomainError("native_candidate_snapshot_scope_mismatch", 409)
        draft_bytes = self.read(
            self.root / candidate["representation_path"], candidate["representation_json_sha256"], 16 * 1024**2
        )
        draft = NativeRepresentationDraft.model_validate_json(draft_bytes)
        native = str(candidate["native_id"])
        if (
            draft.profile_sha256 != NATIVE_PROFILE_SHA
            or draft.annotation_errors
            or draft.source_pointer.native_id != native
            or draft.source_pointer.source_revision != snapshot.source_revision
            or draft.source_pointer.source_snapshot_sha256 != SOURCE_MANIFEST
            or draft.source_text_sha256 != candidate["source_text_sha256"]
            or rights["source_text_sha256"] != draft.source_text_sha256
            or rights["rights_state"] != "verified_project_text_indexing_attribution"
            or rights["license_id"] != "CC-BY-4.0"
            or rights["copyright_policy_sha256"] != POLICY_SHA
            or rights["publisher"] != "Association for Computational Linguistics"
            or int(rights["official_year"]) < 2016
        ):
            raise DomainError("native_source_identity_or_rights_not_verified", 422)
        pdf_path = self.root / rights["pdf_path"]
        pdf = self.read(pdf_path, rights["PDF_sha256"], 50 * 1024**2)
        if (
            hashlib.sha1(pdf).hexdigest() != rights["native_SHA1"]
            or f"{zlib.crc32(pdf):08x}" != rights["official_metadata_CRC32"]
        ):
            raise DomainError("native_publisher_PDF_legacy_identity_changed", 422)
        http_receipts = [json.loads(path.read_bytes()) for path in sorted(pdf_path.parent.glob("HTTP-*.json"))]
        completed = [
            item
            for item in http_receipts
            if item.get("http_status") == 200
            and item.get("received_body_bytes") == len(pdf)
            and item.get("usage_complete_for_application_body_accounting") is True
        ]
        if not completed:
            raise DomainError("official_PDF_download_provenance_missing", 422)
        downloaded = completed[-1]
        if not downloaded["url"].startswith("https://aclanthology.org/"):
            raise DomainError("publisher_source_URL_outside_official_scope", 422)
        source_id = uuid5(snapshot_id, "native:" + native)
        existing = self.db.get(m.SourceRow, source_id)
        if existing is not None:
            resolved = Resolver(self.landing).latest(source_id)
            if resolved is not None and resolved.work_id is not None:
                self.catalog.work(resolved.work_id)
        operation_id = uuid5(snapshot_id, "native-chain:" + draft.representation_id)

        def effect() -> dict:
            location = draft.source_pointer.raw_location
            artifact = self.db.get(m.SourceArtifact, (snapshot_id, location.artifact_path))
            if (
                artifact is None
                or artifact.sha256 != location.artifact_sha256
                or artifact.row_group_sizes is None
                or location.row_group >= len(artifact.row_group_sizes)
                or location.row_in_group >= artifact.row_group_sizes[location.row_group]
                or location.row_ordinal != sum(artifact.row_group_sizes[: location.row_group]) + location.row_in_group
            ):
                raise DomainError("native_raw_row_pointer_mismatch", 422)
            raw_values = {
                "id": source_id,
                "snapshot_id": snapshot_id,
                "security_domain": snapshot.security_domain,
                "native_id": native,
                "native_id_source_type": "integer",
                **location.model_dump(),
                "source_pointer_sha256": draft.source_pointer_sha256,
            }
            self.db.execute(
                insert(m.SourceRow)
                .values(**raw_values)
                .on_conflict_do_nothing(index_elements=["snapshot_id", "native_id"])
            )
            row = self.db.get(m.SourceRow, source_id, populate_existing=True)
            if row is None or any(getattr(row, key) != value for key, value in raw_values.items()):
                raise DomainError("native_source_row_identity_conflict", 409)
            request = WorkInput.model_validate(candidate["work_input_draft"])
            obj = CanonicalResolver(self.catalog).resolve(
                "litsearch", native, request, public=snapshot.security_domain == "public"
            )
            before = present_work(self.catalog, obj, public_only=snapshot.security_domain == "public")
            self.catalog.record_source(
                obj.id,
                SourceInput(
                    source="acl_anthology",
                    native_id=rights["acl_id"],
                    source_revision=rights["attribution"]["source_commit"],
                    retrieved_at=datetime.fromisoformat(downloaded["fetched_at"]),
                    raw_sha256=rights["PDF_sha256"],
                    source_url=downloaded["url"],
                    rights_status="verified",
                    license="CC-BY-4.0",
                ),
                pdf,
                self.store,
                public=snapshot.security_domain == "public",
            )
            source_obj = self.db.scalar(
                select(core.SourceRecord).where(
                    core.SourceRecord.work_id == obj.id,
                    core.SourceRecord.security_domain == snapshot.security_domain,
                    core.SourceRecord.source == "acl_anthology",
                    core.SourceRecord.native_id == rights["acl_id"],
                    core.SourceRecord.source_revision == rights["attribution"]["source_commit"],
                    core.SourceRecord.raw_sha256 == rights["PDF_sha256"],
                )
            )
            if source_obj is None or source_obj.rights_status != "verified" or source_obj.license != "CC-BY-4.0":
                raise DomainError("registered_source_record_missing", 422)
            obj = self.db.get(core.Work, obj.id, with_for_update=True, populate_existing=True)
            field_source = {
                "source_record_id": str(source_obj.id),
                "source": "acl_anthology",
                "native_id": rights["acl_id"],
                "raw_sha256": rights["PDF_sha256"],
                "authority": "verified_publisher_metadata",
            }
            provenance = {**obj.field_provenance, **dict.fromkeys(("title", "authors", "year"), field_source)}
            if (
                obj.title != request.title
                or obj.authors != request.authors
                or obj.year != request.year
                or obj.field_provenance != provenance
            ):
                obj.title, obj.authors, obj.year = request.title, request.authors, request.year
                obj.title_normalized = unicodedata.normalize("NFKC", request.title).casefold()
                obj.field_provenance, obj.metadata_revision = provenance, obj.metadata_revision + 1
                self.db.flush()
                self.catalog.outbox(
                    obj.id,
                    obj.metadata_revision,
                    "upsert",
                    present_work(self.catalog, obj, public_only=snapshot.security_domain == "public"),
                    entity_type="Work",
                    security_domain=obj.security_domain,
                )
            version = self.catalog.record_version(
                obj.id,
                rights["PDF_sha256"],
                "published",
                "official-publisher-PDF",
                public=snapshot.security_domain == "public",
            )
            blob = self.store.put(snapshot.security_domain, io.BytesIO(draft_bytes), max_bytes=16 * 1024**2)
            if self.db.get(core.Blob, (blob.security_domain, blob.sha256)) is None:
                self.db.add(
                    core.Blob(security_domain=blob.security_domain, sha256=blob.sha256, size_bytes=blob.size_bytes)
                )
                self.db.flush()
            values = {
                "id": draft.representation_id,
                "source_row_id": source_id,
                "security_domain": snapshot.security_domain,
                "index_projection_id": uuid5(source_id, "native:" + draft.representation_id),
                "source_text_sha256": draft.source_text_sha256,
                "profile": draft.profile,
                "profile_sha256": draft.profile_sha256,
                "storage_security_domain": blob.security_domain,
                "draft_sha256": blob.sha256,
            }
            self.db.execute(
                insert(m.NativeRepresentation).values(**values).on_conflict_do_nothing(index_elements=["id"])
            )
            actual = self.db.get(m.NativeRepresentation, draft.representation_id)
            if actual is None or any(getattr(actual, key) != value for key, value in values.items()):
                raise DomainError("native_representation_identity_conflict", 409)
            reference = PaperSourceReference(
                source_kind="paper",
                work_version_id=UUID(version["version_id"]),
                source_snapshot_sha256=SOURCE_MANIFEST,
                representation_id=draft.representation_id,
            )
            result = Resolver(self.landing).append(
                row,
                "paper_identity_verified",
                {
                    "source_chain_manifest_sha256": CHAIN_SHA,
                    "rights_manifest_sha256": RIGHTS_SHA,
                    "copyright_policy_sha256": POLICY_SHA,
                    "source_record_id": str(source_obj.id),
                    "raw_PDF_sha256": rights["PDF_sha256"],
                    "legacy_SHA1_provenance_only": rights["native_SHA1"],
                    "official_metadata_CRC32": rights["official_metadata_CRC32"],
                    "source_reference": reference.model_dump(mode="json"),
                    "native_source_pointer_sha256": draft.source_pointer_sha256,
                    "whole_native_literal_preserved": True,
                    "canonical_metadata_before": before,
                    "canonical_metadata_after": present_work(
                        self.catalog, obj, public_only=snapshot.security_domain == "public"
                    ),
                    "original_PDF_full_text_coverage_claimed": False,
                    "rights_scope": "project article-text indexing with attribution",
                    "attribution": rights["attribution"],
                    "application_profile_adopted": False,
                    "formal_citation_ready": False,
                },
                work_id=obj.id,
                version_id=UUID(version["version_id"]),
                representation_id=draft.representation_id,
            )
            envelope = {
                "schema_version": "cq.outbox.payload/1",
                "entity_type": "NativeRepresentation",
                "entity_revision": 1,
                "entity_id": draft.representation_id,
                "security_domain": snapshot.security_domain,
                "operation": "stage",
                "entity_schema": "contracts/v2-data/native-index.schema.json#/$defs/NativeRepresentationPointer",
                "profile_adopted": False,
                "data": NativeRepresentationPointer(
                    representation_id=values["id"],
                    snapshot_id=snapshot.id,
                    **{key: value for key, value in values.items() if key != "id"},
                    source_reference=reference,
                ).model_dump(mode="json"),
            }
            self.db.execute(
                insert(core.Outbox)
                .values(
                    idempotency_key=f"NativeRepresentation:{draft.representation_id}:1:stage",
                    entity_id=draft.representation_id,
                    entity_revision=1,
                    operation="stage",
                    entity_type="NativeRepresentation",
                    schema_version="cq.outbox.payload/1",
                    security_domain=snapshot.security_domain,
                    payload=envelope,
                    payload_sha256=digest(envelope),
                )
                .on_conflict_do_nothing(index_elements=["idempotency_key"])
            )
            event = self.db.scalar(
                select(core.Outbox).where(
                    core.Outbox.idempotency_key == f"NativeRepresentation:{draft.representation_id}:1:stage"
                )
            )
            if event is None or event.payload_sha256 != digest(envelope):
                raise DomainError("native_outbox_pointer_conflict", 409)
            return result

        return self.catalog.operation(
            operation_id,
            {
                "operation": "register_audited_native_paper",
                "snapshot_id": str(snapshot_id),
                "rank": rank,
                "representation_id": draft.representation_id,
                "source_chain_manifest_sha256": CHAIN_SHA,
            },
            effect,
        )
