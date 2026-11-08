"""Close native identity against pinned source evidence and current database facts.

Publisher permission is necessary but is not evidence that a supplied Work is
the paper represented by a corpus row. Readers repeat these checks; they do not
trust a previously stored state label or a client-supplied proof dictionary.
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING
from uuid import UUID

from citequest.source_refs_v2 import PaperSourceReference
from citequest.v2 import models as core
from citequest.v2.domain import DomainError
from citequest.v2.services import digest
from citequest.v2.storage_config import app_storage

from . import models as m
from .standard_seal import SOURCE_MANIFEST

if TYPE_CHECKING:
    from .landing import Landing

CHAIN_SHA = "fd09876d6863e51f25f822b3bb03bf41ee3756314a22b952edfb3c5d3b309813"
RIGHTS_SHA = "e43140d1f1fc118d7692ae562cde983f3399ba05309cd4365a6a3b95647766d3"
POLICY_SHA = "0150e838394bae4268abc506fa5612f6ab3a6e7110b9718b52b4a591d754246f"
NATIVE_PROFILE_SHA = "3a95f20025481c7830ff75506df82024d0f90bf75d46530b020fce52d901ddc2"


def common_work(landing: Landing, row: m.SourceRow, snapshot: m.SourceSnapshot, work_id: UUID) -> None:
    if snapshot.source_namespace not in {"litsearch_clean", "litsearch_s2orc"}:
        raise DomainError("unsupported_resolution_source_namespace", 409)
    common = landing.db.get(core.SourceIdentity, (row.security_domain, "litsearch", row.native_id), populate_existing=True)
    if common is None or common.work_id != work_id:
        raise DomainError("resolution_common_source_identity_conflict", 409)
    landing.catalog.work(work_id)


def verify_native_identity(
    landing: Landing, row: m.SourceRow, snapshot: m.SourceSnapshot, *,
    work_id: UUID, version_id: UUID, representation_id: str, proof: dict,
) -> None:
    common_work(landing, row, snapshot, work_id)
    if (
        snapshot.security_domain != "public"
        or snapshot.source_namespace != "litsearch_s2orc"
        or snapshot.purpose != "application"
        or snapshot.representation_profile != "litsearch-s2orc-native-text/1"
        or snapshot.source_manifest_sha256 != SOURCE_MANIFEST
        or row.native_id_source_type != "integer"
        or not isinstance(proof, dict)
    ):
        raise DomainError("native_identity_source_scope_mismatch", 409)
    root = app_storage().workspace.resolve(strict=True)
    path = root / "reports/stages/v2/S05/rights-preparation/pdf-rights-20261104T222257393069Z/source-chain-reconciliation/outcomes.jsonl"
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise DomainError("native_identity_source_chain_symlink", 422)
    with path.open("rb") as stream:
        raw = stream.read(32 * 1024**2 + 1)
    if len(raw) > 32 * 1024**2 or hashlib.sha256(raw).hexdigest() != CHAIN_SHA:
        raise DomainError("native_identity_source_chain_changed", 422)
    candidates = [v for v in map(json.loads, raw.splitlines()) if v.get("component_candidate") and str(v["native_id"]) == row.native_id]
    if len(candidates) != 1:
        raise DomainError("native_identity_not_in_pinned_source_chain", 422)
    candidate = candidates[0]
    representation = landing.db.get(m.NativeRepresentation, representation_id, populate_existing=True)
    version = landing.catalog.version(version_id)
    try:
        source = landing.db.get(core.SourceRecord, UUID(proof["source_record_id"]), populate_existing=True)
        # Persisted JSON contains UUID strings; validate in JSON mode while
        # retaining the strict model's field and type rules.
        reference = PaperSourceReference.model_validate_json(json.dumps(proof["source_reference"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise DomainError("native_identity_proof_invalid", 422) from exc
    attribution = candidate["attribution"]
    expected_reference = PaperSourceReference(
        source_kind="paper", work_version_id=version_id,
        source_snapshot_sha256=SOURCE_MANIFEST, representation_id=representation_id,
    )
    if (
        representation is None
        or representation.source_row_id != row.id
        or representation.security_domain != row.security_domain
        or representation.storage_security_domain != row.security_domain
        or representation.id != candidate["representation_id"]
        or representation.draft_sha256 != candidate["representation_json_sha256"]
        or representation.source_text_sha256 != candidate["source_text_sha256"]
        or representation.profile_sha256 != NATIVE_PROFILE_SHA
        or row.source_pointer_sha256 != candidate["source_pointer_sha256"]
        or version.work_id != work_id
        or version.security_domain != row.security_domain
        or version.source_sha256 != candidate["PDF_sha256"]
        or source is None
        or source.work_id != work_id
        or source.security_domain != row.security_domain
        # Public citation metadata retains publisher-owned raw bytes in the
        # original importer domain; it does not publish the underlying PDF.
        or source.raw_security_domain != f"owner:{snapshot.importer_owner_id}"
        or source.source != "acl_anthology"
        or source.native_id != attribution["official_id"]
        or source.source_revision != attribution["source_commit"]
        or source.raw_sha256 != candidate["PDF_sha256"]
        or source.rights_status != "verified"
        or source.license != "CC-BY-4.0"
        or proof.get("source_chain_manifest_sha256") != CHAIN_SHA
        or proof.get("rights_manifest_sha256") != RIGHTS_SHA
        or proof.get("copyright_policy_sha256") != POLICY_SHA
        or proof.get("raw_PDF_sha256") != candidate["PDF_sha256"]
        or proof.get("native_source_pointer_sha256") != row.source_pointer_sha256
        or proof.get("attribution") != attribution
        or reference != expected_reference
        or proof.get("application_profile_adopted") is not False
        or proof.get("original_PDF_full_text_coverage_claimed") is not False
    ):
        raise DomainError("native_identity_trusted_chain_conflict", 409)


def verify_stored_resolution(
    landing: Landing, row: m.SourceRow, snapshot: m.SourceSnapshot, resolved: m.SourceResolution,
) -> None:
    if resolved.proof_sha256 != digest(resolved.proof):
        raise DomainError("stored_resolution_proof_hash_mismatch", 422)
    verify_native_identity(
        landing, row, snapshot, work_id=resolved.work_id, version_id=resolved.work_version_id,
        representation_id=resolved.representation_id, proof=resolved.proof,
    )
