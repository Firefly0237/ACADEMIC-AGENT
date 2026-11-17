"""Typed native corpus input for index staging, separate from File/PDF and QA grants."""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from citequest.source_refs_v2 import PaperSourceReference
from citequest.v2.domain import SHA, Input
from citequest.v2.storage import CAS, StoredBlob
from citequest.v2.storage_config import app_storage

from . import models as m
from .landing import Landing
from .native_codec import NativeRepresentationDraft
from .resolution import Resolver


class NativeRepresentationPointer(Input):
    representation_id: SHA
    source_row_id: UUID
    snapshot_id: UUID
    security_domain: str
    index_projection_id: UUID
    source_text_sha256: SHA
    profile: Literal["citequest-native-s2orc-literal/1"]
    profile_sha256: SHA
    storage_security_domain: str
    draft_sha256: SHA
    source_reference: PaperSourceReference
    immutable_blob_pointer: Literal[True] = True


class NativeIndexProjection(Input):
    source_kind: Literal["s2orc_native"] = "s2orc_native"
    representation_profile: Literal["litsearch-s2orc-native-text/1"] = "litsearch-s2orc-native-text/1"
    projection_id: UUID
    source_row_id: UUID
    snapshot_id: UUID
    native_id: str
    source_namespace: Literal["litsearch_s2orc"] = "litsearch_s2orc"
    security_domain: str
    source_revision: str
    representation_id: SHA
    projection_sha256: SHA
    source_pointer_sha256: SHA
    source_text_sha256: SHA
    codec_profile_sha256: SHA
    literal_text: Annotated[str, Field(min_length=1, max_length=200000)]
    work_id: UUID
    work_version_id: UUID
    year: Annotated[int, Field(ge=1000, le=2200)] | None
    source_reference: PaperSourceReference
    snapshot_state: Literal["landing", "staged", "disabled"]
    profile_adopted: bool
    coordinate_system: Literal["unicode_codepoint"] = "unicode_codepoint"
    original_PDF_text_coverage_claimed: Literal[False] = False
    index_staging_is_not_QA_permission: Literal[True] = True

    @model_validator(mode="after")
    def identity(self) -> NativeIndexProjection:
        if (
            hashlib.sha256(self.literal_text.encode()).hexdigest() != self.source_text_sha256
            or self.projection_sha256 != self.representation_id
            or self.source_reference.representation_id != self.representation_id
            or self.source_reference.work_version_id != self.work_version_id
        ):
            raise ValueError("native_index_source_identity_mismatch")
        return self


def read_for_index(landing: Landing, store: CAS, representation_id: str) -> NativeIndexProjection:
    """Internal corpus reader. QA adapters must enforce adoption separately."""
    from citequest.v2 import models as core
    from citequest.v2.domain import DomainError

    app_storage().assert_store(store)
    representation = landing.db.get(m.NativeRepresentation, representation_id)
    if representation is None:
        raise DomainError("not_found", 404)
    row = landing.db.get(m.SourceRow, representation.source_row_id)
    snapshot = landing.snapshot(row.snapshot_id)
    if snapshot.status == "disabled":
        raise DomainError("source_snapshot_disabled", 409)
    if snapshot.source_namespace != "litsearch_s2orc" or snapshot.purpose != "application":
        raise DomainError("not_a_native_application_snapshot", 409)
    if snapshot.profile_adopted:
        from .application_freeze import sealed_resolution

        resolved = sealed_resolution(landing, snapshot, row, representation_id)
    else:
        resolved = Resolver(landing).latest(row.id)
    if (
        resolved is None
        or resolved.state != "paper_identity_verified"
        or resolved.representation_id != representation_id
    ):
        raise DomainError("paper_identity_not_verified", 409)
    from .native_identity import verify_stored_resolution

    verify_stored_resolution(landing, row, snapshot, resolved)
    from .application_freeze import current_denials

    original = landing.db.get(core.SourceRecord, UUID(resolved.proof["source_record_id"]), populate_existing=True)
    if current_denials(landing, store, row, original, resolved.proof["attribution"].get("doi")):
        raise DomainError("application_current_source_denial", 409)
    work = landing.catalog.work(resolved.work_id)
    version = landing.catalog.version(resolved.work_version_id)
    if version.work_id != work.id:
        raise DomainError("native_work_version_conflict", 409)
    blob = landing.db.get(core.Blob, (representation.storage_security_domain, representation.draft_sha256))
    if blob is None or blob.size_bytes > 16 * 1024**2:
        raise DomainError("native_blob_record_missing_or_outside_limit", 422)
    draft = NativeRepresentationDraft.model_validate_json(
        store.read(StoredBlob(blob.security_domain, blob.sha256, blob.size_bytes))
    )
    if (
        draft.representation_id != representation_id
        or draft.source_pointer_sha256 != row.source_pointer_sha256
        or draft.source_pointer.native_id != row.native_id
        or draft.source_text_sha256 != representation.source_text_sha256
    ):
        raise DomainError("stored_native_source_identity_mismatch", 422)
    return NativeIndexProjection(
        projection_id=representation.index_projection_id,
        source_row_id=row.id,
        snapshot_id=snapshot.id,
        native_id=row.native_id,
        security_domain=row.security_domain,
        source_revision=snapshot.source_revision,
        representation_id=representation_id,
        projection_sha256=representation_id,
        source_pointer_sha256=draft.source_pointer_sha256,
        source_text_sha256=draft.source_text_sha256,
        codec_profile_sha256=draft.profile_sha256,
        literal_text=draft.literal_text,
        work_id=work.id,
        work_version_id=version.id,
        year=work.year,
        source_reference=PaperSourceReference(
            source_kind="paper",
            work_version_id=version.id,
            source_snapshot_sha256=snapshot.source_manifest_sha256,
            representation_id=representation_id,
        ),
        snapshot_state=snapshot.status,
        profile_adopted=snapshot.profile_adopted,
    )
