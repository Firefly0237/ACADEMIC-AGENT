"""Append-only pointers to derived navigation metadata; never authorization.

Callers obtain a current source from the domain reader before every lookup.
Native cache inputs use immutable draft fields, not mutable display metadata,
catalog epochs or Task identifiers. CAS and SQL are not one storage transaction;
an interrupted write can leave an unreferenced derived blob.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from citequest.v2.domain import DomainError
from citequest.v2.storage import CAS, StoredBlob

from .source_structure import (
    VERSION, StoredStructure, StructurePolicy, StructureSource, StructureUnit,
    load_structure, policy_sha256, source_binding_sha256, store_structure,
)

if TYPE_CHECKING:
    from citequest.data_v2.native_codec import NativeRepresentationDraft

TABLE = "cq_app.source_structure_cache"
APPLICATION_GRANTS = """
GRANT SELECT,INSERT ON cq_app.source_structure_cache TO {application_role};
REVOKE UPDATE,DELETE,TRUNCATE ON cq_app.source_structure_cache FROM {application_role};
"""

KEY_FIELDS = ("security_domain", "source_kind", "source_id", "source_version_sha256", "policy_sha256")
KEY_PREDICATE = " AND ".join(f"{name}=:{name}" for name in KEY_FIELDS)
LOOKUP = text("SELECT * FROM cq_app.source_structure_cache WHERE " + KEY_PREDICATE)


def require_structure_schema(db) -> None:
    ready = db.execute(text("SELECT to_regclass('cq_app.source_structure_cache') IS NOT NULL")).scalar_one()
    if ready is not True:
        raise DomainError("source_structure_migration_required", 503)


def native_structure_source(draft: NativeRepresentationDraft, security_domain: str) -> StructureSource:
    """Shared registration/runtime recipe, tied to the exact native draft identity."""
    draft.validate_identity()
    title_spans = draft.annotation_spans.get("title", [])
    title = draft.literal_text[title_spans[0][0]:title_spans[0][1]] if title_spans else None
    reference = {"kind": "native_representation", "id": draft.representation_id}
    locator = draft.locator(0, len(draft.literal_text)).model_dump(mode="json")
    return StructureSource(selection=reference, source_version_sha256=draft.representation_id,
        representation_id=draft.representation_id, security_domain=security_domain, title=title,
        units=(StructureUnit(reference=reference, text=draft.literal_text,
            source_text_sha256=draft.source_text_sha256, coordinate_system="unicode_codepoints_in_native_literal",
            locator=locator, annotations=draft.annotation_spans),), coverage="native_literal",
        missing_ranges=tuple({"reason": "native_annotation_invalid", "annotation": name}
                             for name in sorted(draft.annotation_errors)))


def _key(source: StructureSource, policy: StructurePolicy) -> dict:
    # Fingerprinting also validates literal hashes and source/metadata bounds.
    source_binding_sha256(source, policy)
    return {"security_domain": source.security_domain, "source_kind": source.selection["kind"],
        "source_id": source.selection["id"], "source_version_sha256": source.source_version_sha256,
        "policy_sha256": policy_sha256(policy)}


def _read(db, store: CAS, source: StructureSource, policy: StructurePolicy, key: dict) -> StoredStructure | None:
    row = db.execute(LOOKUP, key).mappings().one_or_none()
    if row is None:
        return None
    if (row["schema_version"] != VERSION or row["source_binding_sha256"] != source_binding_sha256(source, policy)
            or row["sidecar_size_bytes"] <= 0 or row["sidecar_size_bytes"] > policy.maximum_sidecar_bytes):
        raise DomainError("source_structure_cache_binding_conflict", 409)
    blob = db.execute(text("SELECT size_bytes FROM cq_app.blobs WHERE security_domain=:security_domain AND sha256=:sha256"),
        {"security_domain": key["security_domain"], "sha256": row["sidecar_sha256"]}).mappings().one_or_none()
    if blob is None or blob["size_bytes"] != row["sidecar_size_bytes"]:
        raise DomainError("source_structure_cache_blob_conflict", 409)
    pointer = StoredBlob(key["security_domain"], row["sidecar_sha256"], row["sidecar_size_bytes"])
    return StoredStructure(pointer, load_structure(store, pointer, source, policy))


def cache_structure(db, store: CAS, source: StructureSource,
                    policy: StructurePolicy | None = None) -> StoredStructure:
    """Use the caller's SQL transaction; source admission remains caller-owned."""
    policy = policy or StructurePolicy()
    require_structure_schema(db)
    key = _key(source, policy)
    existing = _read(db, store, source, policy, key)
    if existing is not None:
        return existing
    stored = store_structure(store, source, policy)
    pointer = stored.pointer
    db.execute(text("""INSERT INTO cq_app.blobs(security_domain,sha256,size_bytes)
        VALUES (:security_domain,:sha256,:size_bytes) ON CONFLICT (security_domain,sha256) DO NOTHING"""),
        {"security_domain": pointer.security_domain, "sha256": pointer.sha256, "size_bytes": pointer.size_bytes})
    values = {**key, "sidecar_sha256": pointer.sha256, "sidecar_size_bytes": pointer.size_bytes,
        "source_binding_sha256": source_binding_sha256(source, policy), "schema_version": VERSION}
    db.execute(text("""INSERT INTO cq_app.source_structure_cache
        (security_domain,source_kind,source_id,source_version_sha256,policy_sha256,
         sidecar_sha256,sidecar_size_bytes,source_binding_sha256,schema_version)
        VALUES (:security_domain,:source_kind,:source_id,:source_version_sha256,:policy_sha256,
         :sidecar_sha256,:sidecar_size_bytes,:source_binding_sha256,:schema_version)
        ON CONFLICT (security_domain,source_kind,source_id,source_version_sha256,policy_sha256) DO NOTHING"""), values)
    # Concurrent builders must converge on the same immutable pointer.
    actual = _read(db, store, source, policy, key)
    if actual is None or actual.pointer != pointer:
        raise DomainError("source_structure_cache_pointer_conflict", 409)
    return actual


class SourceStructureCache:
    def __init__(self, factory, store: CAS, policy: StructurePolicy | None = None) -> None:
        self.factory, self.store, self.policy = factory, store, policy or StructurePolicy()
        self.policy.validate()

    def get(self, current_source: StructureSource) -> StoredStructure | None:
        with self.factory.begin() as db:
            require_structure_schema(db)
            return _read(db, self.store, current_source, self.policy, _key(current_source, self.policy))

    def get_or_build(self, current_source: StructureSource) -> StoredStructure:
        with self.factory.begin() as db:
            return cache_structure(db, self.store, current_source, self.policy)
