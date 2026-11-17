"""Typed source staging and generation publication on the shared SQL authority."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy.dialects.postgresql import insert

from citequest.data_v2 import models as sources
from citequest.updates_v2 import models as updates
from citequest.v2 import models as core
from citequest.v2.domain import DomainError
from citequest.v2.services import Catalog, digest
from citequest.v2.storage import CAS

from . import models as m
from .publication import publication_alias
from .repository import IndexRepository
from .rows_v2 import IDENTITY_FIELDS, SHA, canonical_sha, validate_complete_units
from .schema import IndexProfile
from .source_readers import SourceReaders
from .source_variants import (
    CanonicalSourceUnit,
    DeniedSourceUnit,
    MetadataReference,
    NativeReference,
    UpdateReference,
)

CATALOGS = {
    "metadata_snapshot": "standard_metadata",
    "native_snapshot": "application_papers",
    "update_seal": "managed_metadata",
    "publication_snapshot": "standard_metadata",
}


def unit_signature(unit: CanonicalSourceUnit | DeniedSourceUnit) -> str:
    identity = {
        "unit_id": str(unit.source_unit_id),
        "root_kind": unit.root.kind,
        "root_id": str(unit.root.source_root_id),
        "source_manifest_sha256": unit.root.source_manifest_sha256,
        "representation_profile": unit.root.representation_profile,
        "source_namespace": unit.root.source_namespace,
        "security_domain": unit.root.security_domain,
        "catalog_name": unit.root.catalog_name,
        "policy_sha256": unit.root.policy_sha256,
        "evaluation_sha256": unit.root.evaluation_sha256,
        "source_kind": unit.source_kind,
        "reference": unit.source_reference.model_dump(mode="json"),
        "native_id": unit.native_id,
        "source_revision": unit.source_revision,
        "projection_sha256": unit.projection_sha256,
        "source_pointer_sha256": unit.source_pointer_sha256,
        "source_denied": unit.source_denied,
    }
    if isinstance(unit, DeniedSourceUnit):
        identity["denials"] = [item.model_dump(mode="json") for item in unit.source_denials]
    else:
        identity.update(
            literal_text=unit.literal_text,
            unit_text_sha256=unit.unit_text_sha256,
            year=unit.year,
            coordinate_system=unit.coordinate_system,
            builder_version=unit.builder_version,
        )
        if unit.source_kind == "s2orc_native":
            identity.update(work_id=str(unit.work_id), work_version_id=str(unit.work_version_id))
    return canonical_sha(identity)


def verify_compilation(unit: CanonicalSourceUnit | DeniedSourceUnit, compiled: dict, tokenizer_sha: str) -> None:
    if (
        compiled["source_unit_id"] != str(unit.source_unit_id)
        or compiled["source_root_id"] != str(unit.root.source_root_id)
        or compiled["root_kind"] != unit.root.kind
        or compiled["reference"] != unit.source_reference.model_dump(mode="json")
        or compiled["projection_sha256"] != unit.projection_sha256
        or compiled["source_pointer_sha256"] != unit.source_pointer_sha256
        or compiled["publication_performed"] is not False
        or compiled["tokenizer_profile_sha256"] != tokenizer_sha
    ):
        raise DomainError("typed_compilation_identity_mismatch", 409)
    rows = compiled["rows"]
    if compiled.get("index_staging_is_not_QA_permission") is not True:
        raise DomainError("index_staging_cannot_claim_QA_permission", 409)
    if isinstance(unit, DeniedSourceUnit) or unit.source_denied or not unit.searchable_text_present:
        if rows or compiled["expected"] is not None or compiled["omitted_record"] is not False:
            raise DomainError("empty_or_denied_source_must_remain_explicit", 409)
        if isinstance(unit, DeniedSourceUnit) and (
            compiled["source_text_withheld"] is not True or compiled["original_empty_text_not_claimed"] is not True
        ):
            raise DomainError("withheld_text_cannot_be_claimed_empty_or_verified", 409)
        return
    if (
        compiled.get("unit_text_sha256") != unit.unit_text_sha256
        or compiled.get("coordinate_system") != unit.coordinate_system
        or compiled.get("builder_version") != unit.builder_version
    ):
        raise DomainError("typed_compilation_source_representation_metadata_mismatch", 409)
    expected = compiled["expected"]
    if not rows or expected["coverage_scope"] != "complete_units" or expected["count"] != len(rows):
        raise DomainError("complete_source_unit_manifest_required", 409)
    seen = set()
    for row in rows:
        if (
            row["chunk_id"] in seen
            or row["snapshot_id"] != str(unit.root.source_root_id)
            or row["native_id"] != unit.native_id
            or row["source_namespace"] != unit.root.source_namespace
            or row["security_domain"] != unit.root.security_domain
            or row["source_kind"] != unit.source_kind
            or row["projection_sha256"] != unit.projection_sha256
            or row["source_pointer_sha256"] != unit.source_pointer_sha256
            or row["unit_codepoints"] != len(unit.literal_text)
            or row["unit_text_sha256"] != unit.unit_text_sha256
            or row["text"] != unit.literal_text[row["span_start"] : row["span_end"]]
            or not row["text"].strip()
            or hashlib.sha256(row["text"].encode()).hexdigest() != row["source_text_sha256"]
            or row["year"] != unit.year
            or row["tokenizer_profile_sha256"] != tokenizer_sha
            or row["profile_sha256"] != IndexProfile().fingerprint()
            or type(row["token_count"]) is not int
            or not 0 < row["token_count"] <= 2048
            or {"dense", "sparse"} & row.keys()
            or row["row_key"]
            != canonical_sha({"snapshot_id": str(unit.root.source_root_id), "projection_id": str(unit.source_unit_id)})
            or row["chunk_id"] != canonical_sha({key: row[key] for key in IDENTITY_FIELDS})
            or expected["rows"].get(row["chunk_id"]) != canonical_sha(row)
        ):
            raise DomainError("typed_literal_row_source_or_hash_mismatch", 409)
        seen.add(row["chunk_id"])
    validate_complete_units(rows)
    if set(expected["rows"]) != seen or canonical_sha(expected["rows"]) != expected["sha256"]:
        raise DomainError("typed_complete_unit_manifest_hash_mismatch", 409)


class VariantIndexRepository(IndexRepository):
    def __init__(self, catalog: Catalog, *, store: CAS, policy_sha256: str, evaluation_sha256: str) -> None:
        if not hasattr(m.IndexGeneration, "source_root_kind"):
            raise DomainError("typed_index_0002_runtime_required", 409)
        super().__init__(catalog, policy_sha256=policy_sha256, evaluation_sha256=evaluation_sha256)
        self.readers = SourceReaders(
            catalog, store=store, policy_sha256=policy_sha256, evaluation_sha256=evaluation_sha256
        )

    def generation(self, identifier: UUID, *, writing: bool = False) -> m.IndexGeneration:
        if writing:
            self.catalog.lock_owner()
        value = self.db.get(m.IndexGeneration, identifier, populate_existing=True, with_for_update=writing)
        if writing and value is not None and value.security_domain == "public":
            self.catalog.require_publisher()
        return self._validate_generation_value(value)

    def _validate_generation_value(self, value: m.IndexGeneration | None) -> m.IndexGeneration:
        if value is None or value.security_domain not in {"public", self.actor.security_domain}:
            raise DomainError("not_found", 404)
        if (
            value.policy_sha256 != self.landing.policy_sha256
            or value.evaluation_sha256 != self.landing.evaluation_sha256
            or value.index_profile_sha256 != IndexProfile().fingerprint()
        ):
            raise DomainError("current_index_contract_binding_mismatch", 409)
        root = self.readers.root(value)
        if root.state == "disabled" or root.representation_profile != value.representation_profile:
            raise DomainError("index_source_unavailable_or_changed", 409)
        return value

    def begin_source(
        self, kind: str, root_id: UUID, identifier: UUID, *, tokenizer_profile_sha256: str, unit_limit: int | None
    ) -> dict:
        self.catalog.lock_owner()
        if type(tokenizer_profile_sha256) is not str or not SHA.fullmatch(tokenizer_profile_sha256):
            raise DomainError("pinned_tokenizer_sha_required", 400)
        if kind not in CATALOGS or (
            unit_limit is not None and (type(unit_limit) is not int or not 1 <= unit_limit <= 64183)
        ):
            raise DomainError("declared_typed_source_and_bounded_selection_required", 400)
        publication_snapshot_id = None
        if kind == "publication_snapshot":
            from citequest.updates_v2.publication import SnapshotService
            source = SnapshotService(self).snapshot(root_id)
            snapshot_id, update_seal_id, publication_snapshot_id = source.base_snapshot_id, None, root_id
            profile, manifest = source.profile_id, source.seal_sha256
        elif kind == "update_seal":
            source = self.db.get(updates.UpdateSeal, root_id)
            if source is None:
                raise DomainError("not_found", 404)
            snapshot_id, update_seal_id = None, root_id
            profile, manifest = "cq.protocol-metadata/1", source.seal_sha256
        else:
            source = self.landing.snapshot(root_id)
            snapshot_id, update_seal_id = root_id, None
            profile, manifest = source.representation_profile, source.source_manifest_sha256
        if source.security_domain not in {"public", self.actor.security_domain}:
            raise DomainError("not_found", 404)
        if source.security_domain == "public":
            self.catalog.require_publisher()
        desired = {
            "snapshot_id": snapshot_id,
            "update_seal_id": update_seal_id,
            "publication_snapshot_id": publication_snapshot_id,
            "source_root_kind": kind,
            "security_domain": source.security_domain,
            "source_namespace": "bounded_metadata_union" if kind == "publication_snapshot" else source.source_namespace,
            "source_manifest_sha256": manifest,
            "representation_profile": profile,
            "catalog_name": CATALOGS[kind],
            "policy_sha256": self.landing.policy_sha256,
            "evaluation_sha256": self.landing.evaluation_sha256,
            "tokenizer_profile_sha256": tokenizer_profile_sha256,
            "index_profile_sha256": IndexProfile().fingerprint(),
            "build_scope": "full" if unit_limit is None else "pilot",
            "requested_unit_limit": unit_limit,
        }
        self.readers.root(SimpleNamespace(**desired))
        value = self.db.get(m.IndexGeneration, identifier)
        if value is None:
            value = m.IndexGeneration(
                id=identifier,
                creator_id=self.actor.owner_id,
                collection_name="cq_v2_source_" + identifier.hex,
                **desired,
            )
            self.db.add(value)
            self.db.flush()
        elif any(getattr(value, key) != item for key, item in desired.items()):
            raise DomainError("typed_generation_identity_conflict", 409)
        return self.describe(identifier)

    def describe(self, identifier: UUID) -> dict:
        value = self.generation(identifier)
        return {
            "generation_id": str(value.id),
            "source_root_kind": value.source_root_kind,
            "source_root_id": str(value.publication_snapshot_id or value.snapshot_id or value.update_seal_id),
            "snapshot_id": str(value.snapshot_id) if value.snapshot_id else None,
            "update_seal_id": str(value.update_seal_id) if value.update_seal_id else None,
            "collection_name": value.collection_name,
            "catalog_name": value.catalog_name,
            "state": value.state,
            "build_scope": value.build_scope,
            "requested_unit_limit": value.requested_unit_limit,
            "unit_count": value.unit_count,
            "empty_unit_count": value.empty_unit_count,
            "denied_unit_count": value.denied_unit_count,
            "chunk_count": value.chunk_count,
            "expected_manifest_sha256": value.expected_manifest_sha256,
            "membership_sha256": value.membership_sha256,
            "source_seal_sha256": value.source_seal_sha256,
            "index_profile_sha256": value.index_profile_sha256,
            "tokenizer_profile_sha256": value.tokenizer_profile_sha256,
            "source_publication_eligible": self.readers.root(value).publication_eligible,
        }

    def pending_units(self, identifier: UUID, *, limit: int = 100) -> list[CanonicalSourceUnit | DeniedSourceUnit]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise DomainError("bounded_source_batch_required", 400)
        value = self.generation(identifier)
        if value.state != "preparing":
            return []
        root = self.readers.root(value)
        refs = self.readers.references(root, limit=value.requested_unit_limit)
        done = set(self.db.scalars(select(m.IndexUnit.projection_id).where(m.IndexUnit.generation_id == identifier)))
        wanted = [reference for reference in refs if self.readers.unit_id(reference) not in done][:limit]
        return [self.readers.read(root, reference) for reference in wanted]

    def stage_source_unit(self, identifier: UUID, unit: CanonicalSourceUnit | DeniedSourceUnit, compiled: dict) -> dict:
        value = self.generation(identifier, writing=True)
        if value.state != "preparing":
            raise DomainError("generation_not_preparing", 409)
        current = self.readers.read(self.readers.root(value), unit.source_reference)
        if current.source_unit_id != unit.source_unit_id or unit_signature(current) != unit_signature(unit):
            raise DomainError("typed_source_changed_during_compilation", 409)
        verify_compilation(unit, compiled, value.tokenizer_profile_sha256)
        existing = self.db.get(m.IndexUnit, (identifier, unit.source_unit_id))
        if existing:
            if existing.compiled_sha256 != digest(compiled):
                raise DomainError("typed_unit_replay_conflict", 409)
            return {"state": "already_staged", "source_unit_id": str(unit.source_unit_id)}
        reference = unit.source_reference
        if isinstance(reference, MetadataReference):
            entity, entity_id = "SourceProjection", str(reference.metadata_projection_id)
        elif isinstance(reference, NativeReference):
            entity, entity_id = "NativeRepresentation", reference.native_representation_id
        else:
            entity, entity_id = "SourceUpdateObservation", str(reference.update_observation_id)
        event = self.db.scalar(
            select(core.Outbox)
            .where(
                core.Outbox.entity_type == entity, core.Outbox.entity_id == entity_id, core.Outbox.entity_revision == 1
            )
            .with_for_update()
        )
        if (
            event is None
            or event.payload_sha256 != digest(event.payload)
            or event.security_domain != unit.root.security_domain
            or event.operation not in {"stage", "quarantine", "tombstone"}
        ):
            raise DomainError("typed_source_outbox_binding_failed", 409)
        payload = event.payload["data"]
        event_projection = payload.get("projection_sha256") or payload.get("representation_id")
        if event_projection != unit.projection_sha256:
            raise DomainError("source_outbox_projection_hash_mismatch", 409)
        denied = isinstance(unit, DeniedSourceUnit) or unit.source_denied
        work_id = None if isinstance(current, DeniedSourceUnit) else current.work_id
        version_id = None if isinstance(current, DeniedSourceUnit) else current.work_version_id
        row = m.IndexUnit(
            generation_id=identifier,
            projection_id=unit.source_unit_id,
            source_reference_kind=reference.kind,
            metadata_projection_id=reference.metadata_projection_id
            if isinstance(reference, MetadataReference)
            else None,
            native_representation_id=reference.native_representation_id
            if isinstance(reference, NativeReference)
            else None,
            update_observation_id=reference.update_observation_id if isinstance(reference, UpdateReference) else None,
            source_namespace=unit.root.source_namespace,
            source_revision=unit.source_revision,
            work_id=work_id,
            work_version_id=version_id,
            source_denied=denied,
            native_id=unit.native_id,
            projection_sha256=unit.projection_sha256,
            source_pointer_sha256=unit.source_pointer_sha256,
            searchable=not denied and unit.searchable_text_present,
            chunk_count=len(compiled["rows"]),
            compiled_sha256=digest(compiled),
            projection=current.model_dump(mode="json"),
            compilation=compiled,
            source_outbox_id=event.id,
        )
        self.db.add(row)
        self.db.flush()
        self.db.add_all(
            [
                m.IndexChunk(
                    generation_id=identifier,
                    projection_id=unit.source_unit_id,
                    chunk_id=chunk["chunk_id"],
                    row_sha256=canonical_sha(chunk),
                    payload=chunk,
                )
                for chunk in compiled["rows"]
            ]
        )
        event.state, event.attempt_count = "processed", event.attempt_count + 1
        return {
            "state": "staged_not_searchable",
            "source_unit_id": str(unit.source_unit_id),
            "reference_kind": reference.kind,
            "chunks": len(compiled["rows"]),
            "source_denied": denied,
        }

    def seal_preparation(self, identifier: UUID, *, transport_batch_size: int = 4) -> dict:
        value = self.generation(identifier, writing=True)
        if value.state != "preparing":
            return self.describe(identifier)
        if type(transport_batch_size) is not int or not 1 <= transport_batch_size <= 16:
            raise DomainError("bounded_transport_batch_required", 400)
        refs = self.readers.references(self.readers.root(value), limit=value.requested_unit_limit)
        expected = {self.readers.unit_id(reference) for reference in refs}
        units = list(self.db.scalars(select(m.IndexUnit).where(m.IndexUnit.generation_id == identifier)))
        if (
            not expected
            or {unit.projection_id for unit in units} != expected
            or (value.requested_unit_limit is not None and len(expected) != value.requested_unit_limit)
        ):
            raise DomainError("typed_source_members_incomplete_or_extra", 409)
        if value.source_root_kind == "metadata_snapshot" and value.build_scope == "full" and len(expected) != 64183:
            raise DomainError("full_standard_64183_units_required", 409)
        chunks = list(
            self.db.scalars(
                select(m.IndexChunk).where(m.IndexChunk.generation_id == identifier).order_by(m.IndexChunk.chunk_id)
            )
        )
        if sum(unit.chunk_count for unit in units) != len(chunks):
            raise DomainError("typed_source_chunk_count_mismatch", 409)
        for sequence, offset in enumerate(range(0, len(chunks), transport_batch_size), 1):
            rows = {chunk.chunk_id: chunk.row_sha256 for chunk in chunks[offset : offset + transport_batch_size]}
            self.db.add(
                m.IndexBatch(
                    generation_id=identifier,
                    sequence=sequence,
                    chunk_ids=list(rows),
                    batch_manifest={
                        "coverage_scope": "batch_only",
                        "rows": rows,
                        "count": len(rows),
                        "sha256": canonical_sha(rows),
                    },
                )
            )
        value.unit_count, value.empty_unit_count, value.denied_unit_count = (
            len(units),
            sum(not unit.searchable and not unit.source_denied for unit in units),
            sum(unit.source_denied for unit in units),
        )
        value.chunk_count = len(chunks)
        value.membership_sha256 = digest(
            {
                str(unit.projection_id): {
                    "kind": unit.source_reference_kind,
                    "native_id": unit.native_id,
                    "projection_sha256": unit.projection_sha256,
                    "pointer_sha256": unit.source_pointer_sha256,
                }
                for unit in sorted(units, key=lambda unit: str(unit.projection_id))
            }
        )
        value.expected_manifest_sha256 = canonical_sha({chunk.chunk_id: chunk.row_sha256 for chunk in chunks})
        value.state = "building" if chunks else "empty_unindexed"
        return self.describe(identifier)

    def accept_readback(self, identifier: UUID, proof: dict) -> dict:
        value = self.generation(identifier, writing=True)
        outstanding = self.db.scalar(
            select(func.count())
            .select_from(m.IndexBatch)
            .where(m.IndexBatch.generation_id == identifier, m.IndexBatch.state != "verified")
        )
        if (
            outstanding
            or value.server_proof is None
            or proof.get("complete") is not True
            or proof["matched"] != value.chunk_count
            or proof["rows_sha256"] != value.expected_manifest_sha256
        ):
            raise DomainError("complete_typed_generation_readback_required", 409)
        value.source_seal_sha256 = self.readers.root(value).source_seal_sha256
        value.readback_proof, value.state = proof, "verified"
        return self.describe(identifier)

    def remaining_batches(self, identifier: UUID) -> int:
        self.generation(identifier)
        return self.db.scalar(
            select(func.count())
            .select_from(m.IndexBatch)
            .where(m.IndexBatch.generation_id == identifier, m.IndexBatch.state != "verified")
        )

    def _publication_ready(self, value: m.IndexGeneration) -> None:
        self._read_ready(value)
        root = self.readers.root(value)
        refs = self.readers.references(root)
        expected = {self.readers.unit_id(reference) for reference in refs}
        actual = set(self.db.scalars(select(m.IndexUnit.projection_id).where(m.IndexUnit.generation_id == value.id)))
        if actual != expected or len(actual) != value.unit_count:
            raise DomainError("published_source_member_set_must_be_exact", 409)
        if root.kind == "metadata_snapshot" and value.unit_count != 64183:
            raise DomainError("full_standard_64183_required", 409)
        if root.kind == "native_snapshot" and not 500 <= value.unit_count <= 2000:
            raise DomainError("formally_frozen_application_500_to_2000_required", 409)
        if root.kind == "update_seal":
            # Current accepted UpdateIndexProjection/SQL seal is explicitly
            # observed-only. A future source profile needs a separate migration.
            raise DomainError("observed_metadata_profile_cannot_publish", 409)

    def _read_ready(self, value: m.IndexGeneration) -> None:
        # Full membership is checked at publish and is immutable afterwards.
        # Each query checks current root/contract/seal, not 64k full body JSON.
        root = self.readers.root(value)
        if (
            value.build_scope != "full"
            or value.state not in {"verified", "active", "retired"}
            or not root.publication_eligible
            or value.source_seal_sha256 != root.source_seal_sha256
            or value.readback_proof is None
            or value.server_proof is None
            or value.readback_proof["complete"] is not True
            or value.readback_proof["matched"] != value.chunk_count
            or value.readback_proof["rows_sha256"] != value.expected_manifest_sha256
        ):
            raise DomainError("adopted_sealed_source_and_complete_native_index_required", 409)
        if root.kind == "metadata_snapshot" and value.unit_count != 64183:
            raise DomainError("full_standard_64183_required", 409)
        if root.kind == "native_snapshot" and not 500 <= value.unit_count <= 2000:
            raise DomainError("formal_application_500_to_2000_required", 409)

    def _raw_intent(self, intent_id: UUID, *, writing: bool = False) -> m.IndexPublishIntent:
        row = self.db.get(m.IndexPublishIntent, intent_id, populate_existing=True, with_for_update=writing)
        if row is None or row.security_domain not in {"public", self.actor.security_domain}:
            raise DomainError("not_found", 404)
        if row.security_domain == "public":
            self.catalog.require_publisher()
        return row

    def intent(self, intent_id: UUID) -> dict:
        row = self._raw_intent(intent_id)
        # Recovery must remain possible when a target source was disabled after
        # preparing an intent. This is admin metadata, not a source-read grant.
        target = self.db.get(m.IndexGeneration, row.target_generation_id)
        previous = self.db.get(m.IndexGeneration, row.previous_generation_id) if row.previous_generation_id else None
        if target is None or target.catalog_name != row.catalog_name or target.security_domain != row.security_domain:
            raise DomainError("intent_target_catalog_identity_mismatch", 409)
        if previous is not None and (
            previous.catalog_name != row.catalog_name or previous.security_domain != row.security_domain
        ):
            raise DomainError("intent_previous_catalog_identity_mismatch", 409)
        return {
            "intent_id": str(row.id),
            "state": row.state,
            "expected_epoch": row.expected_epoch,
            "catalog_name": row.catalog_name,
            "alias_name": row.alias_name,
            "target_generation_id": str(target.id),
            "target_collection": target.collection_name,
            "previous_generation_id": str(previous.id) if previous else None,
            "previous_collection": previous.collection_name if previous else None,
        }

    def publish_intent(self, identifier: UUID, intent_id: UUID, *, expected_epoch: int) -> dict:
        value = self.generation(identifier, writing=True)
        if type(expected_epoch) is not int or expected_epoch < 0:
            raise DomainError("strict_nonnegative_epoch_required", 400)
        self._publication_ready(value)
        self.db.execute(
            insert(m.IndexCatalog)
            .values(name=value.catalog_name, security_domain=value.security_domain, epoch=0)
            .on_conflict_do_nothing()
        )
        catalog = self.db.get(
            m.IndexCatalog, (value.catalog_name, value.security_domain), with_for_update=True, populate_existing=True
        )
        prior = self.db.get(m.IndexPublishIntent, intent_id)
        if prior:
            if (
                prior.target_generation_id != identifier
                or prior.expected_epoch != expected_epoch
                or prior.catalog_name != value.catalog_name
            ):
                raise DomainError("publish_intent_identity_conflict", 409)
            return self.intent(intent_id)
        if catalog.epoch != expected_epoch or catalog.pending_intent_id is not None:
            raise DomainError("active_epoch_or_pending_publication_conflict", 409)
        self.db.add(
            m.IndexPublishIntent(
                id=intent_id,
                catalog_name=value.catalog_name,
                security_domain=value.security_domain,
                expected_epoch=expected_epoch,
                previous_generation_id=catalog.active_generation_id,
                target_generation_id=identifier,
                alias_name=publication_alias(value.catalog_name, value.security_domain),
            )
        )
        catalog.pending_intent_id = intent_id
        self.db.flush()
        return self.intent(intent_id)

    def record_alias(self, intent_id: UUID, receipt: dict) -> None:
        self.catalog.lock_owner()
        row = self._raw_intent(intent_id, writing=True)
        if row.state not in {"prepared", "alias_confirmed", "reconcile_required"}:
            raise DomainError("publish_intent_not_pending", 409)
        plan = self.intent(intent_id)
        if (
            receipt.get("observed_collection") != plan["target_collection"]
            or receipt.get("alias_name") != row.alias_name
            or receipt.get("verified_by_real_alias_readback") is not True
        ):
            raise DomainError("actual_alias_readback_required", 409)
        row.receipts = [*row.receipts, receipt]
        row.state = "alias_confirmed"

    def commit_publish(self, intent_id: UUID) -> dict:
        self.catalog.lock_owner()
        row = self._raw_intent(intent_id, writing=True)
        catalog = self.db.get(
            m.IndexCatalog, (row.catalog_name, row.security_domain), with_for_update=True, populate_existing=True
        )
        if (
            row.state == "committed"
            and catalog.active_generation_id == row.target_generation_id
            and catalog.epoch == row.expected_epoch + 1
        ):
            return self.intent(intent_id)
        if (
            row.state != "alias_confirmed"
            or catalog.pending_intent_id != intent_id
            or catalog.epoch != row.expected_epoch
            or catalog.active_generation_id != row.previous_generation_id
        ):
            raise DomainError("publish_compare_and_swap_failed", 409)
        target = self.generation(row.target_generation_id, writing=True)
        self._publication_ready(target)
        if row.previous_generation_id:
            self.db.get(m.IndexGeneration, row.previous_generation_id).state = "retired"
        target.state, row.state = "active", "committed"
        catalog.active_generation_id, catalog.epoch, catalog.pending_intent_id = target.id, catalog.epoch + 1, None
        return self.intent(intent_id)

    def recovery_target(self, intent_id: UUID) -> dict:
        plan = self.intent(intent_id)
        row = self._raw_intent(intent_id)
        catalog = self.db.get(m.IndexCatalog, (row.catalog_name, row.security_domain), populate_existing=True)
        if (
            row.state == "committed"
            and catalog.active_generation_id == row.target_generation_id
            and catalog.epoch == row.expected_epoch + 1
        ):
            return {**plan, "authoritative_collection": plan["target_collection"], "action": "repair_committed_alias"}
        if (
            catalog.active_generation_id == row.previous_generation_id
            and catalog.epoch == row.expected_epoch
            and catalog.pending_intent_id == row.id
        ):
            return {
                **plan,
                "authoritative_collection": plan["previous_collection"],
                "action": "rollback_uncommitted_alias",
            }
        raise DomainError("publish_recovery_requires_operator_reconciliation", 409)

    def acknowledge_rollback(self, intent_id: UUID, actual_collection: str | None) -> dict:
        self.catalog.lock_owner()
        row = self._raw_intent(intent_id, writing=True)
        catalog = self.db.get(
            m.IndexCatalog, (row.catalog_name, row.security_domain), with_for_update=True, populate_existing=True
        )
        plan = self.recovery_target(intent_id)
        if plan["action"] != "rollback_uncommitted_alias" or actual_collection != plan["authoritative_collection"]:
            raise DomainError("actual_alias_rollback_required", 409)
        row.receipts = [
            *row.receipts,
            {"observed_collection": actual_collection, "action": "rollback", "at": datetime.now(UTC).isoformat()},
        ]
        row.state, catalog.pending_intent_id = "rolled_back", None
        return self.intent(intent_id)

    def capture_reader(self, *, catalog_name: str = "standard_metadata", seconds: int = 120) -> dict:
        if catalog_name not in CATALOGS.values() or type(seconds) is not int or not 1 <= seconds <= 300:
            raise DomainError("declared_catalog_and_bounded_reader_lease_required", 400)
        generation, epoch = self._capture_pointer_snapshot(catalog_name, "public")
        self._validate_generation_value(generation)
        self._read_ready(generation)
        lease = m.IndexReaderLease(
            generation_id=generation.id,
            owner_id=self.actor.owner_id,
            epoch=epoch,
            expires_at=datetime.now(UTC) + timedelta(seconds=seconds),
        )
        self.db.add(lease)
        self.db.flush()
        return {
            "lease_id": str(lease.id),
            "generation_id": str(generation.id),
            "catalog_name": catalog_name,
            "epoch": epoch,
            "collection_name": generation.collection_name,
            "manifest_sha256": generation.expected_manifest_sha256,
            "source_seal_sha256": generation.source_seal_sha256,
            "expires_at": lease.expires_at.isoformat(),
        }

    def recheck_hits(self, lease_id: UUID, chunk_ids: list[str]) -> dict:
        # Parent recheck pins this same old/new SQL generation and honors source
        # disable/reader expiry; a later publication does not retarget the lease.
        pinned = self.recheck_reader(lease_id)
        generation = self.generation(UUID(pinned["generation_id"]))
        if (
            type(chunk_ids) is not list
            or len(chunk_ids) > 1000
            or any(type(key) is not str or not SHA.fullmatch(key) for key in chunk_ids)
            or len(set(chunk_ids)) != len(chunk_ids)
        ):
            raise DomainError("bounded_unique_hit_IDs_required", 400)
        rows = {
            row.chunk_id: row
            for row in self.db.scalars(
                select(m.IndexChunk).where(
                    m.IndexChunk.generation_id == generation.id, m.IndexChunk.chunk_id.in_(chunk_ids)
                )
            )
        }
        if set(rows) != set(chunk_ids):
            raise DomainError("hit_outside_captured_generation", 409)
        unit = m.IndexUnit
        access = exists(
            select(core.WorkAccess.work_id).where(
                core.WorkAccess.work_id == core.Work.id,
                core.WorkAccess.owner_id == self.actor.owner_id,
                core.WorkAccess.revoked.is_(False),
            )
        )
        work_visible = or_(
            unit.work_id.is_(None),
            exists(select(core.Work.id).where(core.Work.id == unit.work_id, or_(core.Work.public.is_(True), access))),
        )
        version_visible = or_(
            unit.work_version_id.is_(None),
            exists(
                select(core.WorkVersion.id).where(
                    core.WorkVersion.id == unit.work_version_id,
                    core.WorkVersion.work_id == unit.work_id,
                    core.WorkVersion.security_domain.in_(["public", self.actor.security_domain]),
                )
            ),
        )
        denied = exists(
            select(updates.SourceDenial.native_id).where(
                updates.SourceDenial.security_domain == generation.security_domain,
                updates.SourceDenial.source_namespace == unit.source_namespace,
                updates.SourceDenial.native_id == unit.native_id,
                or_(
                    updates.SourceDenial.source_revision == "*",
                    updates.SourceDenial.source_revision == unit.source_revision,
                ),
                updates.SourceDenial.active.is_(True),
            )
        )
        root_visible = exists(
            select(sources.SourceSnapshot.id).where(
                sources.SourceSnapshot.id == generation.snapshot_id,
                sources.SourceSnapshot.security_domain == generation.security_domain,
                sources.SourceSnapshot.source_namespace == generation.source_namespace,
                sources.SourceSnapshot.representation_profile == generation.representation_profile,
                sources.SourceSnapshot.policy_sha256 == generation.policy_sha256,
                sources.SourceSnapshot.evaluation_sha256 == generation.evaluation_sha256,
                sources.SourceSnapshot.status == "staged",
                sources.SourceSnapshot.profile_adopted.is_(True),
                sources.SourceSnapshot.seal_sha256 == generation.source_seal_sha256,
            )
        )
        if generation.source_root_kind == "publication_snapshot":
            root_visible = exists(select(updates.PublicationSnapshot.id).where(
                updates.PublicationSnapshot.id == generation.publication_snapshot_id,
                updates.PublicationSnapshot.security_domain == generation.security_domain,
                updates.PublicationSnapshot.seal_sha256 == generation.source_seal_sha256,
                updates.PublicationSnapshot.policy_sha256 == generation.policy_sha256,
                updates.PublicationSnapshot.evaluation_sha256 == generation.evaluation_sha256))
        # One statement snapshot on scalar FK/index columns. No native body
        # JSON/CAS or per-hit model call is loaded for this response barrier.
        decisions = dict(
            self.db.execute(
                select(unit.projection_id, and_(root_visible, work_visible, version_visible, ~denied)).where(
                    unit.generation_id == generation.id,
                    unit.projection_id.in_({chunk.projection_id for chunk in rows.values()}),
                )
            ).all()
        )
        allowed, masked = [], []
        for identifier in chunk_ids:
            if decisions[rows[identifier].projection_id]:
                allowed.append(identifier)
            else:
                masked.append({"chunk_id": identifier, "reason": "current_source_or_paper_access_denied"})
        return {
            **pinned,
            "allowed_chunk_ids": allowed,
            "masked": masked,
            "policy": "current SQL source/ACL after native search; no alias retargeting",
        }

    def recheck_reader(self, lease_id: UUID) -> dict:
        row = self.db.get(m.IndexReaderLease, lease_id, populate_existing=True)
        if row is None or row.owner_id != self.actor.owner_id or row.expires_at <= datetime.now(UTC):
            raise DomainError("reader_lease_expired_or_not_found", 409)
        generation = self.generation(row.generation_id)
        self._read_ready(generation)
        return {"generation_id": str(generation.id), "collection_name": generation.collection_name, "epoch": row.epoch}
