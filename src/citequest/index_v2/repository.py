"""Short SQL transactions for native-index staging and publication.

No network request is made here. Operators use Landing DTOs, never body-supplied
owners or arbitrary collections. Pilot generations are never public read targets.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import and_, func, select
from sqlalchemy.dialects.postgresql import insert

from citequest.data_v2 import models as data
from citequest.data_v2.landing import Landing
from citequest.data_v2.schemas import IndexProjection
from citequest.v2 import models as core
from citequest.v2.domain import DomainError
from citequest.v2.services import Catalog, digest

from . import models as m
from .projection import BUILDER_VERSION
from .rows import IDENTITY_FIELDS, SHA, canonical_sha, validate_complete_units
from .schema import IndexProfile, checked_collection_name

CATALOG = "standard_metadata"
ALIAS = "cq_v2_standard_metadata"
CLEAN = "litsearch-clean-exact/1"
STANDARD_ROWS = 64183


def compiled_contract(projection: IndexProjection, compiled: dict) -> None:
    if (
        compiled["projection_id"] != str(projection.projection_id)
        or compiled["snapshot_id"] != str(projection.snapshot_id)
        or compiled["projection_sha256"] != projection.projection_sha256
        or compiled["source_pointer_sha256"] != projection.source_pointer_sha256
        or compiled["representation_profile"] != CLEAN
        or compiled["builder_version"] != BUILDER_VERSION
        or compiled["publication_performed"] is not False
    ):
        raise DomainError("index_compilation_identity_mismatch", 409)
    rows = compiled["rows"]
    if projection.searchable_text_present != bool(rows):
        raise DomainError("index_empty_source_misclassified", 409)
    if not rows:
        if (
            compiled["indexability"] != "empty_source_representation"
            or compiled["omitted_record"] is not False
            or compiled["expected"] is not None
        ):
            raise DomainError("empty_source_record_must_be_explicit", 409)
        return
    manifest = compiled["expected"]
    if manifest["coverage_scope"] != "complete_units" or manifest["count"] != len(rows):
        raise DomainError("complete_unit_manifest_required", 409)
    if len({row["chunk_id"] for row in rows}) != len(rows):
        raise DomainError("duplicate_expected_chunk_ID", 409)
    text = (projection.title or "") + "\n" + (projection.abstract or "")
    for row in rows:
        if (
            row["text"] != text[row["span_start"] : row["span_end"]]
            or row["native_id"] != projection.native_id
            or row["snapshot_id"] != str(projection.snapshot_id)
            or row["projection_sha256"] != projection.projection_sha256
            or row["source_pointer_sha256"] != projection.source_pointer_sha256
            or row["year"] != projection.year
            or row["security_domain"] != projection.security_domain
            or row["source_namespace"] != projection.source_namespace
            or row["source_kind"] != "standard_title_abstract"
            or row["profile_sha256"] != IndexProfile().fingerprint()
            or row["tokenizer_profile_sha256"] != compiled["tokenizer_profile_sha256"]
            or row["unit_text_sha256"] != hashlib.sha256(text.encode()).hexdigest()
            or row["source_text_sha256"] != hashlib.sha256(row["text"].encode()).hexdigest()
            or row["unit_codepoints"] != len(text)
            or type(row["token_count"]) is not int
            or not 0 < row["token_count"] <= 2048
            or {"dense", "sparse"} & row.keys()
            or row["chunk_id"] != canonical_sha({key: row[key] for key in IDENTITY_FIELDS})
            or manifest["rows"].get(row["chunk_id"]) != canonical_sha(row)
        ):
            raise DomainError("compiled_row_source_or_identity_mismatch", 409)
    validate_complete_units(rows)
    if manifest["sha256"] != canonical_sha(manifest["rows"]):
        raise DomainError("complete_unit_manifest_hash_mismatch", 409)


class IndexRepository:
    def __init__(self, catalog: Catalog, *, policy_sha256: str, evaluation_sha256: str) -> None:
        self.catalog, self.db, self.actor = catalog, catalog.db, catalog.actor
        self.landing = Landing(catalog, policy_sha256=policy_sha256, evaluation_sha256=evaluation_sha256)

    def generation(self, identifier: UUID, *, writing: bool = False) -> m.IndexGeneration:
        if writing:
            self.catalog.lock_owner()
            self.catalog.require_publisher()
        value = self.db.get(m.IndexGeneration, identifier, populate_existing=True, with_for_update=writing)
        return self._validate_generation_source(value)

    def _validate_generation_source(self, value: m.IndexGeneration | None) -> m.IndexGeneration:
        if value is None or value.security_domain != "public":
            raise DomainError("not_found", 404)
        source = self.landing.snapshot(value.snapshot_id)
        if (
            source.status == "disabled"
            or source.policy_sha256 != self.landing.policy_sha256
            or source.evaluation_sha256 != self.landing.evaluation_sha256
            or source.representation_profile != value.representation_profile
        ):
            raise DomainError("index_source_snapshot_unavailable_or_changed", 409)
        return value

    def _capture_pointer_snapshot(self, catalog_name: str, security_domain: str) -> tuple[m.IndexGeneration, int]:
        """One READ COMMITTED statement; no row lock and no read grant yet."""
        if security_domain not in {"public", self.actor.security_domain}:
            raise DomainError("not_found", 404)
        row = self.db.execute(
            select(m.IndexGeneration, m.IndexCatalog.epoch)
            .join(
                m.IndexCatalog,
                and_(
                    m.IndexCatalog.active_generation_id == m.IndexGeneration.id,
                    m.IndexCatalog.security_domain == m.IndexGeneration.security_domain,
                    m.IndexCatalog.name == m.IndexGeneration.catalog_name,
                ),
            )
            .where(m.IndexCatalog.name == catalog_name, m.IndexCatalog.security_domain == security_domain)
            .execution_options(populate_existing=True, cq_atomic_capture=True)
        ).one_or_none()
        if row is None:
            raise DomainError("search_index_not_published", 409)
        generation, epoch = row
        if generation.state != "active":
            raise DomainError("search_index_state_inconsistent", 409)
        return generation, epoch

    def _selected_ids(self, generation: m.IndexGeneration) -> list[UUID]:
        query = (
            select(data.SourceProjection.id)
            .join(data.SourceRow, data.SourceProjection.source_row_id == data.SourceRow.id)
            .where(
                data.SourceRow.snapshot_id == generation.snapshot_id,
                data.SourceProjection.representation_profile == CLEAN,
            )
            .order_by(data.SourceRow.artifact_path, data.SourceRow.row_ordinal)
        )
        if generation.requested_unit_limit is not None:
            query = query.limit(generation.requested_unit_limit)
        return list(self.db.scalars(query))

    def begin(
        self, snapshot_id: UUID, identifier: UUID, *, tokenizer_profile_sha256: str, unit_limit: int | None
    ) -> dict:
        self.catalog.lock_owner()
        self.catalog.require_publisher()
        if unit_limit is not None and (type(unit_limit) is not int or unit_limit not in {100, 5000}):
            raise DomainError("declared_pilot_100_or_5000_or_full_required", 400)
        source = self.landing.snapshot(snapshot_id)
        if (
            source.security_domain != "public"
            or source.representation_profile != CLEAN
            or source.purpose != "standard"
            or source.status not in {"landing", "staged"}
            or type(tokenizer_profile_sha256) is not str
            or not SHA.fullmatch(tokenizer_profile_sha256)
        ):
            raise DomainError("fixed_public_clean_standard_snapshot_required", 409)
        value = self.db.get(m.IndexGeneration, identifier)
        expected = {
            "snapshot_id": snapshot_id,
            "security_domain": "public",
            "representation_profile": CLEAN,
            "tokenizer_profile_sha256": tokenizer_profile_sha256,
            "index_profile_sha256": IndexProfile().fingerprint(),
            "policy_sha256": source.policy_sha256,
            "evaluation_sha256": source.evaluation_sha256,
            "build_scope": "full" if unit_limit is None else "pilot",
            "requested_unit_limit": unit_limit,
        }
        if value is None:
            value = m.IndexGeneration(
                id=identifier,
                creator_id=self.actor.owner_id,
                collection_name="cq_v2_clean_" + identifier.hex,
                **expected,
            )
            self.db.add(value)
            self.db.flush()
        elif any(getattr(value, key) != item for key, item in expected.items()):
            raise DomainError("generation_identity_conflict", 409)
        self.generation(identifier)
        return self.describe(identifier)

    def describe(self, identifier: UUID) -> dict:
        value = self.generation(identifier)
        return {
            "generation_id": str(value.id),
            "snapshot_id": str(value.snapshot_id),
            "collection_name": value.collection_name,
            "state": value.state,
            "build_scope": value.build_scope,
            "requested_unit_limit": value.requested_unit_limit,
            "unit_count": value.unit_count,
            "empty_unit_count": value.empty_unit_count,
            "chunk_count": value.chunk_count,
            "expected_manifest_sha256": value.expected_manifest_sha256,
            "membership_sha256": value.membership_sha256,
            "source_seal_sha256": value.source_seal_sha256,
            "index_profile_sha256": value.index_profile_sha256,
            "tokenizer_profile_sha256": value.tokenizer_profile_sha256,
        }

    def pending_projections(self, identifier: UUID, *, limit: int = 100) -> list[IndexProjection]:
        if type(limit) is not int or not 1 <= limit <= 500:
            raise DomainError("bounded_projection_batch_required", 400)
        value = self.generation(identifier)
        if value.state != "preparing":
            return []
        done = set(self.db.scalars(select(m.IndexUnit.projection_id).where(m.IndexUnit.generation_id == identifier)))
        wanted = [projection_id for projection_id in self._selected_ids(value) if projection_id not in done][:limit]
        return [self.landing.index_projection(projection_id) for projection_id in wanted]

    def stage_unit(self, identifier: UUID, projection: IndexProjection, compiled: dict) -> dict:
        value = self.generation(identifier, writing=True)
        if value.state != "preparing":
            raise DomainError("generation_not_preparing", 409)
        current = self.landing.index_projection(projection.projection_id)
        if (
            current.model_dump(mode="json", exclude={"work_id"})
            != projection.model_dump(mode="json", exclude={"work_id"})
            or current.snapshot_id != value.snapshot_id
        ):
            raise DomainError("source_projection_changed_during_compilation", 409)
        if compiled["tokenizer_profile_sha256"] != value.tokenizer_profile_sha256:
            raise DomainError("tokenizer_binding_mismatch", 409)
        compiled_contract(projection, compiled)
        prior = self.db.get(m.IndexUnit, (identifier, projection.projection_id))
        compilation_sha = digest(compiled)
        if prior is not None:
            if prior.compiled_sha256 != compilation_sha:
                raise DomainError("index_unit_replay_conflict", 409)
            return {"status": "already_staged", "projection_id": str(projection.projection_id)}
        event = self.db.scalar(
            select(core.Outbox)
            .where(core.Outbox.idempotency_key == f"SourceProjection:{projection.projection_id}:1:stage")
            .with_for_update()
        )
        if (
            event is None
            or event.payload_sha256 != digest(event.payload)
            or event.entity_id != str(projection.projection_id)
            or event.security_domain != projection.security_domain
            or event.payload["data"]["projection_sha256"] != projection.projection_sha256
            or event.payload["data"]["source_pointer_sha256"] != projection.source_pointer_sha256
        ):
            raise DomainError("index_source_outbox_binding_failed", 409)
        self.db.add(
            m.IndexUnit(
                generation_id=identifier,
                projection_id=projection.projection_id,
                native_id=projection.native_id,
                projection_sha256=projection.projection_sha256,
                source_pointer_sha256=projection.source_pointer_sha256,
                searchable=projection.searchable_text_present,
                chunk_count=len(compiled["rows"]),
                compiled_sha256=compilation_sha,
                projection=projection.model_dump(mode="json"),
                compilation=compiled,
                source_outbox_id=event.id,
            )
        )
        self.db.flush()
        self.db.add_all(
            [
                m.IndexChunk(
                    generation_id=identifier,
                    projection_id=projection.projection_id,
                    chunk_id=row["chunk_id"],
                    row_sha256=canonical_sha(row),
                    payload=row,
                )
                for row in compiled["rows"]
            ]
        )
        event.state, event.attempt_count = "processed", event.attempt_count + 1
        return {
            "status": "staged_not_searchable",
            "projection_id": str(projection.projection_id),
            "chunks": len(compiled["rows"]),
        }

    def seal_preparation(self, identifier: UUID, *, transport_batch_size: int = 4) -> dict:
        value = self.generation(identifier, writing=True)
        if value.state != "preparing":
            return self.describe(identifier)
        if type(transport_batch_size) is not int or not 1 <= transport_batch_size <= 16:
            raise DomainError("bounded_transport_batch_size_required", 400)
        expected = self._selected_ids(value)
        units = list(self.db.scalars(select(m.IndexUnit).where(m.IndexUnit.generation_id == identifier)))
        required = STANDARD_ROWS if value.build_scope == "full" else value.requested_unit_limit
        if len(expected) != required or {unit.projection_id for unit in units} != set(expected):
            raise DomainError("selected_source_units_incomplete_or_extra", 409)
        chunks = list(
            self.db.scalars(
                select(m.IndexChunk).where(m.IndexChunk.generation_id == identifier).order_by(m.IndexChunk.chunk_id)
            )
        )
        if sum(unit.chunk_count for unit in units) != len(chunks) or not chunks:
            raise DomainError("prepared_chunk_count_mismatch_or_no_searchable_text", 409)
        for index, offset in enumerate(range(0, len(chunks), transport_batch_size), 1):
            batch = chunks[offset : offset + transport_batch_size]
            manifest = {item.chunk_id: item.row_sha256 for item in batch}
            self.db.add(
                m.IndexBatch(
                    generation_id=identifier,
                    sequence=index,
                    chunk_ids=list(manifest),
                    batch_manifest={
                        "coverage_scope": "batch_only",
                        "count": len(batch),
                        "rows": manifest,
                        "sha256": canonical_sha(manifest),
                    },
                )
            )
        value.unit_count, value.empty_unit_count, value.chunk_count = (
            len(units),
            sum(not unit.searchable for unit in units),
            len(chunks),
        )
        value.membership_sha256 = digest(
            {
                str(unit.projection_id): {
                    "native_id": unit.native_id,
                    "projection_sha256": unit.projection_sha256,
                    "source_pointer_sha256": unit.source_pointer_sha256,
                }
                for unit in sorted(units, key=lambda unit: str(unit.projection_id))
            }
        )
        value.expected_manifest_sha256 = canonical_sha({chunk.chunk_id: chunk.row_sha256 for chunk in chunks})
        value.state = "building"
        return self.describe(identifier)

    def bind_server_proof(self, identifier: UUID, proof: dict) -> None:
        value = self.generation(identifier, writing=True)
        if proof["profile_sha256"] != value.index_profile_sha256:
            raise DomainError("server_profile_proof_mismatch", 409)
        if (
            value.server_proof is not None
            and value.server_proof["configuration_sha256"] != proof["configuration_sha256"]
        ):
            raise DomainError("server_proof_changed", 409)
        if value.server_proof is None:
            value.server_proof = proof

    def batch_claim(self, identifier: UUID) -> dict | None:
        value = self.generation(identifier, writing=True)
        if value.state != "building" or value.server_proof is None:
            raise DomainError("verified_server_required_before_delivery", 409)
        row = self.db.scalar(
            select(m.IndexBatch)
            .where(m.IndexBatch.generation_id == identifier, m.IndexBatch.state != "verified")
            .order_by(m.IndexBatch.sequence)
            .with_for_update()
        )
        if row is None:
            return None
        now = datetime.now(UTC)
        if row.lease_id is not None and row.lease_expires_at and row.lease_expires_at > now:
            return {"status": "busy", "batch_id": str(row.id)}
        was_uncertain = row.state in {"sending", "uncertain"}
        row.state, row.lease_id, row.lease_expires_at = "sending", uuid4(), now + timedelta(seconds=240)
        row.logical_attempts += 1
        chunks = {
            chunk.chunk_id: chunk
            for chunk in self.db.scalars(
                select(m.IndexChunk).where(
                    m.IndexChunk.generation_id == identifier, m.IndexChunk.chunk_id.in_(row.chunk_ids)
                )
            )
        }
        if set(chunks) != set(row.chunk_ids) or any(
            chunks[key].row_sha256 != row.batch_manifest["rows"][key]
            or canonical_sha(chunks[key].payload) != chunks[key].row_sha256
            for key in chunks
        ):
            raise DomainError("persisted_expected_batch_corrupt", 409)
        return {
            "status": "claimed",
            "batch_id": str(row.id),
            "lease_id": str(row.lease_id),
            "generation_id": str(identifier),
            "collection_name": value.collection_name,
            "read_before_write": was_uncertain,
            "rows": [chunks[key].payload for key in row.chunk_ids],
            "batch_manifest": row.batch_manifest,
            "logical_attempt": row.logical_attempts,
        }

    def batch_result(self, identifier: UUID, batch_id: UUID, lease_id: UUID, receipt: dict) -> None:
        self.generation(identifier, writing=True)
        row = self.db.get(m.IndexBatch, batch_id, with_for_update=True)
        if row is None or row.generation_id != identifier or row.lease_id != lease_id:
            raise DomainError("stale_index_batch_lease", 409)
        if receipt["state"] not in {"verified", "uncertain", "failed"}:
            raise DomainError("invalid_index_delivery_receipt", 409)
        if receipt["state"] == "verified" and (
            receipt["readback"]["complete"] is not True
            or receipt["readback"]["rows_sha256"] != row.batch_manifest["sha256"]
        ):
            raise DomainError("index_delivery_readback_mismatch", 409)
        row.state, row.last_receipt, row.lease_id, row.lease_expires_at = receipt["state"], receipt, None, None

    def remaining_batches(self, identifier: UUID) -> int:
        self.generation(identifier)
        return self.db.scalar(
            select(func.count())
            .select_from(m.IndexBatch)
            .where(m.IndexBatch.generation_id == identifier, m.IndexBatch.state != "verified")
        )

    def expected_rows(self, identifier: UUID) -> list[dict]:
        self.generation(identifier)
        rows = list(
            self.db.scalars(
                select(m.IndexChunk).where(m.IndexChunk.generation_id == identifier).order_by(m.IndexChunk.chunk_id)
            )
        )
        if any(canonical_sha(row.payload) != row.row_sha256 for row in rows):
            raise DomainError("persisted_expected_row_corrupt", 409)
        return [row.payload for row in rows]

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
            raise DomainError("complete_generation_readback_required", 409)
        source = self.landing.snapshot(value.snapshot_id)
        value.source_seal_sha256 = source.seal_sha256
        value.readback_proof, value.state = proof, "verified"
        return self.describe(identifier)

    def _publication_ready(self, value: m.IndexGeneration) -> None:
        source = self.landing.snapshot(value.snapshot_id)
        if (
            value.build_scope != "full"
            or value.unit_count != STANDARD_ROWS
            or value.state not in {"verified", "active", "retired"}
            or not source.profile_adopted
            or source.status != "staged"
            or not source.seal_sha256
            or digest(source.seal) != source.seal_sha256
            or value.source_seal_sha256 != source.seal_sha256
            or value.readback_proof is None
            or value.server_proof is None
        ):
            raise DomainError("full_adopted_source_and_complete_index_required", 409)

    def publish_intent(self, identifier: UUID, intent_id: UUID, *, expected_epoch: int) -> dict:
        value = self.generation(identifier, writing=True)
        self._publication_ready(value)
        self.db.execute(
            insert(m.IndexCatalog).values(name=CATALOG, security_domain="public", epoch=0).on_conflict_do_nothing()
        )
        catalog = self.db.get(m.IndexCatalog, (CATALOG, "public"), with_for_update=True, populate_existing=True)
        prior = self.db.get(m.IndexPublishIntent, intent_id)
        if prior:
            if prior.target_generation_id != identifier or prior.expected_epoch != expected_epoch:
                raise DomainError("publish_intent_identity_conflict", 409)
            return self.intent(intent_id)
        if catalog.epoch != expected_epoch or catalog.pending_intent_id is not None:
            raise DomainError("active_epoch_or_pending_publication_conflict", 409)
        self.db.add(
            m.IndexPublishIntent(
                id=intent_id,
                catalog_name=CATALOG,
                security_domain="public",
                expected_epoch=expected_epoch,
                previous_generation_id=catalog.active_generation_id,
                target_generation_id=identifier,
                alias_name=ALIAS,
            )
        )
        catalog.pending_intent_id = intent_id
        self.db.flush()
        return self.intent(intent_id)

    def intent(self, intent_id: UUID) -> dict:
        self.catalog.require_publisher()
        row = self.db.get(m.IndexPublishIntent, intent_id)
        if row is None:
            raise DomainError("not_found", 404)
        target = self.generation(row.target_generation_id)
        previous = self.db.get(m.IndexGeneration, row.previous_generation_id) if row.previous_generation_id else None
        return {
            "intent_id": str(row.id),
            "state": row.state,
            "expected_epoch": row.expected_epoch,
            "alias_name": row.alias_name,
            "target_generation_id": str(target.id),
            "target_collection": target.collection_name,
            "previous_generation_id": str(previous.id) if previous else None,
            "previous_collection": previous.collection_name if previous else None,
        }

    def record_alias(self, intent_id: UUID, receipt: dict) -> None:
        self.catalog.lock_owner()
        self.catalog.require_publisher()
        row = self.db.get(m.IndexPublishIntent, intent_id, with_for_update=True)
        if row is None or row.state not in {"prepared", "alias_confirmed", "reconcile_required"}:
            raise DomainError("publish_intent_not_pending", 409)
        target = self.generation(row.target_generation_id)
        if receipt.get("observed_collection") != target.collection_name or receipt.get("alias_name") != row.alias_name:
            raise DomainError("actual_alias_confirmation_required", 409)
        row.receipts = [*row.receipts, receipt]
        row.state = "alias_confirmed"

    def commit_publish(self, intent_id: UUID) -> dict:
        self.catalog.lock_owner()
        self.catalog.require_publisher()
        row = self.db.get(m.IndexPublishIntent, intent_id, with_for_update=True)
        if row is None:
            raise DomainError("not_found", 404)
        catalog = self.db.get(m.IndexCatalog, (CATALOG, "public"), with_for_update=True, populate_existing=True)
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
        row = self.db.get(m.IndexPublishIntent, intent_id)
        catalog = self.db.get(m.IndexCatalog, (CATALOG, "public"), populate_existing=True)
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
        self.catalog.require_publisher()
        row = self.db.get(m.IndexPublishIntent, intent_id, with_for_update=True)
        catalog = self.db.get(m.IndexCatalog, (CATALOG, "public"), with_for_update=True, populate_existing=True)
        plan = self.recovery_target(intent_id)
        if plan["action"] != "rollback_uncommitted_alias" or actual_collection != plan["authoritative_collection"]:
            raise DomainError("actual_alias_rollback_required", 409)
        row.receipts = [
            *row.receipts,
            {"observed_collection": actual_collection, "action": "rollback", "at": datetime.now(UTC).isoformat()},
        ]
        row.state, catalog.pending_intent_id = "rolled_back", None
        return self.intent(intent_id)

    def capture_reader(self, *, seconds: int = 120) -> dict:
        if type(seconds) is not int or not 1 <= seconds <= 300:
            raise DomainError("bounded_reader_lease_required", 400)
        generation, epoch = self._capture_pointer_snapshot(CATALOG, "public")
        self._validate_generation_source(generation)
        self._publication_ready(generation)
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
            "epoch": epoch,
            "collection_name": checked_collection_name(generation.collection_name),
            "manifest_sha256": generation.expected_manifest_sha256,
            "source_seal_sha256": generation.source_seal_sha256,
            "expires_at": lease.expires_at.isoformat(),
        }

    def recheck_reader(self, lease_id: UUID) -> dict:
        row = self.db.get(m.IndexReaderLease, lease_id, populate_existing=True)
        if row is None or row.owner_id != self.actor.owner_id or row.expires_at <= datetime.now(UTC):
            raise DomainError("reader_lease_expired_or_not_found", 409)
        generation = self.generation(row.generation_id)
        self._publication_ready(generation)
        return {"generation_id": str(generation.id), "collection_name": generation.collection_name, "epoch": row.epoch}

    def close_reader(self, lease_id: UUID) -> None:
        row = self.db.get(m.IndexReaderLease, lease_id)
        if row is None or row.owner_id != self.actor.owner_id:
            raise DomainError("not_found", 404)
        self.db.delete(row)
