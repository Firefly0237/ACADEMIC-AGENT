"""Persistent staging, delivery and publication; SQL is the read authority."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from citequest.data_v2 import models as data_models  # noqa: F401
from citequest.updates_v2 import models as update_models  # noqa: F401
from citequest.v2.models import SCHEMA, Base, Identity, J


class IndexGeneration(Identity, Base):
    __tablename__ = "index_generations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["security_domain", "snapshot_id"],
            [f"{SCHEMA}.ingest_snapshots.security_domain", f"{SCHEMA}.ingest_snapshots.id"],
        ),
        UniqueConstraint("id", "security_domain"),
        UniqueConstraint("id", "security_domain", "catalog_name"),
        CheckConstraint(
            "(source_root_kind IN ('metadata_snapshot','native_snapshot') AND snapshot_id IS NOT NULL AND update_seal_id IS NULL) OR (source_root_kind='update_seal' AND snapshot_id IS NULL AND update_seal_id IS NOT NULL)",
            name="exact_typed_index_source_root",
        ),
        UniqueConstraint("collection_name"),
        CheckConstraint(
            "state IN ('preparing','building','verifying','verified','active','retired','failed','empty_unindexed')",
            name="known_index_generation_state",
        ),
        CheckConstraint(
            "unit_count >= 0 AND empty_unit_count >= 0 AND chunk_count >= 0 AND empty_unit_count <= unit_count",
            name="valid_index_generation_counts",
        ),
        Index("ix_index_generation_snapshot_state", "snapshot_id", "state", "created_at", "id"),
    )
    snapshot_id: Mapped[UUID | None] = mapped_column(Uuid)
    update_seal_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.source_update_seals.id"))
    source_root_kind: Mapped[str] = mapped_column(
        String(30), default="metadata_snapshot", server_default="metadata_snapshot"
    )
    catalog_name: Mapped[str] = mapped_column(
        String(100), default="standard_metadata", server_default="standard_metadata"
    )
    source_namespace: Mapped[str] = mapped_column(String(100))
    source_manifest_sha256: Mapped[str] = mapped_column(String(64))
    security_domain: Mapped[str] = mapped_column(String(100))
    creator_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    collection_name: Mapped[str] = mapped_column(String(128))
    build_scope: Mapped[str] = mapped_column(String(20))
    requested_unit_limit: Mapped[int | None] = mapped_column(Integer)
    representation_profile: Mapped[str] = mapped_column(String(100))
    index_profile_sha256: Mapped[str] = mapped_column(String(64))
    tokenizer_profile_sha256: Mapped[str] = mapped_column(String(64))
    policy_sha256: Mapped[str] = mapped_column(String(64))
    evaluation_sha256: Mapped[str] = mapped_column(String(64))
    source_seal_sha256: Mapped[str | None] = mapped_column(String(64))
    membership_sha256: Mapped[str | None] = mapped_column(String(64))
    expected_manifest_sha256: Mapped[str | None] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(30), default="preparing")
    unit_count: Mapped[int] = mapped_column(Integer, default=0)
    empty_unit_count: Mapped[int] = mapped_column(Integer, default=0)
    denied_unit_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    server_proof: Mapped[dict | None] = mapped_column(J)
    readback_proof: Mapped[dict | None] = mapped_column(J)
    error: Mapped[dict | None] = mapped_column(J)


class IndexUnit(Base):
    __tablename__ = "index_units"
    __table_args__ = (
        UniqueConstraint("generation_id", "native_id"),
        CheckConstraint(
            "(source_reference_kind='metadata_projection' AND metadata_projection_id IS NOT NULL AND native_representation_id IS NULL AND update_observation_id IS NULL) OR (source_reference_kind='native_representation' AND metadata_projection_id IS NULL AND native_representation_id IS NOT NULL AND update_observation_id IS NULL) OR (source_reference_kind='update_observation' AND metadata_projection_id IS NULL AND native_representation_id IS NULL AND update_observation_id IS NOT NULL)",
            name="exact_typed_index_source_reference",
        ),
        Index("ix_index_unit_current_source", "generation_id", "source_namespace", "native_id", "source_revision"),
    )
    generation_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.index_generations.id"), primary_key=True)
    # Retain the physical legacy PK column for online compatibility. It is a
    # logical unit UUID; the following real typed FK establishes provenance.
    projection_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    source_reference_kind: Mapped[str] = mapped_column(
        String(30), default="metadata_projection", server_default="metadata_projection"
    )
    metadata_projection_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.ingest_projections.id"))
    native_representation_id: Mapped[str | None] = mapped_column(
        ForeignKey(f"{SCHEMA}.ingest_native_representations.id")
    )
    update_observation_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.source_update_observations.id"))
    source_namespace: Mapped[str] = mapped_column(String(100))
    source_revision: Mapped[str | None] = mapped_column(String(100))
    work_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.works.id"))
    work_version_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.work_versions.id"))
    source_denied: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    native_id: Mapped[str] = mapped_column(String(512))
    projection_sha256: Mapped[str] = mapped_column(String(64))
    source_pointer_sha256: Mapped[str] = mapped_column(String(64))
    searchable: Mapped[bool] = mapped_column(Boolean)
    chunk_count: Mapped[int] = mapped_column(Integer)
    compiled_sha256: Mapped[str] = mapped_column(String(64))
    # The full DTO and complete-unit expectation remain distinguishable from a
    # transmission-batch proof. No empty source row disappears from this table.
    projection: Mapped[dict] = mapped_column(J)
    compilation: Mapped[dict] = mapped_column(J)
    source_outbox_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.outbox.id"))


class IndexChunk(Base):
    __tablename__ = "index_chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["generation_id", "projection_id"],
            [f"{SCHEMA}.index_units.generation_id", f"{SCHEMA}.index_units.projection_id"],
        ),
        Index("ix_index_chunk_generation_projection", "generation_id", "projection_id", "chunk_id"),
    )
    generation_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    projection_id: Mapped[UUID] = mapped_column(Uuid)
    row_sha256: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(J)


class IndexBatch(Identity, Base):
    __tablename__ = "index_batches"
    __table_args__ = (
        UniqueConstraint("generation_id", "sequence"),
        CheckConstraint("state IN ('ready','sending','uncertain','verified','failed')", name="known_index_batch_state"),
        Index("ix_index_batch_dispatch", "generation_id", "state", "sequence"),
    )
    generation_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.index_generations.id"))
    sequence: Mapped[int] = mapped_column(Integer)
    chunk_ids: Mapped[list[str]] = mapped_column(J)
    batch_manifest: Mapped[dict] = mapped_column(J)
    state: Mapped[str] = mapped_column(String(30), default="ready")
    logical_attempts: Mapped[int] = mapped_column(Integer, default=0)
    lease_id: Mapped[UUID | None] = mapped_column(Uuid)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_receipt: Mapped[dict | None] = mapped_column(J)


class IndexCatalog(Base):
    __tablename__ = "index_catalogs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["active_generation_id", "security_domain", "name"],
            [
                f"{SCHEMA}.index_generations.id",
                f"{SCHEMA}.index_generations.security_domain",
                f"{SCHEMA}.index_generations.catalog_name",
            ],
        ),
        ForeignKeyConstraint(
            ["active_generation_id", "security_domain"],
            [f"{SCHEMA}.index_generations.id", f"{SCHEMA}.index_generations.security_domain"],
        ),
        CheckConstraint("epoch >= 0", name="nonnegative_index_catalog_epoch"),
    )
    name: Mapped[str] = mapped_column(String(100), primary_key=True)
    security_domain: Mapped[str] = mapped_column(String(100), primary_key=True)
    active_generation_id: Mapped[UUID | None] = mapped_column(Uuid)
    epoch: Mapped[int] = mapped_column(BigInteger, default=0)
    pending_intent_id: Mapped[UUID | None] = mapped_column(Uuid)


class IndexPublishIntent(Identity, Base):
    __tablename__ = "index_publish_intents"
    __table_args__ = (
        ForeignKeyConstraint(
            ["catalog_name", "security_domain"],
            [f"{SCHEMA}.index_catalogs.name", f"{SCHEMA}.index_catalogs.security_domain"],
        ),
        CheckConstraint(
            "state IN ('prepared','alias_confirmed','committed','rolled_back','reconcile_required')",
            name="known_index_publish_state",
        ),
        Index("ix_index_publish_recovery", "state", "created_at", "id"),
    )
    catalog_name: Mapped[str] = mapped_column(String(100))
    security_domain: Mapped[str] = mapped_column(String(100))
    expected_epoch: Mapped[int] = mapped_column(BigInteger)
    previous_generation_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.index_generations.id"))
    target_generation_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.index_generations.id"))
    alias_name: Mapped[str] = mapped_column(String(128))
    state: Mapped[str] = mapped_column(String(30), default="prepared")
    receipts: Mapped[list[dict]] = mapped_column(J, default=list)


class IndexReaderLease(Identity, Base):
    __tablename__ = "index_reader_leases"
    __table_args__ = (Index("ix_index_reader_lease_expiry", "generation_id", "expires_at"),)
    generation_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.index_generations.id"))
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    epoch: Mapped[int] = mapped_column(BigInteger)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
