"""S05 source tables on the cq_app registry; DDL is managed only by Alembic."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from citequest.v2.models import SCHEMA, Base, Identity, J


class SourceSnapshot(Identity, Base):
    __tablename__ = "ingest_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "security_domain", "source_namespace", "source_revision", "representation_profile", "source_manifest_sha256"
        ),
        UniqueConstraint("security_domain", "id"),
        Index("ix_ingest_snapshot_source_created", "security_domain", "source_namespace", "created_at", "id"),
        CheckConstraint("status IN ('landing','staged','disabled')", name="snapshot_state"),
        CheckConstraint(
            "(status='landing' AND NOT profile_adopted AND seal IS NULL AND seal_sha256 IS NULL) OR "
            "(status='staged' AND profile_adopted AND seal IS NOT NULL AND seal_sha256 IS NOT NULL) OR status='disabled'",
            name="snapshot_adoption_state",
        ),
    )
    security_domain: Mapped[str] = mapped_column(String(100))
    importer_owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    source_namespace: Mapped[str] = mapped_column(String(100))
    source_revision: Mapped[str] = mapped_column(String(100))
    source_manifest_sha256: Mapped[str] = mapped_column(String(64))
    representation_profile: Mapped[str] = mapped_column(String(100))
    purpose: Mapped[str] = mapped_column(String(40))
    policy_sha256: Mapped[str] = mapped_column(String(64))
    evaluation_sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(40), default="landing")
    # Acquisition is not approval. No code in this module adopts a profile.
    profile_adopted: Mapped[bool] = mapped_column(Boolean, default=False)
    # Only the fixed standard adoption operation can populate this proof.
    seal: Mapped[dict | None] = mapped_column(J)
    seal_sha256: Mapped[str | None] = mapped_column(String(64))


class SourceArtifact(Base):
    __tablename__ = "ingest_artifacts"
    snapshot_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.ingest_snapshots.id"), primary_key=True)
    path: Mapped[str] = mapped_column(String(512), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    format: Mapped[str] = mapped_column(String(30))
    row_group_sizes: Mapped[list[int] | None] = mapped_column(J)


class SourceRow(Identity, Base):
    __tablename__ = "ingest_rows"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "native_id"),
        UniqueConstraint("security_domain", "id"),
        UniqueConstraint("snapshot_id", "artifact_path", "row_ordinal"),
        ForeignKeyConstraint(
            ["snapshot_id", "artifact_path"],
            [f"{SCHEMA}.ingest_artifacts.snapshot_id", f"{SCHEMA}.ingest_artifacts.path"],
        ),
        ForeignKeyConstraint(
            ["security_domain", "snapshot_id"],
            [f"{SCHEMA}.ingest_snapshots.security_domain", f"{SCHEMA}.ingest_snapshots.id"],
        ),
        CheckConstraint("row_group >= 0 AND row_in_group >= 0 AND row_ordinal >= 0", name="nonnegative_raw_position"),
    )
    snapshot_id: Mapped[UUID] = mapped_column(Uuid)
    security_domain: Mapped[str] = mapped_column(String(100))
    native_id: Mapped[str] = mapped_column(String(512))
    native_id_source_type: Mapped[str] = mapped_column(String(20))
    artifact_path: Mapped[str] = mapped_column(String(512))
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    row_group: Mapped[int] = mapped_column(Integer)
    row_in_group: Mapped[int] = mapped_column(Integer)
    row_ordinal: Mapped[int] = mapped_column(Integer)
    source_pointer_sha256: Mapped[str] = mapped_column(String(64))


class SourceProjection(Identity, Base):
    __tablename__ = "ingest_projections"
    __table_args__ = (UniqueConstraint("source_row_id", "representation_profile"),)
    source_row_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.ingest_rows.id"))
    representation_profile: Mapped[str] = mapped_column(String(100))
    projection_sha256: Mapped[str] = mapped_column(String(64))
    title: Mapped[str | None] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text)
    field_provenance: Mapped[dict] = mapped_column(J)
    unresolved_reasons: Mapped[list[str]] = mapped_column(J)
    searchable_text_present: Mapped[bool] = mapped_column(Boolean)


class SourceResolution(Identity, Base):
    __tablename__ = "ingest_resolutions"
    __table_args__ = (
        UniqueConstraint("source_row_id", "revision"),
        ForeignKeyConstraint(
            ["security_domain", "source_row_id"], [f"{SCHEMA}.ingest_rows.security_domain", f"{SCHEMA}.ingest_rows.id"]
        ),
        CheckConstraint("revision >= 1", name="positive_resolution_revision"),
        CheckConstraint(
            "(state='unresolved' AND work_id IS NULL AND work_version_id IS NULL) OR "
            "(state='metadata_resolved' AND work_id IS NOT NULL AND work_version_id IS NULL) OR "
            "(state='paper_identity_verified' AND work_id IS NOT NULL AND work_version_id IS NOT NULL AND representation_id IS NOT NULL)",
            name="resolution_identity_state",
        ),
    )
    source_row_id: Mapped[UUID] = mapped_column(Uuid)
    security_domain: Mapped[str] = mapped_column(String(100))
    revision: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(50))
    work_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.works.id"))
    work_version_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.work_versions.id"))
    representation_id: Mapped[str | None] = mapped_column(String(64))
    proof: Mapped[dict] = mapped_column(J)
    proof_sha256: Mapped[str] = mapped_column(String(64))


class NativeRepresentation(Base):
    __tablename__ = "ingest_native_representations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["security_domain", "source_row_id"], [f"{SCHEMA}.ingest_rows.security_domain", f"{SCHEMA}.ingest_rows.id"]
        ),
        ForeignKeyConstraint(
            ["storage_security_domain", "draft_sha256"], [f"{SCHEMA}.blobs.security_domain", f"{SCHEMA}.blobs.sha256"]
        ),
        UniqueConstraint("index_projection_id"),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_row_id: Mapped[UUID] = mapped_column(Uuid)
    security_domain: Mapped[str] = mapped_column(String(100))
    index_projection_id: Mapped[UUID] = mapped_column(Uuid)
    source_text_sha256: Mapped[str] = mapped_column(String(64))
    profile: Mapped[str] = mapped_column(String(100))
    profile_sha256: Mapped[str] = mapped_column(String(64))
    storage_security_domain: Mapped[str] = mapped_column(String(100))
    draft_sha256: Mapped[str] = mapped_column(String(64))
