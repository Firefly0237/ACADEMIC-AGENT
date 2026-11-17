"""S04 tables share the accepted cq_app registry; no second business database."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from citequest.v2.models import SCHEMA, Base, Identity, J


class MaterialUpload(Identity, Base):
    __tablename__ = "material_uploads"
    __table_args__ = (
        UniqueConstraint("owner_id", "id"),
        ForeignKeyConstraint(["owner_id", "project_id"], [f"{SCHEMA}.projects.owner_id", f"{SCHEMA}.projects.id"]),
        ForeignKeyConstraint(["owner_id", "file_id"], [f"{SCHEMA}.files.owner_id", f"{SCHEMA}.files.id"]),
        CheckConstraint("size_bytes > 0", name="positive_size"),
        Index("ix_material_uploads_owner_expires", "owner_id", "expires_at"),
    )
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    project_id: Mapped[UUID | None] = mapped_column(Uuid)
    file_id: Mapped[UUID | None] = mapped_column(Uuid)
    file_name: Mapped[str] = mapped_column(String(255))
    display_title: Mapped[str] = mapped_column(String(500))
    media_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    expected_sha256: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    file_receipt: Mapped[dict | None] = mapped_column(J)


class MaterialVersion(Identity, Base):
    """Explicitly a material version, never a fabricated WorkVersion."""
    __tablename__ = "material_versions"
    __table_args__ = (
        UniqueConstraint("owner_id", "id"), UniqueConstraint("file_id"),
        ForeignKeyConstraint(["owner_id", "file_id"], [f"{SCHEMA}.files.owner_id", f"{SCHEMA}.files.id"]),
    )
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    file_id: Mapped[UUID] = mapped_column(Uuid)
    security_domain: Mapped[str] = mapped_column(String(100))
    source_sha256: Mapped[str] = mapped_column(String(64))
    media_type: Mapped[str] = mapped_column(String(100))


class ParsedRepresentation(Base):
    __tablename__ = "parsed_representations"
    __table_args__ = (
        UniqueConstraint("owner_id", "id"), UniqueConstraint("file_id", "config_sha256"),
        ForeignKeyConstraint(["owner_id", "file_id"], [f"{SCHEMA}.files.owner_id", f"{SCHEMA}.files.id"]),
        ForeignKeyConstraint(["owner_id", "material_version_id"], [f"{SCHEMA}.material_versions.owner_id", f"{SCHEMA}.material_versions.id"]),
        ForeignKeyConstraint(["security_domain", "artifact_sha256"], [f"{SCHEMA}.blobs.security_domain", f"{SCHEMA}.blobs.sha256"]),
        Index("ix_parsed_representations_file", "owner_id", "file_id"),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    file_id: Mapped[UUID] = mapped_column(Uuid)
    material_version_id: Mapped[UUID] = mapped_column(Uuid)
    security_domain: Mapped[str] = mapped_column(String(100))
    source_sha256: Mapped[str] = mapped_column(String(64))
    config_sha256: Mapped[str] = mapped_column(String(64))
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    page_count: Mapped[int | None] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(J)


class MaterialIngestion(Identity, Base):
    __tablename__ = "material_ingestions"
    __table_args__ = (
        UniqueConstraint("owner_id", "id"), UniqueConstraint("file_id"),
        ForeignKeyConstraint(["owner_id", "file_id"], [f"{SCHEMA}.files.owner_id", f"{SCHEMA}.files.id"]),
        ForeignKeyConstraint(["owner_id", "representation_id"], [f"{SCHEMA}.parsed_representations.owner_id", f"{SCHEMA}.parsed_representations.id"]),
        ForeignKeyConstraint(["owner_id", "binding_check_id"], [f"{SCHEMA}.material_binding_checks.owner_id", f"{SCHEMA}.material_binding_checks.id"]),
        CheckConstraint("revision >= 1", name="positive_revision"),
        CheckConstraint("state IN ('queued','validating','parsing','partial','ready','failed','cancelled')", name="known_state"),
        Index("ix_material_ingestions_queue", "state", "created_at", "id"),
    )
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    file_id: Mapped[UUID] = mapped_column(Uuid)
    state: Mapped[str] = mapped_column(String(20), default="queued")
    revision: Mapped[int] = mapped_column(Integer, default=1)
    lease_id: Mapped[UUID | None] = mapped_column(Uuid)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    profile_sha256: Mapped[str | None] = mapped_column(String(64))
    representation_id: Mapped[str | None] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(100))
    binding_check_id: Mapped[UUID | None] = mapped_column(Uuid)
    execution_id: Mapped[UUID | None] = mapped_column(Uuid)
    execution_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    execution_budget_seconds: Mapped[float | None] = mapped_column(Float)
    active_elapsed_seconds: Mapped[float | None] = mapped_column(Float)
    execution_outcome: Mapped[str] = mapped_column(String(30), default="not_started", server_default="not_started")


class MaterialBindingCheck(Identity, Base):
    """Typed immutable association proof; a user-selected UUID is not proof."""
    __tablename__ = "material_binding_checks"
    __table_args__ = (
        UniqueConstraint("owner_id", "id"),
        ForeignKeyConstraint(["owner_id", "file_id"], [f"{SCHEMA}.files.owner_id", f"{SCHEMA}.files.id"]),
        CheckConstraint("binding_source IN ('user_explicit','trusted_source') AND identity_status IN ('unverified','hash_verified')", name="known_binding_grade"),
        CheckConstraint("identity_status <> 'hash_verified' OR (binding_source='trusted_source' AND source_record_id IS NOT NULL)", name="hash_grade_requires_source"),
        Index("ix_material_binding_checks_file", "owner_id", "file_id", "created_at", "id"),
    )
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    file_id: Mapped[UUID] = mapped_column(Uuid)
    version_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.work_versions.id"))
    binding_source: Mapped[str] = mapped_column(String(40))
    identity_status: Mapped[str] = mapped_column(String(30))
    source_record_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.source_records.id"))
    proof: Mapped[dict] = mapped_column(J)


class MaterialEvent(Base):
    __tablename__ = "material_events"
    __table_args__ = (
        ForeignKeyConstraint(["owner_id", "ingestion_id"], [f"{SCHEMA}.material_ingestions.owner_id", f"{SCHEMA}.material_ingestions.id"]),
    )
    ingestion_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    state: Mapped[str] = mapped_column(String(20))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(100))


class MaterialFragment(Base):
    __tablename__ = "material_fragments"
    __table_args__ = (
        UniqueConstraint("owner_id", "id"), UniqueConstraint("representation_id", "ordinal"),
        ForeignKeyConstraint(["owner_id", "representation_id"], [f"{SCHEMA}.parsed_representations.owner_id", f"{SCHEMA}.parsed_representations.id"]),
        ForeignKeyConstraint(["owner_id", "file_id"], [f"{SCHEMA}.files.owner_id", f"{SCHEMA}.files.id"]),
        CheckConstraint("ordinal >= 1 AND token_count >= 1 AND token_count <= 2048", name="bounded_tokens_ordinal"),
        Index("ix_material_fragments_file_page", "owner_id", "file_id", "ordinal"),
        Index("ix_material_fragments_text", "search_document", postgresql_using="gin"),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    file_id: Mapped[UUID] = mapped_column(Uuid)
    representation_id: Mapped[str] = mapped_column(String(64))
    ordinal: Mapped[int] = mapped_column(Integer)
    source_text: Mapped[str] = mapped_column(Text)
    index_text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    search_document: Mapped[str] = mapped_column(TSVECTOR)
    payload: Mapped[dict] = mapped_column(J)


class MaterialChunk(Base):
    """Only real WorkVersion-backed fragments receive the frozen Chunk DTO."""
    __tablename__ = "material_chunks"
    __table_args__ = (
        UniqueConstraint("owner_id", "id"), UniqueConstraint("fragment_id", "version_id"),
        ForeignKeyConstraint(["owner_id", "fragment_id"], [f"{SCHEMA}.material_fragments.owner_id", f"{SCHEMA}.material_fragments.id"]),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    fragment_id: Mapped[str] = mapped_column(String(64))
    version_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.work_versions.id"))
    payload: Mapped[dict] = mapped_column(J)


class MaterialEvidence(Base):
    __tablename__ = "material_evidence"
    __table_args__ = (
        ForeignKeyConstraint(["owner_id", "chunk_id"], [f"{SCHEMA}.material_chunks.owner_id", f"{SCHEMA}.material_chunks.id"]),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    chunk_id: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(J)


class ArtifactAsset(Base):
    """Immutable, scoped source-map/runtime assets survive PG+CAS restore."""
    __tablename__ = "material_artifact_assets"
    __table_args__ = (
        ForeignKeyConstraint(["owner_id", "representation_id"], [f"{SCHEMA}.parsed_representations.owner_id", f"{SCHEMA}.parsed_representations.id"]),
        ForeignKeyConstraint(["security_domain", "sha256"], [f"{SCHEMA}.blobs.security_domain", f"{SCHEMA}.blobs.sha256"]),
        CheckConstraint("size_bytes > 0", name="positive_asset_size"),
    )
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"), primary_key=True)
    representation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    role: Mapped[str] = mapped_column(String(60), primary_key=True)
    security_domain: Mapped[str] = mapped_column(String(100))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    provenance: Mapped[dict] = mapped_column(J)
