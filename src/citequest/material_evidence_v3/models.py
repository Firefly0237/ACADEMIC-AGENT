"""Private material generations and transactional delivery jobs, independent of papers."""
from datetime import datetime
from uuid import UUID
from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, Index, Integer, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column
from citequest.materials_v2 import models as material_models  # noqa: F401
from citequest.v2.models import SCHEMA, Base, Identity, J


class MaterialIndexGeneration(Identity, Base):
    __tablename__ = "material_index_generations"
    __table_args__ = (
        UniqueConstraint("id", "owner_id", "file_id"),
        UniqueConstraint("id", "owner_id", "file_id", "representation_id"),
        UniqueConstraint("owner_id", "file_id", "representation_id", "file_revision", "backend_profile_sha256", "schema_profile_sha256", "tokenizer_profile_sha256", "collection_name"),
        ForeignKeyConstraint(["owner_id", "file_id"], [f"{SCHEMA}.files.owner_id", f"{SCHEMA}.files.id"]),
        ForeignKeyConstraint(["owner_id", "representation_id"], [f"{SCHEMA}.parsed_representations.owner_id", f"{SCHEMA}.parsed_representations.id"]),
        ForeignKeyConstraint(["owner_id", "material_version_id"], [f"{SCHEMA}.material_versions.owner_id", f"{SCHEMA}.material_versions.id"]),
        Index("ix_material_index_generation_file", "owner_id", "file_id", "state"),
    )
    owner_id: Mapped[UUID] = mapped_column(Uuid)
    file_id: Mapped[UUID] = mapped_column(Uuid)
    representation_id: Mapped[str] = mapped_column(String(64))
    material_version_id: Mapped[UUID] = mapped_column(Uuid)
    file_revision: Mapped[int] = mapped_column(Integer)
    source_file_sha256: Mapped[str] = mapped_column(String(64))
    backend_profile_sha256: Mapped[str] = mapped_column(String(64))
    schema_profile_sha256: Mapped[str] = mapped_column(String(64))
    tokenizer_profile_sha256: Mapped[str] = mapped_column(String(64))
    collection_name: Mapped[str] = mapped_column(String(128))
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    expected_rows: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(30), default="ready")
    readback_proof: Mapped[dict | None] = mapped_column(J)


class MaterialIndexRow(Base):
    __tablename__ = "material_index_rows"
    __table_args__ = (
        ForeignKeyConstraint(["generation_id", "owner_id", "file_id", "representation_id"],
            [f"{SCHEMA}.material_index_generations.id", f"{SCHEMA}.material_index_generations.owner_id", f"{SCHEMA}.material_index_generations.file_id", f"{SCHEMA}.material_index_generations.representation_id"]),
        ForeignKeyConstraint(["owner_id", "fragment_id"], [f"{SCHEMA}.material_fragments.owner_id", f"{SCHEMA}.material_fragments.id"]),
        Index("ix_material_index_row_delivery", "generation_id", "state", "ordinal"),
        UniqueConstraint("generation_id", "fragment_id"),
    )
    generation_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    row_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[UUID] = mapped_column(Uuid)
    file_id: Mapped[UUID] = mapped_column(Uuid)
    representation_id: Mapped[str] = mapped_column(String(64))
    fragment_id: Mapped[str] = mapped_column(String(64))
    ordinal: Mapped[int] = mapped_column(Integer)
    row_sha256: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(J)
    state: Mapped[str] = mapped_column(String(20), default="ready")


class MaterialIndexJob(Identity, Base):
    """A transactionally enqueued PG outbox job, not a second Agent scheduler."""
    __tablename__ = "material_index_jobs"
    __table_args__ = (UniqueConstraint("generation_id", "operation"), Index("ix_material_index_job_queue", "state", "created_at", "id"))
    generation_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.material_index_generations.id"))
    operation: Mapped[str] = mapped_column(String(20), default="upsert")
    state: Mapped[str] = mapped_column(String(20), default="ready")
    epoch: Mapped[int] = mapped_column(Integer, default=0)
    worker_id: Mapped[str | None] = mapped_column(String(100))
    lease_id: Mapped[UUID | None] = mapped_column(Uuid)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    batch_ids: Mapped[list[str]] = mapped_column(J, default=list)
    attempts: Mapped[list[dict]] = mapped_column(J, default=list)
    error_code: Mapped[str | None] = mapped_column(String(100))


class MaterialIndexPointer(Base):
    __tablename__ = "material_index_pointers"
    __table_args__ = (
        ForeignKeyConstraint(["owner_id", "file_id"], [f"{SCHEMA}.files.owner_id", f"{SCHEMA}.files.id"]),
        ForeignKeyConstraint(["generation_id", "owner_id", "file_id"], [f"{SCHEMA}.material_index_generations.id", f"{SCHEMA}.material_index_generations.owner_id", f"{SCHEMA}.material_index_generations.file_id"]),
    )
    owner_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    file_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    generation_id: Mapped[UUID | None] = mapped_column(Uuid)
    epoch: Mapped[int] = mapped_column(Integer, default=0)


class MaterialIndexAttempt(Identity, Base):
    __tablename__ = "material_index_attempts"
    __table_args__ = (Index("ix_material_index_attempt_job_outcome", "job_id", "write_outcome", "created_at"),)
    job_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.material_index_jobs.id"))
    epoch: Mapped[int] = mapped_column(Integer)
    operation: Mapped[str] = mapped_column(String(20))
    request_sha256: Mapped[str] = mapped_column(String(64))
    batch_ids: Mapped[list[str]] = mapped_column(J)
    write_outcome: Mapped[str] = mapped_column(String(20))
    receipt: Mapped[dict | None] = mapped_column(J)
