"""SQLAlchemy 2 business metadata. Alembic owns only cq_app schema."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SCHEMA = "cq_app"
J = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


class Base(DeclarativeBase):
    metadata = MetaData(
        schema=SCHEMA,
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_N_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        },
    )


class Identity:
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Owner(Identity, Base):
    __tablename__ = "owners"
    name: Mapped[str] = mapped_column(String(500))
    can_publish: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")


class LocalSession(Identity, Base):
    __tablename__ = "local_sessions"
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"), index=True)
    token_sha256: Mapped[str] = mapped_column(String(64), unique=True)
    csrf_sha256: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class Project(Identity, Base):
    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("owner_id", "id"),)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    title: Mapped[str] = mapped_column(String(300))
    revision: Mapped[int] = mapped_column(Integer, default=1)


class Work(Identity, Base):
    __tablename__ = "works"
    __table_args__ = (CheckConstraint("year IS NULL OR year BETWEEN 1000 AND 2200", name="year_range"),)
    title: Mapped[str] = mapped_column(Text)
    title_normalized: Mapped[str] = mapped_column(Text, index=True)
    authors: Mapped[list[str] | None] = mapped_column(J)
    year: Mapped[int | None] = mapped_column(Integer)
    public: Mapped[bool] = mapped_column(Boolean, default=False)
    security_domain: Mapped[str] = mapped_column(String(100), index=True)
    metadata_revision: Mapped[int] = mapped_column(Integer, default=1)
    field_provenance: Mapped[dict] = mapped_column(J, default=dict)
    canonical_redirect_work_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.works.id"))


class WorkAccess(Base):
    __tablename__ = "work_access"
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"), primary_key=True)
    work_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.works.id"), primary_key=True)
    acl_epoch: Mapped[int] = mapped_column(Integer, default=0)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class Identifier(Base):
    __tablename__ = "work_identifiers"
    security_domain: Mapped[str] = mapped_column(String(100), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(40), primary_key=True)
    normalized: Mapped[str] = mapped_column(String(512), primary_key=True)
    work_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.works.id"), index=True)
    raw: Mapped[str] = mapped_column(String(512))
    normalizer_version: Mapped[str] = mapped_column(String(50))


class WorkVersion(Identity, Base):
    __tablename__ = "work_versions"
    work_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.works.id"), index=True)
    security_domain: Mapped[str] = mapped_column(String(100), index=True)
    source_sha256: Mapped[str] = mapped_column(String(64))
    version_label: Mapped[str | None] = mapped_column(String(100))
    kind: Mapped[str] = mapped_column(String(40))


class SourceIdentity(Base):
    __tablename__ = "source_identities"
    __table_args__ = (
        UniqueConstraint("security_domain", "source", "native_id", "work_id", name="uq_source_identity_work"),
    )
    security_domain: Mapped[str] = mapped_column(String(100), primary_key=True)
    source: Mapped[str] = mapped_column(String(100), primary_key=True)
    native_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    work_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.works.id"))


class SourceRecord(Identity, Base):
    __tablename__ = "source_records"
    __table_args__ = (
        UniqueConstraint(
            "security_domain",
            "source",
            "native_id",
            "source_revision",
            "raw_sha256",
            name="uq_source_record_scoped_revision",
            postgresql_nulls_not_distinct=True,
        ),
        ForeignKeyConstraint(
            ["security_domain", "source", "native_id", "work_id"],
            [
                f"{SCHEMA}.source_identities.security_domain",
                f"{SCHEMA}.source_identities.source",
                f"{SCHEMA}.source_identities.native_id",
                f"{SCHEMA}.source_identities.work_id",
            ],
            name="fk_source_record_scoped_identity",
        ),
        ForeignKeyConstraint(
            ["raw_security_domain", "raw_sha256"], [f"{SCHEMA}.blobs.security_domain", f"{SCHEMA}.blobs.sha256"]
        ),
    )
    security_domain: Mapped[str] = mapped_column(String(100))
    work_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.works.id"))
    source: Mapped[str] = mapped_column(String(100))
    native_id: Mapped[str] = mapped_column(String(512))
    source_revision: Mapped[str | None] = mapped_column(String(100))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_security_domain: Mapped[str] = mapped_column(String(100))
    raw_sha256: Mapped[str] = mapped_column(String(64))
    source_url: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(Text)
    rights_status: Mapped[str] = mapped_column(String(20), default="unknown")
    payload: Mapped[dict] = mapped_column(J)


class Blob(Base):
    __tablename__ = "blobs"
    security_domain: Mapped[str] = mapped_column(String(100), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    __table_args__ = (CheckConstraint("size_bytes > 0", name="positive_size"),)


class File(Identity, Base):
    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("owner_id", "id"),
        ForeignKeyConstraint(
            ["security_domain", "sha256"], [f"{SCHEMA}.blobs.security_domain", f"{SCHEMA}.blobs.sha256"]
        ),
        Index("ix_files_owner_created", "owner_id", "created_at", "id"),
    )
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    version_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.work_versions.id"))
    security_domain: Mapped[str] = mapped_column(String(100))
    sha256: Mapped[str] = mapped_column(String(64))
    media_type: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    rights_status: Mapped[str] = mapped_column(String(20), default="unknown")
    outbound_policy: Mapped[str] = mapped_column(String(50), default="local_only")
    revision: Mapped[int] = mapped_column(Integer, default=1)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LibraryItem(Identity, Base):
    __tablename__ = "library_items"
    __table_args__ = (
        UniqueConstraint("owner_id", "work_id"),
        UniqueConstraint("owner_id", "id"),
        Index("ix_library_owner_created", "owner_id", "created_at", "id"),
    )
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    work_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.works.id"))
    revision: Mapped[int] = mapped_column(Integer, default=1)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Collection(Identity, Base):
    __tablename__ = "collections"
    __table_args__ = (
        UniqueConstraint("owner_id", "id"),
        ForeignKeyConstraint(["owner_id", "parent_id"], [f"{SCHEMA}.collections.owner_id", f"{SCHEMA}.collections.id"]),
    )
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    name: Mapped[str] = mapped_column(String(500))
    parent_id: Mapped[UUID | None] = mapped_column(Uuid)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ItemCollection(Base):
    __tablename__ = "item_collections"
    __table_args__ = (
        ForeignKeyConstraint(
            ["owner_id", "item_id"], [f"{SCHEMA}.library_items.owner_id", f"{SCHEMA}.library_items.id"]
        ),
        ForeignKeyConstraint(
            ["owner_id", "collection_id"], [f"{SCHEMA}.collections.owner_id", f"{SCHEMA}.collections.id"]
        ),
    )
    owner_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    item_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    collection_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)


class ItemFile(Base):
    __tablename__ = "item_files"
    __table_args__ = (
        ForeignKeyConstraint(
            ["owner_id", "item_id"], [f"{SCHEMA}.library_items.owner_id", f"{SCHEMA}.library_items.id"]
        ),
        ForeignKeyConstraint(["owner_id", "file_id"], [f"{SCHEMA}.files.owner_id", f"{SCHEMA}.files.id"]),
    )
    owner_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    item_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    file_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)


class ItemTag(Base):
    __tablename__ = "item_tags"
    item_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.library_items.id"), primary_key=True)
    tag: Mapped[str] = mapped_column(String(500), primary_key=True)


class Note(Identity, Base):
    __tablename__ = "notes"
    __table_args__ = (
        ForeignKeyConstraint(
            ["owner_id", "library_item_id"], [f"{SCHEMA}.library_items.owner_id", f"{SCHEMA}.library_items.id"]
        ),
        Index("ix_notes_owner_item", "owner_id", "library_item_id"),
    )
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    library_item_id: Mapped[UUID] = mapped_column(Uuid)
    body: Mapped[str] = mapped_column(Text)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class History(Identity, Base):
    __tablename__ = "history"
    __table_args__ = (Index("ix_history_owner_created", "owner_id", "created_at", "id"),)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    kind: Mapped[str] = mapped_column(String(40))
    target_id: Mapped[str] = mapped_column(String(100))
    snapshot_id: Mapped[UUID | None] = mapped_column(Uuid)


class Operation(Base):
    __tablename__ = "idempotency_keys"
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"), primary_key=True)
    key: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    payload_sha256: Mapped[str] = mapped_column(String(64))
    result: Mapped[dict | None] = mapped_column(J)


class Outbox(Identity, Base):
    __tablename__ = "outbox"
    __table_args__ = (
        UniqueConstraint("idempotency_key"),
        Index("ix_outbox_state_created", "state", "created_at", "id"),
    )
    idempotency_key: Mapped[str] = mapped_column(String(200))
    entity_id: Mapped[str] = mapped_column(String(100))
    entity_revision: Mapped[int] = mapped_column(Integer)
    operation: Mapped[str] = mapped_column(String(20))
    payload_sha256: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(J)
    schema_version: Mapped[str] = mapped_column(String(100))
    entity_type: Mapped[str] = mapped_column(String(100))
    security_domain: Mapped[str] = mapped_column(String(100))
    state: Mapped[str] = mapped_column(String(20), default="pending")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
