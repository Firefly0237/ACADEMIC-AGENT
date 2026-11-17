"""Typed saved-paper identities and user organization; importing performs no DDL."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, Integer, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from citequest.v2.models import SCHEMA, Base, Identity, J


class SavedPaperReference(Identity, Base):
    __tablename__ = "saved_paper_references"
    __table_args__ = (UniqueConstraint("owner_id", "id"), UniqueConstraint("owner_id", "paper_kind", "paper_id"))
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    paper_kind: Mapped[str] = mapped_column(String(30))
    paper_id: Mapped[str] = mapped_column(String(64))
    work_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.works.id"))
    source_snapshot_id: Mapped[UUID] = mapped_column(Uuid)
    captured_metadata: Mapped[dict] = mapped_column(J)
    captured_metadata_sha256: Mapped[str] = mapped_column(String(64))
    source_projection_sha256: Mapped[str] = mapped_column(String(64))
    source_pointer_sha256: Mapped[str] = mapped_column(String(64))
    revision: Mapped[int] = mapped_column(Integer, default=1)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SavedPaperCollection(Base):
    __tablename__ = "saved_paper_collections"
    __table_args__ = (
        ForeignKeyConstraint(
            ["owner_id", "saved_paper_id"],
            [f"{SCHEMA}.saved_paper_references.owner_id", f"{SCHEMA}.saved_paper_references.id"],
        ),
        ForeignKeyConstraint(
            ["owner_id", "collection_id"], [f"{SCHEMA}.collections.owner_id", f"{SCHEMA}.collections.id"]
        ),
    )
    owner_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    saved_paper_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    collection_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)


class SavedPaperTag(Base):
    __tablename__ = "saved_paper_tags"
    __table_args__ = (
        ForeignKeyConstraint(
            ["owner_id", "saved_paper_id"],
            [f"{SCHEMA}.saved_paper_references.owner_id", f"{SCHEMA}.saved_paper_references.id"],
        ),
    )
    owner_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    saved_paper_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    tag: Mapped[str] = mapped_column(String(100), primary_key=True)


class SavedPaperNote(Identity, Base):
    __tablename__ = "saved_paper_notes"
    __table_args__ = (
        UniqueConstraint("owner_id", "id"),
        ForeignKeyConstraint(
            ["owner_id", "saved_paper_id"],
            [f"{SCHEMA}.saved_paper_references.owner_id", f"{SCHEMA}.saved_paper_references.id"],
        ),
    )
    owner_id: Mapped[UUID] = mapped_column(Uuid)
    saved_paper_id: Mapped[UUID] = mapped_column(Uuid)
    body: Mapped[str] = mapped_column(Text)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
