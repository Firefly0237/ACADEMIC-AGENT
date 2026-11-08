"""New research intake tables; importing this module never performs DDL."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from citequest.v2.models import SCHEMA, Base, Identity, J


class ResearchBriefRow(Identity, Base):
    __tablename__ = "research_briefs"
    __table_args__ = (
        UniqueConstraint("owner_id", "id"),
        ForeignKeyConstraint(["owner_id", "project_id"], [f"{SCHEMA}.projects.owner_id", f"{SCHEMA}.projects.id"]),
        CheckConstraint("revision>=1", name="positive_revision"),
        Index("ix_research_brief_owner_project_created", "owner_id", "project_id", "created_at", "id"),
    )
    owner_id: Mapped[UUID] = mapped_column(Uuid)
    project_id: Mapped[UUID] = mapped_column(Uuid)
    revision: Mapped[int] = mapped_column(Integer)
    document: Mapped[dict] = mapped_column(J)
    document_sha256: Mapped[str] = mapped_column(String(64))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ResearchBriefRevision(Base):
    __tablename__ = "research_brief_revisions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["owner_id", "brief_id"], [f"{SCHEMA}.research_briefs.owner_id", f"{SCHEMA}.research_briefs.id"]
        ),
        CheckConstraint("revision>=1", name="positive_revision"),
    )
    brief_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[UUID] = mapped_column(Uuid)
    operation_id: Mapped[UUID] = mapped_column(Uuid)
    document: Mapped[dict] = mapped_column(J)
    document_sha256: Mapped[str] = mapped_column(String(64))
    change_kind: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
