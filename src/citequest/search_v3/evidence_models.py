"""Separate chunk retrieval receipts; paper result grouping stays unchanged."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from citequest.index_v2 import models as index_models  # noqa: F401
from citequest.v2.models import SCHEMA, Base, Identity, J


class EvidenceRetrieval(Identity, Base):
    __tablename__ = "evidence_retrievals"
    __table_args__ = (
        UniqueConstraint("id", "generation_id"),
        CheckConstraint("epoch>=1 AND candidate_count BETWEEN 0 AND 20 AND delivered_count>=0 AND masked_count>=0 "
                        "AND candidate_count=delivered_count+masked_count", name="bounded_chunk_retrieval"),
        Index("ix_evidence_retrieval_owner_expiry", "owner_id", "expires_at", "id"),
        Index("ix_evidence_retrieval_generation_expiry", "generation_id", "expires_at"),
    )
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    generation_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.index_generations.id"))
    epoch: Mapped[int] = mapped_column(Integer)
    request: Mapped[dict] = mapped_column(J)
    request_sha256: Mapped[str] = mapped_column(String(64))
    catalog_binding_sha256: Mapped[str] = mapped_column(String(64))
    candidate_count: Mapped[int] = mapped_column(Integer)
    delivered_count: Mapped[int] = mapped_column(Integer)
    masked_count: Mapped[int] = mapped_column(Integer)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class EvidenceRetrievalHit(Base):
    __tablename__ = "evidence_retrieval_hits"
    __table_args__ = (
        ForeignKeyConstraint(["retrieval_id", "generation_id"], [f"{SCHEMA}.evidence_retrievals.id", f"{SCHEMA}.evidence_retrievals.generation_id"], ondelete="CASCADE"),
        ForeignKeyConstraint(["generation_id", "projection_id"], [f"{SCHEMA}.index_units.generation_id", f"{SCHEMA}.index_units.projection_id"]),
        ForeignKeyConstraint(["generation_id", "chunk_id"], [f"{SCHEMA}.index_chunks.generation_id", f"{SCHEMA}.index_chunks.chunk_id"]),
        UniqueConstraint("retrieval_id", "chunk_id"),
        CheckConstraint("ordinal BETWEEN 1 AND 20 AND score>=-1.0e308 AND score<=1.0e308", name="bounded_evidence_hit"),
    )
    retrieval_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    generation_id: Mapped[UUID] = mapped_column(Uuid)
    projection_id: Mapped[UUID] = mapped_column(Uuid)
    chunk_id: Mapped[str] = mapped_column(String(64))
    score: Mapped[float] = mapped_column(Float)
