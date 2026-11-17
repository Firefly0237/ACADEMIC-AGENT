"""Short-lived durable search snapshots also pin old index generations for GC."""
from __future__ import annotations
from datetime import datetime
from uuid import UUID
from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column
from citequest.index_v2 import models as index_models  # noqa: F401
from citequest.v2.models import SCHEMA, Base, Identity, J


class SearchSnapshot(Identity, Base):
    __tablename__ = "search_snapshots"
    __table_args__ = (
        UniqueConstraint("id", "generation_id"),
        CheckConstraint("page_size BETWEEN 1 AND 50 AND candidate_limit BETWEEN 1 AND 200 AND candidate_count BETWEEN 0 AND 200 AND epoch >= 0", name="bounded_search_snapshot"),
        Index("ix_search_snapshot_owner_expiry", "owner_id", "expires_at", "id"),
        Index("ix_search_snapshot_generation_expiry", "generation_id", "expires_at"),
    )
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    generation_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.index_generations.id"))
    epoch: Mapped[int] = mapped_column(Integer)
    request: Mapped[dict] = mapped_column(J)
    query_sha256: Mapped[str] = mapped_column(String(64))
    page_size: Mapped[int] = mapped_column(Integer)
    candidate_limit: Mapped[int] = mapped_column(Integer)
    candidate_count: Mapped[int] = mapped_column(Integer)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class SearchSnapshotHit(Base):
    __tablename__ = "search_snapshot_hits"
    __table_args__ = (
        ForeignKeyConstraint(["snapshot_id", "generation_id"], [f"{SCHEMA}.search_snapshots.id", f"{SCHEMA}.search_snapshots.generation_id"]),
        ForeignKeyConstraint(["generation_id", "projection_id"], [f"{SCHEMA}.index_units.generation_id", f"{SCHEMA}.index_units.projection_id"]),
        ForeignKeyConstraint(["generation_id", "chunk_id"], [f"{SCHEMA}.index_chunks.generation_id", f"{SCHEMA}.index_chunks.chunk_id"]),
        UniqueConstraint("snapshot_id", "projection_id"),
        CheckConstraint("ordinal BETWEEN 1 AND 200 AND score >= -1.0e308 AND score <= 1.0e308", name="finite_bounded_search_hit"),
    )
    snapshot_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    generation_id: Mapped[UUID] = mapped_column(Uuid)
    projection_id: Mapped[UUID] = mapped_column(Uuid)
    chunk_id: Mapped[str] = mapped_column(String(64))
    score: Mapped[float] = mapped_column(Float)
