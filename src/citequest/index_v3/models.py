"""Migrator-only, append-only backend adoption; business clients cannot approve."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from citequest.index_v2 import models as index_models  # noqa: F401
from citequest.v2.models import SCHEMA, Base, J


class BackendAdoption(Base):
    __tablename__ = "index_backend_adoptions"
    generation_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.index_generations.id"), primary_key=True)
    backend_profile_sha256: Mapped[str] = mapped_column(String(64))
    expected_manifest_sha256: Mapped[str] = mapped_column(String(64))
    standard_quality_generation_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.index_generations.id"))
    evaluation_sha256: Mapped[str] = mapped_column(String(64))
    quality_report_sha256: Mapped[str] = mapped_column(String(64))
    approval_sha256: Mapped[str] = mapped_column(String(64))
    evaluation_queries: Mapped[int] = mapped_column(Integer)
    evaluation_qrels_edges: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(J)
    adopted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BackendBinding(Base):
    __tablename__ = "index_backend_bindings"
    generation_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.index_generations.id"), primary_key=True)
    backend_profile_sha256: Mapped[str] = mapped_column(String(64))
    expected_manifest_sha256: Mapped[str] = mapped_column(String(64))
    profile: Mapped[dict] = mapped_column(J)
    profile_canonical: Mapped[str] = mapped_column(Text)
    bound_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
