"""Source update state on the existing PostgreSQL business registry."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from citequest.v2.models import SCHEMA, Base, Identity, J


class NetworkState(Base):
    __tablename__ = "source_network_states"
    __table_args__ = (CheckConstraint("provider IN ('arxiv','crossref') AND fence >= 0", name="source_network_provider_and_fence"),)
    provider: Mapped[str] = mapped_column(String(30), primary_key=True)
    policy_sha256: Mapped[str] = mapped_column(String(64))
    fence: Mapped[int] = mapped_column(BigInteger, default=0)
    last_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_attempt_id: Mapped[UUID | None] = mapped_column(Uuid)


class NetworkAttempt(Identity, Base):
    __tablename__ = "source_network_attempts"
    __table_args__ = (Index("ix_source_network_attempt_provider_state", "provider", "state", "created_at"),
                      CheckConstraint("state IN ('in_flight','settled','settled_error','unknown') AND fence > 0", name="source_network_attempt_state"))
    provider: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.source_network_states.provider"))
    fence: Mapped[int] = mapped_column(BigInteger)
    url: Mapped[str] = mapped_column(Text)
    request_sha256: Mapped[str] = mapped_column(String(64))
    owner_pid: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    state: Mapped[str] = mapped_column(String(30))
    receipt: Mapped[dict | None] = mapped_column(J)


class UpdateWindow(Identity, Base):
    __tablename__ = "source_update_windows"
    __table_args__ = (Index("ix_source_update_window_dispatch", "state", "created_at", "id"),
                      CheckConstraint("lower_bound < upper_bound", name="source_window_bounds"),
                      CheckConstraint("page_count >= 0 AND record_count >= 0 AND network_calls >= 0 AND active_seconds >= 0", name="source_window_counts"),
                      CheckConstraint("state IN ('queued','running','observed','partial','failed','cancelled')", name="source_window_state"))
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    security_domain: Mapped[str] = mapped_column(String(100))
    source_namespace: Mapped[str] = mapped_column(String(100))
    lower_bound: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    upper_bound: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    scope_sha256: Mapped[str] = mapped_column(String(64))
    managed_ids: Mapped[list[str]] = mapped_column(J)
    config: Mapped[dict] = mapped_column(J)
    config_sha256: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(30), default="queued")
    phase: Mapped[str] = mapped_column(String(40), default="discovery")
    cursor: Mapped[dict] = mapped_column(J, default=dict)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    network_calls: Mapped[int] = mapped_column(Integer, default=0)
    active_seconds: Mapped[float] = mapped_column(Float, default=0)
    error: Mapped[dict | None] = mapped_column(J)
    lease_id: Mapped[UUID | None] = mapped_column(Uuid)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class UpdatePage(Identity, Base):
    __tablename__ = "source_update_pages"
    __table_args__ = (UniqueConstraint("window_id", "sequence"), CheckConstraint("sequence > 0", name="source_page_sequence"))
    window_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.source_update_windows.id"))
    sequence: Mapped[int] = mapped_column(Integer)
    phase: Mapped[str] = mapped_column(String(40))
    request_sha256: Mapped[str] = mapped_column(String(64))
    request: Mapped[dict] = mapped_column(J)
    raw_security_domain: Mapped[str] = mapped_column(String(100))
    raw_sha256: Mapped[str | None] = mapped_column(String(64))
    body_complete: Mapped[bool] = mapped_column(Boolean)
    receipt: Mapped[dict] = mapped_column(J)
    parsed: Mapped[dict | None] = mapped_column(J)


class UpdateObservation(Identity, Base):
    __tablename__ = "source_update_observations"
    __table_args__ = (UniqueConstraint("page_id", "record_ordinal"), Index("ix_source_observation_native", "security_domain", "source_namespace", "native_id", "created_at", "id"))
    window_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.source_update_windows.id"))
    page_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.source_update_pages.id"))
    security_domain: Mapped[str] = mapped_column(String(100))
    source_namespace: Mapped[str] = mapped_column(String(100))
    native_id: Mapped[str] = mapped_column(String(512))
    source_revision: Mapped[str | None] = mapped_column(String(100))
    record_ordinal: Mapped[int] = mapped_column(Integer)
    record_sha256: Mapped[str] = mapped_column(String(64))
    projection_sha256: Mapped[str] = mapped_column(String(64))
    locator: Mapped[dict] = mapped_column(J)
    metadata_payload: Mapped[dict] = mapped_column(J)
    state: Mapped[str] = mapped_column(String(50))
    metadata_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    verified_status_asof: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    work_id: Mapped[UUID | None] = mapped_column(ForeignKey(f"{SCHEMA}.works.id"))


class UpdateSeal(Identity, Base):
    __tablename__ = "source_update_seals"
    __table_args__ = (CheckConstraint("NOT publication_eligible", name="accepted_profile_withholds_publication"),)
    window_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.source_update_windows.id"), unique=True)
    security_domain: Mapped[str] = mapped_column(String(100))
    source_namespace: Mapped[str] = mapped_column(String(100))
    scope_sha256: Mapped[str] = mapped_column(String(64))
    membership_sha256: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(J)
    seal_sha256: Mapped[str] = mapped_column(String(64))
    publication_eligible: Mapped[bool] = mapped_column(Boolean, default=False)


class SealMember(Base):
    __tablename__ = "source_update_seal_members"
    seal_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.source_update_seals.id"), primary_key=True)
    source_namespace: Mapped[str] = mapped_column(String(100), primary_key=True)
    native_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    observation_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.source_update_observations.id"))
    projection_sha256: Mapped[str] = mapped_column(String(64))


class Checkpoint(Base):
    __tablename__ = "source_update_checkpoints"
    security_domain: Mapped[str] = mapped_column(String(100), primary_key=True)
    source_namespace: Mapped[str] = mapped_column(String(100), primary_key=True)
    scope_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(30), primary_key=True)
    seal_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.source_update_seals.id"))
    value: Mapped[dict] = mapped_column(J)


class SourceDenial(Base):
    __tablename__ = "source_update_denials"
    __table_args__ = (CheckConstraint("active", name="source_quarantine_is_monotone"),)
    security_domain: Mapped[str] = mapped_column(String(100), primary_key=True)
    source_namespace: Mapped[str] = mapped_column(String(100), primary_key=True)
    native_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    source_revision: Mapped[str] = mapped_column(String(100), primary_key=True)
    observation_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.source_update_observations.id"))
    reason: Mapped[str] = mapped_column(String(80))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    proof: Mapped[dict] = mapped_column(J)
