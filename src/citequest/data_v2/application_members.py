"""Indexed immutable membership; large qualification evidence lives in real CAS."""

from __future__ import annotations

import hashlib
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, ForeignKeyConstraint, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from citequest.v2.models import SCHEMA, Base, J
from citequest.v2.services import digest


class ApplicationMember(Base):
    __tablename__ = "application_corpus_members"
    __table_args__ = (
        ForeignKeyConstraint(
            ["snapshot_id", "native_id"], [f"{SCHEMA}.ingest_rows.snapshot_id", f"{SCHEMA}.ingest_rows.native_id"]
        ),
        UniqueConstraint("snapshot_id", "rank"),
        UniqueConstraint("snapshot_id", "native_id"),
        UniqueConstraint("snapshot_id", "representation_id"),
        UniqueConstraint("snapshot_id", "work_id"),
        UniqueConstraint("snapshot_id", "work_version_id"),
        UniqueConstraint("snapshot_id", "source_record_id"),
        UniqueConstraint("snapshot_id", "raw_PDF_sha256"),
        Index("ix_application_member_version", "work_version_id", "snapshot_id"),
        CheckConstraint(
            "rank >= 0 AND rank < 3000 AND resolution_revision >= 1", name="valid_fixed_application_position"
        ),
    )
    snapshot_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.ingest_snapshots.id"), primary_key=True)
    source_row_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.ingest_rows.id"), primary_key=True)
    rank: Mapped[int] = mapped_column(Integer)
    native_id: Mapped[str] = mapped_column(String(512))
    resolution_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.ingest_resolutions.id"))
    resolution_revision: Mapped[int] = mapped_column(Integer)
    resolution_proof_sha256: Mapped[str] = mapped_column(String(64))
    representation_id: Mapped[str] = mapped_column(ForeignKey(f"{SCHEMA}.ingest_native_representations.id"))
    work_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.works.id"))
    work_version_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.work_versions.id"))
    source_record_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.source_records.id"))
    draft_sha256: Mapped[str] = mapped_column(String(64))
    source_text_sha256: Mapped[str] = mapped_column(String(64))
    source_pointer_sha256: Mapped[str] = mapped_column(String(64))
    source_record_sha256: Mapped[str] = mapped_column(String(64))
    raw_PDF_sha256: Mapped[str] = mapped_column(String(64))
    source_reference: Mapped[dict] = mapped_column(J)
    member_sha256: Mapped[str] = mapped_column(String(64))
    record_sha256: Mapped[str] = mapped_column(String(64))


RECORD_FIELDS = (
    "snapshot_id",
    "source_row_id",
    "rank",
    "native_id",
    "resolution_id",
    "resolution_revision",
    "resolution_proof_sha256",
    "representation_id",
    "work_id",
    "work_version_id",
    "source_record_id",
    "draft_sha256",
    "source_text_sha256",
    "source_pointer_sha256",
    "source_record_sha256",
    "raw_PDF_sha256",
    "member_sha256",
)
UUID_FIELDS = {"snapshot_id", "source_row_id", "resolution_id", "work_id", "work_version_id", "source_record_id"}


def record_sha(value: dict) -> str:
    # These fields have bounded ASCII UUID/int/native-ID/SHA representations.
    # PostgreSQL uses the identical newline framing in its adoption guard.
    return hashlib.sha256("\n".join(str(value[name]) for name in RECORD_FIELDS).encode("ascii")).hexdigest()


def membership_records(snapshot_id: UUID, members: list[dict]) -> list[dict]:
    records = []
    for member in members:
        row = {name: member[name] for name in RECORD_FIELDS if name not in {"snapshot_id", "member_sha256"}}
        row.update(
            snapshot_id=str(snapshot_id), member_sha256=digest(member), source_reference=member["source_reference"]
        )
        row["record_sha256"] = record_sha(row)
        records.append({key: UUID(value) if key in UUID_FIELDS else value for key, value in row.items()})
    return records


def records_sha(records: list[dict]) -> str:
    return hashlib.sha256(
        ("\n".join(row["record_sha256"] for row in sorted(records, key=lambda row: row["rank"])) + "\n").encode("ascii")
    ).hexdigest()
