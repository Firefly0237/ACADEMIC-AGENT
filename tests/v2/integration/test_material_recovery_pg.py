"""Abandoned execution and worker-slot controls on actual PG, with no native work."""
from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, text

from citequest.materials_v2 import models as m
from citequest.materials_v2 import processor as module
from citequest.materials_v2.processor import SLOT, MaterialProcessor
from citequest.materials_v2.services import MaterialService
from citequest.v2 import models as core
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor
from citequest.v2.storage import CAS

ROOT = Path(__file__).resolve().parents[3]
STORE = CAS(ROOT / "data/runtime-v2-app/blobs")


@pytest.fixture(scope="module")
def factory():
    sys.path.insert(0, str(ROOT / "scripts/s03"))
    from bootstrap import dsn

    database = engine(dsn("cq_s03_app"))
    yield sessions(database)
    database.dispose()


def prepare(factory):
    actor = Actor(uuid4())
    with factory.begin() as db:
        db.add(core.Owner(id=actor.owner_id, name="S04 synthetic abandoned execution control"))
    with factory.begin() as db:
        file = MaterialService(db, actor, STORE).ingest_text({"title": "S04 synthetic recovery control",
             "text": "Synthetic orphan execution " + str(uuid4()), "format": "txt"}, key=uuid4())
    processor = MaterialProcessor(factory, STORE, parser_python=ROOT / "data/runtime-v2/.venv/Scripts/python.exe", root=ROOT,
                                  artifact_root=ROOT / "data/runtime-s04/ingestion-artifacts")
    return actor, UUID(file["file_id"]), processor


@pytest.mark.parametrize("state", ["validating", "parsing"])
def test_expired_lease_archived_without_reset_or_implicit_native_retry(factory, monkeypatch, state):
    actor, file_id, processor = prepare(factory)
    claim = processor._claim(file_id, datetime.now(UTC))
    if state == "parsing":
        processor._parsing(claim)
    with factory.begin() as db:
        row = db.get(m.MaterialIngestion, claim.ingestion_id)
        original = (row.execution_id, row.execution_started_at, row.deadline_at, row.execution_budget_seconds)
        row.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    monkeypatch.setattr(module, "run_isolated", lambda *_a, **_k: pytest.fail("Expired execution dispatched native work"))
    result = processor.run_next(file_id=file_id)
    assert result["status"] == "recovered_abandoned_execution" and result["automatic_retry"] is False
    assert result["ingestion_state"] == "failed" and result["execution_success"] is False
    with factory.begin() as db:
        status = MaterialService(db, actor, STORE).get_ingestion(file_id)
        row = db.get(m.MaterialIngestion, claim.ingestion_id)
        assert (row.execution_id, row.execution_started_at, row.deadline_at, row.execution_budget_seconds) == original
        assert row.lease_id is None and row.lease_expires_at is None
        assert status["execution_outcome"] == "failed" and status["representation_id"] is None
        assert status["events"][-1]["error_code"] == "EXPIRED_MATERIAL_LEASE_NO_AUTOMATIC_RETRY"
        assert db.scalar(select(func.count()).select_from(m.ParsedRepresentation).where(m.ParsedRepresentation.file_id == file_id)) == 0
    assert processor.run_next(file_id=file_id) == {"status": "empty"}


def test_busy_global_pg_slot_does_not_claim_another_file(factory, monkeypatch):
    actor, file_id, processor = prepare(factory)
    monkeypatch.setattr(module, "run_isolated", lambda *_a, **_k: pytest.fail("Busy slot dispatched native work"))
    with factory.kw["bind"].connect().execution_options(isolation_level="AUTOCOMMIT") as guard:
        assert guard.execute(text("SELECT pg_try_advisory_lock(:slot)"), {"slot": SLOT}).scalar() is True
        try:
            assert processor.run_next(file_id=file_id) == {"status": "busy"}
        finally:
            guard.execute(text("SELECT pg_advisory_unlock(:slot)"), {"slot": SLOT})
    with factory.begin() as db:
        status = MaterialService(db, actor, STORE).get_ingestion(file_id)
        assert status["state"] == "queued" and status["execution_outcome"] == "not_started"
        MaterialService(db, actor, STORE).cancel(file_id)


def test_expired_deleted_file_is_cancelled_never_resurrected(factory, monkeypatch):
    _, file_id, processor = prepare(factory)
    claim = processor._claim(file_id, datetime.now(UTC))
    with factory.begin() as db:
        db.get(m.MaterialIngestion, claim.ingestion_id).lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        db.get(core.File, file_id).deleted_at = datetime.now(UTC)
    monkeypatch.setattr(module, "run_isolated", lambda *_a, **_k: pytest.fail("Deleted execution dispatched native work"))
    assert processor.run_next(file_id=file_id)["ingestion_state"] == "cancelled"
    with factory.begin() as db:
        row = db.get(m.MaterialIngestion, claim.ingestion_id)
        assert row.execution_outcome == "cancelled" and row.representation_id is None
