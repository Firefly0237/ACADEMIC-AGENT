"""Real PG fault controls with explicitly mocked native results, never corpus evidence."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select

from citequest.materials_v2 import models as m
from citequest.materials_v2 import processor as module
from citequest.materials_v2.processor import MaterialProcessor
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
    source = "S04 SYNTHETIC FAULT CONTROL " + str(uuid4())
    with factory.begin() as db:
        db.add(core.Owner(id=actor.owner_id, name="S04 synthetic fault controls (mock native)"))
    with factory.begin() as db:
        file = MaterialService(db, actor, STORE).ingest_text({"title": "S04 synthetic fault control", "text": source, "format": "txt"}, key=uuid4())
    return actor, UUID(file["file_id"]), source


def make_processor(factory, clock):
    return MaterialProcessor(factory, STORE, parser_python=ROOT / "data/runtime-v2/.venv/Scripts/python.exe", root=ROOT,
                             artifact_root=ROOT / "data/runtime-s04/fault-control-artifacts", clock=lambda: clock[0])


def fake_native(calls: list, *, after_parse=None):
    def run(request: dict, **kwargs: dict) -> dict:
        Path(kwargs["output"]).mkdir(parents=True)
        calls.append((request.get("operation", "parse"), kwargs["deadline_seconds"]))
        source_path = Path(request["source_path"])
        if request.get("operation") == "chunk":
            data = json.loads(source_path.read_text(encoding="utf-8"))
            block = data["blocks"][0]
            chunk = {"source_text": block["source_quote"], "source_text_sha256": hashlib.sha256(block["source_quote"].encode()).hexdigest(),
                     "index_text": block["source_quote"], "locator": block["locator"], "granularity": "paragraph", "token_count": 1,
                     "tokenizer_version": "synthetic-fault-control-NOT-REAL-TOKENS", "chunker_version": "synthetic-fault-control"}
            return {"result": {"state": "ready", "code": "CHUNKS_READY", "original_file_sha256": data["file_sha256"],
                               "tokenizer_profile_sha256": request["expected_tokenizer_profile_sha256"], "chunks": [chunk]}}
        kwargs["on_validated"]()
        quote = source_path.read_text(encoding="utf-8")
        if after_parse:
            after_parse()
        return {"validation_phase_observed": True, "elapsed_seconds": 0.001, "deadline_seconds": kwargs["deadline_seconds"],
                "result": {"state": "ready", "code": "TEXT_PARSED", "source_sha256": request["source_sha256"],
                           "parser_versions": {"synthetic_fault_control_NOT_REAL_PARSER": "1"},
                           "blocks": [{"block_id": "fixture:1", "source_quote": quote,
                                       "locator": {"kind": "text", "file_sha256": request["source_sha256"], "char_start": 0, "char_end": len(quote), "line_1based": 1}}]}}
    return run


def test_deadline_after_parse_does_not_dispatch_chunk_or_publish(factory, monkeypatch):
    actor, file_id, _ = prepare(factory)
    clock, calls = [0.0], []
    monkeypatch.setattr(module, "run_isolated", fake_native(calls, after_parse=lambda: clock.__setitem__(0, 121.0)))
    result = make_processor(factory, clock).run_next(file_id=file_id)
    assert result["ingestion_state"] == "failed" and result["error_code"] == "PRODUCT_DEADLINE_EXCEEDED"
    assert [operation for operation, _ in calls] == ["parse"]
    with factory.begin() as db:
        assert MaterialService(db, actor, STORE).get_ingestion(file_id)["execution_success"] is False
        assert db.scalar(select(func.count()).select_from(m.ParsedRepresentation).where(m.ParsedRepresentation.file_id == file_id)) == 0


def test_expiry_during_publish_rolls_back_ready_representation_and_fragments(factory, monkeypatch):
    actor, file_id, _ = prepare(factory)
    clock, calls = [0.0], []
    monkeypatch.setattr(module, "run_isolated", fake_native(calls))
    original = MaterialService._publish_formal
    reached = []

    def expire(service, file, ingestion):
        original(service, file, ingestion)
        reached.append("before_commit")
        clock[0] = 121.0

    monkeypatch.setattr(MaterialService, "_publish_formal", expire)
    result = make_processor(factory, clock).run_next(file_id=file_id)
    assert result["ingestion_state"] == "failed" and result["error_code"] == "PRODUCT_DEADLINE_EXCEEDED"
    assert reached == ["before_commit"] and len(calls) == 2
    with factory.begin() as db:
        status = MaterialService(db, actor, STORE).get_ingestion(file_id)
        assert status["execution_outcome"] == "failed" and status["material_source_ready"] is False
        assert db.scalar(select(func.count()).select_from(m.MaterialFragment).where(m.MaterialFragment.file_id == file_id)) == 0


@pytest.mark.parametrize("kind", ["late", "ack_lost"])
def test_post_commit_late_or_unknown_is_not_success_and_not_republished(factory, monkeypatch, kind):
    actor, file_id, _ = prepare(factory)
    clock, calls = [0.0], []
    monkeypatch.setattr(module, "run_isolated", fake_native(calls))
    processor = make_processor(factory, clock)
    original = processor._finish

    def after_commit(*args, **kwargs):
        result = original(*args, **kwargs)
        if kind == "late":
            clock[0] = 121.0
        else:
            raise OSError("synthetic lost commit acknowledgement")
        return result

    monkeypatch.setattr(processor, "_finish", after_commit)
    result = processor.run_next(file_id=file_id)
    assert result["execution_success"] is False
    with factory.begin() as db:
        status = MaterialService(db, actor, STORE).get_ingestion(file_id)
        assert status["execution_outcome"] == ("late" if kind == "late" else "uncertain")
        assert status["execution_success"] is False
        count = db.scalar(select(func.count()).select_from(m.MaterialFragment).where(m.MaterialFragment.file_id == file_id))
        assert count == 1  # Synthetic fault output, not a real-parser positive result.
    assert processor.run_next(file_id=file_id) == {"status": "empty"}
    assert len(calls) == 2


def test_cancel_between_nodes_stops_dispatch_and_publication(factory, monkeypatch):
    actor, file_id, _ = prepare(factory)
    clock, calls = [0.0], []

    def cancel():
        with factory.begin() as db:
            MaterialService(db, actor, STORE).cancel(file_id)

    monkeypatch.setattr(module, "run_isolated", fake_native(calls, after_parse=cancel))
    result = make_processor(factory, clock).run_next(file_id=file_id)
    assert result["ingestion_state"] == "cancelled" and len(calls) == 1
    with factory.begin() as db:
        status = MaterialService(db, actor, STORE).get_ingestion(file_id)
        assert status["execution_outcome"] == "cancelled" and status["material_source_ready"] is False
