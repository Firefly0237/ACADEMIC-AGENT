"""Targeted real parser/chunker/PG runs; no synthetic output substitutes."""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from citequest.materials_v2 import models as m
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
    with database.connect() as connection:
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        current = connection.exec_driver_sql("SELECT version_num FROM cq_app.alembic_version").scalar_one()
        assert "s04_0001" in {r.revision for r in ScriptDirectory.from_config(Config(str(ROOT / "alembic-app.ini"))).iterate_revisions(current, "base")}
    yield sessions(database)
    database.dispose()


def test_real_text_pipeline_source_ready_without_fake_paper_or_agent_qa(factory):
    actor = Actor(uuid4())
    source = "S04 synthetic retrieval evidence.\r\n第二行数学𝒙😀 original text.\n"
    with factory.begin() as db:
        db.add(core.Owner(id=actor.owner_id, name="S04 synthetic real processor fixture"))
    with factory.begin() as db:
        receipt = MaterialService(db, actor, STORE).ingest_text({"title": "S04 synthetic processor text", "text": source, "format": "paste"}, key=uuid4())
    file_id = UUID(receipt["file_id"])
    processor = MaterialProcessor(factory, STORE, parser_python=ROOT / "data/runtime-v2/.venv/Scripts/python.exe", root=ROOT,
                                  artifact_root=ROOT / "data/runtime-s04/ingestion-artifacts")
    result = processor.run_next(file_id=file_id)
    assert result["status"] == "processed" and result["ingestion_state"] == "ready", result
    assert result["execution_outcome"] == "within_deadline" and result["execution_success"] is True
    with factory.begin() as db:
        service = MaterialService(db, actor, STORE)
        file = service.get_file(file_id)
        assert file["status"] == "ready" and file["coverage"] == "text_full"
        assert file["can_preview"] and file["can_download"] and file["can_qa"] is False
        status = service.get_ingestion(file_id)
        assert [event["state"] for event in status["events"]] == ["queued", "validating", "parsing", "ready"]
        assert status["material_source_ready"] is True and status["formal_evidence_ready"] is False
        assert status["identity_status"] == "unverified" and status["binding_source"] == "none"
        representation = service.get_representation(file_id)
        fragments = service.get_fragments(file_id)
        assert fragments["items"] and fragments["representation_id"] == representation["representation_id"]
        for item in fragments["items"]:
            loc = item["locator"]
            assert source[loc["char_start"]:loc["char_end"]] == item["source_text"]
            assert 1 <= item["token_count"] <= 2048
            assert item["work_version_id"] is None
        assert service.get_fragments(file_id, query="retrieval")["items"]
        assert service.get_fragments(file_id, query="第二行")["items"]
        blob, _, _ = service.read_content(file_id)
        assert STORE.read(blob).decode() == source
        assert db.scalar(select(m.MaterialChunk.id).where(m.MaterialChunk.owner_id == actor.owner_id)) is None
        assert db.scalar(select(m.MaterialVersion.id).where(m.MaterialVersion.file_id == file_id)) is not None
