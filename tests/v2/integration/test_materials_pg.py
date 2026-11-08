"""S04 uses the real accepted PG/CAS root; all authored content is synthetic."""
from __future__ import annotations

import hashlib
import io
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select

from citequest.materials_v2 import models as m
from citequest.materials_v2.services import MaterialService
from citequest.v2 import models as core
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor, DomainError, SourceInput, WorkInput
from citequest.v2.services import Catalog
from citequest.v2.storage import CAS

ROOT = Path(__file__).resolve().parents[3]
STORE = CAS(ROOT / "data/runtime-v2-app/blobs")


@pytest.fixture(scope="module")
def factory():
    sys.path.insert(0, str(ROOT / "scripts/s03"))
    from bootstrap import dsn

    database = engine(dsn("cq_s03_app"))
    with database.connect() as connection:
        assert connection.exec_driver_sql("SELECT current_database(),current_user").one() == ("citequest_s03", "cq_s03_app")
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        current = connection.exec_driver_sql("SELECT version_num FROM cq_app.alembic_version").scalar_one()
        assert "s04_0001" in {r.revision for r in ScriptDirectory.from_config(Config(str(ROOT / "alembic-app.ini"))).iterate_revisions(current, "base")}
    yield sessions(database)
    database.dispose()


@pytest.fixture
def actors(factory):
    a, b = Actor(uuid4()), Actor(uuid4())
    with factory.begin() as db:
        db.add_all([core.Owner(id=a.owner_id, name="S04 synthetic A"), core.Owner(id=b.owner_id, name="S04 synthetic B")])
    return a, b


def body(data: bytes) -> dict:
    return {"file_name": "synthetic.txt", "media_type": "text/plain", "size_bytes": len(data),
            "expected_sha256": hashlib.sha256(data).hexdigest(), "project_id": None}


def upload(factory, actor: Actor, data: bytes, key=None) -> dict:
    with factory.begin() as db:
        ticket = MaterialService(db, actor, STORE).create_upload(body(data), key=key or uuid4())
    blob = STORE.put(actor.security_domain, io.BytesIO(data), len(data))
    with factory.begin() as db:
        return MaterialService(db, actor, STORE).put_upload_content(UUID(ticket["upload_id"]), blob)


def test_two_step_actual_bytes_idempotency_and_initial_capability(factory, actors):
    a, _ = actors
    data, key = b"S04 synthetic source body", uuid4()
    with factory.begin() as db:
        service = MaterialService(db, a, STORE)
        ticket = service.create_upload(body(data), key=key)
        assert service.create_upload(body(data), key=key) == ticket
        assert service.upload(UUID(ticket["upload_id"])).max_bytes == len(data)
    blob = STORE.put(a.security_domain, io.BytesIO(data), len(data))
    with factory.begin() as db:
        service = MaterialService(db, a, STORE)
        first = service.put_upload_content(UUID(ticket["upload_id"]), blob)
        assert service.put_upload_content(UUID(ticket["upload_id"]), blob) == first
        assert first["status"] == "queued" and first["can_qa"] is False
        status = service.get_ingestion(UUID(first["file_id"]))
        assert status["state"] == "queued" and status["events"][0]["seq"] == 1
        assert status["identity_status"] == "unverified" and status["formal_evidence_ready"] is False


def test_domain_dedup_is_scoped_and_upload_id_not_transferable(factory, actors):
    a, b = actors
    data = b"same synthetic bytes across different owners"
    first, second = upload(factory, a, data), upload(factory, b, data)
    assert first["sha256"] == second["sha256"] and first["security_domain"] != second["security_domain"]
    with factory.begin() as db:
        for action in (lambda: MaterialService(db, b, STORE).get_file(UUID(first["file_id"])),
                       lambda: MaterialService(db, b, STORE).get_ingestion(UUID(first["file_id"]))):
            with pytest.raises(DomainError, match="not_found"):
                action()
        assert db.scalar(select(func.count()).select_from(core.Blob).where(core.Blob.sha256 == first["sha256"],
                         core.Blob.security_domain.in_([a.security_domain, b.security_domain]))) == 2


def test_wrong_hash_or_size_never_registers_a_file(factory, actors):
    a, _ = actors
    with factory.begin() as db:
        ticket = MaterialService(db, a, STORE).create_upload(body(b"expected"), key=uuid4())
    wrong = STORE.put(a.security_domain, io.BytesIO(b"different"), 100)
    with factory.begin() as db:
        with pytest.raises(DomainError, match="upload_hash_or_size_mismatch"):
            MaterialService(db, a, STORE).put_upload_content(UUID(ticket["upload_id"]), wrong)
        assert db.get(m.MaterialUpload, UUID(ticket["upload_id"])).file_id is None


def test_text_500_character_title_is_preserved_and_unsafe_text_rejected(factory, actors):
    a, _ = actors
    title = "研" * 500
    request = {"title": title, "text": "S04 synthetic α😀\r\nnext line", "format": "paste"}
    with factory.begin() as db:
        service = MaterialService(db, a, STORE)
        result = service.ingest_text(request, key=uuid4())
        row = db.scalar(select(m.MaterialUpload).where(m.MaterialUpload.file_id == UUID(result["file_id"])))
        assert row.display_title == title and len(row.file_name) <= 255
        with pytest.raises(DomainError, match="nul_character"):
            service.ingest_text({**request, "text": "bad\x00value"}, key=uuid4())


def test_project_boundary_and_filename_size_inputs_fail_before_ticket(factory, actors):
    a, b = actors
    project = uuid4()
    with factory.begin() as db:
        db.add(core.Project(id=project, owner_id=b.owner_id, title="S04 synthetic private project", revision=1))
    with factory.begin() as db:
        service = MaterialService(db, a, STORE)
        for request, code in (({**body(b"x"), "project_id": str(project)}, "not_found"),
                              ({**body(b"x"), "file_name": "../escape.txt"}, "invalid_file_name"),
                              ({**body(b"x"), "size_bytes": 2097153}, "file_too_large"),
                              ({**body(b"x"), "owner_id": str(b.owner_id)}, "invalid_input")):
            with pytest.raises(DomainError, match=code):
                service.create_upload(request, key=uuid4())


def test_cancel_is_terminal_and_history_is_immutable(factory, actors):
    a, _ = actors
    receipt = upload(factory, a, b"S04 synthetic cancel source")
    file_id = UUID(receipt["file_id"])
    with factory.begin() as db:
        service = MaterialService(db, a, STORE)
        cancelled = service.cancel(file_id)
        assert cancelled["status"] == "cancelled" and cancelled["can_qa"] is False
        assert service.cancel(file_id) == cancelled
        status = service.get_ingestion(file_id)
        assert [event["state"] for event in status["events"]] == ["queued", "cancelled"]
        with pytest.raises(DomainError, match="file_not_available"):
            service.read_content(file_id)


def test_user_version_binding_is_typed_unverified_and_current_acl_wins(factory, actors):
    a, _ = actors
    data = b"S04 synthetic paper fixture"
    receipt = upload(factory, a, data)
    with factory.begin() as db:
        catalog = Catalog(db, a)
        work = catalog.create_work(WorkInput(title="S04 synthetic metadata", authors=None, year=None, client_operation_id=uuid4()))
        catalog.record_source(UUID(work["work_id"]), SourceInput(source="s04_synthetic_metadata", native_id=str(uuid4()), source_revision=None,
                              retrieved_at=datetime.now(UTC), raw_sha256=hashlib.sha256(data).hexdigest(), source_url=None, rights_status="unknown", license=None), data, STORE)
        version = catalog.record_version(UUID(work["work_id"]), hashlib.sha256(data).hexdigest(), "source_unknown", None)
        version_id = UUID(version["version_id"])
        service = MaterialService(db, a, STORE)
        file_id = UUID(receipt["file_id"])
        assert service.bind_version(file_id, version_id)["can_qa"] is False
        status = service.get_ingestion(file_id)
        assert status["binding_source"] == "user_explicit" and status["identity_status"] == "unverified" and status["formal_evidence_ready"] is False
        with pytest.raises(DomainError, match="publisher_required"):
            service.bind_verified_source(file_id, version_id)
        db.get(core.WorkAccess, (a.owner_id, UUID(work["work_id"]))).revoked = True
    with factory.begin() as db, pytest.raises(DomainError, match="not_found"):
        MaterialService(db, a, STORE).get_file(UUID(receipt["file_id"]))


def test_trusted_source_identity_requires_exact_existing_byte_chain(factory):
    publisher = Actor(uuid4(), publisher=True)
    with factory.begin() as db:
        db.add(core.Owner(id=publisher.owner_id, name="S04 synthetic trusted publisher fixture", can_publish=True))
    data = b"S04 synthetic trusted byte chain, not a research paper"
    receipt = upload(factory, publisher, data)
    different = upload(factory, publisher, b"different actual bytes")
    with factory.begin() as db:
        catalog = Catalog(db, publisher)
        work = catalog.create_work(WorkInput(title="S04 synthetic trusted-source fixture", authors=None, year=None, client_operation_id=uuid4()), public=True)
        catalog.record_source(UUID(work["work_id"]), SourceInput(source="licensed_corpus_file", native_id="S04-SYNTHETIC-" + str(uuid4()), source_revision=None,
                              retrieved_at=datetime.now(UTC), raw_sha256=hashlib.sha256(data).hexdigest(), source_url="https://example.org/s04-synthetic-fixture.txt",
                              rights_status="verified", license="CC0-1.0"), data, STORE, public=True)
        version = catalog.record_version(UUID(work["work_id"]), hashlib.sha256(data).hexdigest(), "published", None, public=True)
        service = MaterialService(db, publisher, STORE)
        file_id = UUID(receipt["file_id"])
        result = service.bind_verified_source(file_id, UUID(version["version_id"]))
        status = service.get_ingestion(file_id)
        assert status["identity_status"] == "hash_verified" and status["binding_source"] == "trusted_source"
        assert result["can_qa"] is False and status["formal_evidence_ready"] is False  # Queued; still no parsed evidence.
        with pytest.raises(DomainError, match="source_identity_proof_unavailable"):
            service.bind_verified_source(UUID(different["file_id"]), UUID(version["version_id"]))
