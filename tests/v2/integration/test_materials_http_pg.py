"""Real S04 HTTP/domain integration on the owned PG and canonical CAS.

All inserted materials are explicitly synthetic fixtures; no parser/model call.
"""

from __future__ import annotations

import hashlib
import socket
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from citequest.materials_v2 import models as m
from citequest.materials_v2.api import create_app
from citequest.v2 import models as core
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor, SourceInput, WorkInput
from citequest.v2.security import issue_session
from citequest.v2.services import Catalog
from citequest.v2.storage import CAS, StoredBlob

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def factory():
    sys.path.insert(0, str(ROOT / "scripts/s03"))
    from bootstrap import dsn

    db = engine(dsn("cq_s03_app"))
    with db.connect() as conn:
        assert conn.exec_driver_sql("SELECT current_database(),current_user").one() == ("citequest_s03", "cq_s03_app")
        current = conn.exec_driver_sql("SELECT version_num FROM cq_app.alembic_version").scalar_one()
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        history = ScriptDirectory.from_config(Config(str(ROOT / "alembic-app.ini")))
        assert "s04_0001" in {revision.revision for revision in history.iterate_revisions(current, "base")}
    yield sessions(db)
    db.dispose()


@pytest.fixture
def material_http(factory):
    actor, other = Actor(uuid4()), Actor(uuid4())
    with factory.begin() as db:
        db.add_all([core.Owner(id=a.owner_id, name="S04 synthetic HTTP fixture") for a in [actor, other]])
        db.flush()
        token, csrf = issue_session(db, actor)
        other_token, other_csrf = issue_session(db, other)
    store = CAS(ROOT / "data/runtime-v2-app/blobs")
    app = create_app(
        factory,
        store=store,
        cursor_key=b"s04-real-http-test-only-signing-key",
        contract_path=ROOT / "contracts/v2/domain.schema.json",
        policy_path=ROOT / "contracts/v2/policy.json",
    )
    with TestClient(app, cookies={"citequest_session": token}) as client:
        yield {
            "client": client,
            "actor": actor,
            "other": other,
            "token": token,
            "csrf": csrf,
            "other_token": other_token,
            "other_csrf": other_csrf,
            "store": store,
            "app": app,
            "factory": factory,
        }


def headers(context, key=None, *, binary=False, other=False):
    value = {"Origin": "http://127.0.0.1:18000", "X-CSRF-Token": context["other_csrf"] if other else context["csrf"]}
    if key is not None:
        value["Idempotency-Key"] = str(key)
    if binary:
        value["Content-Type"] = "application/octet-stream"
    return value


def ticket(context, data=None, key=None):
    data = data or "S04 synthetic HTTP bytes; 非论文、非模型结果。\r\nsecond line".encode()
    body = {
        "file_name": "synthetic-http.txt",
        "media_type": "text/plain",
        "size_bytes": len(data),
        "expected_sha256": hashlib.sha256(data).hexdigest(),
        "project_id": None,
    }
    operation = key or uuid4()
    response = context["client"].post("/api/v2/uploads", json=body, headers=headers(context, operation))
    assert response.status_code == 202, response.text
    return data, body, operation, response.json()


def upload(context):
    data, body, key, admission = ticket(context)
    response = context["client"].put(admission["upload_url"], content=data, headers=headers(context, binary=True))
    assert response.status_code == 202, response.text
    return data, body, key, admission, response.json()


def test_real_pg_ticket_raw_upload_exact_replay_and_current_owner_acl(material_http):
    c = material_http
    data, body, key, admission, file = upload(c)
    assert file["status"] == "queued" and file["version_id"] is None and file["can_qa"] is False
    assert c["client"].post("/api/v2/uploads", json=body, headers=headers(c, key)).json() == admission
    assert c["client"].put(admission["upload_url"], content=data, headers=headers(c, binary=True)).json() == file
    changed = c["client"].post("/api/v2/uploads", json={**body, "file_name": "changed.txt"}, headers=headers(c, key))
    assert changed.status_code == 409
    c["store"].verify(StoredBlob(file["security_domain"], file["sha256"], file["size_bytes"]))
    with c["factory"].begin() as db:
        assert (
            db.scalar(
                select(func.count())
                .select_from(m.MaterialIngestion)
                .where(m.MaterialIngestion.file_id == UUID(file["file_id"]))
            )
            == 1
        )
        assert (
            db.scalar(
                select(func.count())
                .select_from(m.MaterialVersion)
                .where(m.MaterialVersion.file_id == UUID(file["file_id"]))
            )
            == 1
        )
    c["client"].cookies.set("citequest_session", c["other_token"])
    for path in [
        f"/api/v2/files/{file['file_id']}",
        f"/api/v2/files/{file['file_id']}/ingestion",
        f"/api/v2/files/{file['file_id']}/representation",
        f"/api/v2/files/{file['file_id']}/fragments",
        f"/api/v2/files/{file['file_id']}/content",
    ]:
        assert c["client"].get(path).status_code == 404
    assert (
        c["client"].put(admission["upload_url"], content=data, headers=headers(c, binary=True, other=True)).status_code
        == 404
    )


def test_real_pg_paste_immutable_receipt_cancel_and_queued_content_denial(material_http):
    c = material_http
    key = uuid4()
    body = {"title": "S04 synthetic pasted material", "text": "仅用于接口测试 α+β=γ\n第二行", "format": "paste"}
    first = c["client"].post("/api/v2/materials/text", json=body, headers=headers(c, key))
    assert first.status_code == 202, first.text
    file = first.json()
    assert c["client"].post("/api/v2/materials/text", json=body, headers=headers(c, key)).json() == file
    url = f"/api/v2/files/{file['file_id']}"
    assert c["client"].get(url + "/content").status_code == 409
    assert c["client"].get(url + "/representation").status_code == 409
    response = c["client"].post(url + "/cancel", headers=headers(c))
    assert response.status_code == 200 and response.json()["status"] == "cancelled"
    ingestion = c["client"].get(url + "/ingestion")
    assert ingestion.status_code == 200, ingestion.text
    value = ingestion.json()
    assert (
        value["state"] == "cancelled" and value["binding_source"] == "none" and value["formal_evidence_ready"] is False
    )
    assert [(row["seq"], row["state"]) for row in value["events"]] == [(1, "queued"), (2, "cancelled")]
    assert c["client"].get(url + "/content").status_code == 409


def test_real_pg_ticket_expiry_and_bad_content_do_not_create_file(material_http):
    c = material_http
    data, _body, _key, admission = ticket(c)
    with c["factory"].begin() as db:
        row = db.get(m.MaterialUpload, UUID(admission["upload_id"]))
        row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert c["client"].put(admission["upload_url"], content=data, headers=headers(c, binary=True)).status_code == 404
    data, _body, _key, admission = ticket(c)
    response = c["client"].put(admission["upload_url"], content=b"x" * len(data), headers=headers(c, binary=True))
    assert response.status_code == 422
    with c["factory"].begin() as db:
        assert db.get(m.MaterialUpload, UUID(admission["upload_id"])).file_id is None
    assert c["client"].put(admission["upload_url"], content=data, headers=headers(c, binary=True)).status_code == 202


def test_real_pg_user_version_association_stays_unverified(material_http):
    c = material_http
    _data, _body, _key, _admission, file = upload(c)
    raw = b"S04 synthetic metadata bytes; deliberately not uploaded material bytes."
    with c["factory"].begin() as db:
        service = Catalog(db, c["actor"])
        work = service.create_work(
            WorkInput(title="S04 synthetic unknown paper", authors=None, year=None, client_operation_id=uuid4())
        )
        source = SourceInput(
            source="synthetic_http_metadata",
            native_id=str(uuid4()),
            source_revision=None,
            retrieved_at=datetime.now(UTC),
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            source_url=None,
            rights_status="unknown",
            license=None,
        )
        service.record_source(UUID(work["work_id"]), source, raw, c["store"])
        version = service.record_version(UUID(work["work_id"]), source.raw_sha256, "source_unknown", None)
    url = f"/api/v2/files/{file['file_id']}"
    response = c["client"].post(url + "/version", json={"version_id": version["version_id"]}, headers=headers(c))
    assert response.status_code == 200, response.text
    assert response.json()["version_id"] == version["version_id"] and response.json()["can_qa"] is False
    grade = c["client"].get(url + "/ingestion").json()
    assert (
        grade["binding_source"] == "user_explicit"
        and grade["identity_status"] == "unverified"
        and grade["formal_evidence_ready"] is False
    )
    assert (
        c["client"]
        .post(url + "/version", json={"version_id": version["version_id"], "proof": "trusted"}, headers=headers(c))
        .status_code
        == 400
    )


def test_real_pg_local_session_csrf_origin_and_body_identity_are_not_bypassed(material_http):
    c = material_http
    body = {"title": "S04 synthetic", "text": "fixture", "format": "txt"}
    assert (
        c["client"].post("/api/v2/materials/text", json=body, headers={"Idempotency-Key": str(uuid4())}).status_code
        == 403
    )
    assert (
        c["client"]
        .post("/api/v2/materials/text", json=body, headers={**headers(c, uuid4()), "Origin": "https://foreign.invalid"})
        .status_code
        == 403
    )
    assert (
        c["client"]
        .post(
            "/api/v2/materials/text", json={**body, "owner_id": str(c["other"].owner_id)}, headers=headers(c, uuid4())
        )
        .status_code
        == 400
    )
    c["client"].cookies.clear()
    assert c["client"].post("/api/v2/materials/text", json=body, headers=headers(c, uuid4())).status_code == 401


def test_real_tcp_loopback_http_uses_same_service_and_committed_queue(material_http):
    c = material_http
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 18000))  # Never displace another process.
    server = uvicorn.Server(uvicorn.Config(c["app"], host="127.0.0.1", port=18000, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("Owned S04 loopback server failed to start")
            time.sleep(0.01)
        with httpx.Client(
            base_url="http://127.0.0.1:18000", trust_env=False, cookies={"citequest_session": c["token"]}, timeout=10
        ) as client:
            response = client.post(
                "/api/v2/materials/text",
                json={"title": "S04 synthetic TCP", "text": "TCP fixture αβ", "format": "txt"},
                headers=headers(c, uuid4()),
            )
            assert response.status_code == 202 and response.headers["x-request-id"], response.text
            file_id = response.json()["file_id"]
            response = client.get(f"/api/v2/files/{file_id}/ingestion")
            assert response.status_code == 200 and response.json()["state"] == "queued", response.text
            assert response.json()["formal_evidence_ready"] is False
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive(), "Owned loopback HTTP server did not stop"


def test_real_parser_to_http_original_bytes_fragments_pagination_and_late_acl_revocation(material_http, monkeypatch):
    from citequest.materials_v2.processor import MaterialProcessor

    c = material_http
    original = "\r\n".join(
        f"line {i:02d} alpha research 示例😀 equation x²=𝒙" + (" 100%_literal" if i == 0 else "") for i in range(55)
    )
    data, _body, _key, admission = ticket(c, original.encode("utf-8"))
    response = c["client"].put(admission["upload_url"], content=data, headers=headers(c, binary=True))
    assert response.status_code == 202, response.text
    file_id = UUID(response.json()["file_id"])
    processor = MaterialProcessor(
        c["factory"],
        c["store"],
        parser_python=ROOT / "data/runtime-v2/.venv/Scripts/python.exe",
        root=ROOT,
        artifact_root=ROOT / "data/runtime-s04/ingestion-artifacts",
    )
    processed = processor.run_next(file_id=file_id)
    assert processed["status"] == "processed" and processed["ingestion_state"] == "ready", processed
    url = f"/api/v2/files/{file_id}"
    file = c["client"].get(url).json()
    assert file["status"] == "ready" and file["can_download"] and file["can_preview"] and file["can_qa"] is False
    representation = c["client"].get(url + "/representation")
    assert representation.status_code == 200, representation.text
    assert representation.json()["file_sha256"] == hashlib.sha256(data).hexdigest()
    state = c["client"].get(url + "/ingestion")
    assert state.status_code == 200, state.text
    assert state.json()["material_source_ready"] is True and state.json()["paper_source_ready"] is False
    assert state.json()["formal_evidence_ready"] is False and state.json()["lease_active"] is False
    assert state.json()["execution_outcome"] == "within_deadline" and state.json()["execution_success"] is True
    assert 0 < state.json()["active_elapsed_seconds"] <= state.json()["execution_budget_seconds"] == 120.0
    assert [row["state"] for row in state.json()["events"]] == ["queued", "validating", "parsing", "ready"]
    fragments, after, seen = [], 0, set()
    while True:
        page = c["client"].get(url + "/fragments", params={"after": after, "page_size": 7})
        assert page.status_code == 200, page.text
        value = page.json()
        assert value["representation_id"] == representation.json()["representation_id"]
        for item in value["items"]:
            assert item["fragment_id"] not in seen
            seen.add(item["fragment_id"])
            fragments.append(item)
        if value["next_after"] is None:
            break
        assert value["next_after"] > after
        after = value["next_after"]
    assert len(fragments) == 55 and [r["ordinal"] for r in fragments] == list(range(1, 56))
    cover = bytearray(len(original))
    for item in fragments:
        locator = item["locator"]
        assert locator["file_sha256"] == file["sha256"] and item["work_version_id"] is None
        assert original[locator["char_start"] : locator["char_end"]] == item["source_text"]
        assert hashlib.sha256(item["source_text"].encode()).hexdigest() == item["source_text_sha256"]
        assert 1 <= item["token_count"] <= 2048
        cover[locator["char_start"] : locator["char_end"]] = b"1" * (locator["char_end"] - locator["char_start"])
    assert all(cover), "Original CRLF/Unicode source characters were omitted"
    assert c["client"].get(url + "/fragments", params={"query": "示例"}).json()["items"]
    literal = c["client"].get(url + "/fragments", params={"query": "%_"})
    assert literal.status_code == 200 and len(literal.json()["items"]) == 1
    raw = c["client"].get(url + "/content")
    assert raw.status_code == 200 and raw.content == data and raw.headers["content-type"] == "application/octet-stream"
    etag = raw.headers["etag"]
    ranged = c["client"].get(url + "/content", headers={"Range": "bytes=3-19", "If-Range": etag})
    assert ranged.status_code == 206 and ranged.content == data[3:20]
    assert c["client"].get(url + "/content", headers={"If-None-Match": etag}).status_code == 304
    assert c["client"].get(url + "/content", headers={"Range": "bytes=999999-"}).status_code == 416
    c["client"].cookies.set("citequest_session", c["other_token"])
    assert c["client"].get(url + "/content", headers={"If-None-Match": etag}).status_code == 404
    assert c["client"].get(url + "/fragments").status_code == 404
    c["client"].cookies.set("citequest_session", c["token"])
    actual_read = c["store"].read

    def delete_after_verified_read(blob):
        received = actual_read(blob)
        with c["factory"].begin() as db:
            Catalog(db, c["actor"]).delete_file(file_id)
        return received

    monkeypatch.setattr(c["store"], "read", delete_after_verified_read)
    denied = c["client"].get(url + "/content")
    assert denied.status_code == 404 and denied.content != data
