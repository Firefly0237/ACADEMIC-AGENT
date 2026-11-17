"""Adapter-only HTTP controls: real CAS, explicitly fake SQL/service/session.

These checks cannot establish database ACL, rollback or parser correctness.
"""

from __future__ import annotations

import hashlib
import io
import json
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from citequest.materials_v2 import api
from citequest.v2.domain import Actor, DomainError
from citequest.v2.storage import CAS

ROOT = Path(__file__).resolve().parents[3]
IN_TRANSACTION = ContextVar("material_http_test_transaction", default=False)


@pytest.fixture
def http(tmp_path, monkeypatch):
    actor, other = Actor(uuid4()), Actor(uuid4())
    file_id, upload_id = uuid4(), uuid4()
    data = b"original verified material bytes\r\nsecond line"

    class Factory:
        commits = 0
        rollbacks = 0
        fail_commit = False

        @contextmanager
        def begin(self):
            token = IN_TRANSACTION.set(True)
            try:
                yield object()
                if self.fail_commit:
                    raise OperationalError("COMMIT", {}, RuntimeError("synthetic commit failure"))
                self.commits += 1
            except BaseException:
                self.rollbacks += 1
                raise
            finally:
                IN_TRANSACTION.reset(token)

    class Store(CAS):
        def __init__(self, root):
            super().__init__(root)
            self.calls = []
            self.after_read = None
            self.before_put = None

        def put(self, *args, **kwargs):
            assert IN_TRANSACTION.get() is False, "raw CAS upload held a SQL transaction"
            self.calls.append("put")
            if self.before_put:
                self.before_put()
            return super().put(*args, **kwargs)

        def read(self, blob):
            assert IN_TRANSACTION.get() is False, "CAS delivery held a SQL transaction"
            self.calls.append("read")
            value = super().read(blob)
            if self.after_read:
                self.after_read()
            return value

    store = Store(tmp_path / "cas")
    blob = CAS.put(store, actor.security_domain, io.BytesIO(data), len(data))
    dto = {
        "file_id": str(file_id),
        "version_id": None,
        "owner_id": str(actor.owner_id),
        "security_domain": actor.security_domain,
        "sha256": blob.sha256,
        "media_type": "text/plain",
        "size_bytes": len(data),
        "page_count": None,
        "rights_status": "unknown",
        "outbound_policy": "local_only",
        "status": "ready",
        "can_preview": False,
        "can_qa": False,
        "can_download": True,
        "coverage": "text_full",
    }
    state = SimpleNamespace(
        dto=dto,
        revoked=False,
        stored=None,
        calls=[],
        actor=actor,
        other=other,
        upload_id=upload_id,
        file_id=file_id,
        data=data,
        blob=blob,
        store=store,
    )
    admission = SimpleNamespace(
        upload_id=upload_id,
        owner_id=actor.owner_id,
        file_name="paper.txt",
        media_type="text/plain",
        size_bytes=len(data),
        expected_sha256=blob.sha256,
        project_id=None,
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
        max_bytes=len(data),
        file_id=None,
    )
    state.admission = admission

    class Service:
        def __init__(self, db, current, cas):
            assert IN_TRANSACTION.get()
            self.actor = current

        def allowed(self):
            if self.actor != actor or state.revoked:
                raise DomainError("not_found", 404)

        def upload(self, identity):
            self.allowed()
            assert identity == upload_id
            if admission.expires_at <= datetime.now(UTC):
                raise DomainError("not_found", 404)
            return admission

        def create_upload(self, body, *, key):
            self.allowed()
            state.calls.append(("ticket", body, key))
            result = {
                "upload_id": str(upload_id),
                "state": "queued",
                "expires_at": admission.expires_at.isoformat(),
                "max_bytes": len(data),
                "upload_url": f"/api/v2/uploads/{upload_id}/content",
            }
            if getattr(state, "bad_ticket", False):
                result["state"] = "ready"
            if getattr(state, "fail_commit", False):
                state.factory.fail_commit = True
            return result

        def put_upload_content(self, identity, received):
            self.upload(identity)
            assert identity == upload_id and received == blob
            state.stored = received
            state.calls.append(("finish", received))
            return {**dto, "status": "queued", "coverage": "metadata_only", "can_download": False}

        def ingest_text(self, body, *, key):
            self.allowed()
            state.calls.append(("text", body, key))
            return {**dto, "status": "queued", "coverage": "metadata_only", "can_download": False}

        def get_file(self, identity):
            self.allowed()
            assert identity == file_id
            return deepcopy(dto)

        def read_content(self, identity):
            self.allowed()
            assert identity == file_id
            return blob, dto["media_type"], admission.file_name

        def get_fragments(self, identity, after=0, limit=50, query=None):
            self.allowed()
            text = data.decode("utf-8")
            locator = {
                "kind": "text",
                "file_sha256": blob.sha256,
                "char_start": 0,
                "char_end": len(text),
                "line_1based": 1,
            }
            if getattr(state, "bad_locator", None) is not None:
                locator = state.bad_locator
            return {
                "schema_version": "cq.material.fragments/1",
                "file_id": str(identity),
                "representation_id": "a" * 64,
                "items": [
                    {
                        "fragment_id": "b" * 64,
                        "ordinal": 1,
                        "material_version_id": str(uuid4()),
                        "work_version_id": None,
                        "source_text": text,
                        "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "index_text": text,
                        "locator": locator,
                        "token_count": 10,
                        "tokenizer_version": "synthetic_http_fixture/1",
                        "tokenizer_profile_sha256": "c" * 64,
                        "chunker_version": "synthetic_http_fixture/1",
                        "source_block_id": "line:1",
                        "granularity": "paragraph",
                        "bbox_granularity": None,
                        "target_exceeded_due_to_source_boundary": False,
                    }
                ],
                "next_after": None,
                "coverage": "text_full",
                "missing_ranges": [],
            }

        def cancel(self, identity):
            self.allowed()
            return {**dto, "status": "cancelled", "coverage": "metadata_only", "can_download": False}

        def bind_version(self, identity, version):
            self.allowed()
            state.calls.append(("bind", version))
            return {**dto, "version_id": str(version)}

    def authenticate(db, token, *, csrf=None, mutation=False):
        if token not in {str(actor.owner_id), str(other.owner_id)}:
            raise DomainError("session_required", 401)
        if mutation and csrf != "test-csrf":
            raise DomainError("csrf_required", 403)
        return actor if token == str(actor.owner_id) else other

    monkeypatch.setattr(api, "_material_service", Service)
    monkeypatch.setattr(api, "authenticate", authenticate)
    state.factory = Factory()
    app = api.create_app(
        state.factory,
        store=store,
        cursor_key=b"material-http-contract-test-only-key",
        contract_path=ROOT / "contracts/v2/domain.schema.json",
        policy_path=ROOT / "contracts/v2/policy.json",
    )
    state.app = app
    with TestClient(app, cookies={"citequest_session": str(actor.owner_id)}) as client:
        yield client, state


def mutation_headers(**kwargs):
    return {"Origin": "http://127.0.0.1:18000", "X-CSRF-Token": "test-csrf", **kwargs}


def upload_body(state):
    return {
        "file_name": "paper.txt",
        "media_type": "text/plain",
        "size_bytes": len(state.data),
        "expected_sha256": state.blob.sha256,
        "project_id": None,
    }


def test_ticket_preserves_frozen_body_and_explicit_idempotency_header(http):
    client, state = http
    key = uuid4()
    response = client.post(
        "/api/v2/uploads", json=upload_body(state), headers=mutation_headers(**{"Idempotency-Key": str(key)})
    )
    assert response.status_code == 202 and response.json()["state"] == "queued"
    assert state.calls[-1] == ("ticket", upload_body(state), key)
    for forbidden in ["owner_id", "path", "version_id", "identity_status"]:
        response = client.post(
            "/api/v2/uploads",
            json={**upload_body(state), forbidden: str(uuid4())},
            headers=mutation_headers(**{"Idempotency-Key": str(uuid4())}),
        )
        assert response.status_code == 400 and response.json()["code"] == "invalid_input"


@pytest.mark.parametrize("bad", [None, "not-a-uuid"])
def test_idempotency_is_required_not_invented_from_body(http, bad):
    client, state = http
    headers = mutation_headers(**({"Idempotency-Key": bad} if bad is not None else {}))
    assert client.post("/api/v2/uploads", json=upload_body(state), headers=headers).status_code == 400
    assert not state.calls


def test_raw_upload_spools_and_writes_cas_outside_sql_then_rechecks(http):
    client, state = http
    response = client.put(
        f"/api/v2/uploads/{state.upload_id}/content",
        content=state.data,
        headers=mutation_headers(**{"Content-Type": "application/octet-stream"}),
    )
    assert response.status_code == 202 and response.json()["can_qa"] is False
    assert state.stored == state.blob and state.store.calls == ["put"]
    assert not list(state.store.root.glob(".staging-*"))


@pytest.mark.parametrize(
    "kind, status", [("overflow", 413), ("short", 422), ("wrong_hash", 422), ("wrong_media", 415), ("compressed", 415)]
)
def test_raw_upload_actual_bounds_do_not_register_bad_bytes(http, kind, status):
    client, state = http
    headers = mutation_headers(**{"Content-Type": "application/octet-stream"})
    content = state.data
    if kind == "overflow":
        content += b"x"
    elif kind == "short":
        content = content[:-1]
    elif kind == "wrong_hash":
        content = b"x" * len(content)
    elif kind == "wrong_media":
        headers["Content-Type"] = "application/pdf"
    else:
        headers["Content-Encoding"] = "gzip"
    response = client.put(f"/api/v2/uploads/{state.upload_id}/content", content=content, headers=headers)
    assert response.status_code == status and state.stored is None and "put" not in state.store.calls


def test_unauthenticated_cross_owner_origin_csrf_are_rejected_before_cas(http):
    client, state = http
    url = f"/api/v2/uploads/{state.upload_id}/content"
    client.cookies.clear()
    assert (
        client.put(
            url, content=state.data, headers=mutation_headers(**{"Content-Type": "application/octet-stream"})
        ).status_code
        == 401
    )
    client.cookies.set("citequest_session", str(state.other.owner_id))
    assert (
        client.put(
            url, content=state.data, headers=mutation_headers(**{"Content-Type": "application/octet-stream"})
        ).status_code
        == 404
    )
    client.cookies.set("citequest_session", str(state.actor.owner_id))
    assert client.put(url, content=state.data, headers={"Content-Type": "application/octet-stream"}).status_code == 403
    assert (
        client.put(
            url,
            content=state.data,
            headers={"Content-Type": "application/octet-stream", "Origin": "http://127.0.0.1:18000"},
        ).status_code
        == 403
    )
    assert not state.store.calls


def test_paste_contract_does_not_accept_owner_and_rejects_duplicate_json_keys(http):
    client, _state = http
    headers = mutation_headers(**{"Idempotency-Key": str(uuid4()), "Content-Type": "application/json"})
    response = client.post(
        "/api/v2/materials/text", json={"title": "草稿", "text": "原始材料", "format": "paste"}, headers=headers
    )
    assert response.status_code == 202
    response = client.post(
        "/api/v2/materials/text",
        content='{"title":"first","title":"second","text":"ok","format":"paste"}',
        headers=headers,
    )
    assert response.status_code == 400 and response.json()["code"] == "invalid_json"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("bytes=0-3", (0, 3)),
        ("BYTES=3-", (3, None)),
        ("bytes=-4", (-4, None)),
        ("bytes=0-999", (0, None)),
        ("items=0-3", None),
    ],
)
def test_single_range_is_exact_original_bytes(http, value, expected):
    client, state = http
    response = client.get(f"/api/v2/files/{state.file_id}/content", headers={"Range": value})
    if expected is None:
        assert response.status_code == 200 and response.content == state.data
    else:
        left, right = expected
        assert response.status_code == 206
        assert response.content == state.data[left : right + 1 if right is not None else None]
        assert response.headers["content-range"].endswith("/" + str(len(state.data)))
    assert response.headers["etag"] == f'"{state.blob.sha256}"'
    assert response.headers["accept-ranges"] == "bytes" and response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-type"] == "application/octet-stream"


@pytest.mark.parametrize(
    "value", ["bytes=999-", "bytes=5-2", "bytes=-0", "bytes=", "bytes=1-2,4-5", "bytes=١-2", "bytes=" + "1" * 201]
)
def test_bad_or_multiple_ranges_return_structured_416(http, value):
    client, state = http
    response = client.get(f"/api/v2/files/{state.file_id}/content", headers=[(b"Range", value.encode("utf-8"))])
    assert response.status_code == 416 and response.json()["request_id"]
    assert response.headers["content-range"] == f"bytes */{len(state.data)}"


def test_cache_and_if_range_preserve_acl_and_head_semantics(http):
    client, state = http
    url = f"/api/v2/files/{state.file_id}/content"
    etag = f'"{state.blob.sha256}"'
    assert client.get(url, headers={"Range": "bytes=0-2", "If-Range": etag}).content == state.data[:3]
    mismatch = client.get(url, headers={"Range": "bytes=0-2", "If-Range": '"stale"'})
    assert mismatch.status_code == 200 and mismatch.content == state.data
    cached = client.get(url, headers={"If-None-Match": "W/" + etag, "Range": "bytes=0-2"})
    assert cached.status_code == 304 and cached.content == b""
    assert client.get(url, headers={"If-Match": '"stale"'}).status_code == 412
    head = client.head(url, headers={"Range": "bytes=0-2"})
    assert head.status_code == 200 and head.content == b"" and head.headers["content-length"] == str(len(state.data))
    state.store.after_read = lambda: setattr(state, "revoked", True)
    assert client.get(url, headers={"If-None-Match": etag}).status_code == 404


def test_content_verifies_delivered_bytes_and_safe_disposition(http):
    client, state = http
    state.admission.file_name = "../../evil\r\nX-Injected:1.html"
    response = client.get(f"/api/v2/files/{state.file_id}/content")
    assert response.status_code == 200 and response.content == state.data
    header = response.headers["content-disposition"]
    assert header.startswith("attachment;") and "\r" not in header and "\n" not in header and "../" not in header
    assert header.endswith(".html.txt") and "x-injected" not in response.headers
    state.store.path(state.blob.security_domain, state.blob.sha256).write_bytes(b"x" * len(state.data))
    assert client.get(f"/api/v2/files/{state.file_id}/content").status_code == 422


def test_inline_is_only_for_authorized_pdf_preview_and_file_get_has_one_route(http):
    client, state = http
    url = f"/api/v2/files/{state.file_id}/content?disposition=inline"
    assert client.get(url).status_code == 409
    state.dto["media_type"] = "application/pdf"
    state.dto["can_preview"] = True
    inline = client.get(url)
    assert (
        inline.headers["content-disposition"].startswith("inline;")
        and inline.headers["content-type"] == "application/pdf"
    )

    def leaves(routes):
        for route in routes:
            included = getattr(route, "original_router", None)
            if included is not None:
                yield from leaves(included.routes)
            else:
                yield route

    routes = [
        route
        for route in leaves(state.app.routes)
        if getattr(route, "path", None) == "/api/v2/files/{file_id}" and "GET" in getattr(route, "methods", set())
    ]
    assert len(routes) == 1 and client.get(f"/api/v2/files/{state.file_id}").json()["can_download"] is True


def test_same_owner_concurrent_upload_is_rejected_without_cas_and_released(http):
    client, state = http
    limiter = state.app.state.material_upload_limiter
    client.portal.call(limiter.acquire_on_behalf_of_nowait, state.actor.owner_id)
    try:
        response = client.put(
            f"/api/v2/uploads/{state.upload_id}/content",
            content=state.data,
            headers=mutation_headers(**{"Content-Type": "application/octet-stream"}),
        )
        assert response.status_code == 429 and response.headers["retry-after"] == "1" and not state.store.calls
    finally:
        client.portal.call(limiter.release_on_behalf_of, state.actor.owner_id)
    assert (
        client.put(
            f"/api/v2/uploads/{state.upload_id}/content",
            content=state.data,
            headers=mutation_headers(**{"Content-Type": "application/octet-stream"}),
        ).status_code
        == 202
    )


def test_global_request_overload_is_structured_and_limiter_recovers(http):
    client, state = http
    limiter = state.app.state.material_request_limiter
    borrowers = [object() for _ in range(8)]
    for borrower in borrowers:
        client.portal.call(limiter.acquire_on_behalf_of_nowait, borrower)
    try:
        response = client.get(f"/api/v2/files/{state.file_id}/content")
        assert response.status_code == 503 and response.json()["code"] == "http_busy"
        assert (
            response.json()["request_id"] == response.headers["x-request-id"] and response.headers["retry-after"] == "1"
        )
        assert not state.store.calls
        assert client.get("/healthz").status_code == 200
    finally:
        for borrower in borrowers:
            client.portal.call(limiter.release_on_behalf_of, borrower)
    assert client.get(f"/api/v2/files/{state.file_id}/content").status_code == 200
    assert limiter.borrowed_tokens == 0


def test_request_permit_remains_held_during_final_asgi_body_send():
    import anyio

    async def run():
        limiter = anyio.CapacityLimiter(1)
        observed = []

        async def inner(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"body", "more_body": False})

        async def send(message):
            if message["type"] == "http.response.body":
                assert limiter.borrowed_tokens == 1
                await anyio.sleep(0)
                observed.append(limiter.borrowed_tokens)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        await api.RequestConcurrency(inner, limiter)({"type": "http", "path": "/bounded-body"}, receive, send)
        assert observed == [1] and limiter.borrowed_tokens == 0

    anyio.run(run)


def test_chunked_stream_overflow_is_measured_even_without_content_length(http):
    client, state = http

    def body():
        yield state.data[:5]
        yield state.data[5:] + b"overflow"

    response = client.put(
        f"/api/v2/uploads/{state.upload_id}/content",
        content=body(),
        headers=mutation_headers(**{"Content-Type": "application/octet-stream"}),
    )
    assert response.status_code == 413 and not state.store.calls and state.stored is None
    assert state.app.state.material_upload_limiter.borrowed_tokens == 0


@pytest.mark.parametrize("revocation", ["deleted", "expired"])
def test_raw_completion_revalidates_admission_after_outside_transaction_io(http, revocation):
    client, state = http
    if revocation == "deleted":
        state.store.before_put = lambda: setattr(state, "revoked", True)
    else:
        state.store.before_put = lambda: setattr(
            state.admission, "expires_at", datetime.now(UTC) - timedelta(seconds=1)
        )
    response = client.put(
        f"/api/v2/uploads/{state.upload_id}/content",
        content=state.data,
        headers=mutation_headers(**{"Content-Type": "application/octet-stream"}),
    )
    assert response.status_code == 404 and state.stored is None
    assert state.store.calls == ["put"] and state.app.state.material_upload_limiter.borrowed_tokens == 0


def test_bounded_json_rejects_oversize_before_parsing(http):
    client, state = http
    response = client.post(
        "/api/v2/materials/text",
        content=b" " * (max(2 * 1024 * 1024, 12 * 200000) + 16385),
        headers=mutation_headers(**{"Content-Type": "application/json", "Idempotency-Key": str(uuid4())}),
    )
    assert response.status_code == 413 and not state.calls


def test_standard_escaped_unicode_json_preserves_full_source_character_allowance(http):
    client, state = http
    headers = mutation_headers(**{"Content-Type": "application/json", "Idempotency-Key": str(uuid4())})
    body = {"title": "supplementary Unicode boundary", "text": "😀" * 200000, "format": "paste"}
    wire = json.dumps(body, ensure_ascii=True).encode()
    assert len(wire) > 2 * 1024 * 1024 and len(body["text"].encode()) < 2 * 1024 * 1024
    response = client.post("/api/v2/materials/text", content=wire, headers=headers)
    assert response.status_code == 202 and state.calls[-1][1]["text"] == body["text"]
    body["text"] += "😀"
    response = client.post(
        "/api/v2/materials/text", content=json.dumps(body, ensure_ascii=True).encode(), headers=headers
    )
    assert response.status_code == 400 and response.json()["code"] == "invalid_input"


def test_escaped_lone_surrogate_is_rejected_before_service_utf8_encoding(http):
    client, state = http
    wire = b'{"title":"bad Unicode","text":"\\ud800","format":"paste"}'
    response = client.post(
        "/api/v2/materials/text",
        content=wire,
        headers=mutation_headers(**{"Content-Type": "application/json", "Idempotency-Key": str(uuid4())}),
    )
    assert response.status_code == 400 and response.json()["code"] == "invalid_json" and not state.calls


@pytest.mark.parametrize(
    "locator",
    [
        {"kind": "guess", "page": 1},
        {"kind": "text", "file_sha256": "a" * 64, "char_start": False, "char_end": 2, "line_1based": 1},
        {"kind": "text", "file_sha256": "a" * 64, "char_start": 0, "char_end": 2, "line_1based": 1, "guessed_page": 7},
    ],
)
def test_fragment_http_refuses_untyped_or_extra_locator_output(http, locator):
    client, state = http
    state.bad_locator = locator
    response = client.get(f"/api/v2/files/{state.file_id}/fragments")
    assert response.status_code == 500 and response.json()["code"] == "response_contract_failed"


def test_fragment_http_positive_locator_uses_frozen_schema_and_typed_openapi(http):
    client, state = http
    response = client.get(f"/api/v2/files/{state.file_id}/fragments")
    assert response.status_code == 200 and response.json()["items"][0]["locator"]["file_sha256"] == state.blob.sha256
    shape = state.app.openapi()["components"]["schemas"]["MaterialFragment"]["properties"]["locator"]
    assert shape["discriminator"]["propertyName"] == "kind" and len(shape["oneOf"]) == 4


def test_invalid_output_does_not_exit_transaction_as_success(http):
    client, state = http
    state.bad_ticket = True
    response = client.post(
        "/api/v2/uploads", json=upload_body(state), headers=mutation_headers(**{"Idempotency-Key": str(uuid4())})
    )
    assert response.status_code == 500 and response.json()["code"] == "response_contract_failed"
    assert state.factory.commits == 1 and state.factory.rollbacks == 1


def test_commit_failure_precedes_successful_http_send(http):
    client, state = http
    state.fail_commit = True
    response = client.post(
        "/api/v2/uploads", json=upload_body(state), headers=mutation_headers(**{"Idempotency-Key": str(uuid4())})
    )
    assert response.status_code == 503 and response.json()["code"] == "persistence_unavailable"


def test_upload_deadline_stops_slow_body_and_releases_gate(http):
    import time

    _, state = http
    app = api.create_app(
        state.factory,
        store=state.store,
        cursor_key=b"material-http-timeout-test-only-key",
        contract_path=ROOT / "contracts/v2/domain.schema.json",
        policy_path=ROOT / "contracts/v2/policy.json",
        upload_timeout_seconds=0.005,
    )

    def slow_body():
        time.sleep(0.03)
        yield state.data

    with TestClient(app, cookies={"citequest_session": str(state.actor.owner_id)}) as client:
        response = client.put(
            f"/api/v2/uploads/{state.upload_id}/content",
            content=slow_body(),
            headers=mutation_headers(**{"Content-Type": "application/octet-stream"}),
        )
    assert response.status_code == 408 and state.stored is None
    assert app.state.material_upload_limiter.borrowed_tokens == 0


def test_version_binding_rejects_fake_identity_proof(http):
    client, state = http
    version = uuid4()
    url = f"/api/v2/files/{state.file_id}/version"
    assert (
        client.post(
            url, json={"version_id": str(version), "identity_status": "hash_verified"}, headers=mutation_headers()
        ).status_code
        == 400
    )
    response = client.post(url, json={"version_id": str(version)}, headers=mutation_headers())
    assert (
        response.status_code == 200
        and response.json()["version_id"] == str(version)
        and response.json()["can_qa"] is False
    )
    response = client.post(
        url,
        content=f'{{"version_id":"{version}","version_id":"{uuid4()}"}}',
        headers=mutation_headers(**{"Content-Type": "application/json"}),
    )
    assert response.status_code == 400 and response.json()["code"] == "invalid_json"


def test_actual_openapi_has_frozen_bodies_header_key_and_no_duplicate_operations(http):
    _, state = http
    schema = state.app.openapi()
    ids = [
        operation["operationId"]
        for methods in schema["paths"].values()
        for method, operation in methods.items()
        if method in {"get", "head", "post", "put", "patch", "delete"}
    ]
    assert len(ids) == len(set(ids))
    for path, resource in [("/uploads", "UploadInput"), ("/materials/text", "TextMaterialInput")]:
        operation = schema["paths"]["/api/v2" + path]["post"]
        assert operation["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/" + resource)
        assert any(p["name"] == "Idempotency-Key" and p["required"] for p in operation["parameters"])
    assert schema["paths"]["/api/v2/files/{file_id}/content"]["get"]["responses"]["206"]["content"].get(
        "application/octet-stream"
    )
    head = schema["paths"]["/api/v2/files/{file_id}/content"]["head"]
    assert "206" not in head["responses"] and "content" not in head["responses"]["200"]


def test_formal_evidence_flag_cannot_be_unverified_even_in_extension_output():
    value = {
        "ingestion_id": str(uuid4()),
        "file_id": str(uuid4()),
        "state": "ready",
        "revision": 1,
        "error_code": None,
        "representation_id": "a" * 64,
        "lease_active": False,
        "events": [],
        "binding_source": "user_explicit",
        "identity_status": "unverified",
        "material_source_ready": True,
        "paper_source_ready": False,
        "formal_evidence_ready": True,
        "execution_outcome": "legacy_not_measured",
        "active_elapsed_seconds": None,
        "execution_budget_seconds": None,
        "deadline_at": None,
        "execution_success": False,
    }
    import json

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        api.MaterialIngestion.model_validate_json(json.dumps(value))


def test_execution_flags_distinguish_material_readiness_from_measured_success():
    from pydantic import ValidationError

    value = {
        "ingestion_id": str(uuid4()),
        "file_id": str(uuid4()),
        "state": "ready",
        "revision": 4,
        "error_code": None,
        "representation_id": "a" * 64,
        "lease_active": False,
        "events": [],
        "binding_source": "none",
        "identity_status": "unverified",
        "material_source_ready": True,
        "paper_source_ready": False,
        "formal_evidence_ready": False,
        "execution_outcome": "within_deadline",
        "active_elapsed_seconds": 0.25,
        "execution_budget_seconds": 120.0,
        "deadline_at": datetime.now(UTC).isoformat(),
        "execution_success": True,
    }
    assert api.MaterialIngestion.model_validate_json(json.dumps(value)).execution_success is True
    for outcome in ["not_started", "running", "late", "uncertain", "legacy_not_measured", "failed"]:
        with pytest.raises(ValidationError):
            api.MaterialIngestion.model_validate_json(json.dumps({**value, "execution_outcome": outcome}))
    with pytest.raises(ValidationError):
        api.MaterialIngestion.model_validate_json(json.dumps({**value, "active_elapsed_seconds": 121.0}))
