"""Actual PG/CAS/HTTP controls; provider fixtures are explicitly synthetic.

Anonymous real official network runs are recorded separately, not mocked here.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from citequest.materials_v2.api import create_app
from citequest.materials_v2.reference_fetch import FetchResult
from citequest.materials_v2.services import MaterialService
from citequest.v2 import models as core
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor, SourceInput, WorkInput
from citequest.v2.security import issue_session
from citequest.v2.services import Catalog
from citequest.v2.storage import CAS, StoredBlob

ROOT = Path(__file__).resolve().parents[3]
ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1706.03762v1</id>
<title>S04 synthetic HTTP fixture</title><summary>Exact synthetic abstract</summary>
<author><name>Synthetic Author</name></author><published>2017-06-12T00:00:00Z</published></entry></feed>"""


@pytest.fixture
def reference_http():
    sys.path.insert(0, str(ROOT / "scripts/s03"))
    from bootstrap import dsn

    connection = engine(dsn("cq_s03_app"))
    factory = sessions(connection)
    actors = [Actor(uuid4()), Actor(uuid4())]
    with factory.begin() as db:
        assert db.execute(select(core.Owner).limit(1)) is not None
        db.add_all([core.Owner(id=actor.owner_id, name="S04 synthetic reference HTTP fixture") for actor in actors])
        db.flush()
        tokens = [issue_session(db, actor) for actor in actors]
    store = CAS(ROOT / "data/runtime-v2-app/blobs")

    class TrackedFactory:
        active = 0

        @contextmanager
        def begin(self):
            self.active += 1
            try:
                with factory.begin() as db:
                    yield db
            finally:
                self.active -= 1

    tracked = TrackedFactory()
    app = create_app(
        tracked,
        store=store,
        cursor_key=b"s04-reference-test-only-signing-key",
        contract_path=ROOT / "contracts/v2/domain.schema.json",
        policy_path=ROOT / "contracts/v2/policy.json",
    )

    class Fetcher:
        calls = 0

        async def fetch(self, url, **kwargs):
            assert tracked.active == 0, "network held a SQL transaction"
            self.calls += 1
            if "/api/query?" in url:
                return FetchResult(url, 200, "application/atom+xml", ATOM)
            if "/pdf/" in url:
                return FetchResult(url, 200, "application/pdf", b"%PDF-1.7\nS04 synthetic bytes; not a real paper")
            return FetchResult(
                url,
                200,
                "application/json",
                json.dumps(
                    {
                        "status": "ok",
                        "message-type": "work",
                        "message": {"DOI": "10.1234/test", "title": ["Synthetic provider metadata"]},
                    }
                ).encode(),
            )

    fetcher = Fetcher()
    # The controller resolves its captured fetcher; patch only its fetch method.
    app.state.reference_fetcher.fetch = fetcher.fetch
    with TestClient(app, cookies={"citequest_session": tokens[0][0]}) as client:
        yield client, actors, tokens, factory, store, fetcher
    connection.dispose()


def headers(tokens, key, actor=0):
    return {"Origin": "http://127.0.0.1:18000", "X-CSRF-Token": tokens[actor][1], "Idempotency-Key": str(key)}


def test_crossref_metadata_no_fake_file_raw_cas_idempotency_and_conflict(reference_http):
    client, actors, tokens, factory, _store, fetcher = reference_http
    key = uuid4()
    body = {"doi": "10.1234/test", "project_id": None}
    response = client.post("/api/v2/imports/doi", json=body, headers=headers(tokens, key))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["state"] == "metadata_only" and result["file"] is None and result["matches"] == []
    assert result["request"] == {"input_type": "doi", "value": body["doi"], "project_id": None}
    assert result["metadata"]["authors"] is None and not result["fulltext"]["rights_verification_performed"]
    store = _store
    raw = result["provenance"]["metadata_blob"]
    data = store.read(StoredBlob(actors[0].security_domain, raw["sha256"], raw["size_bytes"]))
    assert json.loads(data)["message"]["DOI"] == body["doi"]
    with factory.begin() as db:
        assert db.get(core.Blob, (actors[0].security_domain, raw["sha256"])) is not None
        assert db.scalar(select(core.File.id).where(core.File.owner_id == actors[0].owner_id)) is None
    replay = client.post("/api/v2/imports/doi", json=body, headers=headers(tokens, key))
    assert replay.status_code == 200 and replay.json() == result and fetcher.calls == 1
    conflict = client.post("/api/v2/imports/doi", json={**body, "doi": "10.1234/other"}, headers=headers(tokens, key))
    assert conflict.status_code == 409 and fetcher.calls == 1
    client.cookies.set("citequest_session", tokens[1][0])
    second = client.post("/api/v2/imports/doi", json=body, headers=headers(tokens, key, 1))
    assert second.status_code == 200 and fetcher.calls == 2


def test_actual_pdf_bytes_queue_unbound_unknown_rights_and_current_acl_replay(reference_http):
    client, actors, tokens, factory, _store, fetcher = reference_http
    key = uuid4()
    body = {"url": "https://arxiv.org/abs/1706.03762", "project_id": None}
    response = client.post("/api/v2/imports/url", json=body, headers=headers(tokens, key))
    assert response.status_code == 202, response.text
    result = response.json()
    file = result["file"]
    assert (
        result["state"] == "queued"
        and file["version_id"] is None
        and file["rights_status"] == "unknown"
        and not file["can_qa"]
    )
    assert file["sha256"] == hashlib.sha256(b"%PDF-1.7\nS04 synthetic bytes; not a real paper").hexdigest()
    assert (
        client.post("/api/v2/imports/url", json=body, headers=headers(tokens, key)).json() == result
        and fetcher.calls == 2
    )
    with factory.begin() as db:
        Catalog(db, actors[0]).delete_file(UUID(file["file_id"]))
    denied = client.post("/api/v2/imports/url", json=body, headers=headers(tokens, key))
    assert denied.status_code == 404 and fetcher.calls == 2


def test_unsupported_host_and_invalid_input_never_fetch(reference_http):
    client, _actors, tokens, _factory, _store, fetcher = reference_http
    response = client.post(
        "/api/v2/imports/url",
        json={"url": "https://example.com/paper.pdf", "project_id": None},
        headers=headers(tokens, uuid4()),
    )
    assert response.status_code == 200 and response.json()["state"] == "unavailable" and response.json()["file"] is None
    for url in ["http://arxiv.org/abs/1706.03762", "https://u:p@arxiv.org/abs/1706.03762", "https://127.0.0.1/x"]:
        response = client.post(
            "/api/v2/imports/url", json={"url": url, "project_id": None}, headers=headers(tokens, uuid4())
        )
        assert response.status_code in {200, 400}, response.text
    assert fetcher.calls == 0
    response = client.post(
        "/api/v2/imports/doi",
        json={"doi": "10.1234/test", "project_id": str(uuid4())},
        headers=headers(tokens, uuid4()),
    )
    assert response.status_code == 404 and fetcher.calls == 0


def test_additive_reference_and_asset_openapi_is_closed_and_has_no_missing_refs(reference_http):
    client, _actors, _tokens, _factory, _store, fetcher = reference_http
    schema = client.get("/openapi.json").json()
    definitions = schema["components"]["schemas"]
    assert definitions["ReferenceResolution"]["additionalProperties"] is False
    for path, input_type in [("doi", "DoiImportInput"), ("url", "UrlImportInput")]:
        operation = schema["paths"]["/api/v2/imports/" + path]["post"]
        assert operation["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/" + input_type)
        assert {"200", "202"} <= set(operation["responses"])
        assert any(
            parameter["name"] == "Idempotency-Key" and parameter["required"] for parameter in operation["parameters"]
        )
    stack = [schema]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            ref = value.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
                assert ref.rsplit("/", 1)[1] in definitions, ref
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    assert fetcher.calls == 0


def test_metadata_project_and_original_input_survive_receipt_without_losing_current_acl(reference_http):
    client, actors, tokens, factory, _store, fetcher = reference_http
    project_id = uuid4()
    with factory.begin() as db:
        db.add(core.Project(id=project_id, owner_id=actors[0].owner_id, title="S04 synthetic reference context"))
    key = uuid4()
    body = {"doi": "DOI:10.1234/TEST", "project_id": str(project_id)}
    response = client.post("/api/v2/imports/doi", json=body, headers=headers(tokens, key))
    assert response.status_code == 200, response.text
    original = {"input_type": "doi", "value": "DOI:10.1234/TEST", "project_id": str(project_id)}
    assert response.json()["request"] == original and response.json()["identifier"] == "10.1234/test"
    with factory.begin() as db:
        assert db.get(core.Operation, (actors[0].owner_id, key)).result["request"] == original
        # Delete only this synthetic metadata-only Project; no real user row is touched.
        db.delete(db.get(core.Project, project_id))
    denied = client.post("/api/v2/imports/doi", json=body, headers=headers(tokens, key))
    assert denied.status_code == 404 and fetcher.calls == 1




@pytest.mark.parametrize(
    "namespace,identifier,deleted",
    [("arxiv", "1706.03762", False), ("doi", "10.1234/test", True), ("doi", "10.1234/test", False)],
)
def test_work_candidate_does_not_short_circuit_wrong_version_or_deleted_file(
    reference_http, namespace, identifier, deleted
):
    client, actors, tokens, factory, store, fetcher = reference_http
    raw = b"S04 synthetic source metadata for the v1 attachment; not a paper."
    data = b"%PDF-1.7\nS04 synthetic v1 attachment"
    blob = store.put(actors[0].security_domain, io.BytesIO(data), len(data))
    with factory.begin() as db:
        catalog = Catalog(db, actors[0])
        work = catalog.create_work(
            WorkInput(
                title="S04 synthetic version mismatch",
                authors=None,
                year=None,
                identifiers=[{"namespace": namespace, "raw": identifier}],
                client_operation_id=uuid4(),
            )
        )
        source = SourceInput(
            source="s04_synthetic_reference",
            native_id=str(uuid4()),
            source_revision="v1",
            retrieved_at=datetime.now(UTC),
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            source_url=None,
            rights_status="unknown",
            license=None,
        )
        catalog.record_source(UUID(work["work_id"]), source, raw, store)
        version = catalog.record_version(UUID(work["work_id"]), source.raw_sha256, "source_unknown", "v1")
        materials = MaterialService(db, actors[0], store)
        admission = materials.create_upload(
            {
                "file_name": "s04-synthetic-reference.pdf",
                "media_type": "application/pdf",
                "size_bytes": blob.size_bytes,
                "expected_sha256": blob.sha256,
                "project_id": None,
            },
            key=uuid4(),
        )
        attached = materials.put_upload_content(UUID(admission["upload_id"]), blob)
        materials.bind_version(UUID(attached["file_id"]), UUID(version["version_id"]))
        # A synthetic ready row isolates the resolver's version/ACL decision;
        # this does not claim a native parser or trusted identity proof ran.
        db.get(core.File, UUID(attached["file_id"])).status = "ready"
        if deleted:
            catalog.delete_file(UUID(attached["file_id"]))
    path = "url" if namespace == "arxiv" else "doi"
    body = (
        {"url": "https://arxiv.org/abs/1706.03762v2", "project_id": None}
        if namespace == "arxiv"
        else {"doi": identifier, "project_id": None}
    )
    response = client.post("/api/v2/imports/" + path, json=body, headers=headers(tokens, uuid4()))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["state"] != ("existing_material") and result["file"] is None and fetcher.calls == 1
    assert result["matches"] and result["matches"][0]["files"] == []
    if namespace == "doi" and not deleted:
        grade = result["matches"][0]["file_bindings"][0]
        assert (
            grade["binding_source"] == "user_explicit"
            and grade["identity_status"] == "unverified"
            and grade["reuse_eligible"] is False
        )
    if namespace == "arxiv":
        assert result["state"] == "unavailable"
        # Provider's actual v1 is rejected, never re-labelled as requested v2.
        assert result["metadata"] is None
    else:
        assert result["state"] == "metadata_only"
