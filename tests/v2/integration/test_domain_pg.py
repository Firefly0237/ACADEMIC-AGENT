"""Actual isolated S03 PostgreSQL only; missing prerequisites fail, never skip."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from citequest.v2 import models as m
from citequest.v2.api import create_app
from citequest.v2.db import engine, sessions
from citequest.v2.domain import (
    Actor,
    CollectionCreate,
    CollectionPatch,
    DomainError,
    LibraryPatch,
    LibrarySave,
    NoteCreate,
    NotePatch,
    WorkInput,
)
from citequest.v2.pagination import Pager
from citequest.v2.security import issue_session
from citequest.v2.services import Catalog

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def factory():
    import sys

    sys.path.insert(0, str(ROOT / "scripts/s03"))
    from bootstrap import dsn

    db = engine(dsn("cq_s03_app"))
    with db.connect() as conn:
        assert conn.exec_driver_sql("SELECT current_database(),current_user").one() == ("citequest_s03", "cq_s03_app")
        current = conn.exec_driver_sql("SELECT version_num FROM cq_app.alembic_version").scalar_one()
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        history = ScriptDirectory.from_config(Config(str(ROOT / "alembic-app.ini")))
        assert "s03_0005" in {revision.revision for revision in history.iterate_revisions(current, "base")}
    yield sessions(db)
    db.dispose()


@pytest.fixture
def actors(factory):
    a, b = Actor(uuid4()), Actor(uuid4())
    with factory.begin() as db:
        db.add_all([m.Owner(id=a.owner_id, name="S03 synthetic A"), m.Owner(id=b.owner_id, name="S03 synthetic B")])
    return a, b


def create(factory, actor, key=None):
    request = WorkInput(title="Synthetic unknown metadata", authors=None, year=None, client_operation_id=key or uuid4())
    with factory.begin() as db:
        return request, Catalog(db, actor).create_work(request)


def test_unknown_private_metadata_and_immutable_idempotency_receipt(factory, actors):
    a, b = actors
    request, first = create(factory, a)
    assert first["authors"] is None and first["year"] is None
    work_id = UUID(first["work_id"])
    with factory.begin() as db:
        with pytest.raises(DomainError, match="not_found"):
            Catalog(db, b).work(work_id)
        Catalog(db, a).work(work_id).title = "Later metadata version"
    with factory.begin() as db:
        assert Catalog(db, a).create_work(request) == first
        changed = request.model_copy(update={"title": "Different request"})
        with pytest.raises(DomainError, match="idempotency_conflict"):
            Catalog(db, a).create_work(changed)


def test_current_acl_overrides_prior_receipt_and_saved_item(factory, actors):
    a, _b = actors
    request, first = create(factory, a)
    work_id = UUID(first["work_id"])
    with factory.begin() as db:
        saved = Catalog(db, a).save(LibrarySave(work_id=work_id, collection_id=None, client_operation_id=uuid4()))
        db.get(m.WorkAccess, (a.owner_id, work_id)).revoked = True
    with factory.begin() as db:
        for action in (
            lambda: Catalog(db, a).create_work(request),
            lambda: Catalog(db, a).item(UUID(saved["library_item_id"])),
        ):
            with pytest.raises(DomainError, match="not_found"):
                action()
        rows, cursor = Pager(b"x" * 32).page(Catalog(db, a), "library")
        assert rows == [] and cursor is None


def test_transaction_failure_rolls_back_claim_entity_and_outbox(factory, actors):
    a, _b = actors
    key, work_id = uuid4(), uuid4()
    with pytest.raises(RuntimeError, match="injected"), factory.begin() as db:
        service = Catalog(db, a)

        def fail():
            db.add(
                m.Work(
                    id=work_id,
                    security_domain=a.security_domain,
                    title="Rollback fixture",
                    title_normalized="rollback",
                    authors=None,
                    year=None,
                )
            )
            service.outbox(work_id, 1, "upsert", {})
            db.flush()
            raise RuntimeError("injected_after_entity_and_outbox")

        service.operation(key, {"operation": "injected"}, fail)
    with factory.begin() as db:
        assert db.get(m.Operation, (a.owner_id, key)) is None
        assert db.get(m.Work, work_id) is None
        assert db.scalar(select(m.Outbox.id).where(m.Outbox.entity_id == str(work_id))) is None


def test_project_lookup_rejects_other_owner(factory, actors):
    a, b = actors
    project_id = uuid4()
    with factory.begin() as db:
        db.add(m.Project(id=project_id, owner_id=a.owner_id, title="Explicit synthetic project boundary"))
    with factory.begin() as db:
        assert Catalog(db, a).project(project_id).owner_id == a.owner_id
        with pytest.raises(DomainError, match="not_found"):
            Catalog(db, b).project(project_id)


def test_true_concurrent_same_operation_has_one_effect(factory, actors):
    a, _b = actors
    key, barrier = uuid4(), Barrier(2)

    def execute():
        barrier.wait(timeout=10)
        return create(factory, a, key)[1]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: execute(), range(2)))
    assert results[0] == results[1]
    with factory.begin() as db:
        assert db.scalar(select(func.count()).select_from(m.WorkAccess).where(m.WorkAccess.owner_id == a.owner_id)) == 1
        assert (
            db.scalar(select(func.count()).select_from(m.Outbox).where(m.Outbox.entity_id == results[0]["work_id"]))
            == 1
        )


def test_cross_owner_collections_revision_cycle_and_note_crud(factory, actors):
    a, b = actors
    _, work = create(factory, a)
    with factory.begin() as db:
        sa, sb = Catalog(db, a), Catalog(db, b)
        parent = UUID(
            sa.create_collection(CollectionCreate(name="A", parent_id=None, client_operation_id=uuid4()))[
                "collection_id"
            ]
        )
        foreign = UUID(
            sb.create_collection(CollectionCreate(name="B", parent_id=None, client_operation_id=uuid4()))[
                "collection_id"
            ]
        )
        saved = UUID(
            sa.save(LibrarySave(work_id=UUID(work["work_id"]), collection_id=parent, client_operation_id=uuid4()))[
                "library_item_id"
            ]
        )
        with pytest.raises(DomainError, match="not_found"):
            sa.patch_item(saved, LibraryPatch(expected_revision=1, tags=[], collection_ids=[foreign]))
        sa.patch_item(saved, LibraryPatch(expected_revision=1, tags=["tag"], collection_ids=[parent]))
        with pytest.raises(DomainError, match="revision_conflict"):
            sa.patch_item(saved, LibraryPatch(expected_revision=1, tags=[], collection_ids=[]))
        child = UUID(
            sa.create_collection(CollectionCreate(name="child", parent_id=parent, client_operation_id=uuid4()))[
                "collection_id"
            ]
        )
        with pytest.raises(DomainError, match="collection_cycle"):
            sa.patch_collection(parent, CollectionPatch(name="A", parent_id=child, expected_revision=1))
        note = UUID(
            sa.create_note(
                NoteCreate(
                    library_item_id=saved,
                    body="original",
                    evidence_ids=[],
                    expected_revision=None,
                    client_operation_id=uuid4(),
                )
            )["note_id"]
        )
        sa.patch_note(note, NotePatch(body="changed", expected_revision=1))
        assert sa.note(note).body == "changed"
        with pytest.raises(DomainError, match="not_found"):
            sb.note(note)
        sa.delete_note(note)
        with pytest.raises(DomainError, match="not_found"):
            sa.note(note)
        sa.delete_item(saved)
        assert sa.item(saved, deleted=True).deleted_at is not None
        sa.restore_item(saved)
        assert sa.item(saved).deleted_at is None


def test_signed_cursor_owner_binding_and_stable_ties(factory, actors):
    a, b = actors
    for _ in range(3):
        _, work = create(factory, a)
        with factory.begin() as db:
            Catalog(db, a).save(
                LibrarySave(work_id=UUID(work["work_id"]), collection_id=None, client_operation_id=uuid4())
            )
    pager = Pager(b"s" * 32)
    with factory.begin() as db:
        first, cursor = pager.page(Catalog(db, a), "library", 1)
        second, _ = pager.page(Catalog(db, a), "library", 1, cursor)
        assert first[0].id != second[0].id
        with pytest.raises(DomainError, match="cursor"):
            pager.page(Catalog(db, b), "library", 1, cursor)
        with pytest.raises(DomainError, match="cursor"):
            pager.page(Catalog(db, a), "library", 1, cursor[:-5] + "xxxxx")


def test_real_pg_http_session_csrf_and_output_contract(factory, actors):
    a, b = actors
    with factory.begin() as db:
        token, csrf = issue_session(db, a)
        token_b, _csrf_b = issue_session(db, b)
    app = create_app(factory, cursor_key=b"h" * 32, contract_path=ROOT / "contracts/v2/domain.schema.json")
    with TestClient(app, base_url="http://127.0.0.1:18000", cookies={"citequest_session": token}) as client:
        payload = {"title": "HTTP synthetic", "authors": None, "year": None, "client_operation_id": str(uuid4())}
        assert client.post("/api/v2/works", json=payload).status_code == 403
        headers = {"Origin": "http://127.0.0.1:18000", "X-CSRF-Token": csrf}
        response = client.post("/api/v2/works", json=payload, headers=headers)
        assert response.status_code == 201, response.text
        work = response.json()
        assert work["year"] is None
        assert (
            client.post("/api/v2/works", json={**payload, "owner_id": str(b.owner_id)}, headers=headers).status_code
            == 400
        )
        work_ids = [work["work_id"]]
        for _ in range(2):
            work_ids.append(
                client.post(
                    "/api/v2/works", json={**payload, "client_operation_id": str(uuid4())}, headers=headers
                ).json()["work_id"]
            )
        for work_id in work_ids:
            assert (
                client.post(
                    "/api/v2/library/items",
                    json={"work_id": work_id, "collection_id": None, "client_operation_id": str(uuid4())},
                    headers=headers,
                ).status_code
                == 201
            )
        page = client.get("/api/v2/library/items", params={"page_size": 1})
        assert len(page.json()["items"]) == 1 and page.json()["next_cursor"]
        assert client.get("/api/v2/library/items", params={"page_size": 51}).status_code == 400
        assert client.get("/api/v2/library/items", params={"page_size": "bad"}).status_code == 400
        assert (
            client.get(
                "/api/v2/library/items", params={"page_size": 2, "cursor": page.json()["next_cursor"]}
            ).status_code
            == 400
        )
        client.cookies.set("citequest_session", token_b)
        assert client.get("/api/v2/works/" + work["work_id"]).status_code == 404


def test_raw_source_unknown_version_and_scoped_attachment_fk(factory, actors, tmp_path):
    import hashlib
    import io
    from datetime import UTC, datetime

    from sqlalchemy.exc import IntegrityError

    from citequest.v2.domain import SourceInput
    from citequest.v2.storage import CAS

    a, b = actors
    _, work = create(factory, a)
    work_id = UUID(work["work_id"])
    raw = b"S03 synthetic source bytes, not a research paper"
    source = SourceInput(
        source="synthetic",
        native_id="00000123:" + str(a.owner_id),
        source_revision=None,
        retrieved_at=datetime.now(UTC),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        source_url=None,
        rights_status="unknown",
        license=None,
    )
    store = CAS(ROOT / "data/runtime-v2-app/blobs")
    with factory.begin() as db:
        service = Catalog(db, a)
        first = service.record_source(work_id, source, raw, store)
        assert service.record_source(work_id, source, raw, store) == first
        assert first["source_revision"] is None and first["license"] is None and first["native_id"] == source.native_id
        version = service.record_version(work_id, source.raw_sha256, "source_unknown", None)
        assert service.record_version(work_id, source.raw_sha256, "source_unknown", None) == version
        assert version["version_label"] is None
        saved = service.save(LibrarySave(work_id=work_id, collection_id=None, client_operation_id=uuid4()))
        item_id = UUID(saved["library_item_id"])
        blob = store.put(a.security_domain, io.BytesIO(raw), len(raw))
        file = service.register_file(store, blob, "text/plain", UUID(version["version_id"]), uuid4())
        file_id = UUID(file["file_id"])
        assert file["status"] == "queued" and file["can_qa"] is False
        service.attach(item_id, file_id)
        db.flush()
        with pytest.raises(DomainError, match="not_found"):
            Catalog(db, b).file(file_id)
        with pytest.raises(IntegrityError), db.begin_nested():
            db.add(m.ItemFile(owner_id=b.owner_id, item_id=item_id, file_id=file_id))
            db.flush()
        service.delete_file(file_id)
        db.flush()
        assert db.get(m.ItemFile, (a.owner_id, item_id, file_id)) is None
        assert db.get(m.Blob, (a.security_domain, blob.sha256)) is not None
        assert store.read(blob) == raw


def test_history_is_owned_and_deleting_collection_does_not_delete_item(factory, actors):
    a, b = actors
    _, work = create(factory, a)
    with factory.begin() as db:
        sa, sb = Catalog(db, a), Catalog(db, b)
        collection = UUID(
            sa.create_collection(CollectionCreate(name="disposable", parent_id=None, client_operation_id=uuid4()))[
                "collection_id"
            ]
        )
        item = UUID(
            sa.save(LibrarySave(work_id=UUID(work["work_id"]), collection_id=collection, client_operation_id=uuid4()))[
                "library_item_id"
            ]
        )
        history = sa.append_history("view", work["work_id"])
        db.flush()
        with pytest.raises(DomainError, match="not_found"):
            sb.delete_history(history)
        sa.delete_collection(collection)
        assert sa.item(item).deleted_at is None
        sa.delete_history(history)
        db.flush()
        assert db.get(m.History, history) is None


def test_private_doi_and_source_identity_do_not_squat_public_catalog(factory, actors):
    import hashlib
    from datetime import UTC, datetime

    from citequest.v2.domain import IdentifierInput, SourceInput
    from citequest.v2.storage import CAS

    a, b = actors
    doi = "10.12345/" + uuid4().hex

    def request():
        return WorkInput(
            title="Domain-isolated synthetic",
            authors=None,
            year=None,
            identifiers=[IdentifierInput(namespace="doi", raw=doi)],
            client_operation_id=uuid4(),
        )

    with factory.begin() as db:
        wa = Catalog(db, a).create_work(request())
        wb = Catalog(db, b).create_work(request())
        assert wa["work_id"] != wb["work_id"]
        with pytest.raises(DomainError, match="not_found"):
            Catalog(db, b).work(UUID(wa["work_id"]))
        publisher = Actor(uuid4(), publisher=True)
        db.add(m.Owner(id=publisher.owner_id, name="Synthetic trusted publisher", can_publish=True))
        db.flush()
        canonical = Catalog(db, publisher).create_work(request(), public=True)
        assert canonical["work_id"] not in {wa["work_id"], wb["work_id"]}
        assert Catalog(db, a).create_work(request())["work_id"] == canonical["work_id"]
        assert Catalog(db, b).create_work(request())["work_id"] == canonical["work_id"]
        assert db.get(m.Identifier, ("public", "doi", doi)).work_id == UUID(canonical["work_id"])
        raw = b"Explicit synthetic local source; not a public research paper"
        source = SourceInput(
            source="domain-fixture",
            native_id=doi,
            source_revision=None,
            retrieved_at=datetime.now(UTC),
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            source_url=None,
            rights_status="unknown",
            license=None,
        )
        store = CAS(ROOT / "data/runtime-v2-app/blobs")
        Catalog(db, a).record_source(UUID(wa["work_id"]), source, raw, store)
        Catalog(db, b).record_source(UUID(wb["work_id"]), source, raw, store)
        # Public Work accessibility must not grant another owner's private version.
        upload_source = source.model_copy(update={"native_id": "uploaded-file:" + str(uuid4())})
        Catalog(db, a).record_source(UUID(canonical["work_id"]), upload_source, raw, store)
        private_version = Catalog(db, a).record_version(
            UUID(canonical["work_id"]), source.raw_sha256, "source_unknown", None
        )
        with pytest.raises(DomainError, match="not_found"):
            Catalog(db, b).version(UUID(private_version["version_id"]))
        with pytest.raises(DomainError, match="invalid_version_label"):
            Catalog(db, a).record_version(UUID(wa["work_id"]), source.raw_sha256, "source_unknown", "")
        arxiv = "arxiv-private:" + uuid4().hex
        db.add(
            m.Identifier(
                security_domain=a.security_domain,
                namespace="arxiv",
                normalized=arxiv,
                raw=arxiv,
                work_id=UUID(wa["work_id"]),
                normalizer_version="synthetic",
            )
        )
        db.flush()
        # This public DOI identifies a different Work than the private arxiv ID.
        other_doi = "10.12345/" + uuid4().hex
        other = Catalog(db, publisher).create_work(
            WorkInput(
                title="Other public synthetic",
                authors=None,
                year=None,
                identifiers=[IdentifierInput(namespace="doi", raw=other_doi)],
                client_operation_id=uuid4(),
            ),
            public=True,
        )
        with pytest.raises(DomainError, match="identifier_conflict"):
            Catalog(db, a).create_work(
                WorkInput(
                    title="Mixed identities",
                    authors=None,
                    year=None,
                    identifiers=[
                        IdentifierInput(namespace="doi", raw=other_doi),
                        IdentifierInput(namespace="arxiv", raw=arxiv),
                    ],
                    client_operation_id=uuid4(),
                )
            )
        # Defense in depth even if trusted ingestion associates a private alias
        # with an otherwise public Work: a different reader cannot see that ID.
        private_alias = "private-alias:" + uuid4().hex
        db.add(
            m.Identifier(
                security_domain=a.security_domain,
                namespace="arxiv",
                normalized=private_alias,
                raw=private_alias,
                work_id=UUID(other["work_id"]),
                normalizer_version="synthetic",
            )
        )
        db.flush()
        from citequest.v2.presentation import work as present_work

        assert all(
            i["raw"] != private_alias
            for i in present_work(Catalog(db, b), Catalog(db, b).work(UUID(other["work_id"])))["identifiers"]
        )


def test_delete_collection_cannot_race_save_into_a_deleted_parent(factory, actors):
    import time
    from threading import Event

    a, _b = actors
    _, work = create(factory, a)
    with factory.begin() as db:
        collection = UUID(
            Catalog(db, a).create_collection(
                CollectionCreate(name="race-parent", parent_id=None, client_operation_id=uuid4())
            )["collection_id"]
        )
    started, child_pid = Event(), []

    def save_after_delete_started():
        try:
            with factory.begin() as db:
                child_pid.append(db.connection().exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
                started.set()
                Catalog(db, a).save(
                    LibrarySave(work_id=UUID(work["work_id"]), collection_id=collection, client_operation_id=uuid4())
                )
            return "unexpected_success"
        except DomainError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=1) as pool:
        with factory.begin() as deleting:
            Catalog(deleting, a).delete_collection(collection)
            deleting.flush()
            future = pool.submit(save_after_delete_started)
            assert started.wait(5)
            deadline = time.monotonic() + 5
            waiting = False
            while time.monotonic() < deadline and not future.done():
                with factory.begin() as probe:
                    waiting = (
                        probe.connection()
                        .exec_driver_sql(
                            "SELECT wait_event_type='Lock' FROM pg_stat_activity WHERE pid=%s", (child_pid[0],)
                        )
                        .scalar()
                    )
                if waiting:
                    break
                time.sleep(0.01)
            assert waiting, "The writer must wait for the same owner mutation lock"
        assert future.result(timeout=5) == "not_found"
    with factory.begin() as db:
        assert (
            db.scalar(
                select(func.count()).select_from(m.ItemCollection).where(m.ItemCollection.collection_id == collection)
            )
            == 0
        )


def test_delete_file_cannot_race_a_late_attachment(factory, actors):
    import hashlib
    import io
    import time
    from datetime import UTC, datetime
    from threading import Event

    from citequest.v2.domain import SourceInput
    from citequest.v2.storage import CAS

    a, _b = actors
    _, work = create(factory, a)
    store = CAS(ROOT / "data/runtime-v2-app/blobs")
    raw = b"Synthetic attachment race fixture"
    source = SourceInput(
        source="fixture",
        native_id=str(uuid4()),
        source_revision=None,
        retrieved_at=datetime.now(UTC),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        source_url=None,
        rights_status="unknown",
        license=None,
    )
    with factory.begin() as db:
        service = Catalog(db, a)
        service.record_source(UUID(work["work_id"]), source, raw, store)
        version = service.record_version(UUID(work["work_id"]), source.raw_sha256, "source_unknown", None)
        file = service.register_file(
            store,
            store.put(a.security_domain, io.BytesIO(raw), len(raw)),
            "text/plain",
            UUID(version["version_id"]),
            uuid4(),
        )
        item = service.save(LibrarySave(work_id=UUID(work["work_id"]), collection_id=None, client_operation_id=uuid4()))
    item_id, file_id = UUID(item["library_item_id"]), UUID(file["file_id"])
    started, child_pid = Event(), []

    def late_attach():
        try:
            with factory.begin() as db:
                child_pid.append(db.connection().exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
                started.set()
                Catalog(db, a).attach(item_id, file_id)
            return "unexpected_success"
        except DomainError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=1) as pool:
        with factory.begin() as deleting:
            Catalog(deleting, a).delete_file(file_id)
            deleting.flush()
            future = pool.submit(late_attach)
            assert started.wait(5)
            deadline, waiting = time.monotonic() + 5, False
            while time.monotonic() < deadline and not future.done():
                with factory.begin() as probe:
                    waiting = (
                        probe.connection()
                        .exec_driver_sql(
                            "SELECT wait_event_type='Lock' FROM pg_stat_activity WHERE pid=%s", (child_pid[0],)
                        )
                        .scalar()
                    )
                if waiting:
                    break
                time.sleep(0.01)
            assert waiting
        assert future.result(timeout=5) == "not_found"
    with factory.begin() as db:
        assert db.get(m.ItemFile, (a.owner_id, item_id, file_id)) is None


def test_operation_savepoint_is_atomic_even_if_caller_handles_failure(factory, actors):
    a, _b = actors
    key, work_id = uuid4(), uuid4()
    with factory.begin() as db:
        service = Catalog(db, a)

        def fail():
            db.add(
                m.Work(
                    id=work_id,
                    security_domain=a.security_domain,
                    title="Handled failure",
                    title_normalized="handled",
                    authors=None,
                    year=None,
                )
            )
            service.outbox(work_id, 1, "upsert", {})
            db.flush()
            raise RuntimeError("handled_failure")

        with pytest.raises(RuntimeError, match="handled_failure"):
            service.operation(key, {"operation": "injected"}, fail)
    with factory.begin() as db:
        assert db.get(m.Work, work_id) is None
        assert db.get(m.Operation, (a.owner_id, key)) is None
        assert db.scalar(select(m.Outbox.id).where(m.Outbox.entity_id == str(work_id))) is None


def test_outbox_has_exact_immutable_versioned_payload_and_state_can_progress(factory, actors):
    from sqlalchemy.exc import IntegrityError

    from citequest.v2.services import digest

    a, _b = actors
    _, work = create(factory, a)
    with factory.begin() as db:
        event = db.scalar(select(m.Outbox).where(m.Outbox.entity_id == work["work_id"], m.Outbox.entity_revision == 1))
        assert event.payload["data"] == work
        assert digest(event.payload) == event.payload_sha256
        assert event.security_domain == a.security_domain and event.schema_version == "cq.outbox.payload/1"
        original = event.payload
        Catalog(db, a).work(UUID(work["work_id"])).title = "Later live metadata"
        db.flush()
        assert event.payload == original and event.payload["data"]["title"] == "Synthetic unknown metadata"
        event.state, event.attempt_count = "committed", 1
        db.flush()
        with pytest.raises(IntegrityError), db.begin_nested():
            event.payload = {"corrupt": True}
            db.flush()
