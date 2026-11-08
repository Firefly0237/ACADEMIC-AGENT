"""Read-only current real PG/CAS controls; no adopted membership is created."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from citequest.data_v2 import models as m
from citequest.data_v2.application_contract import (
    CHAIN_PATH,
    EVALUATION,
    POLICY,
    RIGHTS_PATH,
    SNAPSHOT,
    original_inventory,
    read_pinned,
)
from citequest.data_v2.application_freeze import qualify_application, qualify_member, sealed_resolution
from citequest.data_v2.application_members import ApplicationMember
from citequest.data_v2.landing import Landing
from citequest.data_v2.native_identity import CHAIN_SHA, RIGHTS_SHA
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor, DomainError
from citequest.v2.services import Catalog, digest
from citequest.v2.storage import CAS
from citequest.v2.storage_config import app_storage

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def factory():
    sys.path.insert(0, str(ROOT / "scripts/s03"))
    from bootstrap import dsn

    db = engine(dsn("cq_s03_app"))
    yield sessions(db)
    db.dispose()


@pytest.fixture
def actual(factory):
    with factory.begin() as db:
        db.execute(text("SET TRANSACTION READ ONLY"))
        snapshot = db.get(m.SourceSnapshot, SNAPSHOT)
        landing = Landing(
            Catalog(db, Actor(snapshot.importer_owner_id, publisher=True)),
            policy_sha256=POLICY,
            evaluation_sha256=EVALUATION,
        )
        chain = original_inventory(read_pinned(ROOT, CHAIN_PATH, CHAIN_SHA, 32 * 1024**2))
        rights = original_inventory(read_pinned(ROOT, RIGHTS_PATH, RIGHTS_SHA, 32 * 1024**2))
        yield landing, app_storage(ROOT).cas(), snapshot, chain[0], rights[0]


def test_real_candidate_has_bytes_rights_all_locators_and_exact_PaperRef(actual):
    landing, store, snapshot, candidate, rights = actual
    value = qualify_member(landing, store, ROOT, snapshot, candidate, rights)
    assert value["rank"] == 0 and value["native_id"] == "253761972"
    assert value["source_reference"]["work_version_id"] == value["work_version_id"]
    assert value["annotation_locator_count"] > 5 and value["characters"] > 2000
    assert value["license_id"] == "CC-BY-4.0" and value["anonymous_download_receipts"]
    assert not value["original_PDF_complete_extraction_claimed"]


def test_wrong_owned_legacy_CAS_root_rejected_before_qualification(actual):
    landing, _, _, _, _ = actual
    with pytest.raises(ValueError, match="application_storage_root_mismatch"):
        qualify_application(landing, CAS(ROOT / "data/runtime-v2-app/cas"), ROOT)


def test_actual_importer_still_requires_publisher_capability(actual):
    landing, store, _, _, _ = actual
    landing.catalog.actor = Actor(landing.catalog.actor.owner_id, publisher=False)
    with pytest.raises(DomainError, match="publisher_required"):
        qualify_application(landing, store, ROOT)


@pytest.mark.parametrize("mutation", ["body_tamper", "missing_proof", "foreign_namespace"])
def test_real_reader_rejects_tamper_missing_proof_or_namespace(actual, monkeypatch, mutation):
    landing, store, snapshot, candidate, rights = actual
    if mutation == "body_tamper":
        read = store.read

        def tampered(blob):
            raw = read(blob)
            return raw + b" " if blob.sha256 == candidate["representation_json_sha256"] else raw

        monkeypatch.setattr(store, "read", tampered)
    elif mutation == "foreign_namespace":
        from citequest.data_v2.landing import Landing as LandingType

        original = LandingType.snapshot

        # Copy attributes separately because explicit duplicate kwargs must not
        # hide the intended namespace rejection with a Python argument error.
        def bad_snapshot(self, *args, **kwargs):
            value = original(self, *args, **kwargs)
            values = {c.name: getattr(value, c.name) for c in value.__table__.columns}
            values["source_namespace"] = "foreign"
            return SimpleNamespace(**values)

        monkeypatch.setattr(LandingType, "snapshot", bad_snapshot)
    else:
        from citequest.data_v2.resolution import Resolver

        latest = Resolver.latest

        def missing(self, row_id):
            value = latest(self, row_id)
            values = {c.name: getattr(value, c.name) for c in value.__table__.columns}
            values.update(proof={}, proof_sha256=digest({}))
            return SimpleNamespace(**values)

        monkeypatch.setattr(Resolver, "latest", missing)
    with pytest.raises((DomainError, ValueError)):
        qualify_member(landing, store, ROOT, snapshot, candidate, rights)


def test_frozen_reader_rejects_unsealed_representation_without_DB_change(actual, monkeypatch):
    landing, _, snapshot, candidate, rights = actual
    value = qualify_member(landing, app_storage(ROOT).cas(), ROOT, snapshot, candidate, rights)
    row = landing.db.get(m.SourceRow, __import__("uuid").UUID(value["source_row_id"]))
    seal = {"schema_version": "cq.application-source-seal/2", "membership_table": "cq_app.application_corpus_members"}
    fake = SimpleNamespace(id=snapshot.id, seal=seal, seal_sha256=digest(seal))
    get = landing.db.get
    monkeypatch.setattr(
        landing.db,
        "get",
        lambda entity, identity, **kwargs: None if entity is ApplicationMember else get(entity, identity, **kwargs),
    )
    with pytest.raises(DomainError, match="not_in_frozen_application"):
        sealed_resolution(landing, fake, row, "0" * 64)
