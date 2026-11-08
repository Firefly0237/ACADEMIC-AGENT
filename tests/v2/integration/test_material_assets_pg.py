"""Real PG/CAS durability controls; the one actual native text input is synthetic."""
from __future__ import annotations

import io
import sys
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError

from citequest.materials_v2 import models as m
from citequest.materials_v2.artifacts import Asset, register_assets
from citequest.materials_v2.processor import MaterialProcessor
from citequest.materials_v2.services import MaterialService
from citequest.v2 import models as core
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor, DomainError
from citequest.v2.storage import CAS, StoredBlob

ROOT = Path(__file__).resolve().parents[3]
STORE = CAS(ROOT / "data/runtime-v2-app/blobs")


@pytest.fixture(scope="module")
def fixture():
    sys.path.insert(0, str(ROOT / "scripts/s03"))
    from bootstrap import dsn

    database = engine(dsn("cq_s03_app"))
    factory = sessions(database)
    actor, stranger = Actor(uuid4()), Actor(uuid4())
    with factory.begin() as db:
        db.add_all([core.Owner(id=actor.owner_id, name="S04 synthetic artifact durability controls"),
                    core.Owner(id=stranger.owner_id, name="S04 synthetic artifact foreign owner")])
    with factory.begin() as db:
        file = MaterialService(db, actor, STORE).ingest_text({"title": "S04 synthetic artifact source",
                 "text": "Original synthetic source\r\n科研𝒙😀 evidence locator.\n", "format": "txt"}, key=uuid4())
    file_id = UUID(file["file_id"])
    result = MaterialProcessor(factory, STORE, parser_python=ROOT / "data/runtime-v2/.venv/Scripts/python.exe", root=ROOT,
                               artifact_root=ROOT / "data/runtime-s04/ingestion-artifacts").run_next(file_id=file_id)
    assert result["execution_success"] is True, result
    yield factory, actor, stranger, file_id, result["representation_id"], file["sha256"]
    database.dispose()


def test_actual_text_publish_retains_and_reverifies_source_assets(fixture):
    factory, actor, _, file_id, _, source_sha = fixture
    with factory.begin() as db:
        service = MaterialService(db, actor, STORE)
        assets = service.get_artifacts(file_id)
        assert {r["role"] for r in assets["items"]} >= {"parser_result", "normalized_text", "parser_resource_receipt", "chunk_resource_receipt"}
        for row in assets["items"]:
            assert row["provenance"]["source_sha256"] == source_sha
            assert row["provenance"]["acquisition_mode"] == "captured_before_publish"
            blob, media, name = service.read_artifact(file_id, row["role"])
            assert media == "application/json" and name == row["role"] + ".json"
            assert len(STORE.read(blob)) == row["size_bytes"]


def test_registered_asset_replay_is_idempotent(fixture):
    factory, actor, _, _, representation, _ = fixture
    with factory.begin() as db:
        rows = db.scalars(select(m.ArtifactAsset).where(m.ArtifactAsset.representation_id == representation)).all()
        assets = [Asset(r.role, StoredBlob(r.security_domain, r.sha256, r.size_bytes), r.provenance) for r in rows]
        register_assets(db, STORE, owner_id=actor.owner_id, representation_id=representation, assets=assets)
        assert db.scalar(select(func.count()).select_from(m.ArtifactAsset).where(m.ArtifactAsset.representation_id == representation)) == len(assets)


def test_cross_owner_manifest_and_asset_reads_fail(fixture):
    factory, _, stranger, file_id, representation, _ = fixture
    with factory.begin() as db:
        service = MaterialService(db, stranger, STORE)
        for operation in (lambda: service.get_artifacts(file_id), lambda: service.read_artifact(file_id, "parser_result"),
                          lambda: register_assets(db, STORE, owner_id=stranger.owner_id, representation_id=representation, assets=[])):
            with pytest.raises(DomainError, match="not_found"):
                operation()


def test_cas_publish_before_sql_rollback_leaves_explicit_unreferenced_orphan(fixture):
    factory, actor, _, _, representation, source_sha = fixture
    payload = ("S04 SYNTHETIC SQL ROLLBACK ASSET " + str(uuid4())).encode()
    blob = STORE.put(actor.security_domain, io.BytesIO(payload), len(payload))
    asset = Asset("synthetic_rollback_control", blob, {"source_sha256": source_sha, "synthetic_control": True})
    with pytest.raises(RuntimeError, match="synthetic transaction failure"), factory.begin() as db:
        register_assets(db, STORE, owner_id=actor.owner_id, representation_id=representation, assets=[asset])
        assert db.get(m.ArtifactAsset, (actor.owner_id, representation, asset.role)) is not None
        raise RuntimeError("synthetic transaction failure")
    assert STORE.read(blob) == payload  # Retained, never silently garbage collected.
    with factory.begin() as db:
        assert db.get(m.ArtifactAsset, (actor.owner_id, representation, asset.role)) is None
        assert db.get(core.Blob, (blob.security_domain, blob.sha256)) is None


def test_changed_asset_cannot_replace_existing_role_and_db_is_immutable(fixture):
    factory, actor, _, _, representation, source_sha = fixture
    payload = ("S04 SYNTHETIC CONFLICT " + str(uuid4())).encode()
    blob = STORE.put(actor.security_domain, io.BytesIO(payload), len(payload))
    with factory.begin() as db:
        old = db.get(m.ArtifactAsset, (actor.owner_id, representation, "parser_result")).sha256
        with pytest.raises(DomainError, match="artifact_asset_immutable_conflict"):
            register_assets(db, STORE, owner_id=actor.owner_id, representation_id=representation,
                            assets=[Asset("parser_result", blob, {"source_sha256": source_sha})])
        assert db.get(m.ArtifactAsset, (actor.owner_id, representation, "parser_result")).sha256 == old
    with pytest.raises(DBAPIError, match="immutable"), factory.begin() as db:
        db.execute(text("UPDATE cq_app.material_artifact_assets SET provenance='{}'::jsonb WHERE owner_id=:owner AND representation_id=:representation"),
                   {"owner": actor.owner_id, "representation": representation})


def test_bad_source_metadata_and_nonexistent_bytes_never_register(fixture):
    factory, actor, _, _, representation, _ = fixture
    blob = StoredBlob(actor.security_domain, "0" * 64, 10)
    with factory.begin() as db:
        with pytest.raises(DomainError, match="artifact_asset_source_mismatch"):
            register_assets(db, STORE, owner_id=actor.owner_id, representation_id=representation,
                            assets=[Asset("synthetic_bad_source", blob, {"source_sha256": "1" * 64})])
        source_sha = db.get(m.ParsedRepresentation, representation).source_sha256
        with pytest.raises(FileNotFoundError):
            register_assets(db, STORE, owner_id=actor.owner_id, representation_id=representation,
                            assets=[Asset("synthetic_missing_bytes", blob, {"source_sha256": source_sha})])
        assert db.get(m.ArtifactAsset, (actor.owner_id, representation, "synthetic_missing_bytes")) is None
