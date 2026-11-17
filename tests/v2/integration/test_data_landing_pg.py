"""Actual product PG; explicit synthetic Parquet controls, real corpus adoption negative separately."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import DBAPIError

from citequest.data_v2 import models as m
from citequest.data_v2.canonical import CanonicalResolver
from citequest.data_v2.landing import Landing
from citequest.data_v2.resolution import Resolver
from citequest.data_v2.schemas import BatchInput, ProjectionInput, RawRowLocator, SnapshotFreezeInput, SnapshotInput
from citequest.data_v2.standard_seal import QUERIES
from citequest.v2 import models as core
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor, DomainError, IdentifierInput, WorkInput
from citequest.v2.services import Catalog

ROOT = Path(__file__).resolve().parents[3]
EXPORT = ROOT / "data/s05/metadata-export/20261104T222945293337Z"
RAW = ROOT / "data/s05/litsearch/raw"
FIXTURE = ROOT / "data/s05/control-fixture"
POLICY = hashlib.sha256((ROOT / "contracts/v2/policy.json").read_bytes()).hexdigest()
EVALUATION = hashlib.sha256((ROOT / "contracts/v2/evaluation.json").read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def factory():
    import sys

    sys.path.insert(0, str(ROOT / "scripts/s03"))
    from bootstrap import dsn

    db = engine(dsn("cq_s03_app"))
    with db.connect() as conn:
        assert conn.exec_driver_sql("SELECT current_database(),current_user").one() == ("citequest_s03", "cq_s03_app")
        assert conn.exec_driver_sql("SELECT to_regclass('cq_app.ingest_resolutions')").scalar_one()
    yield sessions(db)
    db.dispose()


@pytest.fixture
def actors(factory):
    a, b, publisher = Actor(uuid4()), Actor(uuid4()), Actor(uuid4(), publisher=True)
    with factory.begin() as db:
        db.add_all(
            [
                core.Owner(id=a.owner_id, name="S05 control A"),
                core.Owner(id=b.owner_id, name="S05 control B"),
                core.Owner(id=publisher.owner_id, name="S05 control publisher", can_publish=True),
            ]
        )
    return a, b, publisher


def landing(db, actor):
    return Landing(Catalog(db, actor), policy_sha256=POLICY, evaluation_sha256=EVALUATION)


def source_rows(*, real=False):
    if real:
        return [
            ProjectionInput.model_validate_json(line)
            for line in (EXPORT / "batch-0000.jsonl").read_bytes().splitlines()
        ]
    fixture = json.loads((FIXTURE / "verification.json").read_bytes())
    assert not fixture["contains_real_articles"]
    return [
        ProjectionInput(
            native_id=str(row["corpusid"]),
            native_id_source_type="integer",
            locator=RawRowLocator(
                artifact_path=fixture["artifact"]["path"],
                artifact_sha256=fixture["artifact"]["sha256"],
                row_group=0,
                row_in_group=index,
                row_ordinal=index,
            ),
            title=row["title"],
            abstract=row["abstract"],
            field_provenance={"test_fixture": "explicitly synthetic, no real article claim"},
        )
        for index, row in enumerate(fixture["rows"])
    ]


def register(factory, actor, *, profile="litsearch-clean-exact/1", fixed=False, namespace="litsearch_clean"):
    manifest = json.loads((EXPORT / "verification.json").read_bytes())
    request = SnapshotInput(
        source_namespace=namespace,
        source_revision=manifest["source_revision"] if fixed else "service-control-" + uuid4().hex,
        source_manifest_sha256=manifest["source_snapshot_manifest_sha256"]
        if fixed
        else hashlib.sha256((FIXTURE / "verification.json").read_bytes()).hexdigest(),
        policy_sha256=POLICY,
        evaluation_sha256=EVALUATION,
        representation_profile=profile,
        purpose="standard",
        client_operation_id=uuid4(),
        # Private controls use explicitly synthetic, physically real Parquet.
        # Fixed standard adoption uses all six original source artifacts.
        artifacts=manifest["artifacts"]
        if fixed
        else [json.loads((FIXTURE / "verification.json").read_bytes())["artifact"]],
    )
    with factory.begin() as db:
        result = landing(db, actor).register_snapshot(request, RAW if fixed else FIXTURE)
    return UUID(result["snapshot_id"])


def ingest(factory, actor, snapshot_id, rows, key=None):
    request = BatchInput(rows=rows, client_operation_id=key or uuid4())
    with factory.begin() as db:
        return landing(db, actor).ingest_batch(snapshot_id, request)


def test_real_private_acl_metadata_resolution_and_revoked_receipt(factory, actors):
    a, b, _ = actors
    snapshot_id = register(factory, a)
    first = ingest(factory, a, snapshot_id, source_rows()[:1])
    projection = UUID(first["projection_ids"][0])
    with factory.begin() as db:
        assert landing(db, a).index_projection(projection).work_id is None
        with pytest.raises(DomainError, match="not_found"):
            landing(db, b).index_projection(projection)
    key = uuid4()
    with factory.begin() as db:
        result = Resolver(landing(db, a)).resolve_metadata(projection, key)
        assert result["state"] == "metadata_resolved" and result["work_version_id"] is None
        work_id = UUID(result["work_id"])
        assert Catalog(db, a).work(work_id).authors is None
        assert Catalog(db, a).work(work_id).year is None
        assert result["proof"]["license_status"] == "unknown"
        assert not result["application_profile_adopted"]
        assert landing(db, a).index_projection(projection).work_id == work_id
    with factory.begin() as db:
        db.get(core.WorkAccess, (a.owner_id, work_id)).revoked = True
    with factory.begin() as db:
        with pytest.raises(DomainError, match="not_found"):
            Resolver(landing(db, a)).resolve_metadata(projection, key)
        with pytest.raises(DomainError, match="not_found"):
            landing(db, a).index_projection(projection)


def test_duplicate_and_conflicting_batch_is_atomic_and_outbox_replayable(factory, actors):
    a, _, _ = actors
    snapshot_id, rows = register(factory, a), source_rows()[:100]
    first = ingest(factory, a, snapshot_id, rows[:1])
    changed = rows[0].model_copy(update={"title": "Deliberately changed control input, not source truth"})
    with factory.begin() as db:
        with pytest.raises(DomainError, match="projection_changed_within_snapshot"):
            landing(db, a).ingest_batch(snapshot_id, BatchInput(rows=[changed, *rows[1:]], client_operation_id=uuid4()))
        assert (
            db.scalar(select(func.count()).select_from(m.SourceRow).where(m.SourceRow.snapshot_id == snapshot_id)) == 1
        )
    key = uuid4()
    second = ingest(factory, a, snapshot_id, rows, key)
    assert second == ingest(factory, a, snapshot_id, rows, key)
    with factory.begin() as db:
        assert (
            db.scalar(select(func.count()).select_from(m.SourceRow).where(m.SourceRow.snapshot_id == snapshot_id))
            == 100
        )
        event = db.scalar(
            select(core.Outbox).where(
                core.Outbox.entity_id == first["projection_ids"][0], core.Outbox.entity_type == "SourceProjection"
            )
        )
        from citequest.v2.services import digest

        assert digest(event.payload) == event.payload_sha256
        assert event.payload["data"]["title"] == rows[0].title
        assert not event.payload["profile_adopted"] and event.payload["operation"] == "stage"


def test_real_concurrent_duplicate_batches_converge(factory, actors):
    a, _, _ = actors
    snapshot_id, rows, barrier = register(factory, a), source_rows()[:2], Barrier(2)

    def run():
        barrier.wait(timeout=10)
        return ingest(factory, a, snapshot_id, rows)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run) for _ in range(2)]
        results = [future.result(timeout=30) for future in futures]
    assert results[0]["projection_ids"] == results[1]["projection_ids"]
    with factory.begin() as db:
        assert (
            db.scalar(select(func.count()).select_from(m.SourceRow).where(m.SourceRow.snapshot_id == snapshot_id)) == 2
        )


def test_same_common_source_identity_across_profiles_and_immutable_history(factory, actors):
    a, _, _ = actors
    first = register(factory, a)
    second = register(factory, a, profile="litsearch-native-annotation-decimal-offsets/1")
    # This is a cross-profile service control using the same true source value,
    # not a claim that the derived representation has been adopted or parsed.
    projections = [ingest(factory, a, snapshot, source_rows()[:1])["projection_ids"][0] for snapshot in (first, second)]
    with factory.begin() as db:
        results = [Resolver(landing(db, a)).resolve_metadata(UUID(projection), uuid4()) for projection in projections]
        assert results[0]["work_id"] == results[1]["work_id"]
        assert (
            db.scalar(
                select(func.count())
                .select_from(core.SourceIdentity)
                .where(
                    core.SourceIdentity.security_domain == a.security_domain, core.SourceIdentity.source == "litsearch"
                )
            )
            == 1
        )
        row_id = UUID(results[0]["source_row_id"])
        for target, values in (
            (m.SourceRow, {"native_id": "tampered"}),
            (m.SourceResolution, {"proof_sha256": "f" * 64}),
        ):
            with pytest.raises(DBAPIError, match="Immutable"), db.begin_nested():
                condition = target.id == row_id if target is m.SourceRow else target.source_row_id == row_id
                db.execute(update(target).where(condition).values(**values))


def test_shared_canonical_service_rejects_new_conflicting_strong_identifier(factory, actors):
    a, _, _ = actors
    with factory.begin() as db:
        resolver = CanonicalResolver(Catalog(db, a))
        one = resolver.resolve(
            "canonical_control",
            "native-one",
            WorkInput(
                title="Explicit synthetic identity one",
                authors=None,
                year=None,
                identifiers=[IdentifierInput(namespace="doi", raw="10.1234/s05-" + uuid4().hex)],
                client_operation_id=uuid4(),
            ),
            public=False,
        )
        doi_two = "10.1234/s05-" + uuid4().hex
        two = resolver.resolve(
            "canonical_control",
            "native-two",
            WorkInput(
                title="Explicit synthetic identity two",
                authors=None,
                year=None,
                identifiers=[IdentifierInput(namespace="doi", raw=doi_two)],
                client_operation_id=uuid4(),
            ),
            public=False,
        )
        assert one.id != two.id
        with pytest.raises(DomainError, match="identifier_conflict"), db.begin_nested():
            resolver.resolve(
                "canonical_control",
                "native-one",
                WorkInput(
                    title="Does not override native identity",
                    authors=None,
                    year=None,
                    identifiers=[IdentifierInput(namespace="doi", raw=doi_two)],
                    client_operation_id=uuid4(),
                ),
                public=False,
            )




def test_a_foreign_namespace_cannot_be_treated_as_litsearch_identity(factory, actors):
    a, _, _ = actors
    snapshot = register(factory, a, namespace="arxiv_oai_fixture")
    result = ingest(factory, a, snapshot, source_rows()[:1])
    with factory.begin() as db, pytest.raises(DomainError, match="not_a_metadata_projection"):
        Resolver(landing(db, a)).resolve_metadata(UUID(result["projection_ids"][0]), uuid4())
