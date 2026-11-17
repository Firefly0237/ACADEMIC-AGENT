"""Real audited public corpus chains; no synthetic paper is counted in these checks."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError

from citequest.data_v2 import models as m
from citequest.data_v2.landing import Landing
from citequest.data_v2.native_index import NativeRepresentationPointer, read_for_index
from citequest.data_v2.native_registration import NativePaperRegistrar
from citequest.data_v2.resolution import Resolver
from citequest.v2 import models as core
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor, DomainError, LibrarySave
from citequest.v2.services import Catalog, digest
from citequest.v2.storage import CAS
from citequest.v2.storage_config import app_storage

ROOT = Path(__file__).resolve().parents[3]
STANDARD = UUID("6f6e49a4-9e17-5328-9eee-ea55f4dfde82")
NATIVE = UUID("e7cabf2f-bf73-5cb9-8846-c1afb3b3ed35")


@pytest.fixture(scope="module")
def factory():
    import sys

    sys.path.insert(0, str(ROOT / "scripts/s03"))
    from bootstrap import dsn

    db = engine(dsn("cq_s03_app"))
    yield sessions(db)
    db.dispose()


def landing(db, actor):
    import hashlib

    return Landing(
        Catalog(db, actor),
        policy_sha256=hashlib.sha256((ROOT / "contracts/v2/policy.json").read_bytes()).hexdigest(),
        evaluation_sha256=hashlib.sha256((ROOT / "contracts/v2/evaluation.json").read_bytes()).hexdigest(),
    )


def sample(db):
    # Rank zero was committed at the real three-paper canary. Other records
    # need not have completed yet for these identity/permission controls.
    row = db.scalar(select(m.SourceRow).where(m.SourceRow.snapshot_id == NATIVE, m.SourceRow.native_id == "253761972"))
    assert row is not None
    representation = db.scalar(select(m.NativeRepresentation).where(m.NativeRepresentation.source_row_id == row.id))
    assert representation is not None
    return row, representation


def test_source_reference_links_real_version_rights_and_cas_without_qa_upgrade(factory):
    with factory.begin() as db:
        snapshot = db.get(m.SourceSnapshot, NATIVE)
        actor = Actor(snapshot.importer_owner_id, publisher=True)
        row, representation = sample(db)
        source = read_for_index(landing(db, actor), app_storage(ROOT).cas(), representation.id)
        resolution = Resolver(landing(db, actor)).latest(row.id)
        version = db.get(core.WorkVersion, source.work_version_id)
        work = db.get(core.Work, source.work_id)
        original = db.get(core.SourceRecord, UUID(resolution.proof["source_record_id"]))
        assert version.work_id == work.id == original.work_id
        assert version.source_sha256 == original.raw_sha256 == resolution.proof["raw_PDF_sha256"]
        assert original.rights_status == "verified" and original.license == "CC-BY-4.0"
        assert work.authors and work.year is not None
        assert source.source_reference.work_version_id == version.id
        assert source.source_reference.representation_id == representation.id
        assert source.source_pointer_sha256 == row.source_pointer_sha256
        assert source.coordinate_system == "unicode_codepoint"
        assert not source.original_PDF_text_coverage_claimed
        assert source.index_staging_is_not_QA_permission and not source.profile_adopted
        assert snapshot.status == "landing" and snapshot.seal is None
        common = db.get(core.SourceIdentity, ("public", "litsearch", row.native_id))
        assert common.work_id == work.id
        event = db.scalar(
            select(core.Outbox).where(
                core.Outbox.entity_type == "NativeRepresentation", core.Outbox.entity_id == representation.id
            )
        )
        assert event.payload_sha256 == digest(event.payload)
        pointer = NativeRepresentationPointer.model_validate_json(json.dumps(event.payload["data"]))
        assert pointer.draft_sha256 == representation.draft_sha256
        assert pointer.source_reference == source.source_reference


def test_real_source_identity_enables_private_library_without_fabricated_file(factory):
    a, b = Actor(uuid4()), Actor(uuid4())
    with factory.begin() as db:
        db.add_all(
            [
                core.Owner(id=a.owner_id, name="S05 real-paper library control A"),
                core.Owner(id=b.owner_id, name="S05 real-paper library control B"),
            ]
        )
    with factory.begin() as db:
        _, representation = sample(db)
        source = read_for_index(landing(db, a), app_storage(ROOT).cas(), representation.id)
        saved = Catalog(db, a).save(
            LibrarySave(work_id=source.work_id, collection_id=None, client_operation_id=uuid4())
        )
        assert saved["file_ids"] == []
        assert Catalog(db, a).item(UUID(saved["library_item_id"])).work_id == source.work_id
        with pytest.raises(DomainError, match="not_found"):
            Catalog(db, b).item(UUID(saved["library_item_id"]))


def test_canonical_mapping_preserves_frozen_clean_source_and_does_not_claim_body(factory):
    with factory.begin() as db:
        snapshot = db.get(m.SourceSnapshot, STANDARD)
        actor = Actor(snapshot.importer_owner_id, publisher=True)
        native_row, native_representation = sample(db)
        expected_work = Resolver(landing(db, actor)).latest(native_row.id).work_id
        standard_row = db.scalar(
            select(m.SourceRow).where(
                m.SourceRow.snapshot_id == STANDARD, m.SourceRow.native_id == native_row.native_id
            )
        )
        projection = db.scalar(select(m.SourceProjection).where(m.SourceProjection.source_row_id == standard_row.id))
        before = (projection.title, projection.abstract, projection.field_provenance, projection.projection_sha256)
        result = Resolver(landing(db, actor)).resolve_metadata(projection.id, uuid4())
        if projection.title is not None and projection.title.strip() and len(projection.title) <= 500:
            assert result["state"] == "metadata_resolved" and UUID(result["work_id"]) == expected_work
            assert result["work_version_id"] is None and not result["proof"]["fulltext_identity_verified"]
            assert landing(db, actor).index_projection(projection.id).work_id == expected_work
        else:
            assert result["state"] == "unresolved" and result["work_id"] is None
        assert (
            projection.title,
            projection.abstract,
            projection.field_provenance,
            projection.projection_sha256,
        ) == before
        assert native_representation.source_row_id != standard_row.id


def test_db_rejects_cross_work_version_even_with_a_real_native_representation(factory):
    with factory.begin() as db:
        row, representation = sample(db)
        current = db.scalar(
            select(m.SourceResolution)
            .where(m.SourceResolution.source_row_id == row.id)
            .order_by(m.SourceResolution.revision.desc())
            .limit(1)
        )
        other = db.scalar(
            select(m.SourceResolution).where(
                m.SourceResolution.state == "paper_identity_verified", m.SourceResolution.work_id != current.work_id
            )
        )
        assert other is not None
        with pytest.raises(DBAPIError, match="WorkVersion"), db.begin_nested():
            proof = {"fixture": "adversarial other-work version, never accepted as source truth"}
            db.add(
                m.SourceResolution(
                    source_row_id=row.id,
                    security_domain="public",
                    revision=current.revision + 1,
                    state="paper_identity_verified",
                    work_id=current.work_id,
                    work_version_id=other.work_version_id,
                    representation_id=representation.id,
                    proof=proof,
                    proof_sha256=digest(proof),
                )
            )
            db.flush()


def test_real_native_reader_and_writer_reject_retained_wrong_root(factory):
    with factory.begin() as db:
        snapshot = db.get(m.SourceSnapshot, NATIVE)
        actor = Actor(snapshot.importer_owner_id, publisher=True)
        _, representation = sample(db)
        # Retained incorrect cache still has the actual bytes. Passing this
        # test proves the root guard, not merely a missing-file error.
        wrong = CAS(ROOT / "data/runtime-v2-app/cas")
        with pytest.raises(ValueError, match="application_storage_root_mismatch"):
            read_for_index(landing(db, actor), wrong, representation.id)
        with pytest.raises(ValueError, match="application_storage_root_mismatch"):
            NativePaperRegistrar(landing(db, actor), wrong, ROOT)


def test_public_resolution_append_requires_current_publisher_before_idempotent_replay(factory):
    with factory.begin() as db:
        snapshot = db.get(m.SourceSnapshot, NATIVE)
        row, representation = sample(db)
        resolver = Resolver(landing(db, Actor(snapshot.importer_owner_id, publisher=False)))
        current = resolver.latest(row.id)
        with pytest.raises(DomainError, match="publisher_required"):
            resolver.append(row, current.state, current.proof, work_id=current.work_id,
                            version_id=current.work_version_id, representation_id=representation.id)


@pytest.mark.parametrize("mutation", ["coordinated_work_version", "foreign_source_record", "proof_hash", "source_reference"])
def test_native_reader_rechecks_identity_chain_instead_of_trusting_verified_label(factory, monkeypatch, mutation):
    from citequest.data_v2 import native_index

    with factory.begin() as db:
        snapshot = db.get(m.SourceSnapshot, NATIVE)
        row, representation = sample(db)
        resolver = Resolver(landing(db, Actor(snapshot.importer_owner_id, publisher=True)))
        current = resolver.latest(row.id)
        other = db.scalar(select(m.SourceResolution).where(
            m.SourceResolution.state == "paper_identity_verified", m.SourceResolution.work_id != current.work_id))
        values = {name: getattr(current, name) for name in (
            "state", "work_id", "work_version_id", "representation_id", "proof", "proof_sha256")}
        values["proof"] = copy.deepcopy(current.proof)
        if mutation == "coordinated_work_version":
            values.update(work_id=other.work_id, work_version_id=other.work_version_id)
        elif mutation == "foreign_source_record":
            values["proof"]["source_record_id"] = other.proof["source_record_id"]
        elif mutation == "proof_hash":
            values["proof_sha256"] = "0" * 64
        else:
            values["proof"]["source_reference"]["work_version_id"] = str(other.work_version_id)
        if mutation != "proof_hash":
            values["proof_sha256"] = digest(values["proof"])
        monkeypatch.setattr(native_index.Resolver, "latest", lambda self, _: SimpleNamespace(**values))
        with pytest.raises(DomainError, match=r"identity|proof_hash|trusted_chain"):
            read_for_index(resolver.landing, app_storage(ROOT).cas(), representation.id)


def test_public_append_rejects_coordinated_work_and_version_replacement(factory):
    with factory.begin() as db:
        snapshot = db.get(m.SourceSnapshot, NATIVE)
        row, representation = sample(db)
        resolver = Resolver(landing(db, Actor(snapshot.importer_owner_id, publisher=True)))
        current = resolver.latest(row.id)
        other = db.scalar(select(m.SourceResolution).where(
            m.SourceResolution.state == "paper_identity_verified", m.SourceResolution.work_id != current.work_id))
        with pytest.raises(DomainError, match="common_source_identity_conflict"):
            resolver.append(row, current.state, current.proof, work_id=other.work_id,
                            version_id=other.work_version_id, representation_id=representation.id)


@pytest.mark.parametrize("mutation", ["coordinated_work_version", "missing_proof", "foreign_source_record", "source_reference"])
def test_database_rejects_coordinated_or_unclosed_source_chain(factory, mutation):
    with factory.begin() as db:
        row, representation = sample(db)
        current = db.scalar(select(m.SourceResolution).where(m.SourceResolution.source_row_id == row.id)
                            .order_by(m.SourceResolution.revision.desc()).limit(1))
        other = db.scalar(select(m.SourceResolution).where(
            m.SourceResolution.state == "paper_identity_verified", m.SourceResolution.work_id != current.work_id))
        proof = copy.deepcopy(current.proof)
        work_id, version_id = current.work_id, current.work_version_id
        if mutation == "coordinated_work_version":
            work_id, version_id = other.work_id, other.work_version_id
        elif mutation == "missing_proof":
            proof = {}
        elif mutation == "foreign_source_record":
            proof["source_record_id"] = other.proof["source_record_id"]
        else:
            proof["source_reference"]["work_version_id"] = str(other.work_version_id)
        with pytest.raises(DBAPIError, match="canonical source identity or trusted chain"), db.begin_nested():
            db.add(m.SourceResolution(source_row_id=row.id, security_domain=row.security_domain,
                revision=current.revision + 1, state="paper_identity_verified", work_id=work_id,
                work_version_id=version_id, representation_id=representation.id, proof=proof, proof_sha256=digest(proof)))
            db.flush()


def test_database_accepts_coherent_real_chain_without_retaining_test_revision(factory):
    with factory.begin() as db:
        row, representation = sample(db)
        current = db.scalar(select(m.SourceResolution).where(m.SourceResolution.source_row_id == row.id)
                            .order_by(m.SourceResolution.revision.desc()).limit(1))
        nested = db.begin_nested()
        db.add(m.SourceResolution(source_row_id=row.id, security_domain=row.security_domain,
            revision=current.revision + 1, state=current.state, work_id=current.work_id,
            work_version_id=current.work_version_id, representation_id=representation.id,
            proof=copy.deepcopy(current.proof), proof_sha256=current.proof_sha256))
        db.flush()
        nested.rollback()
        assert db.scalar(select(m.SourceResolution.revision).where(m.SourceResolution.source_row_id == row.id)
                         .order_by(m.SourceResolution.revision.desc()).limit(1)) == current.revision
