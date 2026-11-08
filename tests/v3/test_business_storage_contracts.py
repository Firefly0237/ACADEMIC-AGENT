"""Synthetic local bytes/typed boundaries only; no PG, GCS or academic claim."""

from __future__ import annotations

import ast
import dataclasses
import io
import time
from pathlib import Path
from uuid import uuid4

import pytest

from citequest.business_storage_v3.contracts import OperationDeadline, StorageBinding, parse_reference, reference_value
from citequest.business_storage_v3.schema import DDL
from citequest.business_storage_v3.service import BusinessStorage
from citequest.platform_v3.blob_config import BlobStoreConfig
from citequest.platform_v3.blob_store import identity_for_bytes
from citequest.v2.domain import Actor, DomainError
from citequest.v2.storage import CAS, StoredBlob


def configuration(tmp_path: Path) -> BlobStoreConfig:
    return BlobStoreConfig(mode="test", backend="filesystem", store_id="explicit-local-control",
                           spool_dir=tmp_path / "temporary", filesystem_root=tmp_path / "original-cas")


class NoDatabase:
    def begin(self) -> None:
        raise AssertionError("validation_must_reject_before_database_admission")


def test_binding_separates_persistent_identity_from_ephemeral_spool(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    assert StorageBinding.configured(config) == StorageBinding.configured(config.model_copy(update={"spool_dir": tmp_path / "other-spool"}))
    assert StorageBinding.configured(config) != StorageBinding.configured(config.model_copy(update={"filesystem_root": tmp_path / "other-cas"}))


def test_real_gcs_and_emulator_can_never_share_binding(tmp_path: Path) -> None:
    real = BlobStoreConfig(mode="cloud", backend="gcs", store_id="same-logical-name", spool_dir=tmp_path,
                          gcp_project="future-project", gcs_bucket="future-bucket")
    emulator = real.model_copy(update={"mode": "test", "backend": "gcs_emulator", "emulator_endpoint": "http://localhost:14443"})
    assert StorageBinding.configured(real) != StorageBinding.configured(emulator)
    with pytest.raises(DomainError, match="business_cloud_io_supervision_not_integrated"):
        BusinessStorage(NoDatabase(), real)


@pytest.mark.parametrize("expires", [float("inf"), float("nan"), float("-inf"), 0.0])
def test_invalid_or_expired_deadlines_reject(expires: float) -> None:
    with pytest.raises(DomainError, match="business_blob_deadline_exceeded"):
        OperationDeadline(expires, clock=lambda: 1.0).remaining()


@pytest.mark.parametrize("bad", [b"", b"wrong bytes", b"same length wrong"])
def test_bytes_hash_and_size_are_checked_before_database(tmp_path: Path, bad: bytes) -> None:
    service = BusinessStorage(NoDatabase(), configuration(tmp_path))
    actor = Actor(uuid4())
    identity = identity_for_bytes(actor.security_domain, b"expected bytes")
    with pytest.raises(DomainError, match=r"business_blob_upload_(identity|size)_mismatch"):
        service.prepare_upload(actor, uuid4(), identity, io.BytesIO(bad), deadline=OperationDeadline(time.monotonic() + 10))
    assert list(service.config.spool_dir.iterdir()) == []


def test_domain_and_oversize_checks_precede_admission(tmp_path: Path) -> None:
    service = BusinessStorage(NoDatabase(), configuration(tmp_path).model_copy(update={"maximum_object_bytes": 2}))
    actor = Actor(uuid4())
    data = b"EXPLICIT SYNTHETIC BYTES"
    with pytest.raises(DomainError, match="not_found"):
        service.prepare_upload(actor, uuid4(), identity_for_bytes("owner:" + str(uuid4()), data), io.BytesIO(data),
                               deadline=OperationDeadline(time.monotonic() + 10))
    with pytest.raises(DomainError, match="business_blob_size_exceeds_profile"):
        service.prepare_upload(actor, uuid4(), identity_for_bytes(actor.security_domain, data), io.BytesIO(data),
                               deadline=OperationDeadline(time.monotonic() + 10))


def test_local_adapter_preserves_original_CAS_layout_and_receives_no_fake_path(tmp_path: Path) -> None:
    service = BusinessStorage(NoDatabase(), configuration(tmp_path))
    data = b"EXPLICIT LOCAL LAYOUT COMPATIBILITY"
    identity = identity_for_bytes("owner:" + str(uuid4()), data)
    reference = service.store.put(identity, io.BytesIO(data))
    legacy = CAS(service.config.filesystem_root)
    assert legacy.read(StoredBlob(identity.security_domain, identity.sha256, len(data))) == data
    assert parse_reference(reference_value(reference)) == reference
    with service.materialized(reference, deadline=OperationDeadline(time.monotonic() + 10)) as path:
        materialized = path
        assert path.read_bytes() == data and path != legacy.path(identity.security_domain, identity.sha256)
    assert not materialized.exists()
    with pytest.raises(DomainError, match="business_blob_reference_binding_mismatch"):
        service.binding.validate(dataclasses.replace(reference, store_id="forged"), identity)


def test_candidate_migration_literal_and_parent_are_self_contained() -> None:
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "migrations/app/versions/s15_blob_0001.py").read_text())
    assignments = {target.id: ast.literal_eval(item.value) for item in tree.body if isinstance(item, ast.Assign)
                   for target in item.targets if isinstance(target, ast.Name)}
    assert assignments["DDL"] == DDL
    assert assignments["revision"] == "s15_blob_0001" and assignments["down_revision"] == "s15_history_0001"
    assert not any(isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("citequest") for node in tree.body)
