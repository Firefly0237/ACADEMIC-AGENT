"""Portable storage behavior; synthetic bytes, no product DB or cloud account."""
from __future__ import annotations

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage
from pydantic import ValidationError

from citequest.platform_v3.blob_config import BlobStoreConfig, create_blob_store
from citequest.platform_v3.blob_store import (
    BlobStoreError,
    BlobWriteUncertain,
    GCSBlobStore,
    LocalBlobStore,
    identity_for_bytes,
    put_bytes,
    read_bytes,
)


def local(tmp_path: Path) -> LocalBlobStore:
    return LocalBlobStore(tmp_path / "cas", store_id="fixture-local", spool_dir=tmp_path / "tmp")


def test_local_shared_identity_deduplicates_and_security_domains_stay_separate(tmp_path):
    store = local(tmp_path)
    data = b"EXPLICIT_SYNTHETIC_BYTES"
    first = put_bytes(store, "public", data)
    again = put_bytes(store, "public", data)
    private = put_bytes(store, "owner:" + str(uuid4()), data)
    assert first == again and private.object_key != first.object_key
    assert read_bytes(store, first) == read_bytes(store, private) == data
    with pytest.raises(BlobStoreError, match="store_reference"):
        read_bytes(store, replace(first, store_id="another-store"))


def test_bad_upload_never_publishes_an_object_and_spool_is_released(tmp_path):
    store = local(tmp_path)
    identity = identity_for_bytes("public", b"EXPECTED")
    with pytest.raises(BlobStoreError):
        store.put(identity, io.BytesIO(b"CORRUPTED"))
    assert not store.cas.path(identity.security_domain, identity.sha256).exists()
    assert list((tmp_path / "tmp").iterdir()) == []


def test_delivery_uses_already_verified_bytes_even_if_original_path_changes(tmp_path):
    store = local(tmp_path)
    reference = put_bytes(store, "public", b"SYNTHETIC_VERIFIED_CONTENT")
    with store.open_verified(reference) as verified:
        store.cas.path("public", reference.identity.sha256).write_bytes(b"OUT_OF_BAND_MUTATION")
        assert verified.read() == b"SYNTHETIC_VERIFIED_CONTENT"
    with pytest.raises(BlobStoreError):
        read_bytes(store, reference)


def test_store_profile_rejects_resource_excess_before_io(tmp_path):
    store = LocalBlobStore(tmp_path / "cas", store_id="small", spool_dir=tmp_path / "tmp", maximum_object_bytes=3)
    with pytest.raises(BlobStoreError, match="exceeds_store_profile"):
        put_bytes(store, "public", b"FOUR")


def emulator_config(tmp_path, bucket):
    return BlobStoreConfig(mode="test", backend="gcs_emulator", store_id="fixture-gcs",
        spool_dir=tmp_path / "spool", gcp_project="citequest-local", gcs_bucket=bucket,
        emulator_endpoint=os.environ.get("CITEQUEST_TEST_GCS_ENDPOINT", "http://127.0.0.1:14443"))


@pytest.fixture(scope="module")
def emulator():
    from google.auth.credentials import AnonymousCredentials

    endpoint = os.environ.get("CITEQUEST_TEST_GCS_ENDPOINT", "http://127.0.0.1:14443")
    # Validate the test endpoint before constructing any HTTP client. Tests
    # cannot accidentally create a real cloud bucket through an environment.
    emulator_config(Path("/tmp") if os.name != "nt" else Path.cwd(), "cq-endpoint-validation")
    client = storage.Client(project="citequest-local", credentials=AnonymousCredentials(),
                            client_options={"api_endpoint": endpoint})
    assert endpoint == client._connection.API_BASE_URL
    bucket = "cq-local-" + uuid4().hex[:16]
    client.create_bucket(bucket, timeout=(3, 10), retry=None)
    yield client, bucket
    client.close()


def test_sdk_conditional_create_and_verified_reconcile_keep_one_generation(tmp_path, emulator):
    client, bucket = emulator
    store = GCSBlobStore(client, bucket, store_id="fixture-gcs", spool_dir=tmp_path)
    data = b"SYNTHETIC CONDITIONAL CREATE " + uuid4().bytes
    reference = put_bytes(store, "public", data)
    assert put_bytes(store, "public", data) == reference
    assert store.reconcile(reference.identity) == reference
    assert read_bytes(store, reference) == data
    assert reference.generation > 0


def test_independent_sdk_clients_racing_create_share_the_verified_generation(tmp_path, emulator):
    from google.auth.credentials import AnonymousCredentials

    _, bucket = emulator
    endpoint = os.environ.get("CITEQUEST_TEST_GCS_ENDPOINT", "http://127.0.0.1:14443")
    data = b"SYNTHETIC CONCURRENT CLIENTS " + uuid4().bytes

    def create(index):
        client = storage.Client(project="citequest-local", credentials=AnonymousCredentials(), client_options={"api_endpoint": endpoint})
        try:
            store = GCSBlobStore(client, bucket, store_id="fixture-race", spool_dir=tmp_path / str(index))
            return put_bytes(store, "public", data)
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(create, range(4)))
    assert len(set(results)) == 1


def test_sdk_post_commit_ack_loss_is_uncertain_then_reconciled_without_rewrite(tmp_path, emulator, monkeypatch):
    client, bucket = emulator
    store = GCSBlobStore(client, bucket, store_id="fixture-gcs", spool_dir=tmp_path)
    data = b"SYNTHETIC POST-WRITE ACK LOSS " + uuid4().bytes
    identity = identity_for_bytes("public", data)
    original = storage.Blob.upload_from_file

    def lose_ack(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise TimeoutError("injected_after_emulator_committed")

    monkeypatch.setattr(storage.Blob, "upload_from_file", lose_ack)
    with pytest.raises(BlobWriteUncertain):
        store.put(identity, io.BytesIO(data))
    reference = store.reconcile(identity)
    assert reference is not None and read_bytes(store, reference) == data


def test_sdk_wrong_generation_bytes_are_not_delivered(tmp_path, emulator):
    client, bucket = emulator
    store = GCSBlobStore(client, bucket, store_id="fixture-gcs", spool_dir=tmp_path)
    data = b"SYNTHETIC TRUE CONTENT " + uuid4().bytes
    reference = put_bytes(store, "public", data)
    corrupt = client.bucket(bucket).blob(reference.object_key)
    corrupt.upload_from_string(b"bad", if_generation_match=reference.generation, checksum="crc32c", retry=None, timeout=(3, 10))
    with pytest.raises(BlobStoreError):
        read_bytes(store, replace(reference, generation=corrupt.generation))
    with pytest.raises(BlobStoreError):
        store.put(reference.identity, io.BytesIO(data))


def test_real_sdk_resumable_upload_and_disk_spool_roundtrip(tmp_path, emulator):
    client, bucket = emulator
    store = GCSBlobStore(client, bucket, store_id="fixture-gcs", spool_dir=tmp_path)
    data = (b"SYNTHETIC RESUMABLE BYTES\n" * 400000) + uuid4().bytes
    assert len(data) > 8 * 1024**2
    reference = put_bytes(store, "owner:" + str(uuid4()), data)
    with store.open_verified(reference) as result:
        assert result.read() == data
    assert list(tmp_path.iterdir()) == []


def test_emulator_factory_is_anonymous_and_cloud_cannot_choose_it(tmp_path, emulator):
    _, bucket = emulator
    config = emulator_config(tmp_path, bucket)
    store = create_blob_store(config)
    assert read_bytes(store, put_bytes(store, "public", b"factory synthetic")) == b"factory synthetic"
    with pytest.raises(ValidationError):
        create_blob_store(config.model_copy(update={"mode": "cloud"}))


def test_cloud_never_falls_back_to_local_storage_or_anonymous_credentials(tmp_path, monkeypatch):
    with pytest.raises(ValidationError):
        BlobStoreConfig(mode="cloud", backend="filesystem", store_id="wrong", spool_dir=tmp_path, filesystem_root=tmp_path)
    config = BlobStoreConfig(mode="cloud", backend="gcs", store_id="real", spool_dir=tmp_path,
                             gcp_project="configured-future-project", gcs_bucket="configured-future-bucket")
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", "http://127.0.0.1:14443")
    with pytest.raises(ValueError, match="ambient_emulator"):
        create_blob_store(config)
    monkeypatch.delenv("STORAGE_EMULATOR_HOST")
    monkeypatch.delenv("FIREBASE_STORAGE_EMULATOR_HOST", raising=False)
    monkeypatch.setattr(storage, "Client", lambda **kwargs: (_ for _ in ()).throw(DefaultCredentialsError("injected_no_ADC")))
    with pytest.raises(DefaultCredentialsError):
        create_blob_store(config)


def test_cloud_refuses_sdk_ambient_endpoint_override_before_client_io(tmp_path, monkeypatch):
    config = BlobStoreConfig(mode="cloud", backend="gcs", store_id="real", spool_dir=tmp_path,
                             gcp_project="future-project", gcs_bucket="future-bucket")
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    monkeypatch.delenv("FIREBASE_STORAGE_EMULATOR_HOST", raising=False)
    monkeypatch.setenv("API_ENDPOINT_OVERRIDE", "http://127.0.0.1:14443")
    monkeypatch.setattr(storage, "Client", lambda **kwargs: pytest.fail("must reject before client or ADC"))
    with pytest.raises(ValueError, match="ambient_emulator"):
        create_blob_store(config)


def test_real_backend_uses_explicit_google_https_endpoint_and_adc(tmp_path, monkeypatch):
    import google.auth
    from google.auth.credentials import AnonymousCredentials

    for name in ("STORAGE_EMULATOR_HOST", "FIREBASE_STORAGE_EMULATOR_HOST", "API_ENDPOINT_OVERRIDE"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    # Constructor control only. This stub is NOT a cloud authentication test.
    def adc(**kwargs):
        calls.append(kwargs)
        return AnonymousCredentials(), "future-project"
    monkeypatch.setattr(google.auth, "default", adc)
    config = BlobStoreConfig(mode="cloud", backend="gcs", store_id="real", spool_dir=tmp_path,
                             gcp_project="future-project", gcs_bucket="future-bucket")
    store = create_blob_store(config)
    assert calls and store.client.api_endpoint == "https://storage.googleapis.com"
    store.client.close()


@pytest.mark.parametrize("stage", ["metadata", "readback", "conflict_reconcile"])
def test_post_write_verification_failure_remains_uncertain(tmp_path, emulator, monkeypatch, stage):
    client, bucket = emulator
    store = GCSBlobStore(client, bucket, store_id="fixture-gcs", spool_dir=tmp_path)
    data = b"SYNTHETIC VERIFICATION UNCERTAINTY " + uuid4().bytes
    identity = identity_for_bytes("public", data)
    if stage == "conflict_reconcile":
        original = store.put(identity, io.BytesIO(data))
        target = "reconcile"
    else:
        original = None
        target = "_describe" if stage == "metadata" else "open_verified"
    with monkeypatch.context() as patch:
        patch.setattr(store, target, lambda *a, **k: (_ for _ in ()).throw(TimeoutError("synthetic_verification_timeout")))
        with pytest.raises(BlobWriteUncertain):
            store.put(identity, io.BytesIO(data))
    recovered = store.reconcile(identity)
    assert recovered is not None and read_bytes(store, recovered) == data
    if original:
        assert recovered == original


def test_resumable_server_crc32c_and_fixed_generation_verification_without_delete(tmp_path, emulator, monkeypatch):
    import google_crc32c

    client, bucket = emulator
    store = GCSBlobStore(client, bucket, store_id="fixture-gcs", spool_dir=tmp_path)
    data = b"SYNTHETIC RESUMABLE INTEGRITY\n" * 350000 + uuid4().bytes
    assert len(data) > 8 * 1024**2
    expected = base64.b64encode(google_crc32c.Checksum(data).digest()).decode()
    request = client._http.request
    seen = []

    def record(method, url, **kwargs):
        assert method.upper() != "DELETE", "SDK cleanup must never delete an object"
        if method.upper() == "POST" and "uploadType=resumable" in url:
            metadata = json.loads(kwargs["data"])
            assert metadata["crc32c"] == expected and "ifGenerationMatch=0" in url
            seen.append("server_crc32c_and_create_precondition")
        if method.upper() == "GET" and "generation=" in url:
            seen.append("fixed_generation_read")
        return request(method, url, **kwargs)

    monkeypatch.setattr(client._http, "request", record)
    reference = put_bytes(store, "public", data)
    assert read_bytes(store, reference) == data
    assert "server_crc32c_and_create_precondition" in seen and "fixed_generation_read" in seen


def test_resumable_bad_server_crc_metadata_is_uncertain_without_sdk_cleanup(tmp_path, emulator, monkeypatch):
    client, bucket = emulator
    store = GCSBlobStore(client, bucket, store_id="fixture-gcs", spool_dir=tmp_path)
    data = b"SYNTHETIC RETURNED CRC CORRUPTION\n" * 300000 + uuid4().bytes
    identity = identity_for_bytes("public", data)
    request = client._http.request
    changed = []

    def corrupt_response(method, url, **kwargs):
        assert method.upper() != "DELETE", "SDK cleanup must never delete an object"
        response = request(method, url, **kwargs)
        if method.upper() == "PUT" and response.status_code == 200 and "upload_id=" in url:
            metadata = response.json()
            metadata["crc32c"] = "AAAAAA=="
            response._content = json.dumps(metadata).encode()
            changed.append(True)
        return response

    with monkeypatch.context() as patch:
        patch.setattr(client._http, "request", corrupt_response)
        with pytest.raises(BlobWriteUncertain):
            store.put(identity, io.BytesIO(data))
    assert changed
    reference = store.reconcile(identity)
    assert reference and read_bytes(store, reference) == data
