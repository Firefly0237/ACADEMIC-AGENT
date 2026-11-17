"""Filesystem-only integrity/admission tests; no PostgreSQL claim."""

import io

import pytest

from citequest.v2.storage import CAS


def test_cas_domains_are_separate_and_same_domain_deduplicates(tmp_path):
    store = CAS(tmp_path)
    a = store.put("owner:a", io.BytesIO(b"paper"), 5)
    same = store.put("owner:a", io.BytesIO(b"paper"), 5)
    b = store.put("owner:b", io.BytesIO(b"paper"), 5)
    assert a == same
    assert store.path(a.security_domain, a.sha256) != store.path(b.security_domain, b.sha256)
    assert store.read(a) == b"paper"


@pytest.mark.parametrize("body,limit", [(b"", 100), (b"1234", 3)])
def test_rejected_upload_leaves_no_staging_file(tmp_path, body, limit):
    with pytest.raises(ValueError):
        CAS(tmp_path).put("owner:a", io.BytesIO(body), limit)
    assert not list(tmp_path.glob(".staging-*"))


def test_corrupt_existing_blob_cannot_be_reused(tmp_path):
    store = CAS(tmp_path)
    blob = store.put("owner:a", io.BytesIO(b"paper"), 20)
    store.path(blob.security_domain, blob.sha256).write_bytes(b"wrong")
    with pytest.raises(ValueError, match="corrupt"):
        store.put("owner:a", io.BytesIO(b"paper"), 20)


def test_blob_identity_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):
        CAS(tmp_path).path("owner:a", "../../private")


def test_delivery_never_uses_bytes_changed_after_a_separate_verification(tmp_path, monkeypatch):
    import hashlib

    store = CAS(tmp_path)
    blob = store.put("owner:a", io.BytesIO(b"paper"), 5)
    original = store.verify

    def verification_race(value):
        original(value)
        store.path(value.security_domain, value.sha256).write_bytes(b"wrong")

    monkeypatch.setattr(store, "verify", verification_race)
    delivered = store.read(blob)
    assert hashlib.sha256(delivered).hexdigest() == blob.sha256
