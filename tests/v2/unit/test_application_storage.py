"""Product-root guards are independent of equal content bytes or temporary unit CAS."""

import io

import pytest
from pydantic import ValidationError

from citequest.v2.storage import CAS, StoredBlob
from citequest.v2.storage_config import ApplicationStorage, app_storage


def test_product_factory_has_one_fixed_root_and_cannot_be_overridden(tmp_path):
    config = app_storage(tmp_path)
    assert config.blob_root == tmp_path / "data/runtime-v2-app/blobs"
    assert config.cas().root == config.blob_root
    with pytest.raises(ValidationError):
        ApplicationStorage(workspace=tmp_path, blob_root=tmp_path / "wrong")


def test_wrong_root_is_rejected_even_when_it_has_identical_verified_bytes(tmp_path):
    config = app_storage(tmp_path)
    canonical, wrong = config.cas(), CAS(tmp_path / "wrong")
    original = canonical.put("public", io.BytesIO(b"explicit synthetic bytes"), 100)
    duplicate = wrong.put("public", io.BytesIO(b"explicit synthetic bytes"), 100)
    assert original == duplicate and canonical.read(original) == wrong.read(duplicate)
    with pytest.raises(ValueError, match="application_storage_root_mismatch"):
        config.assert_store(wrong)
    config.assert_store(canonical)


def test_readonly_inventory_path_does_not_create_missing_domain_or_root(tmp_path):
    config = app_storage(tmp_path)
    path = config.readonly_path(StoredBlob("owner:synthetic", "a" * 64, 1))
    assert not path.parent.exists() and not config.blob_root.exists()
