"""CF-01 configuration and unchanged legacy defaults; no external DB/network."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from citequest.platform_v3.domain_runtime import create_runtime_app
from citequest.platform_v3.runtime_config import PoolBudget, RuntimeConfig, StorageDeclaration, load_runtime, owned_cf01_dsn
from citequest.v2.api import create_app
from citequest.v2.db import engine

ROOT = Path(__file__).resolve().parents[2]


def profile(tmp_path, **changes):
    values = {"mode": "test", "instance_id": "test-a", "database_dsn_file": tmp_path / "dsn",
        "cursor_key_file": tmp_path / "cursor", "temporary_directory": tmp_path / "tmp",
        "contract_path": ROOT / "contracts/v2/domain.schema.json",
        "storage": StorageDeclaration(shared_root=tmp_path / "shared-cas")}
    return RuntimeConfig(**(values | changes))


def test_cloud_and_unintegrated_capabilities_fail_before_io(tmp_path):
    with pytest.raises(ValidationError, match="cloud_domain_oidc"):
        profile(tmp_path, mode="cloud")
    for flag in ("file_content_io", "search", "agent", "ingestion"):
        with pytest.raises(ValidationError, match="not_integrated"):
            profile(tmp_path, **{flag: True})
    config = profile(tmp_path)
    with pytest.raises(ValidationError, match="cloud_domain_oidc"):
        create_runtime_app(config.model_copy(update={"mode": "cloud"}))


def test_connection_budget_and_numeric_boundaries_are_real():
    assert PoolBudget().planned_connections == 9
    with pytest.raises(ValidationError, match="budget_exceeded"):
        PoolBudget(pool_size=16, max_api_instances=2)
    for options in ({"pool_size": True}, {"max_overflow": -1}, {"pool_timeout": float("nan")}, {"pool_timeout": float("inf")}, {"connect_timeout": 100}, {"max_api_instances": 0}):
        with pytest.raises(ValidationError):
            PoolBudget(**options)


def test_owned_database_guard_refuses_shared_or_other_targets():
    good = "postgresql+psycopg://cq_cf01_app:synthetic@postgres:5432/citequest_cf01"
    assert owned_cf01_dsn(good, "app") == good
    for bad in (good.replace("5432", "25434"), good.replace("5432", "25435"), good.replace("citequest_cf01", "citequest_s03"), good.replace("postgres:5432", "example.com:5432"), good.replace("cq_cf01_app", "cq_cf01_admin")):
        with pytest.raises(ValueError, match="owned_cf01"):
            owned_cf01_dsn(bad, "app")


def test_config_port_override_and_redaction(tmp_path):
    config = profile(tmp_path)
    path = tmp_path / "runtime.json"
    path.write_text(config.model_dump_json(), encoding="utf-8")
    assert load_runtime(path, port="8091").port == 8091
    for port in ("true", "nan", "0", "999999", "8080.0"):
        with pytest.raises(ValueError):
            load_runtime(path, port=port)
    assert config.effective()["database_dsn_file"] == "<file-reference>"
    assert config.fingerprint == config.fingerprint and len(config.fingerprint) == 64


def test_legacy_loopback_origin_and_dependency_free_liveness_remain():
    with pytest.raises(ValueError, match="loopback"):
        create_app(None, cursor_key=b"x" * 32, contract_path=ROOT / "contracts/v2/domain.schema.json", origin="https://example.test")
    app = create_app(None, cursor_key=b"x" * 32, contract_path=ROOT / "contracts/v2/domain.schema.json")
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}


def test_engine_defaults_and_explicit_pool_parameters_stay_bounded():
    url = "postgresql+psycopg://synthetic:synthetic@127.0.0.1:25436/citequest_cf01"
    db = engine(url)
    try:
        assert db.pool.size() == 3 and db.pool._max_overflow == 0
        assert db.hide_parameters is True
    finally:
        db.dispose()
    configured = engine(url, pool_size=2, max_overflow=1, pool_timeout=2.0, connect_timeout=3)
    try:
        assert configured.pool.size() == 2 and configured.pool._max_overflow == 1
    finally:
        configured.dispose()
    for options in ({"pool_size": True}, {"pool_size": 0}, {"max_overflow": -1}, {"max_overflow": 100}, {"pool_timeout": float("nan")}, {"pool_timeout": float("inf")}, {"pool_timeout": True}, {"pool_timeout": 10**1000}, {"connect_timeout": 100}):
        with pytest.raises(ValueError):
            engine(url, **options)
