"""Pure API configuration/constructor checks. All secret paths are fixtures."""
import json

import pytest
from pydantic import ValidationError

from citequest.agents_v3.api_config import AgentAPIConfig
from citequest.agents_v3.catalog_scope import POLICY
from citequest.agents_v3.dispatch import DispatchPolicy


def config(tmp_path):
    value = {"limits": {"mode": "bounded"}}
    for role in ("business", "checkpoint", "memory"):
        path = tmp_path / (role + ".txt")
        path.write_text("postgresql://EXPLICIT_CONFIGURATION_FIXTURE@invalid.example/fixture")
        value[role + "_dsn_file"] = str(path)
    return value


def test_legacy_config_constructs_without_adding_optional_runtime_policies(tmp_path):
    parsed = AgentAPIConfig.model_validate_json(json.dumps(config(tmp_path)), strict=True)
    database = parsed.database()
    assert database.catalog_scope_policy is None and database.dispatch_policy is None


def test_snapshot_and_fifo_configuration_are_actually_forwarded(tmp_path):
    value = config(tmp_path)
    path = tmp_path / "policy.json"
    policy = DispatchPolicy()
    path.write_text(policy.model_dump_json())
    value.update(catalog_scope_policy=POLICY, dispatch_policy_file=str(path))
    database = AgentAPIConfig.model_validate_json(json.dumps(value), strict=True).database()
    assert database.catalog_scope_policy == POLICY and database.dispatch_policy == policy
    assert database.worker_lane is None


def test_API_rejects_model_credentials_or_unknown_scope_instead_of_silently_ignoring_them(tmp_path):
    for extra in ({"api_key_file": "EXPLICIT_FORBIDDEN_FIXTURE"}, {"catalog_scope_policy": "invented-policy"}):
        with pytest.raises(ValidationError):
            AgentAPIConfig.model_validate_json(json.dumps({**config(tmp_path), **extra}), strict=True)
