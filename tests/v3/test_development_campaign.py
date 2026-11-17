"""Pure release binding tests. Proofs, identities and DSNs below are synthetic.

No SQL server, Docker, model, network, dependency install or main ledger is used.
Real PostgreSQL locking/admission remains a separate minimum integration check.
"""
from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import PROFILES, AgentError, RuntimeLimits, digest
from citequest.agents_v3.development_campaign import (
    ALLOWED_PAIRS, BINDING_KEY, VERSION, DevelopmentCampaign, approval_body, load_campaign, require_runtime_task,
)
from citequest.agents_v3.product_profiles import NewTaskProfile, TaskProfilePolicy, public_profile_policy


def store(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def campaign_capsule(directory: Path) -> tuple[Path, str, dict]:
    directory.mkdir()
    owner, project = uuid4(), uuid4()
    limits = RuntimeLimits(mode="bounded", campaign_cash_ncny=500000000000,
        campaign_prior_reserved_ncny=4212736000, model_dispatch_concurrency=1)
    profiles = TaskProfilePolicy(release_id="EXPLICIT_SYNTHETIC_CAMPAIGN", profiles=[
        NewTaskProfile(kind=kind, profile_id=profile, validation_state="development")
        for kind, profile in sorted(ALLOWED_PAIRS)])
    old_profile = {**PROFILES["qa-v5"].model_dump(mode="json"),
        "aggregate_policy": limits.model_dump(mode="json"), "catalog_scope_policy": "task-catalog-snapshot/1"}
    cases = []
    for number in (1, 2):
        outcome = {"attempt_id": str(uuid4()), "task_id": str(uuid4()), "owner_id": str(owner),
            "run_id": str(uuid4()), "epoch": 1, "request_hash": "a" * 64,
            "profile_hash": digest(old_profile), "task_row_sha256": "b" * 64,
            "attempt_row_sha256": "c" * 64, "reserved_ncny": 2129920000,
            "response_sha256": None if number == 1 else "d" * 64}
        stop = {"container_id": str(number) * 64, "image_id": "sha256:" + str(number + 2) * 64,
            "source_manifest_sha256": "5" * 64, "original_launch_sha256": "6" * 64,
            "daemon_id": "EXPLICIT_SYNTHETIC_DAEMON", "state": {"Status": "exited", "Running": False,
                "Pid": 0, "Paused": False, "Restarting": False, "Dead": False,
                "StartedAt": "2026-11-06T09:00:00Z", "FinishedAt": "2026-11-06T09:00:01Z"},
            "restart_policy": {"Name": "no", "MaximumRetryCount": 0},
            "local_client_physical_stop_verified": True, "remote_inference_or_bill_completion_verified": False,
            "at": "2026-11-06T09:00:02Z"}
        ledger = {**outcome, "status": "unknown", "usage": None, "cost_ncny": None,
            "response_missing": number == 1, "global_unknown_models": 2,
            "task_state": "reconciling", "profile": old_profile}
        stop_path, ledger_path = directory / f"stop{number}.json", directory / f"ledger{number}.json"
        cases.append({"outcome": outcome, "stopped_client_proof": {"path": str(stop_path), "sha256": store(stop_path, stop)},
                      "ledger_proof": {"path": str(ledger_path), "sha256": store(ledger_path, ledger)}})
    identity = {"database_name": "citequest_s08_fixture_EXPLICIT_CAMPAIGN", "database_oid": 123456,
                "postgres_system_identifier": "1234567890"}
    identity_path, approval_path = directory / "identity.json", directory / "approval.json"
    value = {"version": VERSION, "scope": "isolated_control", "campaign_id": str(uuid4()),
        "owner_id": str(owner), "project_id": str(project), "database": {**identity,
            "readonly_identity_proof": {"path": str(identity_path), "sha256": store(identity_path, identity)}},
        "limits": limits.model_dump(mode="json"), "task_profile_policy": profiles.model_dump(mode="json"),
        "task_profile_policy_sha256": public_profile_policy(profiles)["policy_sha256"], "historical": cases,
        "root_approval": {"path": str(approval_path), "sha256": "0" * 64}}
    parsed = DevelopmentCampaign.model_validate_json(json.dumps(value), strict=True)
    value["root_approval"]["sha256"] = store(approval_path, approval_body(parsed))
    path = directory / "campaign.json"
    return path, store(path, value), value


def bound_task(policy: DevelopmentCampaign, *, kind="qa", profile_id="qa-v5") -> dict:
    from citequest.agents_v3.catalog_scope import graph_version

    profile = {**PROFILES[profile_id].model_dump(mode="json"), "aggregate_policy": policy.limits.model_dump(mode="json"),
        "catalog_scope_policy": policy.catalog_scope_policy, BINDING_KEY: policy.task_binding()}
    return {"id": uuid4(), "owner_id": policy.owner_id, "project_id": policy.project_id,
        "kind": kind, "profile": profile, "profile_hash": digest(profile),
        "graph_version": graph_version(kind, profile_id, policy.catalog_scope_policy)}


def test_loaded_release_admits_normal_scoped_queue_without_prebound_task_ids(tmp_path):
    path, checksum, _ = campaign_capsule(tmp_path / "valid")
    policy = load_campaign(path, checksum)
    policy.require_worker(None, "fixture")
    assert policy.task_binding()["policy_sha256"] == checksum
    assert policy.public_binding()["unknown_financial_record_changed"] is False
    assert "path" not in json.dumps(policy.public_binding())
    assert "target_task_id" not in policy.public_binding()
    for kind, profile in ALLOWED_PAIRS:
        policy.require_task(bound_task(policy, kind=kind, profile_id=profile))
    with pytest.raises(AgentError, match="worker_mode_mismatch"):
        policy.require_worker(None, "live")
    with pytest.raises(AgentError, match="historical_Task_readonly"):
        policy.require_worker(policy.historical_task_ids[0], "fixture")


def test_changed_campaign_or_artifact_cannot_inherit_approval(tmp_path):
    path, checksum, value = campaign_capsule(tmp_path / "signed")
    with pytest.raises(AgentError, match="artifact_hash_changed"):
        load_campaign(path, "f" * 64)
    changed = copy.deepcopy(value)
    changed["project_id"] = str(uuid4())
    with pytest.raises(AgentError, match="approval_binding_changed"):
        load_campaign(path, store(path, changed))
    store(path, value)
    Path(value["database"]["readonly_identity_proof"]["path"]).write_text("{}")
    with pytest.raises(AgentError, match="artifact_hash_changed"):
        load_campaign(path, checksum)


@pytest.mark.parametrize("change", [
    {"campaign_cash_ncny": 499000000000}, {"campaign_prior_reserved_ncny": 0},
    {"model_dispatch_concurrency": None}, {"owner_active_tasks": 1},
])
def test_release_cannot_omit_original_wallet_unknown_or_concurrency_contract(tmp_path, change):
    _path, _sha, value = campaign_capsule(tmp_path / "limits")
    value["limits"].update(change)
    with pytest.raises(ValidationError):
        DevelopmentCampaign.model_validate_json(json.dumps(value), strict=True)


def test_scoped_binding_is_not_a_default_worker_or_an_old_task_upgrade(tmp_path):
    path, checksum, _ = campaign_capsule(tmp_path / "task")
    policy = load_campaign(path, checksum)
    task = bound_task(policy)
    with pytest.raises(AgentError, match="bound_worker_required"):
        require_runtime_task(task, None)
    old = {**task, "profile": {k:v for k,v in task["profile"].items() if k != BINDING_KEY}}
    require_runtime_task(old, None)
    with pytest.raises(AgentError, match="Task_binding_required"):
        require_runtime_task(old, policy)
    with pytest.raises(AgentError, match="owner_project_not_enabled"):
        policy.require_task({**task, "owner_id": uuid4()})
    with pytest.raises(AgentError, match="Task_profile_changed"):
        policy.require_task({**task, "profile_hash": "f" * 64})


def test_claim_places_scope_in_SQL_before_limit_and_default_excludes_campaign(tmp_path):
    from citequest.agents_v3.database import claim

    path, checksum, _ = campaign_capsule(tmp_path / "queue")
    policy = load_campaign(path, checksum)

    class ReadOnlyQueryCapture:
        def __init__(self): self.queries = []
        def execute(self, query, args=()):
            self.queries.append((query, args))
            return SimpleNamespace(fetchall=lambda: [])

    for configured in (policy, None):
        connection = ReadOnlyQueryCapture()
        @contextmanager
        def connect(): yield connection
        database = SimpleNamespace(connect=connect, catalog_scope_policy="task-catalog-snapshot/1",
            development_dispatch_policy=None, development_campaign_policy=configured, dispatch_policy=None)
        with patch("citequest.agents_v3.database.catalog_scope.require_schema"), \
                patch("citequest.agents_v3.database.dispatch_control.active_policy", return_value=None), \
                patch.object(DevelopmentCampaign, "occupied", return_value=[]):
            with claim(database, "EXPLICIT_QUERY_ONLY", limits=policy.limits) as result:
                assert result is None
        query, params = next(row for row in connection.queries if row[0].startswith("SELECT c.*"))
        if configured is not None:
            assert query.index("t.owner_id=%s") < query.index("LIMIT 16")
            assert query.index("t.profile->'development_campaign'") < query.index("LIMIT 16")
            assert policy.owner_id in params and policy.project_id in params
        else:
            assert "NOT(t.profile ? 'development_campaign')" in query


def test_API_requires_frozen_trusted_release_and_explicit_profile_policy(tmp_path):
    from citequest.agents_v3.api_config import AgentAPIConfig

    path, checksum, _ = campaign_capsule(tmp_path / "composition")
    policy = load_campaign(path, checksum)
    files = {}
    for role in ("business", "checkpoint", "memory"):
        secret = tmp_path / (role + ".txt")
        secret.write_text("postgresql://EXPLICIT_SYNTHETIC_CONFIG@invalid.example/test")
        files[role + "_dsn_file"] = str(secret)
    common = {**files, "limits": policy.limits.model_dump(mode="json"),
        "catalog_scope_policy": policy.catalog_scope_policy,
        "development_campaign_file": str(path), "development_campaign_sha256": checksum}
    api = AgentAPIConfig.model_validate_json(json.dumps({**common,
        "task_profile_policy": policy.task_profile_policy.model_dump(mode="json")}), strict=True)
    assert api.database().development_campaign_policy.source_sha256 == checksum
    with pytest.raises(ValidationError, match="explicit task profile policy"):
        AgentAPIConfig.model_validate_json(json.dumps(common), strict=True)
    with pytest.raises(ValidationError, match="both trusted file"):
        AgentAPIConfig.model_validate_json(json.dumps({**common, "development_campaign_sha256": None}), strict=True)


def test_worker_requires_same_campaign_release_without_accepting_named_task_mode(tmp_path):
    pytest.importorskip("openai", reason="Worker composition requires the existing locked provider runtime, never a stand-in SDK")
    from citequest.agents_v3.worker import WorkerConfig

    path, checksum, _ = campaign_capsule(tmp_path / "worker")
    policy = load_campaign(path, checksum)
    fields = {"limits": policy.limits.model_dump(mode="json"), "catalog_scope_policy": policy.catalog_scope_policy,
        "development_campaign_file": str(path), "development_campaign_sha256": checksum,
        "mode": "fixture", "tool_mode": "search", "skill_root": str(tmp_path), "temporary_directory": str(tmp_path)}
    for role in ("business", "checkpoint", "memory"):
        secret = tmp_path / (role + ".txt")
        secret.write_text("postgresql://EXPLICIT_SYNTHETIC_CONFIG@invalid.example/test")
        fields[role + "_dsn_file"] = str(secret)
    worker = WorkerConfig.model_validate_json(json.dumps(fields), strict=True)
    assert worker.database().development_campaign_policy.source_sha256 == checksum
    with pytest.raises(ValidationError, match="mutually exclusive"):
        WorkerConfig.model_validate_json(json.dumps({**fields,
            "development_dispatch_policy_file": str(path), "development_dispatch_policy_sha256": checksum}), strict=True)
