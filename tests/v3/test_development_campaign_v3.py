"""Explicit synthetic proof/SQL doubles. No Docker, database or model calls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3 import development_campaign as campaign_module
from citequest.agents_v3.contracts import PROFILES, AgentError, digest
from citequest.agents_v3.development_campaign import (
    ALLOWED_PAIRS_V2, VERSION_V2, VERSION_V3, DevelopmentCampaign, approval_body, load_campaign,
)
from citequest.agents_v3.product_profiles import NewTaskProfile, TaskProfilePolicy, public_profile_policy
from test_development_campaign import campaign_capsule, store


def _sign(path, value):
    policy = DevelopmentCampaign.model_validate_json(json.dumps(value), strict=True)
    value["root_approval"]["sha256"] = store(Path(value["root_approval"]["path"]), approval_body(policy))
    return store(path, value)


def _v2(directory):
    path, _checksum, value = campaign_capsule(directory)
    profiles = TaskProfilePolicy(release_id="EXPLICIT_SYNTHETIC_V2", profiles=[
        NewTaskProfile(kind=kind, profile_id=profile, validation_state="development",
            review_transport_protocol="function1" if kind in {"qa", "stance"} else None)
        for kind, profile in sorted(ALLOWED_PAIRS_V2)])
    value.update(version=VERSION_V2, task_profile_policy=profiles.model_dump(mode="json"),
                 task_profile_policy_sha256=public_profile_policy(profiles)["policy_sha256"])
    return path, _sign(path, value), value


def _append_case(path, value):
    value["version"] = VERSION_V3
    cases = value.setdefault("quarantined_cases", [])
    number = len(cases) + 3
    profile = {**PROFILES["qa-v6"].model_dump(mode="json"), "aggregate_policy": value["limits"],
               "catalog_scope_policy": value.get("catalog_scope_policy",
                   DevelopmentCampaign.model_fields["catalog_scope_policy"].default)}
    outcome = {"attempt_id": str(uuid4()), "task_id": str(uuid4()), "owner_id": value["owner_id"],
        "run_id": str(uuid4()), "epoch": 1, "request_hash": "a" * 64, "profile_hash": digest(profile),
        "task_row_sha256": "b" * 64, "attempt_row_sha256": "c" * 64,
        "reserved_ncny": 2129920000, "response_sha256": "d" * 64}
    stop = {"container_id": str(number) * 64, "image_id": "sha256:" + "4" * 64,
        "source_manifest_sha256": "5" * 64, "original_launch_sha256": "6" * 64,
        "daemon_id": "EXPLICIT_SYNTHETIC_DAEMON", "state": {"Status": "exited", "Running": False,
            "Pid": 0, "Paused": False, "Restarting": False, "Dead": False,
            "StartedAt": "2026-11-07T10:00:00Z", "FinishedAt": "2026-11-07T10:00:30Z"},
        "restart_policy": {"Name": "no", "MaximumRetryCount": 0},
        "local_client_physical_stop_verified": True, "remote_inference_or_bill_completion_verified": False,
        "at": "2026-11-07T10:00:32Z", "worker_id": "EXPLICIT_SYNTHETIC_WORKER",
        **{key: outcome[key] for key in ("task_id", "attempt_id", "run_id")}}
    prefix = [case["outcome"]["attempt_id"] for case in value["historical"] + cases] + [outcome["attempt_id"]]
    ledger = {**outcome, "status": "unknown", "usage": None, "cost_ncny": None,
        "response_missing": False, "global_unknown_models": len(prefix), "task_state": "reconciling",
        "profile": profile, "approved_unknown_attempt_ids": prefix, "project_id": value["project_id"],
        "worker_id": stop["worker_id"], "attempt_sent_at": "2026-11-07T10:00:10Z",
        "observed_at": "2026-11-07T10:00:32Z"}
    stop_path, ledger_path = path.parent / f"quarantine-stop-{number}.json", path.parent / f"quarantine-ledger-{number}.json"
    cases.append({"outcome": outcome,
        "stopped_client_proof": {"path": str(stop_path), "sha256": store(stop_path, stop)},
        "ledger_proof": {"path": str(ledger_path), "sha256": store(ledger_path, ledger)}})
    return stop, ledger


def test_old_serialization_and_proofs_stay_exact_while_v3_requires_new_approval(tmp_path):
    v1_path, v1_sha, _ = campaign_capsule(tmp_path / "v1")
    assert "quarantined_cases" not in load_campaign(v1_path, v1_sha).model_dump(mode="json")
    path, checksum, value = _v2(tmp_path / "v2")
    old = load_campaign(path, checksum)
    old_dump = old.model_dump(mode="json")
    assert "quarantined_cases" not in old_dump
    assert approval_body(old) == json.loads(Path(value["root_approval"]["path"]).read_text())
    old_bytes = {case[key]["path"]: Path(case[key]["path"]).read_bytes()
                 for case in value["historical"] for key in ("ledger_proof", "stopped_client_proof")}
    _append_case(path, value)
    with pytest.raises(AgentError, match="approval_binding_changed"):
        load_campaign(path, store(path, value))
    policy = load_campaign(path, _sign(path, value))
    assert len(policy.historical_task_ids) == 3 and policy.public_binding()["historical_outcomes"] == 3
    assert policy.limits.model_dump(mode="json") == value["limits"] == old_dump["limits"]
    assert all(Path(name).read_bytes() == raw for name, raw in old_bytes.items())
    value["version"] = VERSION_V2
    with pytest.raises(ValidationError, match="version3"):
        DevelopmentCampaign.model_validate_json(json.dumps(value), strict=True)


def test_new_capture_requires_exact_prefix_nulls_scope_client_and_stop_lifetime(tmp_path):
    path, _checksum, value = _v2(tmp_path / "capture")
    stop, ledger = _append_case(path, value)
    case = value["quarantined_cases"][0]
    changes = (("ledger_proof", {"approved_unknown_attempt_ids": ledger["approved_unknown_attempt_ids"][:-1]}),
        ("ledger_proof", {"global_unknown_models": 2}), ("ledger_proof", {"cost_ncny": 0}),
        ("ledger_proof", {"project_id": str(uuid4())}),
        ("ledger_proof", {"attempt_sent_at": "2026-11-07T09:00:00Z"}),
        ("stopped_client_proof", {"run_id": str(uuid4())}),
        ("stopped_client_proof", {"restart_policy": {"Name": "always", "MaximumRetryCount": 0}}))
    for field, change in changes:
        for name, original in (("stopped_client_proof", stop), ("ledger_proof", ledger)):
            case[name]["sha256"] = store(Path(case[name]["path"]), original)
        original = stop if field == "stopped_client_proof" else ledger
        case[field]["sha256"] = store(Path(case[field]["path"]), {**original, **change})
        with pytest.raises(AgentError):
            load_campaign(path, _sign(path, value))


def test_each_addition_is_ordered_distinct_and_separately_bound_by_root(tmp_path):
    path, _checksum, value = _v2(tmp_path / "ordered")
    _append_case(path, value)
    _sign(path, value)
    _append_case(path, value)
    with pytest.raises(AgentError, match="approval_binding_changed"):
        load_campaign(path, store(path, value))
    assert len(load_campaign(path, _sign(path, value)).historical_task_ids) == 4
    value["quarantined_cases"].reverse()
    with pytest.raises(AgentError, match="quarantine_ledger_changed"):
        load_campaign(path, _sign(path, value))
    value["quarantined_cases"].reverse()
    value["quarantined_cases"].append(copy.deepcopy(value["quarantined_cases"][0]))
    with pytest.raises(ValidationError, match="distinct"):
        DevelopmentCampaign.model_validate_json(json.dumps(value), strict=True)


def test_scoped_runtime_delegates_all_exact_cases_and_preserves_wallet_projection(tmp_path, monkeypatch):
    path, _checksum, value = _v2(tmp_path / "scope")
    _append_case(path, value)
    policy = load_campaign(path, _sign(path, value))
    seen = []

    def exact_occupied(_db, **scope):
        # Explicit dependency double for the unchanged row-hash/unknown-set
        # validator. This control checks new v3 delegation, not real SQL locks.
        seen.append(scope)
        assert len(scope["historical"]) == 3 and scope["proofs_loaded"] is True
        return []

    class Queries:
        def __init__(self):
            self.reads = []

        def execute(self, query, parameters=()):
            assert query.lstrip().startswith("SELECT")
            self.reads.append((query, parameters))
            row = ({"name": policy.database.database_name, "oid": policy.database.database_oid}
                   if "current_database()" in query else {"active": 1})
            return SimpleNamespace(fetchone=lambda: row)

    monkeypatch.setattr(campaign_module, "historical_occupied_for_scope", exact_occupied)
    db = Queries()
    assert policy.active_tasks(db, policy.limits, policy.owner_id, policy.project_id) == 1
    assert db.reads[-1][1][1] == policy.historical_task_ids
    for task_id in policy.historical_task_ids:
        with pytest.raises(AgentError, match="historical_Task_readonly"):
            policy.require_worker(task_id, "fixture")
    assert seen[0]["project_id"] == policy.project_id
    assert [case.outcome.task_id for case in seen[0]["historical"]] == policy.historical_task_ids
    clause, parameters = policy.queue_predicate()
    assert "profile->'development_campaign'" in clause
    assert json.loads(parameters[2])["version"] == VERSION_V3
    assert policy.limits.campaign_cash_ncny == 500000000000
    assert policy.limits.campaign_prior_reserved_ncny == 4212736000
