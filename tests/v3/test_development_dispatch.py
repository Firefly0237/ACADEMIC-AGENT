"""Trusted policy loading/worker gates; all approvals/proofs are fixtures."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx2
import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import PROFILES, AgentError, RuntimeLimits, digest
from citequest.agents_v3.development_dispatch import SEMANTICS, VERSION, load_policy
from citequest.agents_v3.worker import WorkerConfig, run_once


def store(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capsule(directory: Path, *, change_stop: dict | None = None) -> tuple[Path, str, dict]:
    directory.mkdir()
    profile = {**PROFILES["qa-v5"].model_dump(mode="json"), "aggregate_policy": RuntimeLimits(mode="bounded",
        campaign_cash_ncny=500000000000, campaign_prior_reserved_ncny=4212736000, model_dispatch_concurrency=1).model_dump(mode="json"),
        "catalog_scope_policy": "task-catalog-snapshot/1"}
    historical = {"attempt_id": str(uuid4()), "task_id": str(uuid4()), "owner_id": str(uuid4()), "run_id": str(uuid4()), "epoch": 1,
        "request_hash": "a" * 64, "profile_hash": digest(profile), "task_row_sha256": "b" * 64,
        "attempt_row_sha256": "c" * 64, "reserved_ncny": 2129920000}
    target = {"task_id": str(uuid4()), "owner_id": str(uuid4()), "project_id": str(uuid4()), "operation_id": str(uuid4()),
        "profile_hash": digest(profile), "initial_payload_sha256": "d" * 64, "graph_version": "EXPLICIT_TEST_GRAPH", "followup_operation_ids": []}
    stop = {"container_id": "1" * 64, "image_id": "sha256:" + "2" * 64, "source_manifest_sha256": "3" * 64,
        "original_launch_sha256": "4" * 64, "daemon_id": "EXPLICIT_TEST_DAEMON", "state": {"Status": "exited", "Running": False,
        "Pid": 0, "Paused": False, "Restarting": False, "Dead": False, "StartedAt": "2026-11-06T09:00:00Z", "FinishedAt": "2026-11-06T09:00:01Z"},
        "restart_policy": {"Name": "no", "MaximumRetryCount": 0}, "local_client_physical_stop_verified": True,
        "remote_inference_or_bill_completion_verified": False, "at": "2026-11-06T09:00:02Z"}
    if change_stop:
        stop.update(change_stop)
    ledger = {**historical, "status": "unknown", "usage": None, "cost_ncny": None, "response_missing": True,
        "global_unknown_models": 1, "task_state": "reconciling", "profile": profile}
    stop_path, ledger_path, approval_path = (directory / name for name in ("stop.json", "ledger.json", "approval.json"))
    stop_sha, ledger_sha = store(stop_path, stop), store(ledger_path, ledger)
    approval = {"version": "citequest-development-dispatch-approval/1", "decision": "approve_one_named_development_verification",
        "scope": "isolated_control", "historical_binding_sha256": digest(historical), "target_binding_sha256": digest(target),
        "stopped_client_proof_sha256": stop_sha, "ledger_proof_sha256": ledger_sha,
        "campaign_cash_ncny": 500000000000, "campaign_prior_reserved_ncny": 4212736000, "semantics": SEMANTICS,
        "old_Task_resume_authorized": False, "old_unknown_financial_change_authorized": False, "Main_dispatch_authorized": False,
        "approved_by": "EXPLICIT_ISOLATED_CONTROL"}
    approval_sha = store(approval_path, approval)
    value = {"version": VERSION, "scope": "isolated_control", "database_name": "citequest_s08_fixture_EXPLICIT_UNIT", "semantics": SEMANTICS,
        "historical": historical, "target": target,
        "stopped_client_proof": {"path": str(stop_path), "sha256": stop_sha}, "ledger_proof": {"path": str(ledger_path), "sha256": ledger_sha},
        "root_approval": {"path": str(approval_path), "sha256": approval_sha}}
    path = directory / "policy.json"
    return path, store(path, value), value


def test_trusted_policy_is_exact_and_default_worker_reference_is_none(tmp_path: Path) -> None:
    path, checksum, _value = capsule(tmp_path / "valid")
    policy = load_policy(path, checksum)
    policy.require_worker(policy.target.task_id, "fixture")
    assert policy.source_sha256 == checksum
    assert policy.public_binding()["unknown_financial_record_changed"] is False
    assert "path" not in json.dumps(policy.public_binding())
    with pytest.raises(AgentError, match="explicit_approved_Task"):
        policy.require_task(None)
    with pytest.raises(AgentError, match="worker_mode_mismatch"):
        policy.require_worker(policy.target.task_id, "live")


def test_policy_and_each_artifact_tamper_are_rejected_before_database_IO(tmp_path: Path) -> None:
    path, checksum, value = capsule(tmp_path / "tamper")
    with pytest.raises(AgentError, match="artifact_hash_changed"):
        load_policy(path, "f" * 64)
    Path(value["ledger_proof"]["path"]).write_text("{}")
    with pytest.raises(AgentError, match="artifact_hash_changed"):
        load_policy(path, checksum)


@pytest.mark.parametrize("change", [{"local_client_physical_stop_verified": False},
    {"remote_inference_or_bill_completion_verified": True}, {"restart_policy": {"Name": "always", "MaximumRetryCount": 0}},
    {"state": {"Status": "exited", "Running": True, "Pid": 123, "Restarting": False, "Paused": False, "Dead": False}}])
def test_even_signed_fixture_cannot_claim_unsafe_physical_or_remote_stop(tmp_path: Path, change: dict) -> None:
    path, checksum, _value = capsule(tmp_path / "unsafe", change_stop=change)
    with pytest.raises(AgentError, match="physical_stop_proof_invalid"):
        load_policy(path, checksum)


def test_worker_policy_file_requires_exact_hash_pair(tmp_path: Path) -> None:
    fields = {"limits": RuntimeLimits(mode="bounded", campaign_cash_ncny=500000000000, campaign_prior_reserved_ncny=4212736000, model_dispatch_concurrency=1), "business_dsn_file": tmp_path / "business", "checkpoint_dsn_file": tmp_path / "checkpoint",
        "memory_dsn_file": tmp_path / "memory", "skill_root": tmp_path, "temporary_directory": tmp_path, "mode": "fixture", "tool_mode": "search"}
    config = WorkerConfig(**fields)
    assert config.development_dispatch_policy_file is None and config.development_dispatch_policy_sha256 is None
    with pytest.raises(ValidationError, match="both trusted file and exact SHA"):
        WorkerConfig(**fields, development_dispatch_policy_file=tmp_path / "policy.json")


def test_worker_rejects_ordinary_queue_or_wrong_Task_before_memory_flush_and_claim(tmp_path: Path) -> None:
    path, checksum, _value = capsule(tmp_path / "worker")
    policy = load_policy(path, checksum)
    config = WorkerConfig(limits=RuntimeLimits(mode="bounded", campaign_cash_ncny=500000000000, campaign_prior_reserved_ncny=4212736000, model_dispatch_concurrency=1), business_dsn_file=tmp_path / "UNOPENED_business", checkpoint_dsn_file=tmp_path / "UNOPENED_checkpoint",
        memory_dsn_file=tmp_path / "UNOPENED_memory", skill_root=tmp_path, temporary_directory=tmp_path, mode="fixture", tool_mode="search",
        development_dispatch_policy_file=path, development_dispatch_policy_sha256=checksum)
    for target in (None, uuid4()):
        with patch.object(WorkerConfig, "database", return_value=SimpleNamespace(development_dispatch_policy=policy)), \
                patch("citequest.agents_v3.worker.MemoryService") as memory, patch("citequest.agents_v3.worker.claim") as claim:
            with pytest.raises(AgentError, match="explicit_approved_Task"):
                asyncio.run(run_once(config, transport=httpx2.MockTransport(lambda _request: None), task_id=target))
            memory.assert_not_called()
            claim.assert_not_called()


def test_duplicate_JSON_and_modified_target_cannot_inherit_Root_approval(tmp_path: Path) -> None:
    path, _checksum, value = capsule(tmp_path / "approval")
    altered = copy.deepcopy(value)
    altered["target"]["task_id"] = str(uuid4())
    checksum = store(path, altered)
    with pytest.raises(AgentError, match="approval_binding_changed"):
        load_policy(path, checksum)
    raw = path.read_bytes()
    raw = b'{"scope":"isolated_control",' + raw[1:]
    path.write_bytes(raw)
    with pytest.raises(AgentError, match="duplicate_JSON_property"):
        load_policy(path, hashlib.sha256(raw).hexdigest())
