"""Exactly two stopped historical clients and one named QA development Task.

This is an explicitly adopted experiment, not product quarantine or a growing
allowlist. Only activity projection changes: every old financial row stays in
the ordinary complete SUM, and every new unknown stops all new HTTP entry.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, PrivateAttr, model_validator

from .contracts import AgentError, RuntimeLimits, Strict, digest
from .development_dispatch import (SEMANTICS, SHA, Artifact, HistoricalUnknown, NewTask,
                                   validate_stopped_client, validate_target)

VERSION = "citequest-development-named-unknown-dispatch/2"
ACTIVE = "exact_target_create_and_two_preapproved_followups_exclude_exact_two_stopped_historical_Tasks"


class HistoricalOutcome(HistoricalUnknown):
    response_sha256: SHA | None


class HistoricalCase(Strict):
    outcome: HistoricalOutcome
    stopped_client_proof: Artifact
    ledger_proof: Artifact


class DevelopmentDispatchPolicyV2(Strict):
    version: Literal[VERSION] = VERSION
    scope: Literal["isolated_control", "development_main"]
    database_name: str = Field(min_length=1, max_length=100)
    semantics: Literal[SEMANTICS] = SEMANTICS
    active_projection: Literal[ACTIVE] = ACTIVE
    campaign_cash_ncny: Literal[500000000000] = 500000000000
    campaign_prior_reserved_ncny: Literal[4212736000] = 4212736000
    known_client_concurrency: Literal[1] = 1
    historical: tuple[HistoricalCase, HistoricalCase]
    target: NewTask
    root_approval: Artifact
    _file_sha256: str | None = PrivateAttr(default=None)
    _proofs: dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def exact_independent_scope(self):
        cases = [item.outcome for item in self.historical]
        if (len({item.task_id for item in cases}) != 2 or len({item.attempt_id for item in cases}) != 2
                or self.target.task_id in {item.task_id for item in cases}
                or any(item.owner_id != self.target.owner_id for item in cases)
                or len(self.target.followup_operation_ids) != 2):
            raise ValueError("two distinct old Tasks, the same real owner, one new Task and exactly two named followups required")
        if ((self.scope == "development_main" and self.database_name != "citequest_s03")
                or (self.scope == "isolated_control" and not self.database_name.startswith("citequest_s08_fixture"))):
            raise ValueError("explicit matching Main/fixture database required")
        return self

    @property
    def source_sha256(self) -> str:
        if self._file_sha256 is None:
            raise AgentError("development_policy_must_be_loaded_from_trusted_file")
        return self._file_sha256

    def require_task(self, task_id: UUID | None) -> None:
        _ = self.source_sha256
        if task_id is None or task_id != self.target.task_id:
            raise AgentError("development_dispatch_requires_explicit_approved_Task")

    def require_worker(self, task_id: UUID | None, mode: str) -> None:
        self.require_task(task_id)
        if mode != ("live" if self.scope == "development_main" else "fixture"):
            raise AgentError("development_dispatch_worker_mode_mismatch")

    def public_binding(self) -> dict:
        return {"version": self.version, "policy_sha256": self.source_sha256,
            "approval_sha256": self.root_approval.sha256, "semantics": SEMANTICS, "active_projection": ACTIVE,
            "historical_attempt_ids": [str(item.outcome.attempt_id) for item in self.historical],
            "historical_Task_ids": [str(item.outcome.task_id) for item in self.historical],
            "target_task_id": str(self.target.task_id), "known_client_concurrency": 1,
            "owner_active_tasks_limit": 2, "remote_physical_concurrency_asserted": False,
            "unknown_financial_record_changed": False}


def approval_body(policy: DevelopmentDispatchPolicyV2) -> dict:
    return {"version": "citequest-development-dispatch-approval/2",
        "decision": "approve_one_named_development_verification", "scope": policy.scope,
        "historical_binding_sha256": digest([item.outcome.model_dump(mode="json") for item in policy.historical]),
        "historical_proof_refs_sha256": digest([{key: item.model_dump(mode="json")[key]
            for key in ("stopped_client_proof", "ledger_proof")} for item in policy.historical]),
        "target_binding_sha256": digest(policy.target.model_dump(mode="json")), "active_projection": ACTIVE,
        "campaign_cash_ncny": 500000000000, "campaign_prior_reserved_ncny": 4212736000,
        "owner_active_tasks_limit": 2, "semantics": SEMANTICS,
        "old_Task_resume_authorized": False, "old_unknown_financial_change_authorized": False,
        "Main_dispatch_authorized": policy.scope == "development_main",
        "approved_by": "Root" if policy.scope == "development_main" else "EXPLICIT_ISOLATED_CONTROL"}


def load_v2(value: dict, expected_sha256: str) -> DevelopmentDispatchPolicyV2:
    policy = DevelopmentDispatchPolicyV2.model_validate_json(json.dumps(value), strict=True)
    validate_historical_proofs(policy.historical)
    approval = policy.root_approval.read()
    if approval != approval_body(policy):
        raise AgentError("development_policy_Root_approval_binding_changed")
    policy._file_sha256 = expected_sha256
    policy._proofs = {"approval": approval}
    return policy


def validate_historical_proofs(historical: tuple[HistoricalCase, HistoricalCase]) -> None:
    """Shared exact-two immutable proof contract; no execution or accounting."""
    daemons, containers = set(), set()
    for case in historical:
        stop, ledger = case.stopped_client_proof.read(), case.ledger_proof.read()
        validate_stopped_client(stop)
        daemons.add(stop["daemon_id"])
        containers.add(stop["container_id"])
        outcome = case.outcome.model_dump(mode="json")
        if any(ledger.get(key) != child for key, child in outcome.items()):
            raise AgentError("development_policy_ledger_binding_changed")
        if (ledger.get("status") != "unknown" or ledger.get("usage") is not None or ledger.get("cost_ncny") is not None
                or ledger.get("global_unknown_models") != 2 or ledger.get("task_state") != "reconciling"
                or ledger.get("response_missing") is not (case.outcome.response_sha256 is None)
                or digest(ledger.get("profile")) != case.outcome.profile_hash):
            raise AgentError("development_policy_unknown_facts_changed")
    if len(daemons) != 1:
        raise AgentError("development_policy_stopped_clients_daemon_mismatch")
    if len(containers) != 2:
        raise AgentError("development_policy_distinct_stopped_clients_required")


def require_limits(limits: RuntimeLimits, policy: DevelopmentDispatchPolicyV2) -> None:
    if (limits.mode != "bounded" or limits.model_dispatch_concurrency != 1 or limits.owner_active_tasks != 2
            or limits.campaign_cash_ncny != policy.campaign_cash_ncny
            or limits.campaign_prior_reserved_ncny != policy.campaign_prior_reserved_ncny):
        raise AgentError("development_dispatch_runtime_policy_mismatch")


def require_identity(task: dict, policy: DevelopmentDispatchPolicyV2) -> None:
    target = policy.target
    expected = {"id": target.task_id, "owner_id": target.owner_id, "project_id": target.project_id,
        "operation_id": target.operation_id, "profile_hash": target.profile_hash,
        "create_hash": target.initial_payload_sha256, "graph_version": target.graph_version, "kind": "qa"}
    if any(task.get(key) != child for key, child in expected.items()):
        raise AgentError("development_dispatch_new_Task_binding_changed")
    profile = task["profile"]
    if (digest(profile) != target.profile_hash or profile.get("profile_id") != "qa-v5"
            or profile.get("active_seconds") != 120 or profile.get("catalog_scope_policy") != "task-catalog-snapshot/1"):
        raise AgentError("development_dispatch_profile_or_wallet_changed")
    require_limits(RuntimeLimits.model_validate(profile.get("aggregate_policy"), strict=True), policy)


def historical_occupied(db: Any, policy: DevelopmentDispatchPolicyV2) -> list[dict]:
    _ = policy.source_sha256
    return historical_occupied_for_scope(db, database_name=policy.database_name,
        project_id=policy.target.project_id, historical=policy.historical, proofs_loaded=bool(policy._proofs))


def historical_occupied_for_scope(db: Any, *, database_name: str, project_id: UUID,
                                historical: tuple[HistoricalCase, ...], proofs_loaded: bool) -> list[dict]:
    """Caller owns the ordinary wallet lock; every financial row stays untouched."""
    if not proofs_loaded:
        raise AgentError("development_policy_proofs_not_loaded")
    if db.execute("SELECT current_database() AS name").fetchone()["name"] != database_name:
        raise AgentError("development_dispatch_database_mismatch")
    approved = {case.outcome.attempt_id for case in historical}
    for case in sorted(historical, key=lambda value: str(value.outcome.task_id)):
        item = case.outcome
        task = db.execute("SELECT t.*,row_to_json(t)::text AS _row FROM cq_app.agent_tasks t WHERE id=%s FOR SHARE", (item.task_id,)).fetchone()
        attempt = db.execute("SELECT a.*,row_to_json(a)::text AS _row FROM cq_app.agent_attempts a WHERE id=%s FOR SHARE", (item.attempt_id,)).fetchone()
        if (not task or not attempt or hashlib.sha256(task["_row"].encode()).hexdigest() != item.task_row_sha256
                or hashlib.sha256(attempt["_row"].encode()).hexdigest() != item.attempt_row_sha256):
            raise AgentError("development_dispatch_historical_row_changed")
        response_hash = digest(attempt["response"]) if attempt["response"] is not None else None
        if (attempt["task_id"] != item.task_id or attempt["kind"] != "model" or attempt["status"] != "unknown"
                or attempt["usage"] is not None or attempt["cost_ncny"] is not None or response_hash != item.response_sha256
                or attempt["reserved_ncny"] != item.reserved_ncny or attempt["run_id"] != item.run_id or attempt["epoch"] != item.epoch
                or attempt["request_hash"] != item.request_hash or digest(attempt["request"]) != item.request_hash
                or task["owner_id"] != item.owner_id or task["project_id"] != project_id
                or task["profile_hash"] != item.profile_hash
                or digest(task["profile"]) != item.profile_hash or task["state"] != "reconciling"):
            raise AgentError("development_dispatch_historical_facts_changed")
        profile = task["profile"]
        reserve = profile["context_window"]*profile["input_miss_ncny"]+attempt["output_limit"]*profile["output_ncny"]
        if (reserve != item.reserved_ncny or attempt["request"].get("model") != profile["model_alias"]
                or attempt["request"].get("max_output_tokens") != attempt["output_limit"]):
            raise AgentError("development_dispatch_historical_price_reserve_changed")
    occupied = db.execute("SELECT id,task_id,request_hash,status FROM cq_app.agent_attempts WHERE kind='model' AND status IN ('reserved','sent','unknown')").fetchall()
    if {row["id"] for row in occupied if row["status"] == "unknown"} != approved:
        raise AgentError("development_dispatch_unmatched_unknown_STOP")
    return [row for row in occupied if row["id"] not in approved]


def project_active(db: Any, limits: RuntimeLimits, owner: UUID, project: UUID, task: dict,
                   operation_id: UUID, command_kind: str, policy: DevelopmentDispatchPolicyV2) -> int:
    """Only called by create/followup under the original wallet lock order."""
    require_limits(limits, policy)
    require_identity(task, policy)
    if (owner, project) != (policy.target.owner_id, policy.target.project_id):
        raise AgentError("development_dispatch_new_Task_binding_changed")
    if command_kind == "start":
        if operation_id != policy.target.operation_id:
            raise AgentError("development_dispatch_command_not_approved")
    elif (command_kind != "followup" or task.get("command_seq") not in (1, 2)
            or operation_id != policy.target.followup_operation_ids[task["command_seq"]-1]):
        raise AgentError("development_dispatch_command_not_approved")
    if historical_occupied(db, policy):
        raise AgentError("global_model_dispatch_busy", 429)
    return db.execute("""SELECT count(*) AS active FROM cq_app.agent_tasks WHERE owner_id=%s
        AND state IN ('queued','running','reconciling') AND NOT(id=ANY(%s::uuid[]))""",
        (owner, [item.outcome.task_id for item in policy.historical])).fetchone()["active"]


def validate_occupied_v2(db: Any, task: dict, request: dict, logical_key: str, command: dict,
                         policy: DevelopmentDispatchPolicyV2, *, current_attempt_id: UUID | None = None) -> list[dict]:
    validate_target(db, task, command, logical_key, policy)
    occupied = historical_occupied(db, policy)
    if request.get("model") != task["profile"]["model_alias"]:
        raise AgentError("development_dispatch_new_model_changed")
    if current_attempt_id is not None:
        current = [row for row in occupied if row["id"] == current_attempt_id]
        if (len(current) != 1 or current[0]["task_id"] != task["id"] or current[0]["status"] != "reserved"
                or current[0]["request_hash"] != digest(request)):
            raise AgentError("development_dispatch_current_reservation_changed")
    return [row for row in occupied if row["id"] != current_attempt_id]
