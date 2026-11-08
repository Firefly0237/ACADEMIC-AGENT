"""One named development-only exception to the durable unknown dispatch hold.

No accounting mutation, retry, queue, clock lease, Docker access or network IO.
The caller must hold the existing campaign/Task wallet locks. Product/default
configuration remains None; all unmatched unknown outcomes still stop dispatch.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import Field, PrivateAttr, model_validator

from .contracts import AgentError, Strict, digest

VERSION = "citequest-development-named-unknown-dispatch/1"
SEMANTICS = "known_active_client_HTTP_1_unknown_remote_concurrency_not_asserted"
SHA = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class Artifact(Strict):
    path: str = Field(min_length=1, max_length=2048)
    sha256: SHA

    def read(self) -> dict:
        path = Path(self.path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise AgentError("development_policy_trusted_artifact_required")
        raw = path.read_bytes()
        if len(raw) > 131072 or hashlib.sha256(raw).hexdigest() != self.sha256:
            raise AgentError("development_policy_artifact_hash_changed")
        value = strict_json(raw)
        if not isinstance(value, dict):
            raise AgentError("development_policy_artifact_object_required")
        return value


class HistoricalUnknown(Strict):
    attempt_id: UUID
    task_id: UUID
    owner_id: UUID
    run_id: UUID
    epoch: int = Field(ge=1)
    request_hash: SHA
    profile_hash: SHA
    task_row_sha256: SHA
    attempt_row_sha256: SHA
    reserved_ncny: int = Field(gt=0)


class NewTask(Strict):
    task_id: UUID
    owner_id: UUID
    project_id: UUID
    operation_id: UUID
    profile_hash: SHA
    initial_payload_sha256: SHA
    graph_version: str = Field(min_length=1, max_length=200)
    followup_operation_ids: tuple[UUID, ...] = Field(default=(), max_length=2)

    @model_validator(mode="after")
    def unique_operations(self) -> NewTask:
        operations = (self.operation_id, *self.followup_operation_ids)
        if len(set(operations)) != len(operations):
            raise ValueError("development operations must be unique")
        return self


class DevelopmentDispatchPolicy(Strict):
    version: Literal[VERSION] = VERSION
    scope: Literal["isolated_control", "development_main"]
    database_name: str = Field(min_length=1, max_length=100)
    semantics: Literal[SEMANTICS] = SEMANTICS
    campaign_cash_ncny: Literal[500000000000] = 500000000000
    campaign_prior_reserved_ncny: Literal[4212736000] = 4212736000
    known_client_concurrency: Literal[1] = 1
    historical: HistoricalUnknown
    target: NewTask
    stopped_client_proof: Artifact
    ledger_proof: Artifact
    root_approval: Artifact
    _file_sha256: str | None = PrivateAttr(default=None)
    _proofs: dict[str, dict] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def independent_new_task(self) -> DevelopmentDispatchPolicy:
        if self.target.task_id == self.historical.task_id:
            raise ValueError("an unknown Task cannot be continued")
        if self.scope == "development_main" and self.database_name != "citequest_s03":
            raise ValueError("development Main database identity required")
        if self.scope == "isolated_control" and not self.database_name.startswith("citequest_s08_fixture"):
            raise ValueError("explicit isolated fixture database required")
        return self

    @property
    def source_sha256(self) -> str:
        if self._file_sha256 is None:
            raise AgentError("development_policy_must_be_loaded_from_trusted_file")
        return self._file_sha256

    def require_worker(self, task_id: UUID | None, mode: str) -> None:
        self.require_task(task_id)
        if mode != ("live" if self.scope == "development_main" else "fixture"):
            raise AgentError("development_dispatch_worker_mode_mismatch")

    def require_task(self, task_id: UUID | None) -> None:
        _ = self.source_sha256
        if task_id is None or task_id != self.target.task_id:
            raise AgentError("development_dispatch_requires_explicit_approved_Task")

    def public_binding(self) -> dict:
        return {"version": self.version, "policy_sha256": self.source_sha256,
            "approval_sha256": self.root_approval.sha256, "semantics": self.semantics,
            "historical_attempt_id": str(self.historical.attempt_id), "target_task_id": str(self.target.task_id),
            "known_client_concurrency": 1, "remote_physical_concurrency_asserted": False,
            "unknown_financial_record_changed": False}


def strict_json(raw: bytes) -> object:
    def unique(pairs: list[tuple]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise AgentError("development_policy_duplicate_JSON_property")
            result[key] = value
        return result

    def finite(_value: str) -> None:
        raise AgentError("development_policy_nonfinite_JSON")

    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=finite)
    except (ValueError, UnicodeError, RecursionError):
        raise AgentError("development_policy_invalid_JSON") from None


def load_policy(path: Path, expected_sha256: str) -> DevelopmentDispatchPolicy:
    ref = Artifact(path=str(path.resolve()), sha256=expected_sha256)
    # Check the original reference too; resolve must not hide a final symlink.
    if path.is_symlink():
        raise AgentError("development_policy_symlink_forbidden")
    value = ref.read()
    if value.get("version") == "citequest-development-named-unknown-dispatch/2":
        from .development_dispatch_v2 import load_v2
        return load_v2(value, expected_sha256)
    policy = DevelopmentDispatchPolicy.model_validate_json(json.dumps(value), strict=True)
    policy._file_sha256 = expected_sha256
    stop, ledger, approval = (policy.stopped_client_proof.read(), policy.ledger_proof.read(), policy.root_approval.read())
    validate_proofs(policy, stop, ledger, approval)
    policy._proofs = {"stop": stop, "ledger": ledger, "approval": approval}
    return policy


def validate_proofs(policy: DevelopmentDispatchPolicy, stop: dict, ledger: dict, approval: dict) -> None:
    historical = policy.historical.model_dump(mode="json")
    if any(ledger.get(key) != value for key, value in historical.items()):
        raise AgentError("development_policy_ledger_binding_changed")
    if (ledger.get("status") != "unknown" or ledger.get("usage") is not None or ledger.get("cost_ncny") is not None
            or ledger.get("response_missing") is not True or ledger.get("global_unknown_models") != 1
            or ledger.get("task_state") != "reconciling" or digest(ledger.get("profile")) != policy.historical.profile_hash):
        raise AgentError("development_policy_unknown_facts_changed")
    validate_stopped_client(stop)
    expected = {"version": "citequest-development-dispatch-approval/1", "decision": "approve_one_named_development_verification",
        "scope": policy.scope, "historical_binding_sha256": digest(historical),
        "target_binding_sha256": digest(policy.target.model_dump(mode="json")),
        "stopped_client_proof_sha256": policy.stopped_client_proof.sha256, "ledger_proof_sha256": policy.ledger_proof.sha256,
        "campaign_cash_ncny": 500000000000, "campaign_prior_reserved_ncny": 4212736000,
        "semantics": SEMANTICS, "old_Task_resume_authorized": False, "old_unknown_financial_change_authorized": False,
        "Main_dispatch_authorized": policy.scope == "development_main",
        "approved_by": "Root" if policy.scope == "development_main" else "EXPLICIT_ISOLATED_CONTROL"}
    if approval != expected:
        raise AgentError("development_policy_Root_approval_binding_changed")


def validate_stopped_client(stop: dict) -> None:
    """Same physical-stop checks shared by the fixed v1 and v2 contracts."""
    state = stop.get("state", {})
    restart = stop.get("restart_policy", {})
    if (stop.get("local_client_physical_stop_verified") is not True or stop.get("remote_inference_or_bill_completion_verified") is not False
            or state.get("Running") is not False or type(state.get("Pid")) is not int or state["Pid"] != 0 or state.get("Status") != "exited"
            or state.get("Restarting") is not False or state.get("Paused") is not False
            or state.get("Dead") is not False
            or restart.get("Name") != "no" or restart.get("MaximumRetryCount") != 0
            or not isinstance(stop.get("daemon_id"), str) or not stop["daemon_id"]):
        raise AgentError("development_policy_physical_stop_proof_invalid")
    for key in ("container_id", "source_manifest_sha256", "original_launch_sha256"):
        value = stop.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise AgentError("development_policy_physical_stop_identity_invalid")
    image = stop.get("image_id", "")
    if (not isinstance(image, str) or not image.startswith("sha256:") or len(image) != 71
            or any(char not in "0123456789abcdef" for char in image[7:])):
        raise AgentError("development_policy_physical_stop_identity_invalid")
    try:
        started, finished, observed = (datetime.fromisoformat(value.replace("Z", "+00:00")) for value in
            (state["StartedAt"], state["FinishedAt"], stop["at"]))
        if (any(value.tzinfo is None for value in (started, finished, observed)) or started.year < 2000
                or not started <= finished <= observed):
            raise ValueError("ordered aware physical observations required")
    except (KeyError, TypeError, ValueError):
        raise AgentError("development_policy_physical_stop_time_invalid") from None


def validate_target(db: Any, task: dict, command: dict, logical_key: str, policy: DevelopmentDispatchPolicy) -> None:
    target = policy.target
    expected = {"id": target.task_id, "owner_id": target.owner_id, "project_id": target.project_id,
        "operation_id": target.operation_id, "profile_hash": target.profile_hash, "create_hash": target.initial_payload_sha256,
        "graph_version": target.graph_version, "kind": "qa"}
    if any(task.get(key) != value for key, value in expected.items()):
        raise AgentError("development_dispatch_new_Task_binding_changed")
    profile = task["profile"]
    if (digest(profile) != target.profile_hash or profile.get("profile_id") != "qa-v5"
            or profile.get("active_seconds") != 120 or profile.get("catalog_scope_policy") != "task-catalog-snapshot/1"
            or profile.get("aggregate_policy", {}).get("campaign_cash_ncny") != 500000000000
            or profile.get("aggregate_policy", {}).get("campaign_prior_reserved_ncny") != 4212736000
            or profile.get("aggregate_policy", {}).get("model_dispatch_concurrency") != 1):
        raise AgentError("development_dispatch_profile_or_wallet_changed")
    if not logical_key.startswith(str(command["id"]) + ":") or command.get("task_id") != task["id"]:
        raise AgentError("development_dispatch_command_identity_changed")
    operations = (target.operation_id, *target.followup_operation_ids)
    sequence = command.get("seq")
    if (type(sequence) is not int or not 1 <= sequence <= len(operations)
            or command.get("operation_id") != operations[sequence - 1]
            or command.get("kind") != ("start" if sequence == 1 else "followup")
            or task["command_seq"] != sequence):
        raise AgentError("development_dispatch_command_not_approved")
    initial = db.execute("SELECT payload,payload_hash,operation_id FROM cq_app.agent_commands WHERE task_id=%s AND seq=1 AND kind='start'", (task["id"],)).fetchone()
    if (not initial or initial["operation_id"] != target.operation_id or initial["payload_hash"] != target.initial_payload_sha256
            or digest(initial["payload"]) != target.initial_payload_sha256):
        raise AgentError("development_dispatch_initial_payload_changed")


def validate_occupied(db: Any, task: dict, request: dict, logical_key: str, command: dict,
                      policy: DevelopmentDispatchPolicy, *, current_attempt_id: UUID | None = None) -> list[dict]:
    """Read-only guard, called ONLY under existing campaign wallet locks."""
    if policy.version == "citequest-development-named-unknown-dispatch/2":
        from .development_dispatch_v2 import validate_occupied_v2
        return validate_occupied_v2(db, task, request, logical_key, command, policy,
                                    current_attempt_id=current_attempt_id)
    _ = policy.source_sha256
    if not policy._proofs:
        raise AgentError("development_policy_proofs_not_loaded")
    if db.execute("SELECT current_database() AS name").fetchone()["name"] != policy.database_name:
        raise AgentError("development_dispatch_database_mismatch")
    validate_target(db, task, command, logical_key, policy)
    historical = policy.historical
    old_task = db.execute("SELECT t.*,row_to_json(t)::text AS _row FROM cq_app.agent_tasks t WHERE id=%s FOR SHARE", (historical.task_id,)).fetchone()
    old = db.execute("SELECT a.*,row_to_json(a)::text AS _row FROM cq_app.agent_attempts a WHERE id=%s FOR SHARE", (historical.attempt_id,)).fetchone()
    if (not old_task or not old or hashlib.sha256(old_task["_row"].encode()).hexdigest() != historical.task_row_sha256
            or hashlib.sha256(old["_row"].encode()).hexdigest() != historical.attempt_row_sha256):
        raise AgentError("development_dispatch_historical_row_changed")
    if (old["task_id"] != historical.task_id or old["kind"] != "model" or old["status"] != "unknown"
            or old["usage"] is not None or old["cost_ncny"] is not None or old["response"] is not None
            or old["reserved_ncny"] != historical.reserved_ncny or old["run_id"] != historical.run_id or old["epoch"] != historical.epoch
            or old["request_hash"] != historical.request_hash or digest(old["request"]) != historical.request_hash
            or old_task["owner_id"] != historical.owner_id or old_task["profile_hash"] != historical.profile_hash
            or digest(old_task["profile"]) != historical.profile_hash or old_task["state"] != "reconciling"):
        raise AgentError("development_dispatch_historical_facts_changed")
    profile = old_task["profile"]
    reserve = profile["context_window"] * profile["input_miss_ncny"] + old["output_limit"] * profile["output_ncny"]
    if (reserve != historical.reserved_ncny or old["request"].get("model") != profile["model_alias"]
            or old["request"].get("max_output_tokens") != old["output_limit"]):
        raise AgentError("development_dispatch_historical_price_reserve_changed")
    occupied = db.execute("SELECT id,task_id,request_hash,status FROM cq_app.agent_attempts WHERE kind='model' AND status IN ('reserved','sent','unknown')").fetchall()
    unknown = [row["id"] for row in occupied if row["status"] == "unknown"]
    if unknown != [historical.attempt_id]:
        raise AgentError("development_dispatch_unmatched_unknown_STOP")
    # Old money and records are untouched; only this known-client occupancy
    # projection excludes the approved historical unknown. The ordinary caller
    # still checks at most one remaining reserved/sent request under the lock.
    if request.get("model") != task["profile"]["model_alias"]:
        raise AgentError("development_dispatch_new_model_changed")
    if current_attempt_id is not None:
        current = [row for row in occupied if row["id"] == current_attempt_id]
        if (len(current) != 1 or current[0]["task_id"] != task["id"] or current[0]["status"] != "reserved"
                or current[0]["request_hash"] != digest(request)):
            raise AgentError("development_dispatch_current_reservation_changed")
    return [row for row in occupied if row["id"] not in {historical.attempt_id, current_attempt_id}]
