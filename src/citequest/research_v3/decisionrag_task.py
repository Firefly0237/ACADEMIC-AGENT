"""Trusted, content-addressed Task configuration for real issued-ref navigation."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from citequest.agents_v3.contracts import AgentError, Strict, digest

from .decisionrag import NavigationLimits
from .decisionrag_remote import RemoteBinaryJudge, RemoteJudgeConfig

TASK_PROTOCOL = "laya1"
TOOL_NAME = "navigate_issued_evidence"


class DecisionRAGTaskConfig(Strict):
    version: Literal["citequest-decisionrag-task/1"]
    judge: RemoteJudgeConfig
    document_keep: int = Field(default=1, ge=1, le=8)
    passage_keep: int = Field(default=1, ge=1, le=8)
    max_documents: int = Field(default=2, ge=1, le=8)
    max_passages: int = Field(default=2, ge=1, le=8)
    max_shared_bytes: int = Field(default=32768, ge=1024, le=32768)
    active_seconds: int = Field(default=25, ge=1, le=25)
    local_input_token_limit: int = Field(default=16384, ge=1, le=16384)

    @property
    def navigation_protocol(self) -> str:
        return "laya1"

    @property
    def max_forward_calls(self) -> int:
        return 2

    @model_validator(mode="after")
    def capacity(self):
        if self.version == "citequest-decisionrag-task/1" and self.judge.version != "citequest-decisionrag-remote/1":
            raise ValueError("laya1 requires its original 1024-token probability protocol")
        if self.document_keep > self.max_documents or self.passage_keep > self.max_passages:
            raise ValueError("navigation keep counts cannot exceed the admitted frontier")
        return self

    def limits(self, seconds: float) -> NavigationLimits:
        return NavigationLimits(document_keep=self.document_keep, passage_keep=self.passage_keep,
            max_documents=self.max_documents, max_passages=self.max_passages,
            max_shared_bytes=self.max_shared_bytes, active_seconds=min(seconds, self.active_seconds))


class DecisionRAGTaskConfigV2(DecisionRAGTaskConfig):
    version: Literal["citequest-decisionrag-task/2"]
    max_documents: int = Field(default=20, ge=1, le=20)
    max_passages: int = Field(default=20, ge=1, le=20)
    shortlist_count: int = Field(default=8, ge=1, le=8)
    max_batches_per_level: int = Field(default=4, ge=1, le=4)
    maximum_forward_calls: int = Field(default=10, ge=2, le=10)
    local_input_token_limit: int = Field(default=131072, ge=1, le=131072)

    @property
    def navigation_protocol(self) -> str:
        return "laya2"

    @property
    def max_forward_calls(self) -> int:
        return self.maximum_forward_calls

    @model_validator(mode="after")
    def batched_capacity(self):
        if (self.judge.version != "citequest-decisionrag-remote/2"
                or self.max_batches_per_level > self.shortlist_count
                or max(self.document_keep, self.passage_keep) > self.shortlist_count):
            raise ValueError("laya2 needs its explicit 8192-token gateway and complete shared shortlist capacity")
        return self

    def limits(self, seconds: float):
        from .decisionrag_batched import BatchedNavigationLimits
        return BatchedNavigationLimits(document_keep=self.document_keep, passage_keep=self.passage_keep,
            max_documents=self.max_documents, max_passages=self.max_passages,
            max_shared_bytes=self.max_shared_bytes, active_seconds=min(seconds, self.active_seconds),
            shortlist_count=self.shortlist_count, max_batches_per_level=self.max_batches_per_level,
            max_forward_calls=self.maximum_forward_calls)


def load_task_config(path: Path, expected_sha256: str) -> DecisionRAGTaskConfig | DecisionRAGTaskConfigV2:
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > 16384
            or type(expected_sha256) is not str or len(expected_sha256) != 64):
        raise AgentError("decisionrag_trusted_configuration_required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise AgentError("decisionrag_configuration_changed")
    kind = DecisionRAGTaskConfigV2 if json.loads(raw).get("version") == "citequest-decisionrag-task/2" else DecisionRAGTaskConfig
    return kind.model_validate_json(raw, strict=True)


def task_requests_navigation(task: dict) -> bool:
    value = task["profile"].get("evidence_navigation_protocol")
    if value is None:
        return False
    if (value not in {TASK_PROTOCOL, "laya2"} or task["kind"] != "qa"
            or task["profile"].get("profile_id") != "qa-v9"
            or task["profile"].get("qa_content_protocol") != "isolated1"):
        raise AgentError("decisionrag_Task_profile_not_supported")
    return True


def navigate_task_evidence(config: DecisionRAGTaskConfig, query: str, ref_ids: list[str], *,
                          access, context: Callable[[], dict], remaining_ms: int) -> dict:
    """One durable MCP attempt owns these two real local probability forwards."""
    if isinstance(config, DecisionRAGTaskConfigV2):
        return _navigate_task_evidence_v2(config, query, ref_ids, access=access,
            context=context, remaining_ms=remaining_ms)
    from citequest.v2.domain import Actor
    from .decisionrag_evidence import navigate_issued_evidence

    if type(remaining_ms) is not int or not 1 <= remaining_ms <= 25000:
        raise AgentError("decisionrag_server_execution_deadline_required")
    started = time.monotonic()
    seconds = min(remaining_ms / 1000, config.active_seconds)
    deadline = started + seconds
    initial = context()
    if not task_requests_navigation(initial) or initial["profile"].get("evidence_navigation_protocol") != config.navigation_protocol:
        raise AgentError("decisionrag_Task_profile_not_enabled")

    def scope_hash(task):
        return digest({**{key: str(task[key]) for key in ("id", "owner_id", "project_id")},
            **{key: task[key] for key in ("epoch", "revision", "brief", "profile")}})

    initial_scope = scope_hash(initial)

    def current_scope():
        if time.monotonic() >= deadline:
            raise AgentError("decisionrag_active_budget_exhausted")
        current = context()  # Live epoch/running/owner/cancellation guard.
        if scope_hash(current) != initial_scope:
            raise AgentError("decisionrag_Task_scope_changed")
        return initial_scope

    judge = RemoteBinaryJudge(config.judge, deadline=deadline)
    admissions = []
    admitted_tokens = 0

    def before_decision(request):
        nonlocal admitted_tokens
        current_scope()
        tokens = judge.preflight_tokens.get(request.fingerprint())
        if type(tokens) is not int or tokens < 1:
            raise AgentError("decisionrag_complete_token_preflight_missing")
        if admitted_tokens + tokens > config.local_input_token_limit:
            raise AgentError("decisionrag_local_token_budget_exhausted")
        admitted_tokens += tokens
        admissions.append({"stage": request.stage, "request_sha256": request.fingerprint(),
            "candidate_count": len(request.candidates), "complete_input_tokens": tokens,
            "source_tokens_dropped": 0})

    result = navigate_issued_evidence(query, ref_ids, judge, access=access,
        actor=Actor(initial["owner_id"]), task_id=initial["id"], current_task_scope=current_scope,
        before_decision=before_decision, limits=config.limits(seconds))
    current_scope()
    forwards = sum(item["usage"]["observed_forward_calls"] for item in result["trace"])
    tokens = sum(item["usage"]["input_tokens"] for item in result["trace"])
    if forwards != 2 or tokens != admitted_tokens or tokens > config.local_input_token_limit:
        raise AgentError("decisionrag_local_usage_admission_mismatch")
    return {"status": "ok", **result, "navigation_protocol": TASK_PROTOCOL,
        "local_usage": {"observed_forward_calls": forwards, "input_tokens": tokens,
            "source_tokens_dropped": 0, "external_API_cash_ncny": 0,
            "physical_compute_cost_not_measured": True, "elapsed_seconds": time.monotonic() - started},
        "local_forward_admissions": admissions,
        "resource_accounting": "one_durable_Task_tool_attempt; two_local_forwards; enclosing_Task_active_window",
        "automatic_fallback": False}


def _navigate_task_evidence_v2(config, query, ref_ids, *, access, context, remaining_ms):
    from citequest.v2.domain import Actor
    from .decisionrag_batched import navigate_batched
    from .decisionrag_evidence import navigate_issued_evidence

    if type(remaining_ms) is not int or not 1 <= remaining_ms <= 25000:
        raise AgentError("decisionrag_server_execution_deadline_required")
    started = time.monotonic()
    seconds = min(remaining_ms / 1000, config.active_seconds)
    deadline = started + seconds
    initial = context()
    if not task_requests_navigation(initial) or initial["profile"].get("evidence_navigation_protocol") != "laya2":
        raise AgentError("decisionrag_Task_profile_not_enabled")

    def scope_hash(task):
        return digest({**{key: str(task[key]) for key in ("id", "owner_id", "project_id")},
            **{key: task[key] for key in ("epoch", "revision", "brief", "profile")}})

    initial_scope = scope_hash(initial)

    def current_scope():
        if time.monotonic() >= deadline:
            raise AgentError("decisionrag_active_budget_exhausted")
        if scope_hash(context()) != initial_scope:
            raise AgentError("decisionrag_Task_scope_changed")
        return initial_scope

    judge = RemoteBinaryJudge(config.judge, deadline=deadline)
    admissions, admitted_tokens = [], 0

    def before_decision(request):
        nonlocal admitted_tokens
        current_scope()
        tokens = judge.preflight_tokens.get(request.fingerprint())
        if type(tokens) is not int or tokens < 1:
            raise AgentError("decisionrag_complete_token_preflight_missing")
        if len(admissions) >= config.max_forward_calls:
            raise AgentError("decisionrag_forward_budget_exhausted")
        if admitted_tokens + tokens > config.local_input_token_limit:
            raise AgentError("decisionrag_local_token_budget_exhausted")
        admitted_tokens += tokens
        admissions.append({"stage": request.stage, "request_sha256": request.fingerprint(),
            "candidate_count": len(request.candidates), "complete_input_tokens": tokens,
            "source_tokens_dropped": 0})

    try:
        result = navigate_issued_evidence(query, ref_ids, judge, access=access,
            actor=Actor(initial["owner_id"]), task_id=initial["id"], current_task_scope=current_scope,
            before_decision=before_decision, limits=config.limits(seconds), navigator=navigate_batched)
    except Exception as error:
        result = {"status": "error", "error_code": error.code if isinstance(error, AgentError)
            else "decisionrag_navigation_internal_error", "trace": [], "selection_steps": [],
            "unknown_outcome": False, "decision_calls": 0, "evidence_ref_ids": []}
    try:
        current_scope()
    except AgentError as error:
        result.update(status="error", error_code=error.code, evidence_ref_ids=[], selected_passage_ids=[])
    trace = result["trace"]
    known_tokens = sum(row["usage"]["input_tokens"] for row in trace)
    known_forwards = len(trace)
    unknown = result["unknown_outcome"]
    for index, admission in enumerate(admissions):
        admission["execution_status"] = "completed" if index < len(trace) else "unknown" if unknown else "not_executed"
    body = {**result, "navigation_protocol": "laya2", "input_frontier_ref_ids": list(ref_ids),
        "configuration_value_sha256": digest(config.model_dump(mode="json")),
        "request_context_tokens": 8192, "native_default_context_tokens": 1024,
        "local_usage": {"observed_forward_calls": None if unknown else known_forwards,
            "input_tokens": None if unknown else known_tokens,
            "completed_forward_calls": known_forwards, "known_input_tokens": known_tokens,
            "source_tokens_dropped": None if unknown else 0,
            "completed_source_tokens_dropped": 0, "external_API_cash_ncny": 0,
            "physical_compute_cost_not_measured": True, "elapsed_seconds": time.monotonic() - started},
        "local_forward_admissions": admissions,
        "resource_accounting": "one_durable_Task_tool_attempt; bounded_actual_local_forwards; enclosing_Task_active_window",
        "automatic_fallback": False}
    validate_task_receipt(config, body, ref_ids)
    return body


def validate_task_receipt(config: DecisionRAGTaskConfigV2, body: dict, ref_ids: list[str]) -> None:
    """Validate complete v2 successes and partial failures before ledger settlement."""
    from .decisionrag_batched import shortlist_quotas
    from .decisionrag_remote import require_model_binding

    def require(condition):
        if not condition:
            raise ValueError("receipt contract")

    try:
        require(isinstance(config, DecisionRAGTaskConfigV2) and type(body) is dict)
        require(body.get("navigation_protocol") == "laya2" and body.get("status") in {"ok", "error"}
            and body.get("input_frontier_ref_ids") == ref_ids
            and body.get("configuration_value_sha256") == digest(config.model_dump(mode="json"))
            and body.get("request_context_tokens") == 8192 and body.get("native_default_context_tokens") == 1024
            and body.get("automatic_fallback") is False)
        require(type(ref_ids) is list and 1 <= len(ref_ids) <= config.max_passages
            and len(set(ref_ids)) == len(ref_ids))
        unknown, trace, admissions = body.get("unknown_outcome"), body.get("trace"), body.get("local_forward_admissions")
        require(type(unknown) is bool and type(trace) is list and type(admissions) is list
            and len(trace) <= config.max_forward_calls
            and len(trace) <= len(admissions) <= min(config.max_forward_calls, len(trace) + 1))
        total_tokens, binding = 0, None
        for index, row in enumerate(trace):
            require(type(row) is dict and row.get("stage") in {"documents", "passages"}
                and row.get("phase") in {"screen", "final"}
                and type(row.get("batch_index")) is int and 0 <= row["batch_index"] < config.max_batches_per_level
                and re.fullmatch(r"[a-f0-9]{64}", row.get("request_sha256", "")) is not None)
            ids, probabilities, usage = row["candidate_ids"], row["probabilities"], row["usage"]
            require(type(ids) is list and 1 <= len(ids) <= 8 and len(set(ids)) == len(ids)
                and all(type(value) is str and 1 <= len(value) <= 300 for value in ids)
                and type(probabilities) is dict and set(probabilities) == set(ids)
                and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                    for value in probabilities.values()))
            require(type(usage) is dict and type(usage.get("observed_forward_calls")) is int
                and usage["observed_forward_calls"] == 1 and usage.get("truncated") is False
                and usage.get("state_tokens_dropped") == 0 and usage.get("external_API_cash_ncny") == 0
                and type(usage.get("input_tokens")) is int and usage["input_tokens"] > 0)
            checks = usage.get("exact_token_preflight")
            require(type(checks) is dict and set(checks) == {"N" + str(i) for i in range(1, len(ids) + 1)})
            require(all(type(check) is dict and type(check.get("sequence_tokens")) is int
                and 1 <= check["sequence_tokens"] <= 8192 and check.get("truncated") is False
                and check.get("state_tokens_dropped") == 0 for check in checks.values()))
            require(sum(check["sequence_tokens"] for check in checks.values()) == usage["input_tokens"])
            require_model_binding(row["model_binding"], sequence_tokens=8192)
            require(binding is None or row["model_binding"] == binding)
            binding = row["model_binding"]
            admission = admissions[index]
            require(admission.get("request_sha256") == row["request_sha256"]
                and admission.get("stage") == row["stage"] and admission.get("candidate_count") == len(ids)
                and admission.get("complete_input_tokens") == usage["input_tokens"]
                and admission.get("execution_status") == "completed" and admission.get("source_tokens_dropped") == 0)
            total_tokens += usage["input_tokens"]
        if len(admissions) > len(trace):
            last = admissions[-1]
            require(last.get("execution_status") == ("unknown" if unknown else "not_executed")
                and re.fullmatch(r"[a-f0-9]{64}", last.get("request_sha256", "")) is not None
                and type(last.get("complete_input_tokens")) is int and last["complete_input_tokens"] > 0
                and type(last.get("candidate_count")) is int and 1 <= last["candidate_count"] <= 8
                and last.get("source_tokens_dropped") == 0)
        require(sum(admission["complete_input_tokens"] for admission in admissions) <= config.local_input_token_limit)
        usage = body.get("local_usage")
        require(type(usage) is dict and usage.get("completed_forward_calls") == len(trace)
            and type(usage.get("completed_forward_calls")) is int
            and usage.get("known_input_tokens") == total_tokens and type(usage.get("known_input_tokens")) is int
            and usage.get("completed_source_tokens_dropped") == 0 and usage.get("external_API_cash_ncny") == 0
            and usage.get("physical_compute_cost_not_measured") is True
            and type(usage.get("elapsed_seconds")) in (int, float) and math.isfinite(usage["elapsed_seconds"])
            and usage["elapsed_seconds"] >= 0)
        if unknown:
            require(body["status"] == "error" and len(admissions) == len(trace) + 1
                and usage.get("observed_forward_calls") is None and usage.get("input_tokens") is None
                and usage.get("source_tokens_dropped") is None and body.get("decision_calls") is None)
        else:
            require(type(usage.get("observed_forward_calls")) is int and usage["observed_forward_calls"] == len(trace)
                and type(usage.get("input_tokens")) is int and usage["input_tokens"] == total_tokens
                and usage.get("source_tokens_dropped") == 0 and body.get("decision_calls") == len(trace))
        if body["status"] == "error":
            require(not body.get("evidence_ref_ids") and re.fullmatch(r"[A-Za-z0-9_]{1,150}", body.get("error_code", "")))
            return
        require(not unknown and len(admissions) == len(trace) and 2 <= len(trace) <= config.max_forward_calls)
        steps = body.get("selection_steps")
        require(type(steps) is list and [step.get("stage") for step in steps] == ["documents", "passages"])
        expected_kept = []
        for step, keep in zip(steps, (config.document_keep, config.passage_keep), strict=True):
            rows = [row for row in trace if row["stage"] == step["stage"]]
            require(bool(rows) and rows[-1]["phase"] == "final" and rows[-1]["batch_index"] == 0
                and all(row["phase"] == "screen" for row in rows[:-1]))
            final = rows[-1]
            groups = step["screening_batches"]
            if len(rows) == 1:
                require(groups == [final["candidate_ids"]])
            else:
                screens = rows[:-1]
                require([row["batch_index"] for row in screens] == list(range(len(screens)))
                    and groups == [row["candidate_ids"] for row in screens])
                flattened = [value for group in groups for value in group]
                require(len(flattened) == len(set(flattened)))
                chosen = set()
                for row, quota in zip(screens, shortlist_quotas([len(group) for group in groups], config.shortlist_count), strict=True):
                    ids = row["candidate_ids"]
                    order = {value: i for i, value in enumerate(ids)}
                    chosen.update(sorted(ids, key=lambda value: (-row["probabilities"][value], order[value]))[:quota])
                require(final["candidate_ids"] == [value for value in flattened if value in chosen])
            require(step["shared_final_candidate_ids"] == final["candidate_ids"])
            order = {value: i for i, value in enumerate(final["candidate_ids"])}
            kept = sorted(final["candidate_ids"], key=lambda value: (-final["probabilities"][value], order[value]))[:keep]
            require(step["kept_ids"] == kept)
            expected_kept.append(kept)
        require(body.get("document_ids_kept") == expected_kept[0]
            and body.get("selected_passage_ids") == expected_kept[1]
            and body.get("evidence_ref_ids") == expected_kept[1]
            and set(expected_kept[1]) <= set(ref_ids))
    except (KeyError, TypeError, ValueError, AttributeError, AgentError):
        raise AgentError("decisionrag_MCP_receipt_contract_invalid", 503) from None
