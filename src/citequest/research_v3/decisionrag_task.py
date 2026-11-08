"""Trusted, content-addressed Task configuration for real issued-ref navigation."""
from __future__ import annotations

import hashlib
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

    @model_validator(mode="after")
    def capacity(self):
        if self.document_keep > self.max_documents or self.passage_keep > self.max_passages:
            raise ValueError("navigation keep counts cannot exceed the admitted frontier")
        return self

    def limits(self, seconds: float) -> NavigationLimits:
        return NavigationLimits(document_keep=self.document_keep, passage_keep=self.passage_keep,
            max_documents=self.max_documents, max_passages=self.max_passages,
            max_shared_bytes=self.max_shared_bytes, active_seconds=min(seconds, self.active_seconds))


def load_task_config(path: Path, expected_sha256: str) -> DecisionRAGTaskConfig:
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > 16384
            or type(expected_sha256) is not str or len(expected_sha256) != 64):
        raise AgentError("decisionrag_trusted_configuration_required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise AgentError("decisionrag_configuration_changed")
    return DecisionRAGTaskConfig.model_validate_json(raw, strict=True)


def task_requests_navigation(task: dict) -> bool:
    value = task["profile"].get("evidence_navigation_protocol")
    if value is None:
        return False
    if (value != TASK_PROTOCOL or task["kind"] != "qa"
            or task["profile"].get("profile_id") != "qa-v9"
            or task["profile"].get("qa_content_protocol") != "isolated1"):
        raise AgentError("decisionrag_Task_profile_not_supported")
    return True


def navigate_task_evidence(config: DecisionRAGTaskConfig, query: str, ref_ids: list[str], *,
                          access, context: Callable[[], dict], remaining_ms: int) -> dict:
    """One durable MCP attempt owns these two real local probability forwards."""
    from citequest.v2.domain import Actor
    from .decisionrag_evidence import navigate_issued_evidence

    if type(remaining_ms) is not int or not 1 <= remaining_ms <= 25000:
        raise AgentError("decisionrag_server_execution_deadline_required")
    started = time.monotonic()
    seconds = min(remaining_ms / 1000, config.active_seconds)
    deadline = started + seconds
    initial = context()
    if not task_requests_navigation(initial):
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
