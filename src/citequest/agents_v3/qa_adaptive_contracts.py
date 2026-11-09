"""Lightweight contracts for one corrective retrieval under unchanged goals."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from citequest.search_v3.schemas import SearchRequest

from .contracts import AgentError, Strict, validate_user_text

SUPPLEMENT_ROLE = "supplement_plan"
SUPPLEMENT_SCHEMA_NAME = "citequest_qa5_supplement_plan"
PROTOCOL = "adaptive1"


class SupplementQuery(Strict):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(ge=1, le=20)

    @field_validator("query")
    @classmethod
    def visible_query(cls, value: str) -> str:
        validate_user_text(value)
        SearchRequest(q=value)
        return value


class SupplementPlan(Strict):
    round: Literal[1] = 1
    candidate_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    target_requirement_ids: list[str] = Field(min_length=1, max_length=32)
    action: Literal["retrieve", "keep_bounded_answer"]
    queries: list[SupplementQuery] = Field(max_length=1)
    public_reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def single_action(self) -> SupplementPlan:
        if len(set(self.target_requirement_ids)) != len(self.target_requirement_ids):
            raise ValueError("supplement requirements must be unique")
        if (self.action == "retrieve") != bool(self.queries):
            raise ValueError("only retrieval has exactly one query")
        return self


def validate_supplement_plan(value: dict | SupplementPlan, *, candidate_sha256: str,
                             evidence_sha256: str, target_requirement_ids: list[str],
                             existing_ref_ids: list[str], allowed_limits: list[int]) -> SupplementPlan:
    plan = value if isinstance(value, SupplementPlan) else SupplementPlan.model_validate(value, strict=True)
    if (plan.candidate_sha256 != candidate_sha256 or plan.evidence_sha256 != evidence_sha256
            or not set(plan.target_requirement_ids) <= set(target_requirement_ids)):
        raise AgentError("qa_adaptive_plan_changes_reviewed_scope")
    if len(existing_ref_ids) != len(set(existing_ref_ids)) or len(existing_ref_ids) > 20:
        raise AgentError("qa_adaptive_existing_evidence_invalid")
    if any(query.limit not in allowed_limits or query.limit + len(existing_ref_ids) > 20 for query in plan.queries):
        raise AgentError("qa_adaptive_query_capacity_invalid")
    return plan


def bounded_gap_targets(data: dict, assessment: dict) -> list[str]:
    """Only a safe independently reviewed answer can request more sources."""
    if (not data.get("accepted") or not data.get("terminal") or data.get("feedback")
            or assessment.get("gaps_rejected") or assessment.get("quantitative_rejected_claim_ids")
            or assessment.get("gap_contract_rejected")):
        return []
    return [identifier for identifier, row in assessment.get("requirement_coverage", {}).items()
            if row.get("status") == "bounded_evidence_gap"]


def supplement_capacity(task: dict, *, revisions: int, recoveries: int, evidence_count: int,
                        allowed_limits: list[int], navigation: bool) -> dict:
    """Check the complete path; actual requests keep sequential reservation."""
    profile = task["profile"]
    remaining_revisions = max(0, 1 - revisions)  # This supplement consumes one existing credit.
    models = 3 + 2 * remaining_revisions + max(0, 2 - recoveries)
    tools = 3 + 4 * remaining_revisions + int(navigation)
    slots = 20 - evidence_count
    limits = [value for value in allowed_limits if value <= slots]
    single_cash = (profile["context_window"] * profile["input_miss_ncny"]
                   + profile["per_call_output"] * profile["output_ncny"])
    reason = next((name for name, rejected in (
        ("content_revision", revisions >= 2),
        ("evidence_slots", not limits),
        ("model_calls", profile["model_calls"] - task["model_calls"] < models),
        ("output_tokens", profile["output_tokens"] - task["output_tokens"] < models * profile["per_call_output"]),
        ("tool_calls", profile["tool_calls"] - task["tool_calls"] < tools),
        ("input_tokens", profile["input_tokens"] - task["input_tokens"] <= 0),
        ("cash", profile["cash_ncny"] - task["known_ncny"] - task["reserved_ncny"] < single_cash),
        ("active_time", task.get("elapsed", task.get("active_seconds", 0)) >= profile["active_seconds"]),
    ) if rejected), None)
    return {"admitted": reason is None, "reason": reason, "required_model_calls": models,
            "required_tool_calls": tools, "required_output_tokens": models * profile["per_call_output"],
            "allowed_limits": limits, "remaining_evidence_slots": slots,
            "cash_headroom_basis": "maximum_single_request_sequential_reservation",
            "future_exact_input_tokens_reserved": False}
