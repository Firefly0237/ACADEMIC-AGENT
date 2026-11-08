"""Server-derived research progress, never an unpublished research answer.

Only bounded counts and protocol states cross this boundary. Callers must still
check the current Brief, skills, memory and every source dependency before
returning the metadata. A progress record cannot authorize any candidate text.
"""
from __future__ import annotations

from typing import Literal, get_args

from pydantic import Field, ValidationError, model_validator

from .contracts import Strict

WaitKind = Literal[
    "scope_sync_required", "research_intake_required", "outline_request_gaps",
    "research_evidence_insufficient", "research_motivation_insufficient",
    "research_review_needs_input", "research_request_coverage_partial",
    "research_scope_exceeds_remaining_budget", "revision_needs_user_input",
    "research_conflict_resolution_set_exceeds_bound", "research_draft_input_required",
    "school_writing_incomplete", "other_required_input",
]
WAIT_KINDS = frozenset(get_args(WaitKind))


class ResearchProgress(Strict):
    schema_version: Literal["citequest-research-progress/1"] = "citequest-research-progress/1"
    status: Literal["waiting_input", "partial", "rejected", "structured_plan_ready"]
    research_status: Literal["proposed_not_executed"] = "proposed_not_executed"
    school_format_ready: Literal[False] = False
    research_questions_planned: int = Field(ge=0, le=8)
    literature_reviews_completed: int = Field(ge=0, le=8)
    experiments_planned: int = Field(ge=0, le=16)
    method_proposals_completed: int = Field(ge=0, le=16)
    unit_reviews_completed: int = Field(ge=0, le=8)
    cross_plan_review_completed: bool
    targeted_revision_round: int = Field(ge=0, le=2)
    waiting_kind: WaitKind | None
    school_writing_status: Literal["not_started", "partial"] = "not_started"
    school_sections_written: int = Field(default=0, ge=0, le=14)
    school_section_reviews_completed: int = Field(default=0, ge=0, le=14)

    @model_validator(mode="after")
    def observed_counts_within_plan(self) -> ResearchProgress:
        if (self.literature_reviews_completed > self.research_questions_planned
                or self.unit_reviews_completed > self.research_questions_planned
                or self.method_proposals_completed > self.experiments_planned):
            raise ValueError("research progress cannot report more completed units than planned")
        if self.school_section_reviews_completed > self.school_sections_written:
            raise ValueError("school review count exceeds retained sections")
        return self


def progress_from_state(state: dict, result: dict) -> dict:
    """Derive a metadata snapshot from completed graph nodes, without raw prose."""
    questions = state.get("outline", {}).get("questions", [])
    waiting = state.get("waiting", {})
    interrupts = result.get("interrupts", [])
    if interrupts and not waiting:
        waiting = interrupts[0]
    kind = waiting.get("kind")
    if kind is not None and kind not in WAIT_KINDS:
        kind = "other_required_input"
    status = ("rejected" if result.get("publication_status") == "rejected" else
        "structured_plan_ready" if result.get("status") == "structured_plan_ready" else
        "waiting_input" if result.get("status") == "waiting_input" or interrupts else "partial")
    return ResearchProgress(status=status,
        research_questions_planned=len(questions),
        literature_reviews_completed=len(state.get("literature", [])),
        experiments_planned=sum(len(question["experiment_ids"]) for question in questions),
        method_proposals_completed=len(state.get("methods", [])),
        unit_reviews_completed=len(state.get("unit_reviews", [])),
        cross_plan_review_completed=bool(state.get("global_review")),
        targeted_revision_round=state.get("revision_round", 0),
        waiting_kind=kind).model_dump(mode="json")


def public_research_progress(result: dict) -> dict | None:
    """Strict allowlist for an already current-authorized research result.

    Legacy records remain absent. A malformed record, unknown field, boolean
    count or arbitrary model-produced string is rejected rather than copied.
    This function does not check ACLs and must not be used before those checks.
    """
    if result.get("kind") != "research" or not isinstance(result.get("research_progress"), dict):
        return None
    try:
        return ResearchProgress.model_validate(result["research_progress"], strict=True).model_dump(mode="json")
    except ValidationError:
        return None


def school_partial_progress(upstream: dict, state: dict) -> dict | None:
    """Keep section counts when School writing stops; never expose draft text."""
    value = public_research_progress(upstream)
    if value is None:
        return None
    return ResearchProgress.model_validate({**value, "status": "partial", "school_writing_status": "partial",
        "school_sections_written": len(state.get("sections", {})),
        "school_section_reviews_completed": len(state.get("reviews", {})),
        "waiting_kind": "school_writing_incomplete"}, strict=True).model_dump(mode="json")
