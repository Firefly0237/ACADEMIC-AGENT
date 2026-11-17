"""Allowlisted task metadata for the read-only workspace list."""

from typing import Literal

from citequest.v2.domain import DomainError

TaskStateGroup = Literal["all", "needs_input", "active", "completed", "ended"]

STATE_GROUPS: dict[str, tuple[str, ...]] = {
    "all": (),
    "needs_input": ("waiting_input",),
    "active": ("queued", "running", "reconciling"),
    "completed": ("succeeded",),
    "ended": ("partial", "failed", "cancelled"),
}


def task_states(state_group: str) -> tuple[str, ...]:
    if state_group not in STATE_GROUPS:
        raise DomainError("invalid_task_state_group", 400)
    return STATE_GROUPS[state_group]


def task_category(kind: str, profile_id: str | None, action: str | None) -> str:
    """Return product categories, never arbitrary profile or Brief content."""
    if kind != "research":
        return kind if kind in {"qa", "summary", "compare", "stance"} else "other"
    if profile_id == "research-review-v1":
        return "research_review"
    if profile_id == "research-revision-v1":
        return "research_revision"
    if profile_id not in {"research-v1", "research-v2", "research-v3", "research-v4", "research-v5", "research-v6"}:
        return "research"
    return {
        "plan": "research_plan",
        "experiment": "experiment_design",
        "suggest": "research_suggestions",
        "extract": "draft_intake",
        # Older research tasks omit the action and use the planning default.
        None: "research_plan",
    }.get(action, "research")
