"""One server-bound goal projection for planning, provisional work and review."""
from __future__ import annotations

import copy
import json
from uuid import UUID

from citequest.research_v3.current_requirements import (
    RequirementsCandidate,
    RequirementsState,
    UserTurn,
    active_projection,
)

from .contracts import AgentError, ResearchBrief, digest


def authority_hash(task: dict) -> str:
    """Immutable Task authority, never a model's proposed requirements."""
    brief = ResearchBrief.model_validate_json(json.dumps(task["brief"]), strict=True)
    return digest({"task_id":str(UUID(str(task["id"]))), "owner_id":str(UUID(str(task["owner_id"]))),
        "project_id":str(UUID(str(task["project_id"]))), "profile_hash":task["profile_hash"], "kind":task["kind"],
        "catalog":brief.catalog, "selected_papers":[item.model_dump(mode="json") for item in brief.selected_papers]})


def projection(base: dict, task: dict, accepted: RequirementsState, turn: UserTurn, *,
               candidate: RequirementsCandidate | None = None, reviewer: bool = False, planning: bool = False) -> tuple[dict, dict]:
    if accepted.authority_sha256 != authority_hash(task) or not accepted.accepted:
        raise AgentError("requirements_context_authority_changed")
    if reviewer and base.get("selected_user_skills"):
        raise AgentError("requirements_reviewer_contains_generator_skill_body")
    prompt = copy.deepcopy(base)
    manifest = prompt["context_manifest"]
    for name in ("user_turns", "required_user_turns", "prior_turns", "raw_turn_reference_ids"):
        prompt.pop(name, None)
    prompt["canonical_brief"]["question"] = turn.text
    prompt["current_user_turn"] = {**turn.model_dump(), "sha256":digest(turn.text)}
    prompt["current_requirements"] = active_projection(accepted)
    prompt["raw_turns_are_audit_not_additional_current_goals"] = True
    manifest["requirements_protocol"] = accepted.version
    manifest["requirements_accepted_revision"] = accepted.revision
    manifest["requirements_accepted_sha256"] = digest(accepted.model_dump(mode="json"))
    manifest["current_user_turn"] = {"id":turn.turn_id,"sha256":digest(turn.text)}
    if candidate is not None:
        if candidate.base_sha256 != manifest["requirements_accepted_sha256"] or candidate.user_turn != turn:
            raise AgentError("requirements_context_candidate_or_turn_changed")
        prompt["candidate_requirements_are_not_current_until_independent_review"] = True
        prompt["requirements_candidate_sha256"] = digest(candidate.model_dump(mode="json"))
        manifest["requirements_candidate_sha256"] = prompt["requirements_candidate_sha256"]
        if candidate.projected is not None:
            # Only active provisional instructions and just-retired targets are
            # needed. Unrelated old tombstone prose never becomes a new goal.
            prompt["provisional_requirements"] = [item.model_dump(mode="json") for item in candidate.projected.requirements
                if item.state == "active" or reviewer and item.id in candidate.required_review_ids]
        prompt["required_requirement_review_ids"] = candidate.required_review_ids
    if planning:
        prompt.pop("untrusted_evidence_data", None)
        prompt.pop("quote_scope", None)
    return prompt, manifest
