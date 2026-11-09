"""One currently authorized assistant utterance, separate from facts and goals.

Uses the normal result-projection boundary over server-owned Task history. No
memory store, user-history replay, source-query fallback or private draft exists
here. Whole reference targets use the existing optional prior-answer wire budget.
"""
from __future__ import annotations

import copy
import json
from typing import Any

from citequest.v2.domain import Actor

from .contracts import canonical, digest

VERSION = "citequest-dialogue-reference/1"


def collect_reference(graph: Any, task: dict, current_manifest: dict) -> dict:
    history = task.get("history", [])
    selected_seq = task.get("brief", {}).get("reference_turn_seq")
    selected_visible = None
    if selected_seq is not None:
        if type(selected_seq) is not int or not 1 <= selected_seq <= 9007199254740991:
            return {"version": VERSION, "status": "unavailable", "reason": "invalid_reference_turn_sequence"}
        authorize = getattr(graph, "authorize_dialogue_refs", None)
        if authorize is None:
            return {"version": VERSION, "status": "unavailable", "reason": "current_reference_authorizer_not_configured"}
        from .conversation import ConversationReader

        reader = ConversationReader(_task_service(graph), graph.memory, authorize)
        page = reader.page(Actor(task["owner_id"]), task["id"], after_seq=selected_seq - 1, limit=1)
        rows = page["items"]
        if not rows or rows[0]["command_seq"] != selected_seq:
            return {"version": VERSION, "status": "unavailable", "reason": "selected_answer_not_retained",
                "requested_turn_seq": selected_seq}
        row = rows[0]
        if row["answer_availability"] != "available" or not isinstance(row["answer"], dict):
            return {"version": VERSION, "status": "unavailable", "reason": row["answer_availability"],
                "requested_turn_seq": selected_seq}
        # The archive reader already projects this exact completed turn with its original Brief.
        previous = {"answer": row["answer"], "brief_revision": row["brief_revision"]}
        selected_visible = row["answer"]
    elif not history:
        return {"version": VERSION, "status": "no_previous_published_answer"}
    else:
        previous = history[-1]
    result = previous.get("answer")
    if not isinstance(result, dict) or result.get("publication_status") != "validated_ai_review":
        return {"version": VERSION, "status": "unavailable", "reason": "previous_answer_not_published"}
    if result.get("context_manifest", {}).get("skill_refs", []) != current_manifest.get("skill_refs", []):
        return {"version": VERSION, "status": "unavailable", "reason": "previous_answer_skill_context_changed"}
    authorize = getattr(graph, "authorize_dialogue_refs", None)
    if authorize is None:
        return {"version": VERSION, "status": "unavailable", "reason": "current_reference_authorizer_not_configured"}
    if selected_visible is not None:
        visible = selected_visible
    else:
        from .database import TaskService
        from .result_projection import project_task_result

        database = graph.execution.database
        campaign = getattr(database, "development_campaign_policy", None)
        # The history belongs to the current already owner-locked Execution. Never
        # accept a caller-supplied answer or reconstruct an unpublished checkpoint.
        public = TaskService.public({**task, "result": result},
            catalog_scope_policy=database.catalog_scope_policy, development_campaign_policy=campaign)
        checked = project_task_result(_task_service(graph), graph.memory, authorize, Actor(task["owner_id"]), task["id"], public)
        visible = checked.get("result")
        if visible is None or checked.get("result_availability") != "available":
            return {"version": VERSION, "status": "unavailable", "reason": checked.get("result_availability", "current_authorization_failed")}
    claims = []
    for claim in visible.get("claims", []):
        claims.append({"claim_id": claim["id"], "statement": claim["statement"]})
    for document in visible.get("documents", []):
        if document.get("summary"):
            identifier = document["document"]["document_id"]
            claims.extend({"document_id": identifier, "claim_id": claim["id"], "statement": claim["statement"]}
                for claim in document["summary"]["claims"])
    claims.extend({"claim_id": claim["id"], "statement": claim["statement"]} for claim in visible.get("comparisons", []))
    if not claims:
        return {"version": VERSION, "status": "unavailable", "reason": "previous_answer_has_no_claim_reference_targets"}
    presentation = None
    if visible.get("presentation") is not None:
        from pydantic import ValidationError
        from pydantic_core import PydanticCustomError
        from .qa_presentation import Presentation, validate_partition
        try:
            presentation = Presentation.model_validate(visible["presentation"], strict=True)
            validate_partition(presentation, [claim["id"] for claim in visible.get("claims", [])])
        except (ValidationError, PydanticCustomError):
            return {"version": VERSION, "status": "unavailable", "reason": "previous_answer_layout_invalid"}
    answer_sha = digest(result)
    target = {
        "id": "assistant-answer:" + answer_sha, "answer_sha256": answer_sha,
        "brief_revision": previous.get("brief_revision"), "kind": visible["kind"],
        "completion_status": visible.get("completion_status"), "claims": claims,
        "factual_evidence": False, "active_user_instructions": False,
        "purpose": "resolve references to a prior assistant statement, never establish its truth"}
    if presentation is not None:
        target["presentation"] = presentation.model_dump(mode="json")
    if selected_seq is not None:
        target["command_seq"] = selected_seq
    return {"version": VERSION, "status": "currently_authorized", "target": target}


def _task_service(graph: Any) -> Any:
    from .database import TaskService

    database = graph.execution.database
    campaign = getattr(database, "development_campaign_policy", None)
    return TaskService(database, graph.execution.limits,
        new_task_profile_policy=campaign.task_profile_policy if campaign is not None else None)


def add_reference(prompt: dict, reference: dict | None) -> dict:
    value = copy.deepcopy(prompt)
    if reference is None:
        return value
    metadata = {key: reference[key] for key in ("version", "status", "reason", "requested_turn_seq") if key in reference}
    # Reserve a longer marker before ContextBuilder; the post-fit exact status
    # is shorter and cannot silently increase the admitted request size.
    metadata["wire_status"] = "pending_context_admission"
    target = reference.get("target")
    if target is not None:
        metadata.update(target_id=target["id"], answer_sha256=target["answer_sha256"], claims=len(target["claims"]),
            budget_unit="one_complete_prior_assistant_claim_set", factual_evidence=False, active_user_instructions=False)
        if "command_seq" in target:
            metadata["requested_turn_seq"] = target["command_seq"]
        # No previous question/old user instructions are reintroduced. The
        # existing ContextBuilder includes or excludes this WHOLE prior-answer
        # item and records its exact decision; no statement is truncated.
        value["prior_turns"] = [{"assistant_reference_target": copy.deepcopy(target)}]
    value["context_manifest"]["dialogue_reference"] = metadata
    return value


def record_reference_admission(request: dict) -> dict:
    """Annotate the existing builder decision without changing its selection."""
    value = copy.deepcopy(request)
    prompt = json.loads(value["input"][0]["content"])
    reference = prompt.get("context_manifest", {}).get("dialogue_reference")
    if reference is not None:
        reference["wire_status"] = ("included_complete" if prompt.get("prior_turns") else "excluded_by_wire_budget"
            if reference["status"] == "currently_authorized" else "not_available")
        value["input"][0]["content"] = canonical(prompt).decode()
    return value
