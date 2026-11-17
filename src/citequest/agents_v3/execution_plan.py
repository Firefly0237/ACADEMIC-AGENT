"""Versioned approval of the fixed research handlers on the existing Task ledger.

The graph topology is server-owned. A decision can change retrieval text within
the admitted scope; it cannot add handlers, remove reviews or enlarge resources.
"""
from __future__ import annotations

import copy
import re

from pydantic import ValidationError

from citequest.platform_v3.input_text import request_text

from .contracts import AgentError, PlanDecision, digest

PROTOCOL = "approval1"
INTERRUPT_KIND = "execution_plan_approval_required"
MAX_PLAN_VERSIONS = 5
_SHA = re.compile(r"^[0-9a-f]{64}$")
_QUESTION_ID = re.compile(r"^RQ[1-9][0-9]?$")
_EXPERIMENT_ID = re.compile(r"^E[1-9][0-9]?$")


def _scope_sha256(task: dict) -> str:
    brief = task["brief"]
    return digest({
        "selected_papers": brief.get("selected_papers", []),
        "catalog": brief.get("catalog"),
        "research_brief": brief.get("research_brief"),
        "selected_skills": brief.get("selected_skills", []),
        "research_draft_files": brief.get("research_draft_files", []),
        "research_action": brief.get("research_action", "plan"),
        "profile_sha256": task.get("profile_hash") or digest(task["profile"]),
    })


def _decision(value: PlanDecision | dict) -> PlanDecision:
    try:
        return value if isinstance(value, PlanDecision) else PlanDecision.model_validate(value, strict=True)
    except ValidationError:
        raise AgentError("execution_plan_decision_invalid", 422) from None


def _query(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 2000:
        raise AgentError("execution_plan_query_invalid", 422)
    try:
        return request_text(value)
    except ValueError:
        raise AgentError("execution_plan_query_invalid", 422) from None


def _questions(state: dict) -> list[dict]:
    try:
        outline = state["outline"]
        if outline["brief_sha256"] != state["intake"]["brief_sha256"]:
            raise AgentError("execution_plan_outline_binding_changed")
        questions = outline["questions"]
        if not isinstance(questions, list) or not 1 <= len(questions) <= 8:
            raise AgentError("execution_plan_outline_invalid", 422)
        question_ids, experiment_ids = set(), set()
        for question in questions:
            identifier, goal = question["id"], question["question"]
            experiments = question["experiment_ids"]
            if (not isinstance(identifier, str) or not _QUESTION_ID.fullmatch(identifier)
                    or identifier in question_ids or not isinstance(goal, str) or not goal.strip()
                    or not isinstance(experiments, list) or not 1 <= len(experiments) <= 2):
                raise AgentError("execution_plan_outline_invalid", 422)
            question_ids.add(identifier)
            _query(question["retrieval_query"])
            for experiment in experiments:
                if (not isinstance(experiment, str) or not _EXPERIMENT_ID.fullmatch(experiment)
                        or experiment in experiment_ids):
                    raise AgentError("execution_plan_outline_invalid", 422)
                experiment_ids.add(experiment)
        return questions
    except (KeyError, TypeError):
        raise AgentError("execution_plan_outline_invalid", 422) from None


def build_execution_plan(task: dict, state: dict, *, version: int = 1,
                         parent_sha256: str | None = None, queries: dict[str, str] | None = None) -> dict:
    """Describe the existing handlers without mutating the independently reviewed outline."""
    if type(version) is not int or not 1 <= version <= MAX_PLAN_VERSIONS:
        raise AgentError("execution_plan_version_limit")
    if ((version == 1 and parent_sha256 is not None)
            or (version > 1 and (not isinstance(parent_sha256, str) or not _SHA.fullmatch(parent_sha256)))):
        raise AgentError("execution_plan_parent_invalid", 422)
    questions = _questions(state)
    origin = state.get("execution_plan_origin_command_id")
    if not isinstance(origin, str) or not origin or len(origin) > 64:
        raise AgentError("execution_plan_origin_command_invalid", 422)
    if queries is None:
        queries = {}
    known = {item["id"] for item in questions}
    if not isinstance(queries, dict) or not set(queries) <= known:
        raise AgentError("execution_plan_query_target_invalid", 422)
    changes, retrieval, methods = [], [], []
    for question in questions:
        identifier = question["id"]
        query = _query(queries.get(identifier, question["retrieval_query"]))
        if query != question["retrieval_query"]:
            changes.append({"target_id": identifier, "before": question["retrieval_query"], "after": query})
        retrieval.append({"id": "retrieve:" + identifier, "operation": "retrieve_review",
            "target_id": identifier, "goal": question["question"], "query": query, "depends_on": []})
        for experiment in question["experiment_ids"]:
            methods.append({"id": "method:" + experiment, "operation": "design_method",
                "target_id": experiment, "goal": "Design " + experiment + " for " + identifier,
                "depends_on": ["retrieve:" + identifier]})
    if version == 1 and changes:
        raise AgentError("execution_plan_initial_query_changed", 422)
    final_nodes = [{"id": "assemble", "operation": "assemble_review", "target_id": "plan",
        "goal": "Assemble and independently review the complete plan",
        "depends_on": [item["id"] for item in methods]}]
    if task["brief"].get("research_action", "plan") != "experiment":
        final_nodes.append({"id": "write_review", "operation": "write_review", "target_id": "document",
            "goal": "Generate the document and independently review each section and the complete document",
            "depends_on": ["assemble"]})
    return {"protocol": PROTOCOL, "version": version, "task_id": str(task["id"]), "origin_command_id": origin,
        "brief_sha256": state["intake"]["brief_sha256"], "outline_sha256": digest(state["outline"]),
        "scope_sha256": _scope_sha256(task), "parent_sha256": parent_sha256,
        "nodes": [*retrieval, *methods, *final_nodes],
        "resource_requirement": copy.deepcopy(state.get("research_capacity") or {}), "query_changes": changes}


def validate_execution_plan(plan: dict, task: dict, state: dict) -> dict:
    """Rebuild all immutable structure, including scope, dependencies and resource requirements."""
    if not isinstance(plan, dict):
        raise AgentError("execution_plan_invalid", 422)
    try:
        nodes = plan["nodes"]
        if not isinstance(nodes, list) or any(not isinstance(node, dict) for node in nodes):
            raise AgentError("execution_plan_structure_changed")
        queries = {node["target_id"]: node["query"] for node in nodes if node.get("operation") == "retrieve_review"}
        expected = build_execution_plan(task, state, version=plan["version"],
            parent_sha256=plan["parent_sha256"], queries=queries)
        if digest(plan) != digest(expected):
            raise AgentError("execution_plan_structure_changed")
    except (KeyError, TypeError, ValueError):
        raise AgentError("execution_plan_invalid", 422) from None
    return plan


def pending_plan(task: dict) -> dict | None:
    """Read exactly one durable approval interruption; malformed approval records fail closed."""
    result = task.get("result") or {}
    interrupts = result.get("interrupts", [])
    if not isinstance(interrupts, list):
        raise AgentError("execution_plan_interrupt_invalid")
    matches = [item for item in interrupts if isinstance(item, dict) and item.get("kind") == INTERRUPT_KIND]
    if not matches:
        return None
    if len(interrupts) != 1 or len(matches) != 1:
        raise AgentError("execution_plan_interrupt_ambiguous")
    record = matches[0]
    plan = record.get("plan")
    if not isinstance(plan, dict) or record.get("plan_sha256") != digest(plan) or plan.get("protocol") != PROTOCOL:
        raise AgentError("execution_plan_interrupt_invalid")
    return plan


def validate_plan_decision(task: dict, decision: PlanDecision | dict) -> dict:
    """Admission only; the transaction owner also checks revision, ownership and attempts."""
    decision = _decision(decision)
    if (task.get("kind") != "research" or task.get("profile", {}).get("profile_id") != "research-v6"
            or task.get("brief", {}).get("execution_plan_protocol") != PROTOCOL
            or task.get("state") != "waiting_input"):
        raise AgentError("execution_plan_decision_not_admissible")
    plan = pending_plan(task)
    if plan is None or digest(plan) != decision.plan_sha256:
        raise AgentError("execution_plan_revision_conflict")
    if (plan.get("task_id") != str(task["id"]) or plan.get("scope_sha256") != _scope_sha256(task)
            or plan.get("brief_sha256") != task["brief"].get("research_brief", {}).get("sha256")):
        raise AgentError("execution_plan_scope_changed")
    if decision.action == "adjust_queries":
        if type(plan.get("version")) is not int or plan["version"] >= MAX_PLAN_VERSIONS:
            raise AgentError("execution_plan_version_limit")
        current = {node["target_id"]: node["query"] for node in plan.get("nodes", [])
            if node.get("operation") == "retrieve_review"}
        if not set(decision.queries) <= set(current):
            raise AgentError("execution_plan_query_target_invalid", 422)
        if all(current[key] == value for key, value in decision.queries.items()):
            raise AgentError("execution_plan_adjustment_no_change", 422)
    return plan


def apply_query_adjustment(plan: dict, decision: PlanDecision | dict, task: dict, state: dict) -> dict:
    """Create a child proposal; approval of its parent is never inherited."""
    decision = _decision(decision)
    validate_execution_plan(plan, task, state)
    if decision.plan_sha256 != digest(plan):
        raise AgentError("execution_plan_revision_conflict")
    if decision.action != "adjust_queries":
        raise AgentError("execution_plan_adjustment_required", 422)
    current = {node["target_id"]: node["query"] for node in plan["nodes"] if node["operation"] == "retrieve_review"}
    if not set(decision.queries) <= set(current):
        raise AgentError("execution_plan_query_target_invalid", 422)
    if all(current[key] == value for key, value in decision.queries.items()):
        raise AgentError("execution_plan_adjustment_no_change", 422)
    current.update(decision.queries)
    return build_execution_plan(task, state, version=plan["version"] + 1,
        parent_sha256=digest(plan), queries=current)
