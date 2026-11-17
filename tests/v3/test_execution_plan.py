"""Approval, bounded query changes and executable-handler integrity without external calls."""
import copy

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import AgentError, PlanDecision, digest
from citequest.agents_v3.execution_plan import (
    INTERRUPT_KIND,
    MAX_PLAN_VERSIONS,
    apply_query_adjustment,
    build_execution_plan,
    pending_plan,
    validate_execution_plan,
    validate_plan_decision,
)


def inputs():
    task = {"id": "synthetic-task", "kind": "research", "state": "waiting_input", "revision": 3,
        "profile": {"profile_id": "research-v6", "model_calls": 64, "tool_calls": 80},
        "profile_hash": "c" * 64,
        "brief": {"execution_plan_protocol": "approval1", "catalog": "application_papers",
            "research_action": "plan", "research_brief": {"id": "synthetic-brief", "revision": 2, "sha256": "a" * 64},
            "selected_papers": [{"kind": "native_representation", "id": "b" * 64}], "selected_skills": []}}
    state = {"execution_plan_origin_command_id": "synthetic-origin-command",
        "intake": {"brief_sha256": "a" * 64}, "outline": {"brief_sha256": "a" * 64,
        "questions": [{"id": "RQ1", "question": "Which conditions constrain the proposed method?",
            "retrieval_query": "method conditions", "experiment_ids": ["E1", "E2"]},
            {"id": "RQ2", "question": "Does the comparison include negative results?",
             "retrieval_query": "comparison negative results", "experiment_ids": ["E3"]}]},
        "research_capacity": {"required_model_calls": 24, "required_tool_calls": 34, "admitted": True}}
    return task, state


def waiting(task, plan):
    return {**task, "result": {"interrupts": [{"kind": INTERRUPT_KIND, "plan_sha256": digest(plan), "plan": plan}]}}


def decision(plan, *, action="approve", queries=None):
    return PlanDecision(plan_sha256=digest(plan), action=action, queries=queries or {})


def test_plan_describes_real_handler_dependencies_and_preserves_reviewed_input():
    task, state = inputs()
    original = copy.deepcopy((task, state))
    plan = build_execution_plan(task, state)
    assert validate_execution_plan(plan, task, state) == plan
    nodes = {node["id"]: node for node in plan["nodes"]}
    assert nodes["method:E1"]["depends_on"] == nodes["method:E2"]["depends_on"] == ["retrieve:RQ1"]
    assert nodes["method:E3"]["depends_on"] == ["retrieve:RQ2"]
    assert nodes["assemble"]["depends_on"] == ["method:E1", "method:E2", "method:E3"]
    assert nodes["write_review"]["depends_on"] == ["assemble"]
    assert nodes["write_review"]["target_id"] == "document"
    assert {node["operation"] for node in nodes.values()} == {"retrieve_review", "design_method", "assemble_review", "write_review"}
    assert plan["resource_requirement"] == state["research_capacity"]
    assert validate_plan_decision(waiting(task, plan), decision(plan)) == plan
    assert (task, state) == original
    plan["resource_requirement"]["required_tool_calls"] = 1
    assert state["research_capacity"]["required_tool_calls"] == 34


def test_standalone_experiment_stops_before_document_writing():
    task, state = inputs()
    task["brief"]["research_action"] = "experiment"
    plan = build_execution_plan(task, state)
    assert plan["nodes"][-1]["id"] == "assemble"
    assert "write_review" not in {node["operation"] for node in plan["nodes"]}
    assert validate_execution_plan(plan, task, state) == plan
    plan["nodes"].append({"id": "write_review", "operation": "write_review", "target_id": "document",
        "goal": "Generate an additional document", "depends_on": ["assemble"]})
    with pytest.raises(AgentError, match="execution_plan_structure_changed"):
        validate_execution_plan(plan, task, state)


def test_query_adjustment_requires_new_approval_and_keeps_original_scope_and_outline():
    task, state = inputs()
    original = copy.deepcopy(state)
    first = build_execution_plan(task, state)
    amend = decision(first, action="adjust_queries", queries={"RQ1": "method conditions failure cases"})
    assert validate_plan_decision(waiting(task, first), amend) == first
    second = apply_query_adjustment(first, amend, task, state)
    assert second["version"] == 2 and second["parent_sha256"] == digest(first)
    assert second["outline_sha256"] == first["outline_sha256"]
    assert second["scope_sha256"] == first["scope_sha256"]
    assert second["resource_requirement"] == first["resource_requirement"]
    assert second["query_changes"] == [{"target_id": "RQ1", "before": "method conditions",
        "after": "method conditions failure cases"}]
    with pytest.raises(AgentError, match="execution_plan_revision_conflict"):
        validate_plan_decision(waiting(task, second), decision(first))
    assert validate_plan_decision(waiting(task, second), decision(second)) == second
    assert validate_execution_plan(second, task, state) == second
    assert state == original


def test_adjustments_accumulate_and_reverting_is_a_distinct_candidate():
    task, state = inputs()
    first = build_execution_plan(task, state)
    second = apply_query_adjustment(first, decision(first, action="adjust_queries", queries={"RQ1": "failure cases"}), task, state)
    third = apply_query_adjustment(second, decision(second, action="adjust_queries", queries={"RQ2": "opposing findings"}), task, state)
    assert {item["target_id"] for item in third["query_changes"]} == {"RQ1", "RQ2"}
    fourth = apply_query_adjustment(third, decision(third, action="adjust_queries", queries={"RQ1": "method conditions"}), task, state)
    assert fourth["query_changes"] == [{"target_id": "RQ2", "before": "comparison negative results", "after": "opposing findings"}]
    assert fourth["parent_sha256"] == digest(third)


@pytest.mark.parametrize("mutation", [
    lambda plan: plan["nodes"][0].update(operation="unrestricted_tool"),
    lambda plan: plan["nodes"][0].update(depends_on=["assemble"]),
    lambda plan: plan["nodes"][2].update(depends_on=["missing-input"]),
    lambda plan: plan["nodes"].pop(0),
    lambda plan: plan["nodes"].pop(),
    lambda plan: plan["nodes"].append(copy.deepcopy(plan["nodes"][0])),
    lambda plan: plan.update(outline_sha256="d" * 64),
    lambda plan: plan.update(scope_sha256="d" * 64),
    lambda plan: plan.update(brief_sha256="d" * 64),
    lambda plan: plan.update(task_id="different-task"),
    lambda plan: plan.update(origin_command_id="different-command"),
    lambda plan: plan["resource_requirement"].update(required_model_calls=200),
    lambda plan: plan["nodes"][0].update(goal="A different research question"),
    lambda plan: plan.update(unrecognized_authorization=True),
])
def test_modified_graph_constraints_or_resources_are_rejected(mutation):
    task, state = inputs()
    plan = build_execution_plan(task, state)
    mutation(plan)
    with pytest.raises(AgentError):
        validate_execution_plan(plan, task, state)


@pytest.mark.parametrize("change", [
    lambda task: task["brief"]["selected_papers"].clear(),
    lambda task: task["brief"].update(selected_skills=[{"version_id": "another-skill"}]),
    lambda task: task["brief"].update(catalog="standard_metadata"),
    lambda task: task["brief"]["research_brief"].update(revision=3),
    lambda task: task.update(profile_hash="d" * 64),
    lambda task: task["brief"].update(research_action="experiment"),
])
def test_old_approval_cannot_authorize_a_changed_scope(change):
    task, state = inputs()
    plan = build_execution_plan(task, state)
    changed = copy.deepcopy(task)
    change(changed)
    with pytest.raises(AgentError, match="execution_plan_scope_changed"):
        validate_plan_decision(waiting(changed, plan), decision(plan))
    with pytest.raises(AgentError):
        validate_execution_plan(plan, changed, state)


def test_pending_approval_requires_its_typed_decision_and_exact_envelope_hash():
    task, state = inputs()
    plan = build_execution_plan(task, state)
    assert pending_plan(task) is None
    assert pending_plan(waiting(task, plan)) == plan
    mismatched = waiting(task, plan)
    mismatched["result"]["interrupts"][0]["plan_sha256"] = "f" * 64
    with pytest.raises(AgentError, match="execution_plan_interrupt_invalid"):
        pending_plan(mismatched)
    ambiguous = waiting(task, plan)
    ambiguous["result"]["interrupts"].append({"kind": "another_interrupt"})
    with pytest.raises(AgentError, match="execution_plan_interrupt_ambiguous"):
        pending_plan(ambiguous)
    with pytest.raises(AgentError, match="execution_plan_decision_not_admissible"):
        validate_plan_decision({**waiting(task, plan), "state": "running"}, decision(plan))
    with pytest.raises(AgentError, match="execution_plan_revision_conflict"):
        validate_plan_decision(task, decision(plan))


def test_repeated_adjustment_unknown_target_and_version_exhaustion_do_not_progress():
    task, state = inputs()
    first = build_execution_plan(task, state)
    for queries, code in (({"RQ1": "method conditions"}, "adjustment_no_change"),
                          ({"RQ9": "new target"}, "query_target_invalid")):
        amend = decision(first, action="adjust_queries", queries=queries)
        with pytest.raises(AgentError, match=code):
            apply_query_adjustment(first, amend, task, state)
        with pytest.raises(AgentError, match=code):
            validate_plan_decision(waiting(task, first), amend)
    plan = first
    for version in range(2, MAX_PLAN_VERSIONS + 1):
        plan = apply_query_adjustment(plan, decision(plan, action="adjust_queries", queries={"RQ1": "query " + str(version)}), task, state)
    assert plan["version"] == MAX_PLAN_VERSIONS
    assert validate_plan_decision(waiting(task, plan), decision(plan)) == plan
    extra = decision(plan, action="adjust_queries", queries={"RQ1": "one more edit"})
    with pytest.raises(AgentError, match="execution_plan_version_limit"):
        apply_query_adjustment(plan, extra, task, state)
    with pytest.raises(AgentError, match="execution_plan_version_limit"):
        validate_plan_decision(waiting(task, plan), extra)


@pytest.mark.parametrize("changes", [
    {"queries": {"RQ1": "query"}},
    {"action": "adjust_queries"},
    {"action": "adjust_queries", "queries": {"RQ0": "query"}},
    {"action": "adjust_queries", "queries": {"RQ1": " "}},
    {"action": "adjust_queries", "queries": {"RQ1": "text\x00"}},
    {"action": "adjust_queries", "queries": {"RQ1": "a" * 2001}},
    {"action": "add_handler"},
    {"plan_sha256": "not-a-hash"},
    {"action": "approve", "spending_authority": True},
])
def test_decision_cannot_smuggle_queries_or_authority(changes):
    with pytest.raises(ValidationError):
        PlanDecision.model_validate({"plan_sha256": "f" * 64, "action": "approve", **changes}, strict=True)
