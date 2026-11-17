"""Recovery at the real runner's graph-complete / business-commit boundary."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from citequest.agents_v3.research_task import ResearchTaskGraph, TaskState

_spec = importlib.util.spec_from_file_location(
    "execution_plan_test_support", Path(__file__).with_name("test_execution_plan_workflow.py"))
_support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_support)


def completion_graph(boundary, saver, *, finalize=None):
    """Production approval/retrieval nodes with an explicitly synthetic final result."""
    child = StateGraph(TaskState)
    child.add_node("execution_plan_prepare", boundary.execution_plan_prepare)
    child.add_node("execution_plan_approval", boundary.execution_plan_approval)
    child.add_node("literature_search", boundary.literature_search)
    child.add_node("finish_fixture", finalize or (lambda _state: {"result": {
        "status": "partial", "completion_status": "partial_acceptance_fixture",
        "publication_status": "not_evaluated"}}))
    child.add_edge(START, "execution_plan_prepare")
    child.add_edge("execution_plan_prepare", "execution_plan_approval")
    child.add_conditional_edges("execution_plan_approval", lambda state: state["next"],
        ["execution_plan_approval", "literature_search"])
    child.add_edge("literature_search", "finish_fixture")
    child.add_edge("finish_fixture", END)
    parent = StateGraph(TaskState)
    parent.add_node("research_planning", child.compile(checkpointer=True))
    parent.add_edge(START, "research_planning")
    parent.add_edge("research_planning", END)
    return parent.compile(checkpointer=saver)


def runner(boundary, saver, config, finish, *, finalize=None):
    value = ResearchTaskGraph.__new__(ResearchTaskGraph)
    value.host, value.config = boundary.host, config
    value.host.execution.finish = finish
    value.host.execution.database = None
    # Do not substitute the graph driver or recovery decision under test.
    value.planner = SimpleNamespace(project_result=lambda _state, result: result)
    value.workflow = completion_graph(boundary, saver, finalize=finalize)
    return value


def test_completed_approval_graph_is_not_restarted_after_business_finish_interruption(monkeypatch):
    monkeypatch.setattr("citequest.agents_v3.user_skills.validate_task_skills", lambda *_args: None)
    ledger, saver = _support.TaskLedger(), InMemorySaver()
    config = {"configurable": {"thread_id": str(ledger.task_id)}}
    source = ledger.worker(_support.task_command())
    first = completion_graph(source, saver).invoke(ledger.state(), config, durability="sync")
    plan = _support.interrupted_plan(first)
    approval = _support.task_command("resume", plan)

    def interrupt_business_finish(*_args, **_kwargs):
        raise RuntimeError("synthetic_stop_after_graph_commit_before_business_commit")

    initial = runner(ledger.worker(approval), saver, config, interrupt_business_finish)
    with pytest.raises(RuntimeError, match="synthetic_stop_after_graph_commit"):
        initial.run()
    committed = initial.workflow.get_state(config)
    assert not committed.next and committed.values["execution_plan_decision"]["command_id"] == str(approval["id"])
    calls, tool_count = list(ledger.calls), ledger.task["tool_calls"]

    completed = []
    recovered = runner(ledger.worker(approval), saver, config,
        lambda state, **kwargs: completed.append((state, kwargs["result"])))
    result = recovered.run()
    assert "__interrupt__" not in result
    assert result == committed.values
    assert completed == [("partial", committed.values["result"])]
    assert ledger.calls == calls and ledger.task["tool_calls"] == tool_count


def test_approval_command_resumes_unfinished_graph_after_known_tool_outcome_without_repeating_tools(monkeypatch):
    monkeypatch.setattr("citequest.agents_v3.user_skills.validate_task_skills", lambda *_args: None)
    ledger, saver = _support.TaskLedger(), InMemorySaver()
    config = {"configurable": {"thread_id": str(ledger.task_id)}}
    source = ledger.worker(_support.task_command())
    first = completion_graph(source, saver).invoke(ledger.state(), config, durability="sync")
    approval = _support.task_command("resume", _support.interrupted_plan(first))

    def stop_before_result(_state):
        raise RuntimeError("synthetic_stop_after_known_tool_outcome")

    completed = []
    def finish(state, **kwargs):
        completed.append((state, kwargs["result"]))
    initial = runner(ledger.worker(approval), saver, config, finish, finalize=stop_before_result)
    with pytest.raises(RuntimeError, match="synthetic_stop_after_known_tool_outcome"):
        initial.run()
    assert initial.workflow.get_state(config).next
    calls, tool_count = list(ledger.calls), ledger.task["tool_calls"]
    assert len(calls) == 2 and completed == []
    recovered = runner(ledger.worker(approval), saver, config, finish)
    result = recovered.run()
    assert "__interrupt__" not in result and completed[0][0] == "partial"
    assert ledger.calls == calls and ledger.task["tool_calls"] == tool_count

    # A genuinely new command still starts a new planning/approval cycle.
    following = _support.task_command("followup")
    new_turn = runner(ledger.worker(following), saver, config,
        lambda state, **kwargs: completed.append((state, kwargs["result"])))
    restarted = new_turn.run()
    proposal = _support.interrupted_plan(restarted)
    assert proposal["origin_command_id"] == str(following["id"])
    assert completed[-1][0] == "waiting_input"
    assert ledger.calls == calls and ledger.task["tool_calls"] == tool_count
