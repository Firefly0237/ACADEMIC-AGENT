"""Actual nested Graph interrupts and cold replay of the bounded approval boundary."""
from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.execution_plan import INTERRUPT_KIND
from citequest.agents_v3.research_workflow import PlanningInput, ResearchState, ResearchWorkflow


class TaskLedger:
    """One persistent synthetic task shared by independently reconstructed workers."""

    def __init__(self):
        self.task_id = uuid4()
        source = {"kind": "native_representation", "id": "b" * 64}
        self.intake = PlanningInput(brief_id=str(uuid4()), brief_revision=1, brief_sha256="a" * 64,
            user_confirmed={"topic": "A bounded synthetic question"}, explicit_unknowns={},
            selected_sources=[source]).model_dump(mode="json")
        self.task = {"id": self.task_id, "kind": "research", "owner_id": uuid4(), "project_id": uuid4(),
            "profile_hash": "c" * 64, "model_calls": 2, "tool_calls": 7,
            "input_tokens": 125, "output_tokens": 50, "known_ncny": 17, "reserved_ncny": 0,
            "profile": {"profile_id": "research-v6", "model_calls": 64, "tool_calls": 80},
            "brief": {"execution_plan_protocol": "approval1", "catalog": "application_papers",
                "research_action": "experiment", "research_brief": {"id": self.intake["brief_id"],
                    "revision": 1, "sha256": self.intake["brief_sha256"]}, "selected_papers": [source],
                "research_draft_files": [], "selected_skills": []}}
        self.outline = {"brief_sha256": self.intake["brief_sha256"], "questions": [{"id": "RQ1",
            "question": "Under which conditions does the proposed method fail?", "retrieval_query": "method conditions",
            "experiment_ids": ["E1"], "user_fields": ["topic"], "expected_outputs": ["An unexecuted design"]}],
            "innovation_hypotheses": [], "uncovered_user_requests": []}
        text = "Explicit synthetic evidence with a bounded condition."
        self.evidence = {"ref_id": "fixture:" + digest(text), "text": text, "source_sha256": digest(text)}
        self.calls, self.events, self.tool_cache = [], [], {}

    def state(self):
        return {"intake": copy.deepcopy(self.intake), "outline": copy.deepcopy(self.outline),
            "research_capacity": {"required_model_calls": 8, "required_tool_calls": 3, "admitted": True},
            "hard_memory_refs": [], "rq_index": 0, "evidence": []}

    def worker(self, command):
        execution = SimpleNamespace(command=command, task_id=self.task_id, run_id=uuid4(),
            snapshot=lambda: copy.deepcopy(self.task), progress=lambda phase, state: self.events.append((phase, state)))
        host = SimpleNamespace(execution=execution, memory=SimpleNamespace(current=lambda *_args: []), wait=asyncio.run)
        host.tools = ToolLedger(self)
        boundary = ResearchWorkflow.__new__(ResearchWorkflow)
        boundary.host, boundary.brief_loader = host, lambda: copy.deepcopy(self.intake)
        return boundary


class ToolLedger:
    def __init__(self, ledger):
        self.ledger = ledger

    async def call(self, name, arguments, *, logical_key):
        if logical_key in self.ledger.tool_cache:
            return copy.deepcopy(self.ledger.tool_cache[logical_key])
        if name == "search_evidence":
            result = {"evidence": [self.ledger.evidence]}
        elif name == "read_evidence" and arguments == {"ref_id": self.ledger.evidence["ref_id"]}:
            result = self.ledger.evidence
        else:
            raise AssertionError("Unexpected tool at the synthetic retrieval boundary")
        self.ledger.task["tool_calls"] += 1
        self.ledger.calls.append({"name": name, "arguments": copy.deepcopy(arguments), "logical_key": logical_key})
        self.ledger.tool_cache[logical_key] = copy.deepcopy(result)
        return copy.deepcopy(result)


def task_command(kind="start", plan=None, *, action="approve", queries=None):
    payload = {"operation_id": str(uuid4()), "message": ""}
    if plan is not None:
        payload["plan_decision"] = {"plan_sha256": digest(plan), "action": action,
            **({"queries": queries} if queries is not None else {})}
    return {"id": uuid4(), "kind": kind, "payload": payload}


def nested_graph(boundary, saver):
    # Use production node methods. Only the untested downstream model roles are
    # outside this bounded harness; retrieval still goes through the real tool method.
    child = StateGraph(ResearchState)
    child.add_node("execution_plan_prepare", boundary.execution_plan_prepare)
    child.add_node("execution_plan_approval", boundary.execution_plan_approval)
    child.add_node("literature_search", boundary.literature_search)
    child.add_edge(START, "execution_plan_prepare")
    child.add_edge("execution_plan_prepare", "execution_plan_approval")
    child.add_conditional_edges("execution_plan_approval", lambda state: state["next"],
        ["execution_plan_approval", "literature_search"])
    child.add_edge("literature_search", END)
    parent = StateGraph(ResearchState)
    parent.add_node("research_planning", child.compile(checkpointer=True))
    parent.add_edge(START, "research_planning")
    parent.add_edge("research_planning", END)
    return parent.compile(checkpointer=saver)


def interrupted_plan(result):
    assert len(result["__interrupt__"]) == 1
    record = result["__interrupt__"][0].value
    assert record["kind"] == INTERRUPT_KIND
    assert record["plan_sha256"] == digest(record["plan"])
    return record["plan"]


def test_nested_cold_recovery_requires_new_approval_before_adjusted_query_reaches_tools():
    ledger, saver = TaskLedger(), InMemorySaver()
    config = {"configurable": {"thread_id": str(ledger.task_id)}}
    initial_counts = {key: ledger.task[key] for key in ("model_calls", "tool_calls", "input_tokens", "output_tokens", "known_ncny")}
    original_outline = copy.deepcopy(ledger.outline)
    start = task_command()
    first_graph = nested_graph(ledger.worker(start), saver)
    first = interrupted_plan(first_graph.invoke(ledger.state(), config, durability="sync"))
    assert first["version"] == 1 and first["origin_command_id"] == str(start["id"])
    assert ledger.calls == [] and ledger.task["tool_calls"] == initial_counts["tool_calls"]

    adjustment = task_command("resume", first, action="adjust_queries", queries={"RQ1": "method failure cases bounded conditions"})
    adjusted_graph = nested_graph(ledger.worker(adjustment), saver)
    second = interrupted_plan(adjusted_graph.invoke(Command(resume={"admitted": True}), config, durability="sync"))
    assert second["version"] == 2 and second["parent_sha256"] == digest(first)
    assert second["origin_command_id"] == first["origin_command_id"]
    assert second["query_changes"] == [{"target_id": "RQ1", "before": "method conditions", "after": "method failure cases bounded conditions"}]
    assert ledger.calls == [] and ledger.task["tool_calls"] == initial_counts["tool_calls"]

    approval = task_command("resume", second)
    final_graph = nested_graph(ledger.worker(approval), saver)
    result = final_graph.invoke(Command(resume={"admitted": True}), config, durability="sync")
    assert "__interrupt__" not in result
    assert result["outline"] == original_outline
    assert result["execution_plan"]["outline_sha256"] == digest(original_outline)
    assert result["execution_plan_decision"]["command_id"] == str(approval["id"])
    assert result["execution_plan_decision"]["plan_sha256"] == digest(second)
    assert [(call["name"], call["arguments"]) for call in ledger.calls] == [
        ("search_evidence", {"query": "method failure cases bounded conditions", "limit": 1}),
        ("read_evidence", {"ref_id": ledger.evidence["ref_id"]})]
    assert all(call["logical_key"].startswith(str(approval["id"]) + ":") for call in ledger.calls)
    assert ledger.task["tool_calls"] == initial_counts["tool_calls"] + 2
    assert all(ledger.task[key] == value for key, value in initial_counts.items() if key != "tool_calls")
    assert ledger.outline == original_outline
    retained = copy.deepcopy((ledger.task, ledger.calls, result))
    assert final_graph.invoke(None, config, durability="sync") == retained[2]
    assert (ledger.task, ledger.calls) == retained[:2]


@pytest.mark.parametrize("invalid", ["missing_decision", "stale_hash"])
def test_worker_cannot_bypass_approval_with_missing_or_stale_decision(invalid):
    ledger, saver = TaskLedger(), InMemorySaver()
    config = {"configurable": {"thread_id": str(ledger.task_id)}}
    first = interrupted_plan(nested_graph(ledger.worker(task_command()), saver).invoke(ledger.state(), config))
    command = task_command("resume", first if invalid == "stale_hash" else None)
    if invalid == "stale_hash":
        command["payload"]["plan_decision"]["plan_sha256"] = "f" * 64
    graph = nested_graph(ledger.worker(command), saver)
    code = "execution_plan_revision_conflict" if invalid == "stale_hash" else "execution_plan_decision_required"
    with pytest.raises(AgentError, match=code):
        graph.invoke(Command(resume={"approved": True}), config)
    assert ledger.calls == [] and ledger.task["tool_calls"] == 7 and ledger.task["model_calls"] == 2


def test_retrieval_itself_rejects_unapproved_plan_even_outside_the_graph_route():
    ledger = TaskLedger()
    boundary = ledger.worker(task_command())
    state = ledger.state()
    state.update(boundary.execution_plan_prepare(state))
    with pytest.raises(AgentError, match="execution_plan_approval_required"):
        boundary.literature_search(state)
    assert ledger.calls == [] and ledger.task["tool_calls"] == 7


def approved_state(ledger):
    source = ledger.worker(task_command())
    state = ledger.state()
    state.update(source.execution_plan_prepare(state))
    approval = task_command("resume", state["execution_plan"])
    state["execution_plan_decision"] = {"action": "approve", "plan_sha256": digest(state["execution_plan"]),
        "command_id": str(approval["id"]), "operation_id": approval["payload"]["operation_id"]}
    return state, approval


def test_observed_boundary_returns_to_intake_before_reusing_another_commands_approval():
    ledger = TaskLedger()
    state, _approval = approved_state(ledger)
    boundary = ledger.worker(task_command("resume"))
    called = []
    observed = boundary.observe("literature_search", lambda _state: called.append(True))
    assert observed(state) == {"next": "intake"}
    assert called == [] and ledger.calls == []
    assert ledger.events == [("research_literature_search", "started"), ("research_literature_search", "completed")]


def test_observed_boundary_reloads_changed_brief_before_any_execution():
    ledger = TaskLedger()
    state, approval = approved_state(ledger)
    boundary = ledger.worker(approval)
    ledger.intake["brief_revision"] = 2
    ledger.intake["brief_sha256"] = "d" * 64
    ledger.intake["user_confirmed"]["topic"] = "A changed goal requiring a fresh plan"
    called = []
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": str(ledger.task_id)}}

    def observed_graph(worker):
        builder = StateGraph(ResearchState)
        builder.add_node("literature_search", worker.observe("literature_search", lambda _state: called.append(True)))
        builder.add_edge(START, "literature_search")
        builder.add_edge("literature_search", END)
        return builder.compile(checkpointer=saver)

    first = observed_graph(boundary).invoke(state, config)
    assert first["__interrupt__"][0].value["kind"] == "scope_sync_required"
    assert called == [] and ledger.calls == [] and ledger.task["tool_calls"] == 7

    # Simulate the explicit Task command synchronizing the newly authorized
    # reference. A fresh worker must still discard the old execution approval.
    reference = {"id": ledger.intake["brief_id"], "revision": 2, "sha256": "d" * 64}
    ledger.task["brief"]["research_brief"] = reference
    synchronized = task_command("resume")
    synchronized["payload"].update(message="Continue from the synchronized Brief.", research_brief=reference)
    resumed = observed_graph(ledger.worker(synchronized)).invoke(Command(resume={"synchronized": True}), config)
    assert "__interrupt__" not in resumed and resumed["next"] == "intake"
    assert called == [] and ledger.calls == [] and ledger.task["tool_calls"] == 7


def test_unsynchronized_source_selection_interrupts_before_any_execution():
    ledger = TaskLedger()
    state, approval = approved_state(ledger)
    boundary = ledger.worker(approval)
    ledger.intake["selected_sources"] = [{"kind": "native_representation", "id": "d" * 64}]
    ledger.intake["brief_revision"] = 2
    graph = StateGraph(ResearchState)
    graph.add_node("literature_search", boundary.observe("literature_search", boundary.literature_search))
    graph.add_edge(START, "literature_search")
    graph.add_edge("literature_search", END)
    result = graph.compile(checkpointer=InMemorySaver()).invoke(state,
        {"configurable": {"thread_id": str(ledger.task_id)}})
    assert result["__interrupt__"][0].value["kind"] == "scope_sync_required"
    assert ledger.calls == [] and ledger.task["tool_calls"] == 7
