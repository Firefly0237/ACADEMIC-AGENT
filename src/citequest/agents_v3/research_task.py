"""Persistent parent graph: versioned research planning then school writing.

Both child graphs inherit the parent's official PostgresSaver namespace and
share its Execution, SDK client, MCP connection and lifetime resource ledger.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, StateSnapshot

if TYPE_CHECKING:
    from .graph import AgentGraph

from .research_runtime import current_planning_input
from .research_workflow import ResearchState, ResearchWorkflow


class TaskState(ResearchState, total=False):
    command_id: str
    planning_state: dict
    result: dict


class ResearchTaskGraph:
    strict_identity_binding = False
    review_outline = False

    def __init__(self, host: AgentGraph) -> None:
        self.host = host
        self.planner = ResearchWorkflow(host, lambda: current_planning_input(host.execution), checkpointer=True,
            strict_identity_binding=self.strict_identity_binding, review_outline=self.review_outline)
        builder = StateGraph(TaskState)
        builder.add_node("research_planning", self.planner.workflow if self.planner.shared_revisions()
            else host.observed_node("research_planning", self.plan))
        builder.add_node("school_writing", host.observed_node("school_writing", self.write))
        builder.add_node("intake_suggestions", host.observed_node("research_intake_suggestions", self.help))
        builder.add_node("draft_intake", host.observed_node("research_draft_intake", self.extract))
        builder.add_conditional_edges(START, lambda state: {"suggest": "intake_suggestions", "extract": "draft_intake"}.get(
            self.host.execution.snapshot()["brief"].get("research_action"), "research_planning"),
            ["intake_suggestions", "draft_intake", "research_planning"])
        builder.add_edge("intake_suggestions", END)
        builder.add_edge("draft_intake", END)
        builder.add_conditional_edges("research_planning", lambda state: "school_writing"
            if state["result"].get("status") == "structured_plan_ready" else END, ["school_writing", END])
        builder.add_edge("school_writing", END)
        self.workflow = builder.compile(checkpointer=host.saver)
        self.config = {"configurable": {"thread_id": str(host.execution.task_id)}, "recursion_limit": 384}

    def plan(self, state: dict) -> dict:
        # This is a child invocation, not a second root graph. The parent's
        # invocation context supplies its Saver namespace and resume command.
        # Calling the standalone driver's get_state(config) here would resolve
        # checkpointer=True as a root and is explicitly rejected by LangGraph.
        value = self.planner.workflow.invoke({"run_id": str(self.host.execution.run_id)}, durability="sync")
        return {"planning_state": value, "result": self.planner.project_result(value, value.get("result", {}))}

    def help(self, state: dict) -> dict:
        from .research_help import suggest
        return {"result": suggest(self.planner)}

    def extract(self, state: dict) -> dict:
        from .research_intake import DraftIntakeWorkflow
        return {"result": DraftIntakeWorkflow(self.planner, checkpointer=True).run()}

    def write(self, state: dict) -> dict:
        from .school_document import SchoolDocumentWorkflow
        value = SchoolDocumentWorkflow(self.planner, checkpointer=True,
            reference_preflight=getattr(self.host, "reference_preflight", None)).run(state["result"])
        return {"result": value["result"]}

    @staticmethod
    def interrupted_planning_state(snapshot: StateSnapshot) -> dict:
        # A child interrupted before returning to its parent. Read its Saver
        # state instead of publishing a stale plan from a previous command.
        best = snapshot.values if isinstance(snapshot.values, dict) else {}
        for task in snapshot.tasks:
            child = getattr(task, "state", None)
            if isinstance(child, StateSnapshot):
                candidate = ResearchTaskGraph.interrupted_planning_state(child)
                if candidate.get("intake"):
                    best = candidate
        return best

    def run(self) -> dict:
        previous = self.workflow.get_state(self.config)
        command = self.host.execution.command
        value = (Command(resume={"versioned_brief_must_be_reloaded": True})
            if previous.next and command["kind"] == "resume" else None
            if previous.next else {"command_id": str(command["id"])})
        result = self.workflow.invoke(value, self.config, durability="sync")
        snapshot = self.workflow.get_state(self.config, subgraphs=True)
        checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
        if result.get("__interrupt__"):
            state = self.interrupted_planning_state(snapshot)
            output = self.planner.project_result(state, {"status": "waiting_input",
                "publication_status": "not_evaluated", "interrupts": [item.value for item in result["__interrupt__"]]})
            self.host.execution.finish("waiting_input", result=output, checkpoint_id=checkpoint_id)
        else:
            output = result["result"]
            from .user_skills import validate_task_skills
            validate_task_skills(self.host.execution.database, self.host.execution.snapshot())
            if output.get("publication_status") == "validated_ai_review":
                task = self.host.execution.snapshot()
                self.host.memory.validate_result_context(task, output)
                from .contracts import canonical
                from .research_runtime import validate_wire_intake
                validate_wire_intake(self.host.execution, {"input": [{"content": canonical({
                    "context_manifest": output["context_manifest"]}).decode()}]})
            rejected = output.get("publication_status") == "rejected"
            complete = output.get("status") in {"school_document_ready", "research_suggestions_ready", "research_draft_candidates_ready"}
            if output.get("completion_status", "").startswith("partial"):
                complete = False
            self.host.execution.finish("waiting_input" if output.get("interrupts") else "failed" if rejected else "succeeded" if complete else "partial",
                result=output, checkpoint_id=checkpoint_id,
                error="research_review_rejected" if rejected else None)
        return result


class ResearchTaskGraphV4(ResearchTaskGraph):
    """Explicit new Task version; no change to legacy Research3 wire contracts."""

    strict_identity_binding = True


class ResearchTaskGraphV5(ResearchTaskGraphV4):
    """Reviewed coordinator output before downstream research or user interrupts."""

    review_outline = True
