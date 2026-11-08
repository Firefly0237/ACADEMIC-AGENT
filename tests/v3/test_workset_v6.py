"""Two bounded graph controls with explicit synthetic role/ledger doubles.

No source business result or academic semantic success is claimed. Full-source
Workset5 has separate controls; these exercise the newly added goal boundary.
"""
from __future__ import annotations

import asyncio
import copy
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from tests.v3.test_qa_v5 import ROOT, FixtureMemory, Tools
from tests.v3.test_qa_v5 import Execution as QAExecution

from citequest.agents_v3.catalog_scope import POLICY
from citequest.agents_v3.contracts import PROFILES, digest, graph_version_for_kind
from citequest.agents_v3.graph import AgentGraph
from citequest.agents_v3.workset_v6 import Workset6Sources, WorksetRequirementsGate


class Execution(QAExecution):
    def __init__(self):
        super().__init__()
        profile = {**PROFILES["workset-v6"].model_dump(), "catalog_scope_policy": POLICY}
        text = "摘要两篇文献；用中文并保留数值条件。"
        self.value.update(kind="summary", profile=profile, profile_hash=digest(profile),
            graph_version=graph_version_for_kind("summary", "workset-v6"))
        self.value["brief"].update(kind="summary", question=text)
        self.turns = [{"turn_id": "U1", "text": text}]


class Provider:
    def __init__(self, execution, *, clarify=False):
        self.execution, self.clarify, self.requests = execution, clarify, []

    async def complete(self, request, key):
        if key in self.execution.known:
            assert request == self.execution.known[key]["request"]
            return copy.deepcopy(self.execution.known[key]["response"])
        prompt = json.loads(request["input"][0]["content"])
        role = request["text"]["format"]["name"].removeprefix("citequest_qa5_")
        self.requests.append((role, copy.deepcopy(prompt)))
        turn, text = prompt["current_user_turn"], prompt["current_user_turn"]["text"]
        def operation(action, targets, category, start=0, end=None):
            return {"action": action, "target_ids": targets, "category": category,
                "spans": [{"turn_id": turn["turn_id"], "start": start, "end": len(text) if end is None else end}]}
        if role == "plan":
            active = [item["id"] for item in prompt["current_requirements"]["active"]]
            ambiguous = self.clarify and turn["turn_id"] == "U1"
            if not active:
                operations = [operation("add", [], "question", 0, text.index("；")), operation("add", [], "format", text.index("；") + 1)] if "；" in text else [operation("add", [], "question")]
                retained = []
            elif turn["turn_id"] == "U2":
                operations, retained = [operation("supersede", ["R1"], "focus")], ["R2"]
            else:
                operations, retained = [operation("withdraw", ["R3"], None, 0, text.index("；")),
                    operation("add", [], "detail", text.index("；") + 1)], ["R2"]
            value = {"requirement_delta": {"parent_sha256": prompt["requirements_parent_sha256"],
                "authority_sha256": prompt["Task_source_authority_sha256"], "turn_id": turn["turn_id"],
                "turn_sha256": turn["sha256"], "retain_ids": active if ambiguous else retained,
                "operations": [] if ambiguous else operations, "intent": "needs_clarification" if ambiguous else "clear",
                "clarification": "请明确这次的摘要重点。" if ambiguous else None}}
        else:
            assert role == "review" and not prompt["selected_user_skills"]
            value = {"candidate_sha256": prompt["requirements_candidate_sha256"], "user_turn_sha256": turn["sha256"],
                "opinions": [{"requirement_id": identifier, "verdict": "correct", "rationale": "EXPLICIT_SYNTHETIC_OPINION"}
                    for identifier in prompt["required_requirement_review_ids"]], "current_turn_coverage": "complete", "decision": "accepted"}
        response = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value)}]}]}
        self.execution.value["model_calls"] += 1
        self.execution.value["input_tokens"] += 100
        self.execution.value["output_tokens"] += 40
        self.execution.known[key] = {"id": uuid4(), "request": copy.deepcopy(request), "response": copy.deepcopy(response)}
        return response


class Workset6GoalControls(unittest.IsolatedAsyncioTestCase):
    def make_graph(self, execution, provider):
        with patch("citequest.agents_v3.graph.PostgresSaver", return_value=InMemorySaver()):
            graph = AgentGraph(execution, provider, Tools(execution), FixtureMemory(), ROOT / "runtime/s08/skills")
        graph.user_turns = lambda: copy.deepcopy(execution.turns)
        return graph

    async def test_same_Task_focus_change_preserves_then_withdraws_only_named_goals(self):
        execution, saver = Execution(), InMemorySaver()
        provider = Provider(execution)
        graph = self.make_graph(execution, provider)
        gate = WorksetRequirementsGate(graph, checkpointer=saver)
        config = {"configurable": {"thread_id": str(execution.task_id)}}
        value = await asyncio.to_thread(gate.workflow.invoke, {"question": execution.turns[-1]["text"]}, config)
        self.assertEqual(value["requirements_state"]["revision"], 0)  # Not source-publication commit.
        accepted = value["workset_requirements"]["reviewed_state"]
        for seq, text in ((2, "保留中文和数值条件，重点改为实验局限。"), (3, "取消局限性重点；保留中文及数值条件，补充方法。")):
            execution.command = {"id": uuid4(), "kind": "followup"}
            execution.value["revision"] += 1
            execution.value["brief"]["question"] = text
            execution.turns.append({"turn_id": "U" + str(seq), "text": text})
            value = await asyncio.to_thread(gate.workflow.invoke, {"question": text, "requirements_state": accepted}, config)
            self.assertEqual(value["requirements_state"], accepted)
            accepted = value["workset_requirements"]["reviewed_state"]
            _, anchor = graph.projection(include_user_skill_body=False)
            anchor["required_user_turn_refs"] = [{"id": turn["turn_id"], "sha256": digest(turn["text"])} for turn in execution.turns]
            anchor["typed_requirements"] = value["workset_requirements"]
            source = Workset6Sources(graph)
            author, manifest = source.project(anchor)
            reviewer, _ = source.project(anchor, reviewer=True)
            self.assertEqual(author["current_requirements"], reviewer["current_requirements"])
            self.assertNotIn("required_user_turns", author)
            self.assertEqual(manifest["requirements_accepted_revision"], seq)
        self.assertEqual([item["id"] for item in accepted["requirements"] if item["state"] == "active"], ["R2", "R4"])
        self.assertEqual([item["state"] for item in accepted["requirements"]], ["superseded", "active", "withdrawn", "active"])
        self.assertEqual(execution.value["model_calls"], 6)
        self.assertEqual(execution.content_revision_count(), 0)

    async def test_ambiguous_turn_waits_and_resumes_same_Task_without_committing_guess(self):
        execution, saver = Execution(), InMemorySaver()
        provider = Provider(execution, clarify=True)
        graph = self.make_graph(execution, provider)
        gate = WorksetRequirementsGate(graph, checkpointer=saver)
        config = {"configurable": {"thread_id": str(execution.task_id)}}
        waiting = await asyncio.to_thread(gate.workflow.invoke, {"question": execution.turns[-1]["text"]}, config)
        self.assertTrue(waiting.get("__interrupt__"))
        self.assertEqual(waiting["requirements_state"]["revision"], 0)
        self.assertEqual(execution.value["model_calls"], 1)
        self.assertEqual(execution.content_revision_count(), 0)
        text = "只总结实验局限，保留实际数值与适用条件。"
        execution.command = {"id": uuid4(), "kind": "resume"}
        execution.value["revision"] += 1
        execution.value["brief"]["question"] = text
        execution.turns.append({"turn_id": "U2", "text": text})
        value = await asyncio.to_thread(gate.workflow.invoke, Command(resume={"message": text}), config)
        self.assertEqual(value["requirements_state"]["revision"], 0)
        self.assertEqual([turn["turn_id"] for turn in value["workset_requirements"]["reviewed_state"]["turns"]], ["U2"])
        self.assertEqual(execution.value["model_calls"], 3)
        self.assertEqual(execution.content_revision_count(), 0)  # Normal clarification is not a failed-content repair.


if __name__ == "__main__":
    unittest.main()
