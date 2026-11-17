"""Shared official graph with explicit synthetic roles/tools/ledger only.

These controls test ordering and boundaries, never academic semantic accuracy.
They require the existing locked Agent runtime; no services are started here.
"""
from __future__ import annotations

import asyncio
import copy
import json
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from langgraph.checkpoint.memory import InMemorySaver
from tests.v3.test_qa_v5 import ROOT, Cursor, FixtureMemory
from tests.v3.test_qa_v5 import Execution as QAExecution
from tests.v3.test_qa_v5 import Provider as QAProvider
from tests.v3.test_qa_v5 import Tools as QATools

from citequest.agents_v3.catalog_scope import POLICY
from citequest.agents_v3.contracts import PROFILES, AgentError, digest, graph_version_for_kind
from citequest.agents_v3.graph import AgentGraph
from citequest.agents_v3.stance_v2 import StancePlanV2
from citequest.agents_v3.tools import require_atomic_stance_review


class Execution(QAExecution):
    def __init__(self):
        super().__init__()
        profile = {**PROFILES["stance-v2"].model_dump(), "catalog_scope_policy": POLICY}
        self.value.update(kind="stance", profile=profile, profile_hash=digest(profile),
            graph_version=graph_version_for_kind("stance", "stance-v2"))
        self.value["brief"].update(kind="stance", requested_direction="support", question="检验该方法优于对照的观点。")
        self.turns = [{"turn_id": "U1", "text": self.value["brief"]["question"]}]

    def execute(self, query, args):
        if "SELECT id,kind,status,request_hash,usage,error_code,request,response" in query:
            wanted = {str(identifier) for identifier in args[1]}
            return Cursor([{"id": row["id"], "kind": "model", "status": "settled",
                "request_hash": digest(row["request"]), "usage": {"input": 100, "output": 40, "cached": 0},
                "error_code": None, "request": copy.deepcopy(row["request"]), "response": copy.deepcopy(row["response"]),
                "logical_key": logical_key}
                for logical_key, row in self.known.items() if str(row["id"]) in wanted])
        return super().execute(query, args)


class Tools(QATools):
    def __init__(self, execution, *, empty=False):
        super().__init__(execution, empty=empty)
        self.trace = []

    async def call(self, name, args, *, logical_key, planning_context):
        self.assert_review(planning_context)
        if logical_key not in self.cache:
            self.trace.append("tool:" + name)
        # The existing source fixture has a QA-only four-key assertion. The
        # real seven-key stance gateway is independently checked just above.
        qa_binding = {key: planning_context[key] for key in (
            "planner_attempt_id", "planner_request_sha256", "candidate_sha256", "plan_sha256")}
        return await super().call(name, args, logical_key=logical_key, planning_context=qa_binding)

    def assert_review(self, binding):
        assert len(binding) == 7
        require_atomic_stance_review(self.execution, binding)


class Provider(QAProvider):
    def __init__(self, execution, tools, *, reject_plan_once=False, wrong_statement_once=False, **modes):
        super().__init__(execution, tools, **modes)
        self.calls["plan_review"] = 0
        self.reject_plan_once, self.wrong_statement_once = reject_plan_once, wrong_statement_once

    async def complete(self, payload, key):
        if key in self.execution.known:
            assert payload == self.execution.known[key]["request"]
            return copy.deepcopy(self.execution.known[key]["response"])
        role = payload["text"]["format"]["name"].removeprefix("citequest_qa5_")
        prompt = json.loads(payload["input"][0]["content"])
        self.tools.trace.append("model:" + role)
        if role == "plan_review":
            self.calls[role] += 1
            self.requests.append({"request": copy.deepcopy(payload), "key": key, "role": role})
            self.execution.value["model_calls"] += 1
            self.execution.value["input_tokens"] += 100
            self.execution.value["output_tokens"] += 40
            passed = not self.reject_plan_once or self.calls[role] > 1
            value = {"claims": [{"claim_id": claim["id"], "faithful_normalization": passed,
                "no_new_facts_or_quantifiers": True, "conditions_preserved": True,
                "explanation": "EXPLICIT_SYNTHETIC_PLAN_REVIEW"} for claim in prompt["claim_plan"]["claims"]],
                "all_user_requests_accounted_for": True, "latest_turn_and_explicit_direction_respected": True,
                "unresolved_reference": False, "explanation": "EXPLICIT_SYNTHETIC_OPINION_NOT_ACADEMIC_GOLD"}
        else:
            response = await super().complete(payload, key)
            value = json.loads(response["output"][0]["content"][0]["text"])
            if role == "plan":
                turn = prompt["current_user_turn"]
                value["claim_plan"] = {"claims": [{"id": "C1", "anchor": {"turn_id": turn["turn_id"], "quote": turn["text"]},
                    "atomic_statement": "Explicit synthetic proposition.", "retrieval_question": turn["text"],
                    "requested_direction": self.execution.value["brief"]["requested_direction"], "conditions": []}],
                    "unplanned_requests": []} if value["action"] != "clarify" else None
            elif role == "author":
                value.update(kind="stance", requested_direction=self.execution.value["brief"]["requested_direction"])
                value["claims"][0]["statement"] = ("Wrong replacement proposition." if
                    self.wrong_statement_once and self.calls[role] == 1 else "Explicit synthetic proposition.")
        result = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value)}]}]}
        row = self.execution.known.get(key)
        self.execution.known[key] = {"id": row["id"] if row else uuid4(), "request": copy.deepcopy(payload), "response": copy.deepcopy(result)}
        return result


class StanceV2Controls(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, *, execution=None, saver=None, tools=None, **modes):
        execution, saver = execution or Execution(), saver or InMemorySaver()
        tools = tools or Tools(execution)
        provider = Provider(execution, tools, **modes)
        with patch("citequest.agents_v3.graph.PostgresSaver", return_value=saver), patch.object(AgentGraph, "user_turns", lambda _: execution.turns):
            graph = AgentGraph(execution, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            result = await asyncio.to_thread(graph.run)
        return execution, provider, tools, result

    async def test_independent_atomic_plan_is_accepted_before_first_tool_boundary(self):
        execution, provider, tools, result = await self.run_case()
        self.assertEqual(provider.calls, {"plan": 1, "plan_review": 1, "author": 1, "review": 1})
        self.assertLess(tools.trace.index("model:plan_review"), next(i for i, item in enumerate(tools.trace) if item.startswith("tool:")))
        self.assertEqual(result["result"]["kind"], "stance")
        self.assertEqual(result["result"]["requested_direction"], "support")
        self.assertEqual(result["requirements_state"]["revision"], 1)
        self.assertEqual(execution.finishes[-1]["state"], "succeeded")

    async def test_same_Task_switches_explicit_direction_and_keeps_durable_requirements(self):
        execution, saver = Execution(), InMemorySaver()
        tools = Tools(execution)
        await self.run_case(execution=execution, saver=saver, tools=tools)
        text = "保留上述命题，改为寻找反驳证据。"
        execution.command = {"id": uuid4(), "kind": "followup"}
        execution.value["revision"] += 1
        execution.value["brief"].update(question=text, requested_direction="refute")
        execution.turns.append({"turn_id": "U2", "text": text})
        _, _, _, result = await self.run_case(execution=execution, saver=saver, tools=tools)
        self.assertEqual(result["result"]["requested_direction"], "refute")
        self.assertEqual(result["requirements_state"]["revision"], 2)
        self.assertEqual(execution.value["model_calls"], 8)
        self.assertEqual(execution.content_revision_count(), 0)

    async def test_unfaithful_plan_replans_and_changed_author_repair_share_original_credits(self):
        execution, provider, tools, _ = await self.run_case(reject_plan_once=True, wrong_statement_once=True)
        self.assertEqual(execution.content_revision_count(), 2)
        self.assertEqual(provider.calls, {"plan": 2, "plan_review": 2, "author": 2, "review": 1})
        self.assertEqual(tools.trace[:4], ["model:plan", "model:plan_review", "model:plan", "model:plan_review"])

    async def test_empty_bounded_retrieval_is_honest_partial_not_a_forced_support(self):
        execution = Execution()
        execution, _, _, result = await self.run_case(execution=execution, tools=Tools(execution, empty=True))
        self.assertEqual(result["result"]["claims"][0]["status"], "insufficient")
        self.assertEqual(result["result"]["completion_status"], "partial_stance_evidence_insufficient")
        self.assertEqual(result["result"]["citations"], [])
        self.assertEqual(execution.finishes[-1]["state"], "partial")

    async def test_tool_gateway_rejects_wrong_review_binding_before_any_tool(self):
        execution, _provider, tools, _ = await self.run_case()
        planner = next(row for row in execution.known.values() if row["request"]["text"]["format"]["name"] == "citequest_qa5_plan")
        reviewer = next(row for row in execution.known.values() if row["request"]["text"]["format"]["name"] == "citequest_qa5_plan_review")
        plan = StancePlanV2.model_validate_json(planner["response"]["output"][0]["content"][0]["text"], strict=True)
        prompt = json.loads(reviewer["request"]["input"][0]["content"])
        binding = {"planner_attempt_id": str(planner["id"]), "planner_request_sha256": digest(planner["request"]),
            "plan_sha256": digest(plan.model_dump()), "candidate_sha256": prompt["requirements_candidate_sha256"],
            "plan_review_attempt_id": str(reviewer["id"]), "plan_review_request_sha256": digest(reviewer["request"]),
            "claim_plan_sha256": digest(plan.claim_plan.model_dump())}
        require_atomic_stance_review(execution, binding)
        count = len(tools.calls)
        for change in ({"plan_review_attempt_id": str(UUID(int=1))}, {"claim_plan_sha256": "0" * 64},
                       {"plan_review_request_sha256": "0" * 64}, {"plan_review_attempt_id": binding["planner_attempt_id"]}):
            with self.subTest(change=change), self.assertRaises(AgentError):
                require_atomic_stance_review(execution, {**binding, **change})
        original_command = execution.command
        execution.command = {"id": uuid4(), "kind": "followup"}
        with self.assertRaises(AgentError):
            require_atomic_stance_review(execution, binding)
        execution.command = original_command
        self.assertEqual(len(tools.calls), count)


if __name__ == "__main__":
    unittest.main()
