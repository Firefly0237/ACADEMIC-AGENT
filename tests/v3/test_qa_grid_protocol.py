"""QA3 production payload/adapter with explicit semantic opinions, no provider."""
from __future__ import annotations

import asyncio
import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from test_quote_spans_v3 import source, wire
from test_review_grid import opinion

from citequest.agents_v3.contracts import PROFILES, AgentError, digest, graph_version_for_kind
from citequest.agents_v3.graph import AgentGraph
from citequest.agents_v3.qa_quotes import QuoteQANodes, normalized, require_original_turns
from citequest.research_v3.quote_spans import QuoteRegistry, SelectorAnswer


def prompt():
    return {"canonical_brief": {"question": "现在只讨论fastText，保留数据和评估要求。"},
        "user_turns": [{"turn_id": "U1", "text": "比较效果，并说明数据和评估设置。"},
            {"turn_id": "U2", "text": "现在只讨论fastText，保留数据和评估要求。"}],
        "prior_turns": [], "confirmed_user_memory_data": [], "untrusted_evidence_data": [],
        "selected_user_skills": [], "context_manifest": {"memory_refs": [], "hard_memory_refs": []}}


class FixtureGraph:
    payload = AgentGraph.payload

    def __init__(self, *, coverage="complete", bad_relation=False, profile="qa-v3"):
        self.task = {"profile": PROFILES[profile].model_dump(), "output_tokens": 0, "model_calls": 0,
            "brief": {"requested_direction": "neutral"}}
        self.receipt = None
        self.execution = SimpleNamespace(snapshot=lambda: copy.deepcopy(self.task), command={"id": "fixture-command"},
            settled_model_request=lambda key: self.receipt)
        self.tools = SimpleNamespace(provider_tools=lambda: [{"type": "function", "name": "search_evidence"}])
        self.wires = []

        async def complete(payload, key):
            self.wires.append(payload)
            projected = json.loads(payload["input"][0]["content"])
            evidence = normalized([source()])
            value = {**wire("Q1"), "kind": "qa", "requested_direction": "neutral", "retrieval_scope": "Fixture."}
            if payload["text"]["format"]["name"] == "citequest_qa_review_grid":
                scope = QuoteRegistry(evidence).scope(evidence, candidates=[self.state["answer"]])
                grid = scope.review_grid(self.state["answer"])
                value = opinion(grid)
                if coverage == "partial":
                    value.update(request_coverage="partial", uncovered_requests=["未明确说明评估设置。"])
                if bad_relation:
                    value["claims"]["C1"]["quotes"]["Q1"]["relation"] = "does_not_match"
                assert projected["selected_user_skills"] == []
                assert len(projected["required_user_turns"]) == 2
            return {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value)}]}]}

        self.provider = SimpleNamespace(complete=complete)

    def projection(self, *, evidence=None, include_user_skill_body=True):
        value = prompt()
        value["untrusted_evidence_data"] = evidence or []
        value["context_manifest"]["source_scope"] = []
        value["selected_user_skills"] = ["explicit synthetic generator Skill"] if include_user_skill_body else []
        return value, value["context_manifest"]

    @staticmethod
    def wait(coro):
        return asyncio.run(coro)

    def candidate(self):
        evidence = normalized([source()])
        value = {**wire("Q1"), "kind": "qa", "requested_direction": "neutral", "retrieval_scope": "Fixture."}
        answer = QuoteRegistry(evidence).scope(evidence).expand_answer(SelectorAnswer.model_validate(value, strict=True))
        raw_prompt = prompt()
        require_original_turns(raw_prompt)
        self.state = {"evidence": evidence, "answer": answer.model_dump(), "result": {
            "context_manifest": raw_prompt["context_manifest"], "completion_status": "bounded_windows_evaluated",
            "quality": {"citation_literal": "server_extracted_and_validated"}}}
        return self.state


class QA3ProtocolControls(unittest.TestCase):
    def test_required_original_turns_survive_and_pin_all_IDS(self):
        graph = FixtureGraph()
        value = prompt()
        payload = graph.payload(value, tools=True)
        admitted = json.loads(payload["input"][0]["content"])
        self.assertNotIn("user_turns", admitted)
        self.assertEqual(admitted["required_user_turns"], prompt()["user_turns"])
        self.assertEqual(admitted["context_manifest"]["required_user_turn_refs"],
            [{"id": item["turn_id"], "sha256": digest(item["text"])} for item in prompt()["user_turns"]])
        self.assertEqual(payload["max_output_tokens"], 1024)

    def test_oversized_initial_question_cannot_drop_but_old_QA2_remains_optional(self):
        value = prompt()
        value["user_turns"][0]["text"] = "科研" * 20000
        with self.assertRaises(AgentError) as caught:
            FixtureGraph().payload(copy.deepcopy(value), tools=True)
        self.assertEqual(caught.exception.code, "required_context_exceeds_wire_limit")
        old = FixtureGraph(profile="qa-v2").payload(copy.deepcopy(value), tools=True)
        self.assertTrue(json.loads(old["input"][0]["content"])["context_manifest"]["context_excluded"])

    def test_all_three_role_caps_and_remaining_protocol_preflight(self):
        graph = FixtureGraph()
        self.assertEqual(graph.payload(prompt(), tools=True)["max_output_tokens"], 1024)
        self.assertEqual(graph.payload(prompt())["max_output_tokens"], 3072)
        self.assertEqual(graph.payload(prompt(), review=True)["max_output_tokens"], 4096)
        for remaining_calls, output_used, code in ((2, 0, "qa_grid_remaining_model_budget_insufficient"),
                (6, 8193, "qa_grid_remaining_output_budget_insufficient")):
            graph.task.update(model_calls=6-remaining_calls, output_tokens=output_used)
            with self.subTest(code=code), self.assertRaises(AgentError) as caught:
                graph.payload(prompt(), tools=True)
            self.assertEqual(caught.exception.code, code)

    def test_duplicate_or_conflicting_user_IDS_fail_before_request(self):
        for mutation in ("duplicate", "conflict"):
            value = prompt()
            if mutation == "duplicate":
                value["user_turns"][1]["turn_id"] = "U1"
            else:
                value["required_user_turns"] = []
            with self.subTest(mutation=mutation), self.assertRaises(AgentError):
                FixtureGraph().payload(value, tools=True)

    def test_actual_adapter_keeps_validated_partial_answer_and_grid(self):
        graph = FixtureGraph(coverage="partial")
        result = QuoteQANodes(graph).review(graph.candidate())["result"]
        self.assertEqual(result["publication_status"], "validated_ai_review")
        self.assertEqual(result["completion_status"], "partial_request_coverage")
        self.assertEqual(result["uncovered_requests"], ["未明确说明评估设置。"])
        self.assertEqual(result["request_coverage"], "partial")
        self.assertIn("Q1", result["review_grid"]["claims"]["C1"]["quotes"])

    def test_actual_adapter_preserves_explicit_complete_opinion_fixture(self):
        graph = FixtureGraph()
        result = QuoteQANodes(graph).review(graph.candidate())["result"]
        self.assertEqual(result["request_coverage"], "complete")
        self.assertEqual(result["uncovered_requests"], [])
        self.assertEqual(result["publication_status"], "validated_ai_review")

    def test_only_exact_known_receipt_can_replay_after_all_budget_used(self):
        graph = FixtureGraph()
        original = graph.payload(prompt(), review=True)
        graph.receipt = {"request": copy.deepcopy(original), "request_hash": digest(original)}
        graph.task.update(model_calls=6, output_tokens=16384)
        self.assertEqual(graph.payload(prompt(), review=True), original)
        changed = prompt()
        changed["canonical_brief"]["question"] += " Changed requirement."
        with self.assertRaises(AgentError) as caught:
            graph.payload(changed, review=True)
        self.assertEqual(caught.exception.code, "settled_qa_grid_request_changed")
        graph.receipt = None
        with self.assertRaises(AgentError):
            graph.payload(prompt(), review=True)

    def test_complete_request_cannot_hide_quote_semantic_rejection(self):
        graph = FixtureGraph(bad_relation=True)
        result = QuoteQANodes(graph).review(graph.candidate())["result"]
        self.assertEqual(result["publication_status"], "rejected")
        self.assertEqual(result["quality"]["semantic_support"], "ai_review_rejected")

    def test_profile_versions_and_original_QA2_are_distinct(self):
        self.assertEqual(PROFILES["qa-v2"].output_tokens, 12288)
        self.assertEqual(PROFILES["qa-v2"].per_call_output, 3072)
        self.assertEqual(PROFILES["qa-v3"].output_tokens, 16384)
        self.assertIn("/12-", graph_version_for_kind("qa", "qa-v3"))
        self.assertIn("/10-", graph_version_for_kind("qa", "qa-v2"))


if __name__ == "__main__":
    unittest.main()
