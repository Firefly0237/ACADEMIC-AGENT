"""Official nested StateGraph/Saver with explicit model, tools and ledger fixtures.

No PostgreSQL, real MCP, provider request or scholarly quality is claimed.
"""
import asyncio
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from citequest.agents_v3.contracts import PROFILES, AgentError, digest, graph_version_for_kind
from citequest.agents_v3.graph import AgentGraph

sys.path.insert(0, str(Path(__file__).parent))
from test_agent_workset import FixtureExecution, FixtureMemory, FixtureTools


class Execution(FixtureExecution):
    def __init__(self, kind="summary", documents=1):
        super().__init__(kind, documents=documents)
        profile = PROFILES["workset-v5"].model_dump()
        self.value.update(profile=profile, profile_hash=digest(profile), graph_version=graph_version_for_kind(kind, "workset-v5"))
        self.known, self.credits, self.latest_known = {}, {}, False

    def settled_model_request(self, key):
        value = self.known.get(key)
        return {"request": value["request"], "request_hash": digest(value["request"])} if value else None

    def content_revision_count(self):
        return len(self.credits)

    def reserve_content_revision(self, key, *, phase, error_codes):
        identity = digest({"phase": phase, "error_codes": error_codes})
        if key in self.credits:
            assert self.credits[key]["identity"] == identity
            return self.credits[key]
        if not self.latest_known:
            raise AgentError("content_revision_requires_known_usage")
        if len(self.credits) >= 2:
            raise AgentError("content_revision_limit")
        self.credits[key] = {"ordinal": len(self.credits) + 1, "limit": 2, "identity": identity}
        return self.credits[key]


class Provider:
    def __init__(self, execution, tools, *, fault=None):
        self.execution, self.tools, self.fault = execution, tools, fault
        self.calls, self.originals = [], {}
        self.crashed = False

    async def complete(self, request, key):
        await self.tools.check_scope()
        if key in self.execution.known:
            prior = self.execution.known[key]
            assert prior["request"] == request
            return copy.deepcopy(prior["response"])
        prompt = json.loads(request["input"][0]["content"])
        role = request["text"]["format"]["name"].removeprefix("citequest_workset5_")
        self.calls.append({"role": role, "key": key, "prompt": prompt})
        self.execution.value["model_calls"] += 1
        self.execution.value["input_tokens"] += 100
        self.execution.value["output_tokens"] += 100
        if self.fault == "unknown" and role == "map":
            self.execution.latest_known = False
            raise AgentError("usage_missing")
        if role in {"map", "reduce", "atomic_author"}:
            units = [unit for group in prompt["untrusted_evidence_data"] for unit in group["quote_units"] if unit["eligible_for_citation"]]
            selected = units[:7] or units[:1]
            if role == "atomic_author":
                by_doc = {}
                for unit in units:
                    by_doc.setdefault(unit["document_id"], unit)
                selected = list(by_doc.values())
                claims = [{"id": prompt["immutable_atomic_question"]["claim_id"], "statement": "Compare the two fixture metrics with settings preserved.",
                    "status": "supported", "edges": [{"quote_id": unit["quote_id"], "relation": "supports", "rationale": "Explicit fixture opinion.", "conditions": []} for unit in selected], "limitations": []}]
            else:
                claims = [{"id": "C" + str(i + 1), "statement": "Explicit fixture metric " + str(i + 1), "status": "supported",
                    "edges": [{"quote_id": unit["quote_id"], "relation": "supports", "rationale": "Fixture only.", "conditions": []}], "limitations": []} for i, unit in enumerate(selected)]
            value = {"claims": claims, "gaps": [], "evidence_status": "supported"}
            if role == "map" and self.fault in {"crossfield", "unchanged", "repeat"}:
                job = key.rsplit(":author:", 1)[0]
                if job not in self.originals:
                    value["claims"][0]["edges"][0]["relation"] = "context_only"
                    self.originals[job] = copy.deepcopy(value)
                elif self.fault == "repeat":
                    value = copy.deepcopy(self.originals[job])
                else:
                    value = copy.deepcopy(self.originals[job])
                    value["claims"][0]["edges"][0]["relation"] = "supports"
                    if self.fault == "unchanged" and len(value["claims"]) > 1:
                        value["claims"][1]["statement"] = "Unrelated rewritten proposition"
        elif role in {"document_review", "atomic_review"}:
            expected = {"supported": "entailed", "contradicted": "contradicted", "mixed": "mixed", "insufficient": "insufficient"}
            value = {"claims": {claim["id"]: {"verdict": expected[claim["status"]], "conditions_preserved": True, "explanation": "Explicit semantic fixture, not gold.",
                "quotes": {edge["quote_id"]: {"relation": "matches_declared_relations", "conditions_preserved": True, "explanation": "Fixture only."} for edge in claim["edges"]}} for claim in prompt["candidate"]["claims"]},
                "source_instruction_injection_detected": False, "requested_stance_bias_detected": False,
                "request_coverage": "complete", "uncovered_requests": []}
            if self.fault == "semantic" and role == "atomic_review":
                for claim in value["claims"].values():
                    claim["verdict"] = "invalid"
            if self.fault == "document_partial" and role == "document_review":
                value.update(request_coverage="partial", uncovered_requests=["Fixture requested information remains unanswered."])
            if self.fault == "atomic_partial" and role == "atomic_review":
                value.update(request_coverage="partial", uncovered_requests=["The atomic question is only partly answered."])
        elif role == "plan":
            cards = prompt["routing_cards_not_evidence"]
            questions = []
            for index in range(len(cards)):
                left, right = cards[index], cards[(index + 1) % len(cards)]
                questions.append({"question": "Compare fixture metrics with original settings.", "requirement_ids": ["R1"],
                    "document_ids": [left["document_id"], right["document_id"]],
                    "evidence_route_ids": [left["routing_hints"][0]["evidence_route_ids"][0], right["routing_hints"][0]["evidence_route_ids"][0]]})
            value = {"questions": questions, "unplanned_requirement_ids": [], "limitations": []}
            if self.fault == "missing_document":
                value["questions"] = questions[:1]
            if self.fault == "outside_route":
                value["questions"][0]["evidence_route_ids"][0] = "E9999"
        elif role == "plan_review":
            value = {"all_current_requirements_addressed": True, "all_routes_within_current_scope": True, "all_questions_atomic": True, "omissions": []}
        elif role == "global_coverage":
            value = {"request_coverage": "complete", "uncovered_requirement_ids": [], "missing_document_ids": [], "conflicting_claim_ids": [], "gaps": []}
        else:
            raise AssertionError(role)
        response = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value)}]}]}
        self.execution.known[key] = {"request": copy.deepcopy(request), "response": copy.deepcopy(response)}
        self.execution.latest_known = True
        if self.fault == "crash" and role == "map" and not self.crashed:
            self.crashed = True
            raise RuntimeError("explicit_fixture_death_after_settlement")
        return response


async def run_case(kind="summary", documents=1, windows=None, fault=None, model_used=0):
    execution = Execution(kind, documents)
    execution.value["model_calls"] = model_used
    tools = FixtureTools(execution, window_counts=windows)
    provider = Provider(execution, tools, fault=fault)
    saver = InMemorySaver()
    with patch("citequest.agents_v3.graph.PostgresSaver", return_value=saver):
        graph = AgentGraph(execution, provider, tools, FixtureMemory(), Path("/opt/citequest/runtime/s08/skills"))
    graph.user_turns = lambda: [{"turn_id": "U1", "text": execution.value["brief"]["question"]}]
    result = await asyncio.to_thread(graph.run)
    return execution, tools, provider, graph, result


class Workset5GraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_whole_four_window_group_maps_once_then_independent_review(self):
        execution, tools, provider, graph, result = await run_case(windows=[4])
        self.assertEqual([row["role"] for row in provider.calls], ["map", "document_review", "global_coverage"])
        self.assertEqual(execution.finishes[-1]["state"], "succeeded")
        self.assertEqual(result["result"]["documents"][0]["coverage"]["mapped_windows"], 4)
        self.assertEqual(result["result"]["context_manifest"]["requirements_policy"], "raw_turn_history_not_typed_requirements")

    async def test_three_documents_true_atomic_subgraph_and_twelve_schema_capacity(self):
        execution, tools, provider, graph, result = await run_case("compare", 3, [2, 2, 2])
        self.assertEqual(execution.finishes[-1]["state"], "succeeded")
        self.assertEqual(len(result["result"]["comparisons"]), 3)
        self.assertEqual(sum(row["role"] == "atomic_author" for row in provider.calls), 3)
        for row in provider.calls:
            if row["role"] == "atomic_author":
                self.assertNotIn("routing_cards_not_evidence", row["prompt"])
                self.assertTrue(row["prompt"]["untrusted_evidence_data"])
        state = graph.workflow.get_state(graph.config, subgraphs=True)
        self.assertIsNotNone(state.values["result"])

    async def test_same_task_crossfield_feedback_repairs_exact_claim_only(self):
        execution, tools, provider, graph, result = await run_case(windows=[3], fault="crossfield")
        self.assertEqual(execution.content_revision_count(), 1)
        self.assertEqual(execution.finishes[-1]["state"], "succeeded")
        repair = next(row for row in provider.calls if row["role"] == "map" and "validation_feedback" in row["prompt"])
        self.assertEqual(repair["prompt"]["validation_feedback"]["issues"][0]["code"], "claim_supported_requires_supports")
        self.assertEqual(repair["prompt"]["repair_contract"]["affected_ids"], ["C1"])

    async def test_reviser_cannot_rewrite_unaffected_claim(self):
        execution, tools, provider, graph, result = await run_case(windows=[3], fault="unchanged")
        self.assertEqual(execution.finishes[-1]["state"], "failed")
        self.assertEqual(result["result"]["citations"], [])
        self.assertEqual(execution.content_revision_count(), 1)

    async def test_repeated_content_error_stops_at_two_shared_credits(self):
        execution, tools, provider, graph, result = await run_case(windows=[2], fault="repeat")
        self.assertEqual(execution.content_revision_count(), 2)
        self.assertEqual(sum(row["role"] == "map" for row in provider.calls), 3)
        self.assertEqual(execution.finishes[-1]["state"], "failed")

    async def test_semantic_comparison_rejection_preserves_verified_document_summaries(self):
        execution, tools, provider, graph, result = await run_case("compare", 2, [1, 1], "semantic")
        self.assertEqual(execution.finishes[-1]["state"], "partial")
        self.assertTrue(all(doc["summary"] for doc in result["result"]["documents"]))
        self.assertEqual(result["result"]["comparisons"], [])
        self.assertEqual(execution.content_revision_count(), 2)
        self.assertTrue(all("candidate" not in job for job in result["result"]["review"]["jobs"]))

    async def test_outside_plan_route_uses_no_comparison_writer(self):
        execution, tools, provider, graph, result = await run_case("compare", 2, [1, 1], "outside_route")
        self.assertEqual(execution.finishes[-1]["state"], "partial")
        self.assertFalse(any(row["role"] == "atomic_author" for row in provider.calls))
        self.assertTrue(result["result"]["documents"][0]["summary"])

    async def test_full_source_reading_does_not_override_incomplete_request(self):
        execution, tools, provider, graph, result = await run_case(fault="document_partial")
        self.assertEqual(execution.finishes[-1]["state"], "partial")
        self.assertEqual(result["result"]["documents"][0]["coverage"]["status"], "complete_available_text")
        self.assertEqual(result["result"]["request_coverage"]["documents"][0]["status"], "partial")

    async def test_same_task_used_resources_refuse_before_any_model(self):
        execution, tools, provider, graph, result = await run_case(model_used=62)
        self.assertEqual(len(provider.calls), 0)
        self.assertEqual(execution.finishes[-1]["state"], "partial")

    async def test_global_complete_cannot_erase_atomic_partial(self):
        execution, tools, provider, graph, result = await run_case("compare", 2, [1, 1], "atomic_partial")
        self.assertEqual(result["result"]["request_coverage"]["global"]["request_coverage"], "complete")
        self.assertTrue(all(row["accepted"] for row in result["result"]["request_coverage"]["comparison"]))
        self.assertEqual(execution.finishes[-1]["state"], "partial")
        self.assertTrue(result["result"]["comparisons"])

    async def test_global_complete_cannot_hide_unplanned_selected_document(self):
        execution, tools, provider, graph, result = await run_case("compare", 3, [1, 1, 1], "missing_document")
        self.assertEqual(execution.finishes[-1]["state"], "partial")
        self.assertTrue(all(doc["summary"] for doc in result["result"]["documents"]))
        self.assertIn("comparison_plan_does_not_cover_all_selected_reviewed_documents", result["result"]["gaps"])

    async def test_unknown_never_enters_content_repair(self):
        with self.assertRaisesRegex(AgentError, "usage_missing"):
            await run_case(fault="unknown")

    async def test_ten_documents_are_processed_in_the_same_task(self):
        execution, tools, provider, graph, result = await run_case("compare", 10, [4] * 10)
        self.assertEqual(execution.finishes[-1]["state"], "succeeded")
        self.assertEqual(len(result["result"]["documents"]), 10)
        self.assertEqual(len(result["result"]["comparisons"]), 10)
        self.assertEqual(sum(row["role"] == "map" for row in provider.calls), 10)
        self.assertEqual(execution.value["model_calls"], 43)
        self.assertLessEqual(execution.value["tool_calls"], 64)

    async def test_cold_graph_after_settlement_reuses_one_known_model_attempt(self):
        execution = Execution()
        tools = FixtureTools(execution)
        first_provider = Provider(execution, tools, fault="crash")
        saver = InMemorySaver()
        def make(provider):
            with patch("citequest.agents_v3.graph.PostgresSaver", return_value=saver):
                graph = AgentGraph(execution, provider, tools, FixtureMemory(), Path("/opt/citequest/runtime/s08/skills"))
            graph.user_turns = lambda: [{"turn_id": "U1", "text": execution.value["brief"]["question"]}]
            return graph
        first = make(first_provider)
        with self.assertRaisesRegex(RuntimeError, "explicit_fixture_death_after_settlement"):
            await asyncio.to_thread(first.run)
        snapshot = first.workflow.get_state(first.config, subgraphs=True)
        self.assertTrue(snapshot.next)
        self.assertTrue(snapshot.tasks[0].state)
        restored_provider = Provider(execution, tools)
        restored = make(restored_provider)
        await asyncio.to_thread(restored.run)
        self.assertEqual(execution.value["model_calls"], 3)
        self.assertEqual([row["role"] for row in restored_provider.calls], ["document_review", "global_coverage"])
        self.assertEqual(execution.finishes[-1]["state"], "succeeded")

    async def test_reviewer_does_receive_original_proposition_after_repair(self):
        execution, tools, provider, graph, result = await run_case(windows=[2], fault="crossfield")
        repair = next(row for row in provider.calls if "validation_feedback" in row["prompt"])
        original = repair["prompt"]["original_proposition_context"]
        self.assertEqual(original["claims"][0]["id"], "C1")
        review = next(row for row in provider.calls if row["role"] == "document_review")
        self.assertTrue(review["prompt"]["original_proposition_context"]["claims"])


if __name__ == "__main__":
    unittest.main()
