"""Official LangGraph with explicit known-usage, provider/tool and ledger fixtures."""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[2]


class Execution(FixtureExecution):
    def __init__(self):
        super().__init__("qa")
        profile = PROFILES["qa-v4"].model_dump()
        self.value.update(profile=profile, profile_hash=digest(profile), graph_version=graph_version_for_kind("qa", "qa-v4"))
        self.credits, self.known = {}, {}
        self.latest_known = False

    def settled_model_request(self, key):
        value = self.known.get(key)
        return {"request": copy.deepcopy(value["request"]), "request_hash": digest(value["request"])} if value else None

    def content_revision_count(self):
        return len(self.credits)

    def reserve_content_revision(self, key, *, phase, error_codes):
        identity = digest({"phase": phase, "error_codes": error_codes})
        if key in self.credits:
            if self.credits[key]["identity"] != identity:
                raise AgentError("content_revision_replay_changed")
            return self.credits[key]
        if not self.latest_known:
            raise AgentError("content_revision_requires_known_usage")
        if len(self.credits) >= 2:
            raise AgentError("content_revision_limit")
        self.credits[key] = {"ordinal": len(self.credits) + 1, "limit": 2, "identity": identity}
        return self.credits[key]


class Tools(FixtureTools):
    def __init__(self, execution, *, empty=False):
        super().__init__(execution)
        self.empty = empty
        self.catalog = {"search_evidence": {"type": "object", "additionalProperties": False,
            "properties": {"query": {"type": "string", "minLength": 1}, "limit": {"type": "integer", "minimum": 1, "maximum": 3}},
            "required": ["query", "limit"]}}

    def provider_tools(self):
        return [{"type": "function", "name": "search_evidence", "parameters": self.catalog["search_evidence"]}]

    async def call(self, name, args, *, logical_key):
        await self.check_scope()
        if logical_key in self.cache:
            return copy.deepcopy(self.cache[logical_key])
        self.calls.append((name, args))
        self.execution.value["tool_calls"] += 1
        if name == "search_evidence":
            result = {"status": "ok", "empty": self.empty, "evidence": [] if self.empty else list(self.evidence.values())}
        elif name == "read_evidence":
            result = self.evidence[args["ref_id"]]
        else:
            raise AssertionError(name)
        self.cache[logical_key] = copy.deepcopy(result)
        return copy.deepcopy(result)


class Provider:
    def __init__(self, execution, tools, *, author_fault=None, review_fault=None, repair_fault=None, repeat=False, partial=False):
        self.execution, self.tools = execution, tools
        self.author_fault, self.review_fault, self.repeat, self.partial = author_fault, review_fault, repeat, partial
        self.repair_fault = repair_fault
        self.requests = []
        self.author_calls, self.review_calls = 0, 0

    async def complete(self, payload, key):
        await self.tools.check_scope()
        if key in self.execution.known:
            assert self.execution.known[key]["request"] == payload
            return copy.deepcopy(self.execution.known[key]["response"])
        self.requests.append({"request": copy.deepcopy(payload), "key": key})
        self.execution.value["model_calls"] += 1
        self.execution.value["input_tokens"] += 100
        self.execution.value["output_tokens"] += 40
        self.execution.latest_known = False
        prompt = json.loads(payload["input"][0]["content"])
        if payload.get("tool_choice"):
            value = {"output": [{"type": "function_call", "name": "search_evidence", "call_id": "fixture-search",
                "arguments": json.dumps({"query": "synthetic controlled evidence", "limit": 1})}]}
        elif payload["text"]["format"]["name"] == "citequest_qa4_selector_answer":
            self.author_calls += 1
            units = [unit for group in prompt["untrusted_evidence_data"] for unit in group["quote_units"] if "text" in unit]
            if "untrusted_previous_candidate" in prompt:
                result = json.loads(prompt["untrusted_previous_candidate"])
                for claim in result["claims"]:
                    if claim["id"] in prompt["repair_contract"]["affected_claim_ids"]:
                        claim["status"] = "supported" if units else "insufficient"
            else:
                result = {"kind": "qa", "requested_direction": "neutral", "claims": [{"id": "C" + str(i + 1),
                    "statement": "Explicit fixture only.", "status": "supported" if units else "insufficient",
                    "edges": [{"quote_id": units[0]["quote_id"], "relation": "supports", "rationale": "Fixture.", "conditions": []}] if units else [],
                    "limitations": ["Explicit source and semantic fixture, not academic truth."]} for i in range(6)],
                    "gaps": [], "evidence_status": "supported" if units else "insufficient", "retrieval_scope": "Synthetic fixture scope."}
            fault = self.author_fault if self.author_calls == 1 or self.repeat else None
            if self.author_calls > 1 and self.repair_fault == "delete":
                result["claims"].pop()
            elif self.author_calls > 1 and self.repair_fault == "unaffected":
                result["claims"][0]["statement"] += " Illicit unrelated replacement."
            if fault == "enum":
                result["claims"][-1]["status"] = "context_only"
            elif fault == "duplicate":
                result["claims"][-1]["id"] = "C1"
            elif fault == "unknown":
                raise AgentError("provider_usage_missing")
            value = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "bad-json" if fault == "json" else json.dumps(result)}]}]}
        else:
            self.review_calls += 1
            assert not prompt["selected_user_skills"]
            statuses = {claim["id"]: claim["status"] for claim in prompt["candidate"]["claims"]}
            result = {"claims": {cid: {"verdict": "insufficient" if statuses[cid] == "insufficient" else "entailed",
                "conditions_preserved": True, "explanation": "Explicit opinion fixture.", "quotes": {qid: {
                    "relation": "matches_declared_relations", "conditions_preserved": True, "explanation": "Fixture."} for qid in qids}}
                for cid, qids in prompt["review_grid"]["claim_quote_ids"].items()},
                "source_instruction_injection_detected": False, "requested_stance_bias_detected": False,
                "request_coverage": "partial" if self.partial else "complete",
                "uncovered_requests": ["Explicit unanswered fixture requirement."] if self.partial else []}
            fault = self.review_fault if self.review_calls == 1 or self.repeat else None
            if fault == "missing":
                result["claims"].pop("C6")
            elif fault == "semantic":
                result["claims"]["C1"]["verdict"] = "invalid"
            elif fault == "unknown":
                raise AgentError("provider_usage_missing")
            value = {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(result)}]}]}
        value["status"] = "completed"
        self.execution.known[key] = {"request": copy.deepcopy(payload), "response": copy.deepcopy(value)}
        self.execution.latest_known = True
        return value


class QA4Controls(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, *, execution=None, saver=None, tools=None, **mode):
        execution = execution or Execution()
        tools = tools or Tools(execution)
        provider = Provider(execution, tools, **mode)
        with (patch("citequest.agents_v3.graph.PostgresSaver", return_value=saver or InMemorySaver()),
                patch.object(AgentGraph, "user_turns", lambda graph: [{"turn_id": "U1", "text": execution.value["brief"]["question"]}])):
            graph = AgentGraph(execution, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            result = await asyncio.to_thread(graph.run)
        return execution, provider, tools, result

    async def test_real_observed_enum_shape_revised_without_deleting_C6(self):
        execution, provider, _, result = await self.run_case(author_fault="enum")
        self.assertEqual(execution.content_revision_count(), 1)
        self.assertEqual(execution.value["model_calls"], 4)
        self.assertEqual([claim["id"] for claim in result["answer"]["claims"]], ["C"+str(i) for i in range(1, 7)])
        self.assertEqual(execution.finishes[-1]["state"], "succeeded")
        revised = json.loads(provider.requests[2]["request"]["input"][0]["content"])
        self.assertEqual(revised["repair_contract"]["affected_claim_ids"], ["C6"])
        self.assertEqual(set(revised["repair_contract"]["unchanged_claim_sha256"]), {"C1", "C2", "C3", "C4", "C5"})

    async def test_author_and_reviewer_format_share_exactly_two_Task_credits(self):
        execution, _, _, _ = await self.run_case(author_fault="enum", review_fault="missing")
        self.assertEqual(execution.content_revision_count(), 2)
        self.assertEqual(execution.value["model_calls"], 5)
        self.assertEqual(execution.finishes[-1]["state"], "succeeded")

    async def test_repeat_bad_author_stops_at_two_not_unbounded_retry(self):
        with self.assertRaises(AgentError) as error:
            await self.run_case(author_fault="enum", repeat=True)
        self.assertEqual(error.exception.code, "qa_content_validation_unresolved")

    async def test_repeat_bad_review_stops_at_two(self):
        with self.assertRaises(AgentError) as error:
            await self.run_case(review_fault="missing", repeat=True)
        self.assertEqual(error.exception.code, "qa_content_validation_unresolved")

    async def test_semantic_revise_has_new_full_source_review(self):
        execution, provider, _, _ = await self.run_case(review_fault="semantic")
        self.assertEqual(execution.content_revision_count(), 1)
        self.assertEqual(provider.review_calls, 2)
        self.assertEqual(execution.finishes[-1]["state"], "succeeded")

    async def test_partial_coverage_keeps_valid_answer_but_not_complete(self):
        execution, _, _, result = await self.run_case(partial=True)
        self.assertEqual(execution.content_revision_count(), 0)
        self.assertEqual(execution.finishes[-1]["state"], "partial")
        self.assertEqual(result["result"]["completion_status"], "partial_request_coverage")

    async def test_missing_usage_cannot_enter_author_or_reviewer_repair(self):
        for role in ("author", "review"):
            execution = Execution()
            with self.subTest(role=role), self.assertRaises(AgentError):
                await self.run_case(execution=execution, **{role+"_fault": "unknown"})
            self.assertEqual(execution.content_revision_count(), 0)

    async def test_initial_json_or_duplicate_IDs_never_guess_missing_claims(self):
        for fault in ("json", "duplicate"):
            execution = Execution()
            with self.subTest(fault=fault), self.assertRaises(AgentError) as error:
                await self.run_case(execution=execution, author_fault=fault)
            self.assertEqual(error.exception.code, "qa_revision_claim_identity_ambiguous")
            self.assertEqual(execution.content_revision_count(), 0)

    async def test_empty_source_is_explicit_insufficiency_not_fake_search_success(self):
        execution = Execution()
        execution, _, _, result = await self.run_case(execution=execution, tools=Tools(execution, empty=True))
        self.assertEqual(execution.finishes[-1]["state"], "partial")
        self.assertEqual(result["result"]["citations"], [])

    async def test_existing_credits_do_not_reset_at_followup(self):
        execution = Execution()
        for i in range(2):
            execution.credits["previous-command:"+str(i)] = {"ordinal": i+1, "limit": 2}
        with self.assertRaises(AgentError) as error:
            await self.run_case(execution=execution, author_fault="enum")
        self.assertEqual(error.exception.code, "qa_content_validation_unresolved")
        self.assertEqual(execution.content_revision_count(), 2)

    async def test_reviser_cannot_delete_C6_or_change_an_unaffected_claim(self):
        for fault in ("delete", "unaffected"):
            execution = Execution()
            with self.subTest(fault=fault), self.assertRaises(AgentError) as error:
                await self.run_case(execution=execution, author_fault="enum", repair_fault=fault)
            self.assertEqual(error.exception.code, "qa_content_validation_unresolved")
            self.assertEqual(execution.content_revision_count(), 2)
            self.assertEqual(execution.value["model_calls"], 4)

    async def test_semantic_failure_after_two_credits_is_rejected_not_accepted(self):
        execution, provider, _, result = await self.run_case(review_fault="semantic", repeat=True)
        self.assertEqual(execution.content_revision_count(), 2)
        self.assertEqual(provider.review_calls, 3)
        self.assertEqual(execution.finishes[-1]["state"], "failed")
        self.assertEqual(result["result"]["publication_status"], "rejected")

    async def test_120_seconds_profile_not_extended_for_repairs(self):
        execution, _, _, _ = await self.run_case(author_fault="enum")
        self.assertEqual(execution.value["profile"]["active_seconds"], 120)
        self.assertEqual(execution.value["profile"]["model_calls"], 10)
        self.assertEqual(execution.value["profile"]["output_tokens"], 30720)

    async def test_known_author_receipt_cold_graph_does_not_call_model_again(self):
        execution = Execution()
        tools = Tools(execution)
        provider = Provider(execution, tools, author_fault="enum")
        saver = InMemorySaver()
        original = AgentGraph.observed_node
        crashed = {"done": False}

        def observed(graph, phase, node):
            wrapped = original(graph, phase, node)
            if phase != "qa4_author":
                return wrapped
            def stop(state):
                result = wrapped(state)
                if not crashed["done"]:
                    crashed["done"] = True
                    raise RuntimeError("explicit_fixture_death_after_known_receipt")
                return result
            return stop

        with (patch("citequest.agents_v3.graph.PostgresSaver", return_value=saver),
                patch.object(AgentGraph, "user_turns", lambda graph: [{"turn_id": "U1", "text": execution.value["brief"]["question"]}]),
                patch.object(AgentGraph, "observed_node", observed)):
            graph = AgentGraph(execution, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            with self.assertRaisesRegex(RuntimeError, "explicit_fixture_death"):
                await asyncio.to_thread(graph.run)
        self.assertEqual(execution.value["model_calls"], 2)
        # The author fault is still the cached invalid enum; recovery must
        # validate it and reserve a repair, never regenerate the old attempt.
        with (patch("citequest.agents_v3.graph.PostgresSaver", return_value=saver),
                patch.object(AgentGraph, "user_turns", lambda graph: [{"turn_id": "U1", "text": execution.value["brief"]["question"]}])):
            graph = AgentGraph(execution, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            await asyncio.to_thread(graph.run)
        self.assertEqual(execution.value["model_calls"], 4)
        self.assertEqual(execution.content_revision_count(), 1)
        self.assertEqual(provider.author_calls, 2)
        self.assertEqual(execution.finishes[-1]["state"], "succeeded")


if __name__ == "__main__":
    unittest.main()
