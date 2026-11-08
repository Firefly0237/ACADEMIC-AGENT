"""Official LangGraph/Saver with explicitly synthetic SDK/tool/Memory fixtures."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from citequest.agents_v3.contracts import PROFILES, AgentError, digest
from citequest.agents_v3.graph import AgentGraph

sys.path.insert(0, str(Path(__file__).parent))
from test_agent_workset import FixtureExecution, FixtureMemory, FixtureTools

ROOT = Path(__file__).resolve().parents[2]


class Execution(FixtureExecution):
    def __init__(self, kind="summary", *, documents=1, **changes):
        super().__init__(kind, documents=documents)
        profile = PROFILES["workset-v2"].model_dump()
        profile.update(profile_id="workset-v3", **changes)
        self.value.update(profile=profile, profile_hash=digest(profile))
        self.revisions = {}

    def content_revision_count(self):
        return len(self.revisions)

    def reserve_content_revision(self, logical_key, *, phase, error_codes):
        if logical_key in self.revisions:
            return self.revisions[logical_key]
        if len(self.revisions) >= 2:
            raise AgentError("content_revision_limit")
        value = {"logical_key": logical_key, "ordinal": len(self.revisions) + 1, "limit": 2}
        self.revisions[logical_key] = value
        return value


class Provider:
    def __init__(
        self,
        execution,
        tools,
        *,
        bad_selector=False,
        unknown=False,
        crash_on_repair=False,
        reject_once=False,
        eight=False,
        bad_always=False,
        incomplete_request=False,
        invalid_identity=None,
        drop_repair_claim=False,
    ):
        self.execution, self.tools = execution, tools
        self.bad_selector, self.unknown, self.crash_on_repair, self.reject_once, self.eight = (
            bad_selector,
            unknown,
            crash_on_repair,
            reject_once,
            eight,
        )
        self.requests, self.cache = [], {}
        self.bad_always, self.incomplete_request = bad_always, incomplete_request
        self.invalid_identity, self.drop_repair_claim = invalid_identity, drop_repair_claim

    async def complete(self, payload, logical_key):
        await self.tools.check_scope()
        if logical_key in self.cache:
            saved, response = self.cache[logical_key]
            if saved != digest(payload):
                raise AgentError("fixture_cached_payload_changed")
            return copy.deepcopy(response)
        if self.crash_on_repair and ":content-revision:" in logical_key:
            self.crash_on_repair = False
            raise AgentError("explicit_fixture_process_crash_before_repair")
        if self.unknown:
            raise AgentError("provider_usage_missing")
        prompt = json.loads(payload["input"][0]["content"])
        name = payload["text"]["format"]["name"].removesuffix("_v3")
        self.requests.append({"name": name, "key": logical_key, "payload": copy.deepcopy(payload), "prompt": prompt})
        self.execution.value["model_calls"] += 1
        self.execution.value["input_tokens"] += 120
        self.execution.value["output_tokens"] += 50
        units = [
            unit
            for group in prompt["untrusted_evidence_data"]
            for unit in group["quote_units"]
            if unit["eligible_for_citation"]
        ]

        def claim(identifier, index=1):
            return {
                "id": "C" + str(index),
                "statement": "Explicit fixture statement, not academic evidence.",
                "status": "supported",
                "edges": [
                    {"quote_id": identifier, "relation": "supports", "rationale": "Explicit fixture.", "conditions": []}
                ],
                "limitations": ["Synthetic only."],
            }

        if name == "citequest_window_notes":
            identifier = "Q99999" if self.bad_always or (self.bad_selector and not self.cache) else units[0]["quote_id"]
            value = {
                "claims": [claim(identifier, i + 1) for i in range(8 if self.eight else 1)],
                "gaps": [],
                "evidence_status": "supported",
            }
            if self.invalid_identity == "duplicate":
                value["claims"] *= 2
            elif self.invalid_identity == "invalid":
                value["claims"].append({**claim(identifier, 2), "id": "not-a-claim-id"})
            if self.drop_repair_claim and not self.cache:
                value["claims"].append(claim(identifier, 2))
        elif name == "citequest_document_reduce":
            claims = [claim for note in prompt["mapped_notes"] for claim in note["claims"]]
            value = {
                "claims": [{**item, "id": "C" + str(i + 1)} for i, item in enumerate(claims[:8])],
                "gaps": [],
                "evidence_status": "supported",
            }
        elif name == "citequest_workset_compare":
            selected = [item["summary"]["claims"][0] for item in prompt["documents"]]
            value = {
                "kind": "compare",
                "requested_direction": "neutral",
                "claims": [{**selected[0], "edges": [item["edges"][0] for item in selected[:2]]}],
                "gaps": [],
                "evidence_status": "supported",
                "retrieval_scope": "Explicit synthetic sources.",
            }
        elif name in {"citequest_document_review", "citequest_workset_compare_review"}:
            reject = self.reject_once
            self.reject_once = False
            value = {
                "claims": [
                    {
                        "claim_id": item["id"],
                        "verdict": "insufficient"
                        if reject
                        else {"supported": "entailed", "insufficient": "insufficient"}[item["status"]],
                        "edge_ref_ids": [edge["quote_id"] for edge in item["edges"]],
                        "numeric_and_conditions_preserved": not reject,
                        "explanation": "Explicit independent transport fixture.",
                    }
                    for item in prompt["candidate"]["claims"]
                ],
                "source_instruction_injection_detected": False,
                "requested_stance_bias_detected": False,
                "request_coverage": "partial" if self.incomplete_request else "complete",
                "uncovered_requests": ["The fixture cannot provide the requested dataset."]
                if self.incomplete_request
                else [],
            }
        elif name == "citequest_directed_revise":
            value = copy.deepcopy(prompt["candidate"])
        else:
            raise AssertionError(name)
        response = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "{malformed" if self.invalid_identity == "malformed" else json.dumps(value),
                        }
                    ],
                }
            ]
        }
        self.cache[logical_key] = (digest(payload), copy.deepcopy(response))
        return response


class GraphControls(unittest.IsolatedAsyncioTestCase):
    async def run_graph(self, *, kind="summary", counts=None, profile=None, skill=None, revisions=0, **options):
        task = Execution(kind, documents=len(counts or [1]), **(profile or {}))
        if skill is not None:
            task.value["brief"]["selected_skills"] = copy.deepcopy(skill["pins"])
        task.revisions.update(
            {"explicit-prior-command-" + str(i): {"ordinal": i + 1, "limit": 2} for i in range(revisions)}
        )
        tools = FixtureTools(task, window_counts=counts)
        provider = Provider(task, tools, **options)
        saver = InMemorySaver()
        with ExitStack() as stack:
            stack.enter_context(patch("citequest.agents_v3.graph.PostgresSaver", return_value=saver))
            stack.enter_context(patch.object(AgentGraph, "user_turns", lambda _: []))
            if skill is not None:
                stack.enter_context(patch("citequest.agents_v3.user_skills.activate_for_task", return_value=skill))
            graph = AgentGraph(task, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            state = await asyncio.to_thread(graph.run)
        return task, tools, provider, state

    async def test_summary_canonical_quote_locator_and_eight_claims(self):
        task, tools, provider, state = await self.run_graph(eight=True)
        self.assertEqual(task.finishes[-1]["state"], "succeeded")
        self.assertEqual(len(state["result"]["documents"][0]["summary"]["claims"]), 8)
        self.assertTrue(all("quote_locator" in citation for citation in state["result"]["citations"]))
        self.assertTrue(all(citation["ref_id"] in tools.evidence for citation in state["result"]["citations"]))
        self.assertTrue(all(":wire3:" in item["key"] for item in provider.requests))

    async def test_compare_two_documents_all_windows_and_same_real_refs(self):
        task, _tools, provider, state = await self.run_graph(kind="compare", counts=[2, 2])
        self.assertEqual(task.finishes[-1]["state"], "succeeded")
        self.assertEqual(len(provider.requests), 10)
        self.assertEqual([item["coverage"]["mapped_windows"] for item in state["result"]["documents"]], [2, 2])
        self.assertEqual(state["result"]["admission"]["revision_extra_model_calls"], 4)
        self.assertEqual(state["result"]["admission"]["required_max_output_tokens"], 27648 + 12288)

    async def test_known_selector_error_directed_repair_and_durable_credit(self):
        task, _, provider, _state = await self.run_graph(bad_selector=True)
        self.assertEqual(task.content_revision_count(), 1)
        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(task.finishes[-1]["state"], "succeeded")
        self.assertIn("validation_feedback", provider.requests[1]["prompt"])
        self.assertEqual(
            provider.requests[1]["prompt"]["validation_feedback"]["error_codes"], ["quote_output_selector_not_issued"]
        )

    async def test_semantic_conflict_uses_same_credit_then_independent_review_again(self):
        task, _, provider, state = await self.run_graph(reject_once=True)
        self.assertEqual(task.content_revision_count(), 1)
        self.assertEqual(task.finishes[-1]["state"], "succeeded")
        self.assertEqual(
            [item["name"] for item in provider.requests],
            [
                "citequest_window_notes",
                "citequest_document_review",
                "citequest_directed_revise",
                "citequest_document_review",
            ],
        )
        self.assertEqual(len(state["result"]["review"]["documents"][0]["review_rounds"]), 2)

    async def test_unknown_provider_error_does_not_become_content_repair(self):
        with self.assertRaises(AgentError) as caught:
            await self.run_graph(unknown=True)
        self.assertEqual(caught.exception.code, "provider_usage_missing")

    async def test_request_coverage_remains_partial_despite_full_source_read(self):
        task, _, _, state = await self.run_graph(incomplete_request=True)
        self.assertEqual(task.finishes[-1]["state"], "partial")
        self.assertEqual(state["result"]["documents"][0]["coverage"]["status"], "complete_available_text")
        self.assertIsNotNone(state["result"]["documents"][0]["summary"])
        self.assertEqual(state["result"]["request_coverage"]["status"], "partial")

    async def test_no_semantic_credit_from_prior_commands_rejects_without_reviser(self):
        task, _, provider, state = await self.run_graph(revisions=2, reject_once=True)
        self.assertEqual(task.content_revision_count(), 2)
        self.assertEqual(len(provider.requests), 2)
        self.assertIsNone(state["result"]["documents"][0]["summary"])
        self.assertEqual(state["result"]["documents"][0]["review_status"], "quality_rejected")

    async def test_still_invalid_after_two_repairs_never_dispatches_a_fourth(self):
        task = Execution()
        tools = FixtureTools(task)
        provider = Provider(task, tools, bad_always=True)
        with (
            patch("citequest.agents_v3.graph.PostgresSaver", return_value=InMemorySaver()),
            patch.object(AgentGraph, "user_turns", lambda _: []),
        ):
            graph = AgentGraph(task, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            with self.assertRaises(AgentError) as caught:
                await asyncio.to_thread(graph.run)
        self.assertEqual(caught.exception.code, "workset_content_validation_unresolved")
        self.assertEqual(task.content_revision_count(), 2)
        self.assertEqual(len(provider.requests), 3)

    async def test_generation_Skill_body_excluded_from_independent_review_with_pins(self):
        pin = {
            "skill_id": str(uuid4()),
            "version_id": str(uuid4()),
            "content_sha256": digest("fixture skill"),
            "head_revision": 1,
        }
        skill = {
            "pins": [pin],
            "instructions": [
                {
                    "pin": pin,
                    "name": "Fixture Skill",
                    "body": "EXPLICIT_GENERATOR_ONLY_BODY",
                    "trust": "user_instruction",
                }
            ],
            "total_utf8_bytes": 28,
        }
        _, _, provider, _ = await self.run_graph(counts=[2], skill=skill)
        self.assertEqual([item["payload"]["max_output_tokens"] for item in provider.requests], [3072, 3072, 3072, 1536])
        for item in provider.requests:
            self.assertEqual(item["prompt"]["context_manifest"]["skill_refs"], [pin])
            if "review" in item["name"]:
                self.assertNotIn("selected_user_skills", item["prompt"])
                self.assertNotIn("EXPLICIT_GENERATOR_ONLY_BODY", item["payload"]["input"][0]["content"])
            else:
                self.assertIn("EXPLICIT_GENERATOR_ONLY_BODY", item["payload"]["input"][0]["content"])

    async def test_ambiguous_or_unrecoverable_initial_identity_never_authorizes_repair(self):
        for case in ("duplicate", "invalid", "malformed"):
            with self.subTest(case=case):
                task = Execution()
                tools = FixtureTools(task)
                provider = Provider(task, tools, invalid_identity=case)
                with (
                    patch("citequest.agents_v3.graph.PostgresSaver", return_value=InMemorySaver()),
                    patch.object(AgentGraph, "user_turns", lambda _: []),
                ):
                    graph = AgentGraph(task, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
                    with self.assertRaises(AgentError) as caught:
                        await asyncio.to_thread(graph.run)
                self.assertEqual(caught.exception.code, "workset_claim_identity_ambiguous")
                self.assertEqual(task.content_revision_count(), 0)
                self.assertEqual(len(provider.requests), 1)

    async def test_reviser_cannot_delete_valid_initial_claim_to_repair_another(self):
        task = Execution()
        tools = FixtureTools(task)
        provider = Provider(task, tools, bad_selector=True, drop_repair_claim=True)
        with (
            patch("citequest.agents_v3.graph.PostgresSaver", return_value=InMemorySaver()),
            patch.object(AgentGraph, "user_turns", lambda _: []),
        ):
            graph = AgentGraph(task, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            with self.assertRaises(AgentError) as caught:
                await asyncio.to_thread(graph.run)
        self.assertEqual(caught.exception.code, "workset_content_validation_unresolved")
        self.assertEqual(task.content_revision_count(), 2)
        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(
            provider.requests[-1]["prompt"]["validation_feedback"]["error_codes"],
            ["content_revision_must_preserve_claim_ids"],
        )

    async def test_role_reservation_rejects_before_generation_not_clipped(self):
        task, _, provider, state = await self.run_graph(profile={"output_tokens": 10000})
        self.assertEqual(provider.requests, [])
        self.assertEqual(task.finishes[-1]["state"], "partial")
        self.assertIn(
            "complete_workset_exceeds_remaining_role_output_reservation", state["result"]["admission"]["reasons"]
        )

    async def test_cold_saver_after_known_invalid_response_reserved_repair_no_double_model(self):
        task = Execution()
        tools = FixtureTools(task)
        provider = Provider(task, tools, bad_selector=True, crash_on_repair=True)
        saver = InMemorySaver()
        with (
            patch("citequest.agents_v3.graph.PostgresSaver", return_value=saver),
            patch.object(AgentGraph, "user_turns", lambda _: []),
        ):
            first = AgentGraph(task, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            with self.assertRaises(AgentError):
                await asyncio.to_thread(first.run)
            self.assertEqual(task.content_revision_count(), 1)
            task.run_id = uuid4()
            second = AgentGraph(task, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            await asyncio.to_thread(second.run)
        self.assertEqual(task.content_revision_count(), 1)
        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(task.value["model_calls"], 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GraphControls))
    args.out.write_text(
        json.dumps(
            {
                "status": "PASS" if result.wasSuccessful() else "FAIL",
                "tests": result.testsRun,
                "scope": "official LangGraph/InMemorySaver; explicit synthetic provider/tool/Memory/revision-ledger fixtures",
                "real_model_calls": 0,
                "real_PG_MCP": "not_run",
                "semantic_quality": "not_assessed",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
