"""Official LangGraph controls with explicit model/tool/Memory/Skill fixtures.

No provider calls or semantic-quality conclusions. Legacy v1 controls run too;
the real truncated Task remains failed and is never repaired by these fixtures.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent))
from test_agent_workset import FixtureExecution, FixtureMemory, FixtureProvider, FixtureTools, WorksetGraphControls

from citequest.agents_v3.contracts import PROFILES, AgentError, canonical, digest
from citequest.agents_v3.graph import AgentGraph
from citequest.agents_v3.workset import AliasWorksetReview, WorksetNodes, as_answer
from citequest.research_v3.contracts import validate_literal
from citequest.research_v3.workset_contracts import DocumentSummary
from citequest.research_v3.workset_wire import (
    ROLE_CAPS,
    AliasDocumentSummary,
    EvidenceAliases,
)

ROOT = Path(__file__).resolve().parents[2]


class WireProvider(FixtureProvider):
    def __init__(self, *args, unknown_alias=False, truncated=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.wire_requests = []
        self.unknown_alias, self.truncated = unknown_alias, truncated

    async def complete(self, payload, logical_key):
        self.wire_requests.append(copy.deepcopy(payload))
        fixture_payload = copy.deepcopy(payload)
        fixture_payload["text"]["format"]["name"] = payload["text"]["format"]["name"].removesuffix("_v2")
        response = await super().complete(fixture_payload, logical_key)
        if self.unknown_alias and len(self.wire_requests) == 1:
            value = json.loads(response["output"][0]["content"][0]["text"])
            value["claims"][0]["edges"][0]["ref_id"] = "E99999"
            response["output"][0]["content"][0]["text"] = json.dumps(value)
        if self.truncated and len(self.wire_requests) == 1:
            response["output"][0]["content"][0]["text"] = '{"claims":[{"id":"C1","statement":"incomplete'
        return response


def execution(kind="summary", *, documents=1, **changes):
    value = FixtureExecution(kind, documents=documents)
    profile = PROFILES["workset-v2"].model_dump()
    profile.update(changes)
    value.value["profile"], value.value["profile_hash"] = profile, digest(profile)
    return value


class AliasControls(unittest.TestCase):
    def setUp(self):
        self.execution = execution(documents=2)
        self.tools = FixtureTools(self.execution)
        self.evidence = list(self.tools.evidence.values())

    def candidate(self):
        source = self.evidence[0]
        return DocumentSummary.model_validate(
            {
                "claims": [
                    {
                        "id": f"C{index + 1}",
                        "statement": "Explicit fixture claim.",
                        "status": "supported",
                        "edges": [
                            {
                                "ref_id": source["ref_id"],
                                "quote": source["text"],
                                "relation": "supports",
                                "rationale": "Fixture only.",
                                "conditions": ["Controlled condition"],
                            }
                            for _ in range(count)
                        ],
                        "limitations": ["Synthetic, no scientific claim."],
                    }
                    for index, count in enumerate((3, 3, 2))
                ],
                "gaps": ["Explicit fixture gap."],
                "evidence_status": "supported",
            },
            strict=True,
        )

    def test_three_claims_eight_edges_roundtrip_preserves_every_field_and_saves_608_ID_chars(self):
        scope = EvidenceAliases(self.evidence).scope([self.evidence[0]])
        original = self.candidate()
        compact = scope.compact_candidate(original.model_dump())
        wire = AliasDocumentSummary.model_validate(compact, strict=True)
        expanded = scope.expand_summary(wire)
        self.assertEqual(expanded.model_dump(), original.model_dump())
        self.assertEqual(sum(len(claim.edges) for claim in expanded.claims), 8)
        self.assertEqual(len(canonical(original.model_dump())) - len(canonical(compact)), 608)
        validate_literal(as_answer(expanded), self.evidence)

    def test_bijection_stable_after_source_order_change_and_bound_to_version(self):
        left, right = EvidenceAliases(self.evidence), EvidenceAliases(list(reversed(self.evidence)))
        self.assertEqual(left.forward, right.forward)
        first = left.scope(self.evidence)
        changed = copy.deepcopy(self.evidence)
        changed[0]["source_version_sha256"] = digest("different-current-version")
        second = EvidenceAliases(changed).scope(changed)
        self.assertNotEqual(first.binding["binding_sha256"], second.binding["binding_sha256"])

    def test_binding_hash_includes_alias_assignment_even_when_same_source_context_is_used(self):
        original = EvidenceAliases(self.evidence).scope([self.evidence[0]])
        additional = {**self.evidence[1], "ref_id": "a-new-earlier-server-reference"}
        changed = EvidenceAliases([additional, *self.evidence]).scope([self.evidence[0]])
        self.assertNotEqual(original.binding["allowed_aliases"], changed.binding["allowed_aliases"])
        self.assertNotEqual(original.binding["binding_sha256"], changed.binding["binding_sha256"])

    def test_known_alias_outside_current_context_scope_is_not_a_capability(self):
        scope = EvidenceAliases(self.evidence).scope([self.evidence[0]])
        with self.assertRaises(AgentError):
            scope.reference(scope.registry.forward[self.evidence[1]["ref_id"]])
        with self.assertRaises(AgentError):
            scope.reference("E99999")
        with self.assertRaises(ValidationError):
            AliasDocumentSummary.model_validate(self.candidate().model_dump(), strict=True)

    def test_duplicate_reference_and_tampered_quote_context_fail_closed(self):
        changed = {**self.evidence[0], "source_version_sha256": digest("changed")}
        with self.assertRaises(AgentError):
            EvidenceAliases([self.evidence[0], changed])
        scope = EvidenceAliases(self.evidence)
        changed = {**self.evidence[0], "text": "invented context"}
        with self.assertRaises(AgentError):
            scope.scope([changed])

    def test_literal_semantic_contract_still_rejects_forged_quotes_after_expansion(self):
        scope = EvidenceAliases(self.evidence).scope([self.evidence[0]])
        value = scope.compact_candidate(self.candidate().model_dump())
        value["claims"][0]["edges"][0]["quote"] = "INVENTED QUOTATION"
        expanded = scope.expand_summary(AliasDocumentSummary.model_validate(value, strict=True))
        with self.assertRaises(ValueError):
            validate_literal(as_answer(expanded), self.evidence)


class WireGraphControls(unittest.IsolatedAsyncioTestCase):
    async def run_graph(self, *, kind="summary", counts=None, profile=None, skill=None, **provider_options):
        task = execution(kind, documents=len(counts or [1]), **(profile or {}))
        if skill is not None:
            task.value["brief"]["selected_skills"] = copy.deepcopy(skill["pins"])
        tools = FixtureTools(task, window_counts=counts)
        provider = WireProvider(task, tools, **provider_options)
        patches = [
            patch("citequest.agents_v3.graph.PostgresSaver", return_value=InMemorySaver()),
            patch.object(
                AgentGraph, "user_turns", lambda graph: [{"turn_id": "U1", "text": task.value["brief"]["question"]}]
            ),
        ]
        if skill is not None:
            patches.append(patch("citequest.agents_v3.user_skills.activate_for_task", return_value=skill))
        from contextlib import ExitStack

        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            graph = AgentGraph(task, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            result = await asyncio.to_thread(graph.run)
        return task, tools, provider, result

    async def test_v2_all_nodes_restore_canonical_citations_and_full_window_coverage(self):
        task, tools, provider, state = await self.run_graph(kind="compare", counts=[2, 2])
        self.assertEqual(task.finishes[-1]["state"], "succeeded")
        self.assertEqual(len(provider.requests), 10)
        for request, fixture in zip(provider.wire_requests, provider.requests, strict=True):
            self.assertTrue(request["text"]["format"]["name"].endswith("_v2"))
            self.assertIn(":wire2:", fixture["logical_key"])
            prompt = json.loads(request["input"][0]["content"])
            self.assertTrue(all(item["ref_id"].startswith("E") for item in prompt["untrusted_evidence_data"]))
        self.assertEqual([item["coverage"]["mapped_windows"] for item in state["result"]["documents"]], [2, 2])
        self.assertTrue(all(citation["ref_id"] in tools.evidence for citation in state["result"]["citations"]))
        self.assertEqual(state["result"]["admission"]["required_max_output_tokens"], 27648)

    async def test_role_caps_and_skill_generation_review_separation_with_pins_retained(self):
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
                    "name": "Explicit fixture skill",
                    "body": "EXPLICIT_GENERATION_SKILL_BODY",
                    "trust": "user_instruction",
                }
            ],
            "total_utf8_bytes": len(b"EXPLICIT_GENERATION_SKILL_BODY"),
        }
        _, _, provider, _ = await self.run_graph(counts=[2], skill=skill)
        self.assertEqual([item["max_output_tokens"] for item in provider.wire_requests], [3072, 3072, 3072, 1536])
        for request in provider.wire_requests:
            prompt = json.loads(request["input"][0]["content"])
            self.assertEqual(prompt["context_manifest"]["skill_refs"], [pin])
            if "review" in request["text"]["format"]["name"]:
                self.assertNotIn("selected_user_skills", prompt)
                self.assertNotIn("EXPLICIT_GENERATION_SKILL_BODY", request["input"][0]["content"])
            else:
                self.assertIn("EXPLICIT_GENERATION_SKILL_BODY", request["input"][0]["content"])

    async def test_total_output_upper_reservation_preflight_without_generation(self):
        task, _, provider, state = await self.run_graph(counts=[2], profile={"output_tokens": 10000})
        self.assertEqual(provider.wire_requests, [])
        self.assertEqual(task.finishes[-1]["state"], "partial")
        self.assertIn(
            "complete_workset_exceeds_remaining_role_output_reservation", state["result"]["admission"]["reasons"]
        )

    async def test_low_per_call_profile_rejected_before_generation_not_silently_clipped(self):
        _, _, provider, state = await self.run_graph(profile={"per_call_output": 1500})
        self.assertEqual(provider.wire_requests, [])
        self.assertIn("workset_v2_per_call_output_profile_incompatible", state["result"]["admission"]["reasons"])

    async def test_unknown_alias_truncated_JSON_and_forged_quote_fail_without_retry(self):
        for option in ({"unknown_alias": True}, {"truncated": True}, {"forged_quote": True}):
            with self.subTest(option=option), self.assertRaises((AgentError, ValueError)):
                await self.run_graph(**option)

    async def test_mapper_checkpoint_resume_does_not_repeat_model_or_change_alias_registry(self):
        task = execution()
        tools, saver = FixtureTools(task), InMemorySaver()
        provider = WireProvider(task, tools)
        original, crashes = WorksetNodes.advance_window, []

        def crash_once(nodes, state):
            if not crashes:
                crashes.append(True)
                raise AgentError("EXPLICIT_FIXTURE_crash_after_wire2_mapper")
            return original(nodes, state)

        with (
            patch("citequest.agents_v3.graph.PostgresSaver", return_value=saver),
            patch.object(AgentGraph, "user_turns", lambda _: []),
            patch.object(WorksetNodes, "advance_window", crash_once),
        ):
            graph = AgentGraph(task, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            with self.assertRaises(AgentError):
                await asyncio.to_thread(graph.run)
            self.assertEqual(len(provider.requests), 1)
            task.run_id = uuid4()
            resumed = AgentGraph(task, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            state = await asyncio.to_thread(resumed.run)
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(state["result"]["documents"][0]["coverage"]["mapped_windows"], 1)

    async def test_server_reviewer_flag_enforces_schema_allowlist(self):
        task = execution()
        with patch("citequest.agents_v3.graph.PostgresSaver", return_value=InMemorySaver()):
            graph = AgentGraph(
                task,
                WireProvider(task, FixtureTools(task)),
                FixtureTools(task),
                FixtureMemory(),
                ROOT / "runtime/s08/skills",
            )
            for model, reviewer in ((AliasWorksetReview, False), (AliasDocumentSummary, True)):
                with self.assertRaises(AgentError) as caught:
                    graph.workset.payload({}, {}, model, "fixture", "fixture", reviewer=reviewer)
                self.assertEqual(caught.exception.code, "trusted_workset_role_schema_mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(case)
        for case in (WorksetGraphControls, AliasControls, WireGraphControls)
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "scope": "official LangGraph + InMemorySaver; explicit provider/tool/memory/Skill fixtures",
        "legacy_v1_controls_included": True,
        "alias_protocol": "citequest-workset-wire/2",
        "role_caps": ROLE_CAPS,
        "real_model_calls": 0,
        "real_MCP_PG": "not_run",
        "semantic_quality": "not_assessed",
        "original_real_truncated_task": "remains_failed_not_repaired",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(report) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
