"""Official product graph + in-memory Saver controls; model/tools are explicit fixtures.

No network, credentials, real provider calls, PostgreSQL or scholarly quality
claims. Real MCP/source/PG integration is a distinct acceptance run.
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

from citequest.agents_v3.contracts import PROFILES, AgentError, ResearchBrief, digest
from citequest.agents_v3.graph import AgentGraph
from citequest.agents_v3.workset import WorksetNodes, quoted_contexts
from citequest.research_v3.workset_contracts import WorksetManifest

ROOT = Path(__file__).resolve().parents[2]


class FixtureExecution:
    def __init__(self, kind="summary", *, documents=1, model_limit=64):
        self.task_id, self.run_id = uuid4(), uuid4()
        self.command = {"id": uuid4(), "kind": "start"}
        selected = [
            {"kind": "native_representation", "id": digest("selected-" + str(index))} for index in range(documents)
        ]
        brief = ResearchBrief(kind=kind, question="Summarize the selected evidence.", selected_papers=selected)
        profile = PROFILES["workset-v1"].model_dump()
        profile["model_calls"] = model_limit
        self.value = {
            "id": self.task_id,
            "owner_id": uuid4(),
            "project_id": uuid4(),
            "kind": kind,
            "brief": brief.model_dump(),
            "revision": 1,
            "profile": profile,
            "profile_hash": digest(profile),
            "graph_version": "EXPLICIT_GRAPH_FIXTURE",
            "history": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "model_calls": 0,
            "tool_calls": 0,
            "elapsed": 0,
        }
        self.saver_connection = None
        self.database = None  # Explicit fixture: no selected UserSkill or real DB.
        self.finishes = []
        self.progress_events = []

    def snapshot(self):
        return copy.deepcopy(self.value)

    def finish(self, state, **kwargs):
        self.finishes.append({"state": state, **kwargs})

    def progress(self, phase, state):
        self.progress_events.append({"phase": phase, "state": state})


class FixtureMemory:
    def current(self, *args):
        return []

    def validate_result_context(self, task, result):
        # Explicit empty-memory fixture for the final production interface.
        # Real pending/current Memory validation is a separate PostgreSQL gate.
        manifest = result["context_manifest"]
        if manifest["memory_refs"] or manifest["hard_memory_refs"]:
            raise AssertionError("fixture_has_no_confirmed_Memory")


class FixtureTools:
    def __init__(self, execution, *, window_counts=None, coverage="native_literal"):
        self.execution = execution
        self.calls, self.cache = [], {}
        self.windows, self.evidence = {}, {}
        self.revoked = False
        documents = []
        selected = execution.value["brief"]["selected_papers"]
        for index, selection in enumerate(selected):
            identifier = f"D{index + 1:02}"
            count = (window_counts or [1] * len(selected))[index]
            version = digest({"selection": selection, "windows": count})
            descriptors = []
            size = 0
            for ordinal in range(1, count + 1):
                text = f"Explicit synthetic document {index + 1}, window {ordinal}, reports metric {ordinal + 4}."
                window = digest({"document": identifier, "ordinal": ordinal})
                ref = "source-window:" + digest({"window": window})
                evidence = {
                    "ref_id": ref,
                    "evidence_type": "source_window",
                    "document_id": identifier,
                    "selection": selection,
                    "source_version_sha256": version,
                    "source_ref": selection,
                    "source_pin_sha256": digest(window),
                    "source_sha256": digest(text.encode()),
                    "unit_text_sha256": digest(text.encode()),
                    "text": text,
                    "span_start": 0,
                    "span_end": len(text),
                    "coordinate_system": "unicode_codepoints_in_synthetic_fixture",
                    "coverage_kind": coverage,
                    "fixture": True,
                }
                self.evidence[ref] = evidence
                body = {"window_id": window, "document_id": identifier, "evidence": [evidence]}
                self.windows[window] = body
                descriptors.append(
                    {
                        "window_id": window,
                        "ordinal": ordinal,
                        "source_parts": 1,
                        "text_bytes": len(text.encode()),
                        "encoded_result_bytes": len(json.dumps(body).encode()),
                        "source_pin_sha256": digest({"source_version": version, "ref_ids": [ref]}),
                    }
                )
                size += len(text)
            documents.append(
                {
                    "document_id": identifier,
                    "selection": selection,
                    "title": "EXPLICIT FIXTURE " + identifier,
                    "source_version_sha256": version,
                    "coverage_kind": coverage,
                    "available_text_bytes": size,
                    "available_codepoints": size,
                    "missing_ranges": [],
                    "windows": descriptors,
                    "model_may_claim_complete_pdf": False,
                }
            )
        self.manifest = WorksetManifest.model_validate(
            {
                "schema_version": "citequest-workset-manifest/1",
                "manifest_id": digest(documents),
                "documents": documents,
                "full_coverage_planned": all(doc["windows"] for doc in documents),
                "window_max_encoded_bytes": 16384,
            },
            strict=True,
        )

    async def check_scope(self):
        if self.revoked:
            raise AgentError("EXPLICIT_FIXTURE_source_access_revoked")
        return {"status": "ready", "fixture": True}

    async def call(self, name, arguments, *, logical_key):
        if logical_key in self.cache:
            return copy.deepcopy(self.cache[logical_key])
        await self.check_scope()
        self.execution.value["tool_calls"] += 1
        self.calls.append({"name": name, "arguments": arguments, "logical_key": logical_key})
        if name == "plan_workset":
            result = {"manifest": self.manifest.model_dump()}
        elif name == "read_source_window":
            result = self.windows[arguments["window_id"]]
        elif name == "verify_evidence_batch":
            result = {
                "valid": True,
                "pins": [
                    {
                        key: self.evidence[ref][key]
                        for key in (
                            "ref_id",
                            "document_id",
                            "source_version_sha256",
                            "source_sha256",
                            "unit_text_sha256",
                        )
                    }
                    for ref in arguments["ref_ids"]
                ],
            }
        else:
            raise AssertionError(name)
        self.cache[logical_key] = copy.deepcopy(result)
        return copy.deepcopy(result)


class FixtureProvider:
    def __init__(
        self, execution, tools, *, reject=False, forged_quote=False, revoke_after_map=False, incomplete_request=False
    ):
        self.execution, self.tools = execution, tools
        self.reject, self.forged_quote, self.revoke_after_map = reject, forged_quote, revoke_after_map
        self.incomplete_request = incomplete_request
        self.requests = []

    async def complete(self, payload, logical_key):
        await self.tools.check_scope()
        prompt = json.loads(payload["input"][0]["content"])
        name = payload["text"]["format"]["name"]
        self.requests.append({"name": name, "logical_key": logical_key, "prompt": prompt})
        self.execution.value["model_calls"] += 1
        self.execution.value["input_tokens"] += 120
        self.execution.value["output_tokens"] += 50
        if name == "citequest_window_notes":
            evidence = prompt["untrusted_evidence_data"][0]
            quote = "INVENTED SOURCE WORDING" if self.forged_quote else evidence["text"]
            value = {
                "claims": [
                    {
                        "id": "C1",
                        "statement": evidence["text"],
                        "status": "supported",
                        "edges": [
                            {
                                "ref_id": evidence["ref_id"],
                                "quote": quote,
                                "relation": "supports",
                                "rationale": "Explicit fixture only.",
                                "conditions": [],
                            }
                        ],
                        "limitations": ["No academic meaning; explicit fixture."],
                    }
                ],
                "gaps": [],
                "evidence_status": "supported",
            }
            if self.revoke_after_map:
                self.tools.revoked = True
        elif name == "citequest_document_reduce":
            claims = [claim for notes in prompt["mapped_notes"] for claim in notes["claims"]]
            # Explicit synthetic reducer makes no semantic-quality claim.
            values = []
            for index, claim in enumerate(claims[:3], 1):
                values.append({**claim, "id": "C" + str(index)})
            value = {"claims": values, "gaps": [], "evidence_status": "supported"}
        elif name == "citequest_workset_compare":
            values = [item["summary"]["claims"][0] for item in prompt["documents"]]
            value = {
                "kind": "compare",
                "requested_direction": "neutral",
                "claims": [
                    {
                        "id": "C1",
                        "statement": "The synthetic documents report their respective measurements.",
                        "status": "supported",
                        "edges": [claim["edges"][0] for claim in values[:2]],
                        "limitations": ["Explicit fixture only."],
                    }
                ],
                "gaps": [],
                "evidence_status": "supported",
                "retrieval_scope": "Synthetic selected sources.",
            }
        elif name in {"citequest_document_review", "citequest_workset_compare_review"}:
            value = {
                "claims": [
                    {
                        "claim_id": claim["id"],
                        "verdict": {"supported": "entailed", "insufficient": "insufficient"}[claim["status"]],
                        "edge_ref_ids": [edge["ref_id"] for edge in claim["edges"]],
                        "numeric_and_conditions_preserved": True,
                        "explanation": "Explicit synthetic independent-review transport.",
                    }
                    for claim in prompt["candidate"]["claims"]
                ],
                "source_instruction_injection_detected": self.reject,
                "requested_stance_bias_detected": False,
                "request_coverage": "partial" if self.incomplete_request else "complete",
                "uncovered_requests": ["An explicitly requested synthetic dimension was omitted."]
                if self.incomplete_request
                else [],
            }
        else:
            raise AssertionError(name)
        return {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": json.dumps(value)}],
                }
            ]
        }


class WorksetGraphControls(unittest.IsolatedAsyncioTestCase):
    async def run_graph(
        self, *, kind="summary", counts=None, model_limit=64, coverage="native_literal", **provider_options
    ):
        execution = FixtureExecution(kind, documents=len(counts or [1]), model_limit=model_limit)
        tools = FixtureTools(execution, window_counts=counts, coverage=coverage)
        provider = FixtureProvider(execution, tools, **provider_options)
        with (
            patch("citequest.agents_v3.graph.PostgresSaver", return_value=InMemorySaver()),
            patch.object(
                AgentGraph,
                "user_turns",
                lambda graph: [{"turn_id": "U1", "text": graph.execution.snapshot()["brief"]["question"]}],
            ),
        ):
            graph = AgentGraph(execution, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            result = await asyncio.to_thread(graph.run)
        return execution, tools, provider, result

    async def test_single_document_all_windows_mapped_reduced_and_reviewed(self):
        execution, tools, provider, state = await self.run_graph(counts=[2])
        result = state["result"]
        self.assertEqual(execution.finishes[-1]["state"], "succeeded")
        self.assertEqual(
            [item["name"] for item in provider.requests],
            [
                "citequest_window_notes",
                "citequest_window_notes",
                "citequest_document_reduce",
                "citequest_document_review",
            ],
        )
        self.assertEqual(result["documents"][0]["coverage"]["mapped_windows"], 2)
        self.assertEqual(
            result["documents"][0]["coverage"]["covered_codepoints"], tools.manifest.documents[0].available_codepoints
        )
        self.assertFalse(result["documents"][0]["document"]["model_may_claim_complete_pdf"])
        self.assertTrue(result["fixture"])
        self.assertEqual(len([call for call in tools.calls if call["name"] == "read_source_window"]), 2)

    async def test_two_documents_compared_with_edges_from_both(self):
        execution, tools, provider, state = await self.run_graph(kind="compare", counts=[1, 1])
        result = state["result"]
        self.assertEqual(execution.finishes[-1]["state"], "succeeded")
        self.assertEqual(len(provider.requests), 6)
        sources = {tools.evidence[edge["ref_id"]]["document_id"] for edge in result["comparisons"][0]["edges"]}
        self.assertEqual(sources, {"D01", "D02"})

    async def test_budget_rejects_before_paid_generation_and_keeps_all_selected_sources(self):
        execution, _, provider, state = await self.run_graph(counts=[2, 2], model_limit=2)
        self.assertEqual(len(provider.requests), 0)
        self.assertEqual(execution.finishes[-1]["state"], "partial")
        self.assertEqual(len(state["result"]["documents"]), 2)
        self.assertTrue(all(item["coverage"]["status"] == "not_processed" for item in state["result"]["documents"]))

    async def test_metadata_never_masquerades_as_full_text(self):
        execution, _, _, state = await self.run_graph(coverage="metadata_only")
        self.assertEqual(execution.finishes[-1]["state"], "partial")
        self.assertEqual(state["result"]["documents"][0]["coverage"]["source_coverage_kind"], "metadata_only")

    async def test_semantic_rejection_does_not_publish_candidate_summary(self):
        execution, _, _, state = await self.run_graph(reject=True)
        self.assertEqual(execution.finishes[-1]["state"], "failed")
        self.assertEqual(state["result"]["publication_status"], "rejected")
        self.assertIsNone(state["result"]["documents"][0]["summary"])

    async def test_forged_literal_quote_is_rejected(self):
        with self.assertRaises(ValueError):
            await self.run_graph(forged_quote=True)

    async def test_revocation_between_mapping_and_review_stops_execution(self):
        with self.assertRaises(AgentError) as caught:
            await self.run_graph(revoke_after_map=True)
        self.assertEqual(caught.exception.code, "EXPLICIT_FIXTURE_source_access_revoked")

    async def test_empty_source_is_insufficient_without_model_call(self):
        execution, _, provider, state = await self.run_graph(counts=[0])
        self.assertEqual(len(provider.requests), 0)
        self.assertEqual(execution.finishes[-1]["state"], "partial")
        self.assertEqual(state["result"]["documents"][0]["coverage"]["status"], "insufficient")

    async def test_followup_same_task_keeps_cumulative_usage_and_new_constraints(self):
        execution = FixtureExecution()
        tools = FixtureTools(execution)
        provider = FixtureProvider(execution, tools)
        saver = InMemorySaver()
        turns = [{"turn_id": "U1", "text": execution.value["brief"]["question"]}]
        with (
            patch("citequest.agents_v3.graph.PostgresSaver", return_value=saver),
            patch.object(AgentGraph, "user_turns", lambda graph: copy.deepcopy(turns)),
        ):
            graph = AgentGraph(execution, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            first = await asyncio.to_thread(graph.run)
            execution.value["history"] = [{"question": turns[0]["text"], "answer": first["result"]}]
            execution.value["brief"]["question"] = (
                "Revise the summary: keep the reported conditions and do not rank methods."
            )
            execution.value["brief"]["constraints"] = ["Do not rank methods."]
            execution.value["revision"] += 1
            execution.command = {"id": uuid4(), "kind": "followup"}
            execution.run_id = uuid4()
            turns.append({"turn_id": "U2", "text": execution.value["brief"]["question"]})
            following = AgentGraph(execution, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            second = await asyncio.to_thread(following.run)
        self.assertEqual(execution.value["model_calls"], 4)
        self.assertEqual(execution.value["output_tokens"], 200)
        self.assertEqual(len(second["result"]["documents"]), 1)
        self.assertIn("Do not rank methods.", provider.requests[-1]["prompt"]["canonical_brief"]["constraints"])
        self.assertEqual(second["result"]["context_manifest"]["brief_revision"], 2)
        latest_prompt = provider.requests[-1]["prompt"]
        self.assertEqual(latest_prompt["canonical_brief"]["question"], turns[-1]["text"])
        self.assertTrue(
            any(
                item.get("id") == "U2" and item.get("reason") == "same_current_question_already_mandatory"
                for item in latest_prompt["context_manifest"]["context_excluded"]
            )
        )

    async def test_checkpoint_recovery_does_not_repeat_completed_mapper(self):
        execution = FixtureExecution()
        tools = FixtureTools(execution)
        provider = FixtureProvider(execution, tools)
        saver = InMemorySaver()
        original = WorksetNodes.advance_window
        crashes = []

        def crash_once(nodes, state):
            if not crashes:
                crashes.append("after_mapper_checkpoint")
                raise AgentError("EXPLICIT_FIXTURE_process_crash_after_mapping")
            return original(nodes, state)

        with (
            patch("citequest.agents_v3.graph.PostgresSaver", return_value=saver),
            patch.object(
                AgentGraph,
                "user_turns",
                lambda graph: [{"turn_id": "U1", "text": execution.value["brief"]["question"]}],
            ),
            patch.object(WorksetNodes, "advance_window", crash_once),
        ):
            graph = AgentGraph(execution, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            with self.assertRaises(AgentError):
                await asyncio.to_thread(graph.run)
            self.assertEqual(len(provider.requests), 1)
            self.assertTrue(graph.workflow.get_state(graph.config).next)
            execution.run_id = uuid4()
            resumed = AgentGraph(execution, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            result = await asyncio.to_thread(resumed.run)
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(
            [item["name"] for item in provider.requests], ["citequest_window_notes", "citequest_document_review"]
        )
        self.assertEqual(result["result"]["publication_status"], "validated_ai_review")

    async def test_changed_publication_pin_is_rejected(self):
        original = FixtureTools.call

        async def corrupt_pin(tools, name, arguments, *, logical_key):
            response = await original(tools, name, arguments, logical_key=logical_key)
            if name == "verify_evidence_batch":
                response["pins"][0]["source_sha256"] = "f" * 64
            return response

        with patch.object(FixtureTools, "call", corrupt_pin), self.assertRaises(AgentError) as caught:
            await self.run_graph()
        self.assertEqual(caught.exception.code, "workset_publication_source_pin_changed")

    async def test_public_progress_contains_phases_and_states_only(self):
        execution, _, _, _ = await self.run_graph()
        events = execution.progress_events
        self.assertIn({"phase": "workset_map_window", "state": "started"}, events)
        self.assertIn({"phase": "workset_review_document", "state": "completed"}, events)
        self.assertTrue(all(set(event) == {"phase", "state"} for event in events))

    async def test_reading_every_window_does_not_mean_every_user_request_answered(self):
        execution, _, _, state = await self.run_graph(incomplete_request=True)
        result = state["result"]
        self.assertEqual(result["documents"][0]["coverage"]["status"], "complete_available_text")
        self.assertEqual(result["documents"][0]["review_status"], "ai_review_validated")
        self.assertIsNotNone(result["documents"][0]["summary"])
        self.assertEqual(result["request_coverage"]["status"], "partial")
        self.assertTrue(result["request_coverage"]["documents"][0]["uncovered_requests"])
        self.assertEqual(execution.finishes[-1]["state"], "partial")

    async def test_clarification_interrupt_remains_waiting_and_does_not_consume_calls(self):
        execution = FixtureExecution()
        execution.value["brief"]["question"] = ""
        tools = FixtureTools(execution)
        provider = FixtureProvider(execution, tools)
        with (
            patch("citequest.agents_v3.graph.PostgresSaver", return_value=InMemorySaver()),
            patch.object(AgentGraph, "user_turns", lambda graph: [{"turn_id": "U1", "text": ""}]),
        ):
            graph = AgentGraph(execution, provider, tools, FixtureMemory(), ROOT / "runtime/s08/skills")
            result = await asyncio.to_thread(graph.run)
        self.assertTrue(result.get("__interrupt__"))
        self.assertEqual(execution.finishes[-1]["state"], "waiting_input")
        self.assertEqual(
            execution.progress_events,
            [{"phase": "clarify", "state": "started"}, {"phase": "clarify", "state": "waiting"}],
        )
        self.assertEqual(len(provider.requests), 0)
        self.assertEqual(len(tools.calls), 0)

    def test_ambiguous_quote_has_separate_literal_contexts(self):
        source = {
            "ref_id": "r",
            "text": "yes " + "x" * 50 + " yes",
            "span_start": 10,
            **dict.fromkeys(
                (
                    "document_id",
                    "source_ref",
                    "source_sha256",
                    "unit_text_sha256",
                    "source_version_sha256",
                    "coverage_kind",
                    "coordinate_system",
                ),
                "fixture",
            ),
        }
        contexts = quoted_contexts([{"edges": [{"ref_id": "r", "quote": "yes"}]}], [source], margin=2)
        self.assertEqual(len(contexts), 2)
        self.assertNotEqual(contexts[0]["context_span_start"], contexts[1]["context_span_start"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WorksetGraphControls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "scope": "official_product_LangGraph_with_explicit_model_tool_memory_and_Saver_fixtures",
        "tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "status": "passed" if result.wasSuccessful() else "failed",
        "live_model_calls": 0,
        "real_mcp": "not_run",
        "real_postgres": "not_run",
        "scholarly_quality": "not_assessed",
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(report) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
