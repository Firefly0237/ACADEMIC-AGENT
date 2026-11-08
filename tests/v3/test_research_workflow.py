"""Multi-role product workflow controls. All external services are explicit fixtures."""

from __future__ import annotations

import asyncio
import copy
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.research_workflow import ResearchWorkflow
from citequest.research_v3.brief import REQUIRED


def planning_input():
    facts = {field: "Explicit synthetic " + field + " constraint." for field in REQUIRED}
    facts["resources"] = "Explicit synthetic availability: one CPU, no GPU, budget 100 CNY."
    return {
        "schema_version": "citequest-research-planning-input/1",
        "brief_id": str(uuid4()),
        "brief_revision": 1,
        "brief_sha256": digest(facts),
        "user_confirmed": facts,
        "explicit_unknowns": {},
        "selected_sources": [],
        "research_status": "proposed_not_executed",
        "source_admission_required": True,
        "school_format_ready": False,
    }


class ExecutionFixture:
    def __init__(self, *, model_limit=64):
        self.task_id, self.run_id = uuid4(), uuid4()
        self.database = None  # Explicit no-SQL fixture; selected Skills require a real DB.
        self.command = {"id": uuid4(), "kind": "start"}
        self.value = {
            "owner_id": uuid4(),
            "project_id": uuid4(),
            "brief": {"question": "Design a synthetic research plan."},
            "graph_version": "EXPLICIT_RESEARCH_GRAPH_FIXTURE",
            "profile_hash": digest("explicit-research-profile"),
            "model_calls": 0,
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed": 0,
            "known_ncny": 0,
            "reserved_ncny": 0,
            "profile": {
                "model_calls": model_limit,
                "tool_calls": 64,
                "output_tokens": 96000,
                "input_tokens": 512000,
                "active_seconds": 1800,
                "cash_ncny": 80000000000,
                "per_call_output": 2048,
                "request_bytes": 65536,
            },
        }
        self.events = []

    def snapshot(self):
        return copy.deepcopy(self.value)

    def progress(self, phase, state):
        self.events.append({"phase": phase, "state": state})


class MemoryFixture:
    def current(self, *args):
        return []


class ToolFixture:
    def __init__(self, host, *, empty=False):
        self.host, self.empty = host, empty
        self.revoked = False
        self.calls, self.cache = [], {}
        text = "This is synthetic control evidence. The proposed baseline uses the same data split."
        self.item = {
            "ref_id": "fixture:" + digest(text.encode()),
            "text": text,
            "source_ref": {"kind": "fixture", "id": "control"},
            "source_sha256": digest(text.encode()),
            "unit_text_sha256": digest(text.encode()),
            "coverage": "explicit_fixture",
            "fixture": True,
        }

    async def check_scope(self):
        if self.revoked:
            raise AgentError("EXPLICIT_FIXTURE_research_source_revoked")
        return {"status": "ready"}

    async def call(self, name, arguments, *, logical_key):
        await self.check_scope()
        if logical_key in self.cache:
            return copy.deepcopy(self.cache[logical_key])
        self.host.execution.value["tool_calls"] += 1
        self.calls.append((name, arguments))
        if name == "search_evidence":
            result = {"evidence": [] if self.empty else [self.item]}
        elif name == "read_evidence" and arguments["ref_id"] == self.item["ref_id"]:
            result = self.item
        else:
            raise AssertionError("forbidden fixture tool")
        self.cache[logical_key] = copy.deepcopy(result)
        return copy.deepcopy(result)


class ProviderFixture:
    def __init__(
        self,
        host,
        *,
        infeasible=False,
        fabricated=False,
        conflict=None,
        forged_position=False,
        change_intake=False,
        revoke_before_publish=False,
    ):
        self.host = host
        self.infeasible, self.fabricated = infeasible, fabricated
        self.conflict, self.forged_position = conflict, forged_position
        self.change_intake, self.revoke_before_publish = change_intake, revoke_before_publish
        self.repaired = False
        self.sent = []

    def experiment(self):
        return {
            "id": "E1",
            "research_question_id": "RQ1",
            "hypothesis": "Proposed synthetic hypothesis, not a result.",
            "data_and_access": "Use the user-confirmed synthetic fixture dataset; access not newly granted.",
            "method_and_independent_variables": "Proposed controlled change in one method factor.",
            "baselines_and_controls": "Original synthetic baseline."
            if not self.repaired
            else "Revised synthetic baseline with the same split.",
            "metrics_and_units": "Proposed error rate, percent.",
            "repetitions_and_uncertainty": "Proposed repeated seeds and confidence intervals.",
            "resources_and_upper_bound": "Proposed one CPU; estimate only, not executed.",
            "risks_and_stopping_rules": "Stop on data leakage or exceeding confirmed resources.",
            "evidence_ref_ids": [self.host.tools.item["ref_id"]],
            "status": "proposed_not_executed",
        }

    async def complete(self, payload, logical_key):
        await self.host.tools.check_scope()
        prompt = json.loads(payload["input"][0]["content"])
        name = payload["text"]["format"]["name"]
        self.sent.append({"name": name, "logical_key": logical_key, "prompt": prompt})
        self.host.execution.value["model_calls"] += 1
        self.host.execution.value["input_tokens"] += 120
        self.host.execution.value["output_tokens"] += 50
        identity = prompt["canonical_brief"]["research_intake"]["brief_sha256"]
        if name == "citequest_research_outline":
            value = {
                "brief_sha256": identity,
                "questions": [
                    {
                        "id": "RQ1",
                        "question": "Can the proposed synthetic control improve the chosen metric?",
                        "retrieval_query": "synthetic baseline evidence",
                        "experiment_ids": ["E1"],
                        "user_fields": ["research_questions", "resources"],
                        "expected_outputs": ["A proposed controlled experiment, not a measured result."],
                    }
                ],
                "innovation_hypotheses": ["Proposed synthetic novelty, not established."],
                "uncovered_user_requests": [],
            }
        elif name == "citequest_research_literature":
            source = self.host.tools.item
            value = {
                "kind": "qa",
                "requested_direction": "neutral",
                "claims": [
                    {
                        "id": "C1",
                        "statement": "The fixture describes a baseline using the same split.",
                        "status": "supported",
                        "edges": [
                            {
                                "ref_id": source["ref_id"],
                                "quote": "The proposed baseline uses the same data split.",
                                "relation": "supports",
                                "rationale": "Synthetic direct quote only.",
                                "conditions": ["No academic meaning."],
                            }
                        ],
                        "limitations": ["Explicit fixture."],
                    }
                ],
                "gaps": [],
                "retrieval_scope": "Explicit synthetic bounded source.",
                "evidence_status": "supported",
            }
            if self.change_intake:
                self.change_intake = False
                self.host.intake["user_confirmed"]["resources"] = (
                    "Corrected synthetic resource constraint: one CPU, 50 CNY."
                )
                self.host.intake["brief_revision"] += 1
                self.host.intake["brief_sha256"] = digest(self.host.intake["user_confirmed"])
        elif name == "citequest_research_literature_review":
            value = {
                "claims": [
                    {
                        "claim_id": "C1",
                        "verdict": "entailed",
                        "edge_ref_ids": [self.host.tools.item["ref_id"]],
                        "numeric_and_conditions_preserved": True,
                        "explanation": "Explicit fixture review.",
                    }
                ],
                "source_instruction_injection_detected": False,
                "requested_stance_bias_detected": False,
            }
        elif name == "citequest_research_method":
            value = {
                "brief_sha256": identity,
                "experiment": self.experiment(),
                "resource_assessment": "exceeds_declared_resources" if self.infeasible else "within_declared_resources",
                "reasons": ["Explicit synthetic feasibility estimate only."],
            }
        elif name == "citequest_research_review":
            target_revision = self.conflict == "revision" and not self.repaired
            target_input = self.conflict == "input"
            index = prompt["handoff_index"]
            conflicts = []
            if target_revision or target_input:
                user = next(item for item in index if item["role"] == "user")
                method = next(item for item in index if item["role"] == "methods")
                conflicts = [
                    {
                        "id": "X1",
                        "target_id": "E1",
                        "positions": [
                            {
                                "role": "user",
                                "handoff_sha256": user["sha256"],
                                "quote": "FORGED ROLE QUOTE"
                                if self.forged_position
                                else prompt["canonical_brief"]["research_intake"]["user_confirmed"]["resources"],
                            },
                            {
                                "role": "methods",
                                "handoff_sha256": method["sha256"],
                                "quote": "Proposed one CPU; estimate only, not executed.",
                            },
                        ],
                        "basis": "user_constraint",
                        "action": "needs_user_input" if target_input else "targeted_revision",
                        "explanation": "Explicit role conflict control; not an actual disagreement.",
                    }
                ]
            value = {
                "content_sha256": prompt["content_sha256"],
                "scope": prompt["review_scope"],
                "review": {
                    "brief_sha256": identity,
                    "items": [
                        {
                            "target_id": target,
                            "verdict": "needs_input"
                            if target_input and target == "E1"
                            else "needs_revision"
                            if target_revision and target == "E1"
                            else "passed",
                            "reasons": ["Explicit synthetic review."],
                        }
                        for target in prompt["required_targets"]
                    ],
                    "fabricated_completed_work_detected": self.fabricated,
                    "unapproved_resource_or_data_access_detected": False,
                    "source_instruction_injection_detected": False,
                },
                "request_coverage": "complete",
                "uncovered_user_requests": [],
                "feasibility": "resource_infeasible" if self.infeasible else "proposed_feasible",
                "conflicts": conflicts,
            }
            if self.revoke_before_publish and prompt["review_scope"] == "cross_plan_ledger":
                self.host.tools.revoked = True
        elif name == "citequest_research_target_repair":
            self.repaired = True
            value = {
                "brief_sha256": identity,
                "target_id": prompt["target_id"],
                "replacement": self.experiment(),
                "resolved_conflict_ids": prompt["required_conflict_refs"],
                "evidence_ref_ids": [self.host.tools.item["ref_id"]],
                "basis": "current_evidence_and_user_constraints",
                "public_reason": "Named synthetic correction using confirmed constraints.",
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


class HostFixture:
    def __init__(self, *, missing=False, empty=False, model_limit=64, **options):
        self.loop = asyncio.get_running_loop()
        self.execution = ExecutionFixture(model_limit=model_limit)
        self.memory = MemoryFixture()
        self.intake = planning_input()
        if missing:
            self.intake["user_confirmed"].pop("resources")
        self.tools = ToolFixture(self, empty=empty)
        self.provider = ProviderFixture(self, **options)

    def wait(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    def loader(self):
        return copy.deepcopy(self.intake)


class ResearchWorkflowControls(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, *, revision_limit=2, **options):
        host = HostFixture(**options)
        flow = ResearchWorkflow(host, host.loader, checkpointer=InMemorySaver(), revision_limit=revision_limit)
        return host, flow, await asyncio.to_thread(flow.run)

    async def test_complete_roles_produce_linked_future_plan_and_writing_handoff(self):
        host, _, state = await self.run_case()
        result = state["result"]
        self.assertEqual(result["status"], "structured_plan_ready")
        self.assertEqual(host.execution.value["model_calls"], 6)
        self.assertFalse(result["performed_experiments"])
        self.assertFalse(result["school_format_ready"])
        self.assertEqual(result["plan"]["questions"][0]["experiment_ids"], [result["plan"]["experiments"][0]["id"]])
        self.assertEqual(result["chapter_plan"][0]["minimum_han"], 5000)
        self.assertEqual(result["remaining_same_task_budget"]["model_calls"], 58)
        self.assertEqual(result["remaining_same_task_budget"]["input_tokens"], 512000 - 720)
        self.assertEqual(result["remaining_same_task_budget"]["output_tokens"], 96000 - 300)
        self.assertEqual(result["remaining_same_task_budget"]["cash_ncny"], 80000000000)
        self.assertEqual(result["research_intake"]["brief_sha256"], result["brief_identity"]["brief_sha256"])
        self.assertTrue(result["evidence_ledger"])
        self.assertEqual(result["kind"], "research")
        self.assertTrue(result["citations"])
        self.assertEqual(result["citations"][0]["ref_id"], host.tools.item["ref_id"])
        self.assertEqual(result["context_manifest"]["brief_sha256"], result["brief_identity"]["brief_sha256"])

    async def test_missing_information_interrupts_before_any_model(self):
        host, _, state = await self.run_case(missing=True)
        self.assertTrue(state.get("__interrupt__"))
        self.assertEqual(host.execution.value["model_calls"], 0)
        self.assertEqual(len(host.tools.calls), 0)

    async def test_open_intake_conflict_blocks_models_without_changing_confirmed_facts(self):
        host = HostFixture()
        original = host.intake["user_confirmed"]["resources"]
        host.intake["open_conflicts"] = [
            {
                "id": str(uuid4()),
                "field": "resources",
                "summary": "A user-origin resource proposal conflicts with the current confirmed value.",
            }
        ]
        flow = ResearchWorkflow(host, host.loader, checkpointer=InMemorySaver())
        state = await asyncio.to_thread(flow.run)
        self.assertTrue(state.get("__interrupt__"))
        self.assertEqual(host.execution.value["model_calls"], 0)
        self.assertEqual(host.intake["user_confirmed"]["resources"], original)
        self.assertTrue(state["__interrupt__"][0].value["open_conflicts"])

    async def test_free_text_resume_cannot_confirm_unknown_resource_fact(self):
        host, flow, _ = await self.run_case(missing=True)
        host.execution.command = {"id": uuid4(), "kind": "resume"}
        state = await asyncio.to_thread(flow.run)
        self.assertTrue(state.get("__interrupt__"))
        self.assertEqual(host.execution.value["model_calls"], 0)
        self.assertNotIn("resources", host.intake["user_confirmed"])

    async def test_authorized_brief_update_resumes_same_task_without_reset(self):
        host, flow, _ = await self.run_case(missing=True)
        identifier = host.execution.task_id
        host.intake["user_confirmed"]["resources"] = "Explicit synthetic one CPU and 100 CNY."
        host.intake["brief_revision"] += 1
        host.intake["brief_sha256"] = digest(host.intake["user_confirmed"])
        host.execution.command = {"id": uuid4(), "kind": "resume"}
        state = await asyncio.to_thread(flow.run)
        self.assertEqual(state["result"]["status"], "structured_plan_ready")
        self.assertEqual(host.execution.task_id, identifier)
        self.assertEqual(state["result"]["brief_identity"]["brief_revision"], 2)

    async def test_empty_evidence_waits_without_inventing_literature(self):
        host, _, state = await self.run_case(empty=True)
        self.assertTrue(state.get("__interrupt__"))
        self.assertEqual([item["name"] for item in host.provider.sent], ["citequest_research_outline"])

    async def test_impossible_resources_are_not_declared_feasible(self):
        _, _, state = await self.run_case(infeasible=True)
        self.assertEqual(state["result"]["feasibility"], "resource_infeasible")
        self.assertEqual(state["result"]["status"], "partial")

    async def test_fabricated_completed_work_rejects_plan(self):
        _, _, state = await self.run_case(fabricated=True)
        self.assertEqual(state["result"]["publication_status"], "rejected")

    async def test_unresolved_role_conflict_requests_input_instead_of_voting(self):
        host, _, state = await self.run_case(conflict="input")
        self.assertTrue(state.get("__interrupt__"))
        self.assertFalse(any(item["name"] == "citequest_research_target_repair" for item in host.provider.sent))

    async def test_targeted_repair_is_followed_by_full_target_rereview(self):
        host, _, state = await self.run_case(conflict="revision")
        self.assertEqual(state["result"]["status"], "structured_plan_ready")
        self.assertEqual(host.execution.value["model_calls"], 9)
        self.assertIn("Revised", state["result"]["plan"]["experiments"][0]["baselines_and_controls"])
        self.assertEqual(
            len([item for item in host.provider.sent if item["name"] == "citequest_research_target_repair"]), 1
        )
        self.assertEqual(len(state["result"]["review_history"]), 4)
        self.assertTrue(
            any(
                target["verdict"] == "needs_revision"
                for item in state["result"]["review_history"]
                for target in item["review"]["review"]["items"]
            )
        )

    async def test_forged_conflict_role_quote_is_rejected(self):
        with self.assertRaises(AgentError) as caught:
            await self.run_case(conflict="revision", forged_position=True)
        self.assertEqual(caught.exception.code, "research_conflict_position_not_in_role_handoff")

    async def test_budget_admission_is_before_model_generation(self):
        host, _, state = await self.run_case(model_limit=5)
        self.assertEqual(state["result"]["status"], "partial")
        self.assertEqual(len(host.provider.sent), 0)
        self.assertTrue(state.get("__interrupt__"))

    async def test_reviewer_rejection_cannot_bypass_revision_limit(self):
        _, _, state = await self.run_case(conflict="revision", revision_limit=0)
        self.assertEqual(state["result"]["publication_status"], "rejected")
        self.assertIn("bounded_targeted_revision_exhausted", state["result"]["reasons"])

    async def test_source_revocation_before_publication_stops_plan(self):
        with self.assertRaises(AgentError) as caught:
            await self.run_case(revoke_before_publish=True)
        self.assertEqual(caught.exception.code, "EXPLICIT_FIXTURE_research_source_revoked")

    async def test_changed_conditions_restart_intake_and_keep_spent_calls(self):
        host, _, state = await self.run_case(change_intake=True)
        self.assertEqual(state["result"]["status"], "structured_plan_ready")
        self.assertEqual(state["result"]["brief_identity"]["brief_revision"], 2)
        self.assertEqual(host.execution.value["model_calls"], 8)
        self.assertTrue(state["history"])

    async def test_completed_followup_keeps_same_task_cumulative_budget(self):
        host, flow, first = await self.run_case()
        host.intake["user_confirmed"]["resources"] = "User explicitly corrected the synthetic budget to 50 CNY."
        host.intake["brief_revision"] += 1
        host.intake["brief_sha256"] = digest(host.intake["user_confirmed"])
        host.execution.command = {"id": uuid4(), "kind": "followup"}
        host.execution.run_id = uuid4()
        second = await asyncio.to_thread(flow.run)
        self.assertEqual(first["result"]["brief_identity"]["brief_revision"], 1)
        self.assertEqual(second["result"]["brief_identity"]["brief_revision"], 2)
        self.assertEqual(host.execution.value["model_calls"], 12)

    async def test_role_tool_allowlist_rejects_method_tool_access(self):
        host = HostFixture()
        flow = ResearchWorkflow(host, host.loader, checkpointer=InMemorySaver())
        with self.assertRaises(AgentError):
            flow.tool({"intake": host.intake}, "methods", "search_evidence", {"query": "forbidden"}, "forbidden")
        self.assertEqual(len(host.tools.calls), 0)

    async def test_intake_selection_cannot_widen_or_replace_task_tool_scope(self):
        host = HostFixture()
        host.intake["selected_sources"] = [{"kind": "native_representation", "id": "d" * 64}]
        flow = ResearchWorkflow(host, host.loader, checkpointer=InMemorySaver())
        state = await asyncio.to_thread(flow.run)
        self.assertTrue(state.get("__interrupt__"))
        self.assertEqual(state["__interrupt__"][0].value["kind"], "scope_sync_required")
        self.assertEqual(len(host.provider.sent), 0)
        self.assertEqual(len(host.tools.calls), 0)

    async def test_explicit_scope_sync_resumes_without_new_task_or_scope_fallback(self):
        host = HostFixture()
        host.intake["selected_sources"] = [{"kind": "native_representation", "id": "d" * 64}]
        flow = ResearchWorkflow(host, host.loader, checkpointer=InMemorySaver())
        first = await asyncio.to_thread(flow.run)
        self.assertTrue(first.get("__interrupt__"))
        host.execution.value["brief"]["selected_papers"] = copy.deepcopy(host.intake["selected_sources"])
        host.execution.command = {"id": uuid4(), "kind": "resume"}
        state = await asyncio.to_thread(flow.run)
        self.assertEqual(state["result"]["status"], "structured_plan_ready")
        self.assertEqual(host.execution.value["model_calls"], 6)

    async def test_standard_api_envelope_retains_memory_and_all_literal_dependencies(self):
        host = HostFixture()
        flow = ResearchWorkflow(host, host.loader, checkpointer=InMemorySaver())
        memory = {"id": str(uuid4()), "revision": 2}
        ref = host.tools.item["ref_id"]
        result = flow.project_result(
            {
                "intake": host.intake,
                "hard_memory_refs": [memory],
                "evidence": [host.tools.item],
                "literature": [
                    {
                        "answer": {
                            "claims": [
                                {"edges": [{"ref_id": ref, "quote": "This is synthetic control evidence."}]},
                                {
                                    "edges": [
                                        {"ref_id": ref, "quote": "The proposed baseline uses the same data split."},
                                        {"ref_id": ref, "quote": "This is synthetic control evidence."},
                                    ]
                                },
                            ]
                        }
                    }
                ],
            },
            {"status": "failed", "publication_status": "rejected"},
        )
        self.assertEqual(len(result["citations"]), 2)
        self.assertEqual(result["context_manifest"]["memory_refs"], [memory])
        self.assertEqual(result["evidence_status"], "insufficient")
        self.assertEqual(result["publication_status"], "rejected")
        self.assertNotIn("data split", result["text"])

    async def test_checkpoint_recovery_does_not_repeat_completed_method(self):
        host = HostFixture()
        saver = InMemorySaver()
        original, crashes = ResearchWorkflow.assemble_plan, []

        def crash_once(flow, state):
            if not crashes:
                crashes.append(True)
                raise AgentError("EXPLICIT_FIXTURE_crash_after_methods")
            return original(flow, state)

        with patch.object(ResearchWorkflow, "assemble_plan", crash_once):
            flow = ResearchWorkflow(host, host.loader, checkpointer=saver)
            with self.assertRaises(AgentError):
                await asyncio.to_thread(flow.run)
            self.assertEqual(host.execution.value["model_calls"], 4)
            host.execution.run_id = uuid4()
            restored = ResearchWorkflow(host, host.loader, checkpointer=saver)
            state = await asyncio.to_thread(restored.run)
        self.assertEqual(state["result"]["status"], "structured_plan_ready")
        self.assertEqual(host.execution.value["model_calls"], 6)
        self.assertEqual(len([item for item in host.provider.sent if item["name"] == "citequest_research_method"]), 1)
