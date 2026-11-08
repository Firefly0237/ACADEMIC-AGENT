"""Completed known-content failures use existing credits; no wire/PG/model calls."""
from __future__ import annotations

import asyncio
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.graph import AgentGraph
from citequest.agents_v3.qa_v5 import QA5
from citequest.research_v3.typed_feedback import strict_object

sys.path.insert(0, str(Path(__file__).parent))
import test_qa_v5 as fixtures

ROOT = Path(__file__).resolve().parents[2]


def invalid_json(kind):
    if kind == "eof":
        # Same observed syntax shape: 4349 chars, outer object left open,
        # final char is '}', error at EOF. No real model text is copied.
        prefix, suffix = '{"review":{"note":"', '"}'
        return prefix + "x" * (4349 - len(prefix) - len(suffix)) + suffix
    return {"fence":'```json\n{}\n```', "duplicate":'{"x":1,"x":2}',
        "nonfinite":'{"x":NaN}', "array":'[]'}[kind]


class FormatProvider(fixtures.Provider):
    """An explicit model double may regenerate a whole unreadable first result."""
    def __init__(self, execution, tools, *, faults=None, stop_role=None, **kwargs):
        super().__init__(execution, tools, **kwargs)
        self.faults, self.stop_role = faults or {}, stop_role
        self.templates, self.mutated = {}, set()

    async def complete(self, payload, key):
        role = payload["text"]["format"]["name"].removeprefix("citequest_qa5_")
        if role == self.stop_role:
            raise AgentError("provider_usage_missing")
        if key in self.execution.known:
            return await super().complete(payload, key)
        prompt = json.loads(payload["input"][0]["content"])
        unreadable_previous = False
        if role == "author" and prompt.get("untrusted_previous_candidate"):
            try:
                strict_object(prompt["untrusted_previous_candidate"])
            except AgentError:
                unreadable_previous = True
        if unreadable_previous:
            # Fixture output factory uses its previous complete template;
            # it does not parse, complete or publish the malformed string.
            self.calls[role] += 1
            self.requests.append({"request":copy.deepcopy(payload), "key":key, "role":role})
            self.execution.value["model_calls"] += 1
            self.execution.value["input_tokens"] += 100
            self.execution.value["output_tokens"] += 40
            response = copy.deepcopy(self.templates[role])
            self.execution.known[key] = {"id":uuid4(), "request":copy.deepcopy(payload), "response":copy.deepcopy(response)}
        else:
            response = await super().complete(payload, key)
            self.templates.setdefault(role, copy.deepcopy(response))
        choices = self.faults.get(role, [])
        choice = choices[self.calls[role] - 1] if len(choices) >= self.calls[role] else None
        if choice:
            response = copy.deepcopy(response)
            response["output"][0]["content"][0]["text"] = invalid_json(choice)
            self.execution.known[key]["response"] = copy.deepcopy(response)
            self.mutated.add(key)
        return response


async def run_case(*, faults=None, stop_role=None, execution=None, saver=None, provider=None, review_fault=None):
    execution = execution or fixtures.Execution()
    tools = provider.tools if provider else fixtures.Tools(execution)
    provider = provider or FormatProvider(execution, tools, faults=faults, stop_role=stop_role, review_fault=review_fault)
    with patch("citequest.agents_v3.graph.PostgresSaver", return_value=saver or InMemorySaver()), \
            patch.object(AgentGraph, "user_turns", lambda _:execution.turns):
        graph = AgentGraph(execution, provider, tools, fixtures.FixtureMemory(), ROOT / "runtime/s08/skills")
        result = await asyncio.to_thread(graph.run)
    return execution, provider, result


class QA5KnownFormatRepairControls(unittest.IsolatedAsyncioTestCase):
    async def test_observed_review_EOF_shape_uses_one_receipt_bound_credit_same_grid(self):
        raw = invalid_json("eof")
        self.assertEqual(len(raw), 4349)
        with self.assertRaises(json.JSONDecodeError) as raised:
            json.loads(raw)
        self.assertEqual(raised.exception.pos, 4349)
        execution, provider, result = await run_case(faults={"review":["eof"]})
        self.assertEqual(provider.calls, {"plan":1, "author":1, "review":2})
        self.assertEqual(execution.content_revision_count(), 1)
        credit = next(iter(execution.credits.values()))
        self.assertIn(credit["identity"], [digest({"phase":"qa5_content_revision", "error_codes":["typed_feedback_initial_json_unreadable"],
            "source_logical_key":key}) for key in provider.mutated])
        reviews = [json.loads(item["request"]["input"][0]["content"]) for item in provider.requests if item["role"] == "review"]
        self.assertEqual(reviews[0]["review_grid"], reviews[1]["review_grid"])
        self.assertEqual(reviews[0]["candidate"], reviews[1]["candidate"])
        self.assertEqual(reviews[0]["untrusted_evidence_data"], reviews[1]["untrusted_evidence_data"])
        self.assertTrue(all(not item["selected_user_skills"] for item in reviews))
        self.assertEqual(result["result"]["publication_status"], "validated_ai_review")

    async def test_exact_JSON_content_errors_repair_each_role_without_coercion(self):
        for role in ("plan", "author", "review"):
            for kind in ("eof", "fence", "duplicate", "nonfinite", "array"):
                with self.subTest(role=role, kind=kind):
                    execution, provider, result = await run_case(faults={role:[kind]})
                    self.assertEqual(execution.content_revision_count(), 1)
                    self.assertEqual(provider.calls[role], 2)
                    self.assertEqual(result["result"]["publication_status"], "validated_ai_review")
                    self.assertEqual(len(provider.mutated), 1)

    async def test_repeated_mixed_format_errors_exhaust_two_Task_credits(self):
        execution = fixtures.Execution()
        tools = fixtures.Tools(execution)
        provider = FormatProvider(execution, tools, faults={"review":["eof", "duplicate", "nonfinite"]})
        with self.assertRaises(AgentError) as raised:
            await run_case(execution=execution, provider=provider)
        self.assertEqual(raised.exception.code, "qa5_content_or_goal_review_unresolved")
        self.assertEqual(execution.content_revision_count(), 2)
        self.assertEqual(provider.calls, {"plan":1, "author":1, "review":3})
        self.assertFalse(any(item.get("state") == "succeeded" for item in execution.finishes))

    async def test_unreadable_author_keeps_existing_directed_lock_and_baseline(self):
        execution, provider, result = await run_case(faults={"author":[None, "eof"]}, review_fault="semantic")
        self.assertEqual(execution.content_revision_count(), 2)
        authors = [json.loads(item["request"]["input"][0]["content"]) for item in provider.requests if item["role"] == "author"]
        self.assertEqual(len(authors), 3)
        self.assertEqual(authors[1]["repair_contract"], authors[2]["repair_contract"])
        self.assertTrue(authors[2]["repair_contract"]["required_claim_ids"])
        self.assertEqual(result["qa5"]["original_propositions"]["source"], "first_candidate_not_factual_authority")

    async def test_unknown_transport_never_consumes_content_credit(self):
        for role in ("plan", "author", "review"):
            with self.subTest(role=role):
                execution = fixtures.Execution()
                with self.assertRaises(AgentError) as raised:
                    await run_case(stop_role=role, execution=execution)
                self.assertEqual(raised.exception.code, "provider_usage_missing")
                self.assertEqual(execution.content_revision_count(), 0)

    async def test_format_repair_does_not_increase_remaining_output_cap(self):
        execution = fixtures.Execution()
        boundary = object.__new__(QA5)
        boundary.graph = type("ExplicitGraph", (), {"execution":execution})()
        execution.value["output_tokens"] = execution.value["profile"]["output_tokens"] - 1
        with self.assertRaises(AgentError) as raised:
            boundary.request({"qa5":{"revision":1}}, "review", {"type":"object"}, {"context_manifest":{}})
        self.assertEqual(raised.exception.code, "qa5_complete_remaining_protocol_output_budget")
        self.assertFalse(execution.known)
        self.assertEqual(execution.content_revision_count(), 0)

    async def test_credit_commit_before_checkpoint_replays_same_credit(self):
        execution, saver = fixtures.Execution(), InMemorySaver()
        provider = FormatProvider(execution, fixtures.Tools(execution), faults={"review":["eof"]})
        original, crash = QA5.repair, {"pending":True}

        def stop_after_commit(boundary, state):
            result = original(boundary, state)
            if crash["pending"]:
                crash["pending"] = False
                raise RuntimeError("EXPLICIT death after content credit commit")
            return result

        with patch.object(QA5, "repair", stop_after_commit), self.assertRaises(RuntimeError):
            await run_case(execution=execution, saver=saver, provider=provider)
        self.assertEqual(execution.content_revision_count(), 1)
        _, _, result = await run_case(execution=execution, saver=saver, provider=provider)
        self.assertEqual(execution.content_revision_count(), 1)
        self.assertEqual(provider.calls, {"plan":1, "author":1, "review":2})
        self.assertEqual(result["result"]["publication_status"], "validated_ai_review")


class QA5FormatAllowlistControls(unittest.TestCase):
    def test_nonformat_AgentErrors_propagate_from_each_parser(self):
        boundary = object.__new__(QA5)
        boundary.graph = None
        state = {"qa5":{"plan_raw":"{}", "author_raw":"{}", "review_raw":"{}"}}
        for code in ("provider_usage_missing", "tool_acl_denied", "quote_source_version_changed", "typed_feedback_unregistered_error"):
            for method in ("validate_plan", "validate_author", "validate_review"):
                with self.subTest(code=code, method=method), \
                        patch("citequest.agents_v3.qa_v5.strict_object", side_effect=AgentError(code)), \
                        patch.object(boundary, "scope"), patch("citequest.agents_v3.qa_v5.ReviewGrid"), \
                        patch("citequest.agents_v3.qa_v5.combined_review_model"):
                    with self.assertRaises(AgentError) as raised:
                        getattr(boundary, method)({**state, "answer":{}})
                    self.assertEqual(raised.exception.code, code)
