"""Locked SDK schema compiler and real QA5 request-builder controls; no network."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx2
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, ValidationError

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.provider import ResponsesProvider
from citequest.agents_v3.structured_output import strict_output_schema
from citequest.research_v3.qa_v5_contracts import SelectorAnswerV5

sys.path.insert(0, str(Path(__file__).parent))
import test_provider_http_receipts as receipt_fixtures
import test_qa_v5 as fixtures


def assert_strict_objects(test: unittest.TestCase, value: object) -> None:
    if isinstance(value, dict):
        if "properties" in value:
            test.assertEqual(set(value["properties"]), set(value.get("required", [])))
            test.assertIs(value["additionalProperties"], False)
        for child in value.values():
            assert_strict_objects(test, child)
    elif isinstance(value, list):
        for child in value:
            assert_strict_objects(test, child)


class StrictOutputSchemaControls(unittest.TestCase):
    def test_pydantic_defaults_are_required_without_adding_null_or_mutating_input(self):
        original = SelectorAnswerV5.model_json_schema()
        before = copy.deepcopy(original)
        self.assertEqual(set(original["properties"]) - set(original["required"]), {"kind", "requested_direction"})
        result = strict_output_schema(original)
        self.assertEqual(original, before)
        self.assertEqual(set(result["required"]), set(result["properties"]))
        self.assertEqual(result["properties"]["kind"], original["properties"]["kind"])
        self.assertEqual(result["properties"]["requested_direction"], original["properties"]["requested_direction"])
        assert_strict_objects(self, result)
        Draft202012Validator.check_schema(result)

    def test_dynamic_quote_enum_and_all_field_constraints_are_retained(self):
        original = SelectorAnswerV5.model_json_schema()
        original["$defs"]["SelectorEdgeV5"]["properties"]["quote_id"]["enum"] = ["Q1", "Q37"]
        result = strict_output_schema(original)
        self.assertEqual(result["$defs"], original["$defs"])
        self.assertEqual(result["$defs"]["SelectorEdgeV5"]["properties"]["quote_id"]["enum"], ["Q1", "Q37"])
        candidate = {"kind":"qa", "requested_direction":"neutral", "claims":[{
            "id":"C1", "statement":"Explicit synthetic assertion.", "status":"supported", "limitations":[],
            "edges":[{"quote_id":"Q1", "relation":"supports", "rationale":"Synthetic fixture.", "conditions":[]}]}],
            "gaps":[], "retrieval_scope":"Synthetic fixture only.", "evidence_status":"supported"}
        validator = Draft202012Validator(result)
        validator.validate(candidate)
        SelectorAnswerV5.model_validate(candidate, strict=True)
        for field in ("kind", "requested_direction"):
            changed = copy.deepcopy(candidate)
            del changed[field]
            self.assertFalse(validator.is_valid(changed))
        changed = copy.deepcopy(candidate)
        changed["claims"][0]["edges"][0]["quote_id"] = "Q999"
        self.assertFalse(validator.is_valid(changed))
        changed = copy.deepcopy(candidate)
        changed["claims"][0]["edges"] = []
        with self.assertRaises(ValidationError):
            SelectorAnswerV5.model_validate(changed, strict=True)

    def test_original_nullable_type_stays_nullable_and_required(self):
        class Nullable(BaseModel):
            model_config = ConfigDict(extra="forbid")
            value: str | None = None

        original = Nullable.model_json_schema()
        result = strict_output_schema(original)
        self.assertEqual(result["required"], ["value"])
        self.assertEqual(result["properties"]["value"]["anyOf"], original["properties"]["value"]["anyOf"])
        self.assertNotIn("default", result["properties"]["value"])

    def test_sdk_lock_and_malformed_schema_fail_closed(self):
        with patch("citequest.agents_v3.structured_output.importlib.metadata.version", return_value="unapproved"):
            with self.assertRaises(AgentError) as raised:
                strict_output_schema({"type":"object", "properties":{}})
            self.assertEqual(raised.exception.code, "structured_output_sdk_lock_mismatch")
        with self.assertRaises(AgentError) as raised:
            strict_output_schema({"type":"object", "properties":{"field":42}})
        self.assertEqual(raised.exception.code, "structured_output_schema_compile_failed")


class QA5RequestSchemaControls(unittest.IsolatedAsyncioTestCase):
    async def test_official_sdk_sends_exact_precompiled_request_once(self):
        ledger, received = receipt_fixtures.Ledger(), []
        payload = receipt_fixtures.payload()
        payload["text"] = {"format":{"type":"json_schema", "name":"explicit_schema_control",
            "schema":strict_output_schema(SelectorAnswerV5.model_json_schema()), "strict":True}}

        async def respond(request):
            received.append(json.loads(request.content))
            return httpx2.Response(200, json=receipt_fixtures.completed())

        provider = ResponsesProvider(ledger, transport=httpx2.MockTransport(respond), fixture=True)
        try:
            await provider.complete(payload, str(ledger.command["id"]) + ":explicit-schema")
        finally:
            await provider.close()
        self.assertEqual(received, [payload])
        self.assertEqual(ledger.original_request, payload)
        self.assertEqual(ledger.row["status"], "settled")

    async def test_all_actual_qa5_roles_compile_before_hash_and_known_replay(self):
        runner = fixtures.QA5Controls()
        execution, provider, _, result = await runner.run_case()
        self.assertEqual(provider.calls, {"plan":1, "author":1, "review":1})
        for item in provider.requests:
            request = item["request"]
            schema = request["text"]["format"]["schema"]
            assert_strict_objects(self, schema)
            Draft202012Validator.check_schema(schema)
            self.assertIs(request["text"]["format"]["strict"], True)
            self.assertEqual(digest(request), execution.settled_model_request(item["key"])["request_hash"])
        author = next(item["request"] for item in provider.requests if item["role"] == "author")
        self.assertIn("kind", author["text"]["format"]["schema"]["required"])
        self.assertIn("requested_direction", author["text"]["format"]["schema"]["required"])
        self.assertEqual(result["result"]["publication_status"], "validated_ai_review")
        # The existing parent/child cold control asserts exact regenerated hash
        # and no second author call after a known receipt before checkpoint.
        await runner.test_cold_graph_after_known_author_replays_without_new_model()
