"""Small offline function1 contracts; synthetic fixture, zero provider/DB calls."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.function_output import function_review_request, review_submission, validate_profile_transport
from citequest.research_v3.typed_feedback import strict_object

sys.path.insert(0,str(Path(__file__).parent))
from test_qa5_protocol_binding import fixture


def request(schema):
    return {"model":"deepseek-flash","reasoning":{"effort":"none"},"instructions":"Explicit synthetic offline review.",
            "input":[{"role":"user","content":"Explicit synthetic fixture."}],"max_output_tokens":4096,
            "text":{"format":{"type":"json_schema","name":"citequest_qa5_review","strict":True,"schema":schema}}}


def response(arguments, **item_changes):
    return {"status":"completed","output":[{"type":"function_call","name":"submit_review","call_id":"explicit-fixture-call-id",
            "arguments":arguments,**item_changes}]}


class FunctionOutputControls(unittest.TestCase):
    def test_identical_complete_schema_and_original_request_preserved(self):
        model,grid,_,_ = fixture()
        original = request(model.model_json_schema())
        before = digest(original)
        transformed = function_review_request(original)
        self.assertEqual(digest(original),before)
        self.assertEqual(transformed["tools"][0]["parameters"],original["text"]["format"]["schema"])
        self.assertNotIn("text",transformed)
        self.assertNotIn("parallel_tool_calls",transformed)
        self.assertEqual(transformed["tool_choice"],{"type":"function","name":"submit_review"})
        self.assertEqual(len(grid.identities),8)
        self.assertEqual(sum(map(len,grid.identities.values())),20)

    def test_envelope_then_original_strict_Pydantic_contract_and_vendor_id(self):
        model,_,_,value = fixture()
        raw = json.dumps(value)
        submitted = review_submission(response(raw))
        self.assertEqual(submitted.arguments,raw)
        self.assertEqual(submitted.call_id,"explicit-fixture-call-id")
        self.assertEqual(model.model_validate(strict_object(submitted.arguments),strict=True).model_dump(),value)
        self.assertFalse(submitted.public_transport_metadata()["external_MCP_execution"])

    def test_multiple_wrong_incomplete_message_and_blank_call_id_rejected(self):
        good = response("{}")
        bad = [response("{}",call_id=" "),response("{}",name="different"),response("{}",status="incomplete"),
               {**good,"status":"incomplete"},{**good,"incomplete_details":{"reason":"max_output_tokens"}},
               {**good,"output":good["output"]*2},{**good,"output":[{"type":"message","content":[]}]}]
        for value in bad:
            with self.subTest(value=value),self.assertRaises(AgentError):
                review_submission(value)

    def test_arguments_never_repaired_and_original_goal_Q_gates_remain(self):
        model,_,_,value = fixture()
        for raw in ('{"broken":1', '{"a":1,"a":2}'):
            submitted = review_submission(response(raw))
            self.assertEqual(submitted.arguments,raw)
            with self.assertRaises(AgentError):
                strict_object(submitted.arguments)
        wrong = copy.deepcopy(value)
        wrong["candidate_sha256"] = "0"*64
        with self.assertRaises(ValueError):
            model.model_validate(strict_object(review_submission(response(json.dumps(wrong))).arguments),strict=True)
        wrong = copy.deepcopy(value)
        wrong["claims"]["C1"]["quotes"].pop(next(iter(wrong["claims"]["C1"]["quotes"])))
        with self.assertRaises(ValueError):
            model.model_validate(strict_object(review_submission(response(json.dumps(wrong))).arguments),strict=True)

    def test_immutable_profile_guard_is_noop_for_legacy_and_never_mutates_wire(self):
        model,_,_,_ = fixture()
        old = request(model.model_json_schema())
        original = digest(old)
        validate_profile_transport({},old)
        self.assertEqual(digest(old),original)
        new = function_review_request(old)
        original = digest(new)
        validate_profile_transport({"review_transport_protocol":"function1"},new)
        self.assertEqual(digest(new),original)
        with self.assertRaises(AgentError):
            validate_profile_transport({},new)
        with self.assertRaises(AgentError):
            validate_profile_transport({"review_transport_protocol":"function1"},old)


if __name__ == "__main__":
    unittest.main()
