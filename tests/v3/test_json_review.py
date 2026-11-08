"""Three focused JSON-mode controls using explicit fakes and no network."""
import asyncio
import copy
import json

import httpx2
import pytest

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.json_review import COMPILER, VERSION, compile_json_review, normalize_json_review, recorded_json_review
from citequest.agents_v3.model_contracts import model_contract
from citequest.agents_v3.provider import POLICY_MARKER, ResponsesProvider
from citequest.agents_v3.provider_contract import wire_contract_violation
from citequest.agents_v3.strict_chat import ENVELOPE_GATE_KEY
from citequest.agents_v3.typed_submission import compile_submission, normalize_submission, recorded_submission


def profile():
    return {**model_contract("deepseek-v4-pro"), "profile_id": "qa-v7", "qa_execution_protocol": "grounded3",
        "review_transport_protocol": "json1", "review_api_protocol": VERSION,
        "structured_output_protocol": "chat-functions1", "answer_revision_protocol": "jsonpatch1",
        "request_bytes": 131072}


def request(role="review"):
    return {"model": "deepseek-v4-pro", "reasoning": {"effort": "none"}, "truncation": "disabled",
        "instructions": POLICY_MARKER + " Explicit JSON transport fixture.", "max_output_tokens": 4096,
        "input": [{"role": "user", "content": '{"context_manifest":{}}'}],
        "text": {"format": {"type": "json_schema", "name": "citequest_qa5_" + role, "strict": True,
            "schema": {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["accepted", "rejected"]}},
                "required": ["verdict"], "additionalProperties": False}}}}


def chat(content='{"verdict":"rejected"}'):
    return {"id": "json-fixture", "object": "chat.completion", "created": 1794000000,
        "model": "deepseek-v4-pro", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
            "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 8}}


def test_json_mode_is_syntax_only_and_preserves_full_schema_and_old_role_wire():
    payload = request()
    before = copy.deepcopy(payload)
    compiled = compile_submission(profile(), payload, "explicit:qa5:review:0")
    assert payload == before and compiled == compile_json_review(payload)
    assert compiled.wire["response_format"] == {"type": "json_object"}
    assert "tools" not in compiled.wire and "tool_choice" not in compiled.wire
    assert compiled.receipt["compiler"] == COMPILER and compiled.receipt["protocol"] == VERSION
    assert compiled.receipt["schema_enforcement"] == "local_authoritative"
    assert compiled.receipt["wire_schema_sha256"] == digest({"type": "object"})
    assert compiled.receipt["full_schema_sha256"] == digest(payload["text"]["format"]["schema"])
    assert "function_name" not in compiled.receipt
    for role in ("plan", "author"):
        old = {**profile(), "review_transport_protocol": "function1", "review_api_protocol": "chat-strict2"}
        assert compile_submission(profile(), request(role), f"explicit:qa5:{role}:0") == compile_submission(old, request(role), f"explicit:qa5:{role}:0")
    # A schema-invalid object belongs to unchanged local domain validation,
    # not the provider's JSON-syntax recovery budget.
    raw = chat('{"unexpected":true}')
    recorded = recorded_submission(profile(), payload, "explicit:qa5:review:0", raw, raw, 200)
    value = normalize_submission(profile(), payload, "explicit:qa5:review:0", recorded)
    assert value["output"][0]["content"][0]["text"] == '{"unexpected":true}'
    assert "function_call" not in json.dumps(value)
    assert wire_contract_violation(payload, recorded, profile=profile(), logical_key="explicit:qa5:review:0") is None
    malformed = chat('{"verdict":"rejected"}}')
    recorded = recorded_json_review(malformed, malformed, 200)
    violation = wire_contract_violation(payload, recorded, profile=profile(), logical_key="explicit:qa5:review:0")
    assert violation["code"] == "json_object_wire_contract_violation" and violation["category"] == "invalid_json"
    assert recorded["choices"][0]["message"]["content"] == malformed["choices"][0]["message"]["content"]


def test_json_gate_rejects_blank_length_refusal_thinking_http_and_forged_recorded_gate():
    cases = []
    blank = chat("   ")
    cases.append((blank, 200, "chat_json_blank_content"))
    length = chat()
    length["choices"][0]["finish_reason"] = "length"
    cases.append((length, 200, "chat_json_response_not_completed"))
    refusal = chat()
    refusal["choices"][0]["message"]["refusal"] = "Explicit synthetic refusal."
    cases.append((refusal, 200, "chat_json_message_invalid"))
    thinking = chat()
    thinking["choices"][0]["message"]["reasoning_content"] = "Synthetic private reasoning."
    cases.append((thinking, 200, "chat_json_thinking_output_forbidden"))
    cases.append((chat(), 500, "provider_http_error_with_known_usage"))
    forged = chat()
    forged[ENVELOPE_GATE_KEY] = {"protocol": VERSION, "accepted": True, "error_code": None}
    cases.append((forged, 200, "chat_json_reserved_receive_gate"))
    for raw, status, code in cases:
        stored = recorded_json_review(raw, raw, status)
        # Persisted rejection survives even if redaction removes reasoning.
        stored["choices"][0]["message"].pop("reasoning_content", None)
        with pytest.raises(AgentError) as error:
            normalize_json_review(stored)
        assert error.value.code == code
    with pytest.raises(AgentError):
        normalize_json_review(chat())


def test_locked_sdk_uses_standard_chat_and_settled_replay_without_function_submission():
    class Execution:
        def __init__(self):
            self.row = None
            self.observed = []
        def snapshot(self):
            return {"profile": profile()}
        def reserve(self, key, payload):
            if self.row is None:
                self.row = {"id": key, "status": "reserved", "request_hash": digest(payload)}
            assert self.row["request_hash"] == digest(payload)
            return self.row
        def observe_model_wire(self, attempt_id, receipt):
            self.observed.append(receipt)
        def mark_sent(self, attempt_id):
            return 60
        def observe_model_http_receipt(self, *_args):
            pass
        def settle(self, attempt_id, response, usage):
            self.row.update(status="settled", response=copy.deepcopy(response), usage=usage)
        def unknown(self, *_args):
            raise AssertionError("No unknown usage expected in this explicit fixture.")
        def not_sent(self, *_args, **_kwargs):
            raise AssertionError("The guarded fixture must reach the mocked wire.")

    async def run():
        execution = Execution()
        calls = []
        def handler(wire):
            calls.append(wire)
            assert str(wire.url) == "https://s08-provider.invalid/chat/completions"
            body = json.loads(wire.content)
            assert body["response_format"] == {"type": "json_object"} and "tools" not in body
            return httpx2.Response(200, json=chat())
        provider = ResponsesProvider(execution, fixture=True, transport=httpx2.MockTransport(handler))
        try:
            first = await provider.complete(request(), "explicit:qa5:review:0")
            assert await provider.complete(request(), "explicit:qa5:review:0") == first
            assert len(calls) == len(execution.observed) == 1
            assert first["output"][0]["type"] == "message"
            assert execution.row["usage"] == {"input": 10, "cached": 2, "output": 4}
            assert execution.row["response"][ENVELOPE_GATE_KEY]["protocol"] == VERSION
        finally:
            await provider.close()
    asyncio.run(run())
