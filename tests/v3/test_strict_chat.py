"""Targeted strict transport controls. All HTTP and ledger state are explicit fakes."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace

import httpx2
import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.provider import POLICY_MARKER, ResponsesProvider
from citequest.agents_v3.model_contracts import model_contract
from citequest.agents_v3.strict_chat import (
    ENVELOPE_GATE_KEY,
    FIXTURE_ENDPOINT,
    compile_review_request,
    compile_schema,
    normalize_chat_response,
    usage_counts_chat,
)
from citequest.research_v3.typed_feedback import strict_object


def canonical_review() -> dict:
    return {"model": "deepseek-flash", "reasoning": {"effort": "none"}, "truncation": "disabled",
        "instructions": POLICY_MARKER + " Return the complete independent review.", "max_output_tokens": 4096,
        "input": [{"role": "user", "content": '{"context_manifest":{}}'}],
        "tools": [{"type": "function", "name": "submit_review", "description": "Review submission",
            "parameters": {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["passed", "rejected"]}},
                "required": ["verdict"], "additionalProperties": False}}],
        "tool_choice": {"type": "function", "name": "submit_review"}}


def raw_chat(arguments: str = '{"verdict":"passed"}') -> dict:
    return {"id": "chat_fixture_1", "object": "chat.completion", "created": 1794000000, "model": "deepseek-flash",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "actual_call_1", "type": "function", "function": {"name": "submit_review", "arguments": arguments}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
            "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 8, "prompt_tokens_details": {"cached_tokens": 2}}}


class FakeExecution:
    def __init__(self, *, strict: bool = True) -> None:
        self.profile = {**model_contract("deepseek-flash"), "profile_id": "qa-v6", "qa_execution_protocol": "grounded1",
            "review_transport_protocol": "function1", "request_bytes": 131072}
        if strict:
            self.profile["review_api_protocol"] = "chat-strict1"
        self.attempt = None
        self.reservations = 0
        self.wire_receipts = []
        self.http_receipts = []
        self.unknowns = []
        self.not_sent_attempts = []

    def snapshot(self):
        return {"profile": self.profile}

    def reserve(self, key, request):
        if self.attempt is not None:
            assert self.attempt["request_hash"] == digest(request)
            if self.attempt["status"] == "unknown":
                raise AgentError("attempt_outcome_unresolved")
            return self.attempt
        self.reservations += 1
        self.attempt = {"id": "fake-attempt", "status": "reserved", "request": copy.deepcopy(request), "request_hash": digest(request)}
        return self.attempt

    def observe_model_wire(self, attempt_id, receipt):
        self.wire_receipts.append(copy.deepcopy(receipt))

    def mark_sent(self, attempt_id):
        self.attempt["status"] = "sent"
        return 60

    def observe_model_http_receipt(self, attempt_id, response, observation):
        self.http_receipts.append((copy.deepcopy(response), copy.deepcopy(observation)))

    def settle(self, attempt_id, response, usage):
        self.attempt.update(status="settled", response=copy.deepcopy(response), usage=copy.deepcopy(usage))
        return None

    def unknown(self, attempt_id, code):
        self.attempt["status"] = "unknown"
        self.unknowns.append(code)

    def not_sent(self, attempt_id, code, *, before_wire):
        self.not_sent_attempts.append(code)


def test_full_bound_review_compilation_preserves_local_schema():
    from citequest.agents_v3.structured_output import strict_output_schema
    from citequest.research_v3.qa_v5_contracts import bound_combined_review_model
    from citequest.research_v3.review_grid import ReviewGrid

    candidate = {"claims": [{"id": f"C{index + 1}", "edges": [{"quote_id": f"Q{(index * 5 + offset) % 20 + 1}"}
        for offset in range(5)]} for index in range(8)]}
    scope = SimpleNamespace(compact_candidate=lambda value, **kwargs: value, binding={"binding_sha256": "a" * 64})
    model = bound_combined_review_model(ReviewGrid(scope, candidate), "b" * 64, "c" * 64)
    full = strict_output_schema(model.model_json_schema())
    original = copy.deepcopy(full)
    compiled, local_only, _annotations = compile_schema(full)
    assert full == original
    assert compiled["properties"]["candidate_sha256"]["enum"] == ["b" * 64]
    assert set(compiled["properties"]["claims"]["properties"]) == {f"C{index}" for index in range(1, 9)}
    assert {item["keyword"] for item in local_only} >= {"minLength", "maxLength", "maxItems"}
    assert '"$ref"' not in json.dumps(compiled) and '"$defs"' not in json.dumps(compiled)
    with pytest.raises(ValidationError):
        model.model_validate({"candidate_sha256": "b" * 64}, strict=True)


@pytest.mark.parametrize("unsupported", [{"type": "null"}, {"type": "string", "format": "date-time"}, {"type": "object", "oneOf": []}])
def test_unsupported_schema_never_silently_compiles(unsupported):
    with pytest.raises(AgentError):
        compile_schema({"type": "object", "properties": {"value": unsupported}, "required": ["value"], "additionalProperties": False})


def test_locked_sdk_beta_wire_raw_ledger_and_settled_zero_network_replay():
    async def run():
        execution = FakeExecution()
        payload = canonical_review()
        observed = []

        def handler(request):
            observed.append(request)
            assert str(request.url) == FIXTURE_ENDPOINT
            assert json.loads(request.content) == compile_review_request(payload, fixture=True).wire
            return httpx2.Response(200, json=raw_chat())

        provider = ResponsesProvider(execution, fixture=True, transport=httpx2.MockTransport(handler))
        try:
            first = await provider.complete(payload, "command:review")
            replay = await provider.complete(payload, "command:review")
        finally:
            await provider.close()
        assert first == replay
        assert first["output"][0]["call_id"] == "actual_call_1"
        assert len(observed) == execution.reservations == len(execution.wire_receipts) == 1
        assert execution.attempt["request"] == payload
        recorded = copy.deepcopy(execution.attempt["response"])
        gate = recorded.pop(ENVELOPE_GATE_KEY)
        assert gate == {"protocol": "chat-strict1", "accepted": True, "error_code": None}
        assert recorded == raw_chat() and "status" not in execution.attempt["response"]
        assert execution.attempt["usage"] == {"input": 10, "cached": 2, "output": 4}
        assert execution.wire_receipts[0]["actual_wire_sha256"] == hashlib.sha256(observed[0].content).hexdigest()
        assert execution.wire_receipts[0]["actual_wire_bytes"] == len(observed[0].content)
        assert not execution.unknowns

    asyncio.run(run())


@pytest.mark.parametrize("usage", [None, ["explicit-malformed-usage"], {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
    "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 9}])
def test_missing_or_inconsistent_usage_stops_and_preserves_unknown(usage):
    async def run():
        execution = FakeExecution()
        response = raw_chat()
        response["usage"] = usage
        observed = []
        def handler(request):
            observed.append(request)
            return httpx2.Response(200, json=response)
        provider = ResponsesProvider(execution, fixture=True, transport=httpx2.MockTransport(handler))
        try:
            with pytest.raises(Exception):
                await provider.complete(canonical_review(), "command:review")
            assert execution.attempt["status"] == "unknown" and execution.unknowns
            assert len(execution.http_receipts) == len(observed) == 1
        finally:
            await provider.close()
    asyncio.run(run())


def test_known_usage_with_bad_arguments_is_not_repaired_or_reclassified_unknown():
    async def run():
        execution = FakeExecution()
        response = raw_chat('{"verdict":"passed"}x')
        provider = ResponsesProvider(execution, fixture=True,
            transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=response)))
        try:
            value = await provider.complete(canonical_review(), "command:review")
        finally:
            await provider.close()
        arguments = value["output"][0]["arguments"]
        assert arguments == '{"verdict":"passed"}x'
        with pytest.raises(AgentError):
            strict_object(arguments)
        assert execution.attempt["status"] == "settled" and not execution.unknowns
    asyncio.run(run())


def test_abnormal_finish_or_extra_function_is_rejected_after_known_usage():
    response = raw_chat()
    response["choices"][0]["finish_reason"] = "length"
    assert usage_counts_chat(response) == {"input": 10, "cached": 2, "output": 4}
    with pytest.raises(AgentError):
        normalize_chat_response(response, require_gate=False)
    response = raw_chat()
    response["choices"][0]["message"]["tool_calls"].append(copy.deepcopy(response["choices"][0]["message"]["tool_calls"][0]))
    with pytest.raises(AgentError):
        normalize_chat_response(response, require_gate=False)


def test_private_envelope_gate_survives_sdk_and_negative_settled_replay():
    async def scenario(response, http_status, expected_error):
        execution = FakeExecution()
        observed = []
        def handler(request):
            observed.append(request)
            return httpx2.Response(http_status, json=response)
        provider = ResponsesProvider(execution, fixture=True, transport=httpx2.MockTransport(handler))
        try:
            with pytest.raises(Exception):
                await provider.complete(canonical_review(), "command:review")
            stored = execution.attempt["response"]
            assert stored[ENVELOPE_GATE_KEY] == {"protocol": "chat-strict1", "accepted": False, "error_code": expected_error}
            assert execution.attempt["status"] == "settled" and not execution.unknowns
            assert stored == execution.http_receipts[0][0]
            with pytest.raises(AgentError) as replay:
                await provider.complete(canonical_review(), "command:review")
            assert replay.value.code == expected_error
            assert len(observed) == execution.reservations == len(execution.http_receipts) == 1
            return stored
        finally:
            await provider.close()

    async def run():
        thinking = raw_chat()
        thinking["choices"][0]["message"]["reasoning_content"] = "Explicit synthetic private reasoning fixture."
        stored = await scenario(thinking, 200, "chat_strict_response_message_invalid")
        assert "reasoning_content" not in stored["choices"][0]["message"]
        forged = raw_chat()
        forged[ENVELOPE_GATE_KEY] = {"protocol": "chat-strict1", "accepted": True, "error_code": None}
        await scenario(forged, 200, "chat_strict_reserved_receive_gate")
        await scenario(raw_chat(), 500, "provider_http_error_with_known_usage")
        with pytest.raises(AgentError) as missing:
            normalize_chat_response(raw_chat())
        assert missing.value.code == "chat_strict_recorded_receive_gate_missing"
        invalid = raw_chat()
        invalid[ENVELOPE_GATE_KEY] = {"protocol": "chat-strict1", "accepted": 1, "error_code": None}
        with pytest.raises(AgentError) as malformed:
            normalize_chat_response(invalid)
        assert malformed.value.code == "chat_strict_recorded_receive_gate_invalid"

    asyncio.run(run())


def test_original_responses_path_unchanged_without_new_protocol():
    async def run():
        execution = FakeExecution(strict=False)
        observed = []
        def handler(request):
            observed.append(request)
            assert str(request.url) == "https://s08-provider.invalid/responses"
            return httpx2.Response(200, json={"id": "resp_fixture", "object": "response", "created_at": 1794000000,
                "status": "completed", "model": "deepseek-flash", "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14, "input_tokens_details": {"cached_tokens": 2}}})
        provider = ResponsesProvider(execution, fixture=True, transport=httpx2.MockTransport(handler))
        try:
            result = await provider.complete(canonical_review(), "command:review")
        finally:
            await provider.close()
        assert result["status"] == "completed" and len(observed) == 1
        assert not execution.wire_receipts and not execution.unknowns
        assert execution.attempt["usage"] == {"input": 10, "cached": 2, "output": 4}
    asyncio.run(run())
