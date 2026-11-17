"""Four focused controls with explicit fake execution and zero-network HTTP."""
from __future__ import annotations

import asyncio
import copy
import json

import httpx2
import pytest

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.provider import POLICY_MARKER, ResponsesProvider
from citequest.agents_v3.model_contracts import model_contract
from citequest.agents_v3.provider_contract import wire_contract_violation
from citequest.agents_v3.strict_chat import recorded_chat_response


def request():
    return {
        "model": "deepseek-flash", "reasoning": {"effort": "none"}, "truncation": "disabled",
        "instructions": POLICY_MARKER + " Explicit protocol recovery fixture.", "max_output_tokens": 4096,
        "input": [{"role": "user", "content": '{"context_manifest":{}}'}],
        "tools": [{"type": "function", "name": "submit_review", "parameters": {
            "type": "object", "properties": {"verdict": {"type": "string", "enum": ["passed", "rejected"]}},
            "required": ["verdict"], "additionalProperties": False,
        }}],
        "tool_choice": {"type": "function", "name": "submit_review"},
    }


def chat(arguments='{"verdict":"passed"}'):
    return {
        "id": "explicit-chat-fixture", "object": "chat.completion", "created": 1794000000,
        "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "explicit-call", "type": "function", "function": {
                "name": "submit_review", "arguments": arguments,
            }}],
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
            "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 8},
    }


class FakeExecution:
    """The real DB capability has separate atomic controls in its own tests."""

    def __init__(self, *, protocol="grounded3"):
        self.profile = {**model_contract("deepseek-flash"), "profile_id": "qa-v6", "qa_execution_protocol": protocol,
            "review_api_protocol": "chat-strict1", "review_transport_protocol": "function1", "request_bytes": 131072}
        self.attempts = {}
        self.recoveries = {}
        self.http_receipts = []
        self.wire_receipts = []
        self.unknowns = []

    def snapshot(self):
        return {"profile": self.profile}

    def reserve(self, key, payload):
        if key in self.attempts:
            row = self.attempts[key]
            assert row["request_hash"] == digest(payload)
            if row["status"] == "unknown":
                raise AgentError("attempt_outcome_unresolved")
            return row
        row = {"id": key, "status": "reserved", "request": copy.deepcopy(payload), "request_hash": digest(payload)}
        self.attempts[key] = row
        return row

    def observe_model_wire(self, attempt_id, receipt):
        self.wire_receipts.append((attempt_id, copy.deepcopy(receipt)))

    def mark_sent(self, attempt_id):
        self.attempts[attempt_id]["status"] = "sent"
        return 60

    def observe_model_http_receipt(self, attempt_id, response, metadata):
        self.http_receipts.append((attempt_id, copy.deepcopy(response), copy.deepcopy(metadata)))

    def settle(self, attempt_id, response, usage):
        self.attempts[attempt_id].update(status="settled", response=copy.deepcopy(response), usage=copy.deepcopy(usage))
        return None

    def unknown(self, attempt_id, code):
        self.attempts[attempt_id]["status"] = "unknown"
        self.unknowns.append((attempt_id, code))

    def not_sent(self, attempt_id, code, *, before_wire):
        self.attempts[attempt_id].update(status="not_sent", error_code=code)

    def reserve_protocol_recovery(self, parent_key, payload, violation):
        parent = self.attempts[parent_key]
        assert parent["status"] == "settled" and parent.get("usage") is not None
        assert parent["request_hash"] == digest(payload)
        assert wire_contract_violation(payload, parent["response"]) == violation
        if parent_key not in self.recoveries:
            assert len(self.recoveries) < 2
            self.recoveries[parent_key] = parent_key + ":wire-recovery:1"
        return self.recoveries[parent_key]


def test_classifier_covers_raw_json_and_actual_wire_schema_without_local_only_bounds():
    payload = request()
    examples = [('{"verdict":"passed"}}', "invalid_json"),
        ('{"verdict":"passed","verdict":"rejected"}', "duplicate_property"),
        ('{"verdict":NaN}', "nonfinite_json"),
        ('{"verdict":"passed","unexpected":null}', "wire_schema")]
    for arguments, category in examples:
        raw = chat(arguments)
        received = recorded_chat_response(raw, copy.deepcopy(raw), 200)
        before = copy.deepcopy(received)
        result = wire_contract_violation(payload, received)
        assert result["category"] == category and result["arguments_sha256"] == digest(arguments.encode())
        assert result["arguments_utf8_bytes"] == len(arguments.encode())
        assert set(result) == {"code", "category", "protocol", "compiler", "wire_schema_sha256",
            "arguments_sha256", "arguments_utf8_bytes"}
        assert received == before
    bounded = copy.deepcopy(payload)
    schema = bounded["tools"][0]["parameters"]
    schema["properties"]["notes"] = {"type": "array", "items": {"type": "string"}, "maxItems": 1}
    schema["required"].append("notes")
    raw = chat('{"verdict":"passed","notes":["one","two"]}')
    assert wire_contract_violation(bounded, recorded_chat_response(raw, copy.deepcopy(raw), 200)) is None


def test_known_bad_parent_and_valid_child_replay_without_resend_or_argument_repair():
    async def run():
        execution = FakeExecution()
        observed = []
        bad = '{"verdict":"passed"}}'

        def handler(value):
            observed.append(value)
            return httpx2.Response(200, json=chat(bad if len(observed) == 1 else '{"verdict":"passed"}'))

        provider = ResponsesProvider(execution, fixture=True, transport=httpx2.MockTransport(handler))
        try:
            first = await provider.complete(request(), "command:qa5:review:0")
            replay = await provider.complete(request(), "command:qa5:review:0")
        finally:
            await provider.close()
        assert first == replay and first["output"][0]["arguments"] == '{"verdict":"passed"}'
        assert len(observed) == len(execution.attempts) == len(execution.http_receipts) == 2
        assert len(execution.recoveries) == 1 and not execution.unknowns
        assert all(row["status"] == "settled" and row["usage"] == {"input": 10, "cached": 2, "output": 4}
            for row in execution.attempts.values())
        assert {row["request_hash"] for row in execution.attempts.values()} == {digest(request())}
        parent = execution.attempts["command:qa5:review:0"]
        assert parent["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == bad
        assert all(value.headers.get("x-stainless-retry-count") in (None, "0") for value in observed)
        assert json.loads(observed[0].content) == json.loads(observed[1].content)

    asyncio.run(run())


def test_second_violation_and_unknown_child_stop_with_no_third_physical_request():
    async def run(*, unknown_child):
        execution = FakeExecution()
        observed = []

        def handler(value):
            observed.append(value)
            raw = chat('{"verdict":"passed"}}')
            if unknown_child and len(observed) == 2:
                raw.pop("usage")
            return httpx2.Response(200, json=raw)

        provider = ResponsesProvider(execution, fixture=True, transport=httpx2.MockTransport(handler))
        try:
            if unknown_child:
                with pytest.raises(Exception):
                    await provider.complete(request(), "command:qa5:review:0")
                with pytest.raises(Exception):
                    await provider.complete(request(), "command:qa5:review:0")
                assert execution.attempts["command:qa5:review:0:wire-recovery:1"]["status"] == "unknown"
                assert len(execution.unknowns) == 1
            else:
                for _ in range(2):
                    with pytest.raises(AgentError, match="provider_contract_violation_after_recovery"):
                        await provider.complete(request(), "command:qa5:review:0")
                assert all(row["status"] == "settled" for row in execution.attempts.values())
                assert not execution.unknowns
        finally:
            await provider.close()
        assert len(observed) == len(execution.attempts) == 2 and len(execution.recoveries) == 1

    asyncio.run(run(unknown_child=False))
    asyncio.run(run(unknown_child=True))


def test_http_refusal_and_unknown_usage_do_not_recover_and_grounded2_keeps_single_attempt():
    async def run():
        missing = chat('{"verdict":"passed"}}')
        missing.pop("usage")
        refusal = chat('{"verdict":"passed"}}')
        refusal["choices"][0]["message"]["refusal"] = "Explicit refusal fixture."
        for status, raw in ((503, chat()), (200, refusal), (200, missing)):
            execution = FakeExecution()
            observed = []

            def handler(value):
                observed.append(value)
                return httpx2.Response(status, json=raw)

            provider = ResponsesProvider(execution, fixture=True, transport=httpx2.MockTransport(handler))
            try:
                with pytest.raises(Exception):
                    await provider.complete(request(), "command:qa5:review:0")
            finally:
                await provider.close()
            assert len(observed) == len(execution.attempts) == 1 and not execution.recoveries
        execution = FakeExecution(protocol="grounded2")
        observed = []

        def legacy(value):
            observed.append(value)
            return httpx2.Response(200, json=chat('{"verdict":"passed"}}'))

        provider = ResponsesProvider(execution, fixture=True, transport=httpx2.MockTransport(legacy))
        try:
            result = await provider.complete(request(), "command:qa5:review:0")
        finally:
            await provider.close()
        assert result["output"][0]["arguments"] == '{"verdict":"passed"}}'
        assert len(observed) == 1 and not execution.recoveries

    asyncio.run(run())
