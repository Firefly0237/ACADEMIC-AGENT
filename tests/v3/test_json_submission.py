"""Focused JSON2 transport controls. All HTTP/Execution are explicit fakes."""
import asyncio
import copy
import json

import httpx2
import openai
import pytest

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.json_review import (
    JSON2_COMPILER, JSON2_VERSION, REASONING_KEY, compile_json_review,
    normalize_json_review, normalize_json_submission, recorded_json_review,
)
from citequest.agents_v3.model_contracts import model_contract
from citequest.agents_v3.provider import POLICY_MARKER, ResponsesProvider
from citequest.agents_v3.provider_contract import wire_contract_violation
from citequest.agents_v3.strict_chat import ENVELOPE_GATE_KEY, usage_counts_chat
from citequest.agents_v3.typed_submission import compile_submission, normalize_submission, recorded_submission


def profile(profile_id="qa-v8"):
    return {**model_contract("deepseek-v4-pro"), "profile_id": profile_id, "qa_execution_protocol": "grounded3",
        "review_transport_protocol": "json1", "review_api_protocol": JSON2_VERSION,
        "structured_output_protocol": JSON2_VERSION, "answer_revision_protocol": "jsonpatch1",
        "qa_context_protocol": "references1", "request_bytes": 262144,
        "review_read_seconds": 300 if profile_id == "qa-v9" else 180}


def request(role):
    return {"model": "deepseek-v4-pro", "reasoning": {"effort": "low" if role == "review" else "none"},
        "truncation": "disabled", "instructions": POLICY_MARKER + " Explicit QA8 transport fixture.",
        "max_output_tokens": 8192, "input": [{"role": "user", "content": '{"context_manifest":{}}'}],
        "text": {"format": {"type": "json_schema", "name": "citequest_qa5_" + role, "strict": True,
            "schema": {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["accepted", "rejected"]}},
                "required": ["verdict"], "additionalProperties": False}}}}


def chat(*, reasoning=None, reasoning_tokens=0):
    message = {"role": "assistant", "content": '{"verdict":"rejected"}'}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"id": "qa8-json-fixture", "object": "chat.completion", "created": 1794000000,
        "model": "deepseek-v4-pro", "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20,
            "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}}


def test_qa8_all_roles_compile_exact_modes_and_old_json1_remains_non_thinking():
    for role in ("plan", "author", "review"):
        payload = request(role)
        original = copy.deepcopy(payload)
        compiled = compile_submission(profile(), payload, f"explicit:qa5:{role}:0")
        assert payload == original and compiled.receipt["protocol"] == JSON2_VERSION
        assert compiled.receipt["compiler"] == JSON2_COMPILER
        assert compiled.receipt["role"] == role and compiled.receipt["schema_name"] == "citequest_qa5_" + role
        assert compiled.receipt["structured_output_protocol"] == JSON2_VERSION
        assert compiled.wire["thinking"] == {"type": "enabled" if role == "review" else "disabled"}
        assert compiled.wire["reasoning_effort"] == ("low" if role == "review" else "none")
        assert compiled.wire["response_format"] == {"type": "json_object"} and compiled.wire["max_tokens"] == 8192
        assert "tools" not in compiled.wire and "function_name" not in compiled.receipt
        bad = {**payload, "reasoning": {"effort": "none" if role == "review" else "low"}}
        with pytest.raises(AgentError):
            compile_submission(profile(), bad, f"explicit:qa5:{role}:0")
    with pytest.raises(AgentError):
        compile_submission({**profile(), "profile_id": "qa-v7"}, request("review"), "explicit:qa5:review:0")
    legacy = request("review")
    legacy.update(reasoning={"effort": "none"}, max_output_tokens=4096)
    compiled = compile_json_review(legacy)
    assert compiled.receipt["protocol"] == "chat-json1" and compiled.receipt["structured_output_protocol"] == "chat-functions1"
    assert compiled.wire["thinking"] == {"type": "disabled"} and compiled.wire["reasoning_effort"] == "none"
    raw = chat(reasoning="Synthetic private review reasoning.", reasoning_tokens=7)
    recorded = recorded_json_review(raw, copy.deepcopy(raw), 200)
    with pytest.raises(AgentError, match="chat_json_thinking_output_forbidden"):
        normalize_json_review(recorded)


def test_private_reasoning_hash_gate_preserves_usage_and_rejects_forgery_or_wrong_role():
    key = "explicit:qa5:review:0"
    private_reasoning = "Synthetic private review reasoning."
    raw = chat(reasoning=private_reasoning, reasoning_tokens=7)
    stored = recorded_submission(profile(), request("review"), key, raw, copy.deepcopy(raw), 200)
    assert stored[REASONING_KEY] == {"protocol": JSON2_VERSION, "role": "review", "present": True,
        "characters": len(private_reasoning), "utf8_bytes": len(private_reasoning.encode()),
        "sha256": digest(private_reasoning.encode())}
    assert "reasoning_content" not in stored["choices"][0]["message"]
    assert usage_counts_chat(stored) == {"input": 10, "cached": 2, "output": 10}
    replay = openai.types.chat.ChatCompletion.model_validate(stored).model_dump(mode="json", exclude_none=True)
    normalized = normalize_submission(profile(), request("review"), key, replay)
    assert normalized["output"][0]["content"][0]["text"] == raw["choices"][0]["message"]["content"]
    assert private_reasoning not in json.dumps(normalized) and REASONING_KEY not in normalized
    assert wire_contract_violation(request("review"), replay, profile=profile(), logical_key=key) is None
    for role in ("plan", "author"):
        forbidden = recorded_submission(profile(), request(role), f"explicit:qa5:{role}:0", raw, raw, 200)
        with pytest.raises(AgentError):
            normalize_json_submission(forbidden, role)
    for bad in (True, -1, 11):
        malformed = chat(reasoning=private_reasoning, reasoning_tokens=bad)
        received = recorded_submission(profile(), request("review"), key, malformed, malformed, 200)
        with pytest.raises(AgentError, match="chat_json_reasoning_metadata_invalid"):
            normalize_json_submission(received, "review")
    for reserved in (REASONING_KEY, ENVELOPE_GATE_KEY):
        forged = {**raw, reserved: copy.deepcopy(stored[reserved])}
        received = recorded_submission(profile(), request("review"), key, forged, forged, 200)
        with pytest.raises(AgentError, match="chat_json_reserved_receive_gate"):
            normalize_json_submission(received, "review")
    missing = copy.deepcopy(stored)
    missing.pop(REASONING_KEY)
    with pytest.raises(AgentError):
        normalize_json_submission(missing, "review")


@pytest.mark.parametrize("profile_id,maximum,read_seconds", [
    ("qa-v8", 8192, 180), ("qa-v9", 65536, 300),
], ids=["qa8", "qa9"])
def test_locked_sdk_accounts_full_completion_and_replays_hash_only_private_review(profile_id, maximum, read_seconds):
    class Execution:
        def __init__(self):
            self.row = None
            self.wire = []
        def snapshot(self):
            return {"profile": profile(profile_id)}
        def reserve(self, key, payload):
            if self.row is None:
                self.row = {"id": key, "status": "reserved", "request_hash": digest(payload)}
            assert self.row["request_hash"] == digest(payload)
            return self.row
        def observe_model_wire(self, _attempt, receipt):
            self.wire.append(receipt)
        def mark_sent(self, _attempt):
            return 900
        def observe_model_http_receipt(self, *_args):
            pass
        def settle(self, _attempt, response, usage):
            self.row.update(status="settled", response=copy.deepcopy(response), usage=usage)
        def unknown(self, *_args):
            raise AssertionError("Unexpected unknown usage in this explicit fake.")
        def not_sent(self, *_args, **_kwargs):
            raise AssertionError("Mocked wire must be reached.")

    async def run():
        execution = Execution()
        physical = []
        payload = {**request("review"), "max_output_tokens": maximum}
        # The new budget profile must leave the already admitted QA8 wire unchanged.
        assert compile_submission(profile(), request("review"), "explicit:qa5:review:0").wire["max_tokens"] == 8192
        def handler(wire):
            physical.append(wire)
            assert str(wire.url) == "https://s08-provider.invalid/chat/completions"
            body = json.loads(wire.content)
            assert body["thinking"] == {"type": "enabled"} and body["reasoning_effort"] == "low"
            assert body["max_tokens"] == maximum and "tools" not in body
            assert wire.extensions["timeout"] == {"connect": 5, "pool": 5, "read": read_seconds, "write": 30}
            return httpx2.Response(200, json=chat(reasoning="Synthetic private reasoning.", reasoning_tokens=7))
        provider = ResponsesProvider(execution, fixture=True, transport=httpx2.MockTransport(handler))
        try:
            first = await provider.complete(payload, "explicit:qa5:review:0")
            assert await provider.complete(payload, "explicit:qa5:review:0") == first
            assert len(physical) == len(execution.wire) == 1
            assert execution.row["usage"]["output"] == 10
            assert execution.row["response"][REASONING_KEY]["present"] is True
            assert "reasoning_content" not in execution.row["response"]["choices"][0]["message"]
            assert REASONING_KEY not in first
        finally:
            await provider.close()
    asyncio.run(run())
