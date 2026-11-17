"""Three model-contract controls with explicit SDK transport/ledger doubles."""
import asyncio
import copy
import json
from datetime import UTC, datetime

import httpx2
import pytest

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.model_contracts import model_contract, validated_model_alias
from citequest.agents_v3.provider import POLICY_MARKER, ResponsesProvider, validate_payload
from citequest.agents_v3.strict_chat import compile_review_request, is_strict_review

PRO = "deepseek-v4-pro"


def profile():
    return {**model_contract(PRO), "profile_id": "qa-v7", "qa_execution_protocol": "grounded3",
        "review_transport_protocol": "function1", "review_api_protocol": "chat-strict2",
        "per_call_output": 4096, "request_bytes": 131072}


def payload(*, review=False):
    schema = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["ok"]}},
        "required": ["verdict"], "additionalProperties": False}
    value = {"model": PRO, "reasoning": {"effort": "none"}, "truncation": "disabled",
        "instructions": POLICY_MARKER + " Explicit model-binding fixture.", "max_output_tokens": 4096,
        "input": [{"role": "user", "content": '{"context_manifest":{}}'}]}
    if review:
        value.update(tools=[{"type": "function", "name": "submit_review", "parameters": schema}],
            tool_choice={"type": "function", "name": "submit_review"})
    else:
        value["text"] = {"format": {"type": "json_schema", "name": "citequest_qa5_plan", "schema": schema, "strict": True}}
    return value


def chat(*, model=PRO):
    return {"id": "explicit-chat", "object": "chat.completion", "created": 1794000000, "model": model,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "explicit-call", "type": "function", "function": {
                "name": "submit_review", "arguments": '{"verdict":"ok"}'}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
            "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 8}}


class Ledger:
    def __init__(self):
        self.profile = profile()
        self.attempts, self.http_receipts, self.unknowns = {}, [], []
        self.settlements = 0

    def snapshot(self):
        return {"profile": self.profile}

    def reserve(self, key, request):
        if key in self.attempts:
            row = self.attempts[key]
            assert row["request_hash"] == digest(request)
            if row["status"] == "unknown":
                raise AgentError("attempt_outcome_unresolved")
            return row
        row = {"id": key, "request": copy.deepcopy(request), "request_hash": digest(request), "status": "reserved"}
        self.attempts[key] = row
        return row

    def observe_model_wire(self, attempt_id, metadata):
        assert metadata["canonical_request_sha256"] == self.attempts[attempt_id]["request_hash"]

    def mark_sent(self, attempt_id):
        self.attempts[attempt_id]["status"] = "sent"
        return 60

    def observe_model_http_receipt(self, attempt_id, response, metadata):
        self.http_receipts.append((attempt_id, copy.deepcopy(response), metadata))

    def settle(self, attempt_id, response, usage):
        self.settlements += 1
        self.attempts[attempt_id].update(status="settled", response=copy.deepcopy(response), usage=usage)
        return None

    def unknown(self, attempt_id, code):
        self.attempts[attempt_id].update(status="unknown", error_code=code)
        self.unknowns.append(code)

    def not_sent(self, attempt_id, code, *, before_wire):
        self.attempts[attempt_id].update(status="not_sent", error_code=code)

    def reserve_protocol_recovery(self, *_args):
        raise AssertionError("These valid or identity-rejected replies must not reserve protocol recovery.")


def test_only_registered_frozen_model_contract_can_admit_pro_and_legacy_defaults_remain_flash():
    original = payload(review=True)
    with pytest.raises(AgentError, match="provider_model_policy_changed"):
        validate_payload(original)
    validate_payload(original, expected_model=validated_model_alias(profile()))
    compiled = compile_review_request(original, protocol="chat-strict2")
    assert compiled.wire["model"] == PRO and compiled.wire["tools"][0]["function"]["strict"] is True
    assert original == payload(review=True) and is_strict_review(profile(), original)
    with pytest.raises(AgentError, match="chat_strict_model_protocol_not_supported"):
        compile_review_request(original)
    with pytest.raises(AgentError, match="provider_model_not_registered"):
        validate_payload(original, expected_model="unregistered-model")
    wrong_rates = {**profile(), "output_ncny": 8000}
    with pytest.raises(AgentError, match="provider_frozen_model_contract_mismatch"):
        validated_model_alias(wrong_rates)
    with pytest.raises(AgentError, match="chat_strict_Task_profile_not_supported"):
        is_strict_review({**profile(), "review_api_protocol": "chat-strict1"}, original)
    flash = copy.deepcopy(original)
    flash["model"] = "deepseek-flash"
    validate_payload(flash)
    assert compile_review_request(flash).wire["model"] == "deepseek-flash"


def test_pro_uses_same_frozen_model_on_responses_and_beta_and_replays_without_physical_calls():
    async def run():
        ledger, observed = Ledger(), []

        def handler(request):
            wire = json.loads(request.content)
            observed.append((str(request.url), wire))
            assert wire["model"] == PRO
            if str(request.url).endswith("/beta/chat/completions"):
                return httpx2.Response(200, json=chat())
            return httpx2.Response(200, json={"id": "explicit-response", "object": "response", "created_at": 1794000000,
                "model": PRO, "status": "completed", "output": [{"type": "message", "id": "explicit-message",
                    "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": '{"verdict":"ok"}', "annotations": []}]}],
                "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
                    "input_tokens_details": {"cached_tokens": 2}}})

        provider = ResponsesProvider(ledger, fixture=True, transport=httpx2.MockTransport(handler))
        try:
            first = await provider.complete(payload(), "command:qa5:plan:0")
            second = await provider.complete(payload(review=True), "command:qa5:review:0")
            assert first == await provider.complete(payload(), "command:qa5:plan:0")
            assert second == await provider.complete(payload(review=True), "command:qa5:review:0")
        finally:
            await provider.close()
        assert len(observed) == ledger.settlements == 2 and not ledger.unknowns
        assert first["model"] == PRO and second["_cq_review_api_protocol"] == "chat-strict2"
        assert second["output"][0]["arguments"] == '{"verdict":"ok"}'
        assert all(row["request"]["model"] == PRO and row["request_hash"] == digest(row["request"])
            for row in ledger.attempts.values())

    asyncio.run(run())


def test_new_pro_wrong_reply_or_metadata_model_is_rejected_without_settlement_or_recovery(tmp_path):
    async def run():
        ledger, observed = Ledger(), []

        def handler(request):
            observed.append(request)
            return httpx2.Response(200, json=chat(model="deepseek-flash"))

        provider = ResponsesProvider(ledger, fixture=True, transport=httpx2.MockTransport(handler))
        try:
            for _ in range(2):
                with pytest.raises(Exception):
                    await provider.complete(payload(review=True), "command:qa5:review:0")
        finally:
            await provider.close()
        assert len(observed) == len(ledger.http_receipts) == 1
        assert ledger.settlements == 0 and ledger.unknowns == ["provider_response_model_mismatch"]
        assert ledger.attempts["command:qa5:review:0"]["status"] == "unknown"
        assert ledger.http_receipts[0][1]["usage"]["completion_tokens"] == 4

    asyncio.run(run())
    metadata = {"status": "PASS", "at": datetime.now(UTC).isoformat(),
        "model": {"id": "deepseek-flash", "context_window": 1048576, "max_output_tokens": 393216},
        "model_version_from_pricing": "DeepSeek-V4-Pro-0813",
        "rates_ncny_per_token": {"input_miss": 9000, "input_cached": 300, "output": 27000}}
    path = tmp_path / "explicit-mismatched-metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(AgentError, match="live_provider_metadata_expired_or_changed"):
        ResponsesProvider(Ledger(), api_key_file=tmp_path / "must-not-be-read", metadata_receipt_file=path)
