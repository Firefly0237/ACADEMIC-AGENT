"""Three pure controls for explicit all-role typed submission. No provider I/O."""
import copy
import json

import pytest
from jsonschema import Draft202012Validator

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.model_contracts import model_contract
from citequest.agents_v3.provider_contract import wire_contract_violation
from citequest.agents_v3.strict_chat import compile_review_request, compile_schema_v2
from citequest.agents_v3.typed_submission import (
    COMPILER, NULLABLE_CONTRACT, ROLE_FUNCTIONS, VERSION,
    compile_submission, normalize_submission, recorded_submission, submission_role,
)


def profile():
    return {**model_contract("deepseek-v4-pro"), "profile_id": "qa-v7", "qa_execution_protocol": "grounded3",
        "review_transport_protocol": "function1", "review_api_protocol": "chat-strict2",
        "structured_output_protocol": VERSION}


def request(role):
    schema = ({"type": "object", "properties": {
        "clarification": {"anyOf": [{"type": "string", "maxLength": 500}, {"type": "null"}]},
        "category": {"anyOf": [{"type": "string", "enum": ["question", "focus"]}, {"type": "null"}]},
    }, "required": ["clarification", "category"], "additionalProperties": False} if role == "plan" else
        {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False})
    value = {"model": "deepseek-v4-pro", "reasoning": {"effort": "none"}, "truncation": "disabled",
        "instructions": "Explicit all-role submission fixture.", "max_output_tokens": 4096,
        "input": [{"role": "user", "content": "Explicit all-role submission fixture."}]}
    if role == "review":
        value.update(tools=[{"type": "function", "name": "submit_review", "parameters": schema}],
            tool_choice={"type": "function", "name": "submit_review"})
    else:
        value["text"] = {"format": {"type": "json_schema", "name": "citequest_qa5_" + role, "schema": schema, "strict": True}}
    return value


def chat(function, arguments):
    return {"object": "chat.completion", "model": "deepseek-v4-pro", "choices": [{"index": 0,
        "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "explicit-submission", "type": "function", "function": {
                "name": function, "arguments": arguments}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
            "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 8}}


def test_all_role_compilation_preserves_canonical_schema_and_nullable_is_only_explicit():
    for role, function in ROLE_FUNCTIONS.items():
        payload = request(role)
        before = copy.deepcopy(payload)
        key = "explicit:qa5:" + role + ":0"
        compiled = compile_submission(profile(), payload, key)
        assert payload == before and compiled.receipt["canonical_request_sha256"] == digest(payload)
        assert compiled.receipt["protocol"] == VERSION and compiled.receipt["compiler"] == COMPILER
        assert compiled.receipt["role"] == role and compiled.receipt["function_name"] == function
        assert compiled.receipt["nullable_type_contract"] == NULLABLE_CONTRACT
        assert compiled.wire["tool_choice"] == {"type": "function", "function": {"name": function}}
        assert compiled.wire["tools"][0]["function"]["strict"] is True
        if role == "plan":
            original = payload["text"]["format"]["schema"]
            with pytest.raises(AgentError, match="chat_strict_schema_keyword_or_type_not_supported"):
                compile_schema_v2(original)
            wire = compiled.wire["tools"][0]["function"]["parameters"]
            assert wire["properties"]["clarification"]["anyOf"][1] == {"type": "null"}
            assert Draft202012Validator(original).is_valid({"clarification": None, "category": None})
            assert Draft202012Validator(wire).is_valid({"clarification": None, "category": None})


def test_normalization_preserves_arguments_and_bad_json_is_classified_without_stripping():
    for role, function in ROLE_FUNCTIONS.items():
        payload = request(role)
        key = "explicit:qa5:" + role + ":0"
        arguments = '{"clarification":null,"category":null}' if role == "plan" else '{"answer":"formula $x^2$\\nunchanged"}'
        raw = chat(function, arguments)
        received = recorded_submission(profile(), payload, key, raw, copy.deepcopy(raw), 200)
        result = normalize_submission(profile(), payload, key, received)
        actual = (result["output"][0]["arguments"] if role == "review"
            else result["output"][0]["content"][0]["text"])
        assert actual == arguments
        assert wire_contract_violation(payload, received, profile=profile(), logical_key=key) is None
        fenced = "```json\n" + arguments + "\n```"
        raw = chat(function, fenced)
        received = recorded_submission(profile(), payload, key, raw, copy.deepcopy(raw), 200)
        violation = wire_contract_violation(payload, received, profile=profile(), logical_key=key)
        assert violation["category"] == "invalid_json" and violation["protocol"] == VERSION
        assert violation["compiler"] == COMPILER and violation["arguments_sha256"] == digest(fenced.encode())
        assert received["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == fenced


def test_role_identity_gate_and_legacy_review_compile_remain_explicit():
    payload = request("plan")
    with pytest.raises(AgentError, match="typed_submission_role_schema_required"):
        compile_submission(profile(), payload, "explicit:qa5:author:0")
    with pytest.raises(AgentError, match="typed_submission_role_key_invalid"):
        compile_submission(profile(), payload, "explicit:qa5:plan:0:wire-recovery:1:wire-recovery:1")
    key = "explicit:qa5:plan:0:wire-recovery:1"
    raw = chat("submit_answer", '{"answer":"wrong-role"}')
    received = recorded_submission(profile(), payload, key, raw, copy.deepcopy(raw), 200)
    with pytest.raises(AgentError, match="chat_strict_submission_envelope_invalid"):
        normalize_submission(profile(), payload, key, received)
    legacy = profile()
    legacy.pop("structured_output_protocol")
    review = request("review")
    assert submission_role(legacy, payload, "legacy-key") is None
    assert compile_submission(legacy, review, "legacy-key").receipt == compile_review_request(review, protocol="chat-strict2").receipt
