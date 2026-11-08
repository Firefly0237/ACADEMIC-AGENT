"""Three pure compiler controls, without provider I/O or runtime services."""
import copy
import json
from typing import Annotated

import pytest
from jsonschema import Draft202012Validator
from pydantic import Field, TypeAdapter, ValidationError

from citequest.agents_v3.contracts import canonical, digest
from citequest.agents_v3.provider_contract import wire_contract_violation
from citequest.agents_v3.strict_chat import (
    ENVELOPE_GATE_KEY,
    compile_review_request,
    compile_schema,
    compile_schema_v2,
    recorded_chat_response,
)


def request(schema):
    return {"model": "deepseek-flash", "reasoning": {"effort": "none"}, "truncation": "disabled",
        "instructions": "Explicit pure compiler fixture.", "max_output_tokens": 4096,
        "input": [{"role": "user", "content": "Explicit pure compiler fixture."}],
        "tools": [{"type": "function", "name": "submit_review", "parameters": schema}],
        "tool_choice": {"type": "function", "name": "submit_review"}}


def schema():
    return {"type": "object", "properties": {"explanation": {
        "type": "string", "minLength": 1, "maxLength": 120, "title": "Explanation",
    }}, "required": ["explanation"], "additionalProperties": False}


def response(text, protocol):
    raw = {"object": "chat.completion", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [{"id": "explicit-call", "type": "function",
            "function": {"name": "submit_review", "arguments": json.dumps({"explanation": text}, ensure_ascii=False)}}],
    }}], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
        "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 8}}
    return recorded_chat_response(raw, copy.deepcopy(raw), 200, protocol=protocol)


def test_compiler_one_retains_exact_original_schema_receipt_and_gate_contract():
    original = schema()
    wire, local, annotations = compile_schema(original)
    assert wire == {"type": "object", "properties": {"explanation": {"type": "string"}},
        "required": ["explanation"], "additionalProperties": False}
    assert local == [
        {"path": "/properties/explanation", "keyword": "minLength", "value": 1},
        {"path": "/properties/explanation", "keyword": "maxLength", "value": 120},
    ]
    assert annotations == [{"path": "/properties/explanation", "keyword": "title"}]
    compiled = compile_review_request(request(original), fixture=True)
    assert compiled.receipt["protocol"] == "chat-strict1"
    assert compiled.receipt["compiler"] == "deepseek-chat-schema-subset/1"
    assert set(compiled.receipt) == {"protocol", "compiler", "endpoint", "canonical_request_sha256",
        "wire_request_sha256", "full_schema_sha256", "wire_schema_sha256", "local_only_constraints", "removed_annotations"}
    assert compiled.receipt["local_only_constraints"] == local
    assert compiled.receipt["removed_annotations"] == annotations
    assert compiled.receipt["full_schema_sha256"] == digest(original)
    raw = response("中" * 123, "chat-strict1")
    assert raw[ENVELOPE_GATE_KEY] == {"protocol": "chat-strict1", "accepted": True, "error_code": None}
    assert wire_contract_violation(request(original), raw) is None
    assert original == schema()


def test_compiler_two_expresses_length_and_preserves_authoritative_unicode_newline_gate():
    payload = request(schema())
    compiled = compile_review_request(payload, protocol="chat-strict2")
    expected = "^[\\s\\S]{1,120}$"
    wire = compiled.wire["tools"][0]["function"]["parameters"]
    assert wire["properties"]["explanation"]["pattern"] == expected
    assert compiled.receipt["transformed_constraints"] == [{"path": "/properties/explanation",
        "keyword": "string_length", "min_length": 1, "max_length": 120,
        "representation": "pattern_with_authoritative_local_bounds", "pattern": expected}]
    assert {item["keyword"] for item in compiled.receipt["local_only_constraints"]} == {"minLength", "maxLength"}
    validator = Draft202012Validator(wire)
    local = TypeAdapter(Annotated[str, Field(min_length=1, max_length=120)])
    for text in ("中" * 120, "😀" * 120, "e\u0301" * 60, "第一行\n第二行"):
        assert validator.is_valid({"explanation": text})
        assert local.validate_python(text) == text
    too_long = "中" * 123
    assert not validator.is_valid({"explanation": too_long})
    assert wire_contract_violation(payload, response(too_long, "chat-strict2"), protocol="chat-strict2")["category"] == "wire_schema"
    # Python's $ also matches before a terminal newline. The provider pattern
    # is deliberately not claimed equivalent to the authoritative local bound.
    terminal_newline = "中" * 120 + "\n"
    assert validator.is_valid({"explanation": terminal_newline})
    with pytest.raises(ValidationError):
        local.validate_python(terminal_newline)
    assert wire_contract_violation(payload, response(terminal_newline, "chat-strict2"), protocol="chat-strict2") is None


def test_existing_constraints_are_preserved_and_version_two_receipts_are_stably_ordered():
    original = schema()
    original["properties"].update(
        identifier={"type": "string", "pattern": "^C[1-9]$", "minLength": 2, "maxLength": 2},
        status={"type": "string", "enum": ["ok"], "maxLength": 4},
        email={"type": "string", "format": "email", "maxLength": 99},
        notes={"type": "array", "items": {"type": "string"}, "maxItems": 1},
    )
    original["required"] = list(original["properties"])
    before = copy.deepcopy(original)
    wire, local, annotations, transformed = compile_schema_v2(original)
    properties = wire["properties"]
    assert properties["identifier"]["pattern"] == "^C[1-9]$"
    assert properties["status"]["enum"] == ["ok"] and "pattern" not in properties["status"]
    assert properties["email"]["format"] == "email" and "pattern" not in properties["email"]
    assert all("Authoritative local Unicode code-point length bounds" in properties[name]["description"]
        for name in ("explanation", "identifier", "status", "email"))
    assert [item["path"] for item in transformed] == ["/properties/explanation"]
    assert any(item["keyword"] == "maxItems" for item in local)
    assert local == sorted(local, key=canonical) and annotations == sorted(annotations, key=canonical)
    reordered = json.loads(json.dumps(original, sort_keys=True))
    first = compile_review_request(request(original), protocol="chat-strict2").receipt
    replay = compile_review_request(request(reordered), protocol="chat-strict2").receipt
    assert first == replay and original == before
