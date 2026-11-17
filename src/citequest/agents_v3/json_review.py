"""Explicit JSON-object review transport with authoritative local schemas.

The provider constrains JSON syntax only. Domain schemas, source relations and
publication decisions remain the original local and independent-review gates.
"""
from __future__ import annotations

import copy
import re

from .contracts import AgentError, canonical, digest
from .strict_chat import CompiledChatReview, ENVELOPE_GATE_KEY
from .qa_adaptive_contracts import SUPPLEMENT_ROLE

VERSION = "chat-json1"
COMPILER = "deepseek-chat-json-object/1"
JSON2_VERSION = "chat-json2"
JSON2_COMPILER = "deepseek-chat-json-object/2"
REASONING_KEY = "_cq_private_reasoning"
LIVE_ENDPOINT = "https://api.deepseek.com/chat/completions"
FIXTURE_ENDPOINT = "https://s08-provider.invalid/chat/completions"
WIRE_SCHEMA = {"type": "object"}
ERRORS = frozenset({"chat_json_response_not_object", "chat_json_response_not_completed",
    "chat_json_message_invalid", "chat_json_blank_content", "chat_json_thinking_output_forbidden",
    "chat_json_reserved_receive_gate", "provider_http_error_with_known_usage"})
JSON2_ERRORS = ERRORS | {"chat_json_reasoning_metadata_invalid", "chat_json_recorded_reasoning_content_forbidden"}


def compile_json_review(request: dict, *, fixture: bool = False) -> CompiledChatReview:
    return _compile_json_object(request, "review", fixture=fixture, protocol=VERSION,
        compiler=COMPILER, structured="chat-functions1", maximum=4096, effort="none")


def _require_submission_role(role: str, *, adaptive: bool) -> None:
    if role not in {"plan", "author", "review"} and not (adaptive is True and role == SUPPLEMENT_ROLE):
        raise AgentError("chat_json_submission_role_invalid")


def compile_json_submission(request: dict, role: str, *, fixture: bool = False,
                            maximum: int = 8192, adaptive: bool = False) -> CompiledChatReview:
    _require_submission_role(role, adaptive=adaptive)
    if type(maximum) is not int or maximum not in {8192, 65536}:
        raise AgentError("chat_json_submission_output_contract_invalid")
    return _compile_json_object(request, role, fixture=fixture, protocol=JSON2_VERSION,
        compiler=JSON2_COMPILER, structured=JSON2_VERSION, maximum=maximum,
        effort="low" if role == "review" else "none")


def _compile_json_object(request, role, *, fixture, protocol, compiler, structured, maximum, effort):
    text = request.get("text")
    fmt = text.get("format") if isinstance(text, dict) else None
    inputs = request.get("input")
    if (request.get("model") != "deepseek-v4-pro" or request.get("tools")
            or request.get("tool_choice") is not None or not isinstance(fmt, dict)
            or fmt.get("type") != "json_schema" or fmt.get("name") != "citequest_qa5_" + role
            or fmt.get("strict") is not True or not isinstance(fmt.get("schema"), dict)
            or fmt["schema"].get("type") != "object"
            or request.get("reasoning") != {"effort": effort} or request.get("truncation") != "disabled"
            or not isinstance(request.get("instructions"), str)
            or not isinstance(inputs, list) or len(inputs) != 1 or inputs[0].get("role") != "user"
            or type(inputs[0].get("content")) is not str
            or type(request.get("max_output_tokens")) is not int or not 1 <= request["max_output_tokens"] <= maximum):
        raise AgentError("chat_json_canonical_review_invalid")
    schema = fmt["schema"]
    # Only server-fixed binding strings appear in this syntax illustration.
    # No example supplies a default factual or acceptance opinion.
    example = {}
    for name, field in schema.get("properties", {}).items():
        if name.endswith("sha256"):
            value = field.get("const")
            if value is None and isinstance(field.get("enum"), list) and len(field["enum"]) == 1:
                value = field["enum"][0]
            if isinstance(value, str):
                example[name] = value
    contract = {"instruction": "Return one complete JSON object, without Markdown fences or surrounding prose. Follow every required field of the full schema. The syntax illustration contains only fixed identities, is incomplete, and supplies no review verdicts. The schema and example supply no factual or acceptance opinions. Follow the canonical task instructions to determine field values.",
        "syntax_illustration_only": example, "full_authoritative_schema": schema}
    wire = {"model": request["model"], "messages": [{"role": "system", "content":
        request["instructions"] + "\n<json_output_contract>\n" + canonical(contract).decode()
        + "\n</json_output_contract>"}, copy.deepcopy(inputs[0])],
        "reasoning_effort": effort, "thinking": {"type": "enabled" if effort == "low" else "disabled"},
        "max_tokens": request["max_output_tokens"], "stream": False,
        "response_format": {"type": "json_object"}}
    receipt = {"protocol": protocol, "compiler": compiler,
        "endpoint": FIXTURE_ENDPOINT if fixture else LIVE_ENDPOINT,
        "canonical_request_sha256": digest(request), "wire_request_sha256": digest(wire),
        "full_schema_sha256": digest(schema), "wire_schema_sha256": digest(WIRE_SCHEMA),
        "local_only_constraints": [{"path": "/", "keyword": "full_schema", "authority": "local"}],
        "removed_annotations": [], "transformed_constraints": [],
        "structured_output_protocol": structured, "role": role,
        "schema_name": "citequest_qa5_" + role, "schema_enforcement": "local_authoritative"}
    return CompiledChatReview(wire, receipt)


def normalize_json_review(response: dict, *, require_gate: bool = True) -> dict:
    if not isinstance(response, dict):
        raise AgentError("chat_json_response_not_object")
    if require_gate:
        gate = response.get(ENVELOPE_GATE_KEY)
        if (not isinstance(gate, dict) or set(gate) != {"protocol", "accepted", "error_code"}
                or gate["protocol"] != VERSION or type(gate["accepted"]) is not bool
                or (gate["accepted"] and gate["error_code"] is not None)
                or (not gate["accepted"] and (type(gate["error_code"]) is not str or gate["error_code"] not in ERRORS))):
            raise AgentError("chat_json_recorded_receive_gate_invalid")
        if not gate["accepted"]:
            raise AgentError(gate["error_code"])
    elif ENVELOPE_GATE_KEY in response:
        raise AgentError("chat_json_reserved_receive_gate")
    choices = response.get("choices")
    if (response.get("object") != "chat.completion" or not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict) or type(choices[0].get("index")) is not int
            or choices[0]["index"] != 0 or choices[0].get("finish_reason") != "stop"):
        raise AgentError("chat_json_response_not_completed")
    message = choices[0].get("message")
    if (not isinstance(message, dict) or message.get("role") != "assistant"
            or type(message.get("content")) is not str or message.get("refusal") not in (None, "")
            or message.get("tool_calls") not in (None, []) or message.get("function_call") is not None
            or response.get("error") is not None):
        raise AgentError("chat_json_message_invalid")
    usage = response.get("usage")
    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    if (message.get("reasoning_content") not in (None, "") or details is not None and (
            not isinstance(details, dict) or "reasoning_tokens" in details and (
                type(details["reasoning_tokens"]) is not int or details["reasoning_tokens"] != 0))):
        raise AgentError("chat_json_thinking_output_forbidden")
    if not message["content"].strip():
        raise AgentError("chat_json_blank_content")
    return {"status": "completed", "output": [{"type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": message["content"]}]}],
        "_cq_review_api_protocol": VERSION}


def recorded_json_review(raw: dict, private: dict, http_status: int) -> dict:
    code = None
    if ENVELOPE_GATE_KEY in raw:
        code = "chat_json_reserved_receive_gate"
    elif http_status != 200:
        code = "provider_http_error_with_known_usage"
    else:
        try:
            normalize_json_review(raw, require_gate=False)
        except AgentError as error:
            code = error.code
    value = copy.deepcopy(private)
    value[ENVELOPE_GATE_KEY] = {"protocol": VERSION, "accepted": code is None, "error_code": code}
    return value


def _reasoning_attestation(response: dict, role: str) -> dict:
    choices = response.get("choices")
    message = (choices[0].get("message") if isinstance(choices, list) and choices
        and isinstance(choices[0], dict) else None)
    reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
    if reasoning is None:
        reasoning = ""
    if type(reasoning) is not str:
        raise AgentError("chat_json_reasoning_metadata_invalid")
    try:
        encoded = reasoning.encode("utf-8")
    except UnicodeError as error:
        raise AgentError("chat_json_reasoning_metadata_invalid") from error
    return {"protocol": JSON2_VERSION, "role": role, "present": bool(reasoning),
        "characters": len(reasoning), "utf8_bytes": len(encoded), "sha256": digest(encoded)}


def normalize_json_submission(response: dict, role: str, *, require_gate: bool = True, adaptive: bool = False) -> dict:
    """Accept private reasoning only for the explicitly versioned review role."""
    _require_submission_role(role, adaptive=adaptive)
    if not isinstance(response, dict):
        raise AgentError("chat_json_response_not_object")
    if require_gate:
        gate = response.get(ENVELOPE_GATE_KEY)
        if (not isinstance(gate, dict) or set(gate) != {"protocol", "accepted", "error_code"}
                or gate["protocol"] != JSON2_VERSION or type(gate["accepted"]) is not bool
                or (gate["accepted"] and gate["error_code"] is not None)
                or (not gate["accepted"] and (type(gate["error_code"]) is not str or gate["error_code"] not in JSON2_ERRORS))):
            raise AgentError("chat_json_recorded_receive_gate_invalid")
        if not gate["accepted"]:
            raise AgentError(gate["error_code"])
        attestation = response.get(REASONING_KEY)
        if (not isinstance(attestation, dict)
                or set(attestation) != {"protocol", "role", "present", "characters", "utf8_bytes", "sha256"}
                or attestation["protocol"] != JSON2_VERSION or attestation["role"] != role
                or type(attestation["present"]) is not bool
                or any(type(attestation[key]) is not int or not 0 <= attestation[key] <= 4 * 1024 * 1024
                    for key in ("characters", "utf8_bytes"))
                or attestation["present"] != (attestation["characters"] > 0)
                or not attestation["characters"] <= attestation["utf8_bytes"] <= 4 * attestation["characters"]
                or type(attestation["sha256"]) is not str or re.fullmatch(r"[a-f0-9]{64}", attestation["sha256"]) is None
                or not attestation["present"] and attestation["sha256"] != digest(b"")
                or role != "review" and attestation["present"]):
            raise AgentError("chat_json_reasoning_metadata_invalid")
    elif ENVELOPE_GATE_KEY in response or REASONING_KEY in response:
        raise AgentError("chat_json_reserved_receive_gate")
    choices = response.get("choices")
    if (response.get("object") != "chat.completion" or not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict) or type(choices[0].get("index")) is not int
            or choices[0]["index"] != 0 or choices[0].get("finish_reason") != "stop"):
        raise AgentError("chat_json_response_not_completed")
    message = choices[0].get("message")
    if (not isinstance(message, dict) or message.get("role") != "assistant"
            or type(message.get("content")) is not str or message.get("refusal") not in (None, "")
            or message.get("tool_calls") not in (None, []) or message.get("function_call") is not None
            or response.get("error") is not None):
        raise AgentError("chat_json_message_invalid")
    usage = response.get("usage")
    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    if details is not None:
        if not isinstance(details, dict):
            raise AgentError("chat_json_reasoning_metadata_invalid")
        if "reasoning_tokens" in details:
            count = details["reasoning_tokens"]
            completion = usage.get("completion_tokens")
            if (type(count) is not int or count < 0 or type(completion) is int and count > completion
                    or role != "review" and count != 0):
                raise AgentError("chat_json_reasoning_metadata_invalid")
    if require_gate:
        if message.get("reasoning_content") not in (None, ""):
            raise AgentError("chat_json_recorded_reasoning_content_forbidden")
    else:
        attestation = _reasoning_attestation(response, role)
        if role != "review" and attestation["present"]:
            raise AgentError("chat_json_thinking_output_forbidden")
    if not message["content"].strip():
        raise AgentError("chat_json_blank_content")
    return {"status": "completed", "output": [{"type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": message["content"]}]}],
        "_cq_review_api_protocol": JSON2_VERSION}


def recorded_json_submission(raw: dict, private: dict, http_status: int, role: str, *, adaptive: bool = False) -> dict:
    _require_submission_role(role, adaptive=adaptive)
    code = None
    if ENVELOPE_GATE_KEY in raw or REASONING_KEY in raw:
        code = "chat_json_reserved_receive_gate"
    elif http_status != 200:
        code = "provider_http_error_with_known_usage"
    else:
        try:
            normalize_json_submission(raw, role, require_gate=False, adaptive=adaptive)
        except AgentError as error:
            code = error.code
    try:
        attestation = _reasoning_attestation(raw, role)
    except AgentError:
        attestation = {"protocol": JSON2_VERSION, "role": role, "present": False,
            "characters": 0, "utf8_bytes": 0, "sha256": digest(b"")}
        code = code or "chat_json_reasoning_metadata_invalid"
    value = copy.deepcopy(private)
    # Caller redaction also removes this field. Enforce it here so the recorded
    # contract never depends on a caller remembering to redact private thought.
    choices = value.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                choice["message"].pop("reasoning_content", None)
    value[REASONING_KEY] = attestation
    value[ENVELOPE_GATE_KEY] = {"protocol": JSON2_VERSION, "accepted": code is None, "error_code": code}
    return value
