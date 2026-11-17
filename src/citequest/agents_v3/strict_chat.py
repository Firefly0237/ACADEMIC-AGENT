"""Versioned review-only DeepSeek Beta Chat strict transport adapter.

Canonical business requests and original local validators remain authoritative.
Schema adaptation is recorded, raw provider replies remain ledger evidence, and
normalization changes only the protocol envelope, never function arguments.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from .contracts import AgentError, canonical, digest
from .model_contracts import model_contract

VERSION = "chat-strict1"
PROFILE_KEY = "review_api_protocol"
COMPILER = "deepseek-chat-schema-subset/1"
LENGTH_VERSION = "chat-strict2"
LENGTH_COMPILER = "deepseek-chat-schema-subset/2"
STRUCTURED_VERSION = "chat-functions1"
PROTOCOL_COMPILERS = {VERSION: COMPILER, LENGTH_VERSION: LENGTH_COMPILER}
FUNCTION_NAME = "submit_review"
ENVELOPE_GATE_KEY = "_cq_review_envelope_gate"
ENVELOPE_ERRORS = frozenset({
    "chat_strict_response_not_object", "chat_strict_response_not_completed",
    "chat_strict_response_message_invalid", "chat_strict_thinking_output_forbidden",
    "chat_strict_exactly_one_submission_required", "chat_strict_submission_envelope_invalid",
    "chat_strict_reserved_receive_gate", "provider_http_error_with_known_usage",
})
LIVE_ENDPOINT = "https://api.deepseek.com/beta/chat/completions"
FIXTURE_ENDPOINT = "https://s08-provider.invalid/beta/chat/completions"
FORMATS = frozenset({"email", "hostname", "ipv4", "ipv6", "uuid"})
ANNOTATIONS = frozenset({"title", "examples"})
LOCAL_BOUNDS = frozenset({"minLength", "maxLength", "minItems", "maxItems"})
COMMON = frozenset({"type", "description", "enum"})
TYPE_KEYS = {
    "object": COMMON | {"properties", "required", "additionalProperties"},
    "string": COMMON | {"pattern", "format"},
    "number": COMMON | {"const", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"},
    "integer": COMMON | {"const", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"},
    "boolean": COMMON,
    "array": COMMON | {"items"},
}


def is_strict_review(profile: dict, request: dict) -> bool:
    selected = profile.get(PROFILE_KEY)
    if selected is None:
        return False
    compatible = (selected == VERSION and profile.get("qa_execution_protocol") in {"grounded1", "grounded2", "grounded3"}
        or selected == LENGTH_VERSION and profile.get("qa_execution_protocol") == "grounded3")
    profile_id = profile.get("profile_id")
    if (not compatible or profile_id not in {"qa-v6", "qa-v7"}
            or profile_id == "qa-v7" and (selected != LENGTH_VERSION
                or profile.get("qa_execution_protocol") != "grounded3"
                or profile.get("model_alias") != "deepseek-v4-pro")):
        raise AgentError("chat_strict_Task_profile_not_supported")
    tools = request.get("tools") or []
    if not isinstance(tools, list):
        raise AgentError("chat_strict_review_tool_list_invalid")
    is_submission = any(isinstance(tool, dict) and tool.get("name") == FUNCTION_NAME for tool in tools)
    if not is_submission:
        return False
    if (profile.get("review_transport_protocol") != "function1" or len(tools) != 1
            or tools[0].get("type") != "function"
            or request.get("tool_choice") != {"type": "function", "name": FUNCTION_NAME}):
        raise AgentError("chat_strict_review_submission_required")
    return True


def _pointer(path: str, key: str) -> str:
    return path + "/" + key.replace("~", "~0").replace("/", "~1")


def compile_schema(full_schema: dict, *, allow_nullable: bool = False) -> tuple[dict, list[dict], list[dict]]:
    """Compile only documented structural keys; disclose local-only bounds.

    The original complete schema is neither mutated nor weakened locally.
    Unknown validation semantics, nullable types and recursive references fail
    before a request is sent instead of silently entering an undocumented mode.
    """
    original = copy.deepcopy(full_schema)
    local_only: list[dict] = []
    annotations: list[dict] = []

    def resolve(reference: str) -> dict:
        if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            raise AgentError("chat_strict_schema_reference_not_supported")
        value = original
        for part in reference[2:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(value, dict) or key not in value:
                raise AgentError("chat_strict_schema_reference_missing")
            value = value[key]
        if not isinstance(value, dict):
            raise AgentError("chat_strict_schema_reference_invalid")
        return value

    def visit(node: dict, path: str, references: tuple[str, ...]) -> dict:
        if not isinstance(node, dict):
            raise AgentError("chat_strict_schema_node_invalid")
        if "$ref" in node:
            reference = node["$ref"]
            if reference in references:
                raise AgentError("chat_strict_recursive_schema_not_supported")
            siblings = {key: value for key, value in node.items() if key != "$ref"}
            if set(siblings) - ANNOTATIONS - {"description"}:
                raise AgentError("chat_strict_reference_validation_siblings_not_supported")
            expanded = copy.deepcopy(resolve(reference))
            expanded.update(siblings)
            return visit(expanded, path, (*references, reference))
        value = {}
        for key, child in node.items():
            if key == "$defs":
                continue
            if key in ANNOTATIONS or (key == "default" and node.get("type") not in {"number", "integer"}):
                annotations.append({"path": path or "/", "keyword": key})
            elif key in LOCAL_BOUNDS:
                if type(child) is not int or child < 0:
                    raise AgentError("chat_strict_schema_local_bound_invalid")
                local_only.append({"path": path or "/", "keyword": key, "value": child})
            else:
                value[key] = copy.deepcopy(child)
        if value.get("type") == "string" and "const" in value:
            constant = value.pop("const")
            if type(constant) is not str or ("enum" in value and value["enum"] != [constant]):
                raise AgentError("chat_strict_schema_string_identity_invalid")
            value["enum"] = [constant]
        if "anyOf" in value:
            if set(value) - {"anyOf", "description"} or not isinstance(value["anyOf"], list) or not value["anyOf"]:
                raise AgentError("chat_strict_schema_anyOf_invalid")
            value["anyOf"] = [visit(child, _pointer(path, "anyOf") + "/" + str(index), references)
                for index, child in enumerate(value["anyOf"])]
            return value
        kind = value.get("type")
        if kind == "null" and allow_nullable:
            if set(value) - {"type", "description"}:
                raise AgentError("chat_strict_nullable_schema_not_supported")
            return value
        if not isinstance(kind, str) or kind not in TYPE_KEYS or set(value) - TYPE_KEYS[kind]:
            raise AgentError("chat_strict_schema_keyword_or_type_not_supported")
        if kind == "object":
            properties = value.get("properties")
            required = value.get("required")
            if (not isinstance(properties, dict) or not isinstance(required, list)
                    or len(required) != len(properties) or set(required) != set(properties)
                    or value.get("additionalProperties") is not False):
                raise AgentError("chat_strict_closed_required_object_needed")
            value["properties"] = {key: visit(child, _pointer(_pointer(path, "properties"), key), references)
                for key, child in properties.items()}
        if kind == "array":
            value["items"] = visit(value.get("items"), _pointer(path, "items"), references)
        if kind == "string" and "format" in value and value["format"] not in FORMATS:
            raise AgentError("chat_strict_string_format_not_supported")
        return value

    wire = visit(original, "", ())
    if wire.get("type") != "object":
        raise AgentError("chat_strict_root_object_required")
    return wire, local_only, annotations


def compile_schema_v2(full_schema: dict, *, allow_nullable: bool = False) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Map plain-string bounds without changing the historical compiler output.

    The original local schema remains authoritative. Existing patterns, enums or
    formats are preserved and their length bounds remain explicitly local-only.
    Regex engines can differ on Unicode and terminal-newline handling.
    """
    wire, local_only, annotations = compile_schema(full_schema, allow_nullable=allow_nullable)
    bounds_by_path = {}
    for item in local_only:
        if item["keyword"] in {"minLength", "maxLength"}:
            bounds_by_path.setdefault(item["path"], {})[item["keyword"]] = item["value"]
    transformed = []
    for path, bounds in sorted(bounds_by_path.items()):
        node = wire
        tokens = path.lstrip("/").split("/") if path != "/" else []
        for token in tokens:
            key = token.replace("~1", "/").replace("~0", "~")
            node = node[int(key)] if isinstance(node, list) else node[key]
        minimum, maximum = bounds.get("minLength", 0), bounds.get("maxLength")
        if maximum is not None and minimum > maximum:
            raise AgentError("chat_strict_string_length_bounds_inconsistent")
        upper = str(maximum) if maximum is not None else "unbounded"
        note = f"Authoritative local Unicode code-point length bounds: minimum {minimum}, maximum {upper}."
        node["description"] = (node.get("description", "") + "\n" + note).lstrip("\n")
        if {"pattern", "enum", "format"} & node.keys():
            continue
        upper = str(maximum) if maximum is not None else ""
        pattern = "^[\\s\\S]{" + str(minimum) + "," + upper + "}$"
        node["pattern"] = pattern
        transformed.append({
            "path": path, "keyword": "string_length", "min_length": minimum,
            "max_length": maximum, "representation": "pattern_with_authoritative_local_bounds", "pattern": pattern,
        })
    return wire, sorted(local_only, key=canonical), sorted(annotations, key=canonical), sorted(transformed, key=canonical)


@dataclass(frozen=True)
class CompiledChatReview:
    wire: dict
    receipt: dict

    def sdk_arguments(self) -> dict:
        value = copy.deepcopy(self.wire)
        thinking = value.pop("thinking")
        value["extra_body"] = {"thinking": thinking}
        return value


def compile_review_request(request: dict, *, fixture: bool = False, protocol: str = VERSION) -> CompiledChatReview:
    if not isinstance(protocol, str) or protocol not in PROTOCOL_COMPILERS:
        raise AgentError("chat_strict_compiler_protocol_not_supported")
    tools = request.get("tools")
    inputs = request.get("input")
    model_alias = model_contract(request.get("model"))["model_alias"]
    if model_alias != "deepseek-flash" and protocol != LENGTH_VERSION:
        raise AgentError("chat_strict_model_protocol_not_supported")
    if (not isinstance(tools, list) or len(tools) != 1 or tools[0].get("type") != "function"
            or tools[0].get("name") != FUNCTION_NAME or not isinstance(tools[0].get("parameters"), dict)
            or request.get("tool_choice") != {"type": "function", "name": FUNCTION_NAME}
            or request.get("reasoning") != {"effort": "none"}
            or not isinstance(request.get("instructions"), str)
            or not isinstance(inputs, list) or len(inputs) != 1 or inputs[0].get("role") != "user"
            or type(inputs[0].get("content")) is not str or request.get("text") is not None
            or type(request.get("max_output_tokens")) is not int or not 1 <= request["max_output_tokens"] <= 4096):
        raise AgentError("chat_strict_canonical_review_shape_not_supported")
    if protocol == LENGTH_VERSION:
        schema, local_only, annotations, transformed = compile_schema_v2(tools[0]["parameters"])
    else:
        schema, local_only, annotations = compile_schema(tools[0]["parameters"])
    wire = {
        "model": request["model"],
        "messages": [{"role": "system", "content": request["instructions"]}, copy.deepcopy(inputs[0])],
        "reasoning_effort": "none",
        "thinking": {"type": "disabled"},
        "max_tokens": request["max_output_tokens"],
        "stream": False,
        "tools": [{"type": "function", "function": {"name": FUNCTION_NAME,
            "description": tools[0].get("description", "Submit the complete independent review."),
            "parameters": schema, "strict": True}}],
        "tool_choice": {"type": "function", "function": {"name": FUNCTION_NAME}},
    }
    receipt = {"protocol": protocol, "compiler": PROTOCOL_COMPILERS[protocol],
        "endpoint": FIXTURE_ENDPOINT if fixture else LIVE_ENDPOINT,
        "canonical_request_sha256": digest(request), "wire_request_sha256": digest(wire),
        "full_schema_sha256": digest(tools[0]["parameters"]), "wire_schema_sha256": digest(schema),
        "local_only_constraints": local_only, "removed_annotations": annotations}
    if protocol == LENGTH_VERSION:
        receipt["transformed_constraints"] = transformed
    return CompiledChatReview(wire, receipt)


def usage_counts_chat(response: dict) -> dict:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise AgentError("provider_usage_missing")
    keys = ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
    values = [usage.get(key) for key in keys]
    if (any(type(value) is not int or value < 0 for value in values)
            or values[0] + values[1] != values[2] or values[3] + values[4] != values[0]):
        raise AgentError("provider_usage_malformed")
    detail = usage.get("prompt_tokens_details")
    if detail is not None and (not isinstance(detail, dict)
            or ("cached_tokens" in detail and (type(detail["cached_tokens"]) is not int or detail["cached_tokens"] != values[3]))):
        raise AgentError("provider_usage_unrecognized")
    return {"input": values[0], "cached": values[3], "output": values[1]}


def recorded_chat_response(raw: dict, private: dict, http_status: int, *, protocol: str = VERSION,
                           function_name: str = FUNCTION_NAME) -> dict:
    """Persist the raw-envelope decision beside the redacted receipt atomically.

    A provider-owned field with our reserved name is a conflict, never an
    attestation. The fixed local decision always replaces that untrusted field.
    Usage is interpreted separately after the complete receipt is preserved.
    """
    error_code = None
    if ENVELOPE_GATE_KEY in raw:
        error_code = "chat_strict_reserved_receive_gate"
    elif http_status != 200:
        error_code = "provider_http_error_with_known_usage"
    else:
        try:
            normalize_chat_response(raw, require_gate=False, protocol=protocol, function_name=function_name)
        except AgentError as error:
            error_code = error.code
    value = copy.deepcopy(private)
    value[ENVELOPE_GATE_KEY] = {"protocol": protocol, "accepted": error_code is None, "error_code": error_code}
    return value


def normalize_chat_response(response: dict, *, require_gate: bool = True, protocol: str = VERSION,
                            function_name: str = FUNCTION_NAME) -> dict:
    """Return a canonical envelope without repairing or reserializing arguments."""
    if not isinstance(protocol, str) or protocol not in {*PROTOCOL_COMPILERS, STRUCTURED_VERSION}:
        raise AgentError("chat_strict_compiler_protocol_not_supported")
    if not isinstance(function_name, str) or function_name not in {"submit_plan", "submit_answer", FUNCTION_NAME}:
        raise AgentError("chat_strict_submission_function_not_supported")
    if not isinstance(response, dict):
        raise AgentError("chat_strict_response_not_object")
    if require_gate:
        gate = response.get(ENVELOPE_GATE_KEY)
        if gate is None:
            raise AgentError("chat_strict_recorded_receive_gate_missing")
        if (not isinstance(gate, dict) or set(gate) != {"protocol", "accepted", "error_code"}
                or gate["protocol"] != protocol or type(gate["accepted"]) is not bool
                or (gate["accepted"] and gate["error_code"] is not None)
                or (not gate["accepted"] and (type(gate["error_code"]) is not str or gate["error_code"] not in ENVELOPE_ERRORS))):
            raise AgentError("chat_strict_recorded_receive_gate_invalid")
        if not gate["accepted"]:
            raise AgentError(gate["error_code"])
    elif ENVELOPE_GATE_KEY in response:
        raise AgentError("chat_strict_reserved_receive_gate")
    choices = response.get("choices")
    if (response.get("object") != "chat.completion" or not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict) or type(choices[0].get("index")) is not int or choices[0]["index"] != 0
            or choices[0].get("finish_reason") != "tool_calls"):
        raise AgentError("chat_strict_response_not_completed")
    message = choices[0].get("message")
    if (not isinstance(message, dict) or message.get("role") != "assistant"
            or message.get("content") not in (None, "") or message.get("refusal") not in (None, "")
            or message.get("reasoning_content") not in (None, "")
            or response.get("error") is not None):
        raise AgentError("chat_strict_response_message_invalid")
    usage = response.get("usage")
    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    if details is not None and (not isinstance(details, dict)
            or ("reasoning_tokens" in details and (type(details["reasoning_tokens"]) is not int or details["reasoning_tokens"] != 0))):
        raise AgentError("chat_strict_thinking_output_forbidden")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        raise AgentError("chat_strict_exactly_one_submission_required")
    call = calls[0]
    function = call.get("function")
    if (call.get("type") != "function" or not isinstance(function, dict)
            or function.get("name") != function_name or type(call.get("id")) is not str
            or not 1 <= len(call["id"]) <= 300 or not call["id"].strip()
            or type(function.get("arguments")) is not str or not function["arguments"].strip()):
        raise AgentError("chat_strict_submission_envelope_invalid")
    return {"status": "completed", "output": [{"type": "function_call", "name": function_name,
        "call_id": call["id"], "arguments": function["arguments"], "status": "completed"}],
        "_cq_review_api_protocol": protocol}
