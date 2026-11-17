"""Explicit all-role typed submissions over the existing guarded SDK client.

Tool names capture structured results and never execute an external MCP action.
Nullable schemas are an isolated, undocumented capability requiring admission.
"""
from __future__ import annotations

import copy
import re

from .contracts import AgentError, digest
from .model_contracts import validated_model_alias
from .qa_adaptive_contracts import SUPPLEMENT_ROLE, SUPPLEMENT_SCHEMA_NAME
from .json_review import VERSION as JSON_VERSION
from .json_review import compile_json_review, normalize_json_review, recorded_json_review
from .json_review import JSON2_VERSION, compile_json_submission, normalize_json_submission, recorded_json_submission
from .strict_chat import (
    FIXTURE_ENDPOINT,
    LIVE_ENDPOINT,
    STRUCTURED_VERSION,
    CompiledChatReview,
    compile_review_request,
    compile_schema_v2,
    normalize_chat_response,
    recorded_chat_response,
)

VERSION = STRUCTURED_VERSION
COMPILER = "deepseek-chat-functions-schema/1"
PROFILE_KEY = "structured_output_protocol"
NULLABLE_CONTRACT = "undocumented_null_extension_requires_probe"
ROLE_FUNCTIONS = {"plan": "submit_plan", "author": "submit_answer", "review": "submit_review"}
ROLE_SCHEMAS = {role: "citequest_qa5_" + role for role in ROLE_FUNCTIONS}
ROLE_SCHEMAS[SUPPLEMENT_ROLE] = SUPPLEMENT_SCHEMA_NAME


def submission_role(profile: dict, request: dict, logical_key: str) -> str | None:
    """Require the immutable mode and exact role/key/schema identity together."""
    selected = profile.get(PROFILE_KEY)
    if selected is None:
        return None
    if selected == JSON2_VERSION:
        if (profile.get("profile_id") not in {"qa-v8", "qa-v9"} or profile.get("qa_execution_protocol") != "grounded3"
                or profile.get("review_api_protocol") != JSON2_VERSION
                or profile.get("review_transport_protocol") != "json1"
                or profile.get("qa_context_protocol") != "references1"
                or profile.get("answer_revision_protocol") != "jsonpatch1"
                or validated_model_alias(profile) != "deepseek-v4-pro"):
            raise AgentError("typed_submission_Task_profile_not_supported")
    elif (selected != VERSION or profile.get("profile_id") != "qa-v7"
            or profile.get("qa_execution_protocol") != "grounded3"
            or (profile.get("review_api_protocol"), profile.get("review_transport_protocol"))
                not in {("chat-strict2", "function1"), (JSON_VERSION, "json1")}
            or profile.get("review_api_protocol") == JSON_VERSION
                and profile.get("answer_revision_protocol") != "jsonpatch1"
            or validated_model_alias(profile) != "deepseek-v4-pro"):
        raise AgentError("typed_submission_Task_profile_not_supported")
    match = (re.fullmatch(r".+:qa5:(plan|author|review|supplement_plan):(0|[1-9][0-9]*)(?::wire-recovery:1)?", logical_key)
        if isinstance(logical_key, str) else None)
    if match is None:
        raise AgentError("typed_submission_role_key_invalid")
    role = match.group(1)
    if role == SUPPLEMENT_ROLE:
        from .product_profiles import qa_retrieval_protocol
        if selected != JSON2_VERSION or qa_retrieval_protocol({"kind": "qa", "profile": profile}) != "adaptive1":
            raise AgentError("typed_submission_supplement_protocol_required")
    if role == "review" and profile.get("review_api_protocol") not in {JSON_VERSION, JSON2_VERSION}:
        tools = request.get("tools")
        if (request.get("text") is not None or not isinstance(tools, list) or len(tools) != 1
                or tools[0].get("type") != "function" or tools[0].get("name") != ROLE_FUNCTIONS[role]
                or request.get("tool_choice") != {"type": "function", "name": ROLE_FUNCTIONS[role]}
                or not isinstance(tools[0].get("parameters"), dict)):
            raise AgentError("typed_submission_review_schema_required")
    else:
        text = request.get("text")
        fmt = text.get("format") if isinstance(text, dict) else None
        if (request.get("tools") or request.get("tool_choice") is not None
                or not isinstance(fmt, dict) or fmt.get("type") != "json_schema"
                or fmt.get("name") != ROLE_SCHEMAS[role] or fmt.get("strict") is not True
                or not isinstance(fmt.get("schema"), dict)):
            raise AgentError("typed_submission_role_schema_required")
    return role


def compile_submission(profile: dict, request: dict, logical_key: str, *, fixture: bool = False) -> CompiledChatReview:
    role = submission_role(profile, request, logical_key)
    if role is None:
        return compile_review_request(request, fixture=fixture, protocol=profile.get("review_api_protocol", "chat-strict1"))
    if profile.get(PROFILE_KEY) == JSON2_VERSION:
        maximum = 65536 if profile["profile_id"] == "qa-v9" else 8192
        return compile_json_submission(request, role, fixture=fixture, maximum=maximum,
            adaptive=role == SUPPLEMENT_ROLE)
    if role == "review" and profile.get("review_api_protocol") == JSON_VERSION:
        return compile_json_review(request, fixture=fixture)
    inputs = request.get("input")
    if (request.get("model") != validated_model_alias(profile)
            or request.get("reasoning") != {"effort": "none"}
            or request.get("truncation") != "disabled" or not isinstance(request.get("instructions"), str)
            or not isinstance(inputs, list) or len(inputs) != 1 or inputs[0].get("role") != "user"
            or type(inputs[0].get("content")) is not str
            or type(request.get("max_output_tokens")) is not int or not 1 <= request["max_output_tokens"] <= 4096):
        raise AgentError("typed_submission_canonical_request_invalid")
    schema = (request["tools"][0]["parameters"] if role == "review" else request["text"]["format"]["schema"])
    wire_schema, local, annotations, transformed = compile_schema_v2(schema, allow_nullable=True)
    function = ROLE_FUNCTIONS[role]
    wire = {
        "model": request["model"],
        "messages": [{"role": "system", "content": request["instructions"]}, copy.deepcopy(inputs[0])],
        "reasoning_effort": "none", "thinking": {"type": "disabled"},
        "max_tokens": request["max_output_tokens"], "stream": False,
        "tools": [{"type": "function", "function": {"name": function, "strict": True,
            "description": "Submit the complete typed " + role + " result. This submission executes no external action.",
            "parameters": wire_schema}}],
        "tool_choice": {"type": "function", "function": {"name": function}},
    }
    receipt = {
        "protocol": VERSION, "compiler": COMPILER,
        "endpoint": FIXTURE_ENDPOINT if fixture else LIVE_ENDPOINT,
        "canonical_request_sha256": digest(request), "wire_request_sha256": digest(wire),
        "full_schema_sha256": digest(schema), "wire_schema_sha256": digest(wire_schema),
        "local_only_constraints": local, "removed_annotations": annotations,
        "transformed_constraints": transformed, "structured_output_protocol": VERSION,
        "role": role, "function_name": function, "schema_name": ROLE_SCHEMAS[role],
        "nullable_type_contract": NULLABLE_CONTRACT,
    }
    return CompiledChatReview(wire, receipt)


def normalize_submission(profile: dict, request: dict, logical_key: str, raw_private_chat: dict) -> dict:
    role = submission_role(profile, request, logical_key)
    if role is None:
        return normalize_chat_response(raw_private_chat, protocol=profile.get("review_api_protocol", "chat-strict1"))
    if profile.get(PROFILE_KEY) == JSON2_VERSION:
        return normalize_json_submission(raw_private_chat, role, adaptive=role == SUPPLEMENT_ROLE)
    if role == "review" and profile.get("review_api_protocol") == JSON_VERSION:
        return normalize_json_review(raw_private_chat)
    response = normalize_chat_response(raw_private_chat, protocol=VERSION, function_name=ROLE_FUNCTIONS[role])
    if role == "review":
        return response
    arguments = response["output"][0]["arguments"]
    return {
        "status": "completed", "output": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": arguments},
        ]}], "_cq_structured_output_protocol": VERSION, "_cq_submission_role": role,
    }


def recorded_submission(profile: dict, request: dict, logical_key: str, raw: dict, private: dict, http_status: int) -> dict:
    role = submission_role(profile, request, logical_key)
    if role is None:
        raise AgentError("typed_submission_explicit_protocol_required")
    if profile.get(PROFILE_KEY) == JSON2_VERSION:
        return recorded_json_submission(raw, private, http_status, role, adaptive=role == SUPPLEMENT_ROLE)
    if role == "review" and profile.get("review_api_protocol") == JSON_VERSION:
        return recorded_json_review(raw, private, http_status)
    return recorded_chat_response(raw, private, http_status, protocol=VERSION, function_name=ROLE_FUNCTIONS[role])
