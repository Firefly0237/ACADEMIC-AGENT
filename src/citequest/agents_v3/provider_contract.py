"""Pure classification of an accounted strict-review wire contract violation.

This checks the exact compiled provider schema. It neither repairs arguments nor
classifies full business validators or semantic opinions as provider failures.
"""
from __future__ import annotations

import json
import math

from jsonschema import Draft202012Validator, FormatChecker

from .contracts import AgentError, digest
from .strict_chat import (
    VERSION,
    compile_review_request,
    normalize_chat_response,
    usage_counts_chat,
)
from .typed_submission import compile_submission, normalize_submission, submission_role
from .json_review import VERSION as JSON_VERSION, JSON2_VERSION, WIRE_SCHEMA as JSON_WIRE_SCHEMA


def validate_reply_model(request: dict, response: dict) -> None:
    """The new Pro contract requires an exact reply identity on live and replay.

    Historical Flash receipts retain their existing acceptance contract.
    """
    if request.get("model") == "deepseek-v4-pro" and response.get("model") != request["model"]:
        raise AgentError("provider_response_model_mismatch")


class _DuplicateProperty(ValueError):
    pass


class _NonfiniteNumber(ValueError):
    pass


def _unique_object(pairs: list[tuple]) -> dict:
    value = {}
    for key, child in pairs:
        if key in value:
            raise _DuplicateProperty
        value[key] = child
    return value


def _finite_number(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _NonfiniteNumber
    return parsed


def _reject_constant(_value: str) -> None:
    raise _NonfiniteNumber


def wire_contract_violation(request: dict, raw_private_chat: dict, *, protocol: str = VERSION,
                            profile: dict | None = None, logical_key: str | None = None) -> dict | None:
    """Return only safe deterministic evidence eligible for explicit recovery.

    Invalid/missing usage or a rejected receipt envelope is ineligible. Those
    conditions keep their original failure path and can never authorize a retry.
    The caller and durable reservation also require a known settled parent row.
    """
    role = submission_role(profile, request, logical_key) if profile is not None else None
    try:
        if role is None:
            normalize_chat_response(raw_private_chat, protocol=protocol)
        else:
            normalize_submission(profile, request, logical_key, raw_private_chat)
        usage_counts_chat(raw_private_chat)
    except AgentError:
        return None
    validate_reply_model(request, raw_private_chat)
    compiled = (compile_review_request(request, protocol=protocol) if role is None
        else compile_submission(profile, request, logical_key))
    json_mode = compiled.receipt["protocol"] in {JSON_VERSION, JSON2_VERSION}
    message = raw_private_chat["choices"][0]["message"]
    arguments = (message["content"] if json_mode else message["tool_calls"][0]["function"]["arguments"])
    try:
        argument_bytes = arguments.encode("utf-8")
    except UnicodeError:
        # Do not reinterpret an unencodable string or hash replacement bytes.
        return None
    category = None
    try:
        parsed = json.loads(arguments, object_pairs_hook=_unique_object,
            parse_constant=_reject_constant, parse_float=_finite_number)
    except _DuplicateProperty:
        category = "duplicate_property"
    except _NonfiniteNumber:
        category = "nonfinite_json"
    except json.JSONDecodeError:
        category = "invalid_json"
    except (RecursionError, ValueError) as error:
        raise AgentError("strict_review_wire_validation_resource_limit") from error
    else:
        schema = JSON_WIRE_SCHEMA if json_mode else compiled.wire["tools"][0]["function"]["parameters"]
        if not Draft202012Validator(schema, format_checker=FormatChecker()).is_valid(parsed):
            category = "wire_schema"
    if category is None:
        return None
    return {
        "code": "json_object_wire_contract_violation" if json_mode else "strict_review_wire_contract_violation",
        "category": category,
        "protocol": compiled.receipt["protocol"],
        "compiler": compiled.receipt["compiler"],
        "wire_schema_sha256": compiled.receipt["wire_schema_sha256"],
        "arguments_sha256": digest(argument_bytes),
        "arguments_utf8_bytes": len(argument_bytes),
    }
