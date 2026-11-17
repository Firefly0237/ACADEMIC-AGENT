"""Validate a whole untrusted search batch before any MCP side effect.

The provider may return multiple calls even when a compatibility parameter asks
for one. The server controls sequential execution, capability and resource caps.
"""
from __future__ import annotations

import json

from jsonschema import Draft202012Validator, ValidationError

from .contracts import AgentError, validate_user_text
from .provider import output_items


def unique_object(pairs: list[tuple]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON argument key")
        value[key] = item
    return value


def admit_search_batch(response: dict, schema: dict, *, tools_remaining: int,
                       models_remaining: int) -> list[dict]:
    calls = [item for item in output_items(response) if item["type"] == "function_call"]
    if not 1 <= len(calls) <= 2:
        raise AgentError("search_batch_call_count_invalid")
    identifiers = [item.get("call_id") for item in calls]
    if (any(not isinstance(value, str) or not 1 <= len(value) <= 200 for value in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise AgentError("search_batch_call_id_invalid")
    validated = []
    for item in calls:
        if item["name"] != "search_evidence":
            raise AgentError("search_batch_tool_not_allowed")
        try:
            arguments = json.loads(item["arguments"], object_pairs_hook=unique_object)
            if not isinstance(arguments, dict) or set(arguments) - {"query", "limit"}:
                raise ValueError("unexpected tool argument")
            Draft202012Validator(schema).validate(arguments)
            validate_user_text(arguments["query"])
            limit = arguments.get("limit", 2)
            if type(limit) is not int or not 1 <= limit <= 3:
                raise ValueError("unexpected tool evidence limit")
        except (KeyError, TypeError, ValueError, ValidationError):
            raise AgentError("search_batch_arguments_invalid") from None
        validated.append({**item, "validated_arguments": arguments, "maximum_evidence": limit})
    # A turn also needs writer+independent reviewer and two current source reads
    # per distinct returned evidence (pre-write and pre-publication). Assume no
    # deduplication benefit when admitting a batch; never execute a cheap prefix.
    maximum_tools = len(calls) + 2 * sum(item["maximum_evidence"] for item in validated)
    if maximum_tools > tools_remaining or models_remaining < 2:
        raise AgentError("search_batch_downstream_budget_insufficient")
    return validated
