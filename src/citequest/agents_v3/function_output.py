"""Explicit Responses function1 transport for the complete review answer.

This is typed answer submission, not an executed MCP tool. No provider call,
parser repair, retry, model/endpoint switch or vendor strict guarantee lives here.
Callers keep the original strict Pydantic and claim/requirement acceptance gates.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from .contracts import AgentError, digest

VERSION = "function1"
FUNCTION_NAME = "submit_review"
PROFILE_BINDING_KEY = "review_transport_protocol"


def validate_profile_transport(profile: dict, request: dict) -> None:
    """Validate immutable Task binding without late mutation of hashed bytes."""
    selected = profile.get(PROFILE_BINDING_KEY)
    if selected == "json1":
        if profile.get("profile_id") in {"qa-v8", "qa-v9"}:
            from .product_profiles import require_qa8_protocol
            require_qa8_protocol({"kind": "qa", "profile": profile})
            fmt = request.get("text", {}).get("format", {})
            if (request.get("tools") or request.get("tool_choice") is not None
                    or fmt.get("type") != "json_schema" or fmt.get("strict") is not True
                    or fmt.get("name") not in {"citequest_qa5_plan", "citequest_qa5_author", "citequest_qa5_review"}):
                raise AgentError("json2_explicit_text_submission_contract_invalid")
            return
        fmt = request.get("text", {}).get("format", {})
        if (profile.get("profile_id") != "qa-v7" or profile.get("qa_execution_protocol") != "grounded3"
                or profile.get("review_api_protocol") != "chat-json1"
                or profile.get("structured_output_protocol") != "chat-functions1"
                or profile.get("answer_revision_protocol") != "jsonpatch1"
                or request.get("tools") or request.get("tool_choice") is not None
                or fmt.get("type") != "json_schema" or fmt.get("strict") is not True
                or fmt.get("name") not in {"citequest_qa5_plan", "citequest_qa5_author", "citequest_qa5_review"}):
            raise AgentError("json1_explicit_text_submission_contract_invalid")
        return
    if selected is not None and (type(selected) is not str or selected not in {"flat1",VERSION}):
        raise AgentError("review_transport_profile_not_supported")
    tools = request.get("tools") or []
    if not isinstance(tools,list):
        raise AgentError("function1_wire_tool_list_required")
    uses_submission = any(isinstance(tool,dict) and tool.get("name") == FUNCTION_NAME for tool in tools)
    forced = request.get("tool_choice") == {"type":"function","name":FUNCTION_NAME}
    text = request.get("text")
    fmt = text.get("format") if isinstance(text,dict) else None
    schema_review = isinstance(fmt,dict) and fmt.get("name") == "citequest_qa5_review"
    if uses_submission or forced:
        if selected != VERSION:
            raise AgentError("function1_requires_immutable_Task_profile")
        if (not isinstance(tools,list) or len(tools) != 1 or tools[0].get("type") != "function"
                or tools[0].get("name") != FUNCTION_NAME or not forced or "text" in request
                or any(key in request for key in ("parallel_tool_calls","max_tool_calls"))
                or not isinstance(tools[0].get("parameters"),dict) or tools[0]["parameters"].get("type") != "object"):
            raise AgentError("function1_wire_submission_contract_invalid")
    elif selected == VERSION and schema_review:
        raise AgentError("function1_wire_must_be_compiled_before_hash")


def function_review_request(request: dict) -> dict:
    """Move the already SDK-compiled identical schema into function parameters.

    Caller must redo full-wire context admission and request hashing afterwards.
    Responses ignores parallel_tool_calls/max_tool_calls and documents no strict
    tool guarantee; do not advertise either as a server-side single-call gate.
    """
    value = copy.deepcopy(request)
    fmt = value.get("text",{}).get("format",{})
    if (fmt.get("type") != "json_schema" or fmt.get("name") != "citequest_qa5_review"
            or not isinstance(fmt.get("schema"),dict) or fmt["schema"].get("type") != "object"
            or any(key in value for key in ("tools","tool_choice","parallel_tool_calls","max_tool_calls"))):
        raise AgentError("function1_complete_review_schema_required")
    schema = fmt["schema"]
    del value["text"]
    value["tools"] = [{"type":"function", "name":FUNCTION_NAME,
        "description":"Submit the complete independent requirement and evidence review using the exact advertised schema.",
        "parameters":schema}]
    value["tool_choice"] = {"type":"function","name":FUNCTION_NAME}
    value["instructions"] += (
        "\nOutput transport: submit exactly one submit_review function call. Its arguments are the complete "
        "advertised review object. Do not emit a prose message, omit cells, or change a verdict merely to fit the transport.")
    return value


@dataclass(frozen=True)
class ReviewSubmission:
    arguments: str
    call_id: str

    def public_transport_metadata(self) -> dict:
        return {"transport":VERSION,"function_name":FUNCTION_NAME,"vendor_call_id_sha256":digest(self.call_id),
                "arguments_sha256":digest(self.arguments),"external_MCP_execution":False}


def review_submission(response: dict) -> ReviewSubmission:
    """Validate the envelope; return raw arguments for ORIGINAL strict validation.

    Overall completed plus one permitted call is mandatory. Item.status is
    optional in the official SDK; an explicitly incomplete item is rejected.
    Never pick one of multiple calls or concatenate/repair model arguments.
    Envelope acceptance alone is not JSON, schema, semantic or academic success.
    """
    if (not isinstance(response,dict) or response.get("status") != "completed"
            or response.get("incomplete_details") is not None or response.get("error") is not None):
        raise AgentError("function1_response_not_completed")
    output = response.get("output")
    if not isinstance(output,list) or len(output) != 1:
        raise AgentError("function1_exactly_one_submission_required")
    item = output[0]
    if (not isinstance(item,dict) or item.get("type") != "function_call"
            or item.get("name") != FUNCTION_NAME or item.get("status") not in {None,"completed"}
            or type(item.get("call_id")) is not str or not 1 <= len(item["call_id"]) <= 300 or not item["call_id"].strip()
            or type(item.get("arguments")) is not str or not item["arguments"].strip()):
        raise AgentError("function1_submission_envelope_invalid")
    return ReviewSubmission(item["arguments"],item["call_id"])
