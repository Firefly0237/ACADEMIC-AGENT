"""Content-free links to completed model attempts, never synthetic executions."""
from __future__ import annotations

from .contracts import AgentError, digest


def model_receipt(row: dict, logical_key: str, profile: dict) -> dict:
    if (row.get("status") != "settled" or row.get("error_code") is not None
            or not isinstance(row.get("request"), dict) or not isinstance(row.get("response"), dict)):
        raise AgentError("handoff_model_receipt_unavailable")
    usage = row.get("usage")
    if (not isinstance(usage, dict)
            or any(type(usage.get(key)) is not int or usage[key] < 0 for key in ("input", "cached", "output"))
            or usage["cached"] > usage["input"]):
        raise AgentError("handoff_model_usage_unknown")
    request = row["request"]
    if digest(request) != row.get("request_hash"):
        raise AgentError("handoff_model_request_hash_changed")
    completed = row["response"].get("status") == "completed"
    if profile.get("review_api_protocol") is not None:
        from .strict_chat import is_strict_review
        from .typed_submission import normalize_submission, submission_role
        role = submission_role(profile, request, logical_key)
        if role is not None or is_strict_review(profile, request):
            normalize_submission(profile, request, logical_key, row["response"])
            completed = True
    if not completed:
        raise AgentError("handoff_model_response_incomplete")
    text = request.get("text")
    fmt = text.get("format") if isinstance(text, dict) else None
    name = fmt.get("name") if isinstance(fmt, dict) else None
    if not isinstance(name, str) or not 1 <= len(name) <= 128:
        raise AgentError("handoff_model_schema_identity_missing")
    return {"attempt_id": str(row["id"]), "logical_key": logical_key,
            "request_sha256": row["request_hash"], "response_sha256": digest(row["response"]),
            "schema_name": name, "usage": {key: usage[key] for key in ("input", "cached", "output")}}
