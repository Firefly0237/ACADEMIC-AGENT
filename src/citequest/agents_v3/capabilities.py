"""Trusted effects of this runtime's fixed MCP capabilities, never model metadata."""
FREE_READ_EFFECT = "read_only_zero_cash/1"
FREE_READ_TOOLS = frozenset({"search_evidence", "read_evidence", "plan_workset", "read_source_window", "verify_evidence_batch", "search_chunk_evidence"})
LOCAL_COMPUTE_EFFECT = "read_only_local_compute_zero_external_cash/1"
LOCAL_COMPUTE_TOOLS = frozenset({"navigate_issued_evidence"})
# This distinct effect is deliberately not added to free_read_attempt: a lost
# local-compute receipt remains unknown under the existing conservative gate.


def free_read_attempt(attempt: dict) -> bool:
    request = attempt.get("request") or {}
    return (attempt.get("kind") == "tool" and attempt.get("reserved_ncny") == 0 and
        request.get("effect") == FREE_READ_EFFECT and request.get("name") in FREE_READ_TOOLS)
