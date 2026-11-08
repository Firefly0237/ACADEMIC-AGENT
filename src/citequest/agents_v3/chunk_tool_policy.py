"""Explicit trusted kind/profile pairs sharing one current-pinned chunk tool ABI."""
from __future__ import annotations

from .contracts import AgentError, canonical

TYPED_CHUNK_TASKS = frozenset({("qa", "qa-v5"), ("qa", "qa-v6"), ("qa", "qa-v7"), ("qa", "qa-v8"), ("qa", "qa-v9"), ("stance", "stance-v2"), ("stance", "stance-v3")})


def uses_typed_chunk_tools(task: dict) -> bool:
    return (task["kind"], task["profile"].get("profile_id")) in TYPED_CHUNK_TASKS


def deduplicate_issued_evidence(evidence: list[dict]) -> tuple[list[dict], dict]:
    """Stable union of current-authorized DTOs, retaining original issued refs.

    The caller MUST first verify every input against this Task's issuance and
    current source authority. Equality grants no permission. Only retrieval
    receipt ID and ref alias are excluded; all source/version/generation/ACL,
    literal coordinates, text, and future DTO fields remain part of equality.
    This helper neither changes issuance rows nor fuses retrieval scores.
    """
    if (type(evidence) is not list or len(evidence) > 20
            or any(not isinstance(item, dict) or not isinstance(item.get("ref_id"), str)
                or not item["ref_id"] or not isinstance(item.get("text"), str) for item in evidence)):
        raise AgentError("issued_evidence_union_invalid_or_over_capacity")
    first, references, output, aliases = {}, {}, [], []
    for item in evidence:
        ref = item["ref_id"]
        complete = canonical(item)
        if ref in references and references[ref] != complete:
            raise AgentError("issued_evidence_reference_changed_during_union")
        references[ref] = complete
        identity = canonical({key: value for key, value in item.items() if key not in {"ref_id", "snapshot_id"}})
        if identity in first:
            aliases.append({"duplicate_ref_id": ref, "retained_ref_id": first[identity]})
        else:
            first[identity] = ref
            output.append(item)
    return output, {"version": "citequest-current-issued-evidence-union/1",
        "input_ref_ids": [item["ref_id"] for item in evidence],
        "retained_ref_ids": [item["ref_id"] for item in output], "duplicate_aliases": aliases,
        "input_characters": sum(len(item["text"]) for item in evidence),
        "retained_characters": sum(len(item["text"]) for item in output),
        "count_basis": "Unicode_codepoints_not_model_tokens",
        "identity_excludes_only": ["ref_id", "snapshot_id"], "permission_grant": False,
        "source_bytes_modified": False, "original_query_receipts_and_issued_rows_retained": True}
