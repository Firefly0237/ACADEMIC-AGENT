"""Versioned, data-minimizing local validation feedback and directed repair locks.

This module owns no model client, execution budget or retry loop. Callers may
use it only after the existing Execution has settled a completed known-usage
response. Transport, permission and source-version exceptions are not accepted
as validation feedback. Old output contracts and their errors stay unchanged.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from citequest.agents_v3.contracts import AgentError, digest

VERSION = "citequest-typed-feedback/1"

# These are server-authored fixed explanations, never error.msg/input/ctx. In
# particular a PydanticCustomError's context is not trusted model feedback.
RULES = {
    "qa_presentation_claim_partition": "Repair only presentation.paragraphs so every current claim ID appears exactly once in nonempty groups. Preserve the facts, claim IDs and source edges. Do not add prose, claims or inferred support.",
    "qa_presentation_inline_statement": "Repair only prose-mode paragraph formatting. Keep each statement on one physical line with inline math, preserving facts, quantities, claim IDs and source edges. Use blocks only when the user's actual requested mathematics/code requires that format.",
    "claim_supported_requires_supports": "A supported claim must have at least one genuinely supporting source edge. context_only is background, not support. Reassess the proposition and evidence without relabeling an unrelated quote as support.",
    "claim_contradicted_requires_contradicts": "A contradicted claim must have at least one genuinely contradicting source edge. Background does not establish contradiction.",
    "claim_mixed_requires_both_relations": "A mixed claim needs both a supports and a contradicts edge for this same proposition. Preserve the actual applicable conditions.",
    "claim_insufficient_requires_gap": "An insufficient claim must name its actual evidence gap in limitations; do not invent missing evidence.",
    "summary_supported_requires_supported_claim": "Supported evidence_status requires at least one claim whose status is not insufficient. Do not invent a claim to satisfy this rule.",
    "literal_error": "Use exactly one advertised enum value for this field. claim.status is supported, contradicted, mixed or insufficient; edge.relation is supports, contradicts or context_only. They are different fields.",
    "missing": "Supply this required schema field with a value meeting its declared type. Preserve every required claim ID.",
    "extra_forbidden": "Remove the undeclared property; do not alter any unrelated claim or add factual prose outside the schema.",
    "string_too_long": "Make this field concise within its advertised bound while preserving its proposition, negation, quantities, conditions and limitations.",
    "string_too_short": "Supply the required nonempty bounded text, without inventing factual support.",
    "string_pattern_mismatch": "Use the exact server-issued identity syntax. Do not invent or substitute an identity.",
    "too_long": "Respect this field's advertised item limit without deleting required claims or opposing findings. If that is impossible, retain an explicit gap.",
    "too_short": "Supply the advertised minimum number of items without inventing evidence or deleting required claims.",
    "list_type": "This field must be a JSON array meeting the advertised element schema.",
    "string_type": "This field must be a JSON string, not a coerced number, object or array.",
    "bool_type": "This field must be a JSON boolean.",
    "model_type": "This field must be a JSON object with exactly its advertised schema.",
    "model_attributes_type": "This field must be a JSON object with exactly its advertised schema.",
}
_FIELDS = frozenset({"claims", "id", "statement", "status", "edges", "quote_id", "relation", "rationale", "presentation", "paragraphs", "format",
    "conditions", "limitations", "gaps", "evidence_status", "kind", "requested_direction", "retrieval_scope"})
_ID = re.compile(r"C[1-9][0-9]?")


def strict_object(raw: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AgentError("typed_feedback_duplicate_json_property")
            result[key] = value
        return result

    def nonfinite(_):
        raise AgentError("typed_feedback_nonfinite_json")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise AgentError("typed_feedback_initial_json_unreadable") from error
    if not isinstance(value, dict):
        raise AgentError("typed_feedback_initial_object_required")
    return value


def claim_objects(candidate: dict, *, maximum: int = 8) -> dict[str, dict]:
    claims = candidate.get("claims")
    if (not isinstance(claims, list) or not 1 <= len(claims) <= maximum
            or any(not isinstance(claim, dict) or not isinstance(claim.get("id"), str)
                or _ID.fullmatch(claim["id"]) is None for claim in claims)
            or len({claim["id"] for claim in claims}) != len(claims)):
        raise AgentError("typed_feedback_claim_identity_ambiguous")
    return {claim["id"]: claim for claim in claims}


def validation_feedback(error: ValidationError, candidate: dict, *, maximum: int = 8) -> dict:
    """Bind local error locations to this response's stable server-checked IDs.

    Unknown validation codes are fail-closed for repair; no exception text or
    model-supplied ctx is echoed. The caller retains the actual failed response
    in its private attempt ledger rather than copying it into public events.
    """
    if not isinstance(error, ValidationError):
        raise TypeError("only local Pydantic ValidationError is accepted")
    claims = claim_objects(candidate, maximum=maximum)
    identifiers = list(claims)
    issues, affected, unknown = [], set(), False
    for row in error.errors(include_input=False, include_url=False, include_context=False):
        code = row["type"]
        advice = RULES.get(code)
        unknown = unknown or advice is None
        original = row["loc"]
        loc = [part if type(part) is int and 0 <= part <= 1000 else part if isinstance(part, str) and part in _FIELDS else "unrecognized_field" for part in original]
        identifier = None
        if len(original) >= 2 and original[0] == "claims" and type(original[1]) is int and 0 <= original[1] < len(identifiers):
            identifier = identifiers[original[1]]
            affected.add(identifier)
        else:
            affected.update(identifiers)
        issues.append({"code": code if advice is not None else "unregistered_validation_rule", "location": loc,
            "claim_id": identifier, "rule": advice or "An unregistered local validation rule failed; automatic repair is not authorized."})
    return {"schema_version": VERSION, "repairable": bool(issues) and not unknown,
        "error_codes": list(dict.fromkeys(issue["code"] for issue in issues))[:8],
        "required_claim_ids": identifiers, "affected_claim_ids": sorted(affected), "issues": issues,
        "instruction": "Correct only affected claims using current original evidence. Preserve all required IDs and every unaffected claim object exactly. Never map a label, invent support, or delete a claim merely to satisfy a validator."}


@dataclass(frozen=True)
class RepairLock:
    """Immutable baseline identity and object bindings for one directed repair."""
    baseline_json: str
    required_ids: tuple[str, ...]
    affected_ids: tuple[str, ...]
    maximum: int = 8

    @classmethod
    def bind(cls, candidate: dict, affected: list[str], *, maximum: int = 8) -> RepairLock:
        objects = claim_objects(candidate, maximum=maximum)
        if (len(affected) != len(set(affected)) or not affected or not set(affected) <= objects.keys()):
            raise AgentError("typed_feedback_affected_identity_invalid")
        return cls(json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False),
            tuple(objects), tuple(sorted(affected)), maximum)

    def verify(self, candidate: dict) -> None:
        current = claim_objects(candidate, maximum=self.maximum)
        baseline = claim_objects(json.loads(self.baseline_json), maximum=self.maximum)
        if set(current) != set(self.required_ids):
            raise AgentError("typed_feedback_required_claim_set_changed")
        if any(current[identifier] != baseline[identifier] for identifier in self.required_ids if identifier not in self.affected_ids):
            raise AgentError("typed_feedback_unaffected_claim_changed")

    def contract(self) -> dict[str, Any]:
        baseline = claim_objects(json.loads(self.baseline_json), maximum=self.maximum)
        return {"schema_version": VERSION, "required_claim_ids": list(self.required_ids),
            "affected_claim_ids": list(self.affected_ids), "unaffected_claim_sha256": {
                identifier: digest(claim) for identifier, claim in baseline.items() if identifier not in self.affected_ids},
            "baseline_sha256": digest(json.loads(self.baseline_json))}
