"""Deterministic failed-invariant feedback for the versioned source-review loop."""
from __future__ import annotations

from citequest.research_v3.typed_feedback import claim_objects

from .contracts import AgentError, digest

VERSION = "citequest-source-review-feedback/1"
EXPECTED_VERDICTS = {
    "supported": "entailed",
    "contradicted": "contradicted",
    "mixed": "mixed",
    "insufficient": "insufficient",
}
REASSESS_RULE = (
    "Reassess the exact written proposition, its status and selected source relations together. "
    "Evidence supporting a bounded statement about missing information differs from evidence "
    "for the requested unknown value. Verify against the original issued sources and preserve "
    "their scope. Do not copy an expected verdict, change a label merely to pass, erase a real "
    "gap or force the independent reviewer to agree. Preserve every locked claim and all required IDs."
)


def source_review_feedback(candidate: dict, canonical_review: dict, raw_grid: dict) -> dict | None:
    """Expose actual failed predicates without changing any candidate or opinion.

    Inputs have already passed the strict answer and fixed C/Q review schemas.
    This projection complements the original complete independent review. It
    supplies diagnostic facts, not a replacement semantic verdict or repair.
    """
    claims = claim_objects(candidate)
    rows = {row["claim_id"]: row for row in canonical_review["claims"]}
    if (len(rows) != len(canonical_review["claims"]) or set(rows) != set(claims)
            or set(raw_grid["claims"]) != set(claims)):
        raise AgentError("qa_source_feedback_review_identity_changed")
    issues = []
    for identifier, claim in claims.items():
        row = rows[identifier]
        raw = raw_grid["claims"][identifier]
        expected = EXPECTED_VERDICTS[claim["status"]]
        failed_quotes = [
            {"quote_id": quote_id, **opinion}
            for quote_id, opinion in sorted(raw["quotes"].items())
            if opinion["relation"] != "matches_declared_relations" or not opinion["conditions_preserved"]
        ]
        failures = []
        if row["verdict"] != expected:
            failures.append("claim_status_verdict_mismatch")
        if any(opinion["relation"] != "matches_declared_relations" for opinion in raw["quotes"].values()):
            failures.append("selected_quote_relation_failed")
        if not row["numeric_and_conditions_preserved"]:
            failures.append("claim_or_quote_conditions_not_preserved")
        if failures:
            issues.append({
                "claim_id": identifier,
                "failed_invariants": failures,
                "declared_status": claim["status"],
                "expected_verdict_for_declared_status": expected,
                "reviewer_verdict": raw["verdict"],
                "effective_verdict_after_quote_checks": row["verdict"],
                "claim_conditions_preserved": raw["conditions_preserved"],
                "effective_conditions_preserved": row["numeric_and_conditions_preserved"],
                "failed_selected_quotes": failed_quotes,
                "independent_explanation": raw["explanation"],
            })
    if not issues:
        return None
    return {
        "protocol": VERSION,
        "reviewed_answer_sha256": digest(candidate),
        "canonical_review_sha256": digest(canonical_review),
        "raw_review_grid_sha256": digest(raw_grid),
        "affected_claim_ids": [issue["claim_id"] for issue in issues],
        "issues": issues,
        "rule": REASSESS_RULE,
    }
