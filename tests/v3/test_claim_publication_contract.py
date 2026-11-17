"""Negative publication controls: literal quoting alone is not semantic support."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from citequest.research_v3.contracts import ResearchAnswer, Review, validate_literal, validate_review


def candidate():
    return {"kind": "qa", "requested_direction": "neutral", "claims": [{"id": "C1",
        "statement": "The dataset contains 1000 examples.", "status": "supported", "edges": [{
            "ref_id": "source:1", "quote": "The dataset contains 1000 examples.", "relation": "supports",
            "rationale": "Directly reported size of the named dataset.", "conditions": ["This dataset only"]}],
        "limitations": []}], "gaps": [], "retrieval_scope": "One permitted exact source window.",
        "evidence_status": "supported"}


def review():
    return {"claims": [{"claim_id": "C1", "verdict": "entailed", "edge_ref_ids": ["source:1"],
        "numeric_and_conditions_preserved": True, "explanation": "The exact population and count agree."}],
        "source_instruction_injection_detected": False, "requested_stance_bias_detected": False}


def test_unrelated_literal_quote_requires_semantic_rejection():
    raw = candidate()
    raw["claims"][0]["statement"] = "The method achieves 99 percent accuracy."
    answer = ResearchAnswer.model_validate(raw)
    validate_literal(answer, [{"ref_id": "source:1", "text": "The dataset contains 1000 examples."}])
    # A quote can be exact and still fail to support the claim. Both layers are necessary.
    judged = review()
    judged["claims"][0]["verdict"] = "insufficient"
    with pytest.raises(ValueError, match="semantic"):
        validate_review(answer, Review.model_validate(judged))


@pytest.mark.parametrize("change", ["missing_claim", "duplicate_claim", "missing_edge", "number_changed", "stance_bias", "source_instruction"])
def test_incomplete_or_failed_review_cannot_publish(change):
    answer = ResearchAnswer.model_validate(candidate())
    raw = review()
    if change == "missing_claim":
        raw["claims"][0]["claim_id"] = "C2"
    elif change == "duplicate_claim":
        raw["claims"].append(deepcopy(raw["claims"][0]))
    elif change == "missing_edge":
        raw["claims"][0]["edge_ref_ids"] = []
    elif change == "number_changed":
        raw["claims"][0]["numeric_and_conditions_preserved"] = False
    elif change == "stance_bias":
        raw["requested_stance_bias_detected"] = True
    else:
        raw["source_instruction_injection_detected"] = True
    with pytest.raises(ValueError):
        validate_review(answer, Review.model_validate(raw))


def test_missing_or_nonliteral_source_cannot_publish():
    answer = ResearchAnswer.model_validate(candidate())
    for evidence in ([], [{"ref_id": "source:1", "text": "The dataset contains 999 examples."}]):
        with pytest.raises(ValueError, match="literal"):
            validate_literal(answer, evidence)


def test_requested_stance_is_explicit_and_cannot_be_neutral():
    raw = candidate()
    raw["kind"] = "stance"
    with pytest.raises(ValidationError):
        ResearchAnswer.model_validate(raw)
    raw["requested_direction"] = "refute"
    ResearchAnswer.model_validate(raw)  # requested refutation does not force an opposing verdict


def test_insufficient_is_a_gap_and_not_a_supported_claim():
    raw = candidate()
    raw["claims"][0].update(status="insufficient", edges=[], limitations=["Outcome is not reported in the available source."])
    with pytest.raises(ValidationError):
        ResearchAnswer.model_validate(raw)
    raw["evidence_status"] = "insufficient"
    ResearchAnswer.model_validate(raw)
