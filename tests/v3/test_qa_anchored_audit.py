"""Three pure anchor controls using explicit synthetic scope/opinion doubles.

These check location and contract consistency, never scientific truth or product
success. No model, HTTP, filesystem evidence, database or runtime is exercised.
"""
import copy
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import digest
from citequest.agents_v3.qa_anchored_audit import AnchoredQuantitativeAnswerAudit
from citequest.agents_v3.qa_quantitative_audit import CRITERIA
from citequest.research_v3.qa_v5_contracts import bound_combined_review_model
from citequest.research_v3.quote_spans import Unit
from citequest.research_v3.review_grid import ReviewGrid

sys.path.insert(0, str(Path(__file__).parent))
from test_quantitative_reference_input import reference_fixture


def anchored_fixture():
    _old, opinions, candidate, original_grid, _base = reference_fixture()
    scope = original_grid.scope
    first = copy.deepcopy(scope.contexts[0]["quote_units"][0])
    second = {**first, "quote_id": "Q2"}
    scope.contexts[0]["quote_units"].append(second)
    unit = scope.units["Q1"]
    scope.units["Q2"] = Unit(unit.reference, unit.start, unit.end, unit.text)
    views = {"Q1": first, "Q2": second}
    scope.view = lambda selector: copy.deepcopy(views[selector])
    answer = copy.deepcopy(original_grid.candidate)
    answer["claims"][0]["statement"] = "测量🧪 " + answer["claims"][0]["statement"]
    grid = ReviewGrid(scope, answer, request_coverage=False)
    base = bound_combined_review_model(grid, digest(candidate.model_dump()), digest(candidate.user_turn.text))
    audit = AnchoredQuantitativeAnswerAudit(base, grid, candidate)
    opinions["answer_candidate_sha256"] = grid.binding["candidate_sha256"]
    opinions["anchor_binding_sha256"] = audit.binding["binding_sha256"]
    phrases = {"metric": "Spearman correlation", "unit": "7pp", "setting": "method A on dataset T"}
    opinions["criterion_evidence"] = {
        "C1": {criterion: {"body": [{"claim_id": "C1", "literal": phrase}],
                           "source": [{"quote_id": "Q1", "literal": phrase}]}
               for criterion, phrase in phrases.items()},
        "C2": {criterion: {"body": [], "source": []} for criterion in CRITERIA},
    }
    opinions["gap_evidence"] = {"G1": {"basis": "bounded_uncertainty", "source": [], "metadata": []}}
    return audit, opinions, candidate, grid, base


def test_anchors_derive_unicode_locations_and_keep_original_opinions_and_schema_fields():
    audit, opinions, _candidate, grid, base = anchored_fixture()
    result = audit.assess(opinions)
    statement = grid.candidate["claims"][0]["statement"]
    metric = result["criterion_evidence_receipts"]["C1"]["metric"]
    body = metric["body"][0]
    assert statement[body["span_start"]:body["span_end"]] == "Spearman correlation"
    assert body["span_start"] == statement.index("Spearman correlation")
    assert body["coordinate_system"] == "unicode_codepoints_in_claim_statement"
    source = metric["source"][0]
    unit = grid.scope.units["Q1"]
    assert unit.start + unit.text.index("Spearman correlation") == source["quote_locator"]["span_start"]
    assert result["quantitative_review"] == opinions["quantitative_review"]
    assert result["gap_review"] == opinions["gap_review"]
    assert result["requirement_coverage"] == opinions["requirement_coverage"]
    assert set(base.model_fields) <= set(audit.model.model_fields)
    assert {"quantitative_review", "criterion_evidence", "gap_evidence"} <= set(audit.model.model_fields)
    assert "claim_statements" not in audit.anchor_context
    assert "Q2" in audit.projection["untrusted_quantitative_input"]["quote_sources"]

    # A legitimate shared body anchor is structural evidence only. The semantic
    # grader still decides whether sharing is justified, without forced acceptance.
    shared = copy.deepcopy(opinions)
    shared["criterion_evidence"]["C1"]["setting"]["body"] = [
        {"claim_id": "C2", "literal": "Synthetic qualitative scope statement."}]
    assert audit.assess(shared)["quantitative_review"] == opinions["quantitative_review"]


def test_body_source_and_identity_fabrication_are_local_validation_errors():
    audit, opinions, _candidate, _grid, _base = anchored_fixture()
    for fault in ("invented_body", "normalized_body", "unselected_source", "invented_source",
                  "missing_positive_body", "wrong_body_id", "duplicate_anchor", "na_with_anchor", "wrong_binding"):
        wrong = copy.deepcopy(opinions)
        row = wrong["criterion_evidence"]["C1"]["metric"]
        if fault == "invented_body":
            row["body"][0]["literal"] = "Twitter"
        elif fault == "normalized_body":
            row["body"][0]["literal"] = "spearman correlation"
        elif fault == "unselected_source":
            row["source"][0]["quote_id"] = "Q2"
        elif fault == "invented_source":
            row["source"][0]["literal"] = "A measured accuracy of 99 percent."
        elif fault == "missing_positive_body":
            row["body"] = []
        elif fault == "wrong_body_id":
            row["body"][0]["claim_id"] = "C99"
        elif fault == "duplicate_anchor":
            row["body"].append(copy.deepcopy(row["body"][0]))
        elif fault == "na_with_anchor":
            wrong["criterion_evidence"]["C2"]["metric"]["body"] = copy.deepcopy(row["body"])
        else:
            wrong["anchor_binding_sha256"] = "0" * 64
        with pytest.raises(ValidationError) as captured:
            audit.model.model_validate(wrong, strict=True)
        error = captured.value.errors(include_input=False, include_url=False, include_context=False)[0]
        expected = {
            "invented_body": "qa_anchor_C1_metric_body_invalid",
            "normalized_body": "qa_anchor_C1_metric_body_invalid",
            "unselected_source": "qa_anchor_C1_metric_source_invalid",
            "invented_source": "qa_anchor_C1_metric_source_invalid",
            "missing_positive_body": "qa_anchor_C1_metric_body_required",
            "duplicate_anchor": "qa_anchor_C1_metric_body_duplicate",
            "na_with_anchor": "qa_anchor_C2_metric_not_applicable_evidence",
        }
        if fault in expected:
            assert error["type"] == expected[fault]
        elif fault == "wrong_body_id":
            assert error["loc"] == ("criterion_evidence", "C1", "metric", "body", 0, "claim_id")
        else:
            assert error["loc"] == ("anchor_binding_sha256",)

    # Repeated text cannot be silently assigned to the first occurrence.
    audit.body["C1"] += " Spearman correlation"
    with pytest.raises(ValidationError) as captured:
        audit.model.model_validate(opinions, strict=True)
    assert captured.value.errors(include_input=False)[0]["type"] == "qa_anchor_C1_metric_body_invalid"
    longer = copy.deepcopy(opinions)
    longer["criterion_evidence"]["C1"]["metric"]["body"][0]["literal"] = audit.body["C1"]
    assert audit.model.model_validate(longer, strict=True)


def test_negative_criteria_and_bounded_gap_bases_preserve_existing_publication_gates():
    audit, opinions, _candidate, _grid, _base = anchored_fixture()
    negative = copy.deepcopy(opinions)
    negative["quantitative_review"]["C1"]["metric"]["status"] = "missing"
    negative["criterion_evidence"]["C1"]["metric"] = {"body": [], "source": []}
    negative["gap_review"]["G1"]["verdict"] = "unsubstantiated"
    negative["gap_evidence"]["G1"]["basis"] = "not_supported"
    negative["requirement_coverage"]["R2"].update(status="bounded_evidence_gap", locations=["G1"])
    negative.update(request_coverage="partial", uncovered_requests=["R2"])
    result = audit.assess(negative)
    assert result["quantitative_rejected_claim_ids"] == ["C1"]
    assert result["gaps_rejected"] == ["G1"]
    assert result["wire_uncovered_requirement_ids"] == ["R2"]
    assert result["request_coverage"] == "partial"
    bounded = audit.assess(opinions)["gap_evidence_receipts"]["G1"]
    assert bounded["source"] == []
    assert bounded["scope_binding_sha256"] == audit.grid.scope.binding["binding_sha256"]

    metadata = copy.deepcopy(opinions)
    pointer = audit.anchor_context["metadata_pointers"][0]
    metadata["gap_evidence"]["G1"] = {"basis": "source_metadata", "source": [], "metadata": [pointer]}
    assert audit.assess(metadata)["gap_evidence_receipts"]["G1"]["metadata"][0]["pointer"] == pointer
    for basis in ("source_statement", "source_metadata", "not_supported"):
        wrong = copy.deepcopy(opinions)
        wrong["gap_evidence"]["G1"]["basis"] = basis
        with pytest.raises(ValidationError) as captured:
            audit.model.model_validate(wrong, strict=True)
        assert captured.value.errors(include_input=False)[0]["type"] == "qa_anchor_G1_gap_basis_invalid"
    metadata["gap_evidence"]["G1"]["metadata"] = ["/untrusted/private/path"]
    with pytest.raises(ValidationError):
        audit.model.model_validate(metadata, strict=True)
