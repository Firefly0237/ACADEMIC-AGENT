"""Two pure controls over explicit synthetic grader opinions and source doubles.

They exercise aggregate, literal and catalog contracts, not scientific truth,
real model performance, database behavior or whole-product acceptance.
"""
import copy
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.qa_decomposed_audit import DecomposedAnchoredAnswerAudit, SETTING_FACETS
from citequest.agents_v3.qa_gap_catalog import GapCatalog
from citequest.research_v3.qa_v5_contracts import bound_combined_review_model
from citequest.research_v3.review_grid import ReviewGrid

sys.path.insert(0, str(Path(__file__).parent))
from test_qa_anchored_audit import anchored_fixture


def decomposed_fixture(*, catalog=None):
    _old, opinions, candidate, grid, base = anchored_fixture()
    audit = DecomposedAnchoredAnswerAudit(base, grid, candidate, gap_catalog=catalog)
    opinions["anchor_binding_sha256"] = audit.binding["binding_sha256"]
    unused = {"status": "not_applicable", "reason": "Inapplicable in this synthetic opinion.", "body": [], "source": []}
    opinions["setting_facets"] = {
        "C1": {facet: copy.deepcopy(unused) for facet in SETTING_FACETS},
        "C2": {facet: copy.deepcopy(unused) for facet in SETTING_FACETS},
    }
    opinions["setting_facets"]["C1"]["data_population"] = {
        "status": "missing", "reason": "An explicit synthetic negative population opinion.", "body": [], "source": []}
    opinions["setting_facets"]["C1"]["method_pipeline"] = {
        "status": "explicit", "reason": "A separately located synthetic method opinion.",
        "body": [{"claim_id": "C1", "literal": "method A"}],
        "source": [{"quote_id": "Q1", "literal": "method A"}]}
    opinions["quantitative_review"]["C1"]["setting"]["status"] = "missing"
    return audit, opinions, candidate, grid, base


def test_missing_population_cannot_be_replaced_by_a_positive_method_opinion():
    audit, opinions, _candidate, _grid, base = decomposed_fixture()
    result = audit.assess(opinions)
    assert result["quantitative_rejected_claim_ids"] == ["C1"]
    assert result["setting_facets"]["C1"]["data_population"]["status"] == "missing"
    method = result["setting_facet_receipts"]["C1"]["method_pipeline"]
    assert method["body"][0]["claim_id"] == "C1"
    assert method["source"][0]["quote_id"] == "Q1"
    assert result["gap_review"] == opinions["gap_review"]
    assert set(base.model_fields) <= set(audit.model.model_fields)
    assert {"criterion_evidence", "gap_evidence", "quantitative_review", "setting_facets"} <= set(audit.model.model_fields)
    for fault, expected in (
        ("aggregate", "qa_anchor_C1_setting_aggregate_invalid"),
        ("unselected_source", "qa_anchor_C1_method_pipeline_source_invalid"),
        ("invented_body", "qa_anchor_C1_method_pipeline_body_invalid"),
        ("blank_reason", "qa_anchor_C1_method_pipeline_reason_invalid"),
        ("all_inapplicable", "qa_anchor_C1_setting_applicability_invalid"),
        ("other_applicable", "qa_anchor_C2_setting_other_facets_invalid"),
    ):
        wrong = copy.deepcopy(opinions)
        row = wrong["setting_facets"]["C1"]["method_pipeline"]
        if fault == "aggregate":
            wrong["quantitative_review"]["C1"]["setting"]["status"] = "explicit"
        elif fault == "unselected_source":
            row["source"][0]["quote_id"] = "Q2"
        elif fault == "invented_body":
            row["body"][0]["literal"] = "Twitter"
        elif fault == "blank_reason":
            row["reason"] = " "
        elif fault == "all_inapplicable":
            for facet in SETTING_FACETS:
                wrong["setting_facets"]["C1"][facet].update(status="not_applicable", body=[], source=[])
        else:
            wrong["setting_facets"]["C2"]["method_pipeline"]["status"] = "missing"
        with pytest.raises(ValidationError) as captured:
            audit.model.model_validate(wrong, strict=True)
        assert captured.value.errors(include_input=False, include_context=False)[0]["type"] == expected


def test_catalog_admissibility_is_bound_and_separate_from_actual_gap_verdicts():
    _old, _opinions, candidate, _grid, _base = anchored_fixture()
    catalog = GapCatalog(candidate)
    audit, opinions, _candidate, grid, _base = decomposed_fixture(catalog=catalog)
    result = audit.assess(opinions)
    assert audit.binding["gap_catalog_binding_sha256"] == catalog.binding["binding_sha256"]
    assert audit.projection["gap_catalog"] == catalog.projection
    assert result["gap_review"]["G1"]["verdict"] == "faithful"
    assert result["gap_contract_rejected"] == ["G1"]
    assert result["bad_gap_ids"] == result["gaps_rejected"] == ["G1"]

    # An admissible gap still retains a genuine negative semantic opinion.
    answer = copy.deepcopy(grid.candidate)
    answer["gaps"] = [catalog.projection["typed_options"][0]["text"]]
    current_grid = ReviewGrid(grid.scope, answer, request_coverage=False)
    base = bound_combined_review_model(current_grid, digest(candidate.model_dump()), digest(candidate.user_turn.text))
    current = DecomposedAnchoredAnswerAudit(base, current_grid, candidate, gap_catalog=catalog)
    negative = copy.deepcopy(opinions)
    negative.update(answer_candidate_sha256=current_grid.binding["candidate_sha256"],
        anchor_binding_sha256=current.binding["binding_sha256"])
    negative["gap_review"]["G1"]["verdict"] = "unsubstantiated"
    negative["gap_evidence"]["G1"]["basis"] = "not_supported"
    checked = current.assess(negative)
    assert checked["gap_contract_rejected"] == []
    assert checked["gap_review"]["G1"]["verdict"] == "unsubstantiated"
    assert checked["gaps_rejected"] == ["G1"]
    assert checked["quantitative_rejected_claim_ids"] == ["C1"]

    changed = copy.deepcopy(catalog)
    changed.binding["requirements_candidate_sha256"] = "0" * 64
    with pytest.raises(AgentError, match="qa_decomposed_gap_catalog_candidate_mismatch"):
        DecomposedAnchoredAnswerAudit(base, current_grid, candidate, gap_catalog=changed)
