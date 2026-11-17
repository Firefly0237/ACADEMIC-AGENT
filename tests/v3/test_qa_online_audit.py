"""Two pure online-contract controls using explicit synthetic reviewer opinions."""
import copy
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from citequest.agents_v3.qa_gap_catalog import GapCatalog
from citequest.agents_v3.qa_online_audit import OnlineAnswerAudit, SETTING_FACETS

sys.path.insert(0, str(Path(__file__).parent))
from test_quantitative_reference_input import reference_fixture


def online_fixture(*, catalog=False):
    original, opinions, candidate, grid, base = reference_fixture()
    audit = OnlineAnswerAudit(base, grid, candidate,
        gap_catalog=GapCatalog(candidate) if catalog else None)
    opinions["online_binding_sha256"] = audit.binding["binding_sha256"]
    inactive = {"status": "not_applicable", "reason": "Synthetic inapplicability opinion."}
    opinions["setting_facets"] = {
        identifier: {facet: copy.deepcopy(inactive) for facet in SETTING_FACETS}
        for identifier in audit.claim_ids}
    opinions["setting_facets"]["C1"]["method_pipeline"] = {
        "status": "explicit", "reason": "Synthetic method opinion."}
    return original, audit, opinions, base


def test_dimensionless_is_recorded_without_relabeling_and_na_has_actionable_feedback():
    original, audit, opinions, base = online_fixture()
    opinions["quantitative_review"]["C1"]["unit"] = {
        "status": "dimensionless", "reason": "Synthetic scale opinion, not a scientific truth assertion."}
    result = audit.assess(opinions)
    assert result["quantitative_review"]["C1"]["unit"]["status"] == "dimensionless"
    assert result["quantitative_rejected_claim_ids"] == []
    assert set(base.model_fields) <= set(audit.model.model_fields)
    assert not {"criterion_evidence", "gap_evidence", "anchor_binding_sha256"} & set(audit.model.model_fields)
    references = audit.projection["untrusted_quantitative_input"]
    assert references["expanded_input_sha256"] == original.binding["quantitative_input_sha256"]
    wrong = copy.deepcopy(opinions)
    wrong["quantitative_review"]["C1"]["unit"]["status"] = "not_applicable"
    with pytest.raises(ValidationError) as captured:
        audit.model.model_validate(wrong, strict=True)
    assert captured.value.errors(include_input=False, include_context=False)[0]["type"] == "qa_online_unit_requires_scale_judgment"
    wrong = copy.deepcopy(opinions)
    wrong["quantitative_review"]["C2"]["unit"]["status"] = "dimensionless"
    with pytest.raises(ValidationError) as captured:
        audit.model.model_validate(wrong, strict=True)
    assert captured.value.errors(include_input=False, include_context=False)[0]["type"] == "qa_online_other_requires_not_applicable"


def test_missing_facets_and_catalog_rejections_preserve_independent_negative_opinions():
    _original, audit, opinions, _base = online_fixture(catalog=True)
    opinions["setting_facets"]["C1"]["data_population"] = {
        "status": "missing", "reason": "Synthetic population omission."}
    opinions["quantitative_review"]["C1"]["setting"]["status"] = "missing"
    result = audit.assess(opinions)
    assert result["quantitative_rejected_claim_ids"] == ["C1"]
    assert result["gap_review"] == opinions["gap_review"]
    assert result["gap_review"]["G1"]["verdict"] == "faithful"
    assert result["gap_contract_rejected"] == result["gaps_rejected"] == ["G1"]
    wrong = copy.deepcopy(opinions)
    wrong["quantitative_review"]["C1"]["setting"]["status"] = "explicit"
    with pytest.raises(ValidationError) as captured:
        audit.model.model_validate(wrong, strict=True)
    assert captured.value.errors(include_input=False, include_context=False)[0]["type"] == "qa_online_C1_setting_aggregate_invalid"
