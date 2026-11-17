"""Pure controls with explicit scope, source and execution substitutes.

Synthetic criterion opinions exercise structure and routing. They establish no
numerical truth, source entailment or real-product acceptance.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import AgentError, canonical, digest
from citequest.agents_v3.qa_answer_audit import AnswerAudit
from citequest.agents_v3.qa_quantitative_audit import CRITERIA, QuantitativeAnswerAudit
from citequest.agents_v3.qa_verified_dialogue import QADialogueVerified
from citequest.research_v3.qa_v5_contracts import bound_combined_review_model
from citequest.research_v3.review_grid import ReviewGrid
from citequest.research_v3.typed_feedback import RepairLock

sys.path.insert(0, str(Path(__file__).parent))
from test_qa_answer_audit import SyntheticScope, fixture as requirement_fixture


class SyntheticEvidenceScope(SyntheticScope):
    """Expose exactly the original QuoteScope interfaces consumed by the audit."""

    def __init__(self):
        text = "Synthetic Spearman correlation rises by 7pp for method A on dataset T."
        self.units = {"Q1": SimpleNamespace(reference="synthetic-ref", start=100, end=100 + len(text), text=text)}
        self.registry = SimpleNamespace(sources={"synthetic-ref": {
            "source_ref": {"kind": "native_representation", "id": "f" * 64},
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "unit_text_sha256": "d" * 64, "span_start": 100, "span_end": 100 + len(text),
            "coverage": "native_literal", "coordinate_system": "unicode_codepoints_in_native_literal",
        }})

    def locator(self, unit):
        return {"span_start": unit.start, "span_end": unit.end,
                "quote_sha256": hashlib.sha256(unit.text.encode()).hexdigest()}


def audited_fixture():
    _, value, candidate, old_grid, _ = requirement_fixture()
    answer = copy.deepcopy(old_grid.candidate)
    answer["claims"][0]["statement"] = "Synthetic Spearman correlation improves by 7pp for method A on dataset T."
    answer["claims"].append({"id": "C2", "statement": "Synthetic qualitative scope statement.", "edges": [{"quote_id": "Q1"}]})
    answer["presentation"]["paragraphs"] = [["C1", "C2"]]
    grid = ReviewGrid(SyntheticEvidenceScope(), answer, request_coverage=False)
    base = bound_combined_review_model(grid, digest(candidate.model_dump()), digest(candidate.user_turn.text))
    audit = QuantitativeAnswerAudit(base, grid, candidate)
    value["answer_candidate_sha256"] = grid.binding["candidate_sha256"]
    value["claims"]["C2"] = copy.deepcopy(value["claims"]["C1"])
    value["quantitative_review"] = {
        "C1": {"kind": "quantitative_result", **{criterion: {"status": "explicit", "reason": "Explicit synthetic judgment."}
                                                 for criterion in CRITERIA}},
        "C2": {"kind": "other", **{criterion: {"status": "not_applicable", "reason": "Synthetic qualitative claim."}
                                   for criterion in CRITERIA}},
    }
    return audit, value, candidate, grid, base


def node(*, revisions=0, recoveries=0):
    value = object.__new__(QADialogueVerified)
    value.graph = SimpleNamespace(execution=SimpleNamespace(
        command={"id": "explicit-test-command"}, content_revision_count=lambda: revisions,
        protocol_recovery_count=lambda: recoveries, effective_model_source_key=lambda key: key,
    ))
    value.scope = lambda _state: SimpleNamespace(compact_candidate=copy.deepcopy)
    return value


def state_and_data(grid):
    return {"answer": copy.deepcopy(grid.candidate)}, {
        "revision": 0, "audit": [], "accepted": False, "terminal": False,
        "lock": None, "gap_only_lock": None, "feedback": None,
    }


def test_fixed_c_criteria_and_original_quote_input_preserve_closed_identity():
    audit, value, candidate, grid, base = audited_fixture()
    original_schema = base.model_json_schema()
    schema = audit.model.model_json_schema()

    def closed_objects(item):
        if isinstance(item, dict):
            if item.get("type") == "object":
                assert item["additionalProperties"] is False
                assert set(item.get("required", [])) == set(item.get("properties", {}))
            for child in item.values():
                closed_objects(child)
        elif isinstance(item, list):
            for child in item:
                closed_objects(child)

    closed_objects(schema)
    parsed = audit.model.model_validate(value, strict=True)
    assert set(type(parsed.quantitative_review).model_fields) == {"C1", "C2"}
    projected = audit.projection["untrusted_quantitative_input"]["claim_evidence"][0]["selected_quote_evidence"][0]
    assert projected["original_quote"] == grid.scope.units["Q1"].text
    assert projected["quote_locator"] == grid.scope.locator(grid.scope.units["Q1"])
    assert projected["source_window"]["span_start"] == 100
    assert audit.binding["answer_candidate_sha256"] == grid.binding["candidate_sha256"]
    assert audit.binding["requirements_candidate_sha256"] == digest(candidate.model_dump())
    assert base.model_json_schema() == original_schema
    assert "quantitative_review" not in AnswerAudit(base, grid, candidate).model.model_fields
    for fault in ("missing_C", "extra_C", "quantitative_NA", "other_explicit"):
        wrong = copy.deepcopy(value)
        if fault == "missing_C":
            wrong["quantitative_review"].pop("C1")
        elif fault == "extra_C":
            wrong["quantitative_review"]["C3"] = copy.deepcopy(wrong["quantitative_review"]["C2"])
        elif fault == "quantitative_NA":
            wrong["quantitative_review"]["C1"]["unit"]["status"] = "not_applicable"
        else:
            wrong["quantitative_review"]["C2"]["metric"]["status"] = "explicit"
        with pytest.raises(ValidationError):
            audit.model.model_validate(wrong, strict=True)


def test_each_missing_quantitative_criterion_preserves_negative_r_g_opinions():
    audit, value, _, _, _ = audited_fixture()
    value["gap_review"]["G1"]["verdict"] = "unsubstantiated"
    value["requirement_coverage"]["R2"].update(status="bounded_evidence_gap", locations=["G1"])
    value.update(request_coverage="partial", uncovered_requests=["R2"])
    for criterion in CRITERIA:
        negative = copy.deepcopy(value)
        negative["quantitative_review"]["C1"][criterion]["status"] = "missing"
        result = audit.assess(negative)
        assert result["quantitative_rejected_claim_ids"] == ["C1"]
        assert result["quantitative_review"]["C1"][criterion]["status"] == "missing"
        assert result["gaps_rejected"] == ["G1"]
        assert result["request_coverage"] == "partial"
        assert result["wire_uncovered_requirement_ids"] == ["R2"]
    assert audit.assess(value)["quantitative_rejected_claim_ids"] == []


def test_quantitative_failure_targets_only_affected_claims_and_preserves_gap_gate():
    audit, value, _, grid, _ = audited_fixture()
    value["quantitative_review"]["C1"]["setting"]["status"] = "missing"
    assessment = audit.assess(value)
    state, data = state_and_data(grid)
    workflow = node()
    repaired = workflow.apply_answer_audit(state, data, assessment)["qa5"]
    assert not repaired["accepted"] and not repaired["terminal"]
    assert repaired["target"] == "author"
    assert repaired["feedback"]["error_codes"] == ["qa_quantitative_publication_rejected"]
    assert repaired["author_raw"] == canonical(state["answer"]).decode()
    lock = RepairLock(**repaired["lock"])
    assert lock.affected_ids == ("C1",)
    changed = copy.deepcopy(state["answer"])
    changed["claims"][1]["statement"] = "Unreviewed change to another claim."
    with pytest.raises(AgentError, match="typed_feedback_unaffected_claim_changed"):
        lock.verify(changed)
    state, data = state_and_data(grid)
    assessment["gaps_rejected"] = ["G1"]
    gap_repair = workflow.apply_answer_audit(state, data, assessment)["qa5"]
    assert gap_repair["gap_only_lock"] is not None and gap_repair["lock"] is None
    assert gap_repair["feedback"]["error_codes"] == ["qa_audit_gap_publication_rejected"]


def test_exhaustion_and_publication_guard_reject_missing_conditions_with_bounded_reserve():
    audit, value, candidate, grid, _ = audited_fixture()
    value["quantitative_review"]["C1"]["metric"]["status"] = "missing"
    assessment = audit.assess(value)
    workflow = node(revisions=2)
    state, data = state_and_data(grid)
    repaired = workflow.apply_answer_audit(state, data, assessment)["qa5"]

    def exhausted(_key, *, phase, error_codes, source_logical_key):
        assert phase == "qa5_content_revision"
        assert error_codes == ["qa_quantitative_publication_rejected"]
        assert source_logical_key.endswith(":qa5:review:0")
        raise AgentError("content_revision_limit")

    workflow.graph.execution.reserve_content_revision = exhausted
    rejected = workflow.repair({"qa5": repaired})["qa5"]
    assert rejected["terminal"] and not rejected["accepted"]
    with pytest.raises(AgentError, match="qa5_content_or_goal_review_unresolved"):
        workflow.finalize({"qa5": rejected})
    for used, expected in ((0, 2), (1, 1), (2, 0)):
        assert node(recoveries=used).protocol_call_reserve({}, "review") == expected
    workflow.answer_audit = lambda _state: audit
    recorded = copy.deepcopy(assessment)
    recorded["quantitative_rejected_claim_ids"] = []
    restored = {"answer": grid.candidate, "qa5": {
        "accepted": True, "candidate": candidate.model_dump(), "review_raw": json.dumps(value),
        "answer_audit": recorded, "answer_audit_binding": audit.binding,
    }}
    with pytest.raises(AgentError, match="qa_quantitative_publication_audit_changed"):
        workflow.finalize(restored)


def test_content_revision_records_requested_and_effective_source_without_fallback():
    audit, value, _, grid, _ = audited_fixture()
    value["quantitative_review"]["C1"]["unit"]["status"] = "missing"
    state, data = state_and_data(grid)
    workflow = node()
    data = workflow.apply_answer_audit(state, data, audit.assess(value))["qa5"]
    requested = data["feedback_source_key"]
    effective = requested + ":protocol-recovery:1"
    resolutions, revisions = [], []

    def resolve(key):
        resolutions.append(key)
        return effective

    def reserve(key, *, phase, error_codes, source_logical_key):
        revisions.append(source_logical_key)
        assert phase == "qa5_content_revision"
        assert error_codes == ["qa_quantitative_publication_rejected"]
        return {"ordinal": 1}

    workflow.graph.execution.effective_model_source_key = resolve
    workflow.graph.execution.reserve_content_revision = reserve
    result = workflow.repair({"qa5": data})["qa5"]
    assert resolutions == [requested] and revisions == [effective]
    assert data["feedback_source_key"] == requested
    assert result["feedback_source_key"] == effective
    assert result["revision"] == 1
    assert result["audit"][0] == {
        "phase": "content_revision_source_resolution",
        "requested_base_source_key": requested,
        "effective_source_key": effective,
    }
    assert result["audit"][1]["phase"] == "durable_content_revision"

    def unknown(_key):
        raise AgentError("protocol_recovery_source_not_settled")

    workflow.graph.execution.effective_model_source_key = unknown
    with pytest.raises(AgentError, match="protocol_recovery_source_not_settled"):
        workflow.repair({"qa5": data})
    assert revisions == [effective]
