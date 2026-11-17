"""One bounded corrective retrieval; no provider, SQL or retrieval-quality calls."""
import copy

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import AgentError, PROFILES
from citequest.agents_v3.qa_adaptive_contracts import (
    SupplementPlan, bounded_gap_targets, supplement_capacity, validate_supplement_plan,
)


def test_supplement_plan_binds_reviewed_goals_and_shared_evidence_capacity():
    values = {"candidate_sha256": "a" * 64, "evidence_sha256": "b" * 64,
        "target_requirement_ids": ["R2"], "action": "retrieve",
        "queries": [{"query": "specific missing measurement", "limit": 2}], "public_reason": "A bounded second query."}
    contract = {"candidate_sha256": "a" * 64, "evidence_sha256": "b" * 64,
        "target_requirement_ids": ["R2"], "existing_ref_ids": [f"ref:{index}" for index in range(18)],
        "allowed_limits": [1, 2, 3]}
    plan = validate_supplement_plan(values, **contract)
    assert plan.round == 1 and plan.queries[0].limit == 2
    for mutation in ({"candidate_sha256": "c" * 64}, {"target_requirement_ids": ["R3"]},
                     {"queries": [{"query": "specific missing measurement", "limit": 3}]}):
        with pytest.raises(AgentError):
            validate_supplement_plan({**values, **mutation}, **contract)
    with pytest.raises(ValidationError):
        SupplementPlan.model_validate({**values, "round": 2}, strict=True)
    with pytest.raises(ValidationError):
        SupplementPlan.model_validate({**values, "queries": [{"query": "bad\x00query", "limit": 1}]}, strict=True)
    kept = validate_supplement_plan({**values, "action": "keep_bounded_answer", "queries": []}, **contract)
    assert kept.queries == []


def test_only_safe_bounded_gaps_are_candidates_for_another_search():
    data = {"accepted": True, "terminal": True, "feedback": None}
    assessment = {"requirement_coverage": {"R1": {"status": "fulfilled"},
        "R2": {"status": "bounded_evidence_gap"}, "R3": {"status": "format_mismatch"}},
        "gaps_rejected": [], "quantitative_rejected_claim_ids": []}
    assert bounded_gap_targets(data, assessment) == ["R2"]
    for changes in ({"accepted": False}, {"terminal": False}, {"feedback": {"source": "rejected"}}):
        assert bounded_gap_targets({**data, **changes}, assessment) == []
    assert bounded_gap_targets(data, {**assessment, "gaps_rejected": ["G1"]}) == []
    assert bounded_gap_targets(data, {**assessment, "quantitative_rejected_claim_ids": ["C1"]}) == []


def test_complete_following_path_preserves_shared_credits_and_sequential_cash_admission():
    profile = PROFILES["qa-v9"].model_dump()
    single_cash = profile["context_window"] * profile["input_miss_ncny"] + profile["per_call_output"] * profile["output_ncny"]
    task = {"profile": profile, "model_calls": 4, "tool_calls": 2,
        "input_tokens": 1000, "output_tokens": 1000, "known_ncny": profile["cash_ncny"] - single_cash,
        "reserved_ncny": 0, "elapsed": 1}
    original = copy.deepcopy(task)
    parameters = {"revisions": 0, "recoveries": 0, "evidence_count": 10,
        "allowed_limits": list(range(1, 21)), "navigation": False}
    receipt = supplement_capacity(task, **parameters)
    assert receipt["admitted"] is True and receipt["required_model_calls"] == 7
    assert receipt["allowed_limits"] == list(range(1, 11)) and task == original
    assert receipt["cash_headroom_basis"] == "maximum_single_request_sequential_reservation"
    assert supplement_capacity(task, **{**parameters, "revisions": 2})["reason"] == "content_revision"
    assert supplement_capacity(task, **{**parameters, "evidence_count": 20})["reason"] == "evidence_slots"
    assert supplement_capacity({**task, "model_calls": profile["model_calls"]}, **parameters)["reason"] == "model_calls"
    assert supplement_capacity({**task, "known_ncny": task["known_ncny"] + 1}, **parameters)["reason"] == "cash"
