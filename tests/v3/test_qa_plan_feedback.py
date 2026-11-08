"""Pure controls for aggregate diagnostics, without providers or persistence."""
from __future__ import annotations

import copy
import json

import pytest

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.qa_plan_feedback import plan_preflight
from citequest.agents_v3.requirements_turn_units import resolve_selected_plan, turn_units
from citequest.research_v3.current_requirements import RequirementsState, UserTurn, propose_delta


def _inputs():
    previous = UserTurn(turn_id="U1", text="比较两种方法，说明实验设置，保留未改善的条件。")
    span = {"turn_id": previous.turn_id, "start": 0, "end": len(previous.text),
            "text": previous.text, "turn_sha256": digest(previous.text)}
    accepted = RequirementsState.model_validate({"authority_sha256": "a" * 64, "revision": 1,
        "turns": [{"turn_id": previous.turn_id, "text_sha256": digest(previous.text)}],
        "requirements": [{"id": f"R{index}", "category": category, "spans": [span],
            "state": "active", "introduced_turn_id": "U1", "replacement_ids": []}
            for index, category in enumerate(("question", "detail", "constraint"), start=1)],
        "accepted": True, "review_sha256": "b" * 64}, strict=True)
    turn = UserTurn(turn_id="U2", text="保留实验设置，取消总体比较，只关注一种方法。")
    unit = turn_units(turn)["units"][0]["unit_id"]
    plan = {"requirement_delta": {"parent_sha256": digest(accepted.model_dump(mode="json")),
        "authority_sha256": accepted.authority_sha256, "turn_id": turn.turn_id,
        "turn_sha256": digest(turn.text), "retain_ids": ["R2", "R3"],
        "operations": [{"action": "supersede", "target_ids": ["R1"], "category": "question", "spans": [unit]}],
        "intent": "clear", "clarification": None}, "action": "retrieve",
        "queries": [{"query": "experimental evidence", "limit": 6}],
        "existing_ref_ids": [f"issued-fixture-{index}" for index in range(14)]}
    return accepted, turn, plan


def test_child_failure_does_not_hide_duplicate_address_or_shared_capacity():
    accepted, turn, plan = _inputs()
    plan["requirement_delta"]["operations"].append({"action": "withdraw", "target_ids": ["R1"],
        "category": "focus", "spans": plan["requirement_delta"]["operations"][0]["spans"]})
    plan["queries"] = [{"query": "first query", "limit": 10}, {"query": "second query", "limit": 10}]
    original = copy.deepcopy(plan)
    feedback = plan_preflight(json.dumps(plan), accepted, turn)
    assert feedback["error_codes"] == ["qa_plan_operation_category", "qa_plan_active_addresses", "qa_plan_result_slots"]
    capacity = next(issue for issue in feedback["details"]["issues"] if issue["code"] == "qa_plan_result_slots")
    assert capacity["parameters"] == {"query_slots": 20, "reused_slots": 14, "total_slots": 34, "maximum": 20}
    assert feedback["details"]["checks"] == "server_deterministic_preflight/1"
    assert plan == original
    retained_and_targeted = copy.deepcopy(plan)
    retained_and_targeted["requirement_delta"]["retain_ids"].append("R1")
    assert "qa_plan_active_addresses" in plan_preflight(
        json.dumps(retained_and_targeted), accepted, turn)["error_codes"]


def test_legal_plan_still_uses_original_validator_and_strict_json_errors():
    accepted, turn, plan = _inputs()
    raw = json.dumps(plan)
    assert plan_preflight(raw, accepted, turn) is None
    candidate = propose_delta(accepted, turn, resolve_selected_plan(raw, accepted, turn).requirement_delta)
    assert candidate.projected is not None
    for malformed in ('{"action":', '{"action":"retrieve","action":"clarify"}', '{"value":NaN}'):
        assert plan_preflight(malformed, accepted, turn) is None
        with pytest.raises(AgentError):
            resolve_selected_plan(malformed, accepted, turn)


def test_diagnostics_never_authorize_unknown_refs_or_replace_server_identity():
    accepted, turn, plan = _inputs()
    plan["existing_ref_ids"] = ["unissued-private-reference"]
    raw = json.dumps(plan)
    assert plan_preflight(raw, accepted, turn) is None  # Original issued-ref gate must still run.
    assert json.loads(raw) == plan
    plan["requirement_delta"]["authority_sha256"] = "c" * 64
    raw = json.dumps(plan)
    feedback = plan_preflight(raw, accepted, turn)
    assert "qa_plan_field_invalid" in feedback["error_codes"]
    assert any(issue["path"] == ["requirement_delta", "authority_sha256"]
               for issue in feedback["details"]["issues"])
    assert "unissued-private-reference" not in json.dumps(feedback)
    assert "c" * 64 not in json.dumps(feedback)
    assert json.loads(raw)["requirement_delta"]["authority_sha256"] == "c" * 64


def test_original_query_reference_and_batch_bounds_remain_unchanged():
    accepted, turn, baseline = _inputs()
    for limits, ref_count, expected_code in (([20], 0, None), ([10, 10], 1, "qa_plan_result_slots"),
            ([21], 0, "qa_plan_field_invalid"), ([1], 21, "qa_plan_field_invalid")):
        plan = copy.deepcopy(baseline)
        plan["queries"] = [{"query": "bounded query", "limit": limit} for limit in limits]
        plan["existing_ref_ids"] = [f"issued-fixture-{index}" for index in range(ref_count)]
        raw = json.dumps(plan)
        feedback = plan_preflight(raw, accepted, turn)
        if expected_code is None:
            assert feedback is None
            resolve_selected_plan(raw, accepted, turn)
        else:
            assert expected_code in feedback["error_codes"]
        assert json.loads(raw) == plan
