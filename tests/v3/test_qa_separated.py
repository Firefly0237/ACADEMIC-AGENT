"""Two goal/source separation controls with explicit receipt and execution doubles."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import canonical, digest
from citequest.agents_v3.qa_gap_catalog import GapCatalog
from citequest.agents_v3.qa_online_audit import SETTING_FACETS
from citequest.agents_v3.qa_separated import QASeparated, SourceAnswerAudit
from citequest.agents_v3.qa_v5 import QA5
from citequest.research_v3.current_requirements import (
    RequirementDelta, RequirementReview, UserTurn, accept_review, initial_state, propose_delta,
)

sys.path.insert(0, str(Path(__file__).parent))
from test_quantitative_reference_input import reference_fixture


def goal_review(opinions):
    return {key: copy.deepcopy(opinions[key]) for key in RequirementReview.model_fields}


def test_source_negative_opinions_cannot_revote_the_independently_reviewed_goal():
    _old, opinions, candidate, grid, _base = reference_fixture()
    accepted_goal = goal_review(opinions)
    audit = SourceAnswerAudit(grid, candidate, {"explicit_fixture": "known independent receipt"},
        gap_catalog=GapCatalog(candidate))
    assert not {"opinions", "current_turn_coverage", "decision"} & set(audit.model.model_fields)
    raw = {key: copy.deepcopy(value) for key, value in opinions.items() if key in audit.model.model_fields}
    raw["online_binding_sha256"] = audit.binding["binding_sha256"]
    raw["setting_facets"] = {identifier: {facet: {
        "status": "not_applicable", "reason": "Synthetic inapplicability."}
        for facet in SETTING_FACETS} for identifier in audit.claim_ids}
    raw["setting_facets"]["C1"]["data_population"] = {
        "status": "missing", "reason": "Synthetic missing answer condition."}
    raw["quantitative_review"]["C1"]["setting"]["status"] = "missing"
    result = audit.assess(raw)
    assert result["quantitative_rejected_claim_ids"] == ["C1"]
    assert result["gap_contract_rejected"] == ["G1"]
    injected = {**raw, "decision": "needs_revision"}
    with pytest.raises(ValidationError):
        audit.model.model_validate(injected, strict=True)

    current = QASeparated.__new__(QASeparated)
    state = {"qa5": {"candidate": candidate.model_dump(), "turn": candidate.user_turn.model_dump()}}
    context = {"review": accepted_goal}
    current._verified_goals = lambda _state: (context, None)  # Explicit receipt lookup double.
    saved = current.requirement_review_for(state, raw)
    assert saved.model_dump() == accepted_goal
    assert saved.decision == "accepted"


def test_goal_review_precedes_sources_keeps_commit_pending_and_routes_real_clarification(monkeypatch):
    _old, opinions, candidate, _grid, _base = reference_fixture()
    first_review = RequirementReview.model_validate(goal_review(opinions), strict=True)
    accepted = accept_review(candidate, first_review, current_state=initial_state("a" * 64))
    turn = UserTurn(turn_id="U2", text="保留已有要求，补充条件。")
    start = turn.text.index("补充")
    delta = RequirementDelta.model_validate({
        "parent_sha256": digest(accepted.model_dump()), "authority_sha256": accepted.authority_sha256,
        "turn_id": turn.turn_id, "turn_sha256": digest(turn.text),
        "retain_ids": [item.id for item in accepted.requirements], "operations": [{
            "action": "add", "target_ids": [], "category": "detail", "spans": [{
                "turn_id": turn.turn_id, "start": start, "end": len(turn.text)}]}],
        "intent": "clear", "clarification": None}, strict=True)
    candidate = propose_delta(accepted, turn, delta)
    accepted_review = {"candidate_sha256": digest(candidate.model_dump()),
        "user_turn_sha256": digest(turn.text), "opinions": [{"requirement_id": identifier,
            "verdict": "correct", "rationale": "Synthetic retained or added requirement opinion."}
            for identifier in candidate.required_review_ids],
        "current_turn_coverage": "complete", "decision": "accepted"}
    command = uuid4()
    current = QASeparated.__new__(QASeparated)
    current.graph = SimpleNamespace(execution=SimpleNamespace(command={"id": command}))
    planner_receipt = {"attempt_id": str(uuid4()), "request_sha256": "a" * 64,
        "logical_key": str(command) + ":qa5:plan:0"}
    reviewer_receipt = {"attempt_id": str(uuid4()), "request_sha256": "b" * 64,
        "logical_key": str(command) + ":qa5:plan:1"}
    current.receipt = lambda _data, _role: copy.deepcopy(reviewer_receipt)  # Explicit settled-receipt double.
    observed = []

    def source_free_base_prompt(_self, state, *, reviewer=False, planner=False):
        assert reviewer is True and state["evidence"] == [] and state["answer"] is None
        observed.append("source_free_goal_prompt")
        return {"untrusted_evidence_data": ["excluded fixture"], "quote_scope": {},
            "source_warnings": ["excluded fixture"], "context_manifest": {}}, {}

    monkeypatch.setattr(QA5, "prompt", source_free_base_prompt)
    for clarification in (False, True):
        review = copy.deepcopy(accepted_review)
        if clarification:
            review.update(decision="needs_clarification", current_turn_coverage="ambiguous")
            for row in review["opinions"]:
                row["verdict"] = "ambiguous"
        captured = []

        def synthetic_send(state, role, schema, prompt):
            assert state["qa5"]["reviewing_goals"] is True and role == "review"
            assert "untrusted_evidence_data" not in prompt and "quote_scope" not in prompt
            captured.append(current.key(state["qa5"], role))
            return canonical(review).decode(), {}

        current.send = synthetic_send  # Explicit model-output double, no provider invocation.
        binding = {"planner_attempt_id": planner_receipt["attempt_id"],
            "planner_request_sha256": planner_receipt["request_sha256"],
            "candidate_sha256": "c" * 64, "plan_sha256": "d" * 64}
        state = {"requirements_state": accepted.model_dump(),
            "evidence": ["unrelated old source fixture"], "answer": {"unrelated": True},
            "qa5": {"revision": 0, "candidate": candidate.model_dump(), "turn": candidate.user_turn.model_dump(),
                "target": "plan", "feedback": None, "audit": [], "accepted": False, "terminal": False,
                "goal_planner_receipt": planner_receipt, "goal_planner_base_key": planner_receipt["logical_key"],
                "goal_review_raw": "", "review_raw": "", "review": None, "next_requirements": None,
                "binding": binding}}
        result = current.review_plan(state)
        data = result["qa5"]
        assert captured == [reviewer_receipt["logical_key"]]
        assert data["binding"] == binding and data["reviewing_goals"] is False
        assert data["accepted"] is False and data["terminal"] is False and data["next_requirements"] is None
        if clarification:
            assert data["candidate"]["projected"] is None and data["goal_review_context"] is None
            assert current.after_plan_review(result) == "clarify"
        else:
            assert data["goal_review_context"]["review"] == review
            assert data["goal_review_context"]["base"] == accepted.model_dump()
            assert data["goal_review_context"]["base"]["revision"] == 1
            assert current.after_plan_review(result) == "retrieve"
    assert observed == ["source_free_goal_prompt", "source_free_goal_prompt"]
