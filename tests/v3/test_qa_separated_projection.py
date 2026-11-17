"""One pure context projection control; ancestor context and receipts are doubles."""
import copy

from citequest.agents_v3.contracts import digest
from citequest.agents_v3.qa_separated import ACTIVE_CONTEXT, QASeparated
from citequest.agents_v3.qa_v5 import QA5
from citequest.agents_v3.qa_v9 import QA9
from citequest.research_v3.current_requirements import (
    RequirementDelta, RequirementReview, UserTurn, accept_review, active_projection,
    initial_state, propose_delta,
)


def test_reviewed_goals_are_the_only_writer_view_without_committing_or_changing_planning(monkeypatch):
    def propose(parent, identifier, text, retain, operations):
        turn = UserTurn(turn_id=identifier, text=text)
        values = []
        for action, targets, category, phrase in operations:
            start = text.index(phrase)
            values.append({"action": action, "target_ids": targets, "category": category,
                "spans": [{"turn_id": identifier, "start": start, "end": start + len(phrase)}]})
        delta = RequirementDelta.model_validate({"parent_sha256": digest(parent.model_dump()),
            "authority_sha256": parent.authority_sha256, "turn_id": identifier,
            "turn_sha256": digest(text), "retain_ids": retain, "operations": values,
            "intent": "clear", "clarification": None}, strict=True)
        candidate = propose_delta(parent, turn, delta)
        review = RequirementReview.model_validate({"candidate_sha256": digest(candidate.model_dump()),
            "user_turn_sha256": digest(text), "opinions": [{"requirement_id": rid,
                "verdict": "correct", "rationale": "Explicit synthetic goal-review fixture."}
                for rid in candidate.required_review_ids],
            "current_turn_coverage": "complete", "decision": "accepted"}, strict=True)
        return candidate, review, accept_review(candidate, review, current_state=parent)

    _, _, first = propose(initial_state("a" * 64), "U1", "解释问题；说明动机；介绍方法。", [], [
        ("add", [], "question", "解释问题"), ("add", [], "detail", "说明动机"),
        ("add", [], "detail", "介绍方法")])
    _, _, parent = propose(first, "U2", "保留问题和方法，撤销动机。", ["R1", "R3"], [
        ("withdraw", ["R2"], None, "撤销动机")])
    candidate, review, reviewed = propose(parent, "U3", "保留前述目标，补充适用范围，分两段中文。", ["R1", "R3"], [
        ("add", [], "detail", "补充适用范围"), ("add", [], "format", "分两段中文")])
    context = {"base": parent.model_dump(), "candidate": candidate.model_dump(),
        "review": review.model_dump(), "reviewed_state": reviewed.model_dump(),
        "binding": {"explicit_fixture": "settled receipts verified by separate receipt controls"}}
    prior = [{"assistant_reference_target": {"claims": [
        {"id": f"C{i}", "statement": f"Complete prior statement {i}."} for i in range(1, 9)]}}]
    state = {"requirements_state": parent.model_dump(), "evidence": ["source fixture"],
        "answer": {"claims": []}, "qa5": {"turn": candidate.user_turn.model_dump(),
            "candidate": candidate.model_dump(), "reviewing_goals": False,
            "goal_review_context": context, "next_requirements": None,
            "accepted": False, "terminal": False}}
    before = copy.deepcopy(state)
    current = QASeparated.__new__(QASeparated)
    current.accepted = lambda _state: parent
    verified = []

    def verified_receipt(_state):
        verified.append(True)
        return context, reviewed

    current._verified_goals = verified_receipt  # Explicit no-SQL receipt boundary double.

    def ancestor_context(_self, data, *, reviewer=False, planner=False):
        # Model the existing ancestor's two views without calling a provider.
        manifest = {"requirements_accepted_revision": parent.revision,
            "requirements_accepted_sha256": digest(parent.model_dump())}
        prompt = {"current_requirements": active_projection(parent),
            "context_manifest": manifest, "prior_turns": copy.deepcopy(prior),
            "requirements_candidate_sha256": digest(candidate.model_dump())}
        if not planner:
            prompt.update(provisional_requirements=[row.model_dump() for row in candidate.projected.requirements
                if row.state == "active"], candidate_requirements_are_not_current_until_independent_review=True)
        return prompt, manifest

    monkeypatch.setattr(QA9, "prompt", ancestor_context)
    for reviewer in (False, True):
        prompt, manifest = current.prompt(state, reviewer=reviewer)
        assert prompt["current_requirements"] == active_projection(reviewed)
        assert [row["id"] for row in prompt["current_requirements"]["active"]] == ["R1", "R3", "R4", "R5"]
        assert "provisional_requirements" not in prompt
        assert "candidate_requirements_are_not_current_until_independent_review" not in prompt
        assert prompt["requirements_goal_review"]["status"] == "independently_reviewed_pending_answer_commit"
        assert manifest["requirements_context_projection"] == ACTIVE_CONTEXT
        assert manifest["requirements_effective_sha256"] == digest(reviewed.model_dump())
        assert manifest["requirements_accepted_sha256"] == digest(parent.model_dump())
        assert prompt["requirements_candidate_sha256"] == digest(candidate.model_dump())
        assert prompt["prior_turns"] == prior and len(prompt["prior_turns"][0]["assistant_reference_target"]["claims"]) == 8
    planned, manifest = current.prompt(state, planner=True)
    assert planned["current_requirements"] == active_projection(parent)
    assert "requirements_context_projection" not in manifest
    assert len(verified) == 2

    def source_free_context(_self, data, *, reviewer=False, planner=False):
        assert data["evidence"] == [] and data["answer"] is None and reviewer
        return ancestor_context(_self, data, reviewer=reviewer, planner=planner)

    monkeypatch.setattr(QA5, "prompt", source_free_context)
    goal_state = copy.deepcopy(state)
    goal_state["qa5"]["reviewing_goals"] = True
    goal_prompt, _ = current.prompt(goal_state)
    assert goal_prompt["current_requirements"] == active_projection(parent)
    assert "provisional_requirements" in goal_prompt and len(verified) == 2
    assert state == before  # No pre-publication commit or prior-answer mutation.
    assert state["requirements_state"] == parent.model_dump()
    assert state["qa5"]["next_requirements"] is None and state["qa5"]["accepted"] is False
