"""Four pure controls. All graph/execution objects are explicit test doubles."""
import copy
from types import SimpleNamespace

import pytest

from citequest.agents_v3.contracts import AgentError, canonical, digest
from citequest.agents_v3.qa_audited_dialogue import GapOnlyRepairLock, QADialogueAudited
from citequest.research_v3.typed_feedback import RepairLock, strict_object


def candidate():
    return {
        "claims": [
            {"id": "C1", "statement": "The document states its objective.", "edges": []},
            {"id": "C2", "statement": "Measurements have not been completed.", "edges": []},
        ],
        "gaps": ["The available source does not contain measured outcomes."],
        "presentation": {"format": "prose", "paragraphs": [["C1"], ["C2"]]},
        "retrieval_scope": "Only the two selected source windows.",
        "evidence_status": "supported",
    }


def node(*, revisions=0):
    value = object.__new__(QADialogueAudited)
    value.graph = SimpleNamespace(execution=SimpleNamespace(
        command={"id": "explicit-test-command"}, content_revision_count=lambda: revisions,
    ))
    value.scope = lambda _state: SimpleNamespace(compact_candidate=copy.deepcopy)
    return value


def state_and_data():
    return {"answer": candidate()}, {
        "revision": 0, "audit": [], "accepted": False, "terminal": False,
        "lock": None, "gap_only_lock": None, "feedback": None,
    }


def test_gap_only_lock_preserves_every_other_candidate_value():
    original = candidate()
    lock = GapOnlyRepairLock.bind(original)
    corrected = copy.deepcopy(original)
    corrected["gaps"] = ["A corrected bounded limitation."]
    lock.verify(corrected)
    assert strict_object(lock.baseline_json) == original
    assert lock.contract()["allowed_paths"] == ["/gaps"]
    for changed in (
        {**corrected, "claims": [{**corrected["claims"][0], "statement": "Changed fact."}, corrected["claims"][1]]},
        {**corrected, "presentation": {"format": "prose", "paragraphs": [["C1", "C2"]]}},
        {**corrected, "retrieval_scope": "A wider source scope."},
        {**corrected, "evidence_status": "insufficient"},
        {**corrected, "new_root_field": "Unreviewed content."},
    ):
        with pytest.raises(AgentError, match="qa_audit_gap_repair_scope_changed"):
            lock.verify(changed)


def test_unsafe_gap_routes_to_gap_only_author_repair():
    value = node()
    state, data = state_and_data()
    assessment = {"gaps_rejected": ["G1"], "repair_claim_ids": [], "request_coverage": "partial"}
    result = value.apply_answer_audit(state, data, assessment)["qa5"]
    assert not result["accepted"] and not result["terminal"]
    assert result["target"] == "author" and result["lock"] is None
    assert result["feedback"]["error_codes"] == ["qa_audit_gap_publication_rejected"]
    assert result["author_raw"] == canonical(state["answer"]).decode()
    GapOnlyRepairLock(**result["gap_only_lock"]).verify(candidate())


def test_honest_partial_can_publish_and_existing_claim_omission_uses_original_lock():
    value = node()
    state, data = state_and_data()
    truthful = {"gaps_rejected": [], "repair_claim_ids": [], "request_coverage": "partial",
        "requirement_coverage": {"R1": {"status": "bounded_evidence_gap"}}}
    result = value.apply_answer_audit(state, data, truthful)["qa5"]
    assert result["accepted"] and result["terminal"] and result["feedback"] is None
    state, data = state_and_data()
    missing = {**truthful, "repair_claim_ids": ["C1"]}
    result = value.apply_answer_audit(state, data, missing)["qa5"]
    assert not result["accepted"] and not result["terminal"]
    assert result["target"] == "author"
    lock = RepairLock(**result["lock"])
    changed = copy.deepcopy(candidate())
    changed["claims"][1]["statement"] = "Silently changed an unaffected claim."
    with pytest.raises(AgentError, match="typed_feedback_unaffected_claim_changed"):
        lock.verify(changed)
    state, data = state_and_data()
    exhausted = node(revisions=2).apply_answer_audit(state, data, missing)["qa5"]
    assert exhausted["accepted"] and exhausted["terminal"]
    assert missing["request_coverage"] == "partial"


def test_unsafe_gap_cannot_publish_when_shared_revision_credit_is_exhausted():
    value = node(revisions=2)
    state, data = state_and_data()
    assessment = {"gaps_rejected": ["G1"], "repair_claim_ids": [], "request_coverage": "partial"}
    data = value.apply_answer_audit(state, data, assessment)["qa5"]

    def exhausted(_key, *, phase, error_codes, source_logical_key):
        assert phase == "qa5_content_revision"
        assert error_codes == ["qa_audit_gap_publication_rejected"]
        assert source_logical_key.endswith(":qa5:review:0")
        raise AgentError("content_revision_limit")

    value.graph.execution.reserve_content_revision = exhausted
    rejected = value.repair({"qa5": data})["qa5"]
    assert rejected["terminal"] and not rejected["accepted"]
    with pytest.raises(AgentError, match="qa5_content_or_goal_review_unresolved"):
        value.finalize({"qa5": rejected})
    # A restored accepted state cannot publish a candidate changed after review.
    bound = {"accepted": True, "candidate": {}, "answer_audit": {"gaps_rejected": []},
        "answer_audit_binding": {"answer_candidate_sha256": digest(candidate()),
            "requirements_candidate_sha256": digest({})}}
    changed = candidate()
    changed["gaps"] = ["An unreviewed replacement."]
    with pytest.raises(AgentError, match="qa_audit_publication_binding_changed"):
        value.finalize({"qa5": bound, "answer": changed})
