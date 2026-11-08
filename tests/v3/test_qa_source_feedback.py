"""Three pure feedback controls. The parent node result is an explicit double."""
import copy

from citequest.agents_v3.qa_audited_dialogue import QADialogueAudited
from citequest.agents_v3.qa_source_feedback import source_review_feedback
from citequest.agents_v3.qa_verified_dialogue import QADialogueVerified
from citequest.research_v3.typed_feedback import RepairLock


def fixture():
    candidate = {"claims": [
        {"id": "C1", "statement": "The source describes proposed work.", "status": "supported", "edges": []},
        {"id": "C8", "statement": "The source says the proposed work is uncompleted.",
            "status": "insufficient", "edges": [], "limitations": ["No completed measurement is available."]},
    ]}
    grid = {"claims": {
        identifier: {"verdict": "entailed", "conditions_preserved": True,
            "explanation": "The selected source supports the written proposition.", "quotes": {
                quote: {"relation": "matches_declared_relations", "conditions_preserved": True,
                    "explanation": "The selected quote matches the declared relation."},
            }} for identifier, quote in (("C1", "Q1"), ("C8", "Q8"))
    }}
    review = {"claims": [
        {"claim_id": identifier, "verdict": "entailed", "numeric_and_conditions_preserved": True}
        for identifier in ("C1", "C8")
    ]}
    return candidate, review, grid


def test_c8_mismatch_is_precise_and_grounded3_hook_preserves_original_lock_and_review(monkeypatch):
    candidate, review, grid = fixture()
    before = copy.deepcopy((candidate, review, grid))
    data = {"target": "author", "review": review, "review_grid": grid,
        "lock": RepairLock.bind(candidate, ["C8"]).__dict__,
        "feedback": {"error_codes": ["qa5_source_review_rejected"], "independent_review": copy.deepcopy(grid)}}
    parent = {"qa5": copy.deepcopy(data)}
    monkeypatch.setattr(QADialogueAudited, "validate_review", lambda _self, _state: copy.deepcopy(parent))
    workflow = object.__new__(QADialogueVerified)
    result = workflow.validate_review({"answer": candidate})["qa5"]
    feedback = result["feedback"]["source_invariants"]
    assert feedback["affected_claim_ids"] == ["C8"]
    issue = feedback["issues"][0]
    assert issue["declared_status"] == "insufficient"
    assert issue["expected_verdict_for_declared_status"] == "insufficient"
    assert issue["reviewer_verdict"] == issue["effective_verdict_after_quote_checks"] == "entailed"
    assert issue["failed_invariants"] == ["claim_status_verdict_mismatch"]
    assert issue["failed_selected_quotes"] == []
    assert issue["independent_explanation"] == grid["claims"]["C8"]["explanation"]
    assert result["lock"] == data["lock"] and result["feedback"]["independent_review"] == grid
    assert "change a label merely to pass" in feedback["rule"]
    assert (candidate, review, grid) == before and parent["qa5"] == data


def test_failed_quote_relation_and_conditions_report_actual_cells_and_effective_verdict():
    candidate, review, grid = fixture()
    candidate["claims"][1]["status"] = "supported"
    grid["claims"]["C8"]["quotes"]["Q8"].update(
        relation="does_not_match", conditions_preserved=False,
        explanation="A necessary source condition was omitted.",
    )
    review["claims"][1].update(verdict="invalid", numeric_and_conditions_preserved=False)
    result = source_review_feedback(candidate, review, grid)
    issue = result["issues"][0]
    assert result["affected_claim_ids"] == ["C8"]
    assert issue["expected_verdict_for_declared_status"] == issue["reviewer_verdict"] == "entailed"
    assert issue["effective_verdict_after_quote_checks"] == "invalid"
    assert issue["failed_invariants"] == ["claim_status_verdict_mismatch",
        "selected_quote_relation_failed", "claim_or_quote_conditions_not_preserved"]
    assert issue["claim_conditions_preserved"] and not issue["effective_conditions_preserved"]
    assert issue["failed_selected_quotes"] == [{"quote_id": "Q8", **grid["claims"]["C8"]["quotes"]["Q8"]}]


def test_consistent_supported_and_genuine_insufficient_claims_need_no_feedback(monkeypatch):
    candidate, review, grid = fixture()
    candidate["claims"][1]["statement"] = "The requested measurement is not established in the available evidence."
    review["claims"][1]["verdict"] = "insufficient"
    grid["claims"]["C8"]["verdict"] = "insufficient"
    before = copy.deepcopy((candidate, review, grid))
    assert source_review_feedback(candidate, review, grid) is None
    assert (candidate, review, grid) == before
    parent = {"qa5": {"accepted": True, "terminal": True, "feedback": None}}
    monkeypatch.setattr(QADialogueAudited, "validate_review", lambda _self, _state: copy.deepcopy(parent))
    workflow = object.__new__(QADialogueVerified)
    assert workflow.validate_review({"answer": candidate}) == parent
