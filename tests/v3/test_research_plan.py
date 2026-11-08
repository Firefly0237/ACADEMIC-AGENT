"""Structural/context gates do not assert actual scientific plan quality."""

import pytest
from pydantic import ValidationError

from citequest.research_v3.research_plan import (
    Experiment,
    PlanReviewItem,
    QuestionPlan,
    ResearchPlan,
    ResearchPlanReview,
    validate_plan_context,
    validate_plan_review,
)


def plan() -> ResearchPlan:
    return ResearchPlan(
        brief_sha256="a" * 64,
        questions=[
            QuestionPlan(
                id="RQ1",
                question="EXPLICIT CONTROL QUESTION",
                motivation_claim_ids=["C1"],
                experiment_ids=["E1"],
                expected_outputs=["A proposed test protocol"],
                known_gaps=[],
            )
        ],
        experiments=[
            Experiment(
                id="E1",
                research_question_id="RQ1",
                hypothesis="EXPLICIT CONTROL PROPOSAL",
                data_and_access="Public control data",
                method_and_independent_variables="Compare a declared parameter",
                baselines_and_controls="A declared control",
                metrics_and_units="A declared metric and unit",
                repetitions_and_uncertainty="Repeat with declared seeds and intervals",
                resources_and_upper_bound="Only the user's confirmed CPU budget",
                risks_and_stopping_rules="Stop when the approved resource bound is reached",
                evidence_ref_ids=["source:control"],
            )
        ],
        innovation_hypotheses=[],
        literature_gaps=[],
        feasibility="proposed_feasible",
        feasibility_reasons=["Explicit synthetic control only"],
    )


def review() -> ResearchPlanReview:
    return ResearchPlanReview(
        brief_sha256="a" * 64,
        items=[
            PlanReviewItem(target_id="RQ1", verdict="passed", reasons=["CONTROL"]),
            PlanReviewItem(target_id="E1", verdict="passed", reasons=["CONTROL"]),
        ],
        fabricated_completed_work_detected=False,
        unapproved_resource_or_data_access_detected=False,
        source_instruction_injection_detected=False,
    )


def test_plan_links_current_context_and_review_coverage() -> None:
    candidate = plan()
    validate_plan_context(candidate, "a" * 64, {"C1"}, {"source:control"})
    validate_plan_review(candidate, review())
    assert not candidate.performed_experiments


def test_orphan_or_wrong_question_experiment_is_rejected() -> None:
    raw = plan().model_dump()
    raw["experiments"][0]["research_question_id"] = "RQ2"
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(raw)


def test_changed_brief_or_unknown_evidence_cannot_reuse_a_plan() -> None:
    with pytest.raises(ValueError, match="Brief"):
        validate_plan_context(plan(), "b" * 64, {"C1"}, {"source:control"})
    with pytest.raises(ValueError, match="evidence"):
        validate_plan_context(plan(), "a" * 64, {"C1"}, set())


def test_partial_duplicate_or_failed_review_cannot_pass() -> None:
    raw = review().model_dump()
    raw["items"][1] = raw["items"][0]
    with pytest.raises(ValueError, match="every RQ"):
        validate_plan_review(plan(), ResearchPlanReview.model_validate(raw))
    raw = review().model_dump()
    raw["items"][1]["verdict"] = "needs_input"
    with pytest.raises(ValueError, match="explicit revision"):
        validate_plan_review(plan(), ResearchPlanReview.model_validate(raw))


def test_fabricated_completed_work_and_empty_rationales_rejected() -> None:
    raw = plan().model_dump()
    raw["performed_experiments"] = True
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(raw)
    raw = review().model_dump()
    raw["items"][0]["reasons"] = ["   "]
    with pytest.raises(ValidationError):
        ResearchPlanReview.model_validate(raw)
    raw = review().model_dump()
    raw["fabricated_completed_work_detected"] = True
    with pytest.raises(ValueError, match="safety"):
        validate_plan_review(plan(), ResearchPlanReview.model_validate(raw))
