"""A frozen reasoning reviewer with source- and answer-grounded grading."""
from citequest.research_v3.current_requirements import RequirementsCandidate
from citequest.research_v3.review_grid import ReviewGrid

from .qa_anchored_audit import ANCHORED_REVIEWER_POLICY, AnchoredQuantitativeAnswerAudit
from .qa_v6 import QA6
from .qa_v7 import QA7


class QA8(QA7):
    profile_id = "qa-v8"

    def reasoning_policy(self, state, role):
        return {"effort": "low" if role == "review" else "none"}

    def role_policy(self, state, role):
        if role == "review":
            return ANCHORED_REVIEWER_POLICY
        return super().role_policy(state, role)

    def answer_audit(self, state, grid=None):
        if grid is None:
            grid = ReviewGrid(self.scope(state), state["answer"], request_coverage=False)
        base = QA6.review_model(self, state, grid)
        candidate = RequirementsCandidate.model_validate(state["qa5"]["candidate"], strict=True)
        return AnchoredQuantitativeAnswerAudit(base, grid, candidate, reference_input=True)
