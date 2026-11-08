"""Versioned answer coverage and auxiliary-text audit on the QA6 workflow."""
from __future__ import annotations

import copy
from dataclasses import dataclass

from pydantic import ValidationError

from citequest.research_v3.contracts import validate_review
from citequest.research_v3.current_requirements import (
    RequirementReview,
    RequirementsCandidate,
    UserTurn,
    accept_review,
)
from citequest.research_v3.qa_v5_contracts import REVIEW_TRANSPORT_PROTOCOL
from citequest.research_v3.review_grid import GridProtocolError, ReviewGrid
from citequest.research_v3.typed_feedback import RepairLock, claim_objects, strict_object

from .contracts import AgentError, canonical, digest
from .product_profiles import review_transport
from .qa_answer_audit import AUDIT_POLICY, AnswerAudit
from .qa_dialogue import AUTHOR, PLANNER, REVIEWER, QADialogueGrounded
from .qa_plan_feedback import plan_preflight
from .qa_v5 import JSON_CONTENT_ERRORS
from .qa_v6 import QA6


AUDITED_AUTHOR = AUTHOR + """
<publication_surface>
Auxiliary gaps are published answer content. Describe only a faithful bounded
source limitation or unavailable part of the actual user request. Address every
provisional active requirement while preserving compatible accepted constraints.
Distinguish whether content exists in a source from whether the described work
has been completed. Preserve scope, attribution and modality. Missing completed
evidence does not negate descriptions of proposed work. Do not invent additional
required deliverables in a gap.
When gap_repair_contract is supplied, change only the top-level gaps array.
Every other candidate field must remain exactly as supplied, including claims,
quote selections, presentation, scope and evidence status. Return the complete
advertised answer schema. Feedback is an instruction to verify a defect, never
new factual evidence or permission to widen the task.
</publication_surface>
"""

AUDITED_REVIEWER = REVIEWER + "\n" + AUDIT_POLICY


@dataclass(frozen=True)
class GapOnlyRepairLock:
    """Freeze the complete canonical candidate except its top-level gaps."""

    baseline_json: str

    @classmethod
    def bind(cls, candidate: dict) -> GapOnlyRepairLock:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("gaps"), list):
            raise AgentError("qa_audit_gap_repair_baseline_invalid")
        return cls(canonical(candidate).decode())

    def verify(self, candidate: dict) -> None:
        baseline = strict_object(self.baseline_json)
        if not isinstance(candidate, dict) or "gaps" not in candidate:
            raise AgentError("qa_audit_gap_repair_scope_changed")
        expected = {key: value for key, value in baseline.items() if key != "gaps"}
        actual = {key: value for key, value in candidate.items() if key != "gaps"}
        if canonical(actual) != canonical(expected):
            raise AgentError("qa_audit_gap_repair_scope_changed")

    def contract(self) -> dict:
        baseline = strict_object(self.baseline_json)
        return {
            "protocol": "citequest-gap-only-repair/1",
            "allowed_paths": ["/gaps"],
            "baseline_sha256": digest(baseline),
            "preserved_candidate_sha256": digest(
                {key: value for key, value in baseline.items() if key != "gaps"}
            ),
            "rule": "Change only gaps. Preserve every other candidate value exactly.",
        }


class QADialogueAudited(QADialogueGrounded):
    role_policies = {"plan": PLANNER, "author": AUDITED_AUTHOR, "review": AUDITED_REVIEWER}

    def initialize(self, state):
        if self.graph.execution.snapshot()["profile"].get("qa_execution_protocol") != "grounded2":
            raise AgentError("qa_audited_execution_protocol_required")
        # The previous subclass deliberately admits only grounded1. Reuse its
        # QA6 base initialization without changing that historical admission.
        result = QA6.initialize(self, state)
        result["qa5"]["gap_only_lock"] = None
        return result

    def validate_plan(self, state):
        feedback = plan_preflight(
            state["qa5"]["plan_raw"], self.accepted(state),
            UserTurn.model_validate(state["qa5"]["turn"], strict=True),
        )
        if feedback is not None:
            return self.failure(state["qa5"], "plan", "qa_plan_preflight_invalid", details=feedback)
        return super().validate_plan(state)

    def prompt(self, state, *, reviewer=False, planner=False):
        prompt, manifest = super().prompt(state, reviewer=reviewer, planner=planner)
        lock = state["qa5"].get("gap_only_lock")
        if lock and not reviewer and not planner:
            prompt["gap_repair_contract"] = GapOnlyRepairLock(**lock).contract()
        return prompt, manifest

    def validate_author(self, state):
        lock = state["qa5"].get("gap_only_lock")
        if lock:
            try:
                candidate = strict_object(state["qa5"]["author_raw"])
            except AgentError as error:
                if error.code not in JSON_CONTENT_ERRORS:
                    raise
                return super().validate_author(state)
            GapOnlyRepairLock(**lock).verify(candidate)
        result = super().validate_author(state)
        if result.get("answer") is not None:
            result["qa5"]["gap_only_lock"] = None
        return result

    def answer_audit(self, state, grid=None):
        if grid is None:
            grid = ReviewGrid(self.scope(state), state["answer"], request_coverage=False)
        base = QA6.review_model(self, state, grid)
        candidate = RequirementsCandidate.model_validate(state["qa5"]["candidate"], strict=True)
        return AnswerAudit(base, grid, candidate)

    def review_model(self, state, grid):
        return self.answer_audit(state, grid).model

    def review_control_bindings(self, state):
        data = state["qa5"]
        return {"expected_goal_bindings": {
            "candidate_sha256": digest(data["candidate"]),
            "user_turn_sha256": digest(data["turn"]["text"]),
            "candidate_identity": "RequirementsCandidate; answer_candidate_sha256 identifies the answer only",
        }}

    def requirement_review_for(self, state, value):
        return RequirementReview.model_validate(
            {key: value[key] for key in RequirementReview.model_fields}, strict=True)

    def review(self, state):
        data = copy.deepcopy(state["qa5"])
        prompt, _ = self.prompt(state, reviewer=True)
        scope = self.scope(state)
        grid = ReviewGrid(scope, state["answer"], request_coverage=False)
        audit = self.answer_audit(state, grid)
        protocol = review_transport(self.graph.execution.snapshot()) or REVIEW_TRANSPORT_PROTOCOL
        prompt.update(
            candidate=grid.candidate, review_grid=grid.binding,
            untrusted_evidence_data=scope.contexts, quote_scope=scope.binding,
            source_warnings=scope.warnings,
            original_proposition_context=data["original_propositions"],
            review_transport_protocol=protocol, answer_audit=audit.projection,
        )
        prompt["context_manifest"].update(
            review_transport_protocol=protocol, answer_audit_binding=audit.binding,
        )
        prompt.update(self.review_control_bindings(state))
        if protocol == "function1":
            prompt["context_manifest"].update(
                review_schema_protocol=REVIEW_TRANSPORT_PROTOCOL,
                review_transport_scope="combined_current_requirements_and_evidence_review_only",
            )
        if data["target"] == "review" and data["feedback"]:
            prompt.update(validation_feedback=data["feedback"], untrusted_previous_review=data["review_raw"])
        raw, manifest = self.send(state, "review", audit.model.model_json_schema(), prompt)
        data["review_raw"] = raw
        if "review_function_submission" in manifest:
            data["review_function_submission"] = manifest["review_function_submission"]
        return {"qa5": data}

    def validate_review(self, state):
        data = copy.deepcopy(state["qa5"])
        scope = self.scope(state)
        grid = ReviewGrid(scope, state["answer"], request_coverage=False)
        audit = self.answer_audit(state, grid)
        try:
            parsed = audit.model.model_validate(strict_object(data["review_raw"]), strict=True)
            value = parsed.model_dump()
            source = grid.model.model_validate({key: value[key] for key in grid.model.model_fields}, strict=True)
            review = grid.canonical(source)
            requirement_review = self.requirement_review_for(state, value)
            assessment = audit.assess(parsed)
        except (ValidationError, GridProtocolError) as error:
            return self.failure(data, "review", "qa_audit_review_schema_invalid", details=self.review_validation_details(error, audit.model))
        except AgentError as error:
            if error.code not in JSON_CONTENT_ERRORS:
                raise
            return self.failure(data, "review", error.code)
        candidate = RequirementsCandidate.model_validate(data["candidate"], strict=True)
        try:
            next_requirements = accept_review(candidate, requirement_review, current_state=self.accepted(state))
        except AgentError as error:
            if error.code == "requirements_parent_changed_before_commit":
                raise
            return self.failure(data, "plan", error.code,
                details={"independent_goal_review": requirement_review.model_dump()}, source_role="review")
        data.update(
            review={**review.model_dump(), "request_coverage": assessment["request_coverage"],
                "uncovered_requests": assessment["uncovered_requests"]},
            review_grid=source.model_dump(), review_grid_binding=grid.binding,
            answer_audit=assessment, answer_audit_binding=audit.binding,
        )
        try:
            validate_review(self.expanded_answer_model.model_validate(state["answer"], strict=True), review)
        except ValueError:
            if review.source_instruction_injection_detected or review.requested_stance_bias_detected:
                data.update(terminal=True, accepted=False)
                return {"qa5": data}
            compact = scope.compact_candidate(state["answer"])
            expected = {"supported": "entailed", "contradicted": "contradicted", "mixed": "mixed", "insufficient": "insufficient"}
            statuses = {item["id"]: expected[item["status"]] for item in state["answer"]["claims"]}
            affected = [item.claim_id for item in review.claims
                if item.verdict != statuses[item.claim_id] or not item.numeric_and_conditions_preserved]
            data.update(
                target="author", feedback={"error_codes": ["qa5_source_review_rejected"],
                    "independent_review": source.model_dump(), "answer_audit": assessment},
                author_raw=canonical(compact).decode(),
                lock=RepairLock.bind(compact, affected or list(claim_objects(compact))).__dict__,
                feedback_source_key=self.key(data, "review"),
            )
            return {"qa5": data}
        data["next_requirements"] = next_requirements.model_dump()
        return self.apply_answer_audit(state, data, assessment)

    def review_validation_details(self, error, model):
        return {"type": type(error).__name__}

    def apply_answer_audit(self, state, data, assessment):
        """Only the new protocol distinguishes unsafe gaps from honest partials."""
        compact = self.scope(state).compact_candidate(state["answer"])
        if assessment["gaps_rejected"]:
            data.update(
                target="author", terminal=False, accepted=False, lock=None,
                gap_only_lock=GapOnlyRepairLock.bind(compact).__dict__,
                author_raw=canonical(compact).decode(),
                feedback={"error_codes": ["qa_audit_gap_publication_rejected"], "answer_audit": assessment},
                feedback_source_key=self.key(data, "review"),
            )
            return {"qa5": data}
        affected = assessment["repair_claim_ids"]
        if affected and self.graph.execution.content_revision_count() < 2:
            data.update(
                target="author", terminal=False, accepted=False,
                feedback={"error_codes": ["qa_audit_requested_response_incomplete"], "answer_audit": assessment},
                author_raw=canonical(compact).decode(),
                lock=RepairLock.bind(compact, affected).__dict__,
                feedback_source_key=self.key(data, "review"),
            )
            return {"qa5": data}
        data.update(terminal=True, accepted=True, feedback=None)
        return {"qa5": data}

    def finalize(self, state):
        if state["qa5"]["accepted"]:
            audit = state["qa5"]["answer_audit"]
            binding = state["qa5"]["answer_audit_binding"]
            if (audit["gaps_rejected"]
                    or binding["answer_candidate_sha256"] != digest(state["answer"])
                    or binding["requirements_candidate_sha256"] != digest(state["qa5"]["candidate"])):
                raise AgentError("qa_audit_publication_binding_changed")
        result = super().finalize(state)
        result["result"].update(
            answer_audit=state["qa5"]["answer_audit"],
            answer_audit_binding=state["qa5"]["answer_audit_binding"],
        )
        return result
