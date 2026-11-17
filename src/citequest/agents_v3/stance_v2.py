"""Explicit stance-v2 on the shared typed-requirements/Q evidence protocol.

The same official compiled graph, Saver, provider, revision ledger and MCP
client are reused. A separately reviewed atomic plan is required BEFORE any
retrieval; support/refute is a user objective, never a forced answer verdict.
"""
from __future__ import annotations

import copy
from typing import ClassVar, Literal

from pydantic import ValidationError, model_validator

from citequest.research_v3.qa_v5_contracts import QAPlanV5, SelectorAnswerV5
from citequest.research_v3.typed_feedback import strict_object

from .claim_plan import ClaimPlan, PlanReview, validate_plan, validate_plan_review
from .contracts import AgentError, digest
from .qa_v5 import AUTHOR, COMMON, JSON_CONTENT_ERRORS, QA5, REVIEWER

STANCE_RULE = """The current canonical Brief's explicit support/refute choice is
the search objective. Preserve the exact atomic propositions and their conditions
from the reviewed claim plan. A relevant source may support, contradict or only
contextualize a proposition. Seek and preserve opposing findings; never coerce a
verdict to satisfy the selected objective. Insufficient bounded evidence must be
reported as insufficient, without implying no literature exists elsewhere.
"""

STANCE_PLANNER = COMMON + """Role: current-requirement and atomic-stance planner.
Input: the current raw user turn, accepted active requirements, exact input unit
ranges, immutable Task source authority and bounded retrieval policy. No source
findings or previous model answers are available at this stage.
Task: interpret every current subquestion, condition, detail and format request.
Bind parent/authority/turn hashes exactly. Address every old active ID once by
retain, supersede or withdraw; keep compatible requirements and never restore a
retired topic from raw audit history. Operations use exact current input ranges.
Create at most eight atomic propositions, each with a literal persisted-user
anchor, preserved negation, quantities, baselines and conditions. Normalization
may restore an omitted subject but cannot add facts or a stronger quantifier.
An unresolved or excess request stays an anchored unplanned_request, never an
omitted obligation. Ambiguity in the current scope requires clarification.
Plan at most two specific retrieval queries; new slots plus reused refs <=20,
each limit must cover every selected document. Reuse only listed Task-issued
evidence refs. A paper ID is not a reference ID. No tool runs before review.
Output: exactly StancePlanV2. For clarify, claim_plan is null with no query or
evidence reuse. Otherwise include the complete typed delta, bounded retrieval
intent and claim_plan. Preserve the current explicit support/refute direction.
""" + STANCE_RULE

STANCE_PLAN_REVIEWER = COMMON + """Role: independent pre-retrieval atomic-plan reviewer.
Input: program-checked literal anchors, the exact current raw user turn, accepted
and provisional active requirements, current direction and the full claim plan.
Task: evaluate every planned claim exactly once for faithful normalization,
preserved negation/quantities/conditions, no new fact or stronger quantifier, and
the current explicit direction. Check that every active proposition is planned
or named as an anchored unplanned request. Retired audit text is not a current
obligation. No papers, old answers or generated facts can justify a proposition.
Reject unresolved references or omissions; do not repair the plan or add claims.
Output: exactly PlanReview with the complete fixed claim-ID set and concise
public reasons. A passed plan is permission for bounded evidence gathering, not
proof of the user's proposition or permission to change source/budget authority.
""" + STANCE_RULE


class StancePlanV2(QAPlanV5):
    claim_plan: ClaimPlan | None

    @model_validator(mode="after")
    def claim_plan_matches_action(self) -> StancePlanV2:
        if (self.action == "clarify") != (self.claim_plan is None):
            raise ValueError("only clarification omits the atomic claim plan")
        return self


class SelectorStanceV2(SelectorAnswerV5):
    kind: Literal["stance"] = "stance"
    requested_direction: Literal["support", "refute"]


class StanceV2(QA5):
    task_kind = "stance"
    profile_id = "stance-v2"
    plan_model = StancePlanV2
    answer_model = SelectorStanceV2
    independent_plan_review = True
    role_policies: ClassVar[dict[str, str]] = {
        "plan": STANCE_PLANNER,
        "plan_review": STANCE_PLAN_REVIEWER,
        "author": AUTHOR + "\n" + STANCE_RULE + "\nReturn kind=stance and the current explicit direction. Return every reviewed planned claim ID exactly once, with statement exactly equal to its atomic_statement. Missing evidence changes status/limitations, never the proposition.",
        "review": REVIEWER + "\n" + STANCE_RULE + "\nThe atomic plan has an independent pre-retrieval review. Still verify the resulting claims and current requirement coverage yourself. Explicit unplanned requests are uncovered, not completed merely because they were listed.",
    }
    required_role_calls: ClassVar[dict[str, int]] = {"plan": 4, "plan_review": 3, "author": 2, "review": 1}
    revision_call_reserve = 4

    def prompt(self, state: dict, *, reviewer: bool = False, planner: bool = False) -> tuple[dict, dict]:
        prompt, manifest = super().prompt(state, reviewer=reviewer, planner=planner)
        task = self.graph.execution.snapshot()
        # Match the stance policy: optional preferences must not bias evidence
        # verdicts, and the manifest must not claim omitted prose was included.
        prompt["confirmed_user_memory_data"] = [item for item in prompt["confirmed_user_memory_data"]
            if item.get("kind") != "preference"]
        manifest["memory_refs"] = list(manifest["hard_memory_refs"])
        manifest["excluded"].append("optional_preferences_not_used_by_stance")
        prompt["stance_objective"] = task["brief"]["requested_direction"]
        prompt["objective_is_not_a_required_verdict"] = True
        plan = state["qa5"].get("plan")
        if plan and plan.get("claim_plan"):
            prompt["claim_plan"] = plan["claim_plan"]
            prompt["claim_plan_sha256"] = digest(plan["claim_plan"])
        return prompt, manifest

    def validate_plan(self, state: dict) -> dict:
        changed = super().validate_plan(state)
        data = changed["qa5"]
        if data["feedback"] or data["candidate"]["projected"] is None:
            return changed
        try:
            plan = self.plan_model.model_validate(data["plan"], strict=True)
            validate_plan(plan.claim_plan, self.graph.user_turns(), self.graph.execution.snapshot()["brief"]["requested_direction"])
        except ValueError:
            return self.failure(data, "plan", "stance2_atomic_plan_not_bound_to_current_user")
        return changed

    def review_plan(self, state: dict) -> dict:
        data = copy.deepcopy(state["qa5"])
        plan = self.plan_model.model_validate(data["plan"], strict=True)
        prompt, _ = self.prompt(state, reviewer=True)
        if data["target"] == "review_plan" and data["feedback"]:
            prompt.update(validation_feedback=data["feedback"], untrusted_previous_plan_review=data.get("stance_plan_review_raw", ""))
        raw, _ = self.send(state, "plan_review", PlanReview.model_json_schema(), prompt)
        data["stance_plan_review_raw"] = raw
        try:
            review = PlanReview.model_validate(strict_object(raw), strict=True)
        except ValidationError:
            return self.failure(data, "review_plan", "stance2_atomic_plan_review_schema_invalid", source_role="plan_review")
        except AgentError as error:
            if error.code not in JSON_CONTENT_ERRORS:
                raise
            return self.failure(data, "review_plan", error.code, source_role="plan_review")
        try:
            validate_plan_review(plan.claim_plan, review)
        except ValueError:
            return self.failure(data, "plan", "stance2_atomic_plan_independent_review_rejected",
                details={"independent_plan_review": review.model_dump()}, source_role="plan_review")
        key = self.key(data, "plan_review")
        with self.graph.execution.database.connect() as db:
            receipt = db.execute("""SELECT id,request_hash,status,usage,error_code FROM cq_app.agent_attempts
                WHERE task_id=%s AND kind='model' AND logical_key=%s""", (self.graph.execution.task_id, key)).fetchone()
        if not receipt or receipt["status"] != "settled" or receipt["usage"] is None or receipt["error_code"]:
            raise AgentError("stance2_atomic_plan_review_receipt_not_known")
        data.update(stance_plan_review=review.model_dump(), feedback=None,
            binding={**data["binding"], "plan_review_attempt_id": str(receipt["id"]),
                "plan_review_request_sha256": receipt["request_hash"], "claim_plan_sha256": digest(plan.claim_plan.model_dump())})
        return {"qa5": data}

    def retrieve(self, state: dict) -> dict:
        data = state["qa5"]
        plan = self.plan_model.model_validate(data["plan"], strict=True)
        if not data.get("stance_plan_review") or data["binding"].get("claim_plan_sha256") != digest(plan.claim_plan.model_dump()):
            raise AgentError("stance2_independent_atomic_plan_review_required_before_retrieval")
        validate_plan_review(plan.claim_plan, PlanReview.model_validate(data["stance_plan_review"], strict=True))
        validate_plan(plan.claim_plan, self.graph.user_turns(), self.graph.execution.snapshot()["brief"]["requested_direction"])
        return super().retrieve(state)

    def validate_author(self, state: dict) -> dict:
        changed = super().validate_author(state)
        if changed["qa5"]["feedback"]:
            return changed
        plan = self.plan_model.model_validate(state["qa5"]["plan"], strict=True)
        expected = {claim.id: claim.atomic_statement for claim in plan.claim_plan.claims}
        actual = {claim["id"]: claim["statement"] for claim in changed["answer"]["claims"]}
        direction = self.graph.execution.snapshot()["brief"]["requested_direction"]
        if (actual != expected or changed["answer"]["kind"] != "stance"
                or changed["answer"]["requested_direction"] != direction):
            return self.failure(changed["qa5"], "author", "stance2_author_changed_reviewed_plan_or_direction",
                details={"required_claims": expected, "required_direction": direction})
        return changed

    def validate_review(self, state: dict) -> dict:
        changed = super().validate_review(state)
        data = changed["qa5"]
        plan = self.plan_model.model_validate(data["plan"], strict=True)
        if data["accepted"] and plan.claim_plan.unplanned_requests and data["review"]["request_coverage"] == "complete":
            data.update(accepted=False, terminal=False, next_requirements=None)
            return self.failure(data, "review", "stance2_unplanned_requests_cannot_be_complete",
                details={"unplanned_request_count": len(plan.claim_plan.unplanned_requests)})
        return changed

    def finalize(self, state: dict) -> dict:
        changed = super().finalize(state)
        result = changed["result"]
        data = state["qa5"]
        plan = self.plan_model.model_validate(data["plan"], strict=True)
        gaps = ["未处理的用户要求：" + item.anchor.quote + "；" + item.reason for item in plan.claim_plan.unplanned_requests]
        insufficient = [claim for claim in result["claims"] if claim["status"] == "insufficient"]
        gaps += ["论断 " + claim["id"] + " 证据不足：" + "；".join(claim["limitations"]) for claim in insufficient]
        result.update(claim_plan=plan.claim_plan.model_dump(), plan_review=data["stance_plan_review"],
            claim_coverage=[{"claim_id": claim["id"], "state": "reviewed", "status": claim["status"],
                "evidence_ref_ids": list(dict.fromkeys(edge["ref_id"] for edge in claim["edges"]))} for claim in result["claims"]],
            coverage_gaps=gaps, stance_protocol="citequest-stance/2-typed-atomic-plan-Q-review")
        result["uncovered_requests"] = list(dict.fromkeys([*result["uncovered_requests"], *gaps]))
        if plan.claim_plan.unplanned_requests:
            result["request_coverage"] = "partial"
            result["completion_status"] = "partial_unplanned_stance_requests"
        elif insufficient:
            result["completion_status"] = "partial_stance_evidence_insufficient"
        return changed
