"""Typed current goals followed by the unchanged complete-source Workset5 core.

Both compiled child graphs inherit the official parent Saver. Goal interpretation
is independently reviewed first, but accepted goals are committed to the Task
checkpoint only together with a freshly source-authorized public answer.
"""
from __future__ import annotations

import copy
from typing import Annotated, Any, ClassVar

from langgraph.graph import END, START, StateGraph
from pydantic import Field, ValidationError

from citequest.research_v3.contracts import Strict
from citequest.research_v3.current_requirements import (
    RequirementDelta,
    RequirementReview,
    RequirementsCandidate,
    RequirementsState,
    UserTurn,
    accept_review,
    propose_delta,
)
from citequest.research_v3.typed_feedback import strict_object
from citequest.research_v3.workset_v5_contracts import (
    ComparisonPlanV5,
    ComparisonQuestionV5,
    GlobalCoverageV5,
)
from citequest.research_v3.workset_v5_planning import resource_plan

from .contracts import AgentError, digest
from .qa_v5 import COMMON, JSON_CONTENT_ERRORS, QA5, QA5State
from .requirements_context import authority_hash, projection
from .workset_v5 import Workset5

VERSION = "citequest-typed-workset/6"
RID = Annotated[str, Field(pattern=r"^R[1-9][0-9]{0,3}$")]


class Workset6State(QA5State, total=False):
    data: dict
    workset_requirements: dict


class RequirementsOnlyPlan(Strict):
    requirement_delta: RequirementDelta


class ComparisonQuestionV6(ComparisonQuestionV5):
    requirement_ids: list[RID] = Field(min_length=1, max_length=24)


class ComparisonPlanV6(ComparisonPlanV5):
    questions: list[ComparisonQuestionV6] = Field(min_length=1, max_length=12)
    unplanned_requirement_ids: list[RID] = Field(max_length=24)


class GlobalCoverageV6(GlobalCoverageV5):
    uncovered_requirement_ids: list[RID] = Field(max_length=24)


PLAN_POLICY = COMMON + """Role: current-requirement interpreter for selected-document
summary/comparison. Return exactly RequirementsOnlyPlan. Bind the supplied parent,
authority and current-turn hashes. Address every previous active ID exactly once
by retain or a targeted operation. Explicit current changes retire conflicting old
goals; preserve compatible conditions, focus, detail and output format requests.
New requirements must use supplied exact current input ranges, without summaries,
invented IDs, source permissions or budget changes. Do not infer that a short
follow-up withdraws unrelated conditions. Ambiguity requires needs_clarification
and a concise question, never a guessed mutation. This role does not retrieve,
summarize sources or assert research facts. All raw older turns are audit only.
"""
REVIEW_POLICY = COMMON + """Role: independent current-requirement reviewer.
Return exactly RequirementReview. Audit the candidate against the exact current
raw turn, previous accepted requirements and every proposed retirement. Emit
exactly the supplied required review IDs with actual independent opinions. Reject
omitted detail, negation/condition changes, resurrected retired topics, unsupported
withdrawals and scope expansion. The generator's Skill body and old draft are not
authority. An ambiguous user instruction requires needs_clarification, not an
accepted guess. This role checks interpretation, never factual source support.
"""


def reviewed_goal_context(value: dict, task: dict, current_turn: dict) -> tuple[RequirementsState, UserTurn]:
    """Revalidate a private checkpoint DTO; a model never supplies this wrapper."""
    base = RequirementsState.model_validate(value["base"], strict=True)
    candidate = RequirementsCandidate.model_validate(value["candidate"], strict=True)
    review = RequirementReview.model_validate(value["review"], strict=True)
    turn = UserTurn.model_validate(current_turn, strict=True)
    if base.authority_sha256 != authority_hash(task) or candidate.user_turn != turn:
        raise AgentError("workset_v6_current_requirements_or_turn_changed")
    accepted = accept_review(candidate, review, current_state=base)
    if accepted.model_dump() != value["reviewed_state"]:
        raise AgentError("workset_v6_reviewed_requirement_state_changed")
    return accepted, turn


class WorksetRequirementsGate(QA5):
    profile_id = "workset-v6"
    state_type = Workset6State
    requirements_only = True
    phase_prefix = "workset6_goals_"
    plan_model = RequirementsOnlyPlan
    role_policies: ClassVar[dict[str, str]] = {"plan": PLAN_POLICY, "review": REVIEW_POLICY}
    required_role_calls: ClassVar[dict[str, int]] = {"plan": 2, "review": 1}
    revision_call_reserve = 2

    def __init__(self, graph: Any, *, checkpointer: Any = True) -> None:
        self.task_kind = graph.execution.snapshot()["kind"]
        super().__init__(graph, checkpointer=checkpointer)

    def initialize(self, state: dict) -> dict:
        task = self.graph.execution.snapshot()
        if task["kind"] not in {"summary", "compare"}:
            raise AgentError("workset_v6_summary_or_compare_required")
        count = len(task["brief"]["selected_papers"])
        # A one-group-per-document lower bound catches an impossible complete
        # protocol before paid goal interpretation. Actual groups/input/wire fit
        # are admitted after reading every source, before any source author.
        if not 1 <= count <= 10 or (task["kind"] == "compare" and count < 2):
            raise AgentError("workset_v5_explicit_source_count_invalid")
        remaining_credits = 2 - self.graph.execution.content_revision_count()
        minimum = resource_plan(documents=count, windows=count, groups=count,
            comparison_units=12 if task["kind"] == "compare" else 0,
            source_parts=count, revision_credits=0, kind=task["kind"],
            review_output_cap=getattr(self, "review_output_cap", None))
        needed_models = minimum["required_models"] + 2 + 2 * remaining_credits
        needed_output = minimum["reserved_output_tokens"] + (2 + 2 * remaining_credits) * 4096
        settled = self.graph.execution.settled_model_request(str(self.graph.execution.command["id"]) + ":qa5:plan:0")
        if settled is None and (needed_models > task["profile"]["model_calls"] - task["model_calls"]
                or needed_output > task["profile"]["output_tokens"] - task["output_tokens"]):
            raise AgentError("workset_v6_minimum_complete_protocol_budget_unavailable")
        value = super().initialize(state)
        value["workset_requirements"] = {}
        return value

    def receipt(self, data: dict, role: str) -> dict:
        key = self.key(data, role)
        with self.graph.execution.database.connect() as db:
            row = db.execute("SELECT id,request_hash,status,usage,error_code FROM cq_app.agent_attempts WHERE task_id=%s AND kind='model' AND logical_key=%s",
                (self.graph.execution.task_id, key)).fetchone()
        if not row or row["status"] != "settled" or row["usage"] is None or row["error_code"]:
            raise AgentError("workset_v6_independent_goal_receipt_not_known")
        return {"attempt_id": str(row["id"]), "request_sha256": row["request_hash"], "logical_key": key}

    def validate_plan(self, state: dict) -> dict:
        data = copy.deepcopy(state["qa5"])
        try:
            plan = self.plan_output_model(state).model_validate(strict_object(data["plan_raw"]), strict=True)
            candidate = propose_delta(self.accepted(state), UserTurn.model_validate(data["turn"], strict=True), plan.requirement_delta)
        except ValidationError:
            return self.failure(data, "plan", "workset_v6_typed_requirement_plan_invalid")
        except AgentError as error:
            if error.code not in JSON_CONTENT_ERRORS and not error.code.startswith("requirements_"):
                raise
            return self.failure(data, "plan", error.code)
        data.update(plan=plan.model_dump(), candidate=candidate.model_dump(), feedback=None,
            binding={"planner": self.receipt(data, "plan"), "candidate_sha256": digest(candidate.model_dump())})
        return {"qa5": data}

    def review(self, state: dict) -> dict:
        data = copy.deepcopy(state["qa5"])
        prompt, _ = self.prompt(state, reviewer=True)
        if data["target"] == "review" and data["feedback"]:
            prompt.update(validation_feedback=data["feedback"], untrusted_previous_review=data["review_raw"])
        raw, _ = self.send(state, "review", self.goal_review_model(state).model_json_schema(), prompt)
        data["review_raw"] = raw
        return {"qa5": data}

    def validate_review(self, state: dict) -> dict:
        data = copy.deepcopy(state["qa5"])
        try:
            review = self.goal_review_model(state).model_validate(strict_object(data["review_raw"]), strict=True)
        except ValidationError:
            return self.failure(data, "review", "workset_v6_requirement_review_schema_invalid")
        except AgentError as error:
            if error.code not in JSON_CONTENT_ERRORS:
                raise
            return self.failure(data, "review", error.code)
        receipt = self.receipt(data, "review")
        if receipt["attempt_id"] == data["binding"]["planner"]["attempt_id"]:
            raise AgentError("workset_v6_goal_review_not_independent")
        candidate = RequirementsCandidate.model_validate(data["candidate"], strict=True)
        try:
            accepted = accept_review(candidate, review, current_state=self.accepted(state))
        except AgentError as error:
            if error.code == "requirements_parent_changed_before_commit":
                raise
            # Only a correctly bound, complete identity grid may ask the user.
            if (error.code == "requirements_projection_not_accepted" and review.decision == "needs_clarification"):
                clarification = "请明确本次修改中要保留和撤销的条件，以及希望重点回答的问题。"
                data.update(candidate={**data["candidate"], "projected": None, "clarification": clarification}, goal_clarification=True)
                return {"qa5": data}
            return self.failure(data, "plan", error.code, details={"independent_goal_review": review.model_dump()}, source_role="review")
        data.update(review=review.model_dump(), next_requirements=accepted.model_dump(), terminal=True,
            accepted=True, feedback=None, binding={**data["binding"], "reviewer": receipt})
        return {"qa5": data}

    def goal_review_model(self, state: dict) -> type[RequirementReview]:
        return RequirementReview

    def clarify(self, state: dict) -> dict:
        value = super().clarify(state)
        value["qa5"].pop("goal_clarification", None)
        return value

    def finalize(self, state: dict) -> dict:
        data = state["qa5"]
        if not data["accepted"]:
            raise AgentError("workset_v6_current_requirement_review_unresolved")
        self.prompt(state, reviewer=True)  # Current Brief/Skill/Memory guard.
        if digest(self.accepted(state).model_dump()) != data["candidate"]["base_sha256"]:
            raise AgentError("requirements_parent_changed_before_commit")
        return {"workset_requirements": {"base": self.accepted(state).model_dump(), "candidate": data["candidate"],
            "review": data["review"], "reviewed_state": data["next_requirements"], "binding": data["binding"]}}


class Workset6Sources(Workset5):
    profile_id = "workset-v6"
    state_type = Workset6State
    typed_requirements = True
    comparison_plan_model = ComparisonPlanV6
    global_coverage_model = GlobalCoverageV6
    protocol_version = VERSION

    def admit(self, state: dict) -> dict:
        value = super().admit(state)
        if value["data"].get("admission"):
            value["data"]["admission"].update(requirements_protocol=self.protocol_version,
                normal_goal_model_calls=2, goal_calls_already_charged_to_same_task=True,
                goal_and_source_revisions_share_two_credits=True,
                source_admission_uses_remaining_after_goal_calls=True)
        return value

    def prepare_anchor(self, state: dict, anchor: dict) -> dict:
        value = copy.deepcopy(state["workset_requirements"])
        reviewed_goal_context(value, self.graph.execution.snapshot(), self.graph.user_turns()[-1])
        anchor["typed_requirements"] = value
        return anchor

    def project(self, anchor: dict, *, reviewer: bool = False, evidence: list[dict] | None = None) -> tuple[dict, dict]:
        base, manifest = self.graph.projection(evidence=evidence, include_user_skill_body=not reviewer)
        for key in ("hard_memory_refs", "skill_refs", "brief_sha256"):
            if manifest.get(key) != anchor.get(key):
                raise AgentError("workset_v5_current_context_changed")
        turns = self.graph.user_turns()
        if [{"id": item["turn_id"], "sha256": digest(item["text"])} for item in turns] != anchor["required_user_turn_refs"]:
            raise AgentError("workset_v5_required_turns_changed")
        accepted, turn = reviewed_goal_context(anchor["typed_requirements"], self.graph.execution.snapshot(), turns[-1])
        base.pop("untrusted_evidence_data", None)
        base["confirmed_user_memory_data"] = [item for item in base["confirmed_user_memory_data"] if item.get("kind") != "preference"]
        base["context_manifest"]["memory_refs"] = list(base["context_manifest"]["hard_memory_refs"])
        prompt, manifest = projection(base, self.graph.execution.snapshot(), accepted, turn, reviewer=reviewer)
        manifest.update(workset_protocol=self.protocol_version, requirements_goal_review="independently_reviewed_pending_answer_commit",
            requirements_candidate_sha256=digest(anchor["typed_requirements"]["candidate"]))
        return prompt, manifest

    def requirement_ids(self, data: dict) -> set[str]:
        state, _ = reviewed_goal_context(data["anchor"]["typed_requirements"], self.graph.execution.snapshot(), self.graph.user_turns()[-1])
        return {item.id for item in state.requirements if item.state == "active"}

    def validate_plan_coverage(self, data: dict, plan: ComparisonPlanV6, requirements: set[str]) -> None:
        covered = {identifier for question in plan.questions for identifier in question.requirement_ids}
        omitted = set(plan.unplanned_requirement_ids)
        if (covered | omitted != requirements or covered & omitted or len(omitted) != len(plan.unplanned_requirement_ids)
                or any(len(set(question.requirement_ids)) != len(question.requirement_ids) for question in plan.questions)):
            raise AgentError("workset_v5_plan_outside_server_scope")

    def validate_global_coverage(self, data: dict, review: GlobalCoverageV6) -> None:
        requirements = self.requirement_ids(data)
        omitted = set(review.uncovered_requirement_ids)
        documents = {item["document"]["document_id"] for item in data["documents"]}
        claims = {item["id"] for item in data["comparisons"]} | {claim["id"] for item in data["documents"]
            if item["summary"] for claim in item["summary"]["claims"]}
        if (not omitted <= requirements or len(omitted) != len(review.uncovered_requirement_ids)
                or not set(review.missing_document_ids) <= documents or not set(review.conflicting_claim_ids) <= claims
                or (review.request_coverage == "complete" and (omitted or review.missing_document_ids or review.conflicting_claim_ids or review.gaps))):
            raise AgentError("workset_v6_global_coverage_outside_current_requirements")


class Workset6:
    gate_class = WorksetRequirementsGate
    sources_class = Workset6Sources
    protocol_version = VERSION

    def __init__(self, graph: Any, *, checkpointer: Any = True) -> None:
        self.graph = graph
        self.gate = self.gate_class(graph, checkpointer=checkpointer)
        self.sources = self.sources_class(graph, checkpointer=checkpointer)
        builder = StateGraph(Workset6State)
        builder.add_node("current_requirements", self.gate.workflow)
        builder.add_node("complete_sources", self.sources.workflow)
        builder.add_node("commit", graph.observed_node("workset6_commit", self.commit))
        builder.add_edge(START, "current_requirements")
        builder.add_edge("current_requirements", "complete_sources")
        builder.add_edge("complete_sources", "commit")
        builder.add_edge("commit", END)
        self.workflow = builder.compile(checkpointer=checkpointer)

    def commit(self, state: dict) -> dict:
        value = state["workset_requirements"]
        if digest(state["requirements_state"]) != digest(value["base"]):
            raise AgentError("requirements_parent_changed_before_commit")
        accepted, _ = reviewed_goal_context(value, self.graph.execution.snapshot(), self.graph.user_turns()[-1])
        # Fresh source/ACL verification and literal checks already ran in
        # Workset5.publish; repeat the cheap current-context guard at commit.
        self.graph.wait(self.graph.tools.check_scope())
        self.sources.project(state["data"]["anchor"], reviewer=True, evidence=state["evidence"])
        result = copy.deepcopy(state["result"])
        committed = result.get("publication_status") == "validated_ai_review"
        result.update(workset_protocol=self.protocol_version, requirements_committed=committed,
            requirements_revision=accepted.revision if committed else state["requirements_state"]["revision"],
            requirements_sha256=digest(accepted.model_dump()) if committed else digest(state["requirements_state"]),
            requirements_candidate_sha256=digest(value["candidate"]), requirements_planning_provenance=value["binding"])
        return {"result": result, **({"requirements_state": accepted.model_dump()} if committed else {})}
