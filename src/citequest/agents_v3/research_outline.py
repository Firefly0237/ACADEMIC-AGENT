"""Source-free coordinator/reviewer child on the research Task's Saver and ledger."""
from __future__ import annotations

import copy
import json
from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import Field, ValidationError, create_model

from citequest.research_v3.brief import QUESTIONS, BriefField
from citequest.research_v3.research_plan import PLANNER_POLICY
from citequest.research_v3.typed_feedback import strict_object

from .contracts import AgentError, Strict, digest
from .provider import output_text
from .research_capacity import research_v5_capacity, reviewed_outline_capacity
from .research_workflow import IntakeChanged, Outline, ResearchState


class QuestionOpinion(Strict):
    verdict: Literal["passed", "needs_revision"]
    reason: str = Field(min_length=1, max_length=400)


class FieldCoverage(Strict):
    state: Literal["represented", "explicit_unknown", "needs_revision"]
    question_ids: list[str] = Field(max_length=8)
    reason: str = Field(min_length=1, max_length=400)


class OutlineReview(Strict):
    brief_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    question_opinions: dict[str, QuestionOpinion]
    field_coverage: dict[str, FieldCoverage]
    missing_user_fields: list[BriefField] = Field(max_length=16)
    unsupported_user_fact_assumptions: bool
    decision: Literal["accepted", "needs_revision", "needs_input"]
    reason: str = Field(min_length=1, max_length=600)


REVIEW_POLICY = """Role: independent research outline reviewer before retrieval.
Review only the current versioned Brief, confirmed facts, explicit unknowns and
coordinator outline. No literature has been retrieved or verified in this stage.
Do not make factual novelty, source support, completed experiment or data-access
claims. Judge each supplied RQ and each fixed user field exactly once. Preserve
the user's scope, resource limits and proposed-not-executed status. A proposed
experimental design can be formulated from constraints; it is not a missing user
fact merely because the coordinator must design it. Structural omissions, wrong
IDs, bad decomposition, unsupported assumptions or failure to cover information
already supplied require needs_revision, not a question back to the user.
Use needs_input only for a genuine decision/fact absent from confirmed and
explicit_unknown fields, naming those exact missing fields. An explicit unknown
is not permission to fabricate it. Missing source evidence belongs to the later
literature stage. Retain independent negative findings; do not majority vote.
For accepted, every question must pass, each confirmed field must be represented,
explicit unknowns must be preserved, and no uncovered request or assumption may
remain. Bind the exact Brief and outline hashes. Give concise public reasons.
"""


class ResearchOutlineFlow:
    def __init__(self, boundary, *, checkpointer=True):
        self.boundary, self.host = boundary, boundary.host
        graph = StateGraph(ResearchState)
        for name in ("initialize", "write", "validate_outline", "review", "validate_review", "reserve", "finish"):
            graph.add_node(name, self.observe(name, getattr(self, name)))
        graph.add_edge(START, "initialize")
        graph.add_conditional_edges("initialize", self.route, ["write", "finish"])
        graph.add_edge("write", "validate_outline")
        graph.add_conditional_edges("validate_outline", self.route, ["review", "reserve", "finish"])
        graph.add_edge("review", "validate_review")
        graph.add_conditional_edges("validate_review", self.route, ["reserve", "finish"])
        graph.add_conditional_edges("reserve", self.route, ["write", "review", "finish"])
        graph.add_edge("finish", END)
        self.workflow = graph.compile(checkpointer=checkpointer)

    @staticmethod
    def route(state):
        return state["outline_control"]["route"]

    def observe(self, phase, node):
        def observed(state):
            self.host.execution.progress("research_outline_" + phase, "started")
            try:
                if not (phase == "finish" and state.get("outline_control", {}).get("restart_intake")):
                    self.boundary.require_current_input(state)
                value = node(state)
            except IntakeChanged:
                value = {"outline_control": {**state.get("outline_control", {}),
                    "route": "finish", "restart_intake": True}}
                if phase == "finish":
                    value["next"] = "intake"
            self.host.execution.progress("research_outline_" + phase, "completed")
            return value
        return observed

    def initialize(self, state):
        revisions = max(0, 2 - self.host.execution.content_revision_count())
        capacity = reviewed_outline_capacity(self.boundary.task(), remaining_revisions=revisions)
        return {"outline_control": {"revision": 0, "candidate": None, "review": None,
            "raw": "", "review_raw": "", "feedback": None, "source_key": "",
            "capacity": capacity, "accepted": False, "missing_fields": [], "audit": [],
            "route": "write" if capacity["admissible_model_tool_and_output_scope_pairs"] else "finish",
            "failure": None if capacity["admissible_model_tool_and_output_scope_pairs"] else "complete_research_and_school_capacity_unavailable"}}

    def key(self, state, role):
        return self.boundary.model_key(state, "outline-v5:" + role + ":" + str(state["outline_control"]["revision"]))

    def send(self, state, role, model, policy, data):
        key = self.key(state, role)
        payload = self.boundary.payload(state, role, model, "citequest_research_outline_" + role,
            policy, data, logical_key=key)
        response = self.host.wait(self.host.provider.complete(payload, key))
        if any(part.get("type") == "refusal" for item in response.get("output", []) for part in item.get("content", [])):
            raise AgentError("model_refusal_not_repaired")
        return output_text(response), key

    def write(self, state):
        control = copy.deepcopy(state["outline_control"])
        key = self.key(state, "planner")
        settled = self.host.execution.settled_model_request(key)
        data = {"planning_capacity": control["capacity"], "source_contents_included": False,
            "planning_rule": "Cover all supplied objectives with the smallest admitted RQ/experiment decomposition. User facts and proposed designs are distinct."}
        if control["feedback"]:
            data.update(validation_feedback=control["feedback"], untrusted_previous_outline=control["candidate"])
        if settled is not None:
            prior = json.loads(settled["request"]["input"][0]["content"])
            data["planning_capacity"] = prior["planning_capacity"]
        raw, key = self.send(state, "planner", self.boundary.identity_model(Outline,
            brief_sha256=state["intake"]["brief_sha256"]), PLANNER_POLICY, data)
        control.update(raw=raw, source_key=key, route="review")
        if self.boundary.audited_handoffs():
            control["author_model_key"] = key
        return {"outline_control": control}

    def feedback(self, state, code, *, target, details=None):
        control = copy.deepcopy(state["outline_control"])
        control.update(feedback={"error_codes": [code], "details": details or {}}, target=target, route="reserve")
        return {"outline_control": control}

    def validate_outline(self, state):
        try:
            value = self.boundary.identity_model(Outline, brief_sha256=state["intake"]["brief_sha256"]).model_validate(
                strict_object(state["outline_control"]["raw"]), strict=True)
        except (ValidationError, AgentError) as error:
            if isinstance(error, AgentError) and not error.code.startswith("typed_feedback_"):
                raise
            return self.feedback(state, "research_outline_structure_invalid", target="write")
        if any(not set(item.user_fields) <= set(state["intake"]["user_confirmed"]) for item in value.questions):
            return self.feedback(state, "research_outline_uses_unconfirmed_user_fields", target="write")
        control = copy.deepcopy(state["outline_control"])
        control.update(candidate=value.model_dump(mode="json"), feedback=None, route="review")
        return {"outline_control": control}

    def review_model(self, state):
        candidate = state["outline_control"]["candidate"]
        questions = create_model("OutlineQuestions_" + digest(candidate)[:12], __base__=Strict,
            **{item["id"]: (QuestionOpinion, ...) for item in candidate["questions"]})
        fields = sorted(set(state["intake"]["user_confirmed"]) | set(state["intake"]["explicit_unknowns"]))
        coverage = create_model("OutlineFields_" + digest(candidate)[:12], __base__=Strict,
            **{field: (FieldCoverage, ...) for field in fields})
        bound = self.boundary.identity_model(OutlineReview, brief_sha256=state["intake"]["brief_sha256"],
            content_sha256=digest(candidate))
        return create_model("ReviewedOutline_" + digest(candidate)[:16], __base__=bound,
            question_opinions=(questions, ...), field_coverage=(coverage, ...))

    def review(self, state):
        control = copy.deepcopy(state["outline_control"])
        data = {"candidate": control["candidate"], "content_sha256": digest(control["candidate"]),
            "source_contents_included": False, "review_scope": "source_free_intent_structure_and_user_fact_boundary"}
        if control["feedback"]:
            data["validation_feedback"] = control["feedback"]
        raw, key = self.send(state, "reviewer", self.review_model(state), REVIEW_POLICY, data)
        control.update(review_raw=raw, source_key=key, route="finish")
        if self.boundary.audited_handoffs():
            control["review_model_key"] = key
        return {"outline_control": control}

    def validate_review(self, state):
        try:
            review = self.review_model(state).model_validate(strict_object(state["outline_control"]["review_raw"]), strict=True).model_dump(mode="json")
        except (ValidationError, AgentError) as error:
            if isinstance(error, AgentError) and not error.code.startswith("typed_feedback_"):
                raise
            return self.feedback(state, "research_outline_review_structure_invalid", target="review")
        control = copy.deepcopy(state["outline_control"])
        candidate = control["candidate"]
        ids = {item["id"] for item in candidate["questions"]}
        known = set(state["intake"]["user_confirmed"]) | set(state["intake"]["explicit_unknowns"])
        if (len(review["missing_user_fields"]) != len(set(review["missing_user_fields"]))
                or set(review["missing_user_fields"]) & known
                or any(not set(row["question_ids"]) <= ids for row in review["field_coverage"].values())):
            return self.feedback(state, "research_outline_review_fact_or_identity_invalid", target="review")
        control["review"] = review
        if review["decision"] == "needs_input" and review["missing_user_fields"]:
            control.update(missing_fields=review["missing_user_fields"], route="finish")
            return {"outline_control": control}
        complete = (review["decision"] == "accepted" and not review["missing_user_fields"]
            and not review["unsupported_user_fact_assumptions"] and not candidate["uncovered_user_requests"]
            and all(row["verdict"] == "passed" for row in review["question_opinions"].values())
            and all(row["state"] == ("explicit_unknown" if field in state["intake"]["explicit_unknowns"] else "represented")
                and (row["question_ids"] or field in state["intake"]["explicit_unknowns"])
                for field, row in review["field_coverage"].items()))
        if not complete:
            revised = {**state, "outline_control": control}
            return self.feedback(revised, "research_outline_independent_review_rejected", target="write", details=review)
        capacity = research_v5_capacity(self.boundary.task(), len(candidate["questions"]),
            sum(len(item["experiment_ids"]) for item in candidate["questions"]),
            remaining_revisions=max(0, 2 - self.host.execution.content_revision_count()))
        control.update(accepted=capacity["admitted"], admission=capacity, route="finish",
            failure=None if capacity["admitted"] else "complete_research_and_school_capacity_unavailable")
        return {"outline_control": control}

    def reserve(self, state):
        control = copy.deepcopy(state["outline_control"])
        revision = control["revision"] + 1
        try:
            receipt = self.host.execution.reserve_content_revision(
                self.boundary.model_key(state, "outline-v5:credit:" + str(revision)),
                phase="research_outline_content_revision", error_codes=control["feedback"]["error_codes"],
                source_logical_key=control["source_key"])
        except AgentError as error:
            if error.code != "content_revision_limit":
                raise
            control.update(route="finish", failure="research_outline_shared_revision_limit")
            return {"outline_control": control}
        control.update(revision=revision, route=control["target"],
            audit=[*control["audit"], {"ordinal": receipt["ordinal"], "target": control["target"],
                "error_codes": control["feedback"]["error_codes"]}])
        return {"outline_control": control}

    def finish(self, state):
        control = state["outline_control"]
        if control.get("restart_intake"):
            return {"next": "intake"}
        if control.get("missing_fields"):
            return {"next": "await_input", "waiting": {"kind": "outline_request_gaps",
                "fields": control["missing_fields"], "questions": [{"field": field, "question": QUESTIONS[field]}
                    for field in control["missing_fields"]], "basis": "independently_reviewed_missing_user_facts"}}
        if control["accepted"]:
            sources = ({control["author_model_key"]: "citequest_research_outline_planner"}
                if self.boundary.audited_handoffs() else {})
            handoffs = self.boundary.handoff(state, "planner", control["candidate"], sources=sources,
                parents={"brief": state["intake"]["brief_sha256"]})
            sources = ({control["review_model_key"]: "citequest_research_outline_reviewer"}
                if self.boundary.audited_handoffs() else {})
            handoffs.append(self.boundary.handoff_entry("reviewer", control["review"], sources=sources,
                parents={"outline": digest(control["candidate"])}))
            return {"outline": control["candidate"], "outline_review": control["review"],
                "research_capacity": control["admission"], "handoffs": handoffs, "next": "literature_search"}
        capacity_limited = control.get("failure") == "complete_research_and_school_capacity_unavailable"
        result = {"status": "partial" if capacity_limited else "failed",
            "publication_status": "not_evaluated" if capacity_limited else "rejected",
            "reasons": [control.get("failure") or "research_outline_not_accepted"],
            "capacity": control.get("admission", control["capacity"]), "school_format_ready": False}
        return {"next": END, "result": self.boundary.project_result(state, result)}
