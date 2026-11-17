"""Research-v3 literal-Q literature child graph on the parent's Execution.

Known model content can consume the Task's shared two revision credits. Source,
permission, usage, transport and timeout failures escape without content retry.
The independent reviewer supplies every claim/Q opinion; the server only binds
identities and expands admitted original quotations.
"""
from __future__ import annotations

import copy
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from citequest.research_v3.contracts import validate_review
from citequest.research_v3.literature_grounding import (
    LITERATURE_AUTHOR_POLICY,
    LITERATURE_REVIEW_POLICY,
    LiteratureGrounding,
)
from citequest.research_v3.quote_spans import QuoteProtocolError, ResearchAnswerV3
from citequest.research_v3.review_grid import GridProtocolError
from citequest.research_v3.typed_feedback import RepairLock, claim_objects, strict_object, validation_feedback

from .contracts import AgentError, canonical, digest
from .provider import output_text


class LiteratureState(TypedDict, total=False):
    anchor: dict
    raw: str
    candidate: dict | None
    review_raw: str
    review: dict | None
    review_grid_opinions: dict | None
    original_propositions: dict
    feedback: dict | None
    lock: dict | None
    revision: int
    target: str
    source_model_key: str
    author_model_key: str
    review_model_key: str
    accepted: bool
    terminal: bool
    audit: list[dict]
    outcome: dict


class ResearchLiteratureFlow:
    def __init__(self, boundary: Any, *, checkpointer: Any = True) -> None:
        self.boundary, self.host = boundary, boundary.host
        builder = StateGraph(LiteratureState)
        for name in ("initialize", "author", "validate_author", "review", "validate_review", "reserve", "finish"):
            builder.add_node(name, self.observe(name, getattr(self, name)))
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "author")
        builder.add_edge("author", "validate_author")
        builder.add_conditional_edges("validate_author", lambda state: "finish" if state["terminal"] else "reserve" if state["feedback"] else "review", ["finish", "reserve", "review"])
        builder.add_edge("review", "validate_review")
        builder.add_conditional_edges("validate_review", lambda state: "finish" if state["terminal"] else "reserve", ["finish", "reserve"])
        builder.add_conditional_edges("reserve", lambda state: "finish" if state["terminal"] else state["target"], ["finish", "author", "review"])
        builder.add_edge("finish", END)
        self.workflow = builder.compile(checkpointer=checkpointer)

    def observe(self, phase: str, node: Any) -> Any:
        from langgraph.errors import GraphInterrupt

        def observed(state: dict) -> dict:
            name = "research_literature_v3_" + phase
            self.host.execution.progress(name, "started")
            try:
                self.boundary.require_current_input(state["anchor"])
                result = node(state)
            except GraphInterrupt:
                self.host.execution.progress(name, "waiting")
                raise
            except Exception:
                # IntakeChanged must escape to ResearchWorkflow's parent node;
                # translating it into an inner `next=intake` would skip this
                # child graph's edges and could continue with stale content.
                self.host.execution.progress(name, "failed")
                raise
            self.host.execution.progress(name, "completed")
            return result

        return observed

    def initialize(self, state: dict) -> dict:
        return {"anchor": copy.deepcopy(state["anchor"]), "raw": "", "candidate": None,
            "review_raw": "", "review": None, "review_grid_opinions": None, "original_propositions": {}, "feedback": None,
            "lock": None, "revision": 0, "target": "author", "source_model_key": "",
            "accepted": False, "terminal": False, "audit": []}

    def grounding(self, state: dict) -> LiteratureGrounding:
        return LiteratureGrounding(state["anchor"]["current_evidence"])

    def key(self, state: dict, role: str) -> str:
        question = self.boundary.question(state["anchor"])
        return self.boundary.model_key(state["anchor"], question["id"] + ":literature-v3:" + role + ":" + str(state["revision"]))

    def send(self, state: dict, *, reviewer: bool, schema: dict, data: dict) -> tuple[str, str]:
        role = "reviewer" if reviewer else "literature"
        policy = LITERATURE_REVIEW_POLICY if reviewer else LITERATURE_AUTHOR_POLICY
        key = self.key(state, "review" if reviewer else "author")
        request = self.boundary.payload(state["anchor"], role, schema, "citequest_research_literature_v3_review" if reviewer else "citequest_research_literature_v3", policy, data, logical_key=key)
        response = self.host.wait(self.host.provider.complete(request, key))
        if any(content.get("type") == "refusal" for item in response.get("output", []) for content in item.get("content", [])):
            raise AgentError("model_refusal_not_repaired")
        return output_text(response), key

    def data(self, state: dict) -> dict:
        grounding = self.grounding(state)
        return {"research_question": self.boundary.question(state["anchor"]),
            "untrusted_evidence_data": grounding.scope.contexts, "quote_scope": grounding.scope.binding,
            "source_warnings": grounding.scope.warnings, "original_proposition_context": state["original_propositions"]}

    def author(self, state: dict) -> dict:
        data = self.data(state)
        if state["feedback"]:
            data.update(validation_feedback=state["feedback"], repair_contract=RepairLock(**state["lock"]).contract(),
                        untrusted_previous_candidate=state["raw"])
        raw, key = self.send(state, reviewer=False, schema=self.grounding(state).author_schema(), data=data)
        return {"raw": raw, "source_model_key": key,
            **({"author_model_key": key} if self.boundary.audited_handoffs() else {})}

    def validate_author(self, state: dict) -> dict:
        try:
            loose = strict_object(state["raw"])
            objects = claim_objects(loose, maximum=3)
            if state["lock"]:
                RepairLock(**state["lock"]).verify(loose)
        except AgentError as error:
            return self.reject(state, error.code)
        original = state["original_propositions"] or {"source": "first_candidate_not_factual_authority",
            "research_question": self.boundary.question(state["anchor"]),
            "claims": [{"id": identifier, "statement": value.get("statement"), "status": value.get("status")} for identifier, value in objects.items()]}
        try:
            candidate = self.grounding(state).parse_author(state["raw"])
        except ValidationError as error:
            feedback = validation_feedback(error, loose, maximum=3)
            if not feedback["repairable"]:
                return self.reject(state, "research_literature_unregistered_validation_rule")
            lock = state["lock"] or RepairLock.bind(loose, feedback["affected_claim_ids"], maximum=3).__dict__
            return {"original_propositions": original, "feedback": feedback, "lock": lock, "target": "author",
                "audit": state["audit"] + [{"phase": "author_validation", "raw_sha256": digest(state["raw"]), "error_codes": feedback["error_codes"]}]}
        except QuoteProtocolError as error:
            if error.code != "quote_output_selector_not_issued":
                raise
            return {"original_propositions": original, "target": "author",
                "feedback": {"error_codes": [error.code], "rule": "Select only current issued Q IDs for this original proposition. Do not guess an identity or change the research question."},
                "lock": state["lock"] or RepairLock.bind(loose, list(objects), maximum=3).__dict__,
                "audit": state["audit"] + [{"phase": "quote_validation", "raw_sha256": digest(state["raw"]), "error_codes": [error.code]}]}
        return {"candidate": candidate.model_dump(), "feedback": None, "lock": None, "original_propositions": original}

    def review(self, state: dict) -> dict:
        grounding = self.grounding(state)
        candidate = ResearchAnswerV3.model_validate(state["candidate"], strict=True)
        grid = grounding.review_grid(candidate)
        data = {**self.data(state), "candidate": grid.candidate, "review_grid": grid.binding}
        if state["target"] == "review" and state["feedback"]:
            data.update(validation_feedback=state["feedback"], untrusted_previous_review=state["review_raw"])
        raw, key = self.send(state, reviewer=True, schema=grid.schema(), data=data)
        return {"review_raw": raw, "source_model_key": key,
            **({"review_model_key": key} if self.boundary.audited_handoffs() else {})}

    def validate_review(self, state: dict) -> dict:
        grounding = self.grounding(state)
        candidate = ResearchAnswerV3.model_validate(state["candidate"], strict=True)
        grid = grounding.review_grid(candidate)
        try:
            parsed = grid.parse(state["review_raw"])
            review = grid.canonical(parsed)
        except ValidationError as error:
            feedback = validation_feedback(error, grounding.scope.compact_candidate(state["candidate"]), maximum=3)
            if not feedback["repairable"]:
                return self.reject(state, "research_literature_unregistered_review_rule")
            return {"feedback": feedback, "target": "review",
                "audit": state["audit"] + [{"phase": "review_format", "raw_sha256": digest(state["review_raw"]), "error_codes": feedback["error_codes"]}]}
        except GridProtocolError as error:
            if error.code not in {"review_grid_invalid_json", "review_grid_duplicate_property"}:
                raise
            return {"feedback": {"error_codes": [error.code], "rule": "Return exactly the complete fixed-key claim/Q review object; do not omit or duplicate any opinion."},
                "target": "review", "audit": state["audit"] + [{"phase": "review_format", "raw_sha256": digest(state["review_raw"]), "error_codes": [error.code]}]}
        if review.source_instruction_injection_detected or review.requested_stance_bias_detected:
            return self.reject(state, "research_literature_source_instruction_or_bias_rejected")
        try:
            validate_review(candidate, review)
            if review.request_coverage != "complete":
                raise ValueError("research_question_coverage_partial")
        except ValueError:
            expected = {"supported": "entailed", "contradicted": "contradicted", "mixed": "mixed", "insufficient": "insufficient"}
            statuses = {claim.id: expected[claim.status] for claim in candidate.claims}
            affected = [row.claim_id for row in review.claims if row.verdict != statuses[row.claim_id] or not row.numeric_and_conditions_preserved]
            compact = grounding.scope.compact_candidate(state["candidate"])
            return {"review": review.model_dump(), "target": "author", "raw": canonical(compact).decode(),
                "feedback": {"error_codes": ["research_literature_semantic_review_rejected"], "independent_review": parsed.model_dump(),
                    "rule": "Revise only rejected propositions or explicit request gaps using current evidence; retain negation and conditions. Do not substitute an easier research question."},
                "lock": RepairLock.bind(compact, affected or list(claim_objects(compact, maximum=3)), maximum=3).__dict__,
                "audit": state["audit"] + [{"phase": "independent_review", "accepted": False, "raw_sha256": digest(state["review_raw"])}]}
        return {"review": review.model_dump(), "review_grid_opinions": parsed.model_dump(),
            "accepted": True, "terminal": True, "feedback": None,
            "audit": state["audit"] + [{"phase": "independent_review", "accepted": True, "raw_sha256": digest(state["review_raw"])}]}

    def reserve(self, state: dict) -> dict:
        revision = state["revision"] + 1
        question = self.boundary.question(state["anchor"])
        try:
            receipt = self.host.execution.reserve_content_revision(
                self.boundary.model_key(state["anchor"], question["id"] + ":literature-v3:credit:" + str(revision)),
                phase="research_literature_content_revision", error_codes=state["feedback"]["error_codes"],
                source_logical_key=state["source_model_key"])
        except AgentError as error:
            if error.code != "content_revision_limit":
                raise
            return self.reject(state, "research_shared_content_revision_limit")
        return {"revision": revision, "audit": state["audit"] + [{"phase": "durable_content_revision",
            "ordinal": receipt["ordinal"], "target": state["target"], "error_codes": state["feedback"]["error_codes"]}]}

    def reject(self, state: dict, code: str) -> dict:
        return {"accepted": False, "terminal": True,
            "audit": state["audit"] + [{"phase": "quality_rejection", "error_code": code}]}

    def finish(self, state: dict) -> dict:
        grounding = self.grounding(state)
        return {"outcome": {"accepted": state["accepted"], "candidate": state["candidate"] if state["accepted"] else None,
            "review": state["review"], "audit": state["audit"], "source_protocol": grounding.scope.binding,
            "review_grid": {"binding": grounding.review_grid(ResearchAnswerV3.model_validate(state["candidate"], strict=True)).binding,
                "opinions": state["review_grid_opinions"]} if state["accepted"] else None,
            "revision_credits_local": state["revision"], "independent_review_required": True,
            **({"model_sources": {state["author_model_key"]: "citequest_research_literature_v3",
                state["review_model_key"]: "citequest_research_literature_v3_review"}}
                if self.boundary.audited_handoffs() and state["accepted"] else {})}}

    def run(self, anchor: dict) -> dict:
        return self.workflow.invoke({"anchor": anchor}, durability="sync")["outcome"]
