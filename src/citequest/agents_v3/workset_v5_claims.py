"""Workset5 directed validation/review child graph on the same Execution.

Every completed response settles through ResponsesProvider before validation.
Only local content failures enter the shared durable two-credit branch. Unknown
usage, incomplete transport, timeout, refusal, permission and version failures
escape; no SDK retry, second model client or new Task is created.
"""
from __future__ import annotations

import copy
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from citequest.research_v3.contracts import validate_literal, validate_review
from citequest.research_v3.quote_spans import QuoteProtocolError, QuoteRegistry
from citequest.research_v3.review_grid import GridProtocolError, ReviewGrid
from citequest.research_v3.typed_feedback import RepairLock, claim_objects, strict_object, validation_feedback
from citequest.research_v3.workset_v5_contracts import SelectorSummaryV5, SummaryV5

from .contracts import AgentError, canonical, digest
from .provider import output_text
from .workset_v5_payloads import payload, role_cap


class ClaimFlowState(TypedDict, total=False):
    job: dict
    raw: str
    candidate: dict | None
    review_raw: str
    review: dict | None
    accepted: bool
    feedback: dict | None
    lock: dict | None
    local_revision: int
    target: str
    terminal: bool
    outcome: dict
    audit: list[dict]


class DirectedClaimFlow:
    def __init__(self, owner, *, checkpointer=True):
        self.owner = owner
        self.graph = owner.graph
        builder = StateGraph(ClaimFlowState)
        for name in ("initialize", "author", "validate_author", "review", "validate_review", "reserve", "finish"):
            builder.add_node(name, self.graph.observed_node("workset5_claim_" + name, getattr(self, name)))
        builder.add_edge(START, "initialize")
        builder.add_conditional_edges("initialize", lambda state: "review" if state["job"].get("candidate") else "author", ["review", "author"])
        builder.add_edge("author", "validate_author")
        builder.add_conditional_edges("validate_author", lambda state: "finish" if state["terminal"] else "reserve" if state["feedback"] else "review", ["finish", "reserve", "review"])
        builder.add_edge("review", "validate_review")
        builder.add_conditional_edges("validate_review", lambda state: "finish" if state["terminal"] else "reserve", ["finish", "reserve"])
        builder.add_conditional_edges("reserve", lambda state: "finish" if state["terminal"] else state["target"], ["finish", "author", "review"])
        builder.add_edge("finish", END)
        self.workflow = builder.compile(checkpointer=checkpointer)

    def key(self, state, role):
        return str(self.graph.execution.command["id"]) + ":workset5:" + state["job"]["job_id"] + ":" + role + ":" + str(state["local_revision"])

    def scope(self, state):
        job = state["job"]
        candidates = [state["candidate"]] if state.get("candidate") else None
        return QuoteRegistry(job["evidence"]).scope(job["contexts"], candidates=candidates)

    def initialize(self, state):
        job = copy.deepcopy(state["job"])
        if job["role"] not in {"map", "reduce", "atomic_author"}:
            raise AgentError("workset_v5_generation_role_invalid")
        candidate = job.get("candidate")
        if candidate is not None:
            SummaryV5.model_validate(candidate, strict=True)
            claim_objects(candidate)
            if job.get("original_proposition_context") is None:
                job["original_proposition_context"] = {"source": "pre_review_candidate_not_a_factual_authority",
                    "claims": [{"id": claim["id"], "statement": claim["statement"], "status": claim["status"]} for claim in candidate["claims"]]}
        return {"job": job, "raw": "", "candidate": candidate, "review_raw": "", "review": None,
            "accepted": False, "feedback": None, "lock": None, "local_revision": 0,
            "target": "author", "terminal": False, "audit": []}

    def send(self, state, role, schema, data, *, reviewer=False):
        projection, manifest = self.owner.project(state["job"]["anchor"], reviewer=reviewer, evidence=state["job"]["evidence"])
        maximum = self.graph.execution.snapshot()["profile"]["request_bytes"]
        cap = role_cap(role)
        if reviewer and getattr(self.owner, "review_output_cap", None) is not None:
            cap = self.owner.review_output_cap
        if state["local_revision"]:
            cap = max(cap, 2048 if reviewer else 4096)
        request = payload(projection, role=role, schema=schema, task_data=data, maximum=maximum, cap_override=cap,
            lifecycle_policy=getattr(self.owner, "lifecycle_policy", None))
        key = self.key(state, "review" if reviewer else "author")
        task = self.graph.execution.snapshot()
        # Reconstruct exactly for settled replay; a free replay is not a new
        # call reservation. The provider still checks the full request hash.
        if self.graph.execution.settled_model_request(key) is None:
            if task["model_calls"] >= task["profile"]["model_calls"] or task["profile"]["output_tokens"] - task["output_tokens"] < cap:
                raise AgentError("workset_v5_remaining_role_budget_insufficient")
        response = self.graph.wait(self.graph.provider.complete(request, key))
        if any(content.get("type") == "refusal" for item in response.get("output", []) for content in item.get("content", [])):
            raise AgentError("model_refusal_not_repaired")
        return output_text(response), manifest

    def author_parts(self, state):
        scope = self.scope(state)
        data = {**copy.deepcopy(state["job"].get("task_data", {})), "untrusted_evidence_data": scope.contexts,
            "source_transport": scope.binding, "source_warnings": scope.warnings,
            "immutable_atomic_question": state["job"].get("atomic_question"),
            "original_proposition_context": state["job"].get("original_proposition_context")}
        if state["feedback"]:
            data.update(validation_feedback=state["feedback"], repair_contract=state["lock"],
                untrusted_previous_candidate=state["raw"] or canonical(scope.compact_candidate(state["candidate"])).decode())
        schema = scope.schema(SelectorSummaryV5)
        required = state["job"].get("required_claim_ids")
        if required:
            schema["properties"]["claims"]["minItems"] = len(required)
            schema["properties"]["claims"]["maxItems"] = len(required)
            schema["$defs"]["SelectorClaimV5"]["properties"]["id"]["enum"] = required
        return schema, data

    def preview_author(self, job):
        state = self.initialize({"job": job})
        schema, data = self.author_parts(state)
        projection, _ = self.owner.project(job["anchor"], evidence=job["evidence"])
        return payload(projection, role=job["role"], schema=schema, task_data=data,
            maximum=16 * 1024 * 1024, lifecycle_policy=getattr(self.owner, "lifecycle_policy", None))

    def author(self, state):
        schema, data = self.author_parts(state)
        raw, _ = self.send(state, state["job"]["role"], schema, data)
        return {"raw": raw}

    def validate_author(self, state):
        try:
            loose = strict_object(state["raw"])
            objects = claim_objects(loose)
        except AgentError as error:
            return self.reject(state, error.code)
        if state["job"].get("original_proposition_context") is None:
            state = {**state, "job": {**state["job"], "original_proposition_context": {
                "source": "first_model_candidate_not_a_factual_authority", "claims": [
                    {"id": identifier, "statement": claim.get("statement"), "status": claim.get("status")}
                    for identifier, claim in objects.items()]}}}
        required = state["job"].get("required_claim_ids")
        if required and set(objects) != set(required):
            return self.reject(state, "workset_v5_atomic_claim_identity_changed")
        if state["lock"]:
            lock = RepairLock(**state["lock"])
            try:
                lock.verify(loose)
            except AgentError as error:
                return self.reject(state, error.code)
        try:
            parsed = SelectorSummaryV5.model_validate(loose, strict=True)
            scope = self.scope(state)
            candidate = scope.expand(parsed, SummaryV5)
            validate_literal(candidate, state["job"]["evidence"])
            if state["job"].get("atomic_question"):
                expected_docs = set(state["job"]["atomic_question"]["document_ids"])
                by_ref = {item["ref_id"]: item for item in state["job"]["evidence"]}
                for claim in candidate.claims:
                    if claim.status != "insufficient" and {by_ref[edge.ref_id]["document_id"] for edge in claim.edges} != expected_docs:
                        return self.reject(state, "workset_v5_atomic_document_coverage_changed")
        except ValidationError as error:
            feedback = validation_feedback(error, loose)
            if not feedback["repairable"]:
                return self.reject(state, "workset_v5_unregistered_validation_rule")
            lock = state["lock"] or RepairLock.bind(loose, feedback["affected_claim_ids"]).__dict__
            return {"job": state["job"], "feedback": feedback, "lock": lock, "target": "author", "terminal": False,
                "audit": state["audit"] + [{"phase": "local_validation", "raw_sha256": digest(state["raw"]), "feedback": feedback}]}
        except QuoteProtocolError as error:
            if error.code != "quote_output_selector_not_issued":
                raise
            feedback = {"schema_version": "citequest-typed-feedback/1", "error_codes": ["quote_output_selector_not_issued"],
                "rule": "Choose only current issued Q IDs for the same proposition; do not invent source selectors or change the required claim set."}
            return {"job": state["job"], "feedback": feedback, "lock": state["lock"] or RepairLock.bind(loose, list(objects)).__dict__,
                "target": "author", "terminal": False,
                "audit": state["audit"] + [{"phase": "quote_validation", "raw_sha256": digest(state["raw"]), "error_code": error.code}]}
        return {"job": state["job"], "candidate": candidate.model_dump(), "feedback": None, "lock": None,
            "accepted": not state["job"]["review_required"], "terminal": not state["job"]["review_required"]}

    def review(self, state):
        scope = self.scope(state)
        grid = ReviewGrid(scope, state["candidate"])
        data = {"candidate": grid.candidate, "review_grid": grid.binding, "untrusted_evidence_data": scope.contexts,
            "source_transport": scope.binding, "source_warnings": scope.warnings,
            "immutable_atomic_question": state["job"].get("atomic_question"),
            "original_proposition_context": state["job"].get("original_proposition_context")}
        if state["job"].get("coverage_assignment") is not None:
            data["coverage_assignment"] = copy.deepcopy(state["job"]["coverage_assignment"])
        if state["target"] == "review" and state["feedback"]:
            data.update(validation_feedback=state["feedback"], untrusted_previous_review=state["review_raw"])
        role = "atomic_review" if state["job"].get("atomic_question") else "document_review"
        raw, _ = self.send(state, role, grid.schema(), data, reviewer=True)
        return {"review_raw": raw}

    def validate_review(self, state):
        scope = self.scope(state)
        grid = ReviewGrid(scope, state["candidate"])
        try:
            parsed = grid.parse(state["review_raw"])
            review = grid.canonical(parsed)
        except ValidationError as error:
            compact = scope.compact_candidate(state["candidate"])
            feedback = validation_feedback(error, compact)
            if not feedback["repairable"]:
                return self.reject(state, "workset_v5_unregistered_review_rule")
            return {"feedback": feedback, "target": "review", "terminal": False,
                "audit": state["audit"] + [{"phase": "review_format", "raw_sha256": digest(state["review_raw"]), "feedback": feedback}]}
        except GridProtocolError as error:
            if error.code not in {"review_grid_invalid_json", "review_grid_duplicate_property"}:
                raise
            return {"feedback": {"error_codes": [error.code], "rule": "Return exactly the required full fixed-key review grid. Do not alter any claim/Q identity or omit a cell."},
                "target": "review", "terminal": False,
                "audit": state["audit"] + [{"phase": "review_json", "raw_sha256": digest(state["review_raw"]), "error_code": error.code}]}
        candidate = SummaryV5.model_validate(state["candidate"], strict=True)
        try:
            validate_review(candidate, review)
        except ValueError:
            if review.source_instruction_injection_detected or review.requested_stance_bias_detected:
                return self.reject(state, "workset_v5_source_instruction_or_bias_rejected")
            expected = {"supported": "entailed", "contradicted": "contradicted", "mixed": "mixed", "insufficient": "insufficient"}
            affected = [row.claim_id for row in review.claims if row.verdict != expected[next(claim.status for claim in candidate.claims if claim.id == row.claim_id)] or not row.numeric_and_conditions_preserved]
            compact = scope.compact_candidate(state["candidate"])
            return {"review": review.model_dump(), "feedback": {"error_codes": ["workset_v5_semantic_review_rejected"],
                    "independent_review": parsed.model_dump(), "rule": "Revise only the rejected proposition using the current original evidence. Do not replace the original question or proposition with an easier unrelated fact."},
                "lock": RepairLock.bind(compact, affected or list(claim_objects(compact))).__dict__,
                "raw": canonical(compact).decode(), "target": "author", "terminal": False,
                "audit": state["audit"] + [{"phase": "semantic_review", "accepted": False, "raw_sha256": digest(state["review_raw"])}]}
        return {"review": review.model_dump(), "accepted": True, "terminal": True, "feedback": None,
            "audit": state["audit"] + [{"phase": "semantic_review", "accepted": True, "raw_sha256": digest(state["review_raw"])}]}

    def reserve(self, state):
        local = state["local_revision"] + 1
        key = str(self.graph.execution.command["id"]) + ":workset5:" + state["job"]["job_id"] + ":repair:" + str(local)
        codes = state["feedback"]["error_codes"]
        try:
            receipt = self.graph.execution.reserve_content_revision(key, phase="workset_v5_content_revision", error_codes=codes)
        except AgentError as error:
            if error.code != "content_revision_limit":
                raise
            return self.reject(state, "workset_v5_content_revision_limit")
        return {"local_revision": local, "audit": state["audit"] + [{"phase": "durable_revision", "ordinal": receipt["ordinal"], "error_codes": codes}]}

    def reject(self, state, code):
        return {"job": state["job"], "accepted": False, "terminal": True,
            "audit": state["audit"] + [{"phase": "quality_rejection", "error_code": code}]}

    def finish(self, state):
        return {"outcome": {"accepted": state["accepted"], "candidate": state["candidate"] if state["accepted"] else None,
            "review": state["review"], "audit": state["audit"], "job_id": state["job"]["job_id"],
            "original_atomic_question_sha256": digest(state["job"].get("atomic_question")),
            "source_reviewed": bool(state["accepted"] and state["job"]["review_required"])}}
