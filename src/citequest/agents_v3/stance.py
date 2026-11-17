"""Bounded stance nodes registered in the single product Graph API graph."""
from __future__ import annotations

from citequest.research_v3.contracts import (Claim, ResearchAnswer, Review,
    SYSTEM_POLICY, REVIEW_POLICY, render_text, validate_literal, validate_review)
from .claim_plan import (ClaimPlan, PlanReview, PLAN_POLICY, PLAN_REVIEW_POLICY,
    validate_plan, validate_plan_review)
from .contracts import AgentError, canonical, digest
from .provider import MODEL, POLICY_MARKER, output_text
from .context import fit_context, wire_bytes
from .structured_output import strict_output_schema
import json


class StanceNodes:
    def __init__(self, graph) -> None:
        self.graph = graph

    def payload(self, prompt: dict, model, name: str, policy: str, cap: int) -> dict:
        task = self.graph.execution.snapshot()
        independent = name in {"citequest_plan_review", "citequest_stance_claim_review"}
        context, manifest = self.graph.projection(include_user_skill_body=not independent)
        prompt = {**prompt, "confirmed_user_memory_data": [item for item in context["confirmed_user_memory_data"] if item.get("kind") != "preference"],
            "context_manifest": manifest, "selected_user_skills": context["selected_user_skills"]}
        manifest["excluded"].append("optional_preferences_not_used_by_stance")
        remaining = task["profile"]["output_tokens"] - task["output_tokens"]
        if remaining <= 0: raise AgentError("output_budget_exhausted")
        payload = {"model": MODEL, "reasoning": {"effort": "none"}, "truncation": "disabled",
            "max_output_tokens": min(cap, task["profile"]["per_call_output"], remaining),
            "instructions": POLICY_MARKER + "\n" + policy + "\nUse the current user's language for conclusions; preserve source-language quotations.",
            "input": [{"role": "user", "content": canonical(prompt).decode()}],
            "text": {"format": {"type": "json_schema", "name": name, "schema": strict_output_schema(model.model_json_schema()), "strict": True}}}
        return fit_context(payload, task["profile"]["request_bytes"])

    def check_memories(self, state: dict) -> None:
        _, current = self.graph.projection()
        if current["hard_memory_refs"] != state["plan_context_manifest"]["hard_memory_refs"]:
            raise AgentError("hard_memory_changed_after_claim_plan")

    def complete(self, payload: dict, suffix: str):
        return self.graph.wait(self.graph.provider.complete(payload, str(self.graph.execution.command["id"]) + ":stance:" + suffix))

    def plan(self, state: dict) -> dict:
        task = self.graph.execution.snapshot()
        turns = self.graph.user_turns()
        prompt = {"canonical_brief": task["brief"], "user_turns": turns, "current_user_turn_id": turns[-1]["turn_id"],
            "context_rule": "Only persisted user inputs may anchor claims; previous answers are not premises."}
        payload = self.payload(prompt, ClaimPlan, "citequest_claim_plan", PLAN_POLICY, 1500)
        response = self.complete(payload, "plan")
        plan = ClaimPlan.model_validate_json(output_text(response), strict=True)
        validate_plan(plan, prompt["user_turns"], task["brief"]["requested_direction"])
        return {"claim_plan": plan.model_dump(), "claim_index": 0, "claim_results": [], "claim_reviews": [],
            "claim_coverage": [], "evidence": [], "stance_rejected": False, "answer": None,
            "plan_context_manifest": json.loads(payload["input"][0]["content"])["context_manifest"]}

    def review_plan(self, state: dict) -> dict:
        task = self.graph.execution.snapshot()
        self.check_memories(state)
        prompt = {"canonical_brief": task["brief"],
            "user_turns": self.graph.user_turns(), "current_user_turn_id": self.graph.user_turns()[-1]["turn_id"], "claim_plan": state["claim_plan"]}
        response = self.complete(self.payload(prompt, PlanReview, "citequest_plan_review", PLAN_REVIEW_POLICY, 1200), "plan-review")
        review = PlanReview.model_validate_json(output_text(response), strict=True)
        try:
            validate_plan_review(ClaimPlan.model_validate(state["claim_plan"], strict=True), review)
        except ValueError:
            raise AgentError("claim_plan_semantic_review_rejected") from None
        return {"plan_review": review.model_dump()}

    def route_next(self, state: dict) -> str:
        if state.get("stance_rejected") or state["claim_index"] >= len(state["claim_plan"]["claims"]): return "assemble_stance"
        task = self.graph.execution.snapshot()
        # Reserve room for both independent model operations and current-source checks.
        if (task["model_calls"] + 2 > task["profile"]["model_calls"] or
            task["tool_calls"] + 3 > task["profile"]["tool_calls"] or
            task["profile"]["output_tokens"] - task["output_tokens"] < 1400 or
            task["input_tokens"] >= task["profile"]["input_tokens"]):
            return "assemble_stance"
        return "retrieve_claim"

    def current(self, state: dict) -> dict:
        return state["claim_plan"]["claims"][state["claim_index"]]

    def prompt(self, state: dict, evidence: list[dict]) -> dict:
        self.check_memories(state)
        task = self.graph.execution.snapshot()
        return {"task_id": str(task["id"]), "brief_revision": task["revision"], "brief_sha256": digest(task["brief"]),
            "plan_sha256": digest(state["claim_plan"]), "claim": self.current(state),
            "constraints": task["brief"]["constraints"], "untrusted_evidence_data": evidence,
            "output_rule": "Return exactly this claim ID and atomic_statement. No new propositions. Missing evidence requires insufficient with a gap."}

    def retrieve(self, state: dict) -> dict:
        claim = self.current(state)
        result = self.graph.wait(self.graph.tools.call("search_evidence", {"query": claim["retrieval_question"], "limit": 1},
            logical_key=str(self.graph.execution.command["id"]) + ":stance:" + claim["id"] + ":search"))
        evidence = result.get("evidence", [])
        if not isinstance(evidence, list) or len(evidence) > 1: raise AgentError("stance_search_window_count_invalid")
        for item in evidence:
            current = self.graph.wait(self.graph.tools.call("read_evidence", {"ref_id": item["ref_id"]},
                logical_key=str(self.graph.execution.run_id) + ":stance:" + claim["id"] + ":recheck"))
            if current["source_sha256"] != item["source_sha256"] or current["unit_text_sha256"] != item["unit_text_sha256"]:
                raise AgentError("evidence_version_changed")
        prompt = self.prompt(state, evidence)
        maximum = self.graph.execution.snapshot()["profile"]["request_bytes"]
        try:
            writer = self.payload(prompt, Claim, "citequest_stance_claim", SYSTEM_POLICY, 900)
            # Independent review adds candidate JSON. This is admission, never text truncation.
            review = self.payload({**prompt, "candidate": {}}, Review, "citequest_stance_claim_review", REVIEW_POLICY, 500)
            skip = "request_wire_bound" if max(wire_bytes(writer), wire_bytes(review) + 6000) > maximum else None
        except AgentError as exc:
            if exc.code != "required_context_exceeds_wire_limit": raise
            skip = "request_wire_bound"
        existing = {item["ref_id"]: item for item in state["evidence"]}
        existing.update({item["ref_id"]: item for item in evidence})
        return {"current_claim_evidence": evidence, "claim_skip_reason": skip, "evidence": list(existing.values())}

    def route_retrieved(self, state: dict) -> str:
        return "skip_claim" if state.get("claim_skip_reason") else "write_claim"

    def skip(self, state: dict) -> dict:
        claim = self.current(state)
        return {"claim_index": state["claim_index"] + 1, "claim_coverage": state["claim_coverage"] + [{
            "claim_id": claim["id"], "state": "not_evaluated", "reason": state["claim_skip_reason"],
            "evidence_ref_ids": [item["ref_id"] for item in state["current_claim_evidence"]]}]}

    def write(self, state: dict) -> dict:
        planned = self.current(state)
        payload = self.payload(self.prompt(state, state["current_claim_evidence"]), Claim, "citequest_stance_claim", SYSTEM_POLICY, 900)
        response = self.complete(payload, planned["id"] + ":write")
        claim = Claim.model_validate_json(output_text(response), strict=True)
        if claim.id != planned["id"] or claim.statement != planned["atomic_statement"]:
            raise AgentError("candidate_changed_planned_claim")
        candidate = self.answer([claim], "insufficient")
        validate_literal(candidate, state["current_claim_evidence"])
        return {"current_claim_candidate": claim.model_dump()}

    def review(self, state: dict) -> dict:
        planned = self.current(state)
        prompt = {**self.prompt(state, state["current_claim_evidence"]), "candidate": state["current_claim_candidate"]}
        response = self.complete(self.payload(prompt, Review, "citequest_stance_claim_review", REVIEW_POLICY, 500), planned["id"] + ":review")
        review = Review.model_validate_json(output_text(response), strict=True)
        claim = Claim.model_validate(state["current_claim_candidate"], strict=True)
        accepted = True
        try:
            validate_review(self.answer([claim], "insufficient"), review)
        except ValueError:
            accepted = False
        return {"claim_index": state["claim_index"] + 1,
            "claim_results": state["claim_results"] + [claim.model_dump()],
            "claim_reviews": state["claim_reviews"] + [review.model_dump()],
            "stance_rejected": not accepted,
            "claim_coverage": state["claim_coverage"] + [{"claim_id": claim.id,
                "state": "reviewed" if accepted else "quality_rejected", "status": claim.status,
                "evidence_ref_ids": [item["ref_id"] for item in state["current_claim_evidence"]]}]}

    def answer(self, claims: list[Claim], evidence_status: str) -> ResearchAnswer:
        task = self.graph.execution.snapshot()
        return ResearchAnswer(kind="stance", requested_direction=task["brief"]["requested_direction"], claims=claims,
            gaps=[], retrieval_scope="Server-issued single literal window per evaluated planned claim; not exhaustive literature or document coverage.",
            evidence_status=evidence_status)

    def assemble(self, state: dict) -> dict:
        covered = {item["claim_id"] for item in state["claim_coverage"]}
        ledger = state["claim_coverage"] + [{"claim_id": claim["id"], "state": "not_evaluated", "reason":
            "prior_quality_rejection" if state.get("stance_rejected") else "remaining_same_Task_budget",
            "evidence_ref_ids": []} for claim in state["claim_plan"]["claims"] if claim["id"] not in covered]
        incomplete = bool(state["claim_plan"]["unplanned_requests"]) or any(item["state"] != "reviewed" or item.get("status") == "insufficient" for item in ledger)
        gaps = ["论断 " + item["claim_id"] + " 未评估：" + {
            "request_wire_bound": "完整证据请求超过本任务容量，未截断原文继续生成。",
            "remaining_same_Task_budget": "本任务剩余资源不足，未继续生成。",
            "prior_quality_rejection": "先前论断未通过独立核验，已停止后续生成。",
        }.get(item["reason"], item["reason"]) for item in ledger if item["state"] == "not_evaluated"]
        gaps += ["未处理的用户要求：" + item["anchor"]["quote"] + "；" + item["reason"]
                 for item in state["claim_plan"]["unplanned_requests"]]
        claims = [Claim.model_validate(item, strict=True) for item in state["claim_results"]]
        gaps += ["论断 " + claim.id + " 证据不足：" + "；".join(claim.limitations)
                 for claim in claims if claim.status == "insufficient"]
        answer = self.answer(claims, "insufficient" if incomplete else "supported") if claims else None
        _, manifest = self.graph.projection(evidence=state["evidence"])
        self.check_memories(state)
        for key in ("memory_refs", "hard_memory_refs"):
            manifest[key] = state["plan_context_manifest"][key]
        result = {**(answer.model_dump() if answer else {"kind": "stance", "requested_direction": self.graph.execution.snapshot()["brief"]["requested_direction"],
            "claims": [], "evidence_status": "insufficient"}), "text": render_text(answer) if answer else "",
            "gaps": gaps,
            "request_coverage": "partial" if incomplete else "complete", "uncovered_requests": gaps,
            "coverage_gaps": gaps, "citations": [{"ref_id": edge.ref_id, "quote": edge.quote} for claim in claims for edge in claim.edges],
            "claim_plan": state["claim_plan"], "plan_review": state["plan_review"], "claim_coverage": ledger,
            "review": {"claims": [item for review in state["claim_reviews"] for item in review["claims"]],
                "source_instruction_injection_detected": any(review["source_instruction_injection_detected"] for review in state["claim_reviews"]),
                "requested_stance_bias_detected": any(review["requested_stance_bias_detected"] for review in state["claim_reviews"])},
            "context_manifest": manifest,
            "source_scope": manifest["source_scope"],
            "completion_status": "partial_named_claim_coverage" if incomplete else "bounded_planned_claims_evaluated",
            "publication_status": "rejected" if state.get("stance_rejected") else ("validated_ai_review" if claims else "not_evaluated"),
            "quality": {"format": "validated", "citation_literal": "validated", "semantic_support": "per_claim_ai_review_not_human_calibrated"},
            "fixture": any(item.get("fixture") for item in state["evidence"])}
        return {"answer": answer.model_dump() if answer else None, "result": result}
