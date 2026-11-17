"""Grouped full-source mapping and independently reviewed atomic comparisons.

Official nested LangGraph state inherits the parent Saver namespace. This
candidate uses the existing Execution/ResponsesProvider/MCPTools/Memory only.
Raw turns remain explicit audit inputs until the common typed requirements
projection is integrated; this implementation does not close multi-turn gates.
"""
from __future__ import annotations

import copy
import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from citequest.research_v3.contracts import validate_literal
from citequest.research_v3.quote_spans import QuoteRegistry, source_warnings
from citequest.research_v3.typed_feedback import strict_object
from citequest.research_v3.workset_contracts import DocumentCoverage, WorksetManifest
from citequest.research_v3.workset_v5_contracts import (ComparisonPlanV5, DocumentResultV5, GlobalCoverageV5,
    PlanReviewV5, SelectorSummaryV5, SummaryV5, WorksetAnswerV5)
from citequest.research_v3.workset_v5_planning import group_windows, raw_quote_pack, resource_plan

from .context import wire_bytes
from .contracts import AgentError, digest
from .provider import output_text
from .workset_v5_claims import DirectedClaimFlow
from .workset_v5_payloads import payload


class Workset5State(TypedDict, total=False):
    command_id: str
    question: str
    data: dict
    evidence: list[dict]
    result: dict


def document_coverage_ledger(documents: list[dict]) -> list[dict]:
    """Give the coverage reviewer the complete public coverage declarations.

    Literal traversal is measured by the source reader, not inferred from
    generated claims. Preserve limitations/version alongside the claims; this
    ledger is not raw evidence and cannot establish semantic support.
    """
    return [{"document_id": document["document"]["document_id"],
             "review_status": document["review_status"],
             "source_coverage": copy.deepcopy(document["coverage"]),
             "declared_gaps": copy.deepcopy(document["summary"]["gaps"]) if document["summary"] else [],
             "evidence_status": document["summary"]["evidence_status"] if document["summary"] else "insufficient",
             "claims": [{key: claim[key] for key in ("id", "statement", "status", "limitations")}
                        for claim in document["summary"]["claims"]] if document["summary"] else []}
            for document in documents]


class Workset5:
    profile_id = "workset-v5"
    state_type = Workset5State
    typed_requirements = False
    comparison_plan_model = ComparisonPlanV5
    global_coverage_model = GlobalCoverageV5
    lifecycle_policy = None

    def __init__(self, graph, *, checkpointer=True):
        self.graph = graph
        self.claims = DirectedClaimFlow(self)
        builder = StateGraph(self.state_type)
        for name in ("intake", "read_sources", "admit", "map_group", "reduce_group", "advance_group",
                "review_document", "finish_document", "plan", "review_plan", "atomic_claim", "coverage", "publish"):
            builder.add_node(name, graph.observed_node("workset5_" + name, getattr(self, name)))
        builder.add_edge(START, "intake")
        builder.add_conditional_edges("intake", lambda s: "publish" if s["data"]["stopped"] else "read_sources", ["publish", "read_sources"])
        builder.add_edge("read_sources", "admit")
        builder.add_conditional_edges("admit", lambda s: "publish" if s["data"]["stopped"] else "map_group", ["publish", "map_group"])
        builder.add_conditional_edges("map_group", lambda s: "finish_document" if s["data"]["document_failed"] else "reduce_group" if s["data"]["document_summary"] else "advance_group", ["finish_document", "reduce_group", "advance_group"])
        builder.add_conditional_edges("reduce_group", lambda s: "finish_document" if s["data"]["document_failed"] else "advance_group", ["finish_document", "advance_group"])
        builder.add_conditional_edges("advance_group", lambda s: "map_group" if s["data"]["group_index"] < len(s["data"]["groups"][s["data"]["document_index"]]) else "review_document", ["map_group", "review_document"])
        builder.add_edge("review_document", "finish_document")
        builder.add_conditional_edges("finish_document", self.route_document, ["map_group", "plan", "coverage", "publish"])
        builder.add_conditional_edges("plan", lambda s: "coverage" if s["data"]["comparison_stopped"] else "review_plan", ["coverage", "review_plan"])
        builder.add_conditional_edges("review_plan", lambda s: "coverage" if s["data"]["comparison_stopped"] else "atomic_claim", ["coverage", "atomic_claim"])
        builder.add_conditional_edges("atomic_claim", lambda s: "atomic_claim" if s["data"]["unit_index"] < len(s["data"]["questions"]) else "coverage", ["atomic_claim", "coverage"])
        builder.add_edge("coverage", "publish")
        builder.add_edge("publish", END)
        self.workflow = builder.compile(checkpointer=checkpointer)

    def invoke(self, parent_state):
        if self.graph.execution.snapshot()["profile"]["profile_id"] != self.profile_id:
            raise AgentError("workset_v5_profile_required")
        value = self.workflow.invoke({"command_id": str(self.graph.execution.command["id"]), "question": parent_state["question"]}, durability="sync")
        return {"result": value["result"], "evidence": value["data"]["evidence"]}

    def tool(self, name, arguments, suffix):
        if name not in {"plan_workset", "read_source_window", "verify_evidence_batch"}:
            raise AgentError("workset_v5_tool_not_allowed")
        return self.graph.wait(self.graph.tools.call(name, arguments, logical_key=str(self.graph.execution.command["id"]) + ":workset5:" + suffix))

    def project(self, anchor, *, reviewer=False, evidence=None):
        projection, manifest = self.graph.projection(evidence=evidence, include_user_skill_body=not reviewer)
        for key in ("hard_memory_refs", "skill_refs", "brief_sha256"):
            if manifest.get(key) != anchor.get(key):
                raise AgentError("workset_v5_current_context_changed")
        turns = self.graph.user_turns()
        refs = [{"id": item["turn_id"], "sha256": digest(item["text"])} for item in turns]
        if refs != anchor["required_user_turn_refs"]:
            raise AgentError("workset_v5_required_turns_changed")
        projection.pop("user_turns", None)
        projection.pop("prior_turns", None)
        projection.pop("untrusted_evidence_data", None)  # Only source-reading roles add actual raw contexts.
        projection["required_user_turns"] = turns
        projection["raw_turn_reference_ids"] = [{"id": "R" + str(index + 1), "turn_id": turn["turn_id"],
            "meaning": "raw_user_instruction_reference_not_typed_current_requirement"} for index, turn in enumerate(turns)]
        projection["confirmed_user_memory_data"] = [item for item in projection["confirmed_user_memory_data"] if item.get("kind") != "preference"]
        manifest["memory_refs"] = list(manifest["hard_memory_refs"])
        manifest.update(required_user_turn_refs=refs, requirements_policy="raw_turn_history_not_typed_requirements")
        projection["context_manifest"] = manifest
        return projection, manifest

    def intake(self, state):
        task = self.graph.execution.snapshot()
        if task["profile"]["profile_id"] != self.profile_id:
            raise AgentError("workset_v5_profile_required")
        selected = task["brief"]["selected_papers"]
        if not 1 <= len(selected) <= 10 or task["kind"] == "compare" and len(selected) < 2:
            raise AgentError("workset_v5_explicit_source_count_invalid")
        _, anchor = self.graph.projection(include_user_skill_body=False)
        turns = self.graph.user_turns()
        anchor["required_user_turn_refs"] = [{"id": item["turn_id"], "sha256": digest(item["text"])} for item in turns]
        anchor = self.prepare_anchor(state, anchor)
        manifest = WorksetManifest.model_validate(self.tool("plan_workset", {}, "manifest")["manifest"], strict=True)
        if [item.selection.model_dump() for item in manifest.documents] != selected:
            raise AgentError("workset_manifest_differs_from_current_selection")
        windows = sum(len(doc.windows) for doc in manifest.documents)
        parts = sum(window.source_parts for doc in manifest.documents for window in doc.windows)
        reasons = []
        if windows + 1 + (parts + 63) // 64 > task["profile"]["tool_calls"] - task["tool_calls"]:
            reasons.append("full_original_window_read_exceeds_remaining_tool_budget")
        if not all(doc.windows for doc in manifest.documents):
            reasons.append("some_selected_sources_have_no_available_text")
        if not turns or not self.typed_requirements and len(turns) > 12:
            reasons.append("raw_turn_capacity_needs_typed_requirements_or_user_clarification")
        return {"data": {"manifest": manifest.model_dump(), "anchor": anchor, "start_counters": {key: task[key] for key in ("tool_calls", "model_calls", "output_tokens")},
            "evidence": [], "windows": {}, "groups": [], "documents": [], "document_index": 0, "group_index": 0,
            "document_summary": None, "current_map": None, "mapped_windows": [], "document_failed": False,
            "document_review": None, "document_request_coverage": [], "reviews": [], "questions": [], "unit_index": 0, "comparisons": [], "comparison_coverage": [],
            "comparison_stopped": False, "routes": {}, "global_review": None, "reasons": reasons, "stopped": bool(reasons)}}

    def prepare_anchor(self, state, anchor):
        return anchor

    def requirement_ids(self, data):
        return {"R" + str(i + 1) for i in range(len(data["anchor"]["required_user_turn_refs"]))}

    def validate_plan_coverage(self, data, plan, requirements):
        pass

    def validate_global_coverage(self, data, review):
        pass

    def manifest(self, data):
        return WorksetManifest.model_validate(data["manifest"], strict=True)

    def read_sources(self, state):
        data = copy.deepcopy(state["data"])
        # Reuse the existing tested source descriptor/version/hash/member gate.
        legacy = {"workset_manifest": data["manifest"], "evidence": []}
        for index, document in enumerate(self.manifest(data).documents):
            for ordinal, window in enumerate(document.windows):
                legacy.update(workset_document_index=index, workset_window_index=ordinal)
                changed = self.graph.workset.read_window(legacy)
                legacy.update(changed)
                data["windows"][window.window_id] = {"window_id": window.window_id,
                    "document_id": document.document_id, "evidence": copy.deepcopy(changed["workset_current_window"])}
        data["evidence"] = legacy["evidence"]
        return {"data": data}

    def admit(self, state):
        data = copy.deepcopy(state["data"])
        try:
            for document in self.manifest(data).documents:
                ids = [window.window_id for window in document.windows]
                original = {key: data["windows"][key] for key in ids}
                def encoder(responses):
                    items = [item for response in responses for item in response["evidence"]]
                    job = self.make_job(data, job_id="preview", role="map", evidence=items, contexts=items,
                        task_data={"document_id": document.document_id, "original_window_ids": [response["window_id"] for response in responses]})
                    request = self.claims.preview_author(job)
                    return wire_bytes(request)
                groups = group_windows(ids, original, encoder, maximum=self.graph.execution.snapshot()["profile"]["request_bytes"])
                data["groups"].append([list(group.window_ids) for group in groups])
        except AgentError as error:
            if error.code not in {"workset_group_single_window_exceeds_request_capacity", "required_context_exceeds_wire_limit"}:
                raise
            data["reasons"].append(error.code)
            data["stopped"] = True
            return {"data": data}
        task = self.graph.execution.snapshot()
        manifest = self.manifest(data)
        plan = resource_plan(documents=len(manifest.documents), windows=sum(len(doc.windows) for doc in manifest.documents),
            groups=sum(map(len, data["groups"])), comparison_units=12 if task["kind"] == "compare" else 0,
            source_parts=len(data["evidence"]), revision_credits=2 - self.graph.execution.content_revision_count(), kind=task["kind"],
            review_output_cap=getattr(self, "review_output_cap", None))
        data["admission"] = plan
        used_tools = task["tool_calls"] - data["start_counters"]["tool_calls"]
        if (not plan["capacity_admitted"] or plan["required_models"] > task["profile"]["model_calls"] - task["model_calls"]
                or max(0, plan["required_tools"] - used_tools) > task["profile"]["tool_calls"] - task["tool_calls"]
                or plan["reserved_output_tokens"] > task["profile"]["output_tokens"] - task["output_tokens"]):
            data["reasons"].append("complete_planned_protocol_exceeds_same_task_remaining_resources")
            data["stopped"] = True
        return {"data": data}

    def document(self, data):
        return self.manifest(data).documents[data["document_index"]]

    def evidence(self, data, document=None):
        return [item for item in data["evidence"] if document is None or item["document_id"] == document]

    def make_job(self, data, *, job_id, role, evidence, contexts, candidate=None, review_required=False, task_data=None, atomic=None):
        return {"job_id": job_id, "role": role, "evidence": evidence, "contexts": contexts, "candidate": candidate,
            "review_required": review_required, "anchor": data["anchor"], "task_data": task_data or {},
            "atomic_question": atomic, "original_proposition_context": copy.deepcopy(atomic),
            "required_claim_ids": [atomic["claim_id"]] if atomic else None}

    def claim_job(self, data, **kwargs):
        job = self.make_job(data, **kwargs)
        try:
            return self.claims.workflow.invoke({"job": job}, durability="sync")["outcome"]
        except AgentError as error:
            if error.code not in {"required_context_exceeds_wire_limit", "workset_v5_remaining_role_budget_insufficient"}:
                raise
            # This is a known local non-dispatch/content boundary, never a
            # transport, usage, ACL or source-version exception.
            return {"job_id": job["job_id"], "accepted": False, "candidate": None, "review": None,
                "source_reviewed": False, "audit": [{"phase": "local_admission", "error_code": error.code}]}

    def map_group(self, state):
        data = copy.deepcopy(state["data"])
        document = self.document(data)
        ids = data["groups"][data["document_index"]][data["group_index"]]
        evidence = [item for key in ids for item in data["windows"][key]["evidence"]]
        outcome = self.claim_job(data, job_id=document.document_id + ":map:" + str(data["group_index"]), role="map",
            evidence=evidence, contexts=evidence, task_data={"document_id": document.document_id, "original_window_ids": ids})
        data["current_map"] = outcome["candidate"]
        data["document_failed"] = not outcome["accepted"]
        data["reviews"].append(outcome)
        if outcome["accepted"]:
            data["mapped_windows"] += ids
        return {"data": data}

    def reduce_group(self, state):
        data = copy.deepcopy(state["data"])
        document = self.document(data)
        evidence = self.evidence(data, document.document_id)
        scope = QuoteRegistry(evidence).scope(evidence, candidates=[data["document_summary"], data["current_map"]])
        outcome = self.claim_job(data, job_id=document.document_id + ":reduce:" + str(data["group_index"]), role="reduce",
            evidence=evidence, contexts=evidence, task_data={"document_id": document.document_id,
                "mapped_notes": [scope.compact_candidate(item) for item in (data["document_summary"], data["current_map"])]})
        data["current_map"] = outcome["candidate"]
        data["document_failed"] = not outcome["accepted"]
        data["reviews"].append(outcome)
        return {"data": data}

    def advance_group(self, state):
        data = copy.deepcopy(state["data"])
        data["document_summary"] = data["current_map"]
        data["group_index"] += 1
        return {"data": data}

    def review_document(self, state):
        data = copy.deepcopy(state["data"])
        document = self.document(data)
        evidence = self.evidence(data, document.document_id)
        outcome = self.claim_job(data, job_id=document.document_id + ":final_review", role="reduce", evidence=evidence,
            contexts=evidence, candidate=data["document_summary"], review_required=True, task_data={"document_id": document.document_id})
        data["document_summary"] = outcome["candidate"]
        data["document_failed"] = not outcome["accepted"]
        data["document_review"] = outcome["review"]
        data["reviews"].append(outcome)
        return {"data": data}

    def finish_document(self, state):
        data = copy.deepcopy(state["data"])
        document = self.document(data)
        ids = data["mapped_windows"]
        covered = sum(len(item["text"]) for key in ids for item in data["windows"][key]["evidence"])
        complete = len(ids) == len(document.windows) and covered == document.available_codepoints
        coverage = DocumentCoverage(document_id=document.document_id, source_version_sha256=document.source_version_sha256,
            status="complete_available_text" if complete else "partial" if ids else "insufficient", total_windows=len(document.windows),
            mapped_windows=len(ids), mapped_window_ids=ids, covered_codepoints=covered, available_codepoints=document.available_codepoints,
            source_coverage_kind=document.coverage_kind, reasons=["Available literal text is not proof of complete PDF/image extraction."])
        data["documents"].append(DocumentResultV5(document=document, coverage=coverage,
            summary=SummaryV5.model_validate(data["document_summary"], strict=True) if not data["document_failed"] else None,
            review_status="quality_rejected" if data["document_failed"] else "ai_review_validated").model_dump())
        review = data.get("document_review")
        data["document_request_coverage"].append({"document_id": document.document_id,
            "status": review.get("request_coverage", "partial") if review else "not_assessed",
            "uncovered_requests": review.get("uncovered_requests", []) if review else ["No independently reviewed document answer was produced."]})
        data["document_index"] += 1
        data.update(group_index=0, document_summary=None, current_map=None, mapped_windows=[], document_failed=False, document_review=None)
        return {"data": data}

    def route_document(self, state):
        data = state["data"]
        if data["document_index"] < len(self.manifest(data).documents):
            return "map_group"
        if not any(doc["summary"] for doc in data["documents"]):
            return "publish"
        return "plan" if self.graph.execution.snapshot()["kind"] == "compare" else "coverage"

    def model(self, data, role, schema, task_data, suffix):
        projection, _ = self.project(data["anchor"], reviewer=role in {"plan_review", "global_coverage"}, evidence=data["evidence"])
        request = payload(projection, role=role, schema=schema, task_data=task_data,
            maximum=self.graph.execution.snapshot()["profile"]["request_bytes"], lifecycle_policy=self.lifecycle_policy)
        if self.lifecycle_policy:
            data["last_wire_context_manifest"] = json.loads(request["input"][0]["content"])["context_manifest"]
        key = str(self.graph.execution.command["id"]) + ":workset5:" + suffix
        response = self.graph.wait(self.graph.provider.complete(request, key))
        if any(content.get("type") == "refusal" for item in response.get("output", []) for content in item.get("content", [])):
            raise AgentError("model_refusal_not_repaired")
        return output_text(response)

    def plan(self, state):
        data = copy.deepcopy(state["data"])
        cards, routes = [], {}
        for document in data["documents"]:
            if document["summary"] is None:
                continue
            hints = []
            for claim in document["summary"]["claims"]:
                identifiers = []
                for edge in claim["edges"]:
                    identifier = "E" + str(len(routes) + 1)
                    routes[identifier] = copy.deepcopy(edge)
                    identifiers.append(identifier)
                hints.append({"claim_id": claim["id"], "statement_hint": claim["statement"][:240], "hint_truncated": len(claim["statement"]) > 240,
                    "statement_sha256": digest(claim["statement"]), "evidence_route_ids": identifiers, "hint_is_not_raw_evidence": True})
            cards.append({"document_id": document["document"]["document_id"], "routing_hints": hints})
        data["routes"] = routes
        try:
            raw = self.model(data, "plan", self.comparison_plan_model.model_json_schema(), {"routing_cards_not_evidence": cards}, "comparison_plan")
            plan = self.comparison_plan_model.model_validate(strict_object(raw), strict=True)
            allowed_docs = {card["document_id"] for card in cards}
            requirements = self.requirement_ids(data)
            self.validate_plan_coverage(data, plan, requirements)
            for index, question in enumerate(plan.questions):
                value = question.model_dump()
                if (not set(question.document_ids) <= allowed_docs or not set(question.evidence_route_ids) <= routes.keys()
                        or not set(question.requirement_ids) <= requirements):
                    raise AgentError("workset_v5_plan_outside_server_scope")
                data["questions"].append({"claim_id": "C" + str(index + 1), **value})
            data["plan"] = plan.model_dump()
            if plan.unplanned_requirement_ids:
                data["reasons"].append("comparison_plan_explicitly_leaves_requirements_unplanned")
            if {identifier for item in data["questions"] for identifier in item["document_ids"]} != allowed_docs:
                data["reasons"].append("comparison_plan_does_not_cover_all_selected_reviewed_documents")
            data["routing_cards"] = cards
        except (ValidationError, AgentError) as error:
            if isinstance(error, AgentError) and error.code not in {"workset_v5_plan_outside_server_scope", "typed_feedback_initial_json_unreadable", "typed_feedback_duplicate_json_property", "required_context_exceeds_wire_limit"}:
                raise
            data["comparison_stopped"] = True
            data["reasons"].append("comparison_plan_invalid_or_exceeds_capacity")
        return {"data": data}

    def review_plan(self, state):
        data = copy.deepcopy(state["data"])
        try:
            raw = self.model(data, "plan_review", PlanReviewV5.model_json_schema(), {"candidate_question_plan": data["plan"],
                "routing_cards_not_evidence": data["routing_cards"]}, "comparison_plan_review")
            review = PlanReviewV5.model_validate(strict_object(raw), strict=True)
            if not (review.all_current_requirements_addressed and review.all_routes_within_current_scope and review.all_questions_atomic) or review.omissions:
                data["comparison_stopped"] = True
                data["reasons"].append("comparison_plan_independent_review_rejected")
            data["plan_review"] = review.model_dump()
            # Preflight every actual raw pack before the first comparison writer.
            registry = QuoteRegistry(data["evidence"])
            for question in data["questions"]:
                edges = [data["routes"][identifier] for identifier in question["evidence_route_ids"]]
                scope, _ = raw_quote_pack(registry, edges)
                actual_docs = {registry.sources[edge["ref_id"]]["document_id"] for edge in edges}
                if actual_docs != set(question["document_ids"]):
                    raise AgentError("workset_v5_plan_document_route_mismatch")
                projection, _ = self.project(data["anchor"])
                payload(projection, role="atomic_author", schema=scope.schema(SelectorSummaryV5), task_data={
                    "immutable_atomic_question": question, "untrusted_evidence_data": scope.contexts,
                    "source_transport": scope.binding, "source_warnings": scope.warnings},
                    maximum=self.graph.execution.snapshot()["profile"]["request_bytes"])
        except (ValidationError, AgentError) as error:
            if isinstance(error, AgentError) and error.code not in {"workset_v5_plan_document_route_mismatch", "workset_atomic_duplicate_evidence_locator",
                    "workset_atomic_comparison_requires_distinct_documents", "typed_feedback_initial_json_unreadable", "typed_feedback_duplicate_json_property", "required_context_exceeds_wire_limit"}:
                raise
            data["comparison_stopped"] = True
            data["reasons"].append("comparison_plan_or_raw_pack_preflight_rejected")
        return {"data": data}

    def verify_refs(self, data, refs, suffix):
        expected = {item["ref_id"]: item for item in data["evidence"]}
        for start in range(0, len(refs), 64):
            batch = refs[start:start + 64]
            current = self.tool("verify_evidence_batch", {"ref_ids": batch}, suffix + ":" + str(start // 64))
            if current.get("valid") is not True or {row["ref_id"] for row in current.get("pins", [])} != set(batch):
                raise AgentError("workset_v5_current_source_verification_failed")
            for pin in current["pins"]:
                if any(pin.get(field) != expected[pin["ref_id"]].get(field) for field in ("document_id", "source_version_sha256", "source_sha256", "unit_text_sha256")):
                    raise AgentError("workset_v5_current_source_pin_changed")

    def atomic_claim(self, state):
        data = copy.deepcopy(state["data"])
        question = data["questions"][data["unit_index"]]
        registry = QuoteRegistry(data["evidence"])
        scope, parents = raw_quote_pack(registry, [data["routes"][identifier] for identifier in question["evidence_route_ids"]])
        self.verify_refs(data, parents, "atomic:" + question["claim_id"] + ":verify")
        outcome = self.claim_job(data, job_id="atomic:" + question["claim_id"], role="atomic_author", evidence=data["evidence"],
            contexts=[group["context"] for group in scope.raw_groups], review_required=True, atomic=question)
        data["reviews"].append(outcome)
        if outcome["accepted"]:
            data["comparisons"] += outcome["candidate"]["claims"]
        data["comparison_coverage"].append({"claim_id": question["claim_id"], "question": question["question"],
            "requirement_ids": question["requirement_ids"], "document_ids": question["document_ids"], "accepted": outcome["accepted"],
            "request_coverage": outcome["review"].get("request_coverage", "partial") if outcome.get("review") else "not_assessed"})
        data["unit_index"] += 1
        return {"data": data}

    def coverage(self, state):
        data = copy.deepcopy(state["data"])
        ledger = document_coverage_ledger(data["documents"])
        try:
            comparison_ledger = [{key: claim[key] for key in ("id", "statement", "status", "limitations")}
                | {"source_conditions": [{"ref_id": edge["ref_id"], "conditions": edge["conditions"],
                    "locator_sha256": digest(edge["quote_locator"])} for edge in claim["edges"]]} for claim in data["comparisons"]]
            raw = self.model(data, "global_coverage", self.global_coverage_model.model_json_schema(), {"reviewed_document_ledger": ledger,
                "source_reviewed_comparison_condition_ledger": comparison_ledger, "comparison_question_coverage": data["comparison_coverage"]}, "global_coverage")
            review = self.global_coverage_model.model_validate(strict_object(raw), strict=True)
            self.validate_global_coverage(data, review)
            data["global_review"] = review.model_dump()
        except (ValidationError, AgentError) as error:
            if isinstance(error, AgentError) and error.code not in {"typed_feedback_initial_json_unreadable", "typed_feedback_duplicate_json_property", "required_context_exceeds_wire_limit", "workset_v6_global_coverage_outside_current_requirements"}:
                raise
            data["reasons"].append("global_coverage_unassessed_invalid_or_exceeds_capacity")
        return {"data": data}

    def publish(self, state):
        data = copy.deepcopy(state["data"])
        manifest = self.manifest(data)
        documents = [DocumentResultV5.model_validate(doc, strict=True) for doc in data["documents"]]
        for document in manifest.documents[len(documents):]:
            documents.append(DocumentResultV5(document=document, summary=None, review_status="not_assessed", coverage=DocumentCoverage(
                document_id=document.document_id, source_version_sha256=document.source_version_sha256, status="not_processed",
                total_windows=len(document.windows), mapped_windows=0, mapped_window_ids=[], covered_codepoints=0,
                available_codepoints=document.available_codepoints, source_coverage_kind=document.coverage_kind, reasons=data["reasons"])))
        claims = [claim for doc in documents if doc.summary for claim in doc.summary.claims]
        comparison = [copy.deepcopy(claim) for claim in data["comparisons"]]
        global_review = data["global_review"]
        complete = (not data["reasons"] and bool(global_review) and global_review["request_coverage"] == "complete"
            and not any(global_review[key] for key in ("uncovered_requirement_ids", "missing_document_ids", "conflicting_claim_ids", "gaps"))
            and all(doc.summary and doc.review_status == "ai_review_validated" and doc.coverage.status == "complete_available_text" for doc in documents)
            and all(row["status"] == "complete" for row in data["document_request_coverage"]))
        if self.graph.execution.snapshot()["kind"] == "compare":
            complete = complete and bool(data["questions"]) and not data["comparison_stopped"] and all(row["accepted"] and row["request_coverage"] == "complete" for row in data["comparison_coverage"]) and len(data["comparison_coverage"]) == len(data["questions"])
        quality_failed_without_public = not claims and not comparison and bool(data["reviews"])
        answer = WorksetAnswerV5(kind=self.graph.execution.snapshot()["kind"], documents=documents, comparisons=comparison,
            gaps=data["reasons"][:30], evidence_status="supported" if complete else "insufficient",
            completion="failed" if quality_failed_without_public else "complete_available_scope" if complete else "partial", manifest_id=manifest.manifest_id)
        if data["evidence"]:
            self.verify_refs(data, [item["ref_id"] for item in data["evidence"]], "publish")
        self.graph.wait(self.graph.tools.check_scope())
        _, context = self.project(data["anchor"], reviewer=True, evidence=data["evidence"])
        public = claims + list(answer.comparisons)
        for doc in documents:
            if doc.summary:
                validate_literal(doc.summary, data["evidence"])
        for claim in answer.comparisons:
            validate_literal(SummaryV5(claims=[claim], gaps=[], evidence_status="supported" if claim.status != "insufficient" else "insufficient"), data["evidence"])
        text = "\n\n".join("**" + doc.document.document_id + " · " + (doc.document.title or doc.document.selection.id) + "**\n" + (
            "\n".join("- [" + doc.document.document_id + "." + claim.id + "] " + claim.statement for claim in doc.summary.claims) if doc.summary else "未发布摘要；请查看独审/覆盖状态。") for doc in documents)
        if answer.comparisons:
            text += "\n\n**跨文比较**\n" + "\n".join("- [" + claim.id + "] " + claim.statement for claim in answer.comparisons)
        result = {**answer.model_dump(), "text": text, "context_manifest": context,
            "citations": [{"ref_id": edge.ref_id, "quote": edge.quote, "quote_locator": edge.quote_locator.model_dump()} for claim in public for edge in claim.edges],
            "citation_transport": "citequest-quote-spans/3", "source_warnings": source_warnings(data["evidence"]),
            "publication_status": "validated_ai_review" if public else "rejected" if quality_failed_without_public else "not_evaluated",
            "completion_status": "bounded_complete_available_scope" if complete else "partial_request_or_source_coverage",
            "source_scope": [doc.coverage.model_dump() for doc in documents],
            "request_coverage": {"status": "complete" if complete else "partial", "global": global_review, "documents": data["document_request_coverage"], "comparison": data["comparison_coverage"]},
            "review": {"jobs": [{"job_id": row["job_id"], "accepted": row["accepted"], "source_reviewed": row["source_reviewed"],
                "audit": [{key: value for key, value in step.items() if key in {"phase", "raw_sha256", "error_code", "error_codes", "ordinal", "accepted"}}
                    for step in row["audit"]]} for row in data["reviews"]], "plan": data.get("plan_review")}, "admission": data.get("admission"),
            "quality": {"format": "validated", "citation_literal": "validated", "semantic_support": "ai_review_not_human_calibrated" if public else "not_assessed"},
            "fixture": any(item.get("fixture") or item.get("source_ref", {}).get("kind") == "fixture" for item in data["evidence"])}
        return {"data": data, "result": result, "evidence": data["evidence"]}
