"""Optional post-draft review on the original Task/Provider/MCP/Saver harness."""
from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from citequest.research_v3.review_contracts import (
    AGGREGATION_POLICY, ROLE_POLICIES, ROLES, Aggregation, ReviewDiagnostics,
    ReviewReport, RoleReport, bound_aggregation_model, bound_role_model, render_review,
    validate_aggregation, validate_role_report,
)
from citequest.research_v3.school_contracts import SchoolDocument
from citequest.research_v3.typed_feedback import strict_object
from citequest.v2.domain import Actor

from .context import fit_context
from .contracts import AgentError, canonical, digest
from .model_contracts import validated_model_alias
from .provider import POLICY_MARKER, output_text
from .structured_output import strict_output_schema
from .user_skills import activate_for_task, validate_task_skills

if TYPE_CHECKING:
    from .graph import AgentGraph

STEPS = (*ROLES, "aggregate")
JSON_ERRORS = frozenset({"typed_feedback_duplicate_json_property", "typed_feedback_nonfinite_json",
    "typed_feedback_initial_json_unreadable", "typed_feedback_initial_object_required"})


class ReviewState(TypedDict, total=False):
    command_id: str
    document: dict
    document_sha256: str
    binding: dict
    plan: dict
    literature: list[dict]
    intake: dict
    evidence: list[dict]
    memories: list[dict]
    manifest: dict
    reports: dict[str, dict]
    aggregation: dict
    requests: dict[str, dict]
    calls: dict[str, str]
    recoveries: dict[str, dict]
    next: str
    result: dict


class ResearchReviewGraph:
    def __init__(self, host: AgentGraph) -> None:
        self.host = host
        graph = StateGraph(ReviewState)
        graph.add_node("admit", host.observed_node("research_review_admit", self.admit))
        graph.add_edge(START, "admit")
        graph.add_edge("admit", "evidence")
        for role in STEPS:
            for recovery in (False, True):
                name = "recover_" + role if recovery else role
                graph.add_node(name, host.observed_node("research_review_" + name,
                    self.role_node(role, recovery=recovery)))
                graph.add_conditional_edges(name, lambda state: state["next"],
                    [*STEPS, *("recover_" + item for item in STEPS), "publish", END])
        graph.add_node("publish", host.observed_node("research_review_publish", self.publish))
        graph.add_edge("publish", END)
        self.workflow = graph.compile(checkpointer=host.saver)
        self.config = {"configurable": {"thread_id": str(host.execution.task_id)}, "recursion_limit": 32}

    def task(self) -> dict:
        return self.host.execution.snapshot()

    def key(self, role: str, *, recovery: bool = False) -> str:
        base = str(self.host.execution.command["id"]) + ":research-post-review:" + role
        return base + ":protocol-recovery:1" if recovery else base

    def admit(self, state: dict) -> dict:
        task = self.task()
        if (task["kind"] != "research" or task["profile"].get("profile_id") != "research-review-v1"
                or task["brief"].get("research_action") != "review"):
            raise AgentError("research_review_profile_action_required")
        source = self.host.research_history.prepare_child(self.host.execution, task["brief"]["review_source"])
        self.host.wait(self.host.tools.check_scope())
        document = SchoolDocument.model_validate_json(canonical(source.result["school_document"]), strict=True)
        if digest(source.result["school_document"]) != source.binding["document_sha256"]:
            raise AgentError("research_review_document_hash_changed")
        from .research_history import source_ref_ids
        refs = source_ref_ids(source.result)
        expected = {item["ref_id"]: item for item in source.result["context_manifest"]["evidence_refs"]}
        evidence = []
        for ref in refs:
            pinned = self.host.wait(self.host.tools.call("read_evidence", {"ref_id": ref},
                logical_key=str(self.host.execution.command["id"]) + ":research-post-review:read:" + ref))
            if (pinned.get("ref_id") != ref or type(pinned.get("text")) is not str
                    or any(pinned.get(key) != expected[ref].get(key)
                           for key in ("source_sha256", "unit_text_sha256", "source_ref"))):
                raise AgentError("research_review_evidence_changed_during_read")
            evidence.append(pinned)
        memories = [item for item in self.host.memory.current(task["owner_id"], task["project_id"])
            if item.get("kind") != "preference" and not item.get("source_refs")]
        skills = activate_for_task(self.host.execution.database, task)
        identity = {key: source.result["context_manifest"][key] for key in ("brief_id", "brief_revision", "brief_sha256")}
        memory_refs = [{"id": item["memory_id"], "revision": item["revision"]} for item in memories]
        manifest = {**identity, "graph_version": task["graph_version"], "profile_hash": task["profile_hash"],
            "workflow_version": "citequest-research-post-review/1", "review_source": source.binding,
            "memory_refs": memory_refs, "hard_memory_refs": memory_refs, "skill_refs": skills["pins"],
            "source_scope": task["brief"].get("selected_papers", []),
            "evidence_refs": [{key: item[key] for key in ("ref_id", "source_sha256", "unit_text_sha256", "source_ref")}
                for item in evidence]}
        return {"command_id": str(self.host.execution.command["id"]),
            "document": document.model_dump(mode="json"), "document_sha256": source.binding["document_sha256"],
            "binding": source.binding, "plan": source.result["plan"], "literature": source.result["literature_ledger"],
            "intake": source.result["research_intake"], "evidence": evidence, "memories": memories,
            "manifest": manifest, "reports": {}, "aggregation": {}, "requests": {}, "calls": {},
            "recoveries": {}, "result": {}}

    def authorize(self, state: dict) -> None:
        task = self.task()
        if str(self.host.execution.command["id"]) != state["command_id"]:
            raise AgentError("research_review_command_changed")
        current = self.host.research_history.read(Actor(task["owner_id"]), state["binding"])
        if current.binding != state["binding"] or digest(current.result["school_document"]) != state["document_sha256"]:
            raise AgentError("research_review_source_changed")
        validate_task_skills(self.host.execution.database, task)
        self.host.memory.validate_result_context(task, {"context_manifest": state["manifest"]})

    @staticmethod
    def section_identities(state: dict) -> tuple[dict[str, str], dict[str, str]]:
        sections = state["document"]["sections"]
        return ({item["section_id"]: digest(item) for item in sections},
            {item["section_id"]: "\n\n".join(block["text"] for block in item["blocks"]) for item in sections})

    @staticmethod
    def manuscript(state: dict) -> dict:
        document = state["document"]
        # Complete manuscript blocks, with fixed obligations; previous model
        # reviews and the large attempt/handoff ledger are not review authority.
        return {key: document[key] for key in ("title", "profile", "brief_sha256", "sections", "section_specs", "source_aliases")}

    def role_context(self, state: dict, role: str) -> tuple[dict, set[str], dict[str, list[str]]]:
        hashes, _ = self.section_identities(state)
        base = {"manuscript": self.manuscript(state), "document_sha256": state["document_sha256"],
            "section_sha256s": hashes}
        if role == "aggregate":
            return {**base, "independent_reports": [state["reports"][key] for key in ROLES],
                "role_report_sha256s": {key: digest(state["reports"][key]) for key in ROLES}}, set(), {}
        if role in {"methods", "structure"}:
            base["fixed_user_intake"] = {key: state["intake"].get(key, {}) for key in ("user_confirmed", "explicit_unknowns")}
        if role == "methods":
            base["reviewed_proposed_plan"] = state["plan"]
        if role == "structure":
            base["proposed_feasibility_constraints"] = [{key: experiment[key] for key in
                ("id", "research_question_id", "data_and_access", "resources_and_upper_bound", "risks_and_stopping_rules")}
                for experiment in state["plan"]["experiments"]]
            return base, set(), {}
        sections = state["document"]["sections"]
        if role == "methods":
            relevant = {item["id"] for item in state["document"]["section_specs"] if item["group"] in {"methods", "objectives"}}
            sections = [item for item in sections if item["section_id"] in relevant]
        selected = {claim for item in sections for block in item["blocks"] for claim in block["claim_ids"]}
        by_ref = {item["ref_id"]: item for item in state["evidence"]}
        claims, quotes = [], {}
        for ledger in state["literature"]:
            for identifier, claim in zip(ledger["claim_ids"], ledger["answer"]["claims"], strict=True):
                if identifier not in selected:
                    continue
                claims.append({"claim_id": identifier, "claim": claim})
                for edge in claim["edges"]:
                    ref, quote = edge["ref_id"], edge["quote"]
                    if ref not in by_ref or quote not in by_ref[ref]["text"]:
                        raise AgentError("research_review_claim_quote_no_longer_literal")
                    quotes.setdefault(ref, []).append(quote)
        if {item["claim_id"] for item in claims} != selected:
            raise AgentError("research_review_manuscript_claim_ledger_missing")
        return {**base, "admitted_claims": claims, "admitted_quotes": quotes}, selected, quotes

    def request(self, state: dict, role: str, *, recovery: bool) -> dict:
        task = self.task()
        if recovery:
            request = copy.deepcopy(state["requests"][role])
            prompt = json.loads(request["input"][0]["content"])
            prompt["protocol_feedback"] = {"error_codes": state["recoveries"][role]["error_codes"],
                "rule": "Return a complete object under the identical schema and fixed manuscript. Preserve independent negative opinions. No factual, source or policy change is authorized."}
            request["input"][0]["content"] = canonical(prompt).decode()
            return fit_context(request, task["profile"]["request_bytes"])
        data, _, _ = self.role_context(state, role)
        model = (bound_aggregation_model(state["document_sha256"]) if role == "aggregate"
            else bound_role_model(role, state["document_sha256"]))
        payload = {"model": validated_model_alias(task["profile"]), "reasoning": {"effort": "none"},
            "truncation": "disabled", "max_output_tokens": task["profile"]["per_call_output"],
            "instructions": POLICY_MARKER + "\n" + (AGGREGATION_POLICY if role == "aggregate" else ROLE_POLICIES[role]),
            "input": [{"role": "user", "content": canonical({"role": role,
                "canonical_brief": {"question": task["brief"].get("question", "")},
                "confirmed_user_memory_data": state["memories"], "selected_user_skills": [],
                "context_manifest": state["manifest"], **data}).decode()}],
            "text": {"format": {"type": "json_schema", "name": "citequest_research_post_review_" + role,
                "strict": True, "schema": strict_output_schema(model.model_json_schema())}}}
        return fit_context(payload, task["profile"]["request_bytes"])

    def partial(self, state: dict, role: str, codes: list[str]) -> dict:
        completed = [item for item in STEPS if item in state["calls"]]
        diagnostics = ReviewDiagnostics(phase=role, error_codes=codes, completed_roles=completed,
            remaining_roles=[item for item in STEPS if item not in completed])
        return {"next": END, "result": {"kind": "research", "status": "research_review_partial",
            "publication_status": "not_evaluated", "completion_status": "partial_research_review",
            "text": "评审尚未完成。阶段：" + role + "；原因：" + "、".join(codes),
            "diagnostics": diagnostics.model_dump(mode="json"), "context_manifest": state["manifest"],
            "citations": [], "school_format_ready": False}}

    def protocol_failure(self, state: dict, role: str, payload: dict, codes: list[str], *, recovery: bool) -> dict:
        if recovery:
            return self.partial(state, role, ["research_review_protocol_invalid_after_one_recovery", *codes])
        settled = self.host.execution.settled_model_request(self.key(role))
        if settled is None or settled["request_hash"] != digest(payload):
            raise AgentError("research_review_protocol_recovery_requires_known_settled_usage")
        self.authorize(state)
        return {"requests": {**state["requests"], role: payload},
            "recoveries": {**state["recoveries"], role: {"base_key": self.key(role),
                "recovery_key": self.key(role, recovery=True), "error_codes": codes}}, "next": "recover_" + role}

    def role_node(self, role: str, *, recovery: bool):
        def run(state: dict) -> dict:
            self.authorize(state)
            try:
                payload = self.request(state, role, recovery=recovery)
            except AgentError as error:
                if error.code == "required_context_exceeds_wire_limit":
                    return self.partial(state, role, [error.code])
                raise
            response = self.host.wait(self.host.provider.complete(payload, self.key(role, recovery=recovery)))
            if any(part.get("type") == "refusal" for item in response.get("output", []) for part in item.get("content", [])):
                return self.partial(state, role, ["research_review_model_refusal"])
            model = (bound_aggregation_model(state["document_sha256"]) if role == "aggregate"
                else bound_role_model(role, state["document_sha256"]))
            try:
                parsed = model.model_validate(strict_object(output_text(response)), strict=True)
            except ValidationError:
                return self.protocol_failure(state, role, payload, ["research_review_output_schema_invalid"], recovery=recovery)
            except AgentError as error:
                if error.code not in JSON_ERRORS:
                    raise
                return self.protocol_failure(state, role, payload, [error.code], recovery=recovery)
            hashes, texts = self.section_identities(state)
            try:
                if role == "aggregate":
                    validate_aggregation(parsed, [RoleReport.model_validate(state["reports"][key], strict=True) for key in ROLES],
                        document_sha256=state["document_sha256"], section_sha256s=hashes)
                else:
                    context, claims, quotes = self.role_context(state, role)
                    claim_quotes = {}
                    for item in context.get("admitted_claims", []):
                        by_ref = {}
                        for edge in item["claim"]["edges"]:
                            by_ref.setdefault(edge["ref_id"], []).append(edge["quote"])
                        claim_quotes[item["claim_id"]] = by_ref
                    validate_role_report(parsed, role=role, document_sha256=state["document_sha256"],
                        section_sha256s=hashes, section_texts=texts, claim_ids=claims, quotes=quotes,
                        claim_quotes=claim_quotes)
            except AgentError as error:
                return self.partial(state, role, [error.code])
            following = "publish" if role == "aggregate" else STEPS[STEPS.index(role) + 1]
            return {"calls": {**state["calls"], role: self.key(role, recovery=recovery)}, "next": following,
                **({"aggregation": parsed.model_dump(mode="json")} if role == "aggregate"
                    else {"reports": {**state["reports"], role: parsed.model_dump(mode="json")}})}
        return run

    def publish(self, state: dict) -> dict:
        self.authorize(state)
        self.host.wait(self.host.tools.check_scope())
        hashes, _ = self.section_identities(state)
        report = ReviewReport(source_document_sha256=state["document_sha256"],
            source_binding_sha256=digest(state["binding"]), section_sha256s=hashes,
            role_reports=[RoleReport.model_validate(state["reports"][role], strict=True) for role in ROLES],
            aggregation=Aggregation.model_validate(state["aggregation"], strict=True))
        value = report.model_dump(mode="json")
        citations = {(item.ref_id, item.quote): item.model_dump(mode="json")
            for role in report.role_reports for finding in role.findings for item in finding.citations}
        task = self.task()
        result = {"kind": "research", "status": "research_review_ready", "publication_status": "validated_ai_review",
            "completion_status": "research_review_report_ready", "school_format_ready": False,
            "text": render_review(report), "review_report": value, "review_report_sha256": digest(value),
            "context_manifest": {**state["manifest"], "review_report_sha256": digest(value)},
            "citations": list(citations.values()), "review_execution": {"role_model_keys": state["calls"],
                "protocol_recoveries": state["recoveries"],
                "model_receipts": self.host.execution.settled_model_receipts(list(state["calls"].values()))}}
        self.host.memory.validate_result_context(task, result)
        return {"result": result}

    def run(self) -> dict:
        previous = self.workflow.get_state(self.config)
        command_id = str(self.host.execution.command["id"])
        if previous.next and previous.values.get("command_id") != command_id:
            raise AgentError("research_review_checkpoint_command_mismatch")
        value = self.workflow.invoke(None if previous.next else {"command_id": command_id},
            self.config, durability="sync")
        snapshot = self.workflow.get_state(self.config)
        result = value["result"]
        self.host.execution.finish("succeeded" if result.get("status") == "research_review_ready" else "partial",
            result=result, checkpoint_id=snapshot.config.get("configurable", {}).get("checkpoint_id"))
        return value
