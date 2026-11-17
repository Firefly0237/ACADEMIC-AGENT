"""Explicit adopted-finding revision through the existing school quality gates."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langgraph.graph import END

from citequest.research_v3.revision_contracts import (
    RESOLUTION_POLICY, REVISION_POLICY, bound_resolution_model,
    validate_resolution, validate_revision_scope,
)
from citequest.research_v3.review_contracts import ReviewDiagnostics
from citequest.research_v3.school_contracts import SchoolProfile, SchoolSection, SectionSpec
from citequest.v2.domain import Actor

from .contracts import AgentError, digest
from .research_workflow import PlanningInput, ResearchWorkflow
from .school_document import SchoolDocumentWorkflow, SchoolState
from .user_skills import activate_for_task

if TYPE_CHECKING:
    from .graph import AgentGraph


class RevisionState(SchoolState, total=False):
    command_id: str
    source_document: dict
    report: dict
    report_binding: dict
    selected_findings: list[dict]
    target_section_ids: list[str]
    resolution: dict
    feedback_targets: list[str]


class RevisionBoundary(ResearchWorkflow):
    def __init__(self, host: AgentGraph) -> None:
        super().__init__(host, lambda: {}, checkpointer=True, strict_identity_binding=True,
            review_outline=True, writing_reserve_model_calls=0)

    def shared_revisions(self) -> bool:
        return True

    def audited_handoffs(self) -> bool:
        # Parent audit identities remain source references, never child receipts.
        return False

    def revision_inputs(self):
        task = self.task()
        return self.host.research_history.read_revision(Actor(task["owner_id"]),
            task["brief"]["review_source"], task["brief"]["revision_request"])

    def current_input(self) -> PlanningInput:
        return PlanningInput.model_validate(self.revision_inputs().source.result["research_intake"], strict=True)

    def payload(self, state, role, model, name, policy, data, *, logical_key=None):
        return super().payload(state, role, model, name, policy,
            {**data, "context_manifest": state["upstream"]["context_manifest"]}, logical_key=logical_key)


class RevisionDocumentWorkflow(SchoolDocumentWorkflow):
    STATE = RevisionState
    NODES = (*SchoolDocumentWorkflow.NODES[:-1], "resolve_findings", "publish")

    def observe(self, name, node):
        observed = super().observe(name, node)
        def current(state):
            if state.get("source_document"):
                self.profile = SchoolProfile.model_validate(state["source_document"]["profile"], strict=True)
                self.boundary.revision_inputs()
                validate_revision_scope(state["source_document"]["sections"], self.ordered(state), state["target_section_ids"])
            return observed(state)
        return current

    def initial_model_reserve(self, state):
        # Initial writer/reviewer per selected section + cross/request/judge,
        # then two shared feedback credits at writer/reviewer/cross/request/judge.
        return 2 * len(state["target_section_ids"]) + 3 + 2 * 5

    def initial_tool_reserve(self, state, references):
        prefix = (str(self.host.execution.command["id"]) + ":research:"
            + state["upstream"]["research_intake"]["brief_sha256"] + ":school:admit:")
        keys = [prefix + item["ref_id"] for item in references]
        observed = self.host.execution.settled_tool_keys(keys)
        # This is a resource observation only. Every initial read still goes
        # through the existing MCP replay and fresh source/permission guards.
        return len(keys) - len(observed) + len(references)

    def prepare(self, state):
        task = self.task()
        if (task["profile"].get("profile_id") != "research-revision-v1"
                or task["brief"].get("research_action") != "revise"):
            raise AgentError("research_revision_profile_action_required")
        inputs = self.host.research_history.prepare_revision_child(self.host.execution,
            task["brief"]["review_source"], task["brief"]["revision_request"])
        self.host.wait(self.host.tools.check_scope())
        source, document = inputs.source.result, inputs.source.result["school_document"]
        self.profile = SchoolProfile.model_validate(document["profile"], strict=True)
        memories = [{"id": item["memory_id"], "revision": item["revision"]} for item in self.boundary.hard_memories()]
        skills = activate_for_task(self.host.execution.database, task)
        manifest = {**{key: source["context_manifest"][key] for key in
                ("brief_id", "brief_revision", "brief_sha256", "evidence_refs")},
            "graph_version": task["graph_version"], "profile_hash": task["profile_hash"],
            "workflow_version": "citequest-research-targeted-revision/1", "review_source": inputs.source.binding,
            "revision_request": task["brief"]["revision_request"], "review_report_sha256": digest(inputs.report),
            "source_document_sha256": inputs.source.binding["document_sha256"],
            "revision_target_section_ids": inputs.target_section_ids,
            "memory_refs": memories, "hard_memory_refs": memories, "skill_refs": skills["pins"],
            "source_scope": task["brief"].get("selected_papers", []),
            "retained_section_review_source": inputs.source.binding,
            **({"source_handoff_audit_sha256": digest(source["handoff_audit"])} if source.get("handoff_audit") else {})}
        upstream = {key: source[key] for key in ("plan", "research_intake", "literature_ledger", "evidence_ledger", "brief_identity")}
        upstream.update(status="structured_plan_ready", publication_status="validated_ai_review",
            performed_experiments=False, context_manifest=manifest, citations=source.get("citations", []))
        return super().prepare({**state, "upstream": upstream, "source_document": document,
            "report": inputs.report, "report_binding": inputs.report_binding,
            "selected_findings": inputs.selected_findings, "target_section_ids": inputs.target_section_ids,
            "resolution": {}, "feedback_targets": []})

    def seed_sections(self, prepared):
        document = prepared["source_document"]
        if document["source_aliases"] != prepared["aliases"]:
            raise AgentError("research_revision_source_alias_binding_changed")
        return {**prepared, "sections": {item["section_id"]: item for item in document["sections"]},
            "reviews": {item["section_id"]: item for item in document["section_reviews"]},
            "specs": document["section_specs"]}

    def writer_preflight_specs(self, prepared, specs):
        return [SectionSpec.model_validate(item, strict=True) for item in prepared["specs"]
            if item["id"] in prepared["target_section_ids"]]

    def spec(self, state, identifier=None):
        return super().spec(state, identifier or state["target_section_ids"][state["index"]])

    def section_data(self, state, spec):
        return {**super().section_data(state, spec),
            "adopted_findings": [item for item in state["selected_findings"] if spec.id in item["section_ids"]],
            "original_section": next(item for item in state["source_document"]["sections"] if item["section_id"] == spec.id),
            "complete_current_manuscript": self.ordered(state),
            "revision_scope": {"allowed_section_ids": state["target_section_ids"],
                "source_document_sha256": state["upstream"]["context_manifest"]["source_document_sha256"],
                "review_report_sha256": digest(state["report"])}}

    def write_section(self, state):
        spec = self.spec(state)
        value = self.complete(state, "school_targeted_reviser", SchoolSection, "citequest_school_section",
            REVISION_POLICY, self.section_data(state, spec), spec.id + ":adopted-initial")
        return {"sections": {**state["sections"], spec.id: value.model_dump(mode="json")},
            "section_model_keys": {**state["section_model_keys"], spec.id:
                self.boundary.model_key(state, "school:" + spec.id + ":adopted-initial")}, "next": "review_section"}

    def decide_section(self, state):
        value = super().decide_section(state)
        if value.get("next") == "write_section" and value["index"] >= len(state["target_section_ids"]):
            value["next"] = "review_cross"
        return value

    def schedule_repair(self, state, target, issues, after):
        if target not in state["target_section_ids"]:
            return self.partial(state, "research_revision_feedback_requires_unselected_section")
        remaining = max(0, len(state["target_section_ids"]) - state["index"] - 1) if after == "advance" else 0
        if (state["repairs"] >= 2 or self.host.execution.content_revision_count() >= 2
                or self.remaining(state) < 5 + 2 * remaining):
            return self.partial(state, "research_revision_bounded_feedback_capacity_exhausted")
        return {"repair_target": target, "issues": issues, "return_after_repair": after, "next": "repair_section",
            "feedback_targets": [*state.get("feedback_targets", []), target]}

    def repair_section(self, state):
        value = super().repair_section(state)
        if value.get("sections"):
            validate_revision_scope(state["source_document"]["sections"],
                [value["sections"][item["id"]] for item in state["specs"]], state["target_section_ids"])
        return value

    def repair_source_key(self, state, spec, current, review):
        content = digest(self.ordered(state))
        resolution = state.get("resolution", {})
        if (state.get("return_after_repair") == "review_cross"
                and resolution.get("candidate_content_sha256") == content
                and any(item["verdict"] == "unresolved" and spec.id in item["section_ids"] for item in resolution.get("items", []))):
            return self.boundary.model_key(state, "school:resolution:" + content)
        cross = state.get("cross_review", {})
        if (state.get("return_after_repair") == "review_cross" and cross.get("content_sha256") == content
                and any(item["section_id"] == spec.id and item["verdict"] == "needs_revision" for item in cross.get("sections", []))):
            return self.boundary.model_key(state, "school:cross:" + content)
        return super().repair_source_key(state, spec, current, review)

    def repair_error_codes(self, state):
        return (["research_revision_adopted_finding_unresolved"] if any(
            item.startswith("adopted_finding_unresolved:") for item in state["issues"])
            else ["research_revision_independent_quality_review_rejected"])

    def review_requests(self, state):
        value = super().review_requests(state)
        if value.get("next") == "publish":
            value["next"] = "resolve_findings"
        return value

    def resolve_findings(self, state):
        sections = self.ordered(state)
        manifest = state["upstream"]["context_manifest"]
        model = bound_resolution_model(manifest["source_document_sha256"], digest(state["report"]), digest(sections))
        relevant_claims = {identifier for finding in state["selected_findings"] for identifier in finding["claim_ids"]}
        value = self.boundary.complete(state, "reviewer", model, "citequest_research_revision_resolution", RESOLUTION_POLICY,
            {"original_manuscript": state["source_document"]["sections"], "revised_manuscript": sections,
                "adopted_findings": state["selected_findings"],
                "fixed_user_intake": state["intake"], "reviewed_proposed_plan": state["upstream"]["plan"],
                "admitted_claims": {key: item for key, item in state["claims"].items() if key in relevant_claims}},
            "school:resolution:" + digest(sections))
        validate_resolution(value, source_sha=manifest["source_document_sha256"], report_sha=digest(state["report"]),
            sections=sections, selected_findings=state["selected_findings"], evidence=state["evidence"])
        resolution = value.model_dump(mode="json")
        if any(item.verdict == "needs_input" for item in value.items):
            return self.partial({**state, "resolution": resolution}, "research_revision_genuine_input_required")
        unresolved = [item for item in value.items if item.verdict != "resolved"]
        if unresolved:
            targets = {identifier for item in unresolved for identifier in item.section_ids}
            ordered = [identifier for identifier in state["target_section_ids"] if identifier in targets]
            target = next((identifier for identifier in ordered if identifier not in state.get("feedback_targets", [])), ordered[0])
            return {"resolution": resolution, **self.schedule_repair(state, target,
                ["adopted_finding_unresolved:" + item.finding_key + ":" + item.explanation
                    for item in unresolved if target in item.section_ids], "review_cross")}
        return {"resolution": resolution, "next": "publish"}

    def partial(self, state, reason):
        code = reason if re.fullmatch(r"[a-z][a-z0-9_]{0,119}", reason) else "research_revision_quality_gate_failed"
        manifest = state.get("upstream", {}).get("context_manifest", {})
        return {"next": END, "result": {"kind": "research", "status": "research_revision_partial",
            "publication_status": "not_evaluated", "completion_status": "partial_research_revision",
            "text": "定向修订尚未通过全部检查，未发布新稿。", "context_manifest": manifest,
            "diagnostics": ReviewDiagnostics(phase="revision", error_codes=[code],
                completed_roles=[], remaining_roles=[]).model_dump(mode="json"), "citations": [], "school_format_ready": False}}

    def publish(self, state):
        unchanged = validate_revision_scope(state["source_document"]["sections"], self.ordered(state), state["target_section_ids"])
        if not state.get("resolution") or any(item["verdict"] != "resolved" for item in state["resolution"]["items"]):
            return self.partial(state, "research_revision_selected_findings_not_resolved")
        value = super().publish(state)
        if value["result"].get("status") != "school_document_ready":
            return value
        inputs = self.boundary.revision_inputs()
        all_keys = [role["role"] + ":" + item["id"] for role in state["report"]["role_reports"] for item in role["findings"]]
        selected = [item["finding_key"] for item in state["selected_findings"]]
        outcome = {"status": "adopted_findings_resolved", "source_document_sha256": inputs.source.binding["document_sha256"],
            "review_report_sha256": digest(inputs.report), "source_binding": inputs.source.binding, "report_binding": inputs.report_binding,
            "selected_finding_keys": selected, "target_section_ids": state["target_section_ids"],
            "changed_section_ids": [item["section_id"] for item in self.ordered(state)
                if item != next(old for old in state["source_document"]["sections"] if old["section_id"] == item["section_id"])],
            "unchanged_section_sha256s": unchanged, "resolution": state["resolution"],
            "unselected_finding_keys": [key for key in all_keys if key not in selected],
            "unselected_findings_status": "not_selected_not_evaluated_again"}
        value["result"]["revision_outcome"] = outcome
        value["result"]["text"] += ("\n\n## 定向修订说明\n\n已采纳问题：" + "、".join(selected)
            + "\n\n实际修改章节：" + ("、".join(outcome["changed_section_ids"]) or "无文本变化")
            + "\n\n未选问题未再次评估：" + ("、".join(outcome["unselected_finding_keys"]) or "无"))
        value["result"]["text"] += "\n\n" + "\n\n".join(item["finding_key"] + " · " + item["verdict"]
            + "\n\n" + item["explanation"] for item in state["resolution"]["items"])
        keys = [state["section_model_keys"][identifier] for identifier in state["target_section_ids"]]
        keys.extend(self.boundary.model_key(state, "school:" + identifier + ":review:" + digest(state["sections"][identifier]))
            for identifier in state["target_section_ids"])
        content_sha = digest(self.ordered(state))
        keys.extend(self.boundary.model_key(state, key + content_sha) for key in
            ("school:cross:", "school:request-coverage:", "school:resolution:"))
        value["result"]["revision_model_receipts"] = self.host.execution.settled_model_receipts(keys)
        return value


class ResearchRevisionGraph:
    def __init__(self, host: AgentGraph) -> None:
        self.host = host
        self.boundary = RevisionBoundary(host)
        self.document = RevisionDocumentWorkflow(self.boundary, checkpointer=host.saver,
            reference_preflight=getattr(host, "reference_preflight", None))
        self.workflow = self.document.workflow
        self.config = {"configurable": {"thread_id": str(host.execution.task_id)}, "recursion_limit": 96}

    def run(self) -> dict:
        previous = self.workflow.get_state(self.config)
        command = str(self.host.execution.command["id"])
        if previous.next and previous.values.get("command_id") != command:
            raise AgentError("research_revision_checkpoint_command_mismatch")
        value = self.workflow.invoke(None if previous.next else {"command_id": command}, self.config, durability="sync")
        snapshot = self.workflow.get_state(self.config)
        result = value["result"]
        self.host.execution.finish("succeeded" if result.get("status") == "school_document_ready" else "partial",
            result=result, checkpoint_id=snapshot.config.get("configurable", {}).get("checkpoint_id"))
        return value
