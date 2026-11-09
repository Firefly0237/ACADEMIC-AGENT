"""One current-authority Task projection shared by API and artifact workers.

No result cache, model provider, mutation, migration or permissive source fallback.
The original Agent API guards are centralized without changing their semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from citequest.v2.domain import Actor

from .contracts import AgentError
from .database import TaskService, owned_task
from .memory import MemoryService


def authorize_result_dependencies(
    service: TaskService, memory: MemoryService,
    authorize_evidence: Callable[[Actor, UUID, list[str]], None],
    actor: Actor, task_id: UUID, task: dict, result: dict,
) -> None:
    """Current dependency gates shared by current and exact historical readers."""
    try:
        from .user_skills import validate_task_skills
        validate_task_skills(service.database, {**task, "owner_id": actor.owner_id})
    except AgentError:
        raise AgentError("selected_skill_changed_or_deleted", 409) from None
    active = {(item["memory_id"], item["revision"]) for item in memory.current(actor.owner_id, task["project_id"])}
    dependencies = result.get("context_manifest", {}).get("memory_refs", [])
    if any((item["id"], item["revision"]) not in active for item in dependencies):
        raise AgentError("memory_invalidated", 409)
    if "interrupts" not in result:
        try:
            memory.validate_result_context({**task, "owner_id": actor.owner_id}, result)
        except AgentError:
            raise AgentError("current_memory_context_invalidated", 409) from None
    try:
        references = {item["ref_id"] for item in result.get("citations", [])}
        references.update(item["ref_id"] for item in result.get("context_manifest", {}).get("evidence_refs", []))
        references.update(result.get("evidence_navigation", {}).get("input_frontier_ref_ids", []))
        current_refs = sorted(references)
        for start in range(0, max(1, len(current_refs)), 256):
            authorize_evidence(actor, task_id, current_refs[start : start + 256])
    except Exception:
        raise AgentError("evidence_unavailable_or_access_revoked", 409) from None


def project_task_result(
    service: TaskService,
    memory: MemoryService,
    authorize_evidence: Callable[[Actor, UUID, list[str]], None],
    actor: Actor,
    task_id: UUID,
    task: dict,
) -> dict:
    # Metadata is recomputed only after current authority checks, never retained
    # from an earlier projection whose sources may since have been withdrawn.
    task = {key: value for key, value in task.items() if key != "research_progress"}
    result = task.get("result")
    progress_only = False
    review_diagnostics_only = False
    if (result is None and task.get("profile", {}).get("profile_id") in {"research-review-v1", "research-revision-v1"}
            and task.get("result_availability") in {"quality_rejected", "partial_no_evaluated_claims"}):
        with service.database.connect() as db:
            source = owned_task(db, actor.owner_id, task_id)
        if any(source[key] != task[key] for key in ("revision", "brief", "project_id", "state")):
            return {**task, "result": None, "result_availability": "task_changed_during_projection"}
        candidate = source.get("result")
        if isinstance(candidate, dict) and candidate.get("status") in {"research_review_partial", "research_revision_partial"}:
            try:
                from citequest.research_v3.review_contracts import ReviewDiagnostics
                diagnostics = ReviewDiagnostics.model_validate(candidate["diagnostics"], strict=True)
                result = {"kind": "research", "status": candidate["status"],
                    "publication_status": "not_evaluated", "completion_status": "partial_research_workflow",
                    "text": "研究任务尚未完成，请查看诊断与剩余步骤。",
                    "diagnostics": diagnostics.model_dump(mode="json"),
                    "context_manifest": candidate["context_manifest"]}
                task, review_diagnostics_only = {**task, "result": result}, True
            except (KeyError, TypeError, ValueError):
                return {**task, "result": None, "result_availability": "review_diagnostics_invalid"}
    if (result is None and task.get("kind") == "research"
            and task.get("result_availability") in {"quality_rejected", "partial_no_evaluated_claims"}):
        # The service deliberately withheld unpublished prose. Read its owned
        # source only inside this projection; no raw body enters a public DTO.
        with service.database.connect() as db:
            source = owned_task(db, actor.owner_id, task_id)
        if (source["revision"] != task["revision"] or source["brief"] != task["brief"]
                or source["project_id"] != task["project_id"] or source["state"] != task["state"]):
            return {**task, "result": None, "result_availability": "task_changed_during_projection"}
        candidate = source.get("result")
        if isinstance(candidate, dict) and candidate.get("kind") == "research" and isinstance(candidate.get("research_progress"), dict):
            result, progress_only = candidate, True
    if result is None:
        return task
    if task["kind"] == "research":
        try:
            from .research_runtime import load_document
            if task["profile"].get("profile_id") in {"research-review-v1", "research-revision-v1"}:
                from .research_history import HistoricalResearchReader
                history = HistoricalResearchReader(service, memory, authorize_evidence)
                if task["profile"]["profile_id"] == "research-revision-v1":
                    inputs = history.read_revision(actor, task["brief"]["review_source"], task["brief"]["revision_request"])
                    source = inputs.source
                    manifest = result.get("context_manifest", {})
                    if (manifest.get("revision_request") != task["brief"]["revision_request"]
                            or manifest.get("review_report_sha256") != inputs.report_binding["report_sha256"]
                            or manifest.get("source_document_sha256") != source.binding["document_sha256"]
                            or manifest.get("revision_target_section_ids") != inputs.target_section_ids):
                        raise AgentError("research_revision_context_binding_changed", 409)
                else:
                    source = history.read(actor, task["brief"]["review_source"])
                document, sha256 = source.brief, source.binding_brief_sha256
                if result.get("context_manifest", {}).get("review_source") != source.binding:
                    raise AgentError("research_review_source_binding_changed", 409)
            else:
                reference = task["brief"]["research_brief"]
                with service.database.connect() as db:
                    document, sha256 = load_document(db, actor.owner_id, task["project_id"], UUID(reference["id"]))
            manifest = result.get("context_manifest", {})
            if manifest.get("brief_id") is not None and (
                manifest.get("brief_id") != str(document.id)
                or manifest.get("brief_revision") != document.revision
                or manifest.get("brief_sha256") != sha256
            ):
                raise AgentError("research_brief_changed_or_deleted")
            if manifest.get("brief_id") is None and "interrupts" not in result:
                raise AgentError("research_brief_context_missing")
        except AgentError as error:
            availability = error.code if task["profile"].get("profile_id") in {"research-review-v1", "research-revision-v1"} else "research_brief_changed_or_deleted"
            return {**task, "result": None, "result_availability": availability}
    waiting_input = "interrupts" in result
    if not progress_only and not review_diagnostics_only and not waiting_input and result.get("publication_status") != "validated_ai_review":
        return {**task, "result": None, "result_availability": "quality_rejected"}
    try:
        authorize_result_dependencies(service, memory, authorize_evidence, actor, task_id, task, result)
    except AgentError as error:
        return {**task, "result": None, "result_availability": error.code}
    if review_diagnostics_only:
        return {**task, "result_availability": "available_diagnostics"}
    if task["kind"] == "research":
        from .research_progress import public_research_progress

        progress = public_research_progress(result)
        if progress is not None:
            task = {**task, "research_progress": progress}
        if progress_only:
            return {**task, "result": None, "result_availability": "research_progress_only" if progress is not None else "quality_rejected"}
    if waiting_input:
        # A clarification/conflict handoff is not permission to publish an
        # unreviewed plan or answer stored alongside it in the checkpoint.
        visible = {
            key: result[key]
            for key in ("interrupts", "context_manifest", "citations", "kind", "status", "gaps", "brief_identity")
            if key in result
        }
        return {**task, "result": visible, "result_availability": "available_for_input"}
    return {**task, "result_availability": "available_for_input" if waiting_input else "available"}


class ProjectedTaskReader:
    def __init__(
        self, service: TaskService, memory: MemoryService, authorize_evidence: Callable[[Actor, UUID, list[str]], None]
    ) -> None:
        self.service, self.memory, self.authorize_evidence = service, memory, authorize_evidence

    def project(self, actor: Actor, task_id: UUID, task: dict) -> dict:
        return project_task_result(self.service, self.memory, self.authorize_evidence, actor, task_id, task)

    def get(self, actor: Actor, task_id: UUID) -> dict:
        return self.project(actor, task_id, self.service.get(actor.owner_id, task_id))
