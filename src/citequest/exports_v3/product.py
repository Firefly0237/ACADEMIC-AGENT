"""Current product Task projection and reference bridge; no model dependency."""

from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.database import TaskService
from citequest.agents_v3.memory import MemoryService
from citequest.agents_v3.result_projection import ProjectedTaskReader
from citequest.platform_v3.blob_store import BlobStore
from citequest.research_v3.evidence import EvidenceAccess, reference_allows
from citequest.research_v3.school_contracts import SchoolDocument
from citequest.research_v3.school_export import current_references
from citequest.search_v3.service import SearchService
from citequest.v2.domain import Actor, DomainError
from citequest.v2.pagination import Pager

from .service import ExportProfile, ExportService


@dataclass(frozen=True)
class ExportDomain:
    service: ExportService
    evidence: EvidenceAccess
    search: SearchService

    def references(self, actor: Actor, task_id: UUID, document: SchoolDocument) -> list[dict]:
        task, current = self.service.document(actor, task_id)
        if digest(document.model_dump(mode="json")) != digest(current.model_dump(mode="json")):
            raise DomainError("export_task_document_or_context_changed", 409)

        def read(ref_id: str) -> dict:
            value = self.evidence.read(actor, task_id, ref_id)
            if not reference_allows(task["brief"], value):
                raise DomainError("export_reference_role_changed", 409)
            return value

        references = current_references(document, read, lambda locator: self.search.detail(actor, locator))
        # Metadata/source IO cannot create an interval of unguarded publication.
        _, after = self.service.document(actor, task_id)
        if digest(after.model_dump(mode="json")) != digest(document.model_dump(mode="json")):
            raise DomainError("export_task_document_or_context_changed", 409)
        return references


def build_export_domain(
    factory: sessionmaker[Session],
    tasks: TaskService,
    memory: MemoryService,
    evidence: EvidenceAccess,
    search: SearchService,
    store: BlobStore,
    profile: ExportProfile,
    pager: Pager,
) -> ExportDomain:
    """No DB query, DDL, Docker process, model request or key read at construction."""
    reader = ProjectedTaskReader(tasks, memory, evidence.authorize)

    def authorize(actor: Actor, task_id: UUID) -> dict:
        task = reader.get(actor, task_id)
        result = task.get("result")
        if (
            task.get("result_availability") == "available"
            and isinstance(result, dict)
            and result.get("school_document")
        ):
            if task.get("runtime_availability") != "current":
                return {**task, "result": None, "result_availability": "legacy_runtime_export_not_supported"}
            document = result["school_document"]
            context = result.get("context_manifest")
            # The producer emits one identical manifest for the visible result
            # and embedded SchoolDocument. Do not authorize only an outer shell
            # while exporting a different nested source/memory/Brief context.
            if (
                not isinstance(context, dict)
                or not context
                or not isinstance(document, dict)
                or document.get("context_manifest") != context
                or document.get("brief_sha256") != context.get("brief_sha256")
                or context.get("source_scope") != task["brief"].get("selected_papers", [])
            ):
                return {**task, "result": None, "result_availability": "school_document_context_mismatch"}
            payload = {"input": [{"content": json.dumps({"context_manifest": context})}]}
            try:
                # Existing dispatch guards also reject new hard constraints or
                # pending confirmations while old dependencies still exist.
                memory.validate_wire_context({**task, "owner_id": actor.owner_id}, payload)
                from citequest.agents_v3.user_skills import validate_task_skills

                validate_task_skills(tasks.database, {**task, "owner_id": actor.owner_id}, payload)
            except AgentError:
                return {**task, "result": None, "result_availability": "export_current_context_invalidated"}
        return task

    service = ExportService(factory, authorize, store, profile, pager)
    return ExportDomain(service, evidence, search)
