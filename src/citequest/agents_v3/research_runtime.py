"""Server-owned, current authorized Brief projection for the research graph.

No model output or free-text resume can confirm an intake fact. The current
document is reloaded at graph boundaries; only an explicit Task command may
synchronize selected sources. Database revision history remains unchanged.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING
from uuid import UUID

from psycopg import Connection

if TYPE_CHECKING:
    from .database import Execution

from citequest.research_v3.brief import BriefDocument

from .contracts import AgentError, ResearchBrief, VersionedBriefRef, digest


def load_document(db: Connection, owner: UUID, project: UUID, brief_id: UUID) -> tuple[BriefDocument, str]:
    row = db.execute("""SELECT b.document,b.document_sha256,b.revision,b.id,b.project_id
        FROM cq_app.research_briefs b
        JOIN cq_app.projects p ON (p.owner_id,p.id)=(b.owner_id,b.project_id)
        WHERE b.id=%s AND b.owner_id=%s AND b.project_id=%s AND b.deleted_at IS NULL""",
        (brief_id, owner, project)).fetchone()
    if row is None:
        raise AgentError("research_brief_not_found", 404)
    document = BriefDocument.model_validate_json(json.dumps(row["document"]), strict=True)
    if (document.id != row["id"] or document.project_id != row["project_id"]
        or document.revision != row["revision"] or digest(row["document"]) != row["document_sha256"]):
        raise AgentError("research_brief_integrity_failure", 500)
    return document, row["document_sha256"]


def bind_reference(db: Connection, owner: UUID, project: UUID, reference: VersionedBriefRef) -> BriefDocument:
    document, sha256 = load_document(db, owner, project, reference.id)
    if document.revision != reference.revision or sha256 != reference.sha256:
        raise AgentError("research_brief_revision_conflict", 409)
    return document


def admit_research(db: Connection, owner: UUID, project: UUID, brief: ResearchBrief) -> None:
    if brief.kind != "research":
        return
    document = bind_reference(db, owner, project, brief.research_brief)
    if document.selected_sources != brief.selected_papers:
        raise AgentError("research_source_selection_mismatch", 409)
    if document.draft_source_ids != brief.research_draft_files:
        raise AgentError("research_draft_marker_mismatch", 409)


def current_planning_input(execution: Execution) -> dict:
    task = execution.snapshot()
    reference = ResearchBrief.model_validate_json(json.dumps(task["brief"]), strict=True).research_brief
    if task["kind"] != "research" or reference is None:
        raise AgentError("research_runtime_task_kind_invalid")
    with execution.database.connect() as db:
        document, sha256 = load_document(db, task["owner_id"], task["project_id"], reference.id)
    return {
        "schema_version": "citequest-research-planning-input/1",
        "brief_id": str(document.id), "brief_revision": document.revision, "brief_sha256": sha256,
        "user_confirmed": {field: fact.value for field, fact in document.facts.items() if fact.status == "confirmed"},
        "explicit_unknowns": {field: fact.value for field, fact in document.facts.items() if fact.status == "explicit_unknown"},
        "selected_sources": [item.model_dump(mode="json") for item in document.selected_sources],
        **({"draft_source_ids": [str(value) for value in document.draft_source_ids]} if document.draft_source_ids else {}),
        "open_conflicts": [{"id": str(item.id), "field": item.field, "summary": item.candidate.value[:1000]}
                           for item in document.proposals if item.state == "open" and item.kind == "conflict"],
        "research_status": "proposed_not_executed", "source_admission_required": True, "school_format_ready": False,
    }


def validate_wire_intake(execution: Execution, payload: dict) -> None:
    current = current_planning_input(execution)
    manifest = json.loads(payload["input"][0]["content"]).get("context_manifest", {})
    if any(manifest.get(key) != current[key] for key in ("brief_id", "brief_revision", "brief_sha256")):
        raise AgentError("research_brief_changed_before_model_dispatch", 409)
