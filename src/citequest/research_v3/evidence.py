"""Current-source authorization bridge for persisted Task results and citations."""

from __future__ import annotations

import json
from uuid import UUID

from citequest.agents_v3.contracts import AgentError, ResearchBrief
from citequest.agents_v3.database import Database, owned_task
from citequest.material_evidence_v3 import MaterialEvidenceService
from citequest.search_v3.schemas import EvidenceDTO, PaperLocator
from citequest.search_v3.service import SearchService
from citequest.v2.domain import Actor

from .source_windows import FullSourceWindows
from .selections import reference_selections


class EvidenceAccess:
    def __init__(self, database: Database, search: SearchService, materials: MaterialEvidenceService | None) -> None:
        self.database, self.search, self.materials = database, search, materials

    def admit_scope(self, actor: Actor, brief: ResearchBrief) -> None:
        admit_scope(actor, brief, self.search, self.materials, catalog_scope_policy=self.database.catalog_scope_policy)

    def read(self, actor: Actor, task_id: UUID, ref_id: str) -> dict:
        if not isinstance(ref_id, str) or not 1 <= len(ref_id) <= 300:
            raise AgentError("invalid_evidence_reference", 400)
        with self.database.connect() as db:
            task = owned_task(db, actor.owner_id, task_id)
            brief = ResearchBrief.model_validate_json(json.dumps(task["brief"]), strict=True)
            row = db.execute(
                "SELECT payload FROM cq_app.agent_evidence_refs WHERE task_id=%s AND ref_id=%s", (task_id, ref_id)
            ).fetchone()
            if row is None:
                raise AgentError("evidence_not_issued_to_task", 404)
            pinned = row["payload"]
        if task['profile'].get('catalog_scope_policy') is not None:
            from citequest.agents_v3.catalog_scope import CatalogPins
            CatalogPins(self.database, self.search).authorize_evidence(actor, task_id, [pinned])
        source = pinned.get("source_ref") or {}
        selected = {(item.kind, item.id) for item in brief.selected_papers}
        if pinned.get("evidence_type") == "source_window":
            selection = pinned.get("selection") or {}
            if (selection.get("kind"), selection.get("id")) not in selected:
                raise AgentError("source_window_not_in_task_selection", 403)
            FullSourceWindows(self.search, self.materials).verify_pinned_batch(actor, [pinned])
            return pinned
        if source.get("kind") == "material_fragment":
            if ("material_file", pinned.get("file_id")) not in selected:
                raise AgentError("material_not_in_task_selection", 403)
            if self.materials is None:
                raise AgentError("material_evidence_not_configured", 503)
            return self.materials.read_pinned(actor, pinned)
        if source.get("kind") not in {"metadata_projection", "native_representation"}:
            raise AgentError("nonproduct_evidence_not_allowed", 403)
        paper_scope = {(item["kind"], item["id"]) for item in reference_selections(brief)}
        if brief.kind == "research" and brief.research_action == "extract" and (source["kind"], source["id"]) not in paper_scope:
            raise AgentError("paper_not_in_draft_extract_reference_scope", 403)
        if paper_scope and (source["kind"], source["id"]) not in paper_scope:
            raise AgentError("paper_not_in_task_selection", 403)
        value = EvidenceDTO.model_validate_json(json.dumps(pinned), strict=True)
        return self.search.read_pinned_evidence(actor, value).model_dump(mode="json")

    def authorize(self, actor: Actor, task_id: UUID, refs: list[str]) -> None:
        if len(refs) > 256:
            raise AgentError("evidence_result_limit")
        with self.database.connect() as db:
            task = owned_task(db, actor.owner_id, task_id)
            brief = ResearchBrief.model_validate_json(json.dumps(task["brief"]), strict=True)
            unique = set(refs)
            rows = db.execute("SELECT ref_id,payload FROM cq_app.agent_evidence_refs WHERE task_id=%s AND ref_id=ANY(%s)",
                (task_id, list(unique))).fetchall()
        if {item["ref_id"] for item in rows} != unique:
            raise AgentError("evidence_not_issued_to_task", 404)
        if task['profile'].get('catalog_scope_policy') is not None:
            from citequest.agents_v3.catalog_scope import CatalogPins
            CatalogPins(self.database, self.search).authorize_evidence(actor, task_id, [item['payload'] for item in rows])
        selected = {(item.kind, item.id) for item in brief.selected_papers}
        windows = []
        for item in rows:
            pinned = item["payload"]
            if pinned.get("evidence_type") == "source_window":
                selection = pinned.get("selection") or {}
                if (selection.get("kind"), selection.get("id")) not in selected:
                    raise AgentError("source_window_not_in_task_selection", 403)
                windows.append(pinned)
            else:
                self.read(actor, task_id, item["ref_id"])
        for start in range(0, len(windows), 64):
            FullSourceWindows(self.search, self.materials).verify_pinned_batch(actor, windows[start:start + 64])


def draft_selections(brief: dict | ResearchBrief) -> list[dict]:
    """Server-validated draft subset; references never become drafts implicitly."""
    value = brief if isinstance(brief, dict) else brief.model_dump(mode="json")
    if value.get("kind") != "research" or value.get("research_action") != "extract":
        raise AgentError("research_extract_action_required", 422)
    files = value.get("research_draft_files", [])
    if not isinstance(files, list) or not 1 <= len(files) <= 4:
        raise AgentError("research_draft_files_required", 422)
    try:
        identifiers = [str(UUID(str(identifier))) for identifier in files]
    except (ValueError, TypeError):
        raise AgentError("research_draft_file_identity_invalid", 422) from None
    allowed = {str(item["id"]) for item in value.get("selected_papers", []) if item.get("kind") == "material_file"}
    if len(identifiers) > 4 or len(set(identifiers)) != len(identifiers) or not set(identifiers) <= allowed:
        raise AgentError("research_drafts_must_be_explicit_selected_materials", 422)
    return [{"kind": "material_file", "id": identifier} for identifier in identifiers]


def reference_allows(brief: dict | ResearchBrief, pinned: dict) -> bool:
    """Current literature role only; prior issued draft bytes stay in audit."""
    references = reference_selections(brief)
    source = pinned.get("source_ref", {})
    if pinned.get("evidence_type") == "source_window":
        key = pinned.get("selection", {})
    elif source.get("kind") == "material_fragment":
        key = {"kind": "material_file", "id": pinned.get("file_id")}
    else:
        key = source
    if references:
        return (key.get("kind"), key.get("id")) in {(item["kind"], item["id"]) for item in references}
    return key.get("kind") in {"metadata_projection", "native_representation"}


def admit_scope(actor: Actor, brief: ResearchBrief, search: SearchService, materials: MaterialEvidenceService | None,
                *, catalog_scope_policy: str | None = None) -> None:
    if brief.kind == "research" and brief.research_action == "extract":
        FullSourceWindows(search, materials).plan(actor, draft_selections(brief))
        return
    if brief.kind in {"summary", "compare"}:
        FullSourceWindows(search, materials).plan(actor, [item.model_dump(mode="json") for item in brief.selected_papers])
        return
    references = reference_selections(brief)
    files = [UUID(item["id"]) for item in references if item["kind"] == "material_file"]
    papers = [PaperLocator(kind=item["kind"], id=item["id"]) for item in references if item["kind"] != "material_file"]
    if files:
        if materials is None:
            raise AgentError("material_semantic_index_not_configured", 503)
        materials.validate_selection(actor, files)
    # An exclusively selected-material Task must not implicitly expand into the paper catalog.
    if papers or not files:
        if catalog_scope_policy is None:
            search.preflight(actor, brief.catalog, papers)
        else:
            from citequest.agents_v3.catalog_scope import POLICY, required_scopes
            if catalog_scope_policy != POLICY:
                raise AgentError('unsupported_catalog_scope_policy', 503)
            for catalog, selected in required_scopes(brief.model_dump(mode='json')).items():
                search.preflight(actor, catalog, selected)
