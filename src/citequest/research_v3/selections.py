"""Pure source-selection rules shared by admission and authorized retrieval."""
from __future__ import annotations

from uuid import UUID

from citequest.agents_v3.contracts import AgentError, ResearchBrief


def reference_selections(brief: dict | ResearchBrief) -> list[dict]:
    """Research draft markers are intake inputs, never implicit literature."""
    value = brief if isinstance(brief, dict) else brief.model_dump(mode="json")
    selected = value.get("selected_papers", [])
    if value.get("kind") != "research":
        return selected
    files = value.get("research_draft_files", [])
    try:
        identifiers = [str(UUID(str(identifier))) for identifier in files]
    except (ValueError, TypeError):
        raise AgentError("research_draft_file_identity_invalid", 422) from None
    allowed = {item["id"] for item in selected if item["kind"] == "material_file"}
    if len(identifiers) > 4 or len(set(identifiers)) != len(identifiers) or not set(identifiers) <= allowed:
        raise AgentError("research_drafts_must_be_explicit_selected_materials", 422)
    return [item for item in selected if not (item["kind"] == "material_file" and item["id"] in identifiers)]
