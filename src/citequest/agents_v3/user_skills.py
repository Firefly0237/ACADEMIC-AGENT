"""Explicit pinned instruction skills, with fresh guards at each trust boundary."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .database import Database

from .contracts import AgentError, ResearchBrief


def activate_for_task(database: Database | None, task: dict) -> dict:
    brief = ResearchBrief.model_validate_json(json.dumps(task["brief"]), strict=True)
    if not brief.selected_skills:
        return {"pins": [], "instructions": [], "total_utf8_bytes": 0}
    if database is None:
        raise AgentError("selected_skill_database_not_configured", 503)
    from citequest.skills_v3.service import SkillService
    return SkillService(database).activate(task["owner_id"], task["project_id"], brief.selected_skills)


def validate_task_skills(database: Database | None, task: dict, payload: dict | None = None) -> None:
    brief = ResearchBrief.model_validate_json(json.dumps(task["brief"]), strict=True)
    pins = [item.model_dump(mode="json") for item in brief.selected_skills]
    if payload is not None:
        prompt = json.loads(payload["input"][0]["content"])
        if prompt.get("context_manifest", {}).get("skill_refs", []) != pins:
            raise AgentError("selected_skill_context_mismatch")
    if brief.selected_skills:
        if database is None:
            raise AgentError("selected_skill_database_not_configured", 503)
        from citequest.skills_v3.service import SkillService
        SkillService(database).validate_current(task["owner_id"], task["project_id"], brief.selected_skills)
