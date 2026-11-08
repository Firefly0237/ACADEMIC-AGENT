"""Owner/Task-issued pinned evidence revalidation for saved product answers."""
from __future__ import annotations

import json
from uuid import UUID

from citequest.v2.domain import Actor

from .contracts import AgentError
from .database import Database, owned_task


def task_evidence_authorizer(database: Database, search_service=None, material_service=None):
    def authorize(actor: Actor, task_id: UUID, ref_ids: list[str]) -> None:
        from citequest.search_v3.schemas import EvidenceDTO
        with database.connect() as db:
            owned_task(db, actor.owner_id, task_id)
            pins = {}
            for ref in ref_ids:
                row = db.execute("SELECT payload FROM cq_app.agent_evidence_refs WHERE task_id=%s AND ref_id=%s", (task_id, ref)).fetchone()
                if row is None:
                    raise AgentError("evidence_ref_not_issued_to_task", 403)
                pins[ref] = row["payload"]
        for value in pins.values():
            if value["source_ref"]["kind"] == "material_fragment":
                if material_service is None:
                    raise AgentError("material_evidence_bridge_unavailable", 503)
                material_service.read_pinned(actor, value)
            else:
                if search_service is None:
                    raise AgentError("paper_evidence_bridge_unavailable", 503)
                search_service.read_pinned_evidence(actor, EvidenceDTO.model_validate_json(json.dumps(value), strict=True))
    return authorize
