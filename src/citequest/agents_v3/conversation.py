"""Owner-scoped user turns and currently authorized, durably retained answers.

The four-answer model-context cache is not a conversation archive. Older
commands without a paired private completion event retain their user message
and explicitly report not_retained. Checkpoints and raw model IO are never read.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from uuid import UUID

from citequest.v2.domain import Actor

from .contracts import AgentError
from .database import TaskService, owned_task
from .memory import MemoryService
from .result_projection import ProjectedTaskReader

VERSION = "citequest-conversation-turn/1"


class ConversationReader:
    def __init__(self, service: TaskService, memory: MemoryService,
                 authorize_evidence: Callable[[Actor, UUID, list[str]], None]) -> None:
        self.service = service
        self.reader = ProjectedTaskReader(service, memory, authorize_evidence)

    def page(self, actor: Actor, task_id: UUID, *, after_seq: int = 0, limit: int = 20) -> dict:
        if type(after_seq) is not int or after_seq < 0 or type(limit) is not int or not 1 <= limit <= 50:
            raise AgentError("invalid_conversation_cursor_or_limit", 400)
        with self.service.database.connect() as db:
            db.execute("SET TRANSACTION READ ONLY")
            task = owned_task(db, actor.owner_id, task_id)
            commands = db.execute(
                """SELECT id,seq,kind,payload,state,created_at FROM cq_app.agent_commands
                WHERE task_id=%s AND owner_id=%s AND kind <> 'cancel' AND seq>%s
                ORDER BY seq LIMIT %s""", (task_id, actor.owner_id, after_seq, limit + 1)
            ).fetchall()
            identifiers = [str(command["id"]) for command in commands[:limit]]
            records = db.execute(
                """SELECT payload FROM cq_app.agent_events WHERE task_id=%s
                AND event_type='conversation_turn_completed'
                AND payload->>'command_id'=ANY(%s) ORDER BY seq""", (task_id, identifiers)
            ).fetchall() if identifiers else []
        grouped = defaultdict(list)
        for record in records:
            grouped[record["payload"].get("command_id")].append(record["payload"])
        items = []
        for command in commands[:limit]:
            payload = command["payload"]
            text = (payload.get("brief", {}).get("question", "") if command["kind"] == "start"
                    else payload.get("message", ""))
            row = {"command_seq": command["seq"], "command_kind": command["kind"], "user_text": text,
                   "created_at": command["created_at"], "state": command["state"], "brief_revision": None,
                   "answer": None, "answer_availability": "not_retained"}
            paired = grouped.get(str(command["id"]), [])
            if len(paired) > 1:
                row["answer_availability"] = "ambiguous_turn_record"
            elif paired:
                record = paired[0]
                if (record.get("version") != VERSION or record.get("command_seq") != command["seq"]
                        or type(record.get("brief_revision")) is not int or record["brief_revision"] < 1
                        or not isinstance(record.get("brief"), dict) or record["brief"].get("question") != text):
                    row["answer_availability"] = "invalid_turn_binding"
                else:
                    row.update(state=record["state"], brief_revision=record["brief_revision"])
                    if record.get("answer") is not None:
                        historical = self.service.public({**task, "brief": record["brief"],
                            "revision": record["brief_revision"], "result": record["answer"]},
                            catalog_scope_policy=self.service.database.catalog_scope_policy)
                        projected = self.reader.project(actor, task_id, historical)
                        row.update(answer=projected.get("result"),
                                   answer_availability=projected.get("result_availability", "unavailable"))
                    else:
                        row["answer_availability"] = "no_answer"
            elif command["state"] in {"queued", "running"}:
                row["answer_availability"] = "pending"
            items.append(row)
        return {"items": items, "next_after_seq": items[-1]["command_seq"] if len(commands) > limit else None}
