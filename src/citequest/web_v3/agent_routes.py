"""Public process events use resumable SSE, never private model reasoning."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from citequest.agents_v3.api import build_router
from citequest.agents_v3.contracts import AgentError
from citequest.agents_v3.database import TaskService
from citequest.agents_v3.memory import MemoryService
from citequest.research_v3.evidence import EvidenceAccess
from citequest.v2.domain import Actor

from .auth import AuthService


def product_agent_router(
    service: TaskService, memory: MemoryService, auth: AuthService, evidence: EvidenceAccess
) -> APIRouter:
    router = build_router(service, memory, auth.private_actor, evidence.authorize, evidence.admit_scope)
    PrivateActor = Annotated[Actor, Depends(auth.private_actor)]

    @router.get("/tasks")
    def recent_tasks(actor: PrivateActor, project_id: UUID | None = None) -> dict:
        # Bounded recent list; content is fetched through the fully authorized
        # Task route, not leaked through a background-task summary cache.
        with service.database.connect() as db:
            rows = db.execute("""SELECT id,project_id,kind,state,revision,created_at,updated_at,
                profile->>'profile_id' AS profile_id FROM cq_app.agent_tasks
                WHERE owner_id=%s AND (%s::uuid IS NULL OR project_id=%s)
                ORDER BY created_at DESC,id DESC LIMIT 50""", (actor.owner_id, project_id, project_id)).fetchall()
        return {"items": rows, "scope": "latest_50_owned_tasks", "limit": 50}

    @router.get("/tasks/{task_id}/evidence")
    def read_evidence(task_id: UUID, actor: PrivateActor, ref_id: str = Query(min_length=1, max_length=300)) -> dict:
        try:
            return evidence.read(actor, task_id, ref_id)
        except AgentError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None

    @router.get("/tasks/{task_id}/events/stream")
    async def events(
        task_id: UUID,
        request: Request,
        actor: PrivateActor,
        after_seq: int = Query(default=0, ge=0),
        last_event_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        try:
            service.get(actor.owner_id, task_id)  # authorize before sending headers
        except AgentError as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None
        if last_event_id is not None:
            if not last_event_id.isdigit() or len(last_event_id) > 16:
                raise HTTPException(400, detail={"code": "invalid_event_cursor"})
            after_seq = max(after_seq, int(last_event_id))

        async def stream() -> AsyncIterator[str]:
            cursor, started = after_seq, time.monotonic()
            yield "retry: 1500\n\n"
            while time.monotonic() - started < 25:
                if await request.is_disconnected():
                    return
                try:
                    await run_in_threadpool(auth.private_actor, request)
                    rows = await run_in_threadpool(service.events, actor.owner_id, task_id, cursor)
                    task = await run_in_threadpool(service.get, actor.owner_id, task_id)
                except Exception:
                    yield 'event: unavailable\ndata: {"code":"event_access_unavailable"}\n\n'
                    return
                for row in rows:
                    cursor = row["seq"]
                    data = json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str)
                    yield f"id: {cursor}\nevent: progress\ndata: {data}\n\n"
                if cursor < task["event_seq"]:
                    # A terminal Task may have more than one stored event page.
                    # Drain its authorized history before closing the stream.
                    continue
                if task["state"] not in {"queued", "running"}:
                    yield "event: settled\ndata: {}\n\n"
                    return
                yield ": heartbeat\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
        )

    return router
