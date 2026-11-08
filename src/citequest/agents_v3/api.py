"""Authenticated product router; API requests enqueue, never run the graph."""
from __future__ import annotations

from collections.abc import Callable
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from citequest.v2.domain import Actor, DomainError

from .contracts import AgentError, MemoryDelete, MemoryWrite, ResearchBrief, TaskCommand, TaskCreate
from .database import TaskService
from .memory import MemoryService
from .product_profiles import public_profile_policy


def build_router(service: TaskService, memory: MemoryService, get_actor: Callable,
                 authorize_evidence: Callable[[Actor, UUID, list[str]], None],
                 admit_scope: Callable[[Actor, ResearchBrief], None]) -> APIRouter:
    """get_actor is the shared authenticated session/CSRF dependency, including mutation checks.

    authorize_evidence must verify current ACL/version for stored Task citations; no permissive default.
    """
    router = APIRouter(prefix="/api/agent", tags=["agent"])
    from citequest.skills_v3.api import build_router as skill_router
    from citequest.skills_v3.service import SkillService
    router.include_router(skill_router(SkillService(service.database), get_actor,
        prefix="/projects/{project_id}/skills"))

    def invoke(function: Callable, *args: object, **kwargs: object) -> object:
        try:
            return function(*args, **kwargs)
        except (AgentError, DomainError) as exc:
            raise HTTPException(exc.status, detail={"code": exc.code}) from None

    from .result_projection import ProjectedTaskReader
    project_result = ProjectedTaskReader(service, memory, authorize_evidence).project

    @router.get("/capabilities")
    def capabilities(response: Response, actor: Actor = Depends(get_actor)):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Vary"] = "Cookie"
        return public_profile_policy(service.new_task_profile_policy)

    @router.post("/tasks", status_code=202)
    def create_task(body: TaskCreate, actor: Actor = Depends(get_actor)):
        if body.kind != "research":
            invoke(admit_scope, actor, body.brief)
        value = invoke(service.create, actor.owner_id, body)
        return invoke(project_result, actor, value["id"], value)

    @router.get("/tasks/{task_id}")
    def task(task_id: UUID, actor: Actor = Depends(get_actor)):
        value = invoke(service.get, actor.owner_id, task_id)
        return invoke(project_result, actor, task_id, value)

    @router.get("/tasks/{task_id}/conversation")
    def conversation(task_id: UUID, response: Response, after_seq: Annotated[int, Query(ge=0)] = 0,
                     limit: Annotated[int, Query(ge=1, le=50)] = 20,
                     actor: Actor = Depends(get_actor)) -> object:  # noqa: B008 - factory-local FastAPI dependency
        from .conversation import ConversationReader

        response.headers["Cache-Control"] = "no-store"
        response.headers["Vary"] = "Cookie"
        return invoke(ConversationReader(service, memory, authorize_evidence).page,
                      actor, task_id, after_seq=after_seq, limit=limit)

    @router.post("/tasks/{task_id}/commands", status_code=202)
    def command(task_id: UUID, body: TaskCommand, actor: Actor = Depends(get_actor)):
        value = invoke(service.command, actor.owner_id, task_id, body)
        return invoke(project_result, actor, task_id, value)

    @router.get("/tasks/{task_id}/events")
    def events(task_id: UUID, after_seq: int = 0, actor: Actor = Depends(get_actor)):
        values = invoke(service.events, actor.owner_id, task_id, after_seq)
        return {"events": values, "next_seq": values[-1]["seq"] if values else after_seq}

    @router.get("/projects/{project_id}/memories")
    def memories(project_id: UUID, actor: Actor = Depends(get_actor)):
        return {"memories": invoke(memory.current, actor.owner_id, project_id)}

    @router.put("/projects/{project_id}/memories/{memory_id}")
    def write_memory(project_id: UUID, memory_id: UUID, body: MemoryWrite, actor: Actor = Depends(get_actor)):
        return invoke(memory.write, actor.owner_id, project_id, memory_id, body)

    @router.delete("/projects/{project_id}/memories/{memory_id}")
    def delete_memory(project_id: UUID, memory_id: UUID, body: MemoryDelete, actor: Actor = Depends(get_actor)):
        return invoke(memory.delete, actor.owner_id, project_id, memory_id, body)

    return router
