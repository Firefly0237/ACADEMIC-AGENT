"""Skill HTTP adapter; the host supplies authenticated Actor + mutation CSRF."""

from collections.abc import Callable
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from citequest.agents_v3.contracts import AgentError
from citequest.v2.domain import Actor

from .contracts import SkillLifecycle, SkillWrite
from .service import SkillService


def build_router(service: SkillService, get_actor: Callable, *,
                 prefix: str = "/api/agent/projects/{project_id}/skills") -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["user-skills"])

    def invoke(function: Callable, *args: object, **kwargs: object) -> Any:
        try:
            return function(*args, **kwargs)
        except AgentError as error:
            raise HTTPException(error.status, detail={"code": error.code}) from None

    @router.get("")
    def catalog(project_id: UUID, actor: Annotated[Actor, Depends(get_actor)], limit: int = Query(20, ge=1, le=50),
                after_name: str = Query("", max_length=64), include_deleted: bool = False) -> dict:
        return invoke(service.catalog, actor.owner_id, project_id, limit=limit, after_name=after_name, include_deleted=include_deleted)

    @router.put("/{skill_id}")
    def write(project_id: UUID, skill_id: UUID, body: SkillWrite, actor: Annotated[Actor, Depends(get_actor)]) -> dict:
        return invoke(service.write, actor.owner_id, project_id, skill_id, body)

    @router.post("/{skill_id}/lifecycle")
    def lifecycle(project_id: UUID, skill_id: UUID, body: SkillLifecycle, actor: Annotated[Actor, Depends(get_actor)]) -> dict:
        return invoke(service.lifecycle, actor.owner_id, project_id, skill_id, body)

    @router.get("/{skill_id}/versions")
    def versions(project_id: UUID, skill_id: UUID, actor: Annotated[Actor, Depends(get_actor)], limit: int = Query(20, ge=1, le=50),
                 before_revision: int = Query(2147483647, ge=1)) -> dict:
        return invoke(service.versions, actor.owner_id, project_id, skill_id, limit=limit, before_revision=before_revision)

    @router.get("/{skill_id}/versions/{version_id}")
    def read_version(project_id: UUID, skill_id: UUID, version_id: UUID, actor: Annotated[Actor, Depends(get_actor)]) -> dict:
        return invoke(service.read_version, actor.owner_id, project_id, skill_id, version_id)

    return router
