"""Authenticated queue/download routes; the shared dependency enforces CSRF."""
# ruff: noqa: B008 -- FastAPI's supported dependency declaration; get_actor is server-injected

from __future__ import annotations

from collections.abc import Callable
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from citequest.agents_v3.contracts import AgentError
from citequest.v2.domain import Actor, DomainError

from .contracts import ExportCancel, ExportCreate, ExportDTO, ExportPageDTO
from .service import ExportService


def build_router(service: ExportService, get_actor: Callable) -> APIRouter:
    router = APIRouter(prefix="/api/agent", tags=["school-document-exports"])

    def invoke(action: Callable, *args: object, **kwargs: object) -> object:
        try:
            service.ready()
            return action(*args, **kwargs)
        except (AgentError, DomainError) as error:
            raise HTTPException(error.status, detail={"code": error.code}) from None
        except ValueError:
            raise HTTPException(409, detail={"code": "export_document_or_artifact_contract_failed"}) from None

    @router.post("/tasks/{task_id}/exports", status_code=202, responses={202: {"model": ExportDTO}})
    def create(task_id: UUID, body: ExportCreate, actor: Actor = Depends(get_actor)) -> object:
        return invoke(service.create, actor, task_id, body)

    @router.get("/tasks/{task_id}/exports", responses={200: {"model": ExportPageDTO}})
    def page(
        task_id: UUID,
        actor: Actor = Depends(get_actor),
        limit: int = Query(10, ge=1, le=20),
        cursor: str | None = Query(None, max_length=4096),
    ) -> object:
        return invoke(service.page, actor, task_id, limit=limit, cursor=cursor)

    @router.get("/exports/{identifier}", responses={200: {"model": ExportDTO}})
    def get(identifier: UUID, actor: Actor = Depends(get_actor)) -> object:
        return invoke(service.get, actor, identifier)

    @router.post("/exports/{identifier}/cancel", responses={200: {"model": ExportDTO}})
    def cancel(identifier: UUID, body: ExportCancel, actor: Actor = Depends(get_actor)) -> object:
        return invoke(service.cancel, actor, identifier, body)

    @router.get("/exports/{identifier}/artifacts/{kind}")
    def content(identifier: UUID, kind: Literal["docx", "html", "pdf"], actor: Actor = Depends(get_actor)) -> Response:
        data, media, filename = invoke(service.download, actor, identifier, kind)
        return Response(
            data,
            media_type=media,
            headers={
                "Content-Disposition": 'attachment; filename="' + filename + '"',
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'none'; sandbox",
            },
        )

    return router
