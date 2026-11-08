"""Typed bookmark HTTP adapter using the parent's current identity/CSRF boundary."""

from collections.abc import Callable, Iterator
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from citequest.search_v3.batch_headers import CurrentPaperHeader
from citequest.search_v3.schemas import PaperDetail, PaperLocator
from citequest.v2.domain import Actor, DomainError
from citequest.v2.pagination import Pager
from citequest.v2.services import Catalog

from .saved_paper_contracts import (
    PaperActivity,
    SavedPaperCreate,
    SavedPaperLifecycle,
    SavedPaperNoteCreate,
    SavedPaperNoteDelete,
    SavedPaperNotePatch,
    SavedPaperPatch,
)
from .saved_papers import SavedPaperService


def build_router(
    factory: sessionmaker[Session],
    get_actor: Callable,
    pager: Pager,
    read_paper: Callable[[Actor, PaperLocator], PaperDetail],
    *,
    read_many: Callable[[Actor, list[PaperLocator]], dict[str, CurrentPaperHeader]] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/v3/workstation", tags=["saved-papers"])

    def service(actor: Annotated[Actor, Depends(get_actor)]) -> Iterator[SavedPaperService]:
        with factory.begin() as db:
            ready = db.execute(
                text(
                    "SELECT to_regclass('cq_app.saved_paper_references') IS NOT NULL AND to_regclass('cq_app.saved_paper_notes') IS NOT NULL AND to_regclass('cq_app.saved_paper_collections') IS NOT NULL AND to_regclass('cq_app.saved_paper_tags') IS NOT NULL"
                )
            ).scalar_one()
            if not ready:
                raise HTTPException(503, detail={"code": "saved_paper_migration_required"})
            yield SavedPaperService(Catalog(db, actor), pager, read_paper, read_many=read_many)

    Service = Annotated[SavedPaperService, Depends(service)]

    def invoke(function: Callable, *args: object, **kwargs: object) -> Any:
        try:
            return function(*args, **kwargs)
        except DomainError as error:
            raise HTTPException(error.status, detail={"code": error.code}) from None

    @router.post("/saved-papers", status_code=201)
    def create(body: SavedPaperCreate, current: Service) -> dict:
        return invoke(current.create, body)

    @router.get("/saved-papers")
    def page(
        current: Service,
        limit: int = Query(20, ge=1, le=50),
        cursor: str | None = Query(None, max_length=4096),
        query: str = Query("", max_length=200),
        collection_id: UUID | None = None,
        tag: str | None = Query(None, max_length=100),
    ) -> dict:
        return invoke(current.page, limit=limit, cursor=cursor, query=query, collection_id=collection_id, tag=tag)

    @router.get("/saved-papers/{identifier}")
    def get(identifier: UUID, current: Service) -> dict:
        return invoke(current.get, identifier)

    @router.patch("/saved-papers/{identifier}")
    def patch(identifier: UUID, body: SavedPaperPatch, current: Service) -> dict:
        return invoke(current.patch, identifier, body)

    @router.post("/saved-papers/{identifier}/lifecycle")
    def lifecycle(identifier: UUID, body: SavedPaperLifecycle, current: Service) -> dict:
        return invoke(current.lifecycle, identifier, body)

    @router.get("/saved-papers/{identifier}/notes")
    def notes(
        identifier: UUID,
        current: Service,
        limit: int = Query(20, ge=1, le=50),
        cursor: str | None = Query(None, max_length=4096),
    ) -> dict:
        return invoke(current.notes, identifier, limit=limit, cursor=cursor)

    @router.post("/saved-papers/{identifier}/notes", status_code=201)
    def create_note(identifier: UUID, body: SavedPaperNoteCreate, current: Service) -> dict:
        return invoke(current.create_note, identifier, body)

    @router.patch("/saved-paper-notes/{identifier}")
    def patch_note(identifier: UUID, body: SavedPaperNotePatch, current: Service) -> dict:
        return invoke(current.patch_note, identifier, body)

    @router.delete("/saved-paper-notes/{identifier}")
    def delete_note(identifier: UUID, body: SavedPaperNoteDelete, current: Service) -> dict:
        return invoke(current.delete_note, identifier, body)

    @router.get("/library/{identifier}/notes")
    def legacy_notes(
        identifier: UUID,
        current: Service,
        limit: int = Query(20, ge=1, le=50),
        cursor: str | None = Query(None, max_length=4096),
    ) -> dict:
        return invoke(current.legacy_notes, identifier, limit=limit, cursor=cursor)

    @router.post("/paper-activity")
    def paper_activity(body: PaperActivity, current: Service) -> dict:
        return invoke(current.record_view, body.operation_id, body.paper)

    return router
