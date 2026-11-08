"""Authenticated HTTP composition; root application explicitly mounts this router."""

from collections.abc import Callable, Iterator
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from citequest.agents_v3.contracts import EvidenceSelection
from citequest.research_v3.brief import BriefCreate, BriefLifecycle, BriefResolution, BriefTurn
from citequest.research_v3.brief_service import BriefService
from citequest.v2.domain import Actor, DomainError
from citequest.v2.pagination import Pager
from citequest.v2.presentation import collection as present_collection
from citequest.v2.presentation import history as present_history
from citequest.v2.services import Catalog

from .contracts import Activity, BatchSave, ClearHistory
from .search_activity import SearchActivityInput, project_searches, record_search
from .service import WorkstationService


def build_router(
    factory: sessionmaker[Session],
    get_actor: Callable,
    pager: Pager,
    authorize_sources: Callable[[Actor, list[EvidenceSelection]], None],
    *,
    read_history_papers: Callable | None = None,
) -> APIRouter:
    """get_actor must be the shared signed-session/CSRF dependency, never a body owner."""
    router = APIRouter(prefix="/api/v3/workstation", tags=["workstation"])

    def catalog(actor: Annotated[Actor, Depends(get_actor)]) -> Iterator[Catalog]:
        with factory.begin() as db:
            yield Catalog(db, actor)

    Service = Annotated[Catalog, Depends(catalog)]

    def invoke(function: Callable, *args: object, **kwargs: object) -> Any:
        try:
            return function(*args, **kwargs)
        except DomainError as error:
            raise HTTPException(error.status, detail={"code": error.code}) from None

    def brief(service: Catalog) -> BriefService:
        ready = service.db.execute(
            text(
                "SELECT to_regclass('cq_app.research_briefs') IS NOT NULL AND to_regclass('cq_app.research_brief_revisions') IS NOT NULL"
            )
        ).scalar_one()
        if not ready:
            raise HTTPException(503, detail={"code": "research_intake_migration_required"})
        return BriefService(service, authorize_sources)

    @router.post("/library/batch")
    def save_batch(body: BatchSave, service: Service) -> dict:
        return invoke(WorkstationService(service, pager).save_batch, body)

    @router.get("/library")
    def library(
        service: Service,
        limit: int = Query(20, ge=1, le=50),
        cursor: str | None = Query(None, max_length=4096),
        query: str = Query("", max_length=200),
        collection_id: UUID | None = None,
        tag: str | None = Query(None, max_length=500),
    ) -> dict:
        return invoke(
            WorkstationService(service, pager).library,
            limit=limit,
            cursor=cursor,
            query=query,
            collection_id=collection_id,
            tag=tag,
        )

    @router.get("/collections")
    def collections(
        service: Service, limit: int = Query(20, ge=1, le=50), cursor: str | None = Query(None, max_length=4096)
    ) -> dict:
        rows, next_cursor = invoke(pager.page, service, "collections", limit, cursor)
        return {"items": [present_collection(row) for row in rows], "next_cursor": next_cursor}

    @router.get("/history")
    def history(
        service: Service, limit: int = Query(20, ge=1, le=50), cursor: str | None = Query(None, max_length=4096)
    ) -> dict:
        rows, next_cursor = invoke(pager.page, service, "history", limit, cursor)
        from .history_projection import project_history
        return {"items": project_history(invoke(project_searches, service, [present_history(row) for row in rows]), service.actor, read_history_papers),
                "next_cursor": next_cursor}

    @router.post("/search-activities", status_code=201)
    def search_activity(body: SearchActivityInput, service: Service) -> dict:
        return invoke(record_search, service, body)

    @router.post("/activity")
    def activity(body: Activity, service: Service) -> dict:
        return invoke(WorkstationService(service, pager).record_activity, body)

    @router.post("/history/clear")
    def clear_history(body: ClearHistory, service: Service) -> dict:
        return invoke(WorkstationService(service, pager).clear_history, body)

    @router.get("/tasks")
    def tasks(
        service: Service,
        project_id: UUID | None = None,
        limit: int = Query(20, ge=1, le=50),
        cursor: str | None = Query(None, max_length=4096),
    ) -> dict:
        return invoke(WorkstationService(service, pager).tasks, project_id=project_id, limit=limit, cursor=cursor)

    @router.post("/briefs", status_code=201)
    def create_brief(body: BriefCreate, service: Service) -> dict:
        return invoke(brief(service).create, body)

    @router.get("/projects/{project_id}/briefs")
    def list_briefs(
        project_id: UUID,
        service: Service,
        limit: int = Query(20, ge=1, le=50),
        cursor: str | None = Query(None, max_length=4096),
    ) -> dict:
        return invoke(brief(service).list, project_id, pager, limit=limit, cursor=cursor)

    @router.get("/briefs/{brief_id}")
    def get_brief(brief_id: UUID, service: Service) -> dict:
        return invoke(brief(service).get, brief_id)

    @router.patch("/briefs/{brief_id}")
    def update_brief(brief_id: UUID, body: BriefTurn, service: Service) -> dict:
        return invoke(brief(service).turn, brief_id, body)

    @router.post("/briefs/{brief_id}/resolutions")
    def resolve_brief(brief_id: UUID, body: BriefResolution, service: Service) -> dict:
        return invoke(brief(service).resolution, brief_id, body)

    @router.post("/briefs/{brief_id}/lifecycle")
    def lifecycle(brief_id: UUID, body: BriefLifecycle, service: Service) -> dict:
        return invoke(
            brief(service).set_deleted, brief_id, body.operation_id, body.expected_revision, deleted=body.deleted
        )

    @router.get("/briefs/{brief_id}/planning-input")
    def planning(brief_id: UUID, service: Service, expected_revision: int = Query(ge=1)) -> dict:
        return invoke(brief(service).planning, brief_id, expected_revision)

    return router
