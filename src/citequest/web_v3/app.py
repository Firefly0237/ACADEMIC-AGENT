"""Unified same-origin API. Dependencies are injected, not silently substituted."""

from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import exists, or_, select, text, tuple_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from citequest.materials_v2.api import create_app as materials_app
from citequest.materials_v2.models import MaterialUpload
from citequest.materials_v2.services import MaterialService
from citequest.search_v3.router import search_router
from citequest.search_v3.service import SearchService
from citequest.search_v3.query_projection import MaterialSimilarInput
from citequest.search_v3.schemas import SearchRequest
from citequest.v2.domain import Actor, DomainError
from citequest.v2.models import File, Work, WorkAccess, WorkVersion
from citequest.v2.pagination import Pager
from citequest.v2.presentation import file as present_file
from citequest.v2.services import Catalog
from citequest.v2.storage import CAS

from .auth import AuthConfig, AuthService


class BodyLimit:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(("/api/agent/", "/api/v3/")):
            return await self.app(scope, receive, send)
        total = 0

        async def bounded_receive() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > 65536:
                    raise DomainError("request_body_limit", 413)
            return message

        return await self.app(scope, bounded_receive, send)


class Consent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation_id: Annotated[UUID, Field(strict=False)]
    expected_revision: int = Field(ge=1)
    allowed: bool


class WorkSelections(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    work_ids: list[Annotated[UUID, Field(strict=False)]] = Field(min_length=1, max_length=10)
    require_full_text: bool = True


def create_app(
    factory: sessionmaker[Session],
    *,
    store: CAS,
    cursor_key: bytes,
    root: Path,
    auth_config: AuthConfig,
    search: SearchService,
    agent_router: APIRouter | None = None,
) -> FastAPI:
    app = materials_app(
        factory,
        store=store,
        cursor_key=cursor_key,
        contract_path=root / "contracts/v2/domain.schema.json",
        policy_path=root / "contracts/v2/policy.json",
        origin=auth_config.policy.origin,
        origin_policy=auth_config.policy,
    )
    auth = AuthService(factory, auth_config)
    app.state.auth_service = auth
    app.state.required_business_tables = {"oidc_identities", "research_briefs", "research_brief_revisions",
        "saved_paper_references", "saved_paper_collections", "saved_paper_tags", "saved_paper_notes", "search_activities"}
    app.title, app.version = "CiteQuest scholarly search and research workspace", "0.7.0"
    app.openapi_schema = None
    app.include_router(auth.router())
    app.include_router(search_router(search, auth.public_actor))
    from citequest.agents_v3.contracts import EvidenceSelection
    from citequest.search_v3.schemas import PaperLocator

    from .workstation_routes import build_router as workstation_router

    def authorize_research_sources(actor: Actor, selections: list[EvidenceSelection]) -> None:
        # Attaching a draft is an authorization operation, not a claim that it
        # has already parsed/indexed or may be sent to the model. Execution has
        # its separate current source-readiness/outbound gate.
        for selection in selections:
            if selection.kind == "material_file":
                with factory.begin() as db:
                    Catalog(db, actor).file(UUID(selection.id))
            else:
                search.detail(actor, PaperLocator(kind=selection.kind, id=selection.id))

    from citequest.search_v3.batch_headers import BatchPaperHeaders
    app.include_router(workstation_router(factory, auth, cursor_key=cursor_key,
        authorize_sources=authorize_research_sources, read_history_papers=BatchPaperHeaders(search).read_many))
    from citequest.workstation_v3.saved_paper_api import build_router as saved_paper_router
    app.include_router(saved_paper_router(factory, auth.private_actor, Pager(cursor_key), search.detail,
        read_many=BatchPaperHeaders(search).read_many))

    PrivateActor = Annotated[Actor, Depends(auth.private_actor)]

    @app.post("/api/v3/workstation/resolve-work-selections")
    def work_selections(body: WorkSelections, actor: PrivateActor) -> dict:
        from citequest.search_v3.work_locator import resolve_work_selections
        return {"items": [item.model_dump(mode="json") for item in resolve_work_selections(
            search, actor, body.work_ids, require_full_text=body.require_full_text)]}
    if agent_router is not None:
        app.include_router(agent_router)

    @app.middleware("http")
    async def boundary(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        try:
            if request.url.path.startswith("/api/v2/") and request.url.path != "/api/v2/capabilities":
                await run_in_threadpool(auth.private_actor, request)
            # Mutation body length also bounded by ASGI receive below. Never trust only Content-Length.
            if request.url.path.startswith(("/api/agent/", "/api/v3/")):
                length = request.headers.get("content-length")
                if length and (not length.isdigit() or int(length) > 65536):
                    return JSONResponse({"code": "request_body_limit"}, status_code=413)
            response = await call_next(request)
        except DomainError as exc:
            response = JSONResponse({"code": exc.code}, status_code=exc.status)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith(("/api/", "/auth/")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/readyz", response_model=None)
    def ready() -> dict | JSONResponse:
        try:
            with factory.begin() as db:
                head = db.execute(text("SELECT version_num FROM cq_app.alembic_version")).scalar_one()
                missing = list(db.scalars(text("SELECT name FROM unnest(CAST(:names AS text[])) AS name "
                    "WHERE to_regclass('cq_app.' || name) IS NULL"), {"names": sorted(app.state.required_business_tables)}))
                if missing:
                    return JSONResponse({"ready": False, "code": "business_schema_not_ready", "missing": missing}, status_code=503)
        except SQLAlchemyError:
            return JSONResponse({"ready": False, "code": "business_database_unavailable"}, status_code=503)
        return {"ready": True, "business_head": head, "cloud_deployment_verified": False}

    @app.get("/api/v3/materials/{file_id}")
    def material_detail(file_id: UUID, actor: PrivateActor) -> dict:
        # V3 adds the concurrency revision explicitly. The published V2 File
        # schema and immutable operation/outbox payloads keep their old shape.
        with factory.begin() as db:
            service = MaterialService(db, actor, store)
            value = service.get_file(file_id)
            file = service.catalog.file(file_id)
            return {**value, "revision": file.revision}

    @app.get("/api/v3/materials/{file_id}/ai-readiness")
    def material_ai_readiness(file_id: UUID, actor: PrivateActor) -> dict:
        with factory.begin() as db:
            file = Catalog(db, actor).file(file_id)
            state = file.status
        material = getattr(app.state, "material_evidence_service", None)
        if material is None:
            return {"ready": False, "code": "material_semantic_index_not_configured", "file_state": state, "model_calls": 0}
        try:
            material.validate_selection(actor, [file_id])
        except DomainError as exc:
            return {"ready": False, "code": exc.code, "file_state": state, "model_calls": 0}
        return {"ready": True, "code": "ready", "file_state": state, "model_calls": 0}

    @app.post("/api/v3/materials/{file_id}/index", status_code=202)
    def index_material(file_id: UUID, actor: PrivateActor) -> dict:
        material = getattr(app.state, "material_evidence_service", None)
        if material is None or material.native_index is None or material.retrieval_mode != "native_hybrid":
            raise DomainError("material_semantic_index_not_configured", 503)
        with factory.begin() as db:
            material._source(db, actor, file_id)
        return material.native_index.enqueue(actor, file_id)

    @app.post("/api/v3/materials/{file_id}/similar")
    def similar_material(file_id: UUID, body: MaterialSimilarInput, actor: PrivateActor) -> dict:
        material = getattr(app.state, "material_evidence_service", None)
        if material is None:
            raise DomainError("material_source_not_configured", 503)
        prepared = material.query_projection(actor, file_id, query=body.query, selection=body.selection,
            expected_source=body.expected_source)
        if body.action == "prepare":
            return {**prepared, "page": None, "model_calls": 0}
        projection = prepared["projection"]
        if projection is None:
            raise DomainError("material_similarity_query_required", 422)
        page = search.search(actor, SearchRequest(q=projection.query, catalog=body.catalog, mode=body.mode,
            page_size=body.page_size, candidate_limit=body.candidate_limit))
        page = search.exclude_similarity_source(actor, page.snapshot_id,
            work_id=UUID(projection.source.work_id) if projection.source.work_id else None, projection=projection)
        material.verify_query_projection(actor, file_id, projection)
        return {**prepared, "page": page, "model_calls": 0}

    @app.get("/api/v3/materials")
    def material_list(
        actor: PrivateActor,
        page_size: int = Query(default=20, ge=1, le=50),
        cursor: str | None = Query(default=None, max_length=4096),
    ) -> dict:
        pager = Pager(cursor_key)
        identity = {"owner": str(actor.owner_id), "kind": "material_files_v3", "limit": page_size}
        with factory.begin() as db:
            permitted_work = exists(
                select(WorkVersion.id)
                .join(Work, Work.id == WorkVersion.work_id)
                .where(
                    WorkVersion.id == File.version_id,
                    or_(
                        Work.public.is_(True),
                        exists(
                            select(WorkAccess.work_id).where(
                                WorkAccess.work_id == Work.id,
                                WorkAccess.owner_id == actor.owner_id,
                                WorkAccess.revoked.is_(False),
                            )
                        ),
                    ),
                )
            )
            statement = select(File).where(
                File.owner_id == actor.owner_id,
                File.security_domain == actor.security_domain,
                File.deleted_at.is_(None),
                or_(File.version_id.is_(None), permitted_work),
            )
            if cursor:
                previous = pager.decode(cursor, identity)
                statement = statement.where(
                    tuple_(File.created_at, File.id)
                    < tuple_(datetime.fromisoformat(previous["created"]), UUID(previous["id"]))
                )
            rows = db.scalars(statement.order_by(File.created_at.desc(), File.id.desc()).limit(page_size + 1)).all()
            material = MaterialService(db, actor, store)
            titles = dict(
                db.execute(
                    select(MaterialUpload.file_id, MaterialUpload.display_title).where(
                        MaterialUpload.owner_id == actor.owner_id,
                        MaterialUpload.file_id.in_([row.id for row in rows[:page_size]]),
                    )
                ).all()
            )
            next_cursor = None
            if len(rows) > page_size:
                from datetime import UTC, timedelta

                last = rows[page_size - 1]
                next_cursor = pager.encode(
                    {
                        "identity": identity,
                        "created": last.created_at.isoformat(),
                        "id": str(last.id),
                        "expires": (datetime.now(UTC) + timedelta(minutes=15)).timestamp(),
                    }
                )
            return {
                "items": [
                    {**material.get_file(row.id), "revision": row.revision, "display_title": titles.get(row.id, "未命名材料")}
                    for row in rows[:page_size]
                ],
                "next_cursor": next_cursor,
                "scope": "current_owner_materials",
                "snapshot_id": None,
            }

    @app.post("/api/v3/materials/{file_id}/processing-consent")
    def consent(file_id: UUID, body: Consent, actor: PrivateActor) -> dict:
        with factory.begin() as db:
            catalog = Catalog(db, actor)
            catalog.lock_owner()

            def effect() -> dict:
                file = catalog.file(file_id)
                if file.revision != body.expected_revision:
                    raise DomainError("revision_conflict", 409)
                if body.allowed and file.rights_status == "restricted":
                    raise DomainError("restricted_material_processing_forbidden", 403)
                file.outbound_policy = "selected_cloud_processing_allowed" if body.allowed else "local_only"
                file.revision += 1
                value = present_file(catalog, file)
                catalog.outbox(file.id, file.revision, "upsert", value, entity_type="File")
                return value

            catalog.operation(
                body.operation_id,
                {
                    "operation": "material_processing_consent",
                    "file_id": str(file_id),
                    "expected_revision": body.expected_revision,
                    "allowed": body.allowed,
                    "processor": "configured_deepseek_api",
                },
                effect,
            )
            file = catalog.file(file_id)  # A replay never bypasses current deletion/ACL.
            # An old grant replay must never make a later revocation appear
            # granted. Preserve the immutable operation receipt in storage and
            # return the current, authorized projection and its actual revision.
            return {**MaterialService(db, actor, store).get_file(file_id), "revision": file.revision}

    if agent_router is None:

        @app.api_route("/api/agent/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"])
        def agent_not_configured(rest: str) -> JSONResponse:
            return JSONResponse({"code": "agent_runtime_not_configured"}, status_code=503)

    app.add_middleware(
        SessionMiddleware,
        secret_key=auth_config.session_secret,
        session_cookie="cq_oidc_state",
        max_age=600,
        same_site="lax",
        https_only=not auth_config.policy.development,
    )
    app.add_middleware(BodyLimit)
    build = root / "web/dist"
    if build.is_dir():
        app.mount("/assets", StaticFiles(directory=build / "assets"), name="web-assets")
        if (build / "pdfjs").is_dir():
            app.mount("/pdfjs", StaticFiles(directory=build / "pdfjs"), name="pdfjs-assets")

        @app.get("/{path:path}")
        def page(path: str) -> Response:
            if path.startswith(("api/", "auth/", "healthz", "readyz")):
                return JSONResponse({"code": "not_found"}, status_code=404)
            return FileResponse(build / "index.html", headers={"Cache-Control": "no-cache"})

    return app
