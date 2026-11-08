"""Thin S03 HTTP adapter. Caller provisions engine and trusted local sessions."""

import json
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator, FormatChecker
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker
from starlette.responses import Response

from . import models as m
from . import presentation as p
from .domain import (
    CollectionCreate,
    CollectionPatch,
    DomainError,
    LibraryPatch,
    LibrarySave,
    NoteCreate,
    NotePatch,
    WorkInput,
)
from .openapi import response_contracts
from .pagination import Pager
from .security import authenticate
from .origin import OriginPolicy
from .services import Catalog


def create_app(
    factory: sessionmaker[Session],
    *,
    cursor_key: bytes,
    contract_path: Path,
    origin: str = "http://127.0.0.1:18000",
    include_file_routes: bool = True,
    origin_policy: OriginPolicy | None = None,
) -> FastAPI:
    if origin_policy is not None and origin_policy.origin != origin:
        raise ValueError("origin_policy_mismatch")
    if origin_policy is None and origin not in {"http://127.0.0.1:18000", "http://localhost:18000"}:
        raise ValueError("Explicit supported loopback origin required")
    schema = json.loads(contract_path.read_text(encoding="utf-8"))
    if not {"uri", "date-time", "uuid"} <= set(FormatChecker.checkers):
        raise RuntimeError("Required JSON Schema format validators are unavailable")
    validators = {
        name: Draft202012Validator({**schema, "$ref": "#/$defs/" + name}, format_checker=FormatChecker())
        for name in schema["$defs"]
    }
    pager = Pager(cursor_key)
    app = FastAPI(title="CiteQuest v2 domain service", version="0.3.0")
    router = APIRouter(prefix="/api/v2")

    def checked(name: str, value: dict) -> dict:
        if not validators[name].is_valid(value):
            raise DomainError("response_contract_failed", 500)
        return value

    @app.middleware("http")
    async def request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

    def error(request: Request, code: str, status: int) -> JSONResponse:
        value = {
            "code": code,
            "message": code,
            "request_id": request.state.request_id,
            "retryable": status == 503,
            "details": {},
            "capability_state": None,
        }
        return JSONResponse(checked("Error", value), status_code=status)

    @app.exception_handler(DomainError)
    async def domain_error(request: Request, exc: DomainError) -> JSONResponse:
        return error(request, exc.code, exc.status)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error(request, "invalid_input", 400)

    @app.exception_handler(IntegrityError)
    async def conflict(request: Request, exc: IntegrityError) -> JSONResponse:
        return error(request, "constraint_conflict", 409)

    @app.exception_handler(OperationalError)
    async def unavailable(request: Request, exc: OperationalError) -> JSONResponse:
        return error(request, "persistence_unavailable", 503)

    def database() -> Iterator[Session]:
        # Function scope commits before HTTP send; a commit failure cannot
        # silently follow a success response. No persistent Session singleton.
        with factory.begin() as db:
            yield db

    DB = Annotated[Session, Depends(database, scope="function")]

    def service(request: Request, db: DB) -> Catalog:
        mutation = request.method not in {"GET", "HEAD", "OPTIONS"}
        if mutation and request.headers.get("origin") != origin:
            raise DomainError("same_origin_required", 403)
        actor = authenticate(
            db, request.cookies.get("citequest_session"), csrf=request.headers.get("x-csrf-token"), mutation=mutation
        )
        return Catalog(db, actor)

    Service = Annotated[Catalog, Depends(service)]

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        return {"status": "ok"}  # Liveness never connects to dependencies.

    @router.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        names = {"domain_crud": "implemented", "upload_parser": "planned", "search": "planned", "agent": "planned"}
        return checked(
            "CapabilityManifest",
            {
                "capabilities": [
                    {
                        "name": n,
                        "state": state,
                        "requested_mode": None,
                        "effective_mode": None,
                        "fallbacks": [],
                        "missing_evidence": ["release capability registration pending; see stage acceptance package"]
                        if n == "domain_crud"
                        else ["later-stage implementation"],
                        "warnings": [],
                        "verified_at": None,
                        "verification_reference": None,
                    }
                    for n, state in names.items()
                ],
                "corpus_versions": [],
                "active_generation": None,
            },
        )

    @router.post("/works", status_code=201)
    def create_work(body: WorkInput, catalog: Service) -> dict[str, Any]:
        result = catalog.create_work(body)
        return checked("Work", result)

    @router.get("/works/{work_id}")
    def get_work(work_id: UUID, catalog: Service) -> dict[str, Any]:
        return checked("Work", p.work(catalog, catalog.work(work_id)))

    @router.post("/library/items", status_code=201)
    def save(body: LibrarySave, catalog: Service) -> dict[str, Any]:
        result = catalog.save(body)
        return checked("LibraryItem", result)

    @router.get("/library/items/{library_item_id}")
    def get_item(library_item_id: UUID, catalog: Service) -> dict[str, Any]:
        return checked("LibraryItem", p.item(catalog, catalog.item(library_item_id)))

    @router.patch("/library/items/{library_item_id}")
    def patch_item(library_item_id: UUID, body: LibraryPatch, catalog: Service) -> dict[str, Any]:
        catalog.patch_item(library_item_id, body)
        return checked("LibraryItem", p.item(catalog, catalog.item(library_item_id)))

    @router.delete("/library/items/{library_item_id}")
    def delete_item(library_item_id: UUID, catalog: Service) -> dict[str, Any]:
        catalog.delete_item(library_item_id)
        return checked("LibraryItem", p.item(catalog, catalog.item(library_item_id, deleted=True)))

    @router.post("/library/items/{library_item_id}/restore")
    def restore_item(library_item_id: UUID, catalog: Service) -> dict[str, Any]:
        catalog.restore_item(library_item_id)
        return checked("LibraryItem", p.item(catalog, catalog.item(library_item_id)))

    @router.post("/collections", status_code=201)
    def collection(body: CollectionCreate, catalog: Service) -> dict[str, Any]:
        result = catalog.create_collection(body)
        return checked("Collection", result)

    @router.patch("/collections/{collection_id}")
    def patch_collection(collection_id: UUID, body: CollectionPatch, catalog: Service) -> dict[str, Any]:
        catalog.patch_collection(collection_id, body)
        return checked("Collection", p.collection(catalog.collection(collection_id)))

    @router.delete("/collections/{collection_id}")
    def delete_collection(collection_id: UUID, catalog: Service) -> dict[str, Any]:
        obj = catalog.collection(collection_id)
        catalog.delete_collection(collection_id)
        return checked("Collection", p.collection(obj))

    @router.post("/notes", status_code=201)
    def note(body: NoteCreate, catalog: Service) -> dict[str, Any]:
        result = catalog.create_note(body)
        return checked("Note", result)

    @router.patch("/notes/{note_id}")
    def patch_note(note_id: UUID, body: NotePatch, catalog: Service) -> dict[str, Any]:
        catalog.patch_note(note_id, body)
        return checked("Note", p.note(catalog.note(note_id)))

    @router.delete("/notes/{note_id}")
    def delete_note(note_id: UUID, catalog: Service) -> dict[str, Any]:
        obj = catalog.note(note_id)
        catalog.delete_note(note_id)
        return checked("Note", p.note(obj))

    def get_file(file_id: UUID, catalog: Service) -> dict[str, Any]:
        return checked("File", p.file(catalog, catalog.file(file_id)))

    if include_file_routes:
        router.add_api_route("/files/{file_id}", get_file, methods=["GET"])

    @router.delete("/files/{file_id}")
    def delete_file(file_id: UUID, catalog: Service) -> dict[str, Any]:
        obj = catalog.file(file_id)
        catalog.delete_file(file_id)
        return checked("File", p.file(catalog, obj))

    @router.put("/library/items/{library_item_id}/files/{file_id}")
    def attach(library_item_id: UUID, file_id: UUID, catalog: Service) -> dict[str, Any]:
        catalog.attach(library_item_id, file_id)
        return checked("LibraryItem", p.item(catalog, catalog.item(library_item_id)))

    @router.delete("/library/items/{library_item_id}/files/{file_id}")
    def detach(library_item_id: UUID, file_id: UUID, catalog: Service) -> dict[str, Any]:
        catalog.detach(library_item_id, file_id)
        return checked("LibraryItem", p.item(catalog, catalog.item(library_item_id)))

    @router.delete("/history/{history_id}")
    def delete_history(history_id: UUID, catalog: Service) -> dict[str, Any]:
        obj = catalog.db.get(m.History, history_id)
        catalog.delete_history(history_id)  # Checks owner before serialization.
        return checked("History", p.history(obj))

    def register_page(path: str, kind: str) -> None:
        def endpoint(
            catalog: Service,
            limit: Annotated[int, Query(alias="page_size", ge=1, le=50)] = 20,
            cursor: str | None = None,
        ) -> dict[str, Any]:
            rows, next_cursor = pager.page(catalog, kind, limit, cursor)
            serialize = {
                "library": lambda row: p.item(catalog, row),
                "collections": p.collection,
                "notes": p.note,
                "history": p.history,
            }[kind]
            resource = {"library": "LibraryItem", "collections": "Collection", "notes": "Note", "history": "History"}[
                kind
            ]
            return checked(
                "Page",
                {
                    "items": [checked(resource, serialize(row)) for row in rows],
                    "next_cursor": next_cursor,
                    "snapshot_id": None,
                },
            )

        router.add_api_route(path, endpoint, methods=["GET"], name="list_" + kind)

    for path, kind in (
        ("/library/items", "library"),
        ("/collections", "collections"),
        ("/notes", "notes"),
        ("/history", "history"),
    ):
        register_page(path, kind)
    app.include_router(router)
    response_contracts(app, schema)
    return app
