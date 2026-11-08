"""Bounded material HTTP adapter; parsing belongs to the durable worker queue.

Byte ranges/conditional reads follow RFC 9110 sections 13 and 14. Attachment
names follow RFC 6266: never trust an uploaded path, control byte or extension.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
import time
import unicodedata
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal
from urllib.parse import quote
from uuid import UUID, uuid4

import anyio
from fastapi import APIRouter, FastAPI, Header, Query, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Receive, Scope, Send

from citequest.v2.api import create_app as domain_app
from citequest.v2.domain import Actor, DomainError
from citequest.v2.origin import OriginPolicy
from citequest.v2.security import authenticate
from citequest.v2.storage import CAS, StoredBlob

from .artifact_public import INTERNAL_ROLES, PUBLIC_ROLES, PublicAssets, ensure_public_content, public_assets
from .reference_fetch import OfficialFetcher
from .reference_resolution import ReferenceService, acquire, reference_contract
from .references import plan_reference
from .text import MaterialError, load_limits

if TYPE_CHECKING:
    from citequest.business_storage_v3.service import BusinessStorage

SHA = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Idempotency = Annotated[UUID, Header(alias="Idempotency-Key")]
State = Literal["queued", "validating", "parsing", "partial", "ready", "failed", "cancelled"]


class Output(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class PDFLocator(Output):
    kind: Literal["pdf"]
    file_sha256: SHA
    page_1based: Annotated[int, Field(ge=1)]
    char_start: Annotated[int, Field(ge=0)]
    char_end: Annotated[int, Field(ge=1)]
    bbox: Annotated[list[Annotated[float, Field(allow_inf_nan=False)]], Field(min_length=4, max_length=4)] | None


class DOCXLocator(Output):
    kind: Literal["docx"]
    file_sha256: SHA
    part: Annotated[str, Field(pattern=r"^word/[^.]+\.xml$")]
    block_id: Annotated[str, Field(min_length=1)]
    char_start: Annotated[int, Field(ge=0)]
    char_end: Annotated[int, Field(ge=1)]


class TextLocator(Output):
    kind: Literal["text"]
    file_sha256: SHA
    char_start: Annotated[int, Field(ge=0)]
    char_end: Annotated[int, Field(ge=1)]
    line_1based: Annotated[int, Field(ge=1)]


class SourceFieldLocator(Output):
    kind: Literal["source_field"]
    source_sha256: SHA
    field: Literal["title", "abstract", "content.text"]
    char_start: Annotated[int, Field(ge=0)]
    char_end: Annotated[int, Field(ge=1)]


SourceLocator = Annotated[PDFLocator | DOCXLocator | TextLocator | SourceFieldLocator, Field(discriminator="kind")]


class MaterialFragment(Output):
    fragment_id: SHA
    ordinal: Annotated[int, Field(ge=1)]
    material_version_id: UUID
    work_version_id: UUID | None
    source_text: Annotated[str, Field(min_length=1)]
    source_text_sha256: SHA
    index_text: Annotated[str, Field(min_length=1)]
    locator: SourceLocator
    token_count: Annotated[int, Field(ge=1, le=2048)]
    tokenizer_version: Annotated[str, Field(min_length=1)]
    tokenizer_profile_sha256: SHA
    chunker_version: Annotated[str, Field(min_length=1)]
    source_block_id: Annotated[str, Field(min_length=1)]
    granularity: Literal["paragraph", "table", "formula"]
    bbox_granularity: Literal["native region enclosing this character span"] | None
    target_exceeded_due_to_source_boundary: bool


class MaterialFragments(Output):
    schema_version: Literal["cq.material.fragments/1"]
    file_id: UUID
    representation_id: SHA
    items: list[MaterialFragment]
    next_after: Annotated[int, Field(ge=1)] | None
    coverage: Literal["text_full", "abstract_only", "image_only", "partial"]
    missing_ranges: list[Annotated[str, Field(min_length=1)]]


class IngestionEvent(Output):
    seq: Annotated[int, Field(ge=1)]
    state: State
    at: AwareDatetime
    error_code: str | None


class MaterialIngestion(Output):
    ingestion_id: UUID
    file_id: UUID
    state: State
    revision: Annotated[int, Field(ge=1)]
    error_code: str | None
    representation_id: SHA | None
    lease_active: bool
    events: list[IngestionEvent]
    binding_source: Literal["none", "user_explicit", "trusted_source"]
    identity_status: Literal["unverified", "hash_verified"]
    material_source_ready: bool
    paper_source_ready: bool
    formal_evidence_ready: bool
    execution_outcome: Literal[
        "not_started", "running", "within_deadline", "failed", "cancelled", "late", "uncertain", "legacy_not_measured"
    ]
    active_elapsed_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None
    execution_budget_seconds: Annotated[float, Field(gt=0, le=120, allow_inf_nan=False)] | None
    deadline_at: AwareDatetime | None
    execution_success: bool

    @model_validator(mode="after")
    def identity_boundary(self) -> MaterialIngestion:
        if self.formal_evidence_ready and (
            self.identity_status != "hash_verified"
            or self.binding_source != "trusted_source"
            or not self.material_source_ready
            or not self.paper_source_ready
        ):
            raise ValueError("formal_evidence_requires_verified_trusted_identity")
        if self.execution_success and (
            self.execution_outcome != "within_deadline"
            or self.state != "ready"
            or self.active_elapsed_seconds is None
            or self.execution_budget_seconds is None
            or self.deadline_at is None
            or self.active_elapsed_seconds > self.execution_budget_seconds
        ):
            raise ValueError("execution_success_requires_measured_within_deadline_ready_result")
        return self


class BindVersion(Output):
    version_id: Annotated[UUID, Field(strict=False)]


class HTTPBoundaryError(DomainError):
    def __init__(self, code: str, status: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(code, status)
        self.headers = headers or {}


class RequestConcurrency:
    """Hold a standard limiter through the final response body, including HEAD."""

    def __init__(self, app: ASGIApp, limiter: anyio.CapacityLimiter) -> None:
        self.app, self.limiter = app, limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == "/healthz":
            await self.app(scope, receive, send)
            return
        borrower = object()
        try:
            self.limiter.acquire_on_behalf_of_nowait(borrower)
        except anyio.WouldBlock:
            request_id = str(uuid4())
            response = JSONResponse(
                {
                    "code": "http_busy",
                    "message": "http_busy",
                    "request_id": request_id,
                    "retryable": True,
                    "details": {},
                    "capability_state": None,
                },
                status_code=503,
                headers={"Retry-After": "1", "X-Request-Id": request_id},
            )
            await response(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self.limiter.release_on_behalf_of(borrower)


def _material_service(db: Session, actor: Actor, store: CAS | None) -> Any:
    # Import at the adapter boundary to keep schema/HTTP tests DB-free.
    from .services import MaterialService

    return MaterialService(db, actor, store)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite_json_number")


def byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    """One inclusive byte interval; unsupported units are ignored (RFC 9110)."""
    if value is None:
        return None
    if len(value) > 200:
        raise HTTPBoundaryError("invalid_byte_range", 416, {"Content-Range": f"bytes */{size}"})
    unit, separator, interval = value.partition("=")
    if separator and unit.lower() != "bytes":
        return None
    match = re.fullmatch(r"([0-9]*)-([0-9]*)", interval) if separator and unit.lower() == "bytes" else None
    if not match or not any(match.groups()):
        raise HTTPBoundaryError("invalid_byte_range", 416, {"Content-Range": f"bytes */{size}"})
    left, right = match.groups()
    if not left:
        suffix = int(right)
        if suffix == 0:
            raise HTTPBoundaryError("unsatisfiable_byte_range", 416, {"Content-Range": f"bytes */{size}"})
        return max(0, size - suffix), size - 1
    start, end = int(left), int(right) if right else size - 1
    if start >= size or end < start:
        raise HTTPBoundaryError("unsatisfiable_byte_range", 416, {"Content-Range": f"bytes */{size}"})
    return start, min(end, size - 1)


def content_disposition(file_name: str, media_type: str, file_id: UUID, inline: bool) -> str:
    extension = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "text/plain": ".txt",
        "text/markdown": ".md",
    }[media_type]
    name = file_name.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(c for c in name if not unicodedata.category(c).startswith("C")).strip(" .")[:255]
    if not name:
        name = "document"
    if not name.lower().endswith(extension):
        name += extension
    return f"{'inline' if inline else 'attachment'}; filename=\"document-{file_id}{extension}\"; filename*=UTF-8''{quote(name, safe='')}"


def create_app(
    factory: sessionmaker[Session],
    *,
    store: CAS | None,
    cursor_key: bytes,
    contract_path: Path,
    policy_path: Path,
    origin: str = "http://127.0.0.1:18000",
    max_uploads: int = 2,
    max_requests: int = 8,
    upload_timeout_seconds: float = 60,
    reference_fetcher: Any | None = None,
    origin_policy: OriginPolicy | None = None,
    business_storage: BusinessStorage | None = None,
) -> FastAPI:
    """Compose S03 with exact upload contracts and explicit S04 query extensions."""
    if (
        type(max_uploads) is not int
        or not 1 <= max_uploads <= 4
        or type(max_requests) is not int
        or not 1 <= max_requests <= 16
        or type(upload_timeout_seconds) not in {int, float}
        or not 0 < upload_timeout_seconds <= 120
    ):
        raise ValueError("bounded_local_http_configuration_required")
    if store is None and business_storage is None:
        raise ValueError("explicit_material_storage_required")
    app = domain_app(
        factory, cursor_key=cursor_key, contract_path=contract_path, origin=origin, include_file_routes=False,
        origin_policy=origin_policy,
    )
    base_openapi = deepcopy(app.openapi())
    app.title, app.version = "CiteQuest material and domain service", "0.4.0"
    schema = json.loads(contract_path.read_text(encoding="utf-8"))
    validators = {
        name: Draft202012Validator({**schema, "$ref": "#/$defs/" + name}, format_checker=FormatChecker())
        for name in schema["$defs"]
    }
    validators["BindVersion"] = Draft202012Validator(BindVersion.model_json_schema(), format_checker=FormatChecker())
    limits = load_limits(policy_path)
    json_wire_limit = max(limits.text_bytes, 12 * limits.text_characters) + 16384
    uploads = anyio.CapacityLimiter(max_uploads)
    requests = anyio.CapacityLimiter(max_requests)
    reference_slot = anyio.CapacityLimiter(1)
    fetcher = reference_fetcher if reference_fetcher is not None else OfficialFetcher()
    app.state.reference_fetcher = fetcher
    app.state.material_upload_limiter = uploads
    app.state.material_request_limiter = requests
    app.add_middleware(RequestConcurrency, limiter=requests)
    router = APIRouter(prefix="/api/v2")

    def checked(name: str, value: dict) -> dict:
        if not validators[name].is_valid(value):
            raise DomainError("response_contract_failed", 500)
        return value

    def extension(model: type[Output], value: dict) -> dict:
        try:
            model.model_validate_json(json.dumps(value, allow_nan=False))
        except (ValueError, TypeError) as exc:
            raise DomainError("response_contract_failed", 500) from exc
        return value

    def actor_in(db: Session, request: Request) -> Actor:
        mutation = request.method not in {"GET", "HEAD", "OPTIONS"}
        if mutation and request.headers.get("origin") != origin:
            raise DomainError("same_origin_required", 403)
        return authenticate(
            db, request.cookies.get("citequest_session"), csrf=request.headers.get("x-csrf-token"), mutation=mutation
        )

    def transaction(request: Request, action: Callable[[Any], Any]) -> Any:
        with factory.begin() as db:
            return action(_material_service(db, actor_in(db, request), store))

    def initial_actor(request: Request) -> Actor:
        with factory.begin() as db:
            return actor_in(db, request)

    async def invoke(request: Request, action: Callable[[Any], Any]) -> Any:
        return await run_in_threadpool(transaction, request, action)

    def reference_transaction(request: Request, action: Callable[[ReferenceService], Any]) -> Any:
        if store is None:
            raise DomainError("reference_cloud_storage_not_integrated", 503)
        with factory.begin() as db:
            return action(ReferenceService(db, actor_in(db, request), store))

    async def stored_bytes(request: Request, blob: StoredBlob,
                           authorization: Callable[[Any], Any], expected: Any) -> bytes:
        """Short SQL capture, supervised IO, then caller's final current guard."""
        if business_storage is None:
            return await run_in_threadpool(store.read, blob)
        from citequest.business_storage_v3.contracts import OperationDeadline
        from citequest.platform_v3.blob_store import BlobIdentity

        deadline = OperationDeadline(time.monotonic() + upload_timeout_seconds)

        def capture(service: Any) -> Any:
            if authorization(service) != expected:
                raise DomainError("material_context_changed", 409)
            return business_storage.resolve_location(service.db,
                BlobIdentity(blob.security_domain, blob.sha256, blob.size_bytes))

        reference = await invoke(request, capture)

        def read() -> bytes:
            with business_storage.open_verified(reference, deadline=deadline) as stream:
                data = stream.read(blob.size_bytes + 1)
                if len(data) != blob.size_bytes:
                    raise DomainError("material_original_size_changed", 409)
                return data

        return await run_in_threadpool(read)

    @app.exception_handler(HTTPBoundaryError)
    async def boundary_error(request: Request, exc: HTTPBoundaryError) -> JSONResponse:
        value = checked(
            "Error",
            {
                "code": exc.code,
                "message": exc.code,
                "request_id": request.state.request_id,
                "retryable": exc.status in {408, 429, 503},
                "details": {},
                "capability_state": None,
            },
        )
        return JSONResponse(value, status_code=exc.status, headers=exc.headers)

    @app.exception_handler(MaterialError)
    async def material_error(request: Request, exc: MaterialError) -> JSONResponse:
        status = 413 if "EXCEEDED" in exc.code else 415 if "MEDIA" in exc.code else 422
        return await boundary_error(request, HTTPBoundaryError(exc.code, status))

    async def json_body(request: Request, name: str) -> dict:
        if (
            request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json"
            or request.headers.get("content-encoding", "identity").lower() != "identity"
        ):
            raise HTTPBoundaryError("unsupported_request_media", 415)
        raw = bytearray()
        try:
            with anyio.fail_after(upload_timeout_seconds):
                async for part in request.stream():
                    if len(raw) + len(part) > json_wire_limit:
                        raise HTTPBoundaryError("request_body_too_large", 413)
                    raw.extend(part)
        except TimeoutError as exc:
            raise HTTPBoundaryError("request_body_deadline_exceeded", 408) from exc
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_pairs, parse_constant=_reject_constant)
            # A valid UTF-8 JSON wire can still contain an escaped lone surrogate.
            # Materials must have a real, lossless UTF-8 representation.
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise HTTPBoundaryError("invalid_json", 400) from exc
        if not validators[name].is_valid(value):
            raise HTTPBoundaryError("invalid_input", 400)
        return value

    @router.post("/uploads", status_code=202)
    async def create_upload(request: Request, key: Idempotency) -> dict:
        await run_in_threadpool(initial_actor, request)
        body = await json_body(request, "UploadInput")
        return await invoke(request, lambda service: checked("UploadTicket", service.create_upload(body, key=key)))

    @router.put("/uploads/{upload_id}/content", status_code=202)
    async def put_upload(upload_id: UUID, request: Request) -> dict:
        from citequest.business_storage_v3.contracts import OperationDeadline
        from citequest.platform_v3.blob_store import BlobIdentity

        from .services import material_blob_operation

        deadline = OperationDeadline(time.monotonic() + upload_timeout_seconds)
        if (
            request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/octet-stream"
            or request.headers.get("content-encoding", "identity").lower() != "identity"
        ):
            raise HTTPBoundaryError("unsupported_request_media", 415)
        actor = await run_in_threadpool(initial_actor, request)
        admission = await invoke(request, lambda service: service.upload(upload_id))
        length = request.headers.get("content-length")
        if length is not None:
            if not re.fullmatch(r"[0-9]{1,12}", length):
                raise HTTPBoundaryError("invalid_content_length", 400)
            if int(length) > admission.max_bytes:
                raise HTTPBoundaryError("upload_bytes_exceeded", 413)
            if int(length) != admission.size_bytes:
                raise HTTPBoundaryError("upload_size_mismatch", 422)
        if actor.owner_id in uploads.statistics().borrowers:
            raise HTTPBoundaryError("upload_busy", 429, {"Retry-After": "1"})
        try:
            uploads.acquire_on_behalf_of_nowait(actor.owner_id)
        except anyio.WouldBlock as exc:
            raise HTTPBoundaryError("upload_busy", 429, {"Retry-After": "1"}) from exc
        try:
            spool_root = business_storage.config.spool_dir if business_storage is not None else store.root
            with tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b", dir=spool_root) as spool:
                count, sha = 0, hashlib.sha256()
                with anyio.fail_after(upload_timeout_seconds):
                    async for part in request.stream():
                        count += len(part)
                        if count > admission.max_bytes:
                            raise HTTPBoundaryError("upload_bytes_exceeded", 413)
                        sha.update(part)
                        await run_in_threadpool(spool.write, part)
                    if count != admission.size_bytes or sha.hexdigest() != admission.expected_sha256:
                        raise HTTPBoundaryError("upload_content_mismatch", 422)
                    spool.seek(0)
                    try:
                        if business_storage is not None:
                            # Committed replays use the current File, even after
                            # the original storage receipt's window has expired.
                            if admission.file_id is not None:
                                return await invoke(request, lambda service: checked("File", service.get_file(admission.file_id)))
                            identity = BlobIdentity(actor.security_domain, admission.expected_sha256, admission.size_bytes)
                            prepared = await run_in_threadpool(lambda: business_storage.prepare_upload(actor,
                                material_blob_operation("upload", upload_id), identity, spool, deadline=deadline))
                        else:
                            blob = await run_in_threadpool(store.put, actor.security_domain, spool, admission.max_bytes)
                    except ValueError as exc:
                        raise HTTPBoundaryError("stored_blob_corrupt", 422) from exc
                if business_storage is not None:
                    return await invoke(request, lambda service: checked("File",
                        service.put_prepared_upload(upload_id, prepared, business_storage)))
                return await invoke(
                    request, lambda service: checked("File", service.put_upload_content(upload_id, blob))
                )
        except TimeoutError as exc:
            raise HTTPBoundaryError("upload_deadline_exceeded", 408) from exc
        except OSError as exc:
            raise HTTPBoundaryError("material_storage_unavailable", 503) from exc
        finally:
            uploads.release_on_behalf_of(actor.owner_id)

    @router.post("/materials/text", status_code=202)
    async def create_text(request: Request, key: Idempotency) -> dict:
        from citequest.business_storage_v3.contracts import OperationDeadline
        from citequest.platform_v3.blob_store import BlobIdentity

        from .services import MaterialService, material_blob_operation

        deadline = OperationDeadline(time.monotonic() + upload_timeout_seconds)
        actor = await run_in_threadpool(initial_actor, request)
        body = await json_body(request, "TextMaterialInput")
        if business_storage is not None:
            data, _ = MaterialService.text_bytes(body)
            previous = await invoke(request, lambda service: service.text_replay(body, key=key))
            if previous is not None:
                return checked("File", previous)
            identity = BlobIdentity(actor.security_domain, hashlib.sha256(data).hexdigest(), len(data))
            prepared = await run_in_threadpool(lambda: business_storage.prepare_upload(actor,
                material_blob_operation("text", key), identity, io.BytesIO(data), deadline=deadline))
            return await invoke(request, lambda service: checked("File",
                service.ingest_prepared_text(body, key=key, prepared=prepared, storage=business_storage)))
        return await invoke(request, lambda service: checked("File", service.ingest_text(body, key=key)))

    async def import_reference(request: Request, key: UUID, name: str) -> Response:
        body = await json_body(request, name)
        raw = body["doi"] if name == "DoiImportInput" else body["url"]
        try:
            plan = plan_reference(raw)
        except MaterialError as exc:
            # Unsupported HTTPS hosts are a structured capability gap. Invalid
            # references/schemes/credentials are rejected before any networking.
            if exc.code != "UNSUPPORTED_REFERENCE_ADAPTER":
                raise HTTPBoundaryError(exc.code.lower(), 400) from exc
            plan = None
        payload = {"operation": "resolve_reference", "input_schema": name, "body": body}
        borrower = object()
        try:
            reference_slot.acquire_on_behalf_of_nowait(borrower)
        except anyio.WouldBlock as exc:
            raise HTTPBoundaryError("reference_resolution_busy", 429, {"Retry-After": "3"}) from exc
        try:
            replay, local, actor = await run_in_threadpool(
                reference_transaction,
                request,
                lambda service: (service.preflight(key, payload), service.matches(plan), service.actor),
            )
            if replay is not None:
                result = replay
            else:
                acquisition = None
                if plan is not None and not any(match["files"] for match in local):
                    acquisition = await acquire(plan, fetcher, store, actor.security_domain)
                result = await run_in_threadpool(
                    reference_transaction, request, lambda service: service.finalize(key, payload, plan, acquisition)
                )
            return JSONResponse(result, status_code=202 if result["state"] == "queued" else 200)
        finally:
            reference_slot.release_on_behalf_of(borrower)

    @router.post("/imports/doi")
    async def import_doi(request: Request, key: Idempotency) -> Response:
        return await import_reference(request, key, "DoiImportInput")

    @router.post("/imports/url")
    async def import_url(request: Request, key: Idempotency) -> Response:
        return await import_reference(request, key, "UrlImportInput")

    @router.get("/files/{file_id}")
    def get_file(file_id: UUID, request: Request) -> dict:
        return transaction(request, lambda service: checked("File", service.get_file(file_id)))

    @router.get("/files/{file_id}/representation")
    def get_representation(file_id: UUID, request: Request) -> dict:
        return transaction(request, lambda service: checked("ParsedDocument", service.get_representation(file_id)))

    @router.get("/files/{file_id}/fragments")
    def get_fragments(
        file_id: UUID,
        request: Request,
        after: Annotated[int, Query(ge=0)] = 0,
        page_size: Annotated[int, Query(ge=1, le=50)] = 50,
        query: Annotated[str | None, Query(min_length=1, max_length=2000)] = None,
    ) -> dict:
        def read(service: Any) -> dict:
            value = extension(
                MaterialFragments, service.get_fragments(file_id, after=after, limit=page_size, query=query)
            )
            for item in value["items"]:
                checked("Locator", item["locator"])
                if hashlib.sha256(item["source_text"].encode()).hexdigest() != item["source_text_sha256"]:
                    raise DomainError("response_contract_failed", 500)
            return value

        return transaction(request, read)

    @router.get("/files/{file_id}/ingestion")
    def get_ingestion(file_id: UUID, request: Request) -> dict:
        return transaction(request, lambda service: extension(MaterialIngestion, service.get_ingestion(file_id)))

    @router.get("/files/{file_id}/artifacts")
    def get_artifacts(file_id: UUID, request: Request) -> dict:
        return transaction(request, lambda service: public_assets(service.get_artifacts(file_id)))

    @router.get("/files/{file_id}/representations/{representation_id}/artifacts/{role}/content")
    async def get_asset_content(
        file_id: UUID, representation_id: SHA, role: Annotated[str, Field(pattern=r"^[a-z_]{1,60}$")], request: Request
    ) -> Response:
        def metadata(service: Any) -> tuple[StoredBlob, str, str]:
            current = checked("ParsedDocument", service.get_representation(file_id))
            if current["representation_id"] != representation_id:
                raise DomainError("not_found", 404)
            if role not in PUBLIC_ROLES:
                raise (
                    DomainError("artifact_internal_audit_only", 403)
                    if role in INTERNAL_ROLES
                    else DomainError("not_found", 404)
                )
            blob, media, filename = service.read_artifact(file_id, role)
            if blob.size_bytes > 64 * 1024 * 1024 or media != "application/json" or filename != role + ".json":
                raise DomainError("artifact_response_contract_failed", 500)
            return blob, media, filename

        before = await invoke(request, metadata)
        try:
            data = await stored_bytes(request, before[0], metadata, before)
        except ValueError as exc:
            raise HTTPBoundaryError("stored_blob_corrupt", 422) from exc
        except OSError as exc:
            raise HTTPBoundaryError("material_storage_unavailable", 503) from exc
        await run_in_threadpool(ensure_public_content, role, data, root=contract_path.resolve().parents[2])
        current = await invoke(request, metadata)
        if current != before:
            raise DomainError("artifact_changed", 409)
        headers = {
            "ETag": f'"{before[0].sha256}"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-cache",
            "Content-Disposition": f'attachment; filename="{role}.json"',
            "Content-Security-Policy": "sandbox; default-src 'none'",
            "Cross-Origin-Resource-Policy": "same-origin",
        }
        return Response(data, media_type="application/octet-stream", headers=headers)

    @router.post("/files/{file_id}/cancel")
    def cancel(file_id: UUID, request: Request) -> dict:
        return transaction(request, lambda service: checked("File", service.cancel(file_id)))

    @router.post("/files/{file_id}/version")
    async def bind_version(file_id: UUID, request: Request) -> dict:
        await run_in_threadpool(initial_actor, request)
        body = await json_body(request, "BindVersion")
        return await invoke(
            request, lambda service: checked("File", service.bind_version(file_id, UUID(body["version_id"])))
        )

    @router.get("/files/{file_id}/content")
    async def get_content(
        file_id: UUID, request: Request, disposition: Literal["attachment", "inline"] = "attachment"
    ) -> Response:
        def metadata(service: Any) -> tuple[StoredBlob, str, str, dict]:
            blob, media, name = service.read_content(file_id)
            dto = checked("File", service.get_file(file_id))
            if (
                dto["file_id"] != str(file_id)
                or dto["sha256"] != blob.sha256
                or dto["size_bytes"] != blob.size_bytes
                or dto["security_domain"] != blob.security_domain
                or dto["media_type"] != media
            ):
                raise DomainError("response_contract_failed", 500)
            if not dto["can_download"]:
                raise DomainError("file_content_unavailable", 409)
            if disposition == "inline" and (media != "application/pdf" or not dto["can_preview"]):
                raise DomainError("inline_preview_unavailable", 409)
            return blob, media, name, dto

        before = await invoke(request, metadata)
        try:
            data = await stored_bytes(request, before[0], metadata, before)
        except ValueError as exc:
            raise HTTPBoundaryError("stored_blob_corrupt", 422) from exc
        except OSError as exc:
            raise HTTPBoundaryError("material_storage_unavailable", 503) from exc
        current = await invoke(request, metadata)
        if current[:2] != before[:2]:
            raise DomainError("file_content_changed", 409)
        blob, media, name, _dto = current
        response_media = "application/pdf" if disposition == "inline" else "application/octet-stream"
        etag = f'"{blob.sha256}"'
        if any(len(request.headers.get(name, "")) > 4096 for name in ["if-match", "if-none-match", "if-range"]):
            raise HTTPBoundaryError("invalid_conditional_header", 400)
        headers = {
            "ETag": etag,
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, no-cache",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": content_disposition(name, media, file_id, disposition == "inline"),
            "Content-Security-Policy": "sandbox; default-src 'none'",
            "Cross-Origin-Resource-Policy": "same-origin",
        }
        if_match = request.headers.get("if-match")
        if (
            if_match is not None
            and if_match.strip() != "*"
            and etag not in [part.strip() for part in if_match.split(",")]
        ):
            raise HTTPBoundaryError("etag_precondition_failed", 412, headers)
        if_none = request.headers.get("if-none-match")
        if if_none is not None and (
            if_none.strip() == "*" or etag in [part.strip().removeprefix("W/") for part in if_none.split(",")]
        ):
            return Response(status_code=304, headers=headers)
        requested = request.headers.get("range") if request.method == "GET" else None
        if request.headers.get("if-range") is not None and request.headers["if-range"] != etag:
            requested = None  # No Last-Modified validator: HTTP dates cannot match strongly.
        try:
            interval = byte_range(requested, len(data))
        except HTTPBoundaryError as exc:
            exc.headers = {**headers, **exc.headers}
            raise
        if interval is not None:
            start, end = interval
            headers["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
            return Response(data[start : end + 1], status_code=206, headers=headers, media_type=response_media)
        headers["Content-Length"] = str(len(data))
        return Response(data if request.method == "GET" else b"", headers=headers, media_type=response_media)

    router.add_api_route("/files/{file_id}/content", get_content, methods=["HEAD"], name="head_content")

    app.include_router(router)
    app.openapi_schema = None

    def generated() -> dict:
        if app.openapi_schema is not None:
            return app.openapi_schema
        value = get_openapi(title=app.title, version=app.version, routes=app.routes)
        value.setdefault("components", {}).setdefault("schemas", {}).update(
            deepcopy(base_openapi["components"]["schemas"])
        )
        value["components"]["securitySchemes"] = deepcopy(base_openapi["components"]["securitySchemes"])
        for path, methods in base_openapi["paths"].items():
            value["paths"][path].update(deepcopy(methods))
        outputs = {
            "create_upload": ("202", "UploadTicket"),
            "put_upload": ("202", "File"),
            "create_text": ("202", "File"),
            "get_file": ("200", "File"),
            "get_representation": ("200", "ParsedDocument"),
            "get_fragments": ("200", "MaterialFragments"),
            "get_ingestion": ("200", "MaterialIngestion"),
            "get_artifacts": ("200", "PublicAssets"),
            "cancel": ("200", "File"),
            "bind_version": ("200", "File"),
        }
        for model in [MaterialFragments, MaterialIngestion, BindVersion, PublicAssets]:
            model_schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
            value["components"]["schemas"].update(model_schema.pop("$defs", {}))
            value["components"]["schemas"][model.__name__] = model_schema
        value["components"]["schemas"]["ReferenceResolution"] = json.loads(
            json.dumps(reference_contract()["$defs"]["ReferenceResolution"]).replace(
                "#/$defs/", "#/components/schemas/"
            )
        )
        for route in router.routes:
            name, path = route.name, route.path
            name = "get_content" if name == "head_content" else name
            for raw_method in route.methods:
                method = raw_method.lower()
                operation = value["paths"][path][method]
                operation["security"] = [{"localSession": []}]
                operation["x-implementation-state"] = "implemented"
                operation["x-s04-material-contract"] = True
                if method in {"post", "put"}:
                    operation.setdefault("parameters", []).extend(
                        [
                            {"name": header, "in": "header", "required": True, "schema": {"type": "string"}}
                            for header in ["Origin", "X-CSRF-Token"]
                        ]
                    )
                if name in outputs:
                    status, output = outputs[name]
                    operation["responses"] = {
                        status: {
                            "description": "Validated material response; queued does not imply QA readiness",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/" + output}}},
                        }
                    }
                for status in [400, 401, 403, 404, 408, 409, 412, 413, 415, 416, 422, 429, 500, 503]:
                    operation.setdefault("responses", {})[str(status)] = {
                        "description": "Structured material boundary error",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
                    }
                if name in {"create_upload", "create_text"}:
                    resource = "UploadInput" if name == "create_upload" else "TextMaterialInput"
                    operation["requestBody"] = {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/" + resource}}},
                    }
                    operation["x-request-wire-max-bytes"] = json_wire_limit
                    if name == "create_text":
                        operation["x-source-text-limits"] = {
                            "max_characters": limits.text_characters,
                            "max_utf8_bytes": limits.text_bytes,
                        }
                if name in {"import_doi", "import_url"}:
                    resource = "DoiImportInput" if name == "import_doi" else "UrlImportInput"
                    operation["requestBody"] = {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/" + resource}}},
                    }
                    for status in ["200", "202"]:
                        operation["responses"][status] = {
                            "description": "200 metadata/existing/unavailable; 202 only actual safely acquired PDF queued, no identity/rights upgrade",
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/ReferenceResolution"}}
                            },
                        }
                    operation["x-contract-evolution"] = (
                        "Additive typed resolution replaces the prototype's unconditional File response; frozen S02 prototype remains unchanged."
                    )
                    operation["x-network-policy"] = {
                        "https_official_adapters_only": True,
                        "dns_pinned_socket": True,
                        "normal_tls_hostname_verification": True,
                        "maximum_redirects": 3,
                        "total_seconds": 30,
                        "metadata_max_bytes": 2097152,
                        "pdf_max_bytes": 52428800,
                        "ambient_proxy": False,
                        "automatic_retries": 0,
                    }
                if name == "bind_version":
                    operation["requestBody"] = {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/BindVersion"}}},
                    }
                    operation["x-request-wire-max-bytes"] = json_wire_limit
                if name == "put_upload":
                    operation["requestBody"] = {
                        "required": True,
                        "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
                    }
                    operation["x-server-enforced-limits"] = {
                        "pdf_bytes": limits.pdf_bytes,
                        "docx_bytes": limits.docx_bytes,
                        "text_bytes": limits.text_bytes,
                        "verify_actual_mime_hash_size": True,
                    }
                    operation["x-admission-proof"] = ["actual_size", "actual_sha256"]
                    operation["x-pending-validation"] = [
                        "isolated_actual_mime_and_signature",
                        "parser_and_source_anchors",
                    ]
                if name == "get_content":
                    operation.setdefault("parameters", []).extend(
                        [
                            {
                                "name": header,
                                "in": "header",
                                "required": False,
                                "schema": {"type": "string", "maxLength": 200 if header == "Range" else 4096},
                            }
                            for header in (
                                ["Range", "If-Range", "If-Match", "If-None-Match"]
                                if method == "get"
                                else ["If-Match", "If-None-Match"]
                            )
                        ]
                    )
                    content_headers = {
                        name: {"schema": {"type": "string"}}
                        for name in ["ETag", "Accept-Ranges", "Content-Disposition", "Content-Length"]
                    }
                    for status in ["200", "206"] if method == "get" else ["200"]:
                        operation["responses"][status] = {
                            "description": "Current ACL and verified original CAS bytes; single byte range supported",
                            "headers": deepcopy(content_headers),
                        }
                        if method == "get":
                            operation["responses"][status]["content"] = {
                                mime: {"schema": {"type": "string", "format": "binary"}}
                                for mime in ["application/octet-stream", "application/pdf"]
                            }
                        if status == "206":
                            operation["responses"][status]["headers"]["Content-Range"] = {"schema": {"type": "string"}}
                    operation["responses"]["304"] = {"description": "Current authorized representation ETag unchanged"}
                if name == "get_asset_content":
                    operation["responses"]["200"] = {
                        "description": "Current File/representation/role-authorized original safe JSON bytes; internal audit assets are not exposed",
                        "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
                    }
        value["info"]["description"] = (
            "S03 domain plus S04 bounded upload, durable parser state, original byte delivery and source-fragment preview. Unbound materials are not QA/Evidence ready. Local cookie sessions and same-origin CSRF apply."
        )
        app.openapi_schema = value
        return value

    app.openapi = generated
    return app
