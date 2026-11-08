"""FastAPI routes; authentication/anonymous owner isolation comes from the app."""
from uuid import UUID
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from pydantic import ValidationError
from citequest.v2.domain import Actor, DomainError
from .schemas import EvidenceDTO, PaperDetail, PaperLocator, SearchPage, SearchRequest, SimilarRequest
from .service import SearchService


def search_router(service: SearchService, actor_dependency) -> APIRouter:
    router = APIRouter(prefix="/api/v3", tags=["academic-search"])

    @router.post("/search", response_model=SearchPage)
    def search(body: SearchRequest, actor: Actor = Depends(actor_dependency)):
        return service.search(actor, body)

    @router.get("/search/snapshots/{snapshot_id}", response_model=SearchPage)
    def page(snapshot_id: UUID, cursor: str | None = Query(default=None, max_length=1800), actor: Actor = Depends(actor_dependency)):
        return service.page(actor, snapshot_id, cursor)

    @router.get("/papers/{kind}/{paper_id}", response_model=PaperDetail)
    def detail(kind: str, paper_id: str, actor: Actor = Depends(actor_dependency)):
        try:
            locator = PaperLocator(kind=kind, id=paper_id)
        except ValidationError:
            raise DomainError("invalid_typed_paper_locator", 422) from None
        return service.detail(actor, locator)

    @router.post("/papers/{kind}/{paper_id}/similar", response_model=SearchPage)
    def similar(kind: str, paper_id: str, body: SimilarRequest, actor: Actor = Depends(actor_dependency)):
        try:
            locator = PaperLocator(kind=kind, id=paper_id)
        except ValidationError:
            raise DomainError("invalid_typed_paper_locator", 422) from None
        return service.similar(actor, locator, body)

    @router.api_route("/papers/native_representation/{paper_id}/pdf", methods=["GET", "HEAD"], response_class=FileResponse)
    def paper_pdf(paper_id: str, actor: Actor = Depends(actor_dependency)):
        try:
            locator = PaperLocator(kind="native_representation", id=paper_id)
        except ValidationError:
            raise DomainError("invalid_typed_paper_locator", 422) from None
        asset = service.read_pdf(actor, locator)
        headers = {"Cache-Control": "no-store", "ETag": '"sha256-'+asset.sha256+'"',
                   "Cross-Origin-Resource-Policy": "same-origin", "X-Content-Type-Options": "nosniff",
                   "Link": '<https://creativecommons.org/licenses/by/4.0/>; rel="license"'}
        if asset.source_url and "\r" not in asset.source_url and "\n" not in asset.source_url:
            headers["Content-Location"] = asset.source_url
        # The installed Starlette FileResponse implements Range/If-Range/HEAD;
        # the route never accepts a filesystem path, blob domain or arbitrary URL.
        return FileResponse(asset.path, media_type="application/pdf", filename=asset.filename,
                            content_disposition_type="inline", headers=headers)

    @router.get("/search/evidence/{snapshot_id}/{chunk_id}", response_model=EvidenceDTO)
    def evidence(snapshot_id: UUID, chunk_id: str, actor: Actor = Depends(actor_dependency)):
        return service.read_evidence(actor, f"{snapshot_id}:{chunk_id}")

    return router


build_router = search_router
