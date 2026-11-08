"""Public search contract. Missing source metadata remains null, never inferred."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
CatalogName = Literal["application_papers", "standard_metadata"]
SearchMode = Literal["hybrid", "bm25", "dense"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchFilters(Strict):
    year_from: int | None = Field(default=None, ge=1000, le=2200)
    year_to: int | None = Field(default=None, ge=1000, le=2200)

    @model_validator(mode="after")
    def ordered_years(self):
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from must not exceed year_to")
        return self


class SearchRequest(Strict):
    q: str = Field(min_length=1, max_length=2000)
    catalog: CatalogName = "application_papers"
    mode: SearchMode = "hybrid"
    field: Literal["text", "author", "doi"] = "text"
    sort: Literal["relevance", "year_desc", "title"] = "relevance"
    filters: SearchFilters = Field(default_factory=SearchFilters)
    page_size: int = Field(default=10, ge=1, le=50)
    candidate_limit: int = Field(default=100, ge=1, le=200)
    ttl_seconds: int = Field(default=900, ge=30, le=1800)
    selected_papers: list[PaperLocator] = Field(default_factory=list, max_length=20)

    @field_validator("q")
    @classmethod
    def nonempty_query(cls, value: str) -> str:
        if not value.strip() or any(ord(c) < 32 and c not in "\t\n\r" for c in value):
            raise ValueError("query must contain visible text and no binary control characters")
        return value

    @model_validator(mode="after")
    def bounded_page(self):
        if self.page_size > self.candidate_limit:
            raise ValueError("page_size exceeds candidate_limit")
        return self


class PaperLocator(Strict):
    kind: Literal["metadata_projection", "native_representation"]
    id: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def typed_id(self):
        if self.kind == "native_representation":
            if re.fullmatch(r"[a-f0-9]{64}", self.id) is None:
                raise ValueError("native representation needs its actual SHA256 ID")
        else:
            self.id = str(UUID(self.id))
        return self


class SourceCitation(Strict):
    source_namespace: str
    native_id: str
    source_revision: str | None
    projection_sha256: SHA
    source_pointer_sha256: SHA


class PaperSummary(Strict):
    paper: PaperLocator
    title: str | None
    abstract: str | None
    authors: list[str] | None
    year: int | None
    work_id: UUID | None
    work_version_id: UUID | None
    coverage: Literal["metadata", "native_literal"]
    source: SourceCitation
    source_url: str | None = None
    pdf_url: str | None = None
    metadata_revision: int | None = None


class SearchHit(Strict):
    rank: int
    score: float
    paper: PaperSummary
    snippet: str
    evidence_refs: list[str]


class SearchPage(Strict):
    snapshot_id: UUID
    generation_id: UUID
    epoch: int
    query: str
    catalog: CatalogName
    mode: SearchMode
    retrieval_method: Literal["native_bm25", "native_dense", "native_rrf", "structured_doi", "structured_author"]
    sort: Literal["relevance", "year_desc", "title"]
    candidate_scope: Literal["bounded_retrieval_candidates"] = "bounded_retrieval_candidates"
    candidate_limit: int
    candidate_count: int
    facets: dict[str, dict[str, int]]
    masked_since_search: int
    results: list[SearchHit]
    next_cursor: str | None
    expires_at: datetime
    generation_profile_sha256: SHA
    model_calls: Literal[0] = 0


class PaperDetail(PaperSummary):
    pdf_license: Literal["CC-BY-4.0"] | None = None
    pdf_is_unmodified: Literal[True] | None = None
    source_snapshot_id: UUID
    literal_text_available: bool
    coordinate_system: Literal["unicode_codepoints_in_metadata_recipe", "unicode_codepoints_in_native_literal"]
    links_status: Literal["source_provided", "not_in_source"]
    citation_status: Literal["complete", "incomplete_metadata"]


class SimilarRequest(Strict):
    mode: SearchMode = "hybrid"
    page_size: int = Field(default=10, ge=1, le=50)
    candidate_limit: int = Field(default=50, ge=1, le=200)


class EvidenceDTO(Strict):
    ref_id: str
    snapshot_id: UUID
    generation_id: UUID
    epoch: int
    source_ref: PaperLocator
    source: SourceCitation
    source_sha256: SHA
    unit_text_sha256: SHA
    text: str
    span_start: int
    span_end: int
    coordinate_system: Literal["unicode_codepoints_in_metadata_recipe", "unicode_codepoints_in_native_literal"]
    work_id: UUID | None
    work_version_id: UUID | None
    coverage: Literal["metadata", "native_literal"]
    search_staging_is_not_QA_permission: Literal[True] = True


class NativePDFAssetProfile(Strict):
    version: Literal["citequest-native-pdf-access/1"]
    state: Literal["candidate", "adopted"]
    adoption_ref: str = Field(min_length=1, max_length=1000)
    exact_source_license: Literal["CC-BY-4.0"] = "CC-BY-4.0"
    use: Literal["unmodified_source_view_download_with_attribution"] = "unmodified_source_view_download_with_attribution"


SearchRequest.model_rebuild()
