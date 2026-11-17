"""Bounded full-source windows and multi-document results; coverage is server-owned.

Reading all native text is never equated with complete PDF/image extraction.
Model schemas contain claims, not invented source identities or coverage counts.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import Claim

SHA = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class SourceSelection(Strict):
    kind: Literal["metadata_projection", "native_representation", "material_file"]
    id: str = Field(min_length=1, max_length=64)


class WindowDescriptor(Strict):
    window_id: SHA
    ordinal: int = Field(ge=1)
    source_parts: int = Field(ge=1, le=32)
    text_bytes: int = Field(ge=1)
    encoded_result_bytes: int = Field(ge=1, le=16384)
    source_pin_sha256: SHA


class DocumentManifest(Strict):
    document_id: str = Field(pattern=r"^D(0[1-9]|10)$")
    selection: SourceSelection
    title: str | None
    source_version_sha256: SHA
    coverage_kind: Literal["metadata_only", "native_literal", "material_full_text", "material_partial"]
    available_text_bytes: int = Field(ge=0)
    available_codepoints: int = Field(ge=0)
    missing_ranges: list[dict] = Field(max_length=500)
    windows: list[WindowDescriptor] = Field(max_length=1000)
    model_may_claim_complete_pdf: Literal[False] = False


class WorksetManifest(Strict):
    schema_version: Literal["citequest-workset-manifest/1"] = "citequest-workset-manifest/1"
    manifest_id: SHA
    documents: list[DocumentManifest] = Field(min_length=1, max_length=10)
    full_coverage_planned: bool
    window_max_encoded_bytes: Literal[16384] = 16384

    @model_validator(mode="after")
    def unique_ids(self) -> WorksetManifest:
        if len({item.document_id for item in self.documents}) != len(self.documents):
            raise ValueError("duplicate workset document")
        return self


class DocumentSummary(Strict):
    """Model output for one real document. Source/coverage are added by the server."""
    claims: list[Claim] = Field(min_length=1, max_length=3)
    gaps: list[str] = Field(max_length=8)
    evidence_status: Literal["supported", "insufficient"]


class DocumentCoverage(Strict):
    document_id: str
    source_version_sha256: SHA
    status: Literal["complete_available_text", "partial", "insufficient", "failed", "not_processed"]
    total_windows: int = Field(ge=0)
    mapped_windows: int = Field(ge=0)
    mapped_window_ids: list[SHA]
    covered_codepoints: int = Field(ge=0)
    available_codepoints: int = Field(ge=0)
    source_coverage_kind: Literal["metadata_only", "native_literal", "material_full_text", "material_partial"]
    reasons: list[str]

    @model_validator(mode="after")
    def measured(self) -> DocumentCoverage:
        if (self.mapped_windows != len(set(self.mapped_window_ids)) or self.mapped_windows > self.total_windows
                or self.covered_codepoints > self.available_codepoints):
            raise ValueError("invalid measured source coverage")
        if self.status == "complete_available_text" and (self.mapped_windows != self.total_windows
                or self.covered_codepoints != self.available_codepoints):
            raise ValueError("all available source windows must be consumed for complete available-text coverage")
        return self


class DocumentResult(Strict):
    document: DocumentManifest
    coverage: DocumentCoverage
    summary: DocumentSummary | None
    review_status: Literal["ai_review_validated", "quality_rejected", "not_assessed"]


class WorksetAnswer(Strict):
    kind: Literal["summary", "compare"]
    documents: list[DocumentResult] = Field(min_length=1, max_length=10)
    comparisons: list[Claim] = Field(max_length=8)
    gaps: list[str] = Field(max_length=30)
    evidence_status: Literal["supported", "insufficient"]
    completion: Literal["complete_available_scope", "partial", "failed"]
    manifest_id: SHA

    @model_validator(mode="after")
    def publication_scope(self) -> WorksetAnswer:
        if self.completion == "complete_available_scope" and any(
            item.coverage.status != "complete_available_text" or item.review_status != "ai_review_validated"
            for item in self.documents
        ):
            raise ValueError("workset completeness requires every selected document")
        return self
