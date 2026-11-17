"""Literal, versioned material queries; no model, inferred title or body truncation."""
from __future__ import annotations

import hashlib
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from citequest.agents_v3.contracts import canonical, digest, validate_user_text
from citequest.v2.domain import DomainError

from .schemas import CatalogName, SearchMode, SearchRequest, Strict

AUDIT_KEY = "_material_query_projection"
SHA = r"^[a-f0-9]{64}$"


class MaterialSourcePin(Strict):
    file_id: str
    file_revision: int = Field(ge=1)
    file_sha256: str = Field(pattern=SHA)
    representation_id: str = Field(pattern=SHA)
    work_id: str | None = None
    work_version_id: str | None = None
    outbound_policy: Literal["selected_cloud_processing_allowed"]
    rights_status: Literal["verified", "unknown"]

    @field_validator("file_id", "work_id", "work_version_id")
    @classmethod
    def canonical_id(cls, value: str | None) -> str | None:
        return str(UUID(value)) if value is not None else None


class MaterialQuerySelection(Strict):
    kind: Literal["docling_title_fragment", "user_excerpt"] = "user_excerpt"
    fragment_id: str = Field(pattern=SHA)
    span_start: int = Field(ge=0)
    span_end: int = Field(ge=1)

    @model_validator(mode="after")
    def bounded_span(self) -> MaterialQuerySelection:
        if not 0 < self.span_end - self.span_start <= 2000:
            raise ValueError("select one nonempty source span of at most 2000 Unicode codepoints")
        return self


class MaterialSimilarInput(Strict):
    action: Literal["prepare", "search"] = "prepare"
    query: str | None = Field(default=None, min_length=1, max_length=2000)
    selection: MaterialQuerySelection | None = None
    expected_source: MaterialSourcePin | None = None
    catalog: CatalogName = "application_papers"
    mode: SearchMode = "hybrid"
    page_size: int = Field(default=10, ge=1, le=50)
    candidate_limit: int = Field(default=50, ge=1, le=200)

    @field_validator("query")
    @classmethod
    def visible_query(cls, value: str | None) -> str | None:
        if value is not None:
            validate_user_text(value)
            if not value.strip() or any(ord(char) < 32 and char not in "\r\n\t" for char in value):
                raise ValueError("query must contain visible text without binary controls")
        return value

    @model_validator(mode="after")
    def complete_search(self) -> MaterialSimilarInput:
        if self.page_size > self.candidate_limit:
            raise ValueError("page size exceeds the candidate limit")
        if self.action == "search" and (self.query is None or self.expected_source is None):
            raise ValueError("search needs an explicitly confirmed query and current material pin")
        return self


class MaterialQueryProjection(Strict):
    schema_version: Literal["citequest-material-query-projection/1"] = "citequest-material-query-projection/1"
    query: str = Field(min_length=1, max_length=2000)
    origin: Literal["docling_title_fragment", "user_excerpt", "edited_user_query", "user_query"]
    source: MaterialSourcePin
    selection: MaterialQuerySelection | None = None
    seed_sha256: str | None = Field(default=None, pattern=SHA)
    fragment_sha256: str | None = Field(default=None, pattern=SHA)
    query_sha256: str = Field(pattern=SHA)
    projection_sha256: str = Field(pattern=SHA)
    model_calls: Literal[0] = 0

    @model_validator(mode="after")
    def exact_identity(self) -> MaterialQueryProjection:
        if self.query_sha256 != hashlib.sha256(self.query.encode()).hexdigest():
            raise ValueError("material query text hash mismatch")
        if self.projection_sha256 != digest(self.model_dump(mode="json", exclude={"projection_sha256"})):
            raise ValueError("material query provenance hash mismatch")
        if ((self.selection is None and (self.seed_sha256 is not None or self.fragment_sha256 is not None))
            or (self.selection is not None and (self.seed_sha256 is None or self.fragment_sha256 is None))):
            raise ValueError("source-derived queries require complete source span hashes")
        if (self.selection is None and self.origin != "user_query") or (self.selection is not None
            and (self.origin not in {self.selection.kind, "edited_user_query"}
                or (self.origin == self.selection.kind and self.seed_sha256 != self.query_sha256))):
            raise ValueError("material query origin does not match its literal source")
        return self


def title_selection(fragments: list[dict]) -> tuple[MaterialQuerySelection | None, str]:
    """Only a single declared Docling title is suggested; ambiguous labels ask for input."""
    if not fragments:
        return None, "no_declared_docling_title"
    if len(fragments) > 16:
        return None, "title_candidate_limit_exceeded"
    if any(row.get("native_provenance", {}).get("structure_version") != "docling-native-structure/1"
        or row.get("native_provenance", {}).get("item_label") != "title" for row in fragments):
        return None, "no_declared_docling_title"
    candidates = {row["text"] for row in fragments}
    if len(candidates) != 1:
        return None, "ambiguous_docling_titles"
    text = fragments[0]["text"]
    if not text.strip() or len(text) > 2000:
        return None, "declared_title_requires_user_query"
    try:
        MaterialSimilarInput(query=text)
    except ValueError:
        return None, "declared_title_requires_user_query"
    return MaterialQuerySelection(kind="docling_title_fragment", fragment_id=fragments[0]["id"],
        span_start=0, span_end=len(text)), "declared_docling_title"


def project_material_query(source: MaterialSourcePin, *, query: str | None,
                           selection: MaterialQuerySelection | None = None, fragment: dict | None = None) -> MaterialQueryProjection:
    seed, fragment_hash = None, None
    if selection is not None:
        if fragment is None or fragment["id"] != selection.fragment_id or selection.span_end > len(fragment["text"]):
            raise DomainError("material_similarity_selection_changed", 409)
        if selection.kind == "docling_title_fragment" and (fragment.get("native_provenance", {}).get("item_label") != "title"
            or fragment.get("native_provenance", {}).get("structure_version") != "docling-native-structure/1"):
            raise DomainError("material_similarity_title_not_declared", 422)
        seed = fragment["text"][selection.span_start:selection.span_end]
        fragment_hash = hashlib.sha256(fragment["text"].encode()).hexdigest()
    literal = query if query is not None else seed
    if literal is None:
        raise DomainError("material_similarity_query_required", 422)
    MaterialSimilarInput(query=literal)
    origin = selection.kind if selection is not None and literal == seed else "edited_user_query" if selection else "user_query"
    value = {"schema_version": "citequest-material-query-projection/1", "query": literal, "origin": origin,
        "source": source.model_dump(mode="json"), "selection": selection.model_dump(mode="json") if selection else None,
        "seed_sha256": hashlib.sha256(seed.encode()).hexdigest() if seed is not None else None,
        "fragment_sha256": fragment_hash, "query_sha256": hashlib.sha256(literal.encode()).hexdigest(), "model_calls": 0}
    return MaterialQueryProjection.model_validate_json(canonical({**value, "projection_sha256": digest(value)}), strict=True)


def snapshot_search_request(payload: dict) -> SearchRequest:
    """Decode the one server-only envelope key; external SearchRequest stays strict."""
    value = dict(payload)
    has_audit = AUDIT_KEY in value
    audit = value.pop(AUDIT_KEY, None)
    request = SearchRequest.model_validate(value, strict=True)
    if has_audit:
        projection = MaterialQueryProjection.model_validate_json(canonical(audit), strict=True)
        if projection.query != request.q:
            raise DomainError("material_similarity_snapshot_query_changed", 409)
    return request
