"""Typed-paper bookmarks are not fabricated canonical Work records.

The save command accepts only a real locator and user-owned organization. The
service must capture metadata by calling the injected current SearchService.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from citequest.agents_v3.contracts import ID, Strict
from citequest.search_v3.schemas import PaperDetail, PaperLocator

Tag = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
NoteBody = Annotated[str, Field(min_length=1, max_length=15000)]


class SavedPaperCreate(Strict):
    operation_id: ID
    paper: PaperLocator
    collection_ids: list[ID] = Field(default_factory=list, max_length=30)
    tags: list[Tag] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def unique(self) -> SavedPaperCreate:
        if len(set(self.collection_ids)) != len(self.collection_ids) or len(set(self.tags)) != len(self.tags):
            raise ValueError("collections and tags must be unique")
        return self


class SavedPaperPatch(Strict):
    operation_id: ID
    expected_revision: int = Field(ge=1)
    collection_ids: list[ID] = Field(max_length=30)
    tags: list[Tag] = Field(max_length=30)

    @model_validator(mode="after")
    def unique(self) -> SavedPaperPatch:
        if len(set(self.collection_ids)) != len(self.collection_ids) or len(set(self.tags)) != len(self.tags):
            raise ValueError("collections and tags must be unique")
        return self


class SavedPaperLifecycle(Strict):
    operation_id: ID
    expected_revision: int = Field(ge=1)
    deleted: bool


class SavedPaperNoteCreate(Strict):
    operation_id: ID
    body: NoteBody


class SavedPaperNotePatch(Strict):
    operation_id: ID
    expected_revision: int = Field(ge=1)
    body: NoteBody


class SavedPaperNoteDelete(Strict):
    operation_id: ID
    expected_revision: int = Field(ge=1)


class PaperActivity(Strict):
    operation_id: ID
    paper: PaperLocator
    kind: Literal["view", "download_requested"] = Field(default="view", exclude_if=lambda value: value == "view")


class SavedPaperDTO(Strict):
    saved_paper_id: ID
    paper: PaperLocator
    revision: int = Field(ge=1)
    deleted: bool
    source_availability: Literal["available", "unavailable", "temporarily_unavailable"]
    # Re-read metadata only; captured historical metadata is never an authorization fallback.
    metadata: PaperDetail | None
    collection_ids: list[ID]
    tags: list[Tag]
    note_count: int = Field(ge=0)
    qa_scope: Literal["metadata_only", "native_text", "unavailable"]
    fulltext_summary_available: bool

    @model_validator(mode="after")
    def honest_availability(self) -> SavedPaperDTO:
        if self.source_availability != "available":
            if self.metadata is not None or self.qa_scope != "unavailable" or self.fulltext_summary_available:
                raise ValueError("unavailable source must not expose cached metadata or claim AI readiness")
        elif self.metadata is None or self.metadata.paper != self.paper:
            raise ValueError("current metadata must match the saved locator")
        if (
            self.metadata
            and self.metadata.coverage == "metadata"
            and (self.fulltext_summary_available or self.qa_scope != "metadata_only")
        ):
            raise ValueError("metadata-only papers remain bookmarkable, not fulltext summaries")
        return self
