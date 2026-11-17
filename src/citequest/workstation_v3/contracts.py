"""Bounded workstation commands; owner identity is never accepted from the body."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from citequest.agents_v3.contracts import ID, Strict


class BatchSave(Strict):
    operation_id: ID
    work_ids: list[ID] = Field(min_length=1, max_length=50)
    collection_id: ID | None = None

    @model_validator(mode="after")
    def unique(self) -> BatchSave:
        if len(set(self.work_ids)) != len(self.work_ids):
            raise ValueError("work ids must be unique")
        return self


class Activity(Strict):
    operation_id: ID
    kind: Literal["view", "download_requested", "export"]
    target_kind: Literal["work", "file", "task"]
    target_id: ID

    @model_validator(mode="after")
    def valid_kind(self) -> Activity:
        if self.kind == "download_requested" and self.target_kind != "file":
            raise ValueError("downloads target a current file; a URL click is not a completed transfer")
        return self


class ClearHistory(Strict):
    operation_id: ID
    through: Annotated[AwareDatetime, Field(strict=False)]
    kinds: list[Literal["search", "view", "download_requested", "download_completed", "export"]] = Field(
        min_length=1, max_length=5
    )

    @model_validator(mode="after")
    def unique(self) -> ClearHistory:
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError("history kinds must be unique")
        return self
