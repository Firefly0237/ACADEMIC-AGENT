"""Client chooses formats only; toolchains, content and paths are server owned."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from citequest.agents_v3.contracts import ID, Strict

Format = Literal["docx", "html", "pdf"]


class ExportCreate(Strict):
    operation_id: ID
    formats: list[Format] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique(self) -> ExportCreate:
        if len(set(self.formats)) != len(self.formats):
            raise ValueError("export formats must be unique")
        return self


class ExportCancel(Strict):
    operation_id: ID


class ArtifactDTO(Strict):
    format: Format
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_url: str


class ExportDTO(Strict):
    id: ID
    task_id: ID
    state: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    requested_formats: list[Format]
    availability: Literal["pending", "available", "not_available", "source_unavailable"]
    revision: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    deadline_seconds: Literal[90] = 90
    font_profile: str
    error_code: str | None
    artifacts: list[ArtifactDTO]
    scientific_quality: Literal["inherits_task_review_not_new_model_evaluation"] = (
        "inherits_task_review_not_new_model_evaluation"
    )
    submission_ready: Literal[False] = False
    actual_model_requests: Literal[0] = 0


class ExportPageDTO(Strict):
    items: list[ExportDTO] = Field(max_length=20)
    next_cursor: str | None = Field(max_length=4096)
