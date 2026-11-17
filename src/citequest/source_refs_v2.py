"""Typed source identities; references never confer access or paper identity."""
from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class SourceReference(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class PaperSourceReference(SourceReference):
    source_kind: Literal["paper"]
    work_version_id: UUID
    source_snapshot_sha256: Sha256
    representation_id: Sha256


class MaterialSourceReference(SourceReference):
    source_kind: Literal["material"]
    material_version_id: UUID
    file_id: UUID
    raw_file_sha256: Sha256
    representation_id: Sha256


SourceRef = Annotated[PaperSourceReference | MaterialSourceReference, Field(discriminator="source_kind")]
SOURCE_REFERENCE = TypeAdapter(SourceRef)


def source_reference_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://citequest.local/contracts/v2-app/source-reference/1",
        "title": "Typed paper and material source reference",
        "x-contract-version": "1.0.0",
        **SOURCE_REFERENCE.json_schema(),
    }
