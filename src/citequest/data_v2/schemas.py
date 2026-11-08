"""Strict source-side DTOs, separate from qualified nonempty-title Work DTOs."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from citequest.v2.domain import ID, SHA, Input

Profile = Literal[
    "litsearch-clean-exact/1",
    "litsearch-native-annotation-decimal-offsets/1",
    "litsearch-s2orc-native-text/1",
]
MetadataProfile = Literal["litsearch-clean-exact/1", "litsearch-native-annotation-decimal-offsets/1"]


class SourceArtifact(Input):
    path: Annotated[str, Field(min_length=1, max_length=512)]
    sha256: SHA
    size_bytes: Annotated[int, Field(gt=0)]
    format: Literal["parquet", "jsonl", "json"]
    row_group_sizes: Annotated[list[Annotated[int, Field(gt=0)]], Field(min_length=1)] | None

    @field_validator("path")
    @classmethod
    def relative_source_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            "\\" in value
            or ":" in value
            or path.is_absolute()
            or any(p in {".", ".."} for p in value.split("/"))
            or str(path) != value
        ):
            raise ValueError("canonical relative POSIX source path required")
        return value

    @model_validator(mode="after")
    def parquet_groups(self) -> SourceArtifact:
        if (self.format == "parquet") != (self.row_group_sizes is not None):
            raise ValueError("Parquet requires actual row-group counts; other formats have none")
        return self


class SnapshotInput(Input):
    source_namespace: Annotated[str, Field(min_length=1, max_length=100)]
    source_revision: Annotated[str, Field(min_length=1, max_length=100)]
    source_manifest_sha256: SHA
    policy_sha256: SHA
    evaluation_sha256: SHA
    representation_profile: Profile
    purpose: Literal["standard", "application", "metadata_update"]
    artifacts: Annotated[list[SourceArtifact], Field(min_length=1, max_length=1000)]
    client_operation_id: ID

    @model_validator(mode="after")
    def distinct_artifacts(self) -> SnapshotInput:
        if len({item.path for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("duplicate source artifact path")
        if self.purpose == "application" and self.representation_profile != "litsearch-s2orc-native-text/1":
            raise ValueError("title/abstract profiles are not application fulltext profiles")
        if self.purpose != "application" and self.representation_profile == "litsearch-s2orc-native-text/1":
            raise ValueError("native fulltext profile requires application purpose")
        return self


class RawRowLocator(Input):
    artifact_path: Annotated[str, Field(min_length=1, max_length=512)]
    artifact_sha256: SHA
    row_group: Annotated[int, Field(ge=0)]
    row_in_group: Annotated[int, Field(ge=0)]
    row_ordinal: Annotated[int, Field(ge=0)]


class SnapshotFreezeInput(Input):
    client_operation_id: ID
    expected_export_manifest_sha256: SHA
    expected_queries_sha256: SHA


class ProjectionInput(Input):
    native_id: Annotated[str, Field(min_length=1, max_length=512)]
    native_id_source_type: Literal["integer", "string"]
    locator: RawRowLocator
    title: str | None
    abstract: str | None
    field_provenance: dict[str, Any]

    @model_validator(mode="after")
    def truthful_native_type_and_json_provenance(self) -> ProjectionInput:
        if self.native_id_source_type == "integer":
            try:
                canonical = str(int(self.native_id))
            except ValueError as exc:
                raise ValueError("source integer ID requires canonical decimal spelling") from exc
            if canonical != self.native_id:
                raise ValueError("source integer ID requires canonical decimal spelling")
        try:
            json.dumps(self.field_provenance, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("provenance must be finite standard JSON") from exc
        return self


class BatchInput(Input):
    rows: Annotated[list[ProjectionInput], Field(min_length=1, max_length=500)]
    client_operation_id: ID

    @model_validator(mode="after")
    def distinct_native_ids_and_positions(self) -> BatchInput:
        if len({row.native_id for row in self.rows}) != len(self.rows):
            raise ValueError("duplicate native ID in batch")
        positions = {(r.locator.artifact_path, r.locator.row_ordinal) for r in self.rows}
        if len(positions) != len(self.rows):
            raise ValueError("duplicate raw position in batch")
        return self


class ProjectionReceipt(Input):
    projection_id: ID
    source_row_id: ID
    snapshot_id: ID
    native_id: str
    representation_profile: Profile
    projection_sha256: SHA
    source_pointer_sha256: SHA
    title: str | None
    abstract: str | None
    field_provenance: dict[str, Any]
    work_id: ID | None
    resolution: Literal["unresolved", "resolved"]
    unresolved_reasons: list[str]
    searchable_text_present: bool


class IndexProjection(Input):
    """Internal immutable metadata indexing input; never require a fake Work.

    Null and original whitespace remain in title/abstract. A versioned index
    text builder, tokenizer and generation manifest decide how to encode them.
    This DTO itself neither adopts a profile nor authorizes publication.
    """

    projection_id: ID
    source_row_id: ID
    snapshot_id: ID
    native_id: Annotated[str, Field(min_length=1, max_length=512)]
    source_namespace: Annotated[str, Field(min_length=1, max_length=100)]
    source_revision: Annotated[str, Field(min_length=1, max_length=100)]
    security_domain: Annotated[str, Field(min_length=1, max_length=100)]
    source_kind: Literal["litsearch_title_abstract"] = "litsearch_title_abstract"
    purpose: Literal["standard"] = "standard"
    representation_profile: MetadataProfile
    source_pointer_sha256: SHA
    projection_sha256: SHA
    title: str | None
    abstract: str | None
    field_provenance: dict[str, Any]
    year: Annotated[int, Field(ge=1000, le=2200)] | None = None
    work_id: ID | None
    searchable_text_present: bool
    profile_adopted: bool
    snapshot_state: Literal["landing", "staged", "disabled"]

    @model_validator(mode="after")
    def unchanged_projection(self) -> IndexProjection:
        expected_present = bool((self.title and self.title.strip()) or (self.abstract and self.abstract.strip()))
        if self.searchable_text_present != expected_present:
            raise ValueError("text presence flag must reflect actual source strings")
        value = {
            "source_pointer_sha256": self.source_pointer_sha256,
            "representation_profile": self.representation_profile,
            "title": self.title,
            "abstract": self.abstract,
            "field_provenance": self.field_provenance,
        }
        actual = hashlib.sha256(
            json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if actual != self.projection_sha256:
            raise ValueError("index input must bind the entire unchanged projection")
        return self


def unresolved_reasons(title: str | None, abstract: str | None) -> list[str]:
    """Preserve source strings, including whitespace; never create a title."""
    reasons = []
    if title is None or not title.strip():
        reasons.append("source_title_missing")
    elif len(title) > 500:
        reasons.append("exceeds_manual_WorkInput_title_limit_requires_source_import_profile")
    if not (title and title.strip()) and not (abstract and abstract.strip()):
        reasons.append("title_and_abstract_empty")
    return reasons
