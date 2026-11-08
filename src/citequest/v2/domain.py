"""Strict public inputs and immutable, server-derived identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, field_validator

ID = Annotated[UUID, Field(strict=False)]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
SHA = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


@dataclass(frozen=True)
class Actor:
    owner_id: UUID
    publisher: bool = False

    @property
    def security_domain(self) -> str:
        return "owner:" + str(self.owner_id)


class IdentifierInput(Input):
    namespace: Literal["doi", "arxiv", "s2_paper_id", "s2_corpus_id", "openalex", "litsearch_corpus_id"]
    raw: Annotated[str, Field(min_length=1, max_length=512)]


class WorkInput(Input):
    title: Name
    authors: Annotated[list[Name], Field(min_length=1)] | None
    year: Annotated[int, Field(ge=1000, le=2200)] | None
    identifiers: list[IdentifierInput] = Field(default_factory=list, max_length=20)
    client_operation_id: ID


class LibrarySave(Input):
    work_id: ID
    collection_id: ID | None
    client_operation_id: ID


class LibraryPatch(Input):
    expected_revision: int = Field(ge=1)
    tags: list[Name]
    collection_ids: list[ID]


class CollectionCreate(Input):
    name: Name
    parent_id: ID | None
    client_operation_id: ID


class CollectionPatch(Input):
    name: Name
    parent_id: ID | None
    expected_revision: int = Field(ge=1)


class NoteCreate(Input):
    library_item_id: ID
    body: str = Field(max_length=200000)
    evidence_ids: list[SHA]
    expected_revision: Annotated[int, Field(ge=1)] | None
    client_operation_id: ID


class NotePatch(Input):
    body: str = Field(max_length=200000)
    expected_revision: int = Field(ge=1)


class SourceInput(Input):
    source: Annotated[str, Field(min_length=1, max_length=100)]
    native_id: Annotated[str, Field(min_length=1, max_length=512)]
    source_revision: Annotated[str, Field(min_length=1, max_length=100)] | None
    retrieved_at: AwareDatetime
    raw_sha256: SHA
    source_url: str | None
    rights_status: Literal["verified", "unknown", "restricted"]
    license: Annotated[str, Field(min_length=1)] | None

    @field_validator("source_url")
    @classmethod
    def valid_source_uri(cls, value: str | None) -> str | None:
        if value is not None:
            from jsonschema import FormatChecker
            from jsonschema.exceptions import FormatError

            checker = FormatChecker()
            if "uri" not in checker.checkers:
                raise RuntimeError("Required URI format validator unavailable")
            try:
                checker.check(value, "uri")
            except FormatError as exc:
                raise ValueError("source_url must be a valid absolute URI") from exc
        return value


class DomainError(Exception):
    def __init__(self, code: str, status: int = 409) -> None:
        self.code, self.status = code, status
        super().__init__(code)
