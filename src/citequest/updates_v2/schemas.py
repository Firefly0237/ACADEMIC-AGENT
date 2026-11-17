"""Typed protocol origins, bounded jobs and immutable index-source variants."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, Field, model_validator

from citequest.updates_v2.protocols import canonical_arxiv, digest
from citequest.v2.domain import ID, SHA, Input


class WindowInput(Input):
    source_namespace: Literal["arxiv", "crossref"]
    mode: Literal["production", "protocol_canary", "synthetic_fixture"]
    lower_bound: AwareDatetime
    upper_bound: AwareDatetime
    managed_ids: Annotated[list[str], Field(max_length=10000)]
    scope_origin_id: ID | None = None
    query: str = "cat:cs.*"
    page_size: Annotated[int, Field(ge=2, le=100)] = 20
    overlap: Annotated[int, Field(ge=1, le=50)] = 4
    maximum_pages: Annotated[int, Field(ge=1, le=100)] = 10
    maximum_records: Annotated[int, Field(ge=1, le=10000)] = 1000
    maximum_network_calls: Annotated[int, Field(ge=1, le=20000)]
    maximum_active_seconds: Annotated[int, Field(ge=30, le=86400)]
    production_stale_seconds: Annotated[int, Field(ge=1)] | None = None
    client_operation_id: ID
    protocol_profile: Literal["oai-fixed-window/1"] | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def fixed_scope(self) -> WindowInput:
        if self.lower_bound >= self.upper_bound or self.overlap >= self.page_size:
            raise ValueError("fixed_valid_window_and_overlap_required")
        if self.protocol_profile is not None:
            if (self.source_namespace != "arxiv" or self.query != "cat:cs.*"
                or any(value.utcoffset().total_seconds() != 0 or (value.hour, value.minute, value.second, value.microsecond) != (0, 0, 0, 0)
                    for value in (self.lower_bound, self.upper_bound))):
                raise ValueError("OAI_fixed_UTC_day_window_and_cs_set_required")
        if len(set(self.managed_ids)) != len(self.managed_ids):
            raise ValueError("managed_ID_scope_must_be_unique")
        if self.managed_ids != sorted(self.managed_ids):
            raise ValueError("managed_ID_scope_must_be_frozen_sorted")
        if self.source_namespace == "arxiv":
            for identifier in self.managed_ids:
                if canonical_arxiv(identifier)[1] is not None:
                    raise ValueError("managed_scope_uses_base_arxiv_IDs")
            if not re.fullmatch(r"cat:cs\.(?:\*|[A-Z]{2})(?: OR cat:cs\.(?:\*|[A-Z]{2}))*", self.query):
                raise ValueError("explicit_computer_science_category_scope_required")
        elif any(not re.fullmatch(r"10\.\d{4,9}/\S+", value) for value in self.managed_ids):
            raise ValueError("known_DOI_scope_required")
        return self


class ProtocolLocator(Input):
    kind: Literal["atom_entry", "oai_record", "crossref_work", "oai_error"]
    raw_response_sha256: SHA
    record_ordinal: Annotated[int, Field(ge=0)] | None
    identifier: str
    source_url: str
    coordinate_system: Literal["parsed_XML_or_JSON_record_identifier_and_ordinal_not_PDF_or_byte_offsets"]

    @model_validator(mode="after")
    def error_is_not_a_returned_record(self) -> ProtocolLocator:
        if (self.kind == "oai_error") != (self.record_ordinal is None):
            raise ValueError("OAI_error_is_a_protocol_error_not_a_returned_record_ordinal")
        return self


class UpdateIndexProjection(Input):
    observation_id: ID
    unit_id: ID
    seal_id: ID
    security_domain: str
    source_namespace: str
    native_id: str
    source_revision: str | None
    source_kind: Literal["source_update_metadata"] = "source_update_metadata"
    purpose: Literal["metadata_update"] = "metadata_update"
    representation_profile: Literal["cq.protocol-metadata/1"] = "cq.protocol-metadata/1"
    title: str | None
    abstract: str | None
    year: Annotated[int, Field(ge=1000, le=2200)] | None
    work_id: ID | None
    record_sha256: SHA
    projection_sha256: SHA
    source_pointer_sha256: SHA
    locator: ProtocolLocator
    metadata_payload: dict[str, Any]
    seal_sha256: SHA
    membership_sha256: SHA
    publication_eligible: Literal[False]
    current_status: Literal["unknown"] = "unknown"

    @model_validator(mode="after")
    def real_bound_observation(self) -> UpdateIndexProjection:
        if self.unit_id != self.observation_id:
            raise ValueError("stable_unit_ID_must_name_the_actual_observation")
        if self.source_pointer_sha256 != digest(self.locator.model_dump(mode="json")):
            raise ValueError("protocol_source_pointer_mismatch")
        expected = digest({"locator": self.locator.model_dump(mode="json"), "record_sha256": self.record_sha256, "metadata_payload": self.metadata_payload})
        if expected != self.projection_sha256:
            raise ValueError("full_protocol_projection_hash_mismatch")
        for name in ("title", "abstract", "year"):
            if getattr(self, name) != self.metadata_payload.get(name):
                raise ValueError("index_field_differs_from_saved_source_projection")
        return self
