"""Lossless codec for the official S2ORC source format; no PDF parsing or File identity."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator

from citequest.v2.domain import SHA, Input

from .native_text import annotation_spans
from .schemas import RawRowLocator

PROFILE = "citequest-native-s2orc-literal/1"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class NativeRecordPointer(Input):
    source_namespace: Literal["litsearch_s2orc"]
    source_revision: Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
    source_snapshot_sha256: SHA
    native_id: Annotated[str, Field(min_length=1, max_length=512)]
    raw_location: RawRowLocator


class NativeTextLocator(Input):
    """A native-source coordinate, never a frozen File/PDF Locator subtype."""

    kind: Literal["s2orc_native_text"] = "s2orc_native_text"
    source_snapshot_sha256: SHA
    source_pointer_sha256: SHA
    source_text_sha256: SHA
    coordinate_system: Literal["unicode_codepoint"] = "unicode_codepoint"
    char_start: Annotated[int, Field(ge=0)]
    char_end: Annotated[int, Field(gt=0)]


class NativeRepresentationDraft(Input):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    producer: Literal["CiteQuest source codec"] = "CiteQuest source codec"
    source_schema: Literal["S2ORC"] = "S2ORC"
    profile: Literal["citequest-native-s2orc-literal/1"] = PROFILE
    profile_sha256: SHA
    source_pointer: NativeRecordPointer
    source_pointer_sha256: SHA
    literal_text: str
    source_text_sha256: SHA
    original_annotation_strings: dict[str, str | None]
    annotation_values_sha256: SHA
    annotation_spans: dict[str, list[tuple[int, int]]]
    annotation_offset_type_conversions: dict[str, int]
    annotation_errors: dict[str, str]
    representation_id: SHA
    source_text_coverage: Literal["entire_native_literal_text"] = "entire_native_literal_text"
    original_PDF_coverage: Literal["not_claimed_by_native_codec"] = "not_claimed_by_native_codec"
    can_qa: Literal[False] = False
    paper_identity_verified: Literal[False] = False

    @model_validator(mode="after")
    def verify_serialized_identity(self) -> NativeRepresentationDraft:
        self.validate_identity()
        return self

    def validate_identity(self) -> None:
        expected = {
            "profile": self.profile,
            "profile_sha256": self.profile_sha256,
            "source_pointer_sha256": digest(self.source_pointer.model_dump(mode="json")),
            "source_text_sha256": hashlib.sha256(self.literal_text.encode("utf-8")).hexdigest(),
            "annotation_values_sha256": digest(self.original_annotation_strings),
        }
        if (
            expected["source_pointer_sha256"] != self.source_pointer_sha256
            or expected["source_text_sha256"] != self.source_text_sha256
            or expected["annotation_values_sha256"] != self.annotation_values_sha256
            or digest(expected) != self.representation_id
        ):
            raise ValueError("native_source_representation_identity_mismatch")
        spans, conversions, errors = _decode_annotations(self.original_annotation_strings, self.literal_text)
        if (
            spans != self.annotation_spans
            or conversions != self.annotation_offset_type_conversions
            or errors != self.annotation_errors
        ):
            raise ValueError("native_source_derived_annotations_mismatch")

    def original_utf8(self) -> bytes:
        self.validate_identity()
        # Binary CAS storage must use these bytes; Windows text-mode writes
        # would translate native LF and break character/hash identity.
        return self.literal_text.encode("utf-8")

    def locator(self, start: int, end: int) -> NativeTextLocator:
        self.validate_identity()
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(self.literal_text):
            raise ValueError("native_source_span_out_of_bounds")
        return NativeTextLocator(
            source_snapshot_sha256=self.source_pointer.source_snapshot_sha256,
            source_pointer_sha256=self.source_pointer_sha256,
            source_text_sha256=self.source_text_sha256,
            char_start=start,
            char_end=end,
        )

    def read(self, locator: NativeTextLocator) -> str:
        self.validate_identity()
        expected = self.locator(locator.char_start, locator.char_end)
        if locator != expected:
            raise ValueError("native_locator_source_mismatch")
        return self.literal_text[locator.char_start : locator.char_end]

    def read_many(self, locators: list[NativeTextLocator]) -> list[str]:
        """Verify the bounded batch once before and after exact literal reads."""
        if len(locators) > 100000:
            raise ValueError("native_locator_batch_too_large")
        self.validate_identity()
        text = self.literal_text
        results = []
        for locator in locators:
            if (
                locator.source_snapshot_sha256 != self.source_pointer.source_snapshot_sha256
                or locator.source_pointer_sha256 != self.source_pointer_sha256
                or locator.source_text_sha256 != self.source_text_sha256
                or not 0 <= locator.char_start < locator.char_end <= len(text)
            ):
                raise ValueError("native_locator_source_or_span_mismatch")
            results.append(text[locator.char_start : locator.char_end])
        self.validate_identity()
        return results


def _decode_annotations(
    raw: dict[str, str | None], text: str
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, int], dict[str, str]]:
    spans, conversions, errors = {}, {}, {}
    for name, value in raw.items():
        try:
            values, count = annotation_spans(value, text)
            spans[name], conversions[name] = values, count
        except (ValueError, TypeError) as exc:
            errors[name] = str(exc)
    return spans, conversions, errors


def decode_native(
    record: dict[str, Any], pointer: NativeRecordPointer, profile_sha256: str
) -> NativeRepresentationDraft:
    """Called after official Arrow decoding and whole-artifact/row validation.

    This internal draft cannot grant QA, paper identity or publication. The
    qualified WorkVersion/source chain later binds the canonical PaperSourceRef.
    """
    if type(record.get("corpusid")) is not int or str(record["corpusid"]) != pointer.native_id:
        raise ValueError("native_ID_does_not_match_validated_source_pointer")
    content = record.get("content")
    if not isinstance(content, dict) or not isinstance(content.get("text"), str):
        raise ValueError("native_source_text_missing")
    text = content["text"]
    if not text.strip() or "\x00" in text or len(text) > 200000 or len(text.encode("utf-8")) > 2 * 1024 * 1024:
        raise ValueError("native_source_text_outside_declared_admission")
    annotations = content.get("annotations")
    if not isinstance(annotations, dict) or any(
        not isinstance(key, str) or (value is not None and not isinstance(value, str))
        for key, value in annotations.items()
    ):
        raise ValueError("native_annotation_field_types_invalid")
    raw = dict(annotations)
    spans, conversions, errors = _decode_annotations(raw, text)
    identity = {
        "profile": PROFILE,
        "profile_sha256": profile_sha256,
        "source_pointer_sha256": digest(pointer.model_dump(mode="json")),
        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "annotation_values_sha256": digest(raw),
    }
    result = NativeRepresentationDraft(
        profile_sha256=profile_sha256,
        source_pointer=pointer,
        source_pointer_sha256=identity["source_pointer_sha256"],
        literal_text=text,
        source_text_sha256=identity["source_text_sha256"],
        original_annotation_strings=raw,
        annotation_values_sha256=identity["annotation_values_sha256"],
        annotation_spans=spans,
        annotation_offset_type_conversions=conversions,
        annotation_errors=errors,
        representation_id=digest(identity),
    )
    result.validate_identity()
    return result
