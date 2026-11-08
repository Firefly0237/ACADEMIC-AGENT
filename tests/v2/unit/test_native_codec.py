"""Native source codec controls; synthetic strings are never labelled real papers."""

import copy
import hashlib
import json

import pytest
from pydantic import ValidationError

from citequest.data_v2.native_codec import (
    NativeRecordPointer,
    NativeRepresentationDraft,
    NativeTextLocator,
    decode_native,
)
from citequest.data_v2.schemas import RawRowLocator


def pointer(**changes):
    return NativeRecordPointer(
        source_namespace="litsearch_s2orc",
        source_revision="a" * 40,
        source_snapshot_sha256="b" * 64,
        native_id="123",
        raw_location=RawRowLocator(
            artifact_path="raw/source.parquet",
            artifact_sha256="c" * 64,
            row_group=0,
            row_in_group=1,
            row_ordinal=1,
        ),
        **changes,
    )


def record():
    return {
        "corpusid": 123,
        "content": {
            "text": "研究A😀\r\n正文\nEnd",
            "annotations": {
                "title": '[{"start":"0", "end":"4", "attributes":{"k":"原值"}}]',
                "paragraph": '[{"start":6,"end":8}]',
                "empty": None,
                "invalid_optional": '[{"start":100,"end":101}]',
            },
        },
    }


def draft():
    return decode_native(record(), pointer(), "d" * 64)


def test_native_codec_preserves_every_byte_and_original_annotation_string():
    raw, result = record(), draft()
    assert result.original_utf8() == raw["content"]["text"].encode("utf-8")
    assert result.original_annotation_strings == raw["content"]["annotations"]
    assert result.source_text_sha256 == hashlib.sha256(result.original_utf8()).hexdigest()
    assert result.annotation_offset_type_conversions["title"] == 1
    assert result.annotation_spans["title"] == [(0, 4)]
    assert result.annotation_errors == {"invalid_optional": "offset_out_of_bounds"}
    assert not result.can_qa and not result.paper_identity_verified


def test_unicode_locators_are_native_coordinates_not_pdf_or_file_coordinates():
    result = draft()
    locator = result.locator(2, 4)
    assert result.read(locator) == "A😀"
    assert locator.coordinate_system == "unicode_codepoint"
    assert locator.kind == "s2orc_native_text"
    assert not {"file_sha256", "page", "page_number", "file_id"} & locator.model_dump().keys()
    for begin, end in [(True, 2), (2, True), (-1, 3), (3, 3), (5, 4), (0, 100)]:
        with pytest.raises(ValueError, match="out_of_bounds"):
            result.locator(begin, end)


def test_serialized_roundtrip_revalidates_source_and_derived_fields():
    result = draft()
    restored = NativeRepresentationDraft.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.read(restored.locator(0, 4)) == "研究A😀"
    changes = [
        ("literal_text", "changed"),
        ("source_text_sha256", "e" * 64),
        ("source_pointer_sha256", "e" * 64),
        ("representation_id", "e" * 64),
        ("annotation_spans", {}),
        ("annotation_errors", {}),
        ("annotation_offset_type_conversions", {}),
        ("original_annotation_strings", {}),
    ]
    for key, value in changes:
        payload = result.model_dump(mode="json")
        payload[key] = value
        with pytest.raises(ValidationError):
            NativeRepresentationDraft.model_validate_json(json.dumps(payload))


def test_nested_mutation_cannot_bypass_read_time_identity_validation():
    result = draft()
    locator = result.locator(0, 4)
    result.annotation_spans["title"].append((1, 2))
    with pytest.raises(ValueError, match="derived_annotations_mismatch"):
        result.read(locator)


def test_locator_from_another_source_is_rejected():
    result = draft()
    payload = result.locator(0, 2).model_dump()
    payload["source_snapshot_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source_mismatch"):
        result.read(NativeTextLocator.model_validate(payload))


def test_batch_reads_match_individual_literals_and_reject_a_single_wrong_identity():
    result = draft()
    locators = [result.locator(0, 4), result.locator(6, 8)]
    assert result.read_many(locators) == [result.read(locator) for locator in locators]
    payload = locators[1].model_dump()
    payload["source_pointer_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source_or_span_mismatch"):
        result.read_many([locators[0], NativeTextLocator.model_validate(payload)])


@pytest.mark.parametrize("value", [True, "123", 123.0, 124, None])
def test_native_id_must_be_the_exact_original_integer(value):
    raw = record()
    raw["corpusid"] = value
    with pytest.raises(ValueError, match="native_ID"):
        decode_native(raw, pointer(), "d" * 64)


@pytest.mark.parametrize("text", [None, "", " \r\n", "has\x00nul", "x" * 200001])
def test_invalid_source_admission_never_silently_truncates(text):
    raw = record()
    raw["content"]["text"] = text
    with pytest.raises(ValueError):
        decode_native(raw, pointer(), "d" * 64)


def test_invalid_annotation_types_and_qa_grants_are_not_accepted():
    raw = copy.deepcopy(record())
    raw["content"]["annotations"]["paragraph"] = []
    with pytest.raises(ValueError, match="field_types"):
        decode_native(raw, pointer(), "d" * 64)
    payload = draft().model_dump(mode="json")
    payload["can_qa"] = True
    with pytest.raises(ValidationError):
        NativeRepresentationDraft.model_validate_json(json.dumps(payload))
