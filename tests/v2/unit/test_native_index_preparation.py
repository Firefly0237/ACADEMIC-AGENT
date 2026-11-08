"""Source preservation and SDK schema controls, not server/index benchmarks."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from uuid import UUID

import pytest
from tokenizers import Tokenizer

from citequest.data_v2.schemas import IndexProjection
from citequest.index_v2.projection import compile_projection
from citequest.index_v2.rows import compile_row, expected_manifest, validate_complete_units
from citequest.index_v2.schema import IndexProfile, build_schema, checked_collection_name, checked_endpoint
from citequest.materials_v2.chunking import SourceChunker

ROOT = Path(__file__).resolve().parents[3]
SID = UUID("7e1ae64c-a3a0-4fd0-947b-b0730c1167f1")
PID = UUID("0a790f6b-70ce-4fc6-a8db-764315fc42b8")


@pytest.fixture(scope="module")
def counter():
    profile = json.loads((ROOT / "runtime/s04/tokenizer-profile.json").read_text(encoding="utf-8"))
    path = ROOT / profile["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == profile["sha256"]
    tokenizer = Tokenizer.from_file(str(path))
    return lambda text: len(tokenizer.encode(text, add_special_tokens=True).ids)


def make(counter, **changes):
    values = {"snapshot_id": SID, "projection_id": PID, "native_id": "235352554", "security_domain": "public",
              "source_namespace": "litsearch_s2orc", "source_kind": "standard_title_abstract",
              "projection_sha256": "a" * 64, "source_pointer_sha256": "b" * 64, "tokenizer_profile_sha256": "c" * 64,
              "full_unit_text": "中文𝒙😀\r\nLiteral science.", "start": 0, "end": 23, "ordinal": 1, "exact_token_count": counter}
    values["end"] = len(values["full_unit_text"])
    values.update(changes)
    return compile_row(**values)


def test_literal_unicode_and_unknown_year(counter):
    row = make(counter)
    assert row["text"] == "中文𝒙😀\r\nLiteral science."
    assert row["year"] is None
    assert {"dense", "sparse"}.isdisjoint(row)
    assert expected_manifest([row], exact_token_count=counter)["count"] == 1


def test_different_native_IDs_and_profiles_do_not_merge_text(counter):
    left = make(counter)
    right = make(counter, projection_id=UUID("0a790f6b-70ce-4fc6-a8db-764315fc42b9"), native_id="235352555")
    alternate = make(counter, projection_sha256="d" * 64)
    assert len({left["chunk_id"], right["chunk_id"], alternate["chunk_id"]}) == 3
    assert expected_manifest([left, right], exact_token_count=counter)["count"] == 2


@pytest.mark.parametrize("changes", [
    {"start": True}, {"end": 10000}, {"ordinal": 0}, {"snapshot_id": str(SID)}, {"year": 0}, {"year": True},
    {"native_id": "中" * 171}, {"security_domain": "owner:someone"}, {"security_domain": "tenant:any"},
    {"source_kind": "fake_paper"}, {"source_pointer_sha256": "not-a-hash"},
    {"full_unit_text": " \r\n", "end": 3}, {"full_unit_text": "中" * 21846, "end": 21846},
    {"full_unit_text": "\ud800", "end": 1},
])
def test_invalid_rows_are_rejected(counter, changes):
    with pytest.raises((ValueError, UnicodeError)):
        make(counter, **changes)


def test_real_pinned_tokenizer_over_limit_rejected(counter):
    text = "embedding " * 2500
    assert counter(text) > 2048
    with pytest.raises(ValueError, match="token_limit"):
        make(counter, full_unit_text=text, end=len(text))


def test_complete_span_union_and_tail_and_gap(counter):
    text = "alpha beta gamma delta"
    first = make(counter, full_unit_text=text, end=10)
    overlap = make(counter, full_unit_text=text, start=6, end=len(text), ordinal=2)
    validate_complete_units([first, overlap])
    with pytest.raises(ValueError, match="tail"):
        validate_complete_units([first])
    gap = make(counter, full_unit_text=text, start=11, end=len(text), ordinal=2)
    with pytest.raises(ValueError, match="gap"):
        validate_complete_units([first, gap])


@pytest.mark.parametrize("mutation", ["text", "identity", "vector", "duplicate"])
def test_prewrite_manifest_rejects_tamper(counter, mutation):
    row = copy.deepcopy(make(counter))
    rows = [row]
    if mutation == "text":
        row["text"] += "!"
    elif mutation == "identity":
        row["projection_sha256"] = "e" * 64
    elif mutation == "vector":
        row["dense"] = [0.0] * 1024
    else:
        rows.append(row)
    with pytest.raises(ValueError):
        expected_manifest(rows, exact_token_count=counter)


@pytest.mark.parametrize("bad", ["https://example.com", "http://tei-strict:8080/embed", "http://u:p@tei-strict:8080", "http://tei-strict:8080?x=1"])
def test_operator_endpoint_is_not_user_fetch_url(bad):
    with pytest.raises(ValueError):
        checked_endpoint(bad)


def test_profile_and_owned_names_are_explicit():
    assert checked_collection_name("cq_v2_standard_001") == "cq_v2_standard_001"
    with pytest.raises(ValueError):
        checked_collection_name("old_experiment")
    with pytest.raises(ValueError):
        IndexProfile(bm25_k1=2.0)


def test_real_sdk_native_function_schema():
    schema, indexes = build_schema("http://tei-strict:8080")
    value = schema.to_dict()
    assert value["enable_dynamic_field"] is False
    fields = {field["name"]: field for field in value["fields"]}
    assert fields["year"]["nullable"] is True
    assert not any(field.get("is_partition_key") for field in value["fields"])
    assert fields["text"]["params"]["max_length"] == 65535
    functions = {function["name"]: function for function in value["functions"]}
    assert len(functions) == 2 and functions["text_bm25"]["params"] == {}
    assert functions["text_dense"]["params"]["truncate"] == "false"
    assert functions["text_dense"]["params"]["max_client_batch_size"] == 2
    sparse = next(item.to_dict() for item in indexes if item.field_name == "sparse")
    assert sparse["bm25_k1"] == 1.2 and sparse["bm25_b"] == 0.75


def metadata_projection(title, abstract, **changes):
    projection = {"source_pointer_sha256": "b" * 64, "representation_profile": "litsearch-clean-exact/1",
                  "title": title, "abstract": abstract, "field_provenance": {"title": "clean", "abstract": "clean"}}
    value = {"projection_id": PID, "source_row_id": PID, "snapshot_id": SID, "native_id": "235352554",
             "source_namespace": "litsearch_s2orc", "source_revision": "9573fb284a1026c998df47024b888a163f0f0e25",
             "security_domain": "public", "work_id": None, "searchable_text_present": bool((title or "").strip() or (abstract or "").strip()),
             "profile_adopted": False, "snapshot_state": "landing", **projection}
    value["projection_sha256"] = hashlib.sha256(json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    value.update(changes)
    return IndexProjection.model_validate(value)


@pytest.fixture(scope="module")
def source_chunker():
    return SourceChunker(ROOT)


def test_projection_no_fake_Work_no_mutation_and_no_publication(source_chunker):
    original = metadata_projection(None, "Actual abstract 𝒙😀\r\n" * 1200)
    before = original.model_dump(mode="json")
    result = compile_projection(original, source_chunker)
    assert len(result["rows"]) > 1 and result["publication_performed"] is False
    assert original.model_dump(mode="json") == before and original.work_id is None
    assert result["field_ranges"]["title"] == [0, 0]
    assert all(row["year"] is None and row["source_kind"] == "standard_title_abstract" for row in result["rows"])
    assert result["expected"]["count"] == len(result["rows"])


def test_empty_source_record_is_retained_without_fake_title(source_chunker):
    original = metadata_projection(" \r\n", None)
    result = compile_projection(original, source_chunker)
    assert result["indexability"] == "empty_source_representation" and result["rows"] == []
    assert result["omitted_record"] is False and original.title == " \r\n"


def test_disabled_source_is_rejected(source_chunker):
    with pytest.raises(ValueError, match="disabled_source"):
        compile_projection(metadata_projection("Title", "Abstract", snapshot_state="disabled"), source_chunker)


@pytest.mark.parametrize("title,abstract", [("科学𝒙 " * 1200, None), (None, "科学𝒙 " * 1200), ("Title", "Actual abstract " * 2000 + "\r\n")])
def test_whitespace_boundaries_preserve_entire_metadata_unit(source_chunker, title, abstract):
    original = metadata_projection(title, abstract)
    result = compile_projection(original, source_chunker)
    text = (title or "") + "\n" + (abstract or "")
    assert all(row["text"].strip() for row in result["rows"])
    assert result["rows"][0]["span_start"] == 0
    assert result["rows"][-1]["span_end"] == len(text)
    assert all(row["text"] == text[row["span_start"]:row["span_end"]] for row in result["rows"])
