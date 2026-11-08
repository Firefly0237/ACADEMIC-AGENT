"""Synthetic protocol controls, not substitutes for real parsing evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from citequest.materials_v2.provenance import native_pdf_view
from citequest.materials_v2.text import MaterialError


def document() -> dict:
    return {"pages": {"1": {"size": {"width": 600, "height": 800}}, "2": {"size": {"width": 600, "height": 800}}},
            "texts": [{"self_ref": "#/texts/0", "orig": "- alpha beta", "text": "alpha beta", "label": "list_item",
                       "prov": [{"page_no": 1, "charspan": [0, 5], "bbox": {"l": 10, "b": 20, "r": 100, "t": 40, "coord_origin": "BOTTOMLEFT"}},
                                {"page_no": 2, "charspan": [6, 10], "bbox": {"l": 10, "b": 20, "r": 100, "t": 40, "coord_origin": "BOTTOMLEFT"}}]}], "tables": []}


def view(data: dict) -> dict:
    return native_pdf_view(data, source_sha256="a" * 64, native_sha256="b" * 64, config_sha256="c" * 64, page_count=2)


def test_native_text_spans_are_not_applied_to_different_orig() -> None:
    result = view(document())
    assert result["coverage"] == "text_full"
    assert [b["source_quote"] for b in result["blocks"]] == ["alpha", "beta"]
    assert all(b["native_provenance"]["orig_differs_from_text"] for b in result["blocks"])
    assert result["pages"][0]["text"] == "alpha\n"
    assert result["representation_id"] == view(document())["representation_id"]


@pytest.mark.parametrize("span", [[0, 100], [0, 0], [-1, 5], [True, 5], [0, 5.0], [0], [0, 5, 6]])
def test_invalid_span_never_clips_or_maps_partial_item(span: list) -> None:
    data = document()
    data["texts"][0]["prov"][0]["charspan"] = span
    result = view(data)
    assert result["coverage"] == "partial"
    assert result["blocks"] == []
    assert result["missing_ranges"]


@pytest.mark.parametrize("change", ["overlap", "gap", "tail", "no_proof", "out_page", "bool_page", "nan_box", "outside_box", "unknown_origin"])
def test_incomplete_and_impossible_native_proof_fails_closed(change: str) -> None:
    data = document()
    item = data["texts"][0]
    if change == "overlap":
        item["prov"][1]["charspan"] = [4, 10]
    elif change == "gap":
        item["prov"][1]["charspan"] = [7, 10]
    elif change == "tail":
        item["text"] += "!"
    elif change == "no_proof":
        item["prov"] = []
    elif change == "out_page":
        item["prov"][0]["page_no"] = 3
    elif change == "bool_page":
        item["prov"][0]["page_no"] = True
    elif change == "nan_box":
        item["prov"][0]["bbox"]["l"] = float("nan")
    elif change == "outside_box":
        item["prov"][0]["bbox"]["r"] = 601
    else:
        item["prov"][0]["bbox"]["coord_origin"] = "guess"
    result = view(data)
    assert result["coverage"] == "partial"
    assert result["blocks"] == []


def test_every_emitted_locator_and_hash_returns_exact_source_view() -> None:
    result = view(document())
    contract = json.loads((Path(__file__).resolve().parents[3] / "contracts/v2/domain.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator({"$ref": "#/$defs/Locator", "$defs": contract["$defs"]})
    for block in result["blocks"]:
        locator = block["locator"]
        validator.validate(locator)
        page = result["pages"][locator["page_1based"] - 1]
        quote = page["text"][locator["char_start"]:locator["char_end"]]
        assert quote == block["source_quote"]
        assert hashlib.sha256(quote.encode()).hexdigest() == block["source_quote_sha256"]
    changed = document()
    changed["texts"][0]["text"] = "Alpha beta"
    assert view(changed)["representation_id"] != result["representation_id"]


def test_table_cell_requires_one_unambiguous_page_and_real_cell_box() -> None:
    data = document()
    data["tables"] = [{"self_ref": "#/tables/0", "prov": [{"page_no": 1}], "data": {"table_cells": [{"text": "42", "bbox": {"l": 10, "t": 20, "r": 100, "b": 40, "coord_origin": "TOPLEFT"}}]}}]
    result = view(data)
    assert result["blocks"][-1]["locator"]["bbox"] == [10, 760, 100, 780]
    assert result["table_warnings"]
    for mutate in ("no_box", "two_pages"):
        bad = copy.deepcopy(data)
        if mutate == "no_box":
            bad["tables"][0]["data"]["table_cells"][0]["bbox"] = None
        else:
            bad["tables"][0]["prov"].append({"page_no": 2})
        parsed = view(bad)
        assert parsed["coverage"] == "partial"
        assert len(parsed["blocks"]) == 2
        assert parsed["missing_ranges"][0]["item_ref"] == "#/tables/0/cell/0"


def test_duplicate_references_and_page_set_are_not_silently_replaced() -> None:
    data = document()
    data["texts"].append(copy.deepcopy(data["texts"][0]))
    with pytest.raises(MaterialError, match="DUPLICATE_NATIVE_ITEM_REFERENCE"):
        view(data)
    data = document()
    del data["pages"]["2"]
    with pytest.raises(MaterialError, match="NATIVE_DOCUMENT_PAGE_SET_MISMATCH"):
        view(data)
