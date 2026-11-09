"""Versioned Docling source view; PDF bytes have no universal Unicode order.

This module does not repair invalid native spans, infer missing characters,
perform fuzzy matching, or assert correctness of recognized formulas/tables.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .text import MaterialError

BASIS = "docling-text-provenance-page-view/1"
STRUCTURE_VERSION = "docling-native-structure/1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _box(box: dict, width: float, height: float) -> list[float]:
    numbers = [box.get(k) for k in ("l", "b", "r", "t")]
    if any(type(v) not in {int, float} or not math.isfinite(v) for v in numbers):
        raise MaterialError("INVALID_NATIVE_BOUNDING_BOX")
    left, bottom, right, top = numbers
    if box.get("coord_origin") == "TOPLEFT":
        bottom, top = height - bottom, height - top
    elif box.get("coord_origin") != "BOTTOMLEFT":
        raise MaterialError("UNKNOWN_NATIVE_COORDINATE_ORIGIN")
    if not 0 <= left < right <= width or not 0 <= bottom < top <= height:
        raise MaterialError("NATIVE_BOUNDING_BOX_OUT_OF_PAGE")
    return [left, bottom, right, top]


def _native_structure(item: dict, cell: dict | None = None) -> dict:
    """Copy declared Docling structure without inferring reading order or headers."""
    result = {"structure_version": STRUCTURE_VERSION}
    if "label" in item:
        if type(item["label"]) is not str:
            raise MaterialError("NATIVE_STRUCTURE_LABEL_INVALID")
        result["item_label"] = item["label"]
    parent = item.get("parent")
    if parent is not None:
        if not isinstance(parent, dict) or type(parent.get("$ref")) is not str:
            raise MaterialError("NATIVE_STRUCTURE_PARENT_INVALID")
        result["parent_ref"] = parent["$ref"]
    if item.get("label") == "section_header" and "level" in item:
        if type(item["level"]) is not int or item["level"] < 1:
            raise MaterialError("NATIVE_STRUCTURE_SECTION_LEVEL_INVALID")
        result["section_level"] = item["level"]
    if cell is not None:
        structure = {}
        for name in ("row_span", "col_span", "start_row_offset_idx", "end_row_offset_idx",
                     "start_col_offset_idx", "end_col_offset_idx"):
            if name in cell:
                if type(cell[name]) is not int or cell[name] < (1 if name.endswith("_span") else 0):
                    raise MaterialError("NATIVE_TABLE_CELL_STRUCTURE_INVALID")
                structure[name] = cell[name]
        for name in ("column_header", "row_header", "row_section"):
            if name in cell:
                if type(cell[name]) is not bool:
                    raise MaterialError("NATIVE_TABLE_HEADER_FLAG_INVALID")
                structure[name] = cell[name]
        if structure:
            result["table_cell"] = structure
    return result


def native_pdf_view(document: dict, *, source_sha256: str, native_sha256: str, config_sha256: str,
                    page_count: int, structure_version: str | None = STRUCTURE_VERSION) -> dict:
    """Build exact source quotes from native text and validated native proof.

    Locator Unicode offsets address this explicit, hashed page representation,
    never a claimed intrinsic PDF string or offsets in the normalized index.
    """
    if structure_version not in {None, STRUCTURE_VERSION}:
        raise MaterialError("NATIVE_STRUCTURE_VERSION_UNSUPPORTED")
    raw_pages = document.get("pages", {})
    if set(raw_pages) != {str(n) for n in range(1, page_count + 1)}:
        raise MaterialError("NATIVE_DOCUMENT_PAGE_SET_MISMATCH")
    pages = {}
    for key, page in raw_pages.items():
        width, height = page["size"]["width"], page["size"]["height"]
        if any(type(v) not in {int, float} or not math.isfinite(v) or v <= 0 for v in (width, height)):
            raise MaterialError("INVALID_NATIVE_PAGE_SIZE")
        pages[int(key)] = {"page_1based": int(key), "width": width, "height": height, "text": ""}
    blocks, missing, formulas, table_warnings = [], [], [], []
    refs: set[str] = set()

    def append(reference: str, quote: str, page_no: int, bbox: dict, native: dict, granularity: str) -> None:
        if type(page_no) is not int or page_no not in pages:
            raise MaterialError("INVALID_NATIVE_PAGE_NUMBER")
        page = pages[page_no]
        checked_box = _box(bbox, page["width"], page["height"])
        start = len(page["text"])
        page["text"] += quote
        end = len(page["text"])
        page["text"] += "\n"
        blocks.append({"block_id": reference, "text": quote, "source_quote": quote,
                       "source_quote_sha256": hashlib.sha256(quote.encode()).hexdigest(),
                       "locator": {"kind": "pdf", "file_sha256": source_sha256, "page_1based": page_no, "char_start": start, "char_end": end, "bbox": checked_box},
                       "locator_text_basis": BASIS, "bbox_coordinate_origin": "BOTTOMLEFT", "native_provenance": native,
                       "granularity": granularity, "anchor_valid": True})

    for item in document.get("texts", []):
        reference = item["self_ref"]
        if reference in refs:
            raise MaterialError("DUPLICATE_NATIVE_ITEM_REFERENCE")
        refs.add(reference)
        # Official reading-order contract indexes item.text; orig can retain a
        # list marker. Never apply text offsets to a different orig string.
        original = item.get("text")
        if type(original) is not str:
            missing.append({"item_ref": reference, "reason": "missing_native_text"})
            continue
        if not original.strip():
            continue
        proofs = item.get("prov", [])
        previous, planned = 0, []
        try:
            if not proofs:
                raise MaterialError("NATIVE_ITEM_WITHOUT_PROVENANCE")
            for index, proof in enumerate(proofs):
                span = proof.get("charspan", [])
                if len(span) != 2 or any(type(v) is not int for v in span):
                    raise MaterialError("INVALID_NATIVE_CHARSPAN")
                start, end = span
                if not previous <= start < end <= len(original) or original[previous:start].strip():
                    raise MaterialError("NATIVE_CHARSPAN_OUT_OF_RANGE_OR_INCOMPLETE")
                page_no = proof["page_no"]
                if type(page_no) is not int or page_no not in pages:
                    raise MaterialError("INVALID_NATIVE_PAGE_NUMBER")
                _box(proof["bbox"], pages[page_no]["width"], pages[page_no]["height"])
                planned.append((index, proof, original[start:end]))
                previous = end
            if original[previous:].strip():
                raise MaterialError("NATIVE_CHARSPAN_LEAVES_TEXT_UNMAPPED")
        except (MaterialError, KeyError, TypeError) as exc:
            missing.append({"item_ref": reference, "reason": getattr(exc, "code", type(exc).__name__), "native_text_characters": len(original), "native_prov": proofs})
            continue
        for index, proof, quote in planned:
            native = {"item_ref": reference, "prov_index": index, "charspan_in_native_text": proof["charspan"],
                      "native_text_sha256": hashlib.sha256(original.encode()).hexdigest(),
                      "native_orig_sha256": hashlib.sha256(item.get("orig", "").encode()).hexdigest(),
                      "orig_differs_from_text": item.get("orig") != original}
            if structure_version is not None:
                native.update(_native_structure(item))
            append(f"{reference}/prov/{index}", quote, proof["page_no"], proof["bbox"],
                   native,
                   "formula" if item.get("label") == "formula" else "paragraph")
        if item.get("label") == "formula":
            formulas.append(f"{reference}: recognized formula text is unverified; consult the source PDF")
    for table in document.get("tables", []):
        reference = table["self_ref"]
        if reference in refs:
            raise MaterialError("DUPLICATE_NATIVE_ITEM_REFERENCE")
        refs.add(reference)
        proofs = table.get("prov", [])
        for index, cell in enumerate(table.get("data", {}).get("table_cells", [])):
            quote = cell.get("text", "")
            if not quote.strip():
                continue
            cell_ref = f"{reference}/cell/{index}"
            try:
                if len(proofs) != 1 or not cell.get("bbox"):
                    raise MaterialError("TABLE_CELL_PAGE_OR_BOX_NOT_UNAMBIGUOUS")
                native = {"table_ref": reference, "cell_index": index,
                          "cell_text_sha256": hashlib.sha256(quote.encode()).hexdigest()}
                if structure_version is not None:
                    native.update(_native_structure(table, cell))
                append(cell_ref, quote, proofs[0]["page_no"], cell["bbox"], native, "table")
            except (MaterialError, KeyError, TypeError) as exc:
                missing.append({"item_ref": cell_ref, "reason": getattr(exc, "code", type(exc).__name__)})
        table_warnings.append(f"{reference}: native cell text/boxes preserved; cell semantics and merged-cell relationships are not independently verified")
    for page in pages.values():
        if not page["text"].strip():
            missing.append({"item_ref": f"page:{page['page_1based']}", "reason": "NO_VALID_NATIVE_TEXT_FOR_ADMITTED_TEXTLAYER_PAGE"})
        page["text_sha256"] = hashlib.sha256(page["text"].encode()).hexdigest()
    source_view = {"basis": BASIS, "file_sha256": source_sha256, "native_document_sha256": native_sha256, "config_sha256": config_sha256,
                   "pages": [pages[n] for n in sorted(pages)], "missing_ranges": missing}
    if structure_version is not None:
        source_view["native_structure_version"] = structure_version
    representation_id = _digest(source_view)
    for block in blocks:
        block["representation_id"] = representation_id
    return {**source_view, "representation_id": representation_id, "blocks": blocks, "formula_warnings": formulas, "table_warnings": table_warnings,
            "coverage": "partial" if missing else "text_full", "source_semantics_verified": False,
            "qa_scope": "Validated source-view text only; visual figures, formula meaning, numeric/table accuracy and intrinsic PDF reading order are not guaranteed"}
