"""Pure tampering controls for asset snapshot consistency; not PDF parser tests."""
from __future__ import annotations

import hashlib
import io
import json

import pytest

from citequest.materials_v2.artifacts import capture_assets
from citequest.materials_v2.provenance import native_pdf_view
from citequest.materials_v2.text import MaterialError
from citequest.v2.storage import CAS


def prepare(tmp_path):
    store = CAS(tmp_path / "cas")
    output, domain = tmp_path / "output", "owner:synthetic-pure-fixture"
    parse = output / "parse"
    parse.mkdir(parents=True)
    original = store.put(domain, io.BytesIO(b"SYNTHETIC NON-PDF FIXTURE; NOT PARSED"), 100)
    document = {"pages": {"1": {"size": {"width": 100, "height": 100}}},
                "texts": [{"self_ref": "#/texts/0", "text": "Synthetic source", "orig": "Synthetic source", "label": "text",
                           "prov": [{"page_no": 1, "charspan": [0, 16], "bbox": {"l": 1, "b": 1, "r": 50, "t": 10, "coord_origin": "BOTTOMLEFT"}}]}]}
    native = json.dumps(document).encode()
    (parse / "docling-document.json").write_bytes(native)
    view = native_pdf_view(document, source_sha256=original.sha256, native_sha256=hashlib.sha256(native).hexdigest(), config_sha256="b" * 64, page_count=1)
    (parse / "source-representation.json").write_text(json.dumps(view), encoding="utf-8")
    (parse / "source-pages.json").write_text(json.dumps(view["pages"]), encoding="utf-8")
    parsed = {"source_sha256": original.sha256, "representation_id": view["representation_id"], "blocks": view["blocks"], "missing_ranges": []}
    result = store.put(domain, io.BytesIO(json.dumps(parsed).encode()), 10000)
    return store, output, parsed, original, result


def test_coherent_asset_snapshot_positive(tmp_path):
    store, output, parsed, original, result = prepare(tmp_path)
    assets = capture_assets(output, parsed, original, result, store, mode="synthetic_consistency_control")
    assert {a.role for a in assets} == {"parser_result", "native_document", "source_representation", "source_pages"}
    for asset in assets:
        store.verify(asset.blob)


@pytest.mark.parametrize("filename,value", [("docling-document.json", '{"changed":true}'), ("source-pages.json", '[{"changed":true}]')])
def test_changed_native_or_page_asset_cannot_be_published_as_existing_view(tmp_path, filename, value):
    store, output, parsed, original, result = prepare(tmp_path)
    (output / "parse" / filename).write_text(value, encoding="utf-8")
    with pytest.raises(MaterialError, match="ASSET_CAPTURED_NATIVE_PROOF_CHANGED"):
        capture_assets(output, parsed, original, result, store, mode="synthetic_consistency_control")


def test_view_with_different_usable_blocks_is_rejected(tmp_path):
    store, output, parsed, original, result = prepare(tmp_path)
    path = output / "parse/source-representation.json"
    view = json.loads(path.read_text(encoding="utf-8"))
    view["blocks"] = []
    path.write_text(json.dumps(view), encoding="utf-8")
    with pytest.raises(MaterialError, match="ASSET_NATIVE_SOURCE_VIEW_MISMATCH"):
        capture_assets(output, parsed, original, result, store, mode="synthetic_consistency_control")


def test_unsaved_parser_result_cannot_be_labelled_with_other_cas_result(tmp_path):
    store, output, parsed, original, result = prepare(tmp_path)
    parsed["extra_unsaved_data"] = True
    with pytest.raises(MaterialError, match="ASSET_PARSER_RESULT_MISMATCH"):
        capture_assets(output, parsed, original, result, store, mode="synthetic_consistency_control")


def test_coordinated_native_and_view_hash_changes_cannot_keep_old_quotes_or_view_id(tmp_path):
    store, output, parsed, original, result = prepare(tmp_path)
    native_path = output / "parse/docling-document.json"
    document = json.loads(native_path.read_bytes())
    text = "Forged source sentence."
    document["texts"][0]["text"] = document["texts"][0]["orig"] = text
    document["texts"][0]["prov"][0]["charspan"] = [0, len(text)]
    raw = json.dumps(document).encode()
    native_path.write_bytes(raw)
    view_path = output / "parse/source-representation.json"
    view = json.loads(view_path.read_bytes())
    view["native_document_sha256"] = hashlib.sha256(raw).hexdigest()
    view_path.write_text(json.dumps(view), encoding="utf-8")
    with pytest.raises(MaterialError, match="ASSET_NATIVE_VIEW_RECONSTRUCTION_MISMATCH"):
        capture_assets(output, parsed, original, result, store, mode="synthetic_consistency_control")
