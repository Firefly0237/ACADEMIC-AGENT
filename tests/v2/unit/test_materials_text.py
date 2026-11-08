"""Pure real byte/Unicode/hash/Locator tests; no document parser or DB claim."""
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from citequest.materials_v2.text import MaterialError, load_limits, prepare_text

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def limits():
    return load_limits(ROOT / "contracts/v2/policy.json")


def test_frozen_input_contract_is_used(limits):
    assert (limits.text_bytes, limits.text_characters, limits.pdf_bytes, limits.pdf_pages) == (2097152, 200000, 52428800, 200)
    assert (limits.docx_bytes, limits.docx_uncompressed_bytes, limits.docx_zip_entries) == (20971520, 209715200, 5000)
    assert len(limits.contract_sha256) == 64


@pytest.mark.parametrize("media", ["text/plain", "text/markdown"])
def test_lossless_unicode_newline_mapping_and_contract_locators(limits, media):
    import json

    source = "首行\r\n第二行\re\u0301，𠀀\n尾"
    material = prepare_text(source.encode(), media, limits)
    assert material.original_utf8 == source.encode()
    assert material.normalized_text == "首行\n第二行\ne\u0301，𠀀\n尾"
    assert material.source_sha256 == hashlib.sha256(source.encode()).hexdigest()
    assert material.normalized_sha256 == hashlib.sha256(material.normalized_text.encode()).hexdigest()
    assert material.source_sha256 != material.normalized_sha256
    assert material.can_qa is False
    domain = json.loads((ROOT / "contracts/v2/domain.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator({"$ref": "#/$defs/Locator", "$defs": domain["$defs"]})
    # All possible non-empty spans exercise CRLF, lone CR, combining and astral
    # boundaries; the check uses source text, not the mapping implementation.
    for start in range(len(material.normalized_text)):
        for end in range(start + 1, len(material.normalized_text) + 1):
            locator = material.locator(start, end)
            assert not list(validator.iter_errors(locator))
            original_quote = material.read_locator(locator)
            assert original_quote.replace("\r\n", "\n").replace("\r", "\n") == material.normalized_text[start:end]
            assert "page_1based" not in locator


@pytest.mark.parametrize("data,code", [(b"", "EMPTY_TEXT"), (b"  \r\n\t", "EMPTY_TEXT"), (b"\xef\xbb\xbf", "EMPTY_TEXT"),
                                       (b"\xff", "INVALID_UTF8"), (b"text\x00tail", "NUL_CHARACTER"),
                                       (b"a" * 200001, "TEXT_CHARACTERS_EXCEEDED"), (b"a" * 2097153, "TEXT_BYTES_EXCEEDED")])
def test_text_rejections_before_any_ready_state(limits, data, code):
    with pytest.raises(MaterialError, match=code):
        prepare_text(data, "text/plain", limits)


def test_markdown_code_links_remain_unexecuted_data(limits):
    text = "# 草稿\n```python\nprint('data only')\n```\n[untrusted](http://127.0.0.1/private)"
    material = prepare_text(text.encode(), "text/markdown", limits)
    assert material.original_text == text == material.normalized_text
    assert material.can_qa is False


@pytest.mark.parametrize("start,end", [(True, 2), (0, False), (-1, 2), (1, 1), (0, 99)])
def test_invalid_normalized_span(limits, start, end):
    material = prepare_text("甲\r\n乙".encode(), "text/plain", limits)
    with pytest.raises(MaterialError, match="INVALID_NORMALIZED_SPAN"):
        material.locator(start, end)


def test_locator_hash_line_and_original_integrity_are_mandatory(limits):
    material = prepare_text("首\r\n尾".encode(), "text/plain", limits)
    locator = material.locator(2, 3)
    for change in ({"file_sha256": "0" * 64}, {"line_1based": 1}, {"char_start": True}):
        with pytest.raises(MaterialError):
            material.read_locator({**locator, **change})
    with pytest.raises(MaterialError, match="ORIGINAL_TEXT_INTEGRITY_ERROR"):
        replace(material, original_utf8=b"changed").read_locator(locator)
