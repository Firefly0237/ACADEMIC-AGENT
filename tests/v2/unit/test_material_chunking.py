"""Real pinned BGE tokenization of synthetic fixtures; not LLM evaluation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from citequest.materials_v2.chunking import SourceChunker
from citequest.materials_v2.text import MaterialError

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def chunker():
    return SourceChunker(ROOT)


@pytest.mark.parametrize("source", ["研究" * 5000, "alpha beta gamma\n" * 1000, "数学𝒙😀 é \r\n" * 700,
                                    "```python\nprint('literal code only')\n```\n$$E=mc^2$$\n" * 100,
                                    "\u200b" * 2000, "سياق العربية Ελληνικά 中文\t" * 500])
def test_hierarchical_split_preserves_original_unicode_and_exact_counts(chunker, source):
    sha = hashlib.sha256(source.encode()).hexdigest()
    result = chunker.split([{"block_id": "fixture:1", "source_quote": source,
                            "locator": {"kind": "text", "file_sha256": sha, "char_start": 0, "char_end": len(source), "line_1based": 1}}])
    ranges = chunker.span_ranges(source)
    assert ranges == [{"char_start": r["locator"]["char_start"], "char_end": r["locator"]["char_end"],
                       "token_count": r["token_count"], "target_exceeded_due_to_source_boundary": r["target_exceeded_due_to_source_boundary"]} for r in result]
    cover = bytearray(len(source))
    for chunk in result:
        loc = chunk["locator"]
        assert source[loc["char_start"]:loc["char_end"]] == chunk["source_text"]
        assert chunk["source_text"] == chunk["index_text"]
        assert hashlib.sha256(chunk["source_text"].encode()).hexdigest() == chunk["source_text_sha256"]
        assert chunk["token_count"] == len(chunker.tokenizer.encode(chunk["source_text"], add_special_tokens=True).ids)
        assert 1 <= chunk["token_count"] <= 2048
        cover[loc["char_start"]:loc["char_end"]] = b"1" * (loc["char_end"] - loc["char_start"])
    assert all(cover)


def test_wrong_range_and_null_cannot_be_cleaned_into_valid_source(chunker):
    locator = {"kind": "text", "file_sha256": "a" * 64, "char_start": 0, "char_end": 1, "line_1based": 1}
    with pytest.raises(MaterialError, match="CHUNK_INPUT_LOCATOR_SOURCE_MISMATCH"):
        chunker.split([{"block_id": "fixture", "source_quote": "two", "locator": locator}])
    with pytest.raises(MaterialError, match="CHUNK_SOURCE_NUL_UNSUPPORTED"):
        chunker.split([{"block_id": "fixture", "source_quote": "\x00", "locator": locator}])


def test_unidentified_metadata_ranges_do_not_create_file_identity(chunker):
    source = "Metadata title 𝒙😀\n摘要\r\n" * 300
    ranges = chunker.span_ranges(source)
    assert ranges and all(set(r) == {"char_start", "char_end", "token_count", "target_exceeded_due_to_source_boundary"} for r in ranges)
    cover = bytearray(len(source))
    for r in ranges:
        assert 0 <= r["char_start"] < r["char_end"] <= len(source)
        assert chunker.count(source[r["char_start"]:r["char_end"]]) == r["token_count"]
        cover[r["char_start"]:r["char_end"]] = b"1" * (r["char_end"] - r["char_start"])
    assert all(cover)
    assert chunker.span_ranges("") == []
    with pytest.raises(MaterialError, match="CHUNK_SOURCE_TYPE_UNSUPPORTED"):
        chunker.span_ranges(None)
