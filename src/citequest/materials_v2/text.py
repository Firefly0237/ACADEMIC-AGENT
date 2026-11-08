"""Lossless text admission and original-source locators for uploaded materials."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MaterialError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class MaterialLimits:
    text_bytes: int
    text_characters: int
    pdf_bytes: int
    pdf_pages: int
    docx_bytes: int
    docx_uncompressed_bytes: int
    docx_zip_entries: int
    contract_sha256: str


def load_limits(contract_path: Path) -> MaterialLimits:
    raw = contract_path.read_bytes()
    limits = json.loads(raw)["limits"]
    names = ("text_material_bytes", "text_material_characters", "pdf_bytes", "pdf_pages", "docx_bytes", "docx_uncompressed_bytes", "docx_zip_entries")
    if any(type(limits.get(key)) is not int or limits[key] <= 0 for key in names):
        raise MaterialError("INVALID_MATERIAL_LIMIT_CONTRACT")
    return MaterialLimits(*(limits[key] for key in names), hashlib.sha256(raw).hexdigest())


@dataclass(frozen=True)
class TextMaterial:
    original_utf8: bytes
    original_text: str
    normalized_text: str
    source_sha256: str
    normalized_sha256: str
    normalized_boundaries_in_original: tuple[int, ...]
    media_type: str
    normalization_version: str = "newlines-only/1"
    can_qa: bool = False

    def locator(self, normalized_start: int, normalized_end: int) -> dict[str, Any]:
        """Unicode codepoint offsets in original decoded bytes, never PDF pages."""
        if (type(normalized_start) is not int or type(normalized_end) is not int
                or not 0 <= normalized_start < normalized_end <= len(self.normalized_text)):
            raise MaterialError("INVALID_NORMALIZED_SPAN")
        start = self.normalized_boundaries_in_original[normalized_start]
        end = self.normalized_boundaries_in_original[normalized_end]
        return {"kind": "text", "file_sha256": self.source_sha256, "char_start": start, "char_end": end,
                "line_1based": len(re.findall(r"\r\n|\r|\n", self.original_text[:start])) + 1}

    def read_locator(self, locator: dict[str, Any]) -> str:
        required = {"kind", "file_sha256", "char_start", "char_end", "line_1based"}
        if set(locator) != required or locator["kind"] != "text" or locator["file_sha256"] != self.source_sha256:
            raise MaterialError("LOCATOR_SOURCE_MISMATCH")
        start, end, line = locator["char_start"], locator["char_end"], locator["line_1based"]
        if (any(type(value) is not int for value in (start, end, line))
                or not 0 <= start < end <= len(self.original_text)
                or line != len(re.findall(r"\r\n|\r|\n", self.original_text[:start])) + 1):
            raise MaterialError("INVALID_SOURCE_LOCATOR")
        if hashlib.sha256(self.original_utf8).hexdigest() != self.source_sha256 or self.original_utf8.decode("utf-8") != self.original_text:
            raise MaterialError("ORIGINAL_TEXT_INTEGRITY_ERROR")
        return self.original_text[start:end]


def prepare_text(data: bytes, media_type: str, limits: MaterialLimits) -> TextMaterial:
    if media_type not in {"text/plain", "text/markdown"}:
        raise MaterialError("UNSUPPORTED_TEXT_MEDIA")
    if len(data) > limits.text_bytes:
        raise MaterialError("TEXT_BYTES_EXCEEDED")
    try:
        original = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise MaterialError("INVALID_UTF8") from exc
    if "\x00" in original:
        raise MaterialError("NUL_CHARACTER")
    if len(original) > limits.text_characters:
        raise MaterialError("TEXT_CHARACTERS_EXCEEDED")
    if not original.removeprefix("\ufeff").strip():
        raise MaterialError("EMPTY_TEXT")
    normalized, boundaries, index = [], [0], 0
    while index < len(original):
        if original[index] == "\r":
            index += 2 if original[index:index + 2] == "\r\n" else 1
            normalized.append("\n")
        else:
            normalized.append(original[index])
            index += 1
        boundaries.append(index)
    text = "".join(normalized)
    return TextMaterial(data, original, text, hashlib.sha256(data).hexdigest(), hashlib.sha256(text.encode()).hexdigest(), tuple(boundaries), media_type)
