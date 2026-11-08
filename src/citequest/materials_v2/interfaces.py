"""S04 adapter boundary. Implementations must run outside the API process."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .text import MaterialLimits


@dataclass(frozen=True)
class ParserInput:
    file_path: Path
    source_sha256: str
    detected_media_type: str
    limits: MaterialLimits
    parser_profile_sha256: str


@dataclass(frozen=True)
class ParseBlock:
    """Locator refers to this original source version, not cleaned text offsets."""
    block_id: str
    text: str
    source_locator: dict
    source_quote: str
    source_quote_sha256: str


@dataclass(frozen=True)
class ParsedMaterial:
    source_sha256: str
    parser_id: str
    parser_profile_sha256: str
    state: Literal["partial", "ready"]
    blocks: tuple[ParseBlock, ...]
    pages_requiring_ocr: tuple[int, ...]
    warnings: tuple[str, ...]
    representation_id: str | None = None
    missing_ranges: tuple[str, ...] = ()
    usable_block_ids: tuple[str, ...] = ()
    can_qa: bool = False
    original_is_immutable: bool = False


class IsolatedParser(Protocol):
    """Worker interface implemented by adapter.JobParser.

    Admission, parser, memory/deadline limits and source integrity are mandatory.
    `ready` means all included QA scope has validated source anchors; OCR and
    formula understanding cannot become enabled through document instructions.
    API/CAS/SQL readiness promotion is a separate short transaction after proof.
    CAS immutability is a storage guarantee, never inferred from a parser run.
    """

    def parse(self, request: ParserInput) -> ParsedMaterial: ...
