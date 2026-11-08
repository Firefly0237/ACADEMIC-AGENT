"""Official LangChain semantic splitting + HF overflow, with exact source maps."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.metadata
import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tokenizers import Tokenizer

from .text import MaterialError


def _execution_owner() -> tuple[int, object | None]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), task


@dataclass
class _VerificationScope:
    owner: tuple[int, object | None]
    active: bool = True


class SourceChunker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.profile_path = root / "runtime/s04/tokenizer-profile.json"
        self.profile_bytes = self.profile_path.read_bytes()
        self.profile = json.loads(self.profile_bytes)
        self.profile_sha256 = hashlib.sha256(self.profile_bytes).hexdigest()
        tokenizer_path = root / self.profile["file"]
        tokenizer_raw = tokenizer_path.read_bytes()
        if (len(tokenizer_raw) != self.profile["bytes"]
                or hashlib.sha256(tokenizer_raw).hexdigest() != self.profile["sha256"]):
            raise MaterialError("TOKENIZER_BYTES_CHANGED")
        if (importlib.metadata.version("tokenizers") != self.profile["tokenizers_version"]
                or importlib.metadata.version("langchain-text-splitters") != "1.1.3"):
            raise MaterialError("CHUNKER_DEPENDENCY_VERSION_MISMATCH")
        self.tokenizer = Tokenizer.from_str(tokenizer_raw.decode("utf-8"))
        self.tokenizer.no_truncation()
        captured_tokenizer = self.tokenizer
        self._count_cached = lru_cache(maxsize=4096)(lambda value: len(captured_tokenizer.encode(value, add_special_tokens=True).ids))
        self.windows = Tokenizer.from_str(self.tokenizer.to_str())
        self.windows.enable_truncation(max_length=self.profile["chunk_target_tokens"], stride=self.profile["chunk_overlap_tokens"])
        # Coarse units are contiguous semantic boundaries. Native HF overflow
        # supplies the actual within-unit 512/64 windows and their source offsets.
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.profile["maximum_chunk_tokens"], chunk_overlap=0,
                                                       length_function=self.count, strip_whitespace=False, keep_separator=True,
                                                       separators=["\n\n", "\n", "。", "；", " ", ""])
        self.tokenizer_version = f"{self.profile['model_id']}@{self.profile['revision']};tokenizers={self.profile['tokenizers_version']};special=true;sha256={self.profile['sha256']}"
        self.chunker_version = "langchain-recursive1.1.3-2048-0+hf-overflow512-stride64/source-map1"
        self._verification_depth: ContextVar[int] = ContextVar("source_chunker_verification_depth", default=0)
        self._verification_scope: ContextVar[_VerificationScope | None] = ContextVar("source_chunker_verification_scope", default=None)

    def count(self, text: str) -> int:
        return self._count_cached(text)

    def verify_assets(self) -> None:
        """Verify disk assets against the immutable tokenizer loaded in memory."""
        if (hashlib.sha256(self.profile_path.read_bytes()).hexdigest() != self.profile_sha256
                or hashlib.sha256((self.root / self.profile["file"]).read_bytes()).hexdigest() != self.profile["sha256"]):
            raise MaterialError("TOKENIZER_PROFILE_OR_BYTES_CHANGED_DURING_CHUNKING")

    def _owns_active_scope(self) -> bool:
        scope = self._verification_scope.get()
        return scope is not None and scope.active and scope.owner == _execution_owner()

    @contextmanager
    def verified_batch(self) -> Iterator[None]:
        """Check the whole operation before/after, rather than rehash 17MB per block.

        Results inside the context are staged. A caller must exit successfully
        before accepting/publishing them. The loaded Rust tokenizer is unchanged;
        this scope only coalesces disk-integrity checks. Depth is context-local.
        """
        outer = not self._owns_active_scope()
        depth = 0 if outer else self._verification_depth.get()
        scope_token = None
        if outer:
            self.verify_assets()
            scope = _VerificationScope(owner=_execution_owner())
            scope_token = self._verification_scope.set(scope)
        token = self._verification_depth.set(depth + 1)
        try:
            yield
        finally:
            self._verification_depth.reset(token)
            if outer:
                scope.active = False  # Revoke copied task/thread contexts before exit verification.
                self._verification_scope.reset(scope_token)
                self.verify_assets()

    def _windows(self, text: str) -> list[tuple[int, int]]:
        if self.count(text) <= self.profile["chunk_target_tokens"]:
            return [(0, len(text))]
        encoding = self.windows.encode(text, add_special_tokens=True)
        encoded = [encoding, *encoding.overflowing]
        spans = []
        for piece in encoded:
            offsets = [(a, b) for (a, b), special in zip(piece.offsets, piece.special_tokens_mask, strict=True) if not special and a < b]
            if not offsets or any(not 0 <= a < b <= len(text) for a, b in offsets):
                raise MaterialError("TOKENIZER_SOURCE_OFFSETS_INVALID")
            spans.append([min(a for a, _ in offsets), max(b for _, b in offsets)])
        spans[0][0], spans[-1][1] = 0, len(text)
        for previous, current in pairwise(spans):
            if current[0] < previous[0] or current[1] <= previous[1]:
                raise MaterialError("TOKENIZER_OVERFLOW_NOT_PROGRESSING")
            if current[0] > previous[1]:
                current[0] = previous[1]  # Preserve stripped separator/control codepoints too.
        return [(a, b) for a, b in spans]

    def span_ranges(self, source: str) -> list[dict]:
        """Lossless Unicode ranges, without inventing a File/PDF source identity.

        Callers provide their own truthful typed source projection. These exact
        ranges are also used by uploaded-file split(), with the same locked
        tokenizer, semantic boundaries, overflow windows and hard token cap.
        """
        if type(source) is not str:
            raise MaterialError("CHUNK_SOURCE_TYPE_UNSUPPORTED")
        if "\x00" in source:
            raise MaterialError("CHUNK_SOURCE_NUL_UNSUPPORTED")
        coarse = self.splitter.split_text(source)
        if "".join(coarse) != source:
            raise MaterialError("SEMANTIC_SPLITTER_SOURCE_COVERAGE_FAILED")
        ranges, cursor = [], 0
        for unit in coarse:
            for start, end in self._windows(unit):
                a, b = cursor + start, cursor + end
                quote = source[a:b]
                if not quote:
                    raise MaterialError("EMPTY_SOURCE_CHUNK")
                count = self.count(quote)
                if not 1 <= count <= self.profile["maximum_chunk_tokens"]:
                    raise MaterialError("CHUNK_TOKENS_EXCEEDED")
                ranges.append({"char_start": a, "char_end": b, "token_count": count,
                               "target_exceeded_due_to_source_boundary": count > self.profile["chunk_target_tokens"]})
            cursor += len(unit)
        if not self._owns_active_scope():
            self.verify_assets()
        return ranges

    def split(self, blocks: list[dict]) -> list[dict]:
        chunks = []
        with self.verified_batch():
            for block in blocks:
                source = block["source_quote"]
                original_locator = block["locator"]
                if (type(source) is not str or any(type(original_locator.get(k)) is not int for k in ("char_start", "char_end"))
                        or not 0 <= original_locator["char_start"] < original_locator["char_end"]
                        or len(source) != original_locator["char_end"] - original_locator["char_start"]):
                    raise MaterialError("CHUNK_INPUT_LOCATOR_SOURCE_MISMATCH")
                for span in self.span_ranges(source):
                    a, b = span["char_start"], span["char_end"]
                    quote = source[a:b]
                    locator = copy.deepcopy(block["locator"])
                    locator["char_start"] += a
                    locator["char_end"] = block["locator"]["char_start"] + b
                    if locator["kind"] == "text":
                        locator["line_1based"] += len(re.findall(r"\r\n|\r|\n", source[:a]))
                    chunks.append({"source_block_id": block["block_id"], "source_text": quote, "source_text_sha256": hashlib.sha256(quote.encode()).hexdigest(),
                                   "index_text": quote, "locator": locator, "granularity": block.get("granularity", "paragraph"), "token_count": span["token_count"],
                                   "tokenizer_version": self.tokenizer_version, "tokenizer_profile_sha256": self.profile_sha256, "chunker_version": self.chunker_version,
                                   "target_exceeded_due_to_source_boundary": span["target_exceeded_due_to_source_boundary"],
                                   "bbox_granularity": "native region enclosing this character span" if locator["kind"] == "pdf" else None})
        return chunks
