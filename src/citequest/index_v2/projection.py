"""Metadata-only projection adapter; its coordinates are never PDF coordinates."""

from __future__ import annotations

import hashlib

from citequest.data_v2.schemas import IndexProjection
from citequest.index_v2.rows import compile_row, expected_manifest
from citequest.materials_v2.chunking import SourceChunker

BUILDER_VERSION = "litsearch-title-newline-abstract/2-whitespace-boundary-attachment"


def metadata_text(title: str | None, abstract: str | None) -> str:
    if (title is not None and type(title) is not str) or (abstract is not None and type(abstract) is not str):
        raise ValueError("literal_source_strings_or_null_required")
    return (title or "") + "\n" + (abstract or "")


def metadata_ranges(text: str, chunker: SourceChunker) -> list[dict]:
    """Attach whitespace-only splitter units to a neighbour without dropping bytes.

    For example, a long title with no abstract can leave the recipe's terminal
    newline as a standalone LC unit. Keep that newline in the preceding literal
    slice instead of spending an embedding on an empty-information fragment.
    Original source fields, shared splitter and PDF/material chunking are unchanged.
    """
    output, pending_start, pending_end = [], None, None
    for span in chunker.span_ranges(text):
        a, b = span["char_start"], span["char_end"]
        if not text[a:b].strip():
            pending_start = a if pending_start is None else min(pending_start, a)
            pending_end = b
            continue
        if pending_start is not None:
            span = {**span, "char_start": min(a, pending_start)}
            pending_start = pending_end = None
        output.append(span)
    if pending_start is not None and output:
        output[-1] = {**output[-1], "char_end": max(output[-1]["char_end"], pending_end)}
    for span in output:
        span["token_count"] = chunker.count(text[span["char_start"]:span["char_end"]])
        span["target_exceeded_due_to_source_boundary"] = span["token_count"] > chunker.profile["chunk_target_tokens"]
    return output


def compile_projection(projection: IndexProjection, chunker: SourceChunker) -> dict:
    """Prepare immutable SQL metadata using the shared exact, lossless chunker.

    Keeping original nulls in the SQL DTO and using an empty string in this
    declared text recipe are different operations. No body or external metadata
    is substituted. This builder cannot adopt a profile or publish a collection.
    """
    if projection.snapshot_state == "disabled":
        raise ValueError("disabled_source_snapshot")
    identity = {"projection_id": str(projection.projection_id), "snapshot_id": str(projection.snapshot_id),
                "representation_profile": projection.representation_profile, "projection_sha256": projection.projection_sha256,
                "source_pointer_sha256": projection.source_pointer_sha256, "source_revision": projection.source_revision,
                "builder_version": BUILDER_VERSION, "profile_adopted": projection.profile_adopted,
                "publication_performed": False, "tokenizer_profile_sha256": chunker.profile_sha256,
                "coordinate_system": "Unicode codepoints in the declared title-newline-abstract representation, not a PDF page or native body"}
    if not projection.searchable_text_present:
        return {**identity, "indexability": "empty_source_representation", "rows": [], "expected": None,
                "omitted_record": False, "reason": "Preserve the complete source landing row and explicit non-searchable status"}
    title = projection.title or ""
    text = metadata_text(projection.title, projection.abstract)
    ranges = metadata_ranges(text, chunker)
    rows = [compile_row(snapshot_id=projection.snapshot_id, projection_id=projection.projection_id, native_id=projection.native_id,
                        security_domain=projection.security_domain, source_namespace=projection.source_namespace,
                        source_kind="standard_title_abstract", projection_sha256=projection.projection_sha256,
                        source_pointer_sha256=projection.source_pointer_sha256, tokenizer_profile_sha256=chunker.profile_sha256,
                        full_unit_text=text, start=span["char_start"], end=span["char_end"], ordinal=ordinal,
                        exact_token_count=chunker.count, year=projection.year)
            for ordinal, span in enumerate(ranges, 1)]
    return {**identity, "indexability": "compiled_metadata", "unit_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "field_ranges": {"title": [0, len(title)], "separator": [len(title), len(title) + 1], "abstract": [len(title) + 1, len(text)]},
            "rows": rows, "expected": expected_manifest(rows, exact_token_count=chunker.count)}
