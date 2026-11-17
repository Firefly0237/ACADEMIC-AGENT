"""Versioned row compiler for metadata/native/update source variants."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Literal
from uuid import UUID

from citequest.index_v2.schema import IndexProfile

SHA = re.compile(r"[a-f0-9]{64}\Z")
SourceKind = Literal["standard_title_abstract", "s2orc_native", "material_native", "update_title_abstract"]
IDENTITY_FIELDS = (
    "row_key",
    "projection_sha256",
    "unit_text_sha256",
    "span_start",
    "span_end",
    "ordinal",
    "source_text_sha256",
    "tokenizer_profile_sha256",
    "profile_sha256",
    "token_count",
)


def canonical_sha(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def bounded_text(value: str, limit: int, name: str) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > limit:
        raise ValueError(f"invalid_or_oversize_{name}")
    return value


def compile_row(
    *,
    snapshot_id: UUID,
    projection_id: UUID,
    native_id: str,
    security_domain: str,
    source_namespace: str,
    source_kind: SourceKind,
    projection_sha256: str,
    source_pointer_sha256: str,
    tokenizer_profile_sha256: str,
    full_unit_text: str,
    start: int,
    end: int,
    ordinal: int,
    exact_token_count: Callable[[str], int],
    year: int | None = None,
    profile: IndexProfile | None = None,
) -> dict:
    """No normalization or truncation. Token callback must be the pinned host tokenizer.

    This compiler is not an authorization endpoint. Its caller must obtain the
    projection and current security domain from the trusted SQL service.
    """
    config = profile or IndexProfile()
    if type(snapshot_id) is not UUID or type(projection_id) is not UUID:
        raise ValueError("typed_snapshot_and_projection_UUIDs_required")
    for name, value in (
        ("projection", projection_sha256),
        ("source_pointer", source_pointer_sha256),
        ("tokenizer", tokenizer_profile_sha256),
    ):
        if type(value) is not str or not SHA.fullmatch(value):
            raise ValueError(f"invalid_{name}_sha256")
    for name, value in (("start", start), ("end", end), ("ordinal", ordinal)):
        if type(value) is not int:
            raise ValueError(f"strict_integer_{name}_required")
    if type(full_unit_text) is not str or not (0 <= start < end <= len(full_unit_text)) or ordinal < 1:
        raise ValueError("nonempty_literal_span_required")
    text = bounded_text(full_unit_text[start:end], config.maximum_text_bytes, "text")
    if not text.strip():
        raise ValueError("empty_index_representation_keep_landing_row")
    token_count = exact_token_count(text)
    if type(token_count) is not int or not 0 < token_count <= config.maximum_tokens:
        raise ValueError("exact_token_limit_exceeded_or_invalid_count")
    if source_kind not in {"standard_title_abstract", "s2orc_native", "material_native", "update_title_abstract"}:
        raise ValueError("undeclared_source_kind")
    if year is not None and (type(year) is not int or not 1 <= year <= 9999):
        raise ValueError("known_calendar_year_or_null_required")
    security_domain = bounded_text(security_domain, 100, "security_domain")
    if security_domain != "public":
        try:
            kind, identifier = security_domain.split(":", 1)
            valid = kind in {"owner", "project"} and str(UUID(identifier)) == identifier
        except (ValueError, AttributeError):
            valid = False
        if not valid:
            raise ValueError("declared_security_domain_required")
    unit_hash = hashlib.sha256(full_unit_text.encode()).hexdigest()
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    row_key = canonical_sha({"snapshot_id": str(snapshot_id), "projection_id": str(projection_id)})
    identity = {
        "row_key": row_key,
        "projection_sha256": projection_sha256,
        "unit_text_sha256": unit_hash,
        "span_start": start,
        "span_end": end,
        "ordinal": ordinal,
        "source_text_sha256": text_hash,
        "tokenizer_profile_sha256": tokenizer_profile_sha256,
        "profile_sha256": config.fingerprint(),
        "token_count": token_count,
    }
    return {
        "chunk_id": canonical_sha(identity),
        **identity,
        "snapshot_id": str(snapshot_id),
        "native_id": bounded_text(native_id, config.maximum_native_id_bytes, "native_id"),
        "security_domain": security_domain,
        "source_namespace": bounded_text(source_namespace, 100, "source_namespace"),
        "source_kind": source_kind,
        "source_pointer_sha256": source_pointer_sha256,
        "unit_codepoints": len(full_unit_text),
        "year": year,
        "text": text,
    }


def validate_complete_units(rows: list[dict]) -> None:
    """Check interval union per nonempty source unit, allowing declared overlap."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["row_key"], []).append(row)
    for chunks in groups.values():
        first = chunks[0]
        end = 0
        for chunk in sorted(chunks, key=lambda item: (item["span_start"], item["span_end"])):
            if (chunk["unit_text_sha256"], chunk["unit_codepoints"]) != (
                first["unit_text_sha256"],
                first["unit_codepoints"],
            ):
                raise ValueError("source_unit_identity_drift")
            if not 0 <= chunk["span_start"] <= end or chunk["span_end"] > first["unit_codepoints"]:
                raise ValueError("source_unit_coverage_gap_or_out_of_bounds")
            end = max(end, chunk["span_end"])
        if end != first["unit_codepoints"]:
            raise ValueError("source_unit_tail_not_covered")


def expected_manifest(
    rows: list[dict], *, exact_token_count: Callable[[str], int], complete_units: bool = True
) -> dict:
    """Exact expected ID/content map; an empty/duplicate batch cannot pass silently."""
    if not rows:
        raise ValueError("nonempty_expected_rows_required")
    result = {}
    profile = IndexProfile()
    allowed = {
        "chunk_id",
        *IDENTITY_FIELDS,
        "snapshot_id",
        "native_id",
        "security_domain",
        "source_namespace",
        "source_kind",
        "source_pointer_sha256",
        "unit_codepoints",
        "year",
        "text",
    }
    for row in rows:
        if set(row) != allowed:
            raise ValueError("index_row_fields_mismatch_or_client_vectors")
        identifier = row["chunk_id"]
        if identifier in result:
            raise ValueError("duplicate_chunk_ID")
        if {"dense", "sparse"} & row.keys():
            raise ValueError("client_vectors_forbidden")
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["source_text_sha256"]:
            raise ValueError("text_hash_mismatch")
        if identifier != canonical_sha({key: row[key] for key in IDENTITY_FIELDS}):
            raise ValueError("chunk_identity_mismatch")
        if row["profile_sha256"] != profile.fingerprint():
            raise ValueError("index_profile_drift")
        tokens = exact_token_count(bounded_text(row["text"], profile.maximum_text_bytes, "text"))
        if type(tokens) is not int or not 0 < tokens <= profile.maximum_tokens:
            raise ValueError("exact_token_limit_exceeded_or_invalid_count")
        if type(row["token_count"]) is not int or row["token_count"] != tokens:
            raise ValueError("stored_token_count_mismatch")
        if len(row["text"]) != row["span_end"] - row["span_start"]:
            raise ValueError("literal_span_length_mismatch")
        result[identifier] = canonical_sha(row)
    if complete_units:
        validate_complete_units(rows)
    return {
        "count": len(rows),
        "rows": dict(sorted(result.items())),
        "sha256": canonical_sha(result),
        "coverage_scope": "complete_units" if complete_units else "batch_only",
    }
