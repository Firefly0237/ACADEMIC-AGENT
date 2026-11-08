"""Lossless S2ORC character anchors and observable audit signals, not PDF claims.

This module uses only the Python standard library so the fixed, official Arrow
reader can run it without importing a model or the product service runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

PROFILE = "litsearch-s2orc-native-text-audit/1"
OFFSET_PROFILE = "litsearch-native-annotation-decimal-offsets/1"


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_annotation_key")
        result[key] = value
    return result


def offset(value: Any, length: int) -> tuple[int, bool]:
    if type(value) is int:
        result, converted = value, False
    elif isinstance(value, str) and 1 <= len(value) <= 12 and re.fullmatch(r"[0-9]+", value):
        result, converted = int(value), True
    else:
        raise ValueError("offset_requires_int_or_bounded_ascii_decimal")
    if not 0 <= result <= length:
        raise ValueError("offset_out_of_bounds")
    return result, converted


@dataclass(frozen=True)
class Anchor:
    source_text_sha256: str
    character_start: int
    character_end: int
    literal_sha256: str
    coordinate_system: str = "unicode_codepoint"

    def validate(self, text: str) -> None:
        validate_anchors(text, [self])


def validate_anchors(text: str, anchors: list[Anchor]) -> None:
    source_sha = hashlib.sha256(text.encode()).hexdigest()
    for anchor in anchors:
        if (
            anchor.coordinate_system != "unicode_codepoint"
            or source_sha != anchor.source_text_sha256
            or not 0 <= anchor.character_start < anchor.character_end <= len(text)
            or hashlib.sha256(text[anchor.character_start : anchor.character_end].encode()).hexdigest()
            != anchor.literal_sha256
        ):
            raise ValueError("anchor_not_recoverable")


def annotation_spans(raw: Any, text: str) -> tuple[list[tuple[int, int]], int]:
    if raw is None:
        return [], 0
    if not isinstance(raw, str) or len(raw.encode()) > 2 * 1024 * 1024:
        raise ValueError("annotation_not_bounded_json_string")
    values = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(values, list):
        raise ValueError("annotation_list_required")
    spans, converted = [], 0
    for entry in values:
        if not isinstance(entry, dict):
            raise ValueError("annotation_object_required")
        start, a = offset(entry.get("start"), len(text))
        end, b = offset(entry.get("end"), len(text))
        if start >= end or not text[start:end].strip():
            raise ValueError("empty_or_reversed_annotation")
        spans.append((start, end))
        converted += int(a or b)
    return spans, converted


def audit_text(
    text: Any, annotations: Any, *, minimum_fulltext_characters: int = 2000, maximum_characters: int = 200000
) -> dict[str, Any]:
    """Report only observable source conditions; licence/completeness need proof.

    A long string is never labelled a qualified full paper by this function.
    Every positive anchor is checked against the entire unchanged native text.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "profile": PROFILE,
            "technical_status": "rejected",
            "reasons": ["native_text_missing"],
            "eligible": False,
        }
    sha = hashlib.sha256(text.encode()).hexdigest()
    reasons = []
    if len(text) > maximum_characters or len(text.encode()) > 2 * 1024 * 1024:
        reasons.append("source_text_exceeds_preparation_profile_caps")
    if "\x00" in text or "\ufffd" in text:
        reasons.append("source_text_corruption_marker")
    if len(text.strip()) < minimum_fulltext_characters:
        reasons.append("not_enough_body_for_fulltext_candidate")
    values = annotations if isinstance(annotations, dict) else {}
    counts, errors, converted, anchors = {}, {}, 0, []
    for name in ("title", "abstract", "paragraph", "sectionheader", "bibentry"):
        try:
            spans, conversions = annotation_spans(values.get(name), text)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            errors[name] = str(exc)
            counts[name] = 0
            continue
        counts[name] = len(spans)
        converted += conversions
        if name == "paragraph":
            for start, end in spans:
                anchor = Anchor(sha, start, end, hashlib.sha256(text[start:end].encode()).hexdigest())
                anchors.append(asdict(anchor))
    validate_anchors(text, [Anchor(**value) for value in anchors])
    if errors:
        reasons.append("malformed_native_annotations")
    if counts.get("paragraph", 0) < 5 or counts.get("sectionheader", 0) < 2:
        reasons.append("structured_body_not_demonstrated")
    # A licence URL anywhere in a paper may describe a dataset/other work.
    # Preserve the signal; do not turn it into a rights decision.
    license_signals = []
    for match in re.finditer(r"https?://creativecommons\.org/(?:licenses|publicdomain)/[^\s)<>]+", text, re.I):
        license_signals.append(
            {
                "start": match.start(),
                "end": match.end(),
                "literal": match.group(),
                "context_start": max(0, match.start() - 160),
                "context_end": min(len(text), match.end() + 160),
            }
        )
    return {
        "profile": PROFILE,
        "technical_status": "rejected" if reasons else "source_fulltext_candidate",
        "source_text_sha256": sha,
        "characters": len(text),
        "utf8_bytes": len(text.encode()),
        "native_annotation_counts": counts,
        "annotation_errors": errors,
        "normalized_offset_entries": converted,
        "offset_profile": OFFSET_PROFILE,
        "anchors": anchors,
        "license_signals": license_signals,
        "reasons": reasons,
        "eligible": False if reasons else None,
        "remaining_gates": [
            "per_article_rights_evidence",
            "complete_native_article_confirmation",
            "frozen_permission_metadata",
        ],
        "pdf_page_numbers_claimed": False,
        "original_pdf_completeness_claimed": False,
    }


def order_sha(seed: str, namespace: str, native_id: str) -> str:
    if not seed or not namespace or not native_id:
        raise ValueError("nonempty_order_identity_required")
    return hashlib.sha256((seed + "\n" + namespace + ":" + native_id).encode()).hexdigest()
