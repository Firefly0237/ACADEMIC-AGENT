"""No raw audit/path exposure, and original asset bytes remain unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from citequest.materials_v2.artifact_public import ensure_public_content
from citequest.v2.domain import DomainError


@pytest.mark.parametrize(
    "value",
    [
        {"origin": {"filename": "C:\\private\\source.pdf"}},
        {"origin": {"filename": "/home/private/source.pdf"}},
        {"options": {"root": "\\\\server\\share"}},
        {"text": "example local install " + str(Path(__file__).resolve().parents[3])},
    ],
)
def test_actual_local_or_metadata_absolute_paths_are_not_public(value):
    with pytest.raises(DomainError):
        ensure_public_content("native_document", json.dumps(value).encode(), root=Path(__file__).resolve().parents[3])


def test_scientific_path_example_and_self_refs_are_preserved_not_rewritten():
    original = json.dumps(
        {
            "origin": {"filename": "opaque-sha256"},
            "self_ref": "#/texts/0",
            "text": "Use /example/path for the synthetic experiment.",
        }
    ).encode()
    ensure_public_content("native_document", original, root=Path(__file__).resolve().parents[3])
    assert b"/example/path" in original


def test_internal_audit_content_is_not_a_public_role():
    with pytest.raises(DomainError, match="artifact_internal_audit_only"):
        ensure_public_content("parser_resource_receipt", b"{}", root=Path(__file__).resolve().parents[3])


def test_duplicate_metadata_key_cannot_hide_an_original_private_path():
    with pytest.raises(DomainError, match="artifact_invalid_json"):
        ensure_public_content(
            "native_document",
            b'{"filename":"C:/private/source.pdf","filename":"safe.pdf"}',
            root=Path(__file__).resolve().parents[3],
        )
