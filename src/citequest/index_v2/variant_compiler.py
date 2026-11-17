"""One lossless row compiler for explicitly typed independent source families."""

from __future__ import annotations

from citequest.materials_v2.chunking import SourceChunker

from .projection import metadata_ranges
from .rows_v2 import compile_row, expected_manifest
from .source_variants import CanonicalSourceUnit, DeniedSourceUnit


def compile_unit(unit: CanonicalSourceUnit | DeniedSourceUnit, chunker: SourceChunker) -> dict:
    if unit.root.state == "disabled":
        raise ValueError("disabled_source_root")
    if isinstance(unit, DeniedSourceUnit):
        return {
            "source_unit_id": str(unit.source_unit_id),
            "root_kind": unit.root.kind,
            "source_root_id": str(unit.root.source_root_id),
            "reference": unit.source_reference.model_dump(mode="json"),
            "projection_sha256": unit.projection_sha256,
            "source_pointer_sha256": unit.source_pointer_sha256,
            "unit_text_sha256": None,
            "tokenizer_profile_sha256": chunker.profile_sha256,
            "indexability": "current_source_policy_denied",
            "source_text_withheld": True,
            "original_empty_text_not_claimed": True,
            "rows": [],
            "expected": None,
            "omitted_record": False,
            "publication_performed": False,
            "index_staging_is_not_QA_permission": True,
        }
    identity = {
        "source_unit_id": str(unit.source_unit_id),
        "root_kind": unit.root.kind,
        "source_root_id": str(unit.root.source_root_id),
        "reference": unit.source_reference.model_dump(mode="json"),
        "projection_sha256": unit.projection_sha256,
        "unit_text_sha256": unit.unit_text_sha256,
        "source_pointer_sha256": unit.source_pointer_sha256,
        "tokenizer_profile_sha256": chunker.profile_sha256,
        "builder_version": unit.builder_version,
        "coordinate_system": unit.coordinate_system,
        "index_staging_is_not_QA_permission": True,
        "publication_performed": False,
    }
    if unit.source_denied:
        return {
            **identity,
            "indexability": "current_source_policy_denied",
            "rows": [],
            "expected": None,
            "omitted_record": False,
        }
    if not unit.searchable_text_present:
        return {
            **identity,
            "indexability": "empty_source_representation",
            "rows": [],
            "expected": None,
            "omitted_record": False,
        }
    spans = metadata_ranges(unit.literal_text, chunker)
    rows = [
        compile_row(
            snapshot_id=unit.root.source_root_id,
            projection_id=unit.source_unit_id,
            native_id=unit.native_id,
            security_domain=unit.root.security_domain,
            source_namespace=unit.root.source_namespace,
            source_kind=unit.source_kind,
            projection_sha256=unit.projection_sha256,
            source_pointer_sha256=unit.source_pointer_sha256,
            tokenizer_profile_sha256=chunker.profile_sha256,
            full_unit_text=unit.literal_text,
            start=span["char_start"],
            end=span["char_end"],
            ordinal=ordinal,
            exact_token_count=chunker.count,
            year=unit.year,
        )
        for ordinal, span in enumerate(spans, 1)
    ]
    return {
        **identity,
        "indexability": "compiled_literal_source",
        "rows": rows,
        "expected": expected_manifest(rows, exact_token_count=chunker.count),
    }
