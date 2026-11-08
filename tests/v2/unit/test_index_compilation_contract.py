"""Pure source/complete-unit contract controls; no SQL or native service call."""

from __future__ import annotations

import copy
from pathlib import Path
from uuid import uuid4

import pytest

from citequest.data_v2.schemas import IndexProjection
from citequest.index_v2.projection import compile_projection
from citequest.index_v2.repository import compiled_contract
from citequest.index_v2.rows import canonical_sha
from citequest.materials_v2.chunking import SourceChunker
from citequest.v2.domain import DomainError

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def prepared():
    content = {
        "source_pointer_sha256": "a" * 64,
        "representation_profile": "litsearch-clean-exact/1",
        "title": "Real literal title 𝒙😀",
        "abstract": "Exact\r\nsource abstract.",
        "field_provenance": {"title": "clean", "abstract": "clean"},
    }
    projection = IndexProjection(
        projection_id=uuid4(),
        source_row_id=uuid4(),
        snapshot_id=uuid4(),
        native_id="synthetic-raw-001",
        source_namespace="litsearch_clean",
        source_revision="9573fb284a1026c998df47024b888a163f0f0e25",
        security_domain="public",
        work_id=None,
        searchable_text_present=True,
        profile_adopted=False,
        snapshot_state="landing",
        projection_sha256=canonical_sha(content),
        **content,
    )
    chunker = SourceChunker(ROOT)
    with chunker.verified_batch():
        compiled = compile_projection(projection, chunker)
    return projection, compiled


def test_complete_literal_compilation_is_accepted_only_after_verified_context(prepared):
    projection, compiled = prepared
    compiled_contract(projection, compiled)
    assert compiled["expected"]["coverage_scope"] == "complete_units" and projection.work_id is None


@pytest.mark.parametrize(
    "kind",
    [
        "batch_only",
        "omitted_tail",
        "namespace",
        "native",
        "source_kind",
        "tokens_bool",
        "unit_hash",
        "year0",
        "vector",
        "profile",
        "pointer",
    ],
)
def test_coordinated_compilation_tamper_cannot_pass_source_contract(prepared, kind):
    projection, original = prepared
    compiled = copy.deepcopy(original)
    row = compiled["rows"][0]
    if kind == "batch_only":
        compiled["expected"]["coverage_scope"] = "batch_only"
    elif kind == "omitted_tail":
        row["span_end"] -= 1
    elif kind == "namespace":
        row["source_namespace"] = "different_source"
    elif kind == "native":
        row["native_id"] = "another_id"
    elif kind == "source_kind":
        row["source_kind"] = "material_native"
    elif kind == "tokens_bool":
        row["token_count"] = True
    elif kind == "unit_hash":
        row["unit_text_sha256"] = "b" * 64
    elif kind == "year0":
        row["year"] = 0
    elif kind == "vector":
        row["dense"] = [0.0] * 1024
    elif kind == "profile":
        row["profile_sha256"] = "c" * 64
    elif kind == "pointer":
        row["source_pointer_sha256"] = "d" * 64
    # Recompute the outer digest: rejection must include source/structure checks,
    # not merely rely on an accidentally stale batch hash.
    compiled["expected"]["rows"][row["chunk_id"]] = canonical_sha(row)
    compiled["expected"]["sha256"] = canonical_sha(compiled["expected"]["rows"])
    with pytest.raises((DomainError, ValueError)):
        compiled_contract(projection, compiled)
