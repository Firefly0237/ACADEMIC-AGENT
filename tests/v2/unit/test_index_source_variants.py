"""Pure typed-adapter controls; synthetic DTOs are not source admission proof."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.data_v2.native_index import NativeIndexProjection
from citequest.data_v2.schemas import IndexProjection
from citequest.index_v2.projection import compile_projection
from citequest.index_v2.rows import canonical_sha
from citequest.index_v2.source_variants import (
    CanonicalSourceUnit,
    DenialReceipt,
    DeniedSourceUnit,
    NativeReference,
    SourceRoot,
    UpdateReference,
    metadata_unit,
    native_unit,
    update_unit,
)
from citequest.index_v2.variant_compiler import compile_unit
from citequest.materials_v2.chunking import SourceChunker
from citequest.source_refs_v2 import PaperSourceReference
from citequest.updates_v2.schemas import UpdateIndexProjection

ROOT = Path(__file__).resolve().parents[3]


def context(kind="metadata_snapshot"):
    values = {
        "metadata_snapshot": ("standard_metadata", "litsearch-clean-exact/1", "litsearch_clean"),
        "native_snapshot": ("application_papers", "litsearch-s2orc-native-text/1", "litsearch_s2orc"),
        "update_seal": ("managed_metadata", "cq.protocol-metadata/1", "arxiv_managed"),
    }
    catalog, profile, namespace = values[kind]
    return SourceRoot(
        kind=kind,
        source_root_id=uuid4(),
        source_namespace=namespace,
        security_domain="public",
        representation_profile=profile,
        source_manifest_sha256="a" * 64,
        source_seal_sha256=None,
        membership_sha256=None,
        policy_sha256="b" * 64,
        evaluation_sha256="c" * 64,
        state="landing",
        profile_adopted=False,
        publication_eligible=False,
        catalog_name=catalog,
    )


def metadata(root, title="literal title𝒙😀", abstract="Exact\r\nabstract"):
    content = {
        "source_pointer_sha256": "d" * 64,
        "representation_profile": root.representation_profile,
        "title": title,
        "abstract": abstract,
        "field_provenance": {"title": "clean", "abstract": "clean"},
    }
    return IndexProjection(
        projection_id=uuid4(),
        source_row_id=uuid4(),
        snapshot_id=root.source_root_id,
        native_id="raw-id001",
        source_namespace=root.source_namespace,
        source_revision="fixed-source-revision",
        security_domain=root.security_domain,
        work_id=None,
        searchable_text_present=bool((title or "").strip() or (abstract or "").strip()),
        profile_adopted=False,
        snapshot_state="landing",
        projection_sha256=canonical_sha(content),
        **content,
    )


@pytest.fixture(scope="module")
def chunker():
    return SourceChunker(ROOT)


def test_standard_variant_has_identical_rows_to_running_v1_builder(chunker):
    root = context()
    source = metadata(root)
    with chunker.verified_batch():
        legacy = compile_projection(source, chunker)
        variant = compile_unit(metadata_unit(source, root), chunker)
    assert variant["rows"] == legacy["rows"] and variant["expected"] == legacy["expected"]
    assert variant["reference"] == {"kind": "metadata_projection", "metadata_projection_id": str(source.projection_id)}


def test_native_real_reference_type_and_original_full_unicode_spans_are_preserved(chunker):
    root = context("native_snapshot")
    text = "Native source CP coordinate.\r\n𝒙😀 " * 500
    version, work = uuid4(), uuid4()
    source = NativeIndexProjection(
        projection_id=uuid4(),
        source_row_id=uuid4(),
        snapshot_id=root.source_root_id,
        native_id="native-001",
        security_domain="public",
        source_revision="fixed-source-revision",
        representation_id="e" * 64,
        projection_sha256="e" * 64,
        source_pointer_sha256="f" * 64,
        source_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        codec_profile_sha256="a" * 64,
        literal_text=text,
        work_id=work,
        work_version_id=version,
        year=None,
        source_reference=PaperSourceReference(
            source_kind="paper", work_version_id=version, source_snapshot_sha256="a" * 64, representation_id="e" * 64
        ),
        snapshot_state="landing",
        profile_adopted=False,
    )
    unit = native_unit(source, root)
    assert isinstance(unit.source_reference, NativeReference) and unit.source_unit_id == source.projection_id
    with chunker.verified_batch():
        result = compile_unit(unit, chunker)
    assert len(result["rows"]) > 1 and result["expected"]["coverage_scope"] == "complete_units"
    assert all(
        row["text"] == text[row["span_start"] : row["span_end"]]
        and row["source_kind"] == "s2orc_native"
        and row["year"] is None
        for row in result["rows"]
    )
    assert (
        result["coordinate_system"] == "unicode_codepoints_in_native_literal"
        and result["publication_performed"] is False
    )


def test_observed_metadata_keeps_false_publication_and_never_becomes_standard_or_paper(chunker):
    root = context("update_seal")
    identifier = uuid4()
    text = "Observed title\nObserved abstract"
    unit = CanonicalSourceUnit(
        source_unit_id=identifier,
        root=root,
        source_reference=UpdateReference(update_observation_id=identifier),
        source_kind="update_title_abstract",
        native_id="1706.03762",
        source_revision="v1",
        projection_sha256="d" * 64,
        source_pointer_sha256="e" * 64,
        unit_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        literal_text=text,
        searchable_text_present=True,
        year=None,
        work_id=None,
        work_version_id=None,
        coordinate_system="unicode_codepoints_in_metadata_recipe",
        builder_version="observed-title-newline-abstract/1",
        original_source_dto={"fixture": "synthetic observation"},
    )
    with chunker.verified_batch():
        result = compile_unit(unit, chunker)
    assert result["rows"][0]["source_kind"] == "update_title_abstract" and unit.root.publication_eligible is False
    assert unit.work_id is None and unit.work_version_id is None and result["index_staging_is_not_QA_permission"]
    with pytest.raises(ValidationError):
        CanonicalSourceUnit.model_validate({**unit.model_dump(), "source_kind": "s2orc_native"})


def test_empty_source_and_policy_denial_are_distinct_and_preserve_original_record(chunker):
    root = context()
    empty = metadata_unit(metadata(root, None, None), root)
    with chunker.verified_batch():
        result = compile_unit(empty, chunker)
    assert result["rows"] == [] and result["omitted_record"] is False and empty.original_source_dto["title"] is None
    nonempty = metadata_unit(metadata(root), root)
    denied = CanonicalSourceUnit.model_validate({**nonempty.model_dump(), "source_denied": True})
    with chunker.verified_batch():
        result = compile_unit(denied, chunker)
    assert result["indexability"] == "current_source_policy_denied" and result["rows"] == []
    assert denied.searchable_text_present and denied.literal_text == nonempty.literal_text


@pytest.mark.parametrize(
    "change",
    [
        {"publication_eligible": True},
        {"catalog_name": "application_papers"},
        {"representation_profile": "litsearch-s2orc-native-text/1"},
        {"source_seal_sha256": "not-sha"},
    ],
)
def test_root_cannot_self_grant_wrong_profile_or_unsealed_publication(change):
    root = context()
    with pytest.raises(ValidationError):
        SourceRoot.model_validate({**root.model_dump(), **change})




def test_denied_member_never_invents_empty_or_verified_original_body(chunker):
    root = context("native_snapshot")
    denial = DenialReceipt(
        source_namespace=root.source_namespace,
        native_id="same-source",
        source_revision="*",
        reason="withdrawn",
        scope="this_source_all_versions",
        observation_id=uuid4(),
        blocks_current_default_evidence=True,
    )
    unit = DeniedSourceUnit(
        source_unit_id=uuid4(),
        root=root,
        source_reference=NativeReference(native_representation_id="e" * 64),
        source_kind="s2orc_native",
        native_id="same-source",
        source_revision="v1",
        projection_sha256="e" * 64,
        source_pointer_sha256="f" * 64,
        source_denials=[denial],
    )
    with chunker.verified_batch():
        result = compile_unit(unit, chunker)
    assert result["rows"] == [] and result["unit_text_sha256"] is None
    assert result["source_text_withheld"] and result["original_empty_text_not_claimed"]
    assert not hasattr(unit, "literal_text")
    with pytest.raises(ValidationError):
        DeniedSourceUnit.model_validate({**unit.model_dump(), "native_id": "other-paper"})
    with pytest.raises(ValidationError):
        DeniedSourceUnit.model_validate({**unit.model_dump(), "source_denials": []})
