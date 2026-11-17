"""S05 preparation controls: strict source preservation and real local-byte verification, no PG claims."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from citequest.data_v2 import models
from citequest.data_v2.landing import Landing, verify_artifacts
from citequest.data_v2.native_text import Anchor, annotation_spans, audit_text, order_sha
from citequest.data_v2.schemas import (
    BatchInput,
    IndexProjection,
    ProjectionInput,
    RawRowLocator,
    SnapshotInput,
    SourceArtifact,
    unresolved_reasons,
)
from citequest.v2.domain import DomainError
from citequest.v2.models import Base
from citequest.v2.services import digest


def artifact(**changes):
    return SourceArtifact.model_validate(
        {
            "path": "raw/shard.parquet",
            "sha256": "a" * 64,
            "size_bytes": 20,
            "format": "parquet",
            "row_group_sizes": [2, 3],
            **changes,
        }
    )


@pytest.mark.parametrize("path", ["/abs", "../escape", "a/../b", "a\\b", "a//b", "a/./b", "C:/secret", "a/"])
def test_source_paths_are_explicit_relative_logical_names(path):
    with pytest.raises(ValidationError):
        artifact(path=path)


def test_parquet_groups_are_truthful_and_size_types_strict():
    for change in (
        {"row_group_sizes": None},
        {"row_group_sizes": [0]},
        {"row_group_sizes": [True]},
        {"size_bytes": True},
        {"format": "jsonl"},
    ):
        with pytest.raises(ValidationError):
            artifact(**change)
    assert artifact(format="jsonl", row_group_sizes=None).format == "jsonl"


def snapshot(**changes):
    return SnapshotInput.model_validate(
        {
            "source_namespace": "litsearch_s2orc",
            "source_revision": "rev",
            "source_manifest_sha256": "b" * 64,
            "policy_sha256": "c" * 64,
            "evaluation_sha256": "d" * 64,
            "representation_profile": "litsearch-clean-exact/1",
            "purpose": "standard",
            "artifacts": [artifact().model_dump()],
            "client_operation_id": uuid4(),
            **changes,
        }
    )


def test_snapshot_separates_fulltext_and_standard_profiles_and_server_identity():
    with pytest.raises(ValidationError):
        snapshot(purpose="application")
    with pytest.raises(ValidationError):
        snapshot(owner_id=str(uuid4()))
    with pytest.raises(ValidationError):
        snapshot(artifacts=[artifact().model_dump(), artifact().model_dump()])
    assert snapshot(representation_profile="litsearch-native-annotation-decimal-offsets/1").purpose == "standard"
    with pytest.raises(ValidationError):
        snapshot(representation_profile="litsearch-s2orc-native-text/1")
    assert (
        snapshot(purpose="application", representation_profile="litsearch-s2orc-native-text/1").purpose == "application"
    )


def test_metadata_ingest_cannot_label_title_abstract_as_application_fulltext():
    fake_landing = SimpleNamespace(
        snapshot=lambda *args, **kwargs: SimpleNamespace(representation_profile="litsearch-s2orc-native-text/1")
    )
    with pytest.raises(DomainError, match="native_fulltext_requires_native_projection_contract"):
        Landing.ingest_batch(fake_landing, uuid4(), BatchInput(rows=[projection()], client_operation_id=uuid4()))


def projection(**changes):
    return ProjectionInput.model_validate(
        {
            "native_id": "000001",
            "native_id_source_type": "string",
            "locator": RawRowLocator(
                artifact_path="raw/shard.parquet", artifact_sha256="a" * 64, row_group=0, row_in_group=0, row_ordinal=0
            ),
            "title": None,
            "abstract": "",
            "field_provenance": {},
            **changes,
        }
    )


def test_source_missing_whitespace_long_title_and_raw_native_identity_are_not_rewritten():
    value = projection(title="  " + "长" * 8560 + "  ", abstract=None)
    assert value.native_id == "000001" and len(value.title) == 8564 and value.title.startswith("  ")
    assert unresolved_reasons(None, "known abstract") == ["source_title_missing"]
    assert unresolved_reasons(" ", "") == ["source_title_missing", "title_and_abstract_empty"]
    assert "requires_source_import_profile" in unresolved_reasons(value.title, None)[0]


def test_batch_has_real_bounded_nonduplicate_inputs():
    with pytest.raises(ValidationError):
        BatchInput(rows=[projection(), projection()], client_operation_id=uuid4())
    with pytest.raises(ValidationError):
        BatchInput(rows=[], client_operation_id=uuid4())
    with pytest.raises(ValidationError):
        BatchInput(rows=[projection(native_id=str(i)) for i in range(501)], client_operation_id=uuid4())


def test_integer_native_ids_and_provenance_do_not_accept_false_source_types():
    for spelling in ("0001", " 1", "+1", "١", "one"):
        with pytest.raises(ValidationError, match="canonical decimal"):
            projection(native_id=spelling, native_id_source_type="integer")
    assert projection(native_id="1", native_id_source_type="integer").native_id == "1"
    for value in (float("nan"), object()):
        with pytest.raises(ValidationError, match="standard JSON"):
            projection(field_provenance={"value": value})


def test_verified_source_bytes_must_match_bound_hash_and_length(tmp_path: Path):
    path = tmp_path / "raw/shard.parquet"
    path.parent.mkdir()
    data = b"explicit synthetic artifact bytes; not a valid Parquet fixture"
    path.write_bytes(data)
    request = snapshot(artifacts=[artifact(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)).model_dump()])
    verify_artifacts(tmp_path, request)
    path.write_bytes(data + b"changed")
    with pytest.raises(DomainError, match="integrity_failed"):
        verify_artifacts(tmp_path, request)


@pytest.mark.parametrize("bad", [True, 1.0, " 1", "+1", "-1", "١", "１", "1e1", "9" * 13])
def test_native_offsets_do_not_coerce_ambiguous_values(bad):
    with pytest.raises(ValueError):
        annotation_spans(json.dumps([{"start": bad, "end": 2}]), "abcd")


@pytest.mark.parametrize(
    "raw", ['[{"start":0,"start":1,"end":2}]', '[{"start":3,"end":2}]', '[{"start":0,"end":99}]', "{}", "[null]"]
)
def test_invalid_native_annotation_does_not_gain_a_locator(raw):
    with pytest.raises(ValueError):
        annotation_spans(raw, "abcd")


def test_unicode_native_anchor_is_literal_and_never_a_fabricated_pdf_page():
    text = "开头😀中文及formula x²"
    start, end = 2, 5
    spans, conversions = annotation_spans(json.dumps([{"start": "02", "end": str(end)}]), text)
    assert spans == [(start, end)] and conversions == 1
    anchor = Anchor(
        hashlib.sha256(text.encode()).hexdigest(), start, end, hashlib.sha256("😀中文".encode()).hexdigest()
    )
    anchor.validate(text)
    with pytest.raises(ValueError, match="recoverable"):
        anchor.validate("changed" + text)
    with pytest.raises(TypeError):
        Anchor(**{**anchor.__dict__, "pdf_page": 1})
    with pytest.raises(ValueError, match="recoverable"):
        Anchor(**{**anchor.__dict__, "coordinate_system": "utf16"}).validate(text)


def test_long_structured_source_is_only_a_candidate_not_proven_rights_or_complete_pdf():
    parts = [f"Section{i} " + "word " * 100 for i in range(6)]
    text = "\n".join(parts) + "\nDataset license: https://creativecommons.org/licenses/by/4.0/"
    positions, cursor = [], 0
    for part in parts:
        positions.append({"start": cursor, "end": cursor + len(part)})
        cursor += len(part) + 1
    annotations = {"paragraph": json.dumps(positions), "sectionheader": json.dumps(positions[:2])}
    result = audit_text(text, annotations)
    assert result["technical_status"] == "source_fulltext_candidate" and result["eligible"] is None
    assert len(result["anchors"]) == 6 and result["license_signals"]
    assert result["original_pdf_completeness_claimed"] is False and result["pdf_page_numbers_claimed"] is False
    assert "per_article_rights_evidence" in result["remaining_gates"]
    assert audit_text("only an abstract", {})["technical_status"] == "rejected"
    assert audit_text(text, {"paragraph": '[{"start":false,"end":1}]'})["eligible"] is False


def test_frozen_candidate_order_known_vector_and_same_registry():
    assert (
        order_sha("citequest-v2-s02-corpus-20261104", "litsearch_s2orc", "253761972")
        == "000175df55a759a3a435db7c69735727a00783205fdb9a8e6383e0c46856ba54"
    )
    assert models.SourceRow.metadata is Base.metadata
    for table in (
        models.SourceSnapshot.__table__,
        models.SourceArtifact.__table__,
        models.SourceRow.__table__,
        models.SourceProjection.__table__,
    ):
        assert "cq_app." in str(CreateTable(table).compile(dialect=postgresql.dialect()))
    assert models.SourceProjection.__table__.c.title.nullable and models.SourceResolution.__table__.c.work_id.nullable
    assert "work_id" not in models.SourceProjection.__table__.c


def test_index_projection_has_nullable_work_year_and_preserves_exact_source():
    profile = "litsearch-clean-exact/1"
    retrieval = {
        "source_pointer_sha256": "a" * 64,
        "representation_profile": profile,
        "title": None,
        "abstract": "  original abstract\n",
        "field_provenance": {"abstract": "raw"},
    }
    value = IndexProjection(
        projection_id=uuid4(),
        source_row_id=uuid4(),
        snapshot_id=uuid4(),
        native_id="00001",
        source_namespace="litsearch",
        source_revision="pinned",
        security_domain="public",
        projection_sha256=digest(retrieval),
        **retrieval,
        work_id=None,
        searchable_text_present=True,
        profile_adopted=False,
        snapshot_state="landing",
    )
    assert value.work_id is None and value.year is None and value.native_id == "00001"
    assert value.title is None and value.abstract == "  original abstract\n"
    with pytest.raises(ValidationError, match="unchanged projection"):
        IndexProjection.model_validate({**value.model_dump(), "abstract": "changed"})
    with pytest.raises(ValidationError, match="presence flag"):
        IndexProjection.model_validate({**value.model_dump(), "searchable_text_present": False})
    with pytest.raises(ValidationError):
        IndexProjection.model_validate({**value.model_dump(), "year": 0})
    with pytest.raises(ValidationError):
        IndexProjection.model_validate(
            {**value.model_dump(), "representation_profile": "litsearch-s2orc-native-text/1"}
        )
