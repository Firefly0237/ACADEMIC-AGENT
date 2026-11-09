"""Citation hydration keeps source verification without loading display metadata."""
import hashlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from citequest.search_v3.service import SearchService
from citequest.v2.domain import Actor, DomainError


@pytest.mark.parametrize("kind", ["native_representation", "metadata_projection"])
def test_literal_evidence_uses_pinned_identity_without_display_queries(kind):
    literal = "前言：实验结果为 α=0.9。"
    excerpt = literal[3:]
    def checksum(value):
        return hashlib.sha256(value.encode()).hexdigest()

    current = SimpleNamespace(literal_text=literal, unit_text_sha256=checksum(literal),
        coordinate_system="unicode_codepoints_in_native_literal" if kind == "native_representation"
        else "unicode_codepoints_in_metadata_recipe", work_id=uuid4(), work_version_id=uuid4())
    unit = SimpleNamespace(source_reference_kind=kind, native_representation_id="a" * 64,
        metadata_projection_id=uuid4(), source_namespace="explicit_fixture", native_id="fixture-paper",
        source_revision="v1", projection_sha256="b" * 64, source_pointer_sha256="c" * 64)
    payload = {"unit_text_sha256": current.unit_text_sha256, "span_start": 3, "span_end": len(literal),
        "text": excerpt, "source_text_sha256": checksum(excerpt)}
    source_reads = []

    def read(root, reference):
        source_reads.append((root, reference))
        return current

    def unexpected_query(*_args, **_kwargs):
        pytest.fail("citation identity must not load paper display metadata")

    service = object.__new__(SearchService)
    service._repo = lambda *_args: SimpleNamespace(readers=SimpleNamespace(root=lambda _: "fixture-root", read=read))
    service._summaries = unexpected_query
    db = SimpleNamespace(execute=unexpected_query, scalars=unexpected_query)
    generation, snapshot_id = SimpleNamespace(id=uuid4()), uuid4()
    arguments = (db, Actor(uuid4()), generation, SimpleNamespace(payload=payload), unit)
    kwargs = {"snapshot_id": snapshot_id, "epoch": 1, "ref_id": "fixture:chunk"}

    result = service._literal_evidence(*arguments, **kwargs)

    assert result.source_ref.model_dump() == {"kind": kind,
        "id": unit.native_representation_id if kind == "native_representation" else str(unit.metadata_projection_id)}
    assert result.source.model_dump() == {key: getattr(unit, key) for key in
        ("source_namespace", "native_id", "source_revision", "projection_sha256", "source_pointer_sha256")}
    assert result.coverage == ("native_literal" if kind == "native_representation" else "metadata")
    assert (result.work_id, result.work_version_id) == (current.work_id, current.work_version_id)
    assert (result.text, result.span_start, result.span_end) == (excerpt, 3, len(literal))
    assert result.source_sha256 == checksum(excerpt) and result.unit_text_sha256 == checksum(literal)
    assert result.snapshot_id == snapshot_id and result.generation_id == generation.id and result.epoch == 1
    assert len(source_reads) == 1

    payload["source_text_sha256"] = "0" * 64
    with pytest.raises(DomainError, match="evidence_source_hash_or_span_mismatch"):
        service._literal_evidence(*arguments, **kwargs)
    assert len(source_reads) == 2
