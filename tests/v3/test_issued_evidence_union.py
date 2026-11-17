"""Three pure controls for a stable union of explicitly synthetic issued DTOs."""
import copy

import pytest

from citequest.agents_v3.chunk_tool_policy import deduplicate_issued_evidence
from citequest.agents_v3.contracts import AgentError, digest


def source():
    text = "Synthetic table\nA .18\nA+X .20\nΔ=.02; condition α😀."
    return {"ref_id": "rag:synthetic-retrieval-1:chunk", "snapshot_id": "synthetic-retrieval-1",
        "owner_id": "synthetic-owner", "source_ref": {"kind": "native_representation", "id": "a" * 64},
        "source": {"source_namespace": "synthetic", "source_revision": "original"},
        "generation_id": "synthetic-generation", "epoch": 1, "text": text,
        "source_sha256": digest(text.encode()), "unit_text_sha256": "b" * 64,
        "span_start": 100, "span_end": 100 + len(text), "coordinate_system": "unicode_codepoints_in_native_literal",
        "work_version_id": "synthetic-work-version", "rights_status": "open",
        "outbound_policy": "selected_cloud_processing_allowed", "locator": {"kind": "synthetic", "block": 3}}


def test_stable_union_preserves_complete_first_DTO_and_literal_table_formula_coordinates():
    first = source()
    repeated = {**copy.deepcopy(first), "ref_id": "rag:synthetic-retrieval-2:chunk", "snapshot_id": "synthetic-retrieval-2"}
    original = copy.deepcopy([first, repeated])
    retained, receipt = deduplicate_issued_evidence([first, repeated])
    assert len(retained) == 1 and retained[0] is first
    assert [first, repeated] == original
    assert retained[0]["text"] == original[0]["text"]
    assert retained[0]["span_end"] - retained[0]["span_start"] == len(retained[0]["text"])
    assert receipt["duplicate_aliases"] == [{"duplicate_ref_id": repeated["ref_id"], "retained_ref_id": first["ref_id"]}]
    assert receipt["input_characters"] == 2 * receipt["retained_characters"]
    assert receipt["identity_excludes_only"] == ["ref_id", "snapshot_id"]
    assert receipt["permission_grant"] is False and receipt["source_bytes_modified"] is False


def test_same_text_from_different_authority_versions_generations_coordinates_or_new_metadata_is_not_merged():
    first = source()
    changes = [
        {"owner_id": "another-owner"},
        {"source_ref": {"kind": "native_representation", "id": "c" * 64}},
        {"source": {"source_namespace": "synthetic", "source_revision": "updated"}},
        {"unit_text_sha256": "d" * 64}, {"work_version_id": "another-work-version"},
        {"generation_id": "another-generation"}, {"epoch": 2},
        {"span_start": 200, "span_end": 200 + len(first["text"])},
        {"coordinate_system": "unicode_codepoints_in_material_fragment"},
        {"rights_status": "restricted"}, {"outbound_policy": "local_only"},
        {"locator": {"kind": "synthetic", "block": 4}}, {"future_authority_field": "must_remain_distinct"},
    ]
    for change in changes:
        other = {**copy.deepcopy(first), "ref_id": "rag:other:chunk", "snapshot_id": "other", **change}
        retained, receipt = deduplicate_issued_evidence([first, other])
        assert retained == [first, other], change
        assert not receipt["duplicate_aliases"], change


def test_changed_payload_under_one_issued_ref_is_rejected_and_empty_evidence_remains_empty():
    first = source()
    with pytest.raises(AgentError, match="reference_changed_during_union"):
        deduplicate_issued_evidence([first, {**first, "epoch": 2}])
    retained, receipt = deduplicate_issued_evidence([])
    assert retained == [] and receipt["input_ref_ids"] == receipt["retained_ref_ids"] == []
    assert receipt["input_characters"] == receipt["retained_characters"] == 0
