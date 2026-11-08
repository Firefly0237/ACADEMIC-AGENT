"""The global reviewer receives the coverage declarations users actually get."""
import copy

from citequest.agents_v3.workset_v5 import document_coverage_ledger


def test_full_native_traversal_version_and_declared_gaps_reach_global_reviewer():
    documents = [{"document": {"document_id": "D01"}, "review_status": "ai_review_validated",
        "coverage": {"status": "complete_available_text", "mapped_windows": 4, "total_windows": 4,
                     "covered_codepoints": 51512, "available_codepoints": 51512,
                     "source_version_sha256": "a" * 64},
        "summary": {"evidence_status": "supported", "gaps": ["Original PDF layout is unavailable."],
                    "claims": [{"id": "C1", "statement": "Explicit fixture proposition", "status": "supported",
                                "limitations": [], "edges": [{"quote": "explicit fixture, not raw evidence"}]}]}}]
    original = copy.deepcopy(documents)
    row = document_coverage_ledger(documents)[0]
    assert row["source_coverage"] == original[0]["coverage"]
    assert row["declared_gaps"] == original[0]["summary"]["gaps"]
    assert "edges" not in row["claims"][0]
    row["source_coverage"]["mapped_windows"] = 0
    assert documents == original


def test_unprocessed_document_cannot_acquire_claims_or_fake_coverage():
    row = document_coverage_ledger([{"document": {"document_id": "D02"}, "review_status": "not_assessed",
        "coverage": {"status": "not_processed", "mapped_windows": 0}, "summary": None}])[0]
    assert row["claims"] == [] and row["evidence_status"] == "insufficient"
    assert row["source_coverage"]["status"] == "not_processed"
