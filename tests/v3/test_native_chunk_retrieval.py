"""Protocol-only: real PyMilvus request objects, explicit fake RPC/proof hook.

No SQL, server, vector, source, model or retrieval-quality acceptance claim.
"""
import math

import pytest

from citequest.search_v3.native import SearchNative
from citequest.search_v3.schemas import SearchRequest
from citequest.v2.domain import DomainError


def native(rows):
    result = SearchNative(uri="http://127.0.0.1:19530", tei_info_url="http://127.0.0.1:8080/info")
    calls, proofs = [], []
    result.verify = lambda *args, **kwargs: proofs.append((args, kwargs))

    def call(method, **kwargs):
        calls.append((method, kwargs))
        return [rows]

    result.call = call
    return result, calls, proofs


def row(chunk="a", paper="1", score=0.8):
    return {"id": chunk * 64, "entity": {"chunk_id": chunk * 64, "row_key": paper * 64}, "distance": score}


def args(mode="hybrid"):
    return {"collection": "cq_v2_source_" + "a" * 32, "request": SearchRequest(q="FT table results", mode=mode, candidate_limit=8, page_size=8),
            "security_domain": "public", "excluded_row_keys": [], "included_row_keys": ["1" * 64], "expected_proof": {"fixture": True}}


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_paper_path_keeps_original_grouping_and_chunk_path_is_ungrouped(mode):
    gateway, calls, proofs = native([row()])
    assert len(gateway.search(**args(mode))) == 1
    assert calls[0][1]["group_by_field"] == "row_key" and calls[0][1]["group_size"] == 1
    assert len(gateway.search_chunks(**args(mode))) == 1
    assert all(key not in calls[1][1] for key in ("group_by_field", "group_size", "strict_group_size"))
    assert calls[0][0] == calls[1][0] == ("hybrid_search" if mode == "hybrid" else "search")
    assert len(proofs) == 2 and calls[0][1]["consistency_level"] == calls[1][1]["consistency_level"] == "Strong"


def test_two_chunks_of_same_paper_are_delivered_but_paper_path_refuses_duplicate_groups():
    gateway, _, _ = native([row(), row("b")])
    assert [hit.chunk_id for hit in gateway.search_chunks(**args())] == ["a" * 64, "b" * 64]
    with pytest.raises(DomainError, match="native_result_identity_or_grouping_invalid"):
        gateway.search(**args())


@pytest.mark.parametrize("rows", [[row(), row()], [row(score=math.nan)], [row(score=math.inf)]])
def test_chunks_keep_unique_identity_and_finite_score_guards(rows):
    gateway, _, _ = native(rows)
    with pytest.raises(DomainError, match="native_result_identity_or_grouping_invalid"):
        gateway.search_chunks(**args())


def test_non_text_or_non_relevance_rejected_before_any_verification_or_RPC():
    gateway, calls, proofs = native([])
    for update in ({"field": "author"}, {"sort": "title"}):
        parameters = args()
        parameters["request"] = parameters["request"].model_copy(update=update)
        with pytest.raises(DomainError, match="chunk_evidence_requires_text_relevance"):
            gateway.search_chunks(**parameters)
    assert not calls and not proofs
