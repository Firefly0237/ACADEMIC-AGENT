"""Synthetic TEI HTTP fixtures; no service, model or database is contacted."""
import copy
import hashlib
import json
from pathlib import Path

import httpx2
import pytest

from citequest.agents_v3.catalog_scope import POLICY, current_runtime, graph_version, graph_versions
from citequest.agents_v3.contracts import AgentError
from citequest.agents_v3.product_profiles import TaskProfilePolicy
from citequest.research_v3.rerank import RerankConfig, TEICrossEncoder, validate_receipt


def configuration():
    raw = json.dumps({"version": "citequest-tei-rerank/1", "deployment_kind": "self_hosted",
        "base_url": "http://synthetic-reranker.invalid", "expected_model_id": "synthetic-cross-encoder",
        "expected_model_sha": "a" * 40, "expected_tei_version": "1.9.4"}).encode()
    return RerankConfig.model_validate_json(raw, strict=True), hashlib.sha256(raw).hexdigest()


def evidence():
    return [{"ref_id": f"synthetic:{index}", "text": f"Complete synthetic passage {index}: 数学 α = β.",
        "source_ref": {"kind": "native_representation", "id": str(index)},
        "source_sha256": str(index) * 64, "unit_text_sha256": "b" * 64}
        for index in range(3)]


def fixture_handler(config, requests, *, failure=None, headers=None):
    def handle(request):
        requests.append((request.method, request.url.path, request.content))
        if request.url.path == "/info":
            return httpx2.Response(200, json={"model_id": config.expected_model_id,
                "model_sha": None if failure == "identity" else config.expected_model_sha,
                "version": config.expected_tei_version, "model_type": {"reranker": {"id2label": {"0": "score"}}},
                "auto_truncate": False, "max_input_length": 8192, "max_client_batch_size": 32})
        assert request.method == "POST" and request.url.path == "/rerank"
        body = json.loads(request.content)
        assert body == {"query": "Synthetic question", "texts": [item["text"] for item in evidence()],
            "truncate": False, "raw_scores": True, "return_text": False}
        if failure == "http422":
            return httpx2.Response(422, json={"error": "Synthetic length rejection", "error_type": "Tokenizer"})
        if failure == "nonfinite":
            return httpx2.Response(200, content=b'[{"index":0,"score":NaN}]')
        rows = [{"index": 2, "score": 0.9}, {"index": 1, "score": -0.1}, {"index": 0, "score": 0.9}]
        if failure == "duplicate":
            rows[0]["index"] = 0
        return httpx2.Response(200, json=rows, headers=headers)
    return handle


@pytest.mark.parametrize("with_usage", [False, True])
def test_cross_encoder_preserves_every_complete_source_and_records_only_observed_usage(with_usage):
    config, sha = configuration()
    calls, scopes = [], []
    headers = {"x-compute-tokens": "91", "x-total-time": "12", "x-inference-time": "7"} if with_usage else {}
    ranker = TEICrossEncoder(config, sha, transport=httpx2.MockTransport(fixture_handler(config, calls, headers=headers)))
    original = evidence()
    output, receipt = ranker.rerank("Synthetic question", original, remaining_ms=1000,
        check_scope=lambda: scopes.append("authorized"))
    assert [item["ref_id"] for item in output] == ["synthetic:0", "synthetic:2", "synthetic:1"]
    assert {item["ref_id"]: item for item in output} == {item["ref_id"]: item for item in original}
    assert original == evidence()
    assert [(method, path) for method, path, _ in calls] == [("GET", "/info"), ("POST", "/rerank"), ("GET", "/info")]
    assert len(scopes) >= 4
    assert receipt["input_tokens"] == (91 if with_usage else None)
    assert receipt["server_inference_ms"] == (7 if with_usage else None)
    assert receipt["external_API_cash_ncny"] == 0
    result = {"evidence": output, "rerank_receipt": receipt}
    validate_receipt(result, "Synthetic question", config, sha)
    tampered = copy.deepcopy(result)
    tampered["evidence"][0]["text"] = "Changed source bytes"
    with pytest.raises(AgentError, match="rerank_MCP_receipt_contract_invalid"):
        validate_receipt(tampered, "Synthetic question", config, sha)


@pytest.mark.parametrize("failure,error,count", [
    ("identity", "rerank_service_identity_or_capacity_mismatch", 1),
    ("duplicate", "rerank_rank_permutation_invalid", 3),
    ("nonfinite", "rerank_response_json_invalid", 2),
    ("http422", "rerank_http_422", 2),
])
def test_invalid_identity_ranks_or_http_result_never_retries_or_falls_back(failure, error, count):
    config, sha = configuration()
    calls = []
    ranker = TEICrossEncoder(config, sha,
        transport=httpx2.MockTransport(fixture_handler(config, calls, failure=failure)))
    with pytest.raises(AgentError, match=error):
        ranker.rerank("Synthetic question", evidence(), remaining_ms=1000, check_scope=lambda: None)
    assert len(calls) == count


def test_reranking_is_explicit_new_task_metadata_and_old_graph_identity_stays_readable():
    source = json.loads(Path("runtime/s08/api.example.json").read_bytes())["task_profile_policy"]
    old = TaskProfilePolicy.model_validate_json(json.dumps(source), strict=True)
    fresh = copy.deepcopy(source)
    qa = next(item for item in fresh["profiles"] if item["kind"] == "qa")
    qa.update(retrieval_rerank_protocol="tei-cross-encoder1", rerank_config_sha256="a" * 64)
    new = TaskProfilePolicy.model_validate_json(json.dumps(fresh), strict=True)
    assert "retrieval_rerank_protocol" not in old.runtime_metadata("qa", "qa-v9")
    metadata = new.runtime_metadata("qa", "qa-v9")
    assert metadata["rerank_config_sha256"] == "a" * 64
    arguments = ("qa", "qa-v9", POLICY, "json1", "grounded3", "chat-json2", "chat-json2", "jsonpatch1", "references1", "isolated1")
    old_version = graph_version(*arguments)
    new_version = graph_version(*arguments, None, "tei-cross-encoder1", "a" * 64)
    assert old_version != new_version and new_version.endswith("+rerank-tei1" + "+" + "task-catalog-snapshot1")
    assert old_version in graph_versions(POLICY) and new_version in graph_versions(POLICY)
    assert current_runtime({"kind": "qa", "graph_version": new_version,
        "profile": {"profile_id": "qa-v9", "catalog_scope_policy": POLICY, **metadata}}, POLICY)
