"""Pure protocol-admission fixtures; no model, service or database is contacted."""
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from citequest.agents_v3.catalog_scope import POLICY, SUFFIX, current_runtime, graph_version, graph_versions, readable_runtime
from citequest.agents_v3.contracts import AgentError, QA_SEPARATED_GRAPH_VERSION, digest
from citequest.agents_v3.function_output import validate_profile_transport
from citequest.agents_v3.json_review import compile_json_submission
from citequest.agents_v3.model_contracts import model_contract
from citequest.agents_v3.product_profiles import NewTaskProfile, QA8_PROTOCOL_METADATA, TaskProfilePolicy, qa_retrieval_protocol
from citequest.agents_v3.qa_adaptive_contracts import SUPPLEMENT_ROLE, SUPPLEMENT_SCHEMA_NAME, SupplementPlan
from citequest.agents_v3.typed_submission import compile_submission, normalize_submission, recorded_submission, submission_role


def profile(*, adaptive=False):
    result = {"kind": "qa", "profile_id": "qa-v9", "validation_state": "development",
        **QA8_PROTOCOL_METADATA, "qa_content_protocol": "isolated1"}
    if adaptive:
        result["qa_retrieval_protocol"] = "adaptive1"
    return result


def test_new_protocol_is_explicit_and_historical_serialization_is_exact():
    old = profile()
    assert NewTaskProfile.model_validate(old).model_dump(mode="json") == old
    new = NewTaskProfile.model_validate(profile(adaptive=True))
    assert new.model_dump(mode="json")["qa_retrieval_protocol"] == "adaptive1"
    policy = TaskProfilePolicy.model_validate_json(
        Path("runtime/s08/adaptive-retrieval-policy.example.json").read_text(), strict=True)
    assert policy.runtime_metadata("qa", "qa-v9")["qa_retrieval_protocol"] == "adaptive1"
    assert "qa_retrieval_protocol" not in policy.runtime_metadata("research", "research-v5")
    for changes in ({"qa_content_protocol": "anchored2"}, {"profile_id": "qa-v8"},
                    {"kind": "summary", "profile_id": "workset-v7"}):
        with pytest.raises(ValidationError):
            NewTaskProfile.model_validate({**profile(adaptive=True), **changes})
    assert qa_retrieval_protocol({"kind": "qa", "profile": old}) is None


def test_graph_versions_retain_old_bytes_and_freeze_new_protocol():
    common = ("qa", "qa-v9", POLICY, "json1", "grounded3", "chat-json2", "chat-json2",
        "jsonpatch1", "references1", "isolated1")
    old = graph_version(*common)
    assert old == QA_SEPARATED_GRAPH_VERSION + "+review-json1+review-chatjson2" + SUFFIX
    for navigation in (None, "laya1"):
        for rerank in (None, "tei-cross-encoder1"):
            version = graph_version(*common, navigation, rerank, "a" * 64 if rerank else None, "adaptive1")
            assert version.endswith("+retrieval-adaptive1" + SUFFIX)
            assert version in graph_versions(POLICY)
            frozen = {**profile(adaptive=True), "catalog_scope_policy": POLICY}
            if navigation:
                frozen["evidence_navigation_protocol"] = navigation
            if rerank:
                frozen.update(retrieval_rerank_protocol=rerank, rerank_config_sha256="a" * 64)
            task = {"kind": "qa", "profile": frozen, "graph_version": version}
            assert current_runtime(task, POLICY) and readable_runtime(task)
    with pytest.raises(AgentError):
        graph_version(*common[:-1], "anchored2", qa_retrieval="adaptive1")


def test_supplement_role_requires_metadata_and_keeps_json_receipt_identity():
    selected = {**profile(adaptive=True), **model_contract("deepseek-v4-pro")}
    request = {"model": "deepseek-v4-pro", "reasoning": {"effort": "none"},
        "truncation": "disabled", "instructions": "Synthetic supplement protocol fixture.",
        "max_output_tokens": 4096, "input": [{"role": "user", "content": "{}"}],
        "text": {"format": {"type": "json_schema", "name": SUPPLEMENT_SCHEMA_NAME,
            "strict": True, "schema": SupplementPlan.model_json_schema()}}}
    original = copy.deepcopy(request)
    key = "synthetic-command:qa5:supplement_plan:0"
    validate_profile_transport(selected, request)
    assert submission_role(selected, request, key) == SUPPLEMENT_ROLE
    compiled = compile_submission(selected, request, key, fixture=True)
    assert request == original
    assert compiled.receipt["role"] == SUPPLEMENT_ROLE
    assert compiled.receipt["schema_name"] == SUPPLEMENT_SCHEMA_NAME
    assert compiled.receipt["canonical_request_sha256"] == digest(request)
    assert compiled.wire["thinking"] == {"type": "disabled"}
    assert compiled.wire["response_format"] == {"type": "json_object"}
    unmarked = {name: value for name, value in selected.items() if name != "qa_retrieval_protocol"}
    with pytest.raises(AgentError):
        validate_profile_transport(unmarked, request)
    with pytest.raises(AgentError):
        submission_role(unmarked, request, key)
    with pytest.raises(AgentError):
        compile_json_submission(request, SUPPLEMENT_ROLE)
    mismatched = copy.deepcopy(request)
    mismatched["text"]["format"]["name"] = "citequest_qa5_plan"
    with pytest.raises(AgentError):
        submission_role(selected, mismatched, key)
    # The transport preserves bytes; SupplementPlan validation remains a domain gate.
    content = json.dumps({"synthetic_envelope_fixture": True})
    raw = {"object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": content}}]}
    recorded = recorded_submission(selected, request, key, raw, raw, 200)
    normalized = normalize_submission(selected, request, key, recorded)
    assert normalized["output"][0]["content"][0]["text"] == content
