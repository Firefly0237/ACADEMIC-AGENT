"""Pure versioned task/frontier admission; no navigation model is loaded."""
import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from citequest.agents_v3.catalog_scope import POLICY, SUFFIX, graph_version, graph_versions
from citequest.agents_v3.contracts import AgentError, QA_SEPARATED_GRAPH_VERSION
from citequest.agents_v3.evidence_navigation import navigation_frontier
from citequest.agents_v3.product_profiles import NewTaskProfile, QA8_PROTOCOL_METADATA, TaskProfilePolicy, evidence_navigation_protocol


def profile(protocol):
    return {"kind": "qa", "profile_id": "qa-v9", "validation_state": "development",
        **QA8_PROTOCOL_METADATA, "qa_content_protocol": "isolated1", "evidence_navigation_protocol": protocol}


def test_laya2_is_explicit_and_legacy_profile_serialization_is_unchanged():
    legacy = profile("laya1")
    assert NewTaskProfile.model_validate(legacy).model_dump(mode="json") == legacy
    policy = TaskProfilePolicy.model_validate_json(Path("runtime/s08/laya2-policy.example.json").read_text(), strict=True)
    metadata = policy.runtime_metadata("qa", "qa-v9")
    assert metadata["evidence_navigation_protocol"] == "laya2"
    assert evidence_navigation_protocol({"kind": "qa", "profile": {"profile_id": "qa-v9", **metadata}}) == "laya2"
    for changes in ({"profile_id": "qa-v8"}, {"qa_content_protocol": "anchored2"}):
        with pytest.raises(ValidationError):
            NewTaskProfile.model_validate({**profile("laya2"), **changes})


def test_laya2_graph_identity_does_not_upgrade_old_laya1_tasks():
    common = ("qa", "qa-v9", POLICY, "json1", "grounded3", "chat-json2", "chat-json2",
        "jsonpatch1", "references1", "isolated1")
    assert graph_version(*common, "laya1") == QA_SEPARATED_GRAPH_VERSION + "+review-json1+review-chatjson2+evidence-laya1" + SUFFIX
    for rerank in (None, "tei-cross-encoder1"):
        for retrieval in (None, "adaptive1"):
            version = graph_version(*common, "laya2", rerank, "a" * 64 if rerank else None, retrieval)
            assert "+evidence-laya2" in version and version.endswith(SUFFIX)
            assert version in graph_versions(POLICY)


def test_complete_twenty_ref_frontier_needs_the_new_explicit_capacity():
    evidence = [{"ref_id": "issued:" + str(index), "text": "Complete literal α = β. " + str(index),
        "source_ref": {"kind": "native_representation", "id": "document:" + str(index % 4)}} for index in range(20)]
    original = copy.deepcopy(evidence)
    with pytest.raises(AgentError):
        navigation_frontier(evidence, 20)
    refs = navigation_frontier(evidence, 20, capacity_limit=20)
    assert len(refs) == 20 and set(refs) == {item["ref_id"] for item in evidence}
    assert evidence == original
    assert len(navigation_frontier(evidence, 8)) == 8
    with pytest.raises(AgentError):
        navigation_frontier(evidence, 21, capacity_limit=20)
