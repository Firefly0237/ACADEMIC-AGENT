"""One pure control for explicit content versions and legacy serialization."""
import pytest
from pydantic import ValidationError

from citequest.agents_v3.catalog_scope import POLICY, graph_version, graph_versions, readable_runtime
from citequest.agents_v3.contracts import (
    QA_DECOMPOSED_GRAPH_VERSION,
    QA_SEPARATED_GRAPH_VERSION,
    QA_ONLINE_GRAPH_VERSION,
    AgentError,
    PROFILES,
    canonical,
    digest,
)
from citequest.agents_v3.product_profiles import (
    QA8_PROTOCOL_METADATA,
    NewTaskProfile,
    TaskProfilePolicy,
    public_profile_policy,
    qa_content_protocol,
)


def test_content_protocol_is_explicit_and_none_preserves_original_policy_bytes():
    legacy_data = {"kind": "qa", "profile_id": "qa-v9", "validation_state": "development",
        **QA8_PROTOCOL_METADATA}
    legacy = NewTaskProfile(**legacy_data)
    explicit_none = NewTaskProfile(**legacy_data, qa_content_protocol=None)
    assert canonical(legacy.model_dump(mode="json")) == canonical(legacy_data)
    assert canonical(explicit_none.model_dump(mode="json")) == canonical(legacy_data)
    legacy_policy = TaskProfilePolicy(release_id="qa-content-pure-control", profiles=[legacy])
    original_policy = {"schema_version": "new-task-profile-policy/1", "release_id": "qa-content-pure-control",
        "profiles": [legacy_data]}
    assert legacy_policy.model_dump(mode="json") == original_policy
    original_public = {**original_policy,
        "profile_hashes": {"qa-v9": digest(PROFILES["qa-v9"].model_dump(mode="json"))}}
    assert public_profile_policy(legacy_policy)["policy_sha256"] == digest(original_public)
    assert legacy_policy.runtime_metadata("qa", "qa-v9") == QA8_PROTOCOL_METADATA
    old_task = {"kind": "qa", "profile": {"profile_id": "qa-v9", **QA8_PROTOCOL_METADATA}}
    assert qa_content_protocol(old_task) is None
    version_args = ("qa", "qa-v9", POLICY, "json1", "grounded3", "chat-json2", "chat-json2",
        "jsonpatch1", "references1")
    old_graph = graph_version(*version_args)
    original_worker_qa9_graphs = {
        "citequest-agent-graph/34-grounded-grader-and-reasoning-QA+review-json1+review-chatjson2+task-catalog-snapshot1"}
    assert old_graph in original_worker_qa9_graphs
    assert graph_version(*version_args, None) == old_graph
    new_graph = graph_version(*version_args, "anchored2")
    assert new_graph.startswith(QA_DECOMPOSED_GRAPH_VERSION + "+")
    assert {old_graph, new_graph} <= set(graph_versions(POLICY))
    # The old worker's QA9 support fixture cannot select a Graph35 command.
    assert new_graph not in original_worker_qa9_graphs
    online_graph = graph_version(*version_args, "online1")
    assert online_graph.startswith(QA_ONLINE_GRAPH_VERSION + "+")
    assert online_graph in graph_versions(POLICY)
    assert online_graph not in original_worker_qa9_graphs | {new_graph}
    online_profile = NewTaskProfile(**legacy_data, qa_content_protocol="online1")
    online_policy = TaskProfilePolicy(release_id="qa-content-pure-control", profiles=[online_profile])
    online_metadata = {**QA8_PROTOCOL_METADATA, "qa_content_protocol": "online1"}
    assert online_policy.runtime_metadata("qa", "qa-v9") == online_metadata
    assert qa_content_protocol({"kind": "qa", "profile": {"profile_id": "qa-v9", **online_metadata}}) == "online1"
    online_task = {"kind": "qa", "profile": {**PROFILES["qa-v9"].model_dump(mode="json"),
        **online_metadata, "catalog_scope_policy": POLICY}, "graph_version": online_graph}
    assert readable_runtime(online_task)
    assert not readable_runtime({**online_task, "graph_version": old_graph})
    assert not readable_runtime({**online_task, "graph_version": new_graph})

    selected = NewTaskProfile(**legacy_data, qa_content_protocol="anchored2")
    policy = TaskProfilePolicy(release_id="qa-content-pure-control", profiles=[selected])
    expected = {**QA8_PROTOCOL_METADATA, "qa_content_protocol": "anchored2"}
    assert policy.runtime_metadata("qa", "qa-v9") == expected
    assert public_profile_policy(policy)["policy_sha256"] != public_profile_policy(legacy_policy)["policy_sha256"]
    assert qa_content_protocol({"kind": "qa", "profile": {"profile_id": "qa-v9", **expected}}) == "anchored2"
    with pytest.raises(ValidationError):
        NewTaskProfile(**{**legacy_data, "profile_id": "qa-v8"}, qa_content_protocol="anchored2")
    with pytest.raises(AgentError, match="task_qa_content_protocol_not_supported"):
        qa_content_protocol({"kind": "qa", "profile": {"profile_id": "qa-v8", **expected}})

    isolated = NewTaskProfile(**legacy_data, qa_content_protocol="isolated1")
    isolated_policy = TaskProfilePolicy(release_id="qa-content-pure-control", profiles=[isolated])
    isolated_metadata = {**QA8_PROTOCOL_METADATA, "qa_content_protocol": "isolated1"}
    assert isolated_policy.runtime_metadata("qa", "qa-v9") == isolated_metadata
    assert qa_content_protocol({"kind": "qa", "profile": {
        "profile_id": "qa-v9", **isolated_metadata}}) == "isolated1"
    isolated_graph = graph_version(*version_args, "isolated1")
    assert isolated_graph.startswith(QA_SEPARATED_GRAPH_VERSION + "+")
    assert isolated_graph in graph_versions(POLICY)
    assert isolated_graph not in {old_graph, new_graph, graph_version(*version_args, "online1")}
