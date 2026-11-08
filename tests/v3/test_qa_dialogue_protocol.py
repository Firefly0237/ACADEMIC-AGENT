"""Pure P02 release controls; the revision ledger is an explicit SQL double."""
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

from citequest.agents_v3 import catalog_scope
from citequest.agents_v3.contracts import AgentError, PROFILES, QA_GROUNDED_GRAPH_VERSION, QA_V6_GRAPH_VERSION, digest
from citequest.agents_v3.database import Execution
from citequest.agents_v3.product_profiles import NewTaskProfile, TaskProfilePolicy, public_profile_policy, qa_execution_protocol


def test_omitted_and_explicit_none_keep_original_policy_JSON_hash_and_QA6_graph():
    original = {"schema_version": "new-task-profile-policy/1", "release_id": "synthetic-old-QA6-release",
        "profiles": [{"kind": "qa", "profile_id": "qa-v6", "validation_state": "development",
                      "review_transport_protocol": "function1"}]}
    policy = TaskProfilePolicy.model_validate(original, strict=True)
    explicit_none = TaskProfilePolicy.model_validate({**original, "profiles": [
        {**original["profiles"][0], "qa_execution_protocol": None}]}, strict=True)
    assert policy.model_dump(mode="json") == explicit_none.model_dump(mode="json") == original
    assert digest(policy.model_dump(mode="json")) == digest(original)
    expected_public_hash = digest({**original, "profile_hashes": {
        "qa-v6": digest(PROFILES["qa-v6"].model_dump(mode="json"))}})
    assert public_profile_policy(policy)["policy_sha256"] == expected_public_hash
    assert policy.runtime_metadata("qa", "qa-v6") == {"review_transport_protocol": "function1"}
    old_graph = QA_V6_GRAPH_VERSION + catalog_scope.REVIEW_SUFFIX + catalog_scope.SUFFIX
    assert catalog_scope.graph_version("qa", "qa-v6", catalog_scope.POLICY, "function1") == old_graph
    assert catalog_scope.graph_version("qa", "qa-v6", catalog_scope.POLICY, "function1", None) == old_graph
    old_task = {"kind": "qa", "profile": {"profile_id": "qa-v6", "catalog_scope_policy": catalog_scope.POLICY,
        "review_transport_protocol": "function1"}, "graph_version": old_graph}
    assert qa_execution_protocol(old_task) is None
    assert catalog_scope.current_runtime(old_task, catalog_scope.POLICY) and catalog_scope.readable_runtime(old_task)


def test_grounded_protocol_has_distinct_graph_and_same_resource_and_two_revision_ledger(monkeypatch):
    policy = TaskProfilePolicy(release_id="synthetic-grounded-release", profiles=[NewTaskProfile(
        kind="qa", profile_id="qa-v6", validation_state="development", review_transport_protocol="function1",
        qa_execution_protocol="grounded1")])
    metadata = policy.runtime_metadata("qa", "qa-v6")
    assert metadata == {"review_transport_protocol": "function1", "qa_execution_protocol": "grounded1"}
    version = catalog_scope.graph_version("qa", "qa-v6", catalog_scope.POLICY, "function1", "grounded1")
    assert version == QA_GROUNDED_GRAPH_VERSION + catalog_scope.REVIEW_SUFFIX + catalog_scope.SUFFIX
    assert version in catalog_scope.graph_versions(catalog_scope.POLICY)
    assert not any(QA_GROUNDED_GRAPH_VERSION in item for item in catalog_scope.graph_versions(None))
    resources = PROFILES["qa-v6"].model_dump(mode="json")
    assert (resources["model_calls"], resources["active_seconds"], resources["tool_calls"]) == (18, 450, 64)
    assert (resources["input_tokens"], resources["output_tokens"], resources["per_call_output"],
        resources["request_bytes"], resources["cash_ncny"]) == (384000, 73728, 4096, 131072, 15000000000)
    profile = {**resources, **metadata, "catalog_scope_policy": catalog_scope.POLICY}
    task = {"kind": "qa", "profile": profile, "graph_version": version}
    assert qa_execution_protocol(task) == "grounded1"
    assert catalog_scope.current_runtime(task, catalog_scope.POLICY) and catalog_scope.readable_runtime(task)
    assert not catalog_scope.readable_runtime({**task, "graph_version": QA_V6_GRAPH_VERSION +
        catalog_scope.REVIEW_SUFFIX + catalog_scope.SUFFIX})

    queries = []

    class TwoSpentCredits:
        def execute(self, query, parameters):
            queries.append(query)
            assert query.lstrip().startswith("SELECT payload FROM cq_app.agent_events")
            return SimpleNamespace(fetchall=lambda: [{"payload": {"logical_key": "synthetic-old-repair-1"}},
                                                    {"payload": {"logical_key": "synthetic-old-repair-2"}}])

    @contextmanager
    def connect():
        yield TwoSpentCredits()

    execution = object.__new__(Execution)
    execution.database, execution.task_id = SimpleNamespace(connect=connect), UUID(int=901)
    monkeypatch.setattr(execution, "lock_task", lambda _db: task)
    with pytest.raises(AgentError, match="content_revision_limit"):
        execution.reserve_content_revision("synthetic-new-repair", phase="qa_content_revision",
            error_codes=["synthetic_review_rejected"])
    assert len(queries) == 1


def test_grounded_protocol_rejects_kind_profile_transport_catalog_and_immutable_marker_mismatches():
    for kind, profile, transport in (("stance", "stance-v3", "function1"),
                                    ("qa", "qa-v5", "function1"), ("qa", "qa-v6", None)):
        with pytest.raises(ValidationError):
            NewTaskProfile(kind=kind, profile_id=profile, validation_state="development",
                review_transport_protocol=transport, qa_execution_protocol="grounded1")
    for kind, profile, policy, transport, protocol in (
        ("qa", "qa-v6", None, "function1", "grounded1"),
        ("qa", "qa-v6", catalog_scope.POLICY, None, "grounded1"),
        ("qa", "qa-v5", catalog_scope.POLICY, "function1", "grounded1"),
        ("stance", "stance-v3", catalog_scope.POLICY, "function1", "grounded1"),
        ("qa", "qa-v6", catalog_scope.POLICY, "function1", "unregistered"),
    ):
        with pytest.raises(AgentError):
            catalog_scope.graph_version(kind, profile, policy, transport, protocol)
        with pytest.raises(AgentError):
            # A missing catalog is checked by graph construction; a malformed
            # persisted execution marker is checked before graph selection.
            if policy is None:
                catalog_scope.require_runtime({"kind": kind, "profile": {"profile_id": profile,
                    "catalog_scope_policy": policy, "review_transport_protocol": transport,
                    "qa_execution_protocol": protocol}}, catalog_scope.POLICY)
            else:
                qa_execution_protocol({"kind": kind, "profile": {"profile_id": profile,
                    "review_transport_protocol": transport, "qa_execution_protocol": protocol}})
