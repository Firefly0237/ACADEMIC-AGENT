"""Pure version selection controls. No database, model or external HTTP."""
import pytest
from pydantic import ValidationError

from citequest.agents_v3 import catalog_scope
from citequest.agents_v3.contracts import AgentError, QA_AUDITED_GRAPH_VERSION, QA_GROUNDED_GRAPH_VERSION, QA_VERIFIED_GRAPH_VERSION
from citequest.agents_v3.product_profiles import NewTaskProfile, TaskProfilePolicy, qa_execution_protocol
from citequest.agents_v3.strict_chat import is_strict_review


@pytest.mark.parametrize("protocol,base_version,api_protocol", [
    ("grounded2", QA_AUDITED_GRAPH_VERSION, "chat-strict1"),
    ("grounded3", QA_VERIFIED_GRAPH_VERSION, "chat-strict1"),
    ("grounded3", QA_VERIFIED_GRAPH_VERSION, "chat-strict2"),
])
def test_audited_version_is_explicit_and_catalog_identity_remains_complete(protocol, base_version, api_protocol):
    selected = NewTaskProfile(kind="qa", profile_id="qa-v6", validation_state="development",
        review_transport_protocol="function1", qa_execution_protocol=protocol, review_api_protocol=api_protocol)
    policy = TaskProfilePolicy(release_id="explicit-audited-fixture", profiles=[selected])
    metadata = policy.runtime_metadata("qa", "qa-v6")
    task = {"kind": "qa", "profile": {"profile_id": "qa-v6", **metadata}}
    assert qa_execution_protocol(task) == protocol
    version = catalog_scope.graph_version("qa", "qa-v6", catalog_scope.POLICY,
        metadata["review_transport_protocol"], metadata["qa_execution_protocol"], metadata["review_api_protocol"])
    suffix = {"chat-strict1": "+review-chatstrict1", "chat-strict2": "+review-chatstrict2"}[api_protocol]
    assert version == base_version + "+review-function1" + suffix + "+task-catalog-snapshot1"
    assert version in catalog_scope.graph_versions(catalog_scope.POLICY)
    request = {"tools": [{"type": "function", "name": "submit_review"}],
        "tool_choice": {"type": "function", "name": "submit_review"}}
    assert is_strict_review(task["profile"], request)


def test_legacy_omission_and_grounded1_select_their_original_versions():
    old = NewTaskProfile(kind="qa", profile_id="qa-v6", validation_state="development")
    assert set(old.model_dump()) == {"kind", "profile_id", "validation_state"}
    assert catalog_scope.graph_version("qa", "qa-v6", catalog_scope.POLICY, "function1", "grounded1") == (
        QA_GROUNDED_GRAPH_VERSION + "+review-function1+task-catalog-snapshot1")
    assert qa_execution_protocol({"kind": "qa", "profile": {"profile_id": "qa-v6"}}) is None


@pytest.mark.parametrize("protocol", ["grounded2", "grounded3"])
def test_audited_protocol_cannot_enter_without_its_registered_transport(protocol):
    with pytest.raises(ValidationError):
        NewTaskProfile(kind="qa", profile_id="qa-v6", validation_state="development",
            review_transport_protocol="function1", qa_execution_protocol=protocol)
    with pytest.raises(AgentError):
        catalog_scope.graph_version("qa", "qa-v6", catalog_scope.POLICY, "function1", protocol)


def test_length_compiler_receipt_uses_its_bound_protocol(monkeypatch):
    from citequest.agents_v3.contracts import canonical, digest
    from citequest.agents_v3.strict_chat import compile_review_request
    from tests.v3.test_strict_review_binding import Binding

    binding = Binding(monkeypatch)
    binding.task["profile"].update(qa_execution_protocol="grounded3", review_api_protocol="chat-strict2")
    compiled = compile_review_request(binding.attempt["request"], fixture=True, protocol="chat-strict2")
    receipt = {**compiled.receipt, "actual_wire_bytes": len(canonical(compiled.wire)),
        "actual_wire_sha256": digest(canonical(compiled.wire))}
    recorded = binding.execution.observe_model_wire(binding.attempt["id"], receipt)
    assert recorded["protocol"] == "chat-strict2"
    assert recorded["compiler"] == "deepseek-chat-schema-subset/2"
    assert "transformed_constraints" in recorded
    wrong = {**receipt, "compiler": "deepseek-chat-schema-subset/1"}
    with pytest.raises(AgentError, match="model_wire_receipt_invalid"):
        binding.execution.observe_model_wire(binding.attempt["id"], wrong)
