"""Admission and historical identity for reference-only quantitative input."""
import copy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from citequest.agents_v3.catalog_scope import POLICY, graph_version, graph_versions, readable_runtime
from citequest.agents_v3.contracts import AgentError, PROFILES, digest
from citequest.agents_v3.product_profiles import NewTaskProfile, TaskProfilePolicy, qa_context_protocol


def test_context_protocol_is_explicit_and_old_identities_are_unchanged():
    old = {
        "kind": "qa", "profile_id": "qa-v7", "validation_state": "development",
        "review_transport_protocol": "json1", "qa_execution_protocol": "grounded3",
        "review_api_protocol": "chat-json1", "structured_output_protocol": "chat-functions1",
        "answer_revision_protocol": "jsonpatch1",
    }
    old_profile = NewTaskProfile.model_validate(old, strict=True)
    assert old_profile.model_dump(mode="json") == old
    new_profile = NewTaskProfile.model_validate({**old, "qa_context_protocol": "references1"}, strict=True)
    policy = TaskProfilePolicy(release_id="context-control", profiles=[new_profile])
    metadata = policy.runtime_metadata("qa", "qa-v7")
    assert metadata["qa_context_protocol"] == "references1"
    arguments = ("qa", "qa-v7", POLICY, "json1", "grounded3", "chat-json1", "chat-functions1", "jsonpatch1")
    old_version = graph_version(*arguments)
    new_version = graph_version(*arguments, "references1")
    assert old_version.startswith("citequest-agent-graph/32-")
    assert new_version.startswith("citequest-agent-graph/33-")
    assert old_version != new_version
    assert {old_version, new_version}.issubset(graph_versions(POLICY))
    old_task = {"kind": "qa", "profile": {**PROFILES["qa-v7"].model_dump(),
        **{key: value for key, value in metadata.items() if key != "qa_context_protocol"},
        "catalog_scope_policy": POLICY}, "graph_version": old_version}
    assert qa_context_protocol(old_task) is None
    assert readable_runtime(old_task)
    new_task = copy.deepcopy(old_task)
    new_task["profile"]["qa_context_protocol"] = "references1"
    new_task["graph_version"] = new_version
    assert qa_context_protocol(new_task) == "references1"
    assert readable_runtime(new_task)
    for key, value in (("profile_id", "qa-v6"), ("review_transport_protocol", "function1"),
                       ("review_api_protocol", "chat-strict2"), ("answer_revision_protocol", None),
                       ("structured_output_protocol", None)):
        with pytest.raises(ValidationError):
            NewTaskProfile.model_validate({**old, "qa_context_protocol": "references1", key: value}, strict=True)
        invalid = copy.deepcopy(new_task)
        invalid["profile"][key] = value
        with pytest.raises(AgentError):
            qa_context_protocol(invalid)
    with pytest.raises(AgentError):
        graph_version(*arguments, "unknown")
    assert PROFILES["qa-v7"].request_bytes == 131072


def test_source_identity_reaches_author_and_is_not_duplicated_in_review(monkeypatch):
    from citequest.agents_v3 import qa_v7

    def base_prompt(*_args, **_kwargs):
        manifest = {}
        return {"context_manifest": manifest}, manifest

    monkeypatch.setattr(qa_v7.QADialogueVerified, "prompt", base_prompt)
    provenance = {"source_provenance": {"S1": {"source_ref": {"kind": "synthetic", "id": "one"}}},
        "quote_sources": {"Q1": "S1"}}
    monkeypatch.setattr(qa_v7, "quote_source_provenance", lambda _scope: copy.deepcopy(provenance))
    task = {"kind": "qa", "profile": {
        "profile_id": "qa-v7", "qa_execution_protocol": "grounded3", "review_transport_protocol": "json1",
        "review_api_protocol": "chat-json1", "structured_output_protocol": "chat-functions1",
        "answer_revision_protocol": "jsonpatch1", "qa_context_protocol": "references1"}}
    node = object.__new__(qa_v7.QA7)
    node.graph = SimpleNamespace(execution=SimpleNamespace(snapshot=lambda: task))
    node.scope = lambda _state: object()
    prompt, manifest = node.prompt({})
    assert prompt["quote_source_provenance"] == provenance
    assert manifest["quote_source_provenance_sha256"] == digest(provenance)
    for options in ({"reviewer": True}, {"planner": True}):
        assert "quote_source_provenance" not in node.prompt({}, **options)[0]
    task["profile"].pop("qa_context_protocol")
    assert "quote_source_provenance" not in node.prompt({})[0]
