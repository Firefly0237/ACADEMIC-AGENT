"""One pure navigation binding/frontier/withdrawal control; no provider or DB."""
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3.catalog_scope import POLICY, current_runtime, graph_version
from citequest.agents_v3.contracts import AgentError
from citequest.agents_v3.evidence_navigation import navigation_frontier
from citequest.agents_v3.product_profiles import NewTaskProfile, QA8_PROTOCOL_METADATA
from citequest.agents_v3.result_projection import project_task_result


def test_navigation_is_opt_in_bounded_and_rechecks_pruned_source_access(monkeypatch):
    profile = {"kind": "qa", "profile_id": "qa-v9", "validation_state": "development",
        **QA8_PROTOCOL_METADATA, "qa_content_protocol": "isolated1"}
    assert "evidence_navigation_protocol" not in NewTaskProfile(**profile).model_dump()
    enabled = NewTaskProfile(**profile, evidence_navigation_protocol="laya1")
    with pytest.raises(ValidationError):
        NewTaskProfile(**{**profile, "qa_content_protocol": "online1"}, evidence_navigation_protocol="laya1")
    bound = {key: value for key, value in enabled.model_dump().items()
        if key not in {"kind", "validation_state"}}
    bound["catalog_scope_policy"] = POLICY
    version = graph_version("qa", "qa-v9", POLICY, "json1", "grounded3", "chat-json2",
        "chat-json2", "jsonpatch1", "references1", "isolated1", "laya1")
    task = {"kind": "qa", "profile": bound, "graph_version": version}
    assert current_runtime(task, POLICY) and "+evidence-laya1+task-catalog-snapshot1" in version
    assert not current_runtime({**task, "profile": {key: value for key, value in bound.items()
        if key != "evidence_navigation_protocol"}}, POLICY)

    evidence = [{"ref_id": "a1", "file_id": "A"}, {"ref_id": "a2", "file_id": "A"},
        {"ref_id": "b1", "file_id": "B"}]
    assert navigation_frontier(evidence, 2) == ["a1", "b1"]
    assert navigation_frontier(evidence, 8) == ["a1", "b1", "a2"]
    with pytest.raises(AgentError):
        navigation_frontier([evidence[0], evidence[0]], 2)

    # Explicit authority fixture: pruned candidates remain visible in the score trace,
    # so withdrawing one must also suppress the result, not only the selected quote.
    monkeypatch.setattr("citequest.agents_v3.user_skills.validate_task_skills", lambda *_args: None)
    memory = SimpleNamespace(current=lambda *_args: [], validate_result_context=lambda *_args: None)
    service = SimpleNamespace(database=None)
    actor = SimpleNamespace(owner_id=uuid4())
    task_id = uuid4()
    result = {"publication_status": "validated_ai_review", "citations": [{"ref_id": "a1"}],
        "context_manifest": {}, "evidence_navigation": {"input_frontier_ref_ids": ["a1", "b1"]}}
    projection_task = {"kind": "qa", "project_id": uuid4(), "result": result}
    observed = []

    def authorize(_actor, _task_id, references):
        observed.extend(references)
        if "b1" in references:
            raise AgentError("evidence_access_revoked", 403)

    projected = project_task_result(service, memory, authorize, actor, task_id, projection_task)
    assert set(observed) == {"a1", "b1"}
    assert projected["result"] is None
    assert projected["result_availability"] == "evidence_unavailable_or_access_revoked"
