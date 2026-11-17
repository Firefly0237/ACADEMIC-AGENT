"""Pending plans reach the public DTO only after current authority checks."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from citequest.agents_v3.contracts import PROFILES, AgentError, ResearchBrief, digest, graph_version_for_kind
from citequest.agents_v3.database import TaskService
from citequest.agents_v3.execution_plan import INTERRUPT_KIND, build_execution_plan
from citequest.agents_v3.result_projection import project_task_result
from citequest.v2.domain import Actor


def fixture():
    owner, project, task_id, brief_id, memory_id = uuid4(), uuid4(), uuid4(), uuid4(), str(uuid4())
    brief = ResearchBrief(kind="research", question="Design a bounded synthetic experiment.",
        research_action="experiment", execution_plan_protocol="approval1",
        research_brief={"id": brief_id, "revision": 2, "sha256": "a" * 64}).model_dump(mode="json")
    profile = PROFILES["research-v6"].model_dump(mode="json")
    source = {"id": task_id, "owner_id": owner, "project_id": project, "kind": "research", "revision": 3,
        "state": "waiting_input", "brief": brief, "profile": profile, "profile_hash": digest(profile),
        "graph_version": graph_version_for_kind("research", "research-v6"), "command_seq": 1,
        "model_calls": 2, "tool_calls": 0, "input_tokens": 240, "output_tokens": 100,
        "known_ncny": 1, "reserved_ncny": 0, "active_seconds": 1.0,
        "error_code": None, "event_seq": 2, "cancellation_requested": False}
    plan = build_execution_plan(source, {"intake": {"brief_sha256": "a" * 64},
        "execution_plan_origin_command_id": str(uuid4()), "research_capacity": {},
        "outline": {"brief_sha256": "a" * 64, "questions": [{"id": "RQ1", "question": "Synthetic target",
            "retrieval_query": "bounded synthetic query", "experiment_ids": ["E1"]}]}})
    source["result"] = {"kind": "research", "status": "waiting_input", "publication_status": "not_evaluated",
        "interrupts": [{"kind": INTERRUPT_KIND, "plan": plan, "plan_sha256": digest(plan)}],
        "context_manifest": {"brief_id": str(brief_id), "brief_revision": 2, "brief_sha256": "a" * 64,
            "memory_refs": [{"id": memory_id, "revision": 1}], "evidence_refs": []}, "citations": [],
        "text": "PRIVATE_UNREVIEWED_BODY", "plan": {"private": "PRIVATE_UNREVIEWED_PLAN"},
        "role_handoffs": ["PRIVATE_ROLE_PROMPT"], "review_history": ["PRIVATE_REVIEW_BODY"]}

    @contextmanager
    def connect():
        yield object()

    service = SimpleNamespace(database=SimpleNamespace(connect=connect))
    memory = SimpleNamespace(current=Mock(return_value=[{"memory_id": memory_id, "revision": 1}]),
        validate_result_context=Mock())
    document = SimpleNamespace(id=brief_id, revision=2)
    return Actor(owner), service, memory, source, document, plan


def project(actor, service, memory, source, document, *, sha="a" * 64, skill_error=None, evidence_error=None):
    task = TaskService.public(source)
    authorize = Mock(side_effect=evidence_error)
    with patch("citequest.agents_v3.research_runtime.load_document", return_value=(document, sha)) as brief, \
            patch("citequest.agents_v3.user_skills.validate_task_skills", side_effect=skill_error) as skills:
        visible = project_task_result(service, memory, authorize, actor, source["id"], task)
    return visible, brief, skills, authorize


def test_service_public_retains_pending_plan_and_projection_returns_only_reviewable_payload():
    actor, service, memory, source, document, plan = fixture()
    dto = TaskService.public(source)
    assert dto["result"]["interrupts"][0]["plan"] == plan
    visible, brief, skills, authorize = project(actor, service, memory, source, document)
    assert visible["result_availability"] == "available_for_input"
    assert visible["result"]["interrupts"] == source["result"]["interrupts"]
    assert visible["result"]["interrupts"][0]["plan_sha256"] == digest(plan)
    assert "PRIVATE" not in str(visible)
    assert set(visible["result"]) == {"interrupts", "context_manifest", "citations", "kind", "status"}
    brief.assert_called_once()
    skills.assert_called_once()
    memory.current.assert_called_once_with(actor.owner_id, source["project_id"])
    authorize.assert_called_once_with(actor, source["id"], [])
    memory.validate_result_context.assert_not_called()


@pytest.mark.parametrize("barrier", ["brief_revision", "brief_hash", "deleted", "skill", "memory", "evidence"])
def test_current_revocation_or_changed_brief_hides_pending_plan_and_unreviewed_body(barrier):
    actor, service, memory, source, document, _ = fixture()
    if barrier == "brief_revision":
        document.revision += 1
    if barrier == "memory":
        memory.current.return_value = []
    sha = "b" * 64 if barrier == "brief_hash" else "a" * 64
    if barrier == "deleted":
        with patch("citequest.agents_v3.research_runtime.load_document", side_effect=AgentError("not_found", 404)):
            visible = project_task_result(service, memory, Mock(), actor, source["id"], TaskService.public(source))
    else:
        visible, _, _, _ = project(actor, service, memory, source, document, sha=sha,
            skill_error=AgentError("revoked", 403) if barrier == "skill" else None,
            evidence_error=AgentError("revoked", 403) if barrier == "evidence" else None)
    assert visible["result"] is None
    assert visible["result_availability"] != "available_for_input"
    assert "PRIVATE" not in str(visible)
    assert "bounded synthetic query" not in str(visible)
