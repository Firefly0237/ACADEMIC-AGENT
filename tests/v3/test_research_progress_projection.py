"""Current-authority metadata projection, with explicit in-process authorities.

No DB/provider or authentication substitute is represented as a live check.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from citequest.agents_v3.contracts import AgentError
from citequest.agents_v3.contracts import PROFILES
from citequest.agents_v3.database import TaskService
from citequest.agents_v3.research_progress import progress_from_state
from citequest.agents_v3.result_projection import project_task_result
from citequest.v2.domain import Actor


def fixture():
    owner, project, task_id, brief_id = uuid4(), uuid4(), uuid4(), uuid4()
    progress = progress_from_state({"outline": {"questions": [{"experiment_ids": ["E1"]}]},
        "literature": [{}], "methods": [], "unit_reviews": [], "revision_round": 0},
        {"kind": "research", "status": "partial"})
    raw = {"kind": "research", "research_progress": progress, "publication_status": "rejected",
        "plan": {"secret": "UNPUBLISHED_PRIVATE_PLAN"}, "role_handoffs": ["PRIVATE_QUOTE"],
        "context_manifest": {"brief_id": str(brief_id), "brief_revision": 2, "brief_sha256": "a"*64,
            "memory_refs": [], "evidence_refs": [{"ref_id": "EXPLICIT_REF"}]}, "citations": []}
    task = {"id": task_id, "project_id": project, "kind": "research", "revision": 2, "state": "partial",
        "brief": {"research_brief": {"id": str(brief_id)}}, "result": None, "result_availability": "quality_rejected"}
    source = {**copy.deepcopy(task), "owner_id": owner, "result": raw}
    @contextmanager
    def connect(): yield object()
    service = SimpleNamespace(database=SimpleNamespace(connect=connect))
    memory = SimpleNamespace(current=Mock(return_value=[]), validate_result_context=Mock())
    document = SimpleNamespace(id=brief_id, revision=2)
    return Actor(owner), service, memory, task_id, task, source, document, progress


def test_unpublished_body_stays_hidden_after_all_current_authority_checks():
    actor, service, memory, task_id, task, source, document, progress = fixture()
    authorize = Mock()
    with patch("citequest.agents_v3.result_projection.owned_task", return_value=source), \
            patch("citequest.agents_v3.research_runtime.load_document", return_value=(document, "a"*64)), \
            patch("citequest.agents_v3.user_skills.validate_task_skills") as skills:
        result = project_task_result(service, memory, authorize, actor, task_id, task)
    assert result["result"] is None and result["result_availability"] == "research_progress_only"
    assert result["research_progress"] == progress
    assert "PRIVATE" not in str(result) and "role_handoffs" not in result
    skills.assert_called_once()
    memory.validate_result_context.assert_called_once()
    authorize.assert_called_once_with(actor, task_id, ["EXPLICIT_REF"])


@pytest.mark.parametrize("barrier", ["brief", "skill", "memory", "evidence", "row_revision"])
def test_each_current_barrier_hides_even_cached_progress_metadata(barrier):
    actor, service, memory, task_id, task, source, document, progress = fixture()
    task["research_progress"] = progress  # An earlier projection must not survive revocation.
    authorize = Mock(side_effect=AgentError("revoked", 403) if barrier == "evidence" else None)
    if barrier == "brief": document.revision += 1
    if barrier == "memory": memory.validate_result_context.side_effect = AgentError("revoked", 403)
    if barrier == "row_revision": source["revision"] += 1
    with patch("citequest.agents_v3.result_projection.owned_task", return_value=source), \
            patch("citequest.agents_v3.research_runtime.load_document", return_value=(document, "a"*64)), \
            patch("citequest.agents_v3.user_skills.validate_task_skills",
                side_effect=AgentError("revoked", 403) if barrier == "skill" else None):
        result = project_task_result(service, memory, authorize, actor, task_id, task)
    assert result["result"] is None and "research_progress" not in result
    assert result["result_availability"] != "research_progress_only"


def test_old_records_without_safe_metadata_do_not_reconstruct_a_plan_or_checkpoint():
    actor, service, memory, task_id, task, source, document, _ = fixture()
    source["result"].pop("research_progress")
    with patch("citequest.agents_v3.result_projection.owned_task", return_value=source), \
            patch("citequest.agents_v3.research_runtime.load_document") as load, \
            patch("citequest.agents_v3.user_skills.validate_task_skills"):
        result = project_task_result(service, memory, Mock(), actor, task_id, task)
    assert result == task
    load.assert_not_called()


def test_actual_not_evaluated_research_DTO_enters_metadata_projection():
    from citequest.agents_v3.catalog_scope import POLICY, graph_version

    actor, service, memory, task_id, _task, source, document, progress = fixture()
    source["result"]["publication_status"] = "not_evaluated"
    source.update(profile={**PROFILES["research-v3"].model_dump(mode="json"), "catalog_scope_policy": POLICY},
        graph_version=graph_version("research", "research-v3", POLICY), model_calls=2, tool_calls=3,
        input_tokens=100, output_tokens=200, known_ncny=1, reserved_ncny=0, active_seconds=1.0,
        error_code=None, event_seq=1, cancellation_requested=False)
    task = TaskService.public(source, catalog_scope_policy=POLICY)
    assert task["result"] is None and task["result_availability"] == "partial_no_evaluated_claims"
    with patch("citequest.agents_v3.result_projection.owned_task", return_value=source), \
            patch("citequest.agents_v3.research_runtime.load_document", return_value=(document, "a"*64)), \
            patch("citequest.agents_v3.user_skills.validate_task_skills"):
        result = project_task_result(service, memory, Mock(), actor, task_id, task)
    assert result["result"] is None and result["research_progress"] == progress
    assert result["result_availability"] == "research_progress_only"
