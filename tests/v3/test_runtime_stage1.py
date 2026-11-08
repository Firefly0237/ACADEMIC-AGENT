"""P01 regression controls using explicit in-memory services and SQL doubles.

The SSE control uses the actual FastAPI route through TestClient. No database,
network service, model, filesystem fixture or provider is contacted.
"""
import copy
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from citequest.agents_v3 import database as database_module
from citequest.agents_v3.contracts import AgentError, ResearchBrief, RuntimeLimits, TaskCreate, VersionedBriefRef, digest
from citequest.agents_v3.database import Execution, TaskService
from citequest.agents_v3.memory import MemoryService
from citequest.agents_v3.product_profiles import NewTaskProfile, is_profile_implemented
from citequest.v2.domain import Actor
from citequest.web_v3.agent_routes import product_agent_router

OWNER = UUID(int=101)
PROJECT = UUID(int=102)
TASK = UUID(int=103)


def test_unfinished_profile_is_not_new_task_configuration_but_existing_profiles_remain_usable():
    assert not is_profile_implemented("research-v5")
    with pytest.raises(ValidationError, match="implementation is incomplete"):
        NewTaskProfile(kind="research", profile_id="research-v5", validation_state="development")
    for kind, profile in (("qa", "qa-v6"), ("stance", "stance-v3"),
                          ("summary", "workset-v7"), ("research", "research-v4")):
        assert is_profile_implemented(profile)
        assert NewTaskProfile(kind=kind, profile_id=profile, validation_state="development").profile_id == profile


@pytest.mark.parametrize("replay", [False, True])
def test_new_task_implementation_guard_keeps_committed_idempotent_replay_readable(monkeypatch, replay):
    request = TaskCreate(project_id=PROJECT, operation_id=UUID(int=104), kind="research", profile_id="research-v5",
        brief=ResearchBrief(kind="research", question="Explicit synthetic intake fixture",
            research_brief=VersionedBriefRef(id=UUID(int=105), revision=1, sha256="a" * 64)))
    existing = {"id": TASK, "create_hash": digest(request.model_dump(mode="json")), "state": "waiting_input"}
    queries, implementation_checks = [], []

    class ReadOnlyConnection:
        def execute(self, query, parameters):
            queries.append(query)
            if query.startswith("SELECT 1 FROM cq_app.projects"):
                row = {"authorized": True}
            elif query.startswith("SELECT * FROM cq_app.agent_tasks WHERE owner_id="):
                row = existing if replay else None
            else:
                pytest.fail("new-profile guard/replay reached SQL beyond its read-only admission boundary")
            return SimpleNamespace(fetchone=lambda: row)

    @contextmanager
    def connect():
        yield ReadOnlyConnection()

    database = SimpleNamespace(connect=connect, catalog_scope_policy=None, development_dispatch_policy=None,
        development_campaign_policy=None)
    service = TaskService(database, RuntimeLimits(mode="bounded"))
    original_implemented = database_module.is_profile_implemented

    def implementation_check(profile):
        implementation_checks.append(profile)
        return original_implemented(profile)

    monkeypatch.setattr(database_module, "wallet_locks", lambda *_: None)
    monkeypatch.setattr(database_module, "is_profile_implemented", implementation_check)
    monkeypatch.setattr(service, "public", lambda row, **_: row)
    if replay:
        assert service.create(OWNER, request) is existing
        assert implementation_checks == []
    else:
        with pytest.raises(AgentError, match="task_profile_implementation_unavailable") as caught:
            service.create(OWNER, request)
        assert caught.value.status == 503
        assert implementation_checks == ["research-v5"]
    assert len(queries) == 2


def test_second_finish_of_closed_run_cannot_update_result_history_or_events(monkeypatch):
    task = {"state": "waiting_input", "result": {"interrupts": ["synthetic missing information"]},
        "history": [{"question": "synthetic previous turn", "answer": {"text": "retained"}}]}
    original = copy.deepcopy(task)
    queries = []

    class ClosedRunConnection:
        def execute(self, query, parameters):
            queries.append((query, parameters))
            assert query.startswith("SELECT ended_at FROM cq_app.agent_runs")
            assert parameters == (UUID(int=106), TASK, 1)
            return SimpleNamespace(fetchone=lambda: {"ended_at": datetime(2026, 11, 7, tzinfo=UTC)})

    @contextmanager
    def connect():
        yield ClosedRunConnection()

    execution = object.__new__(Execution)
    execution.database = SimpleNamespace(connect=connect)
    execution.run_id, execution.task_id, execution.epoch = UUID(int=106), TASK, 1

    def locked(db, *, dispatch):
        assert isinstance(db, ClosedRunConnection) and dispatch is False
        return task

    monkeypatch.setattr(execution, "lock_task", locked)
    monkeypatch.setattr(database_module, "event", lambda *_: pytest.fail("closed run emitted a duplicate event"))
    execution.finish("failed", error="synthetic_transport_cleanup_failure")
    assert len(queries) == 1
    assert task == original


@pytest.mark.parametrize("after_seq,last_event_id", [(0, None), (17, "201")])
def test_actual_sse_route_drains_451_terminal_events_before_settled_and_resumes_cursor(after_seq, last_event_id):
    rows = [{"seq": seq, "event_type": "workflow_progress", "payload": {"phase": "synthetic"}}
            for seq in range(1, 452)]
    cursors = []

    class NoDatabase:
        def connect(self, *_args, **_kwargs):
            pytest.fail("SSE in-memory control attempted SQL")

    class SyntheticTaskService:
        database = NoDatabase()

        def get(self, owner, task):
            assert (owner, task) == (OWNER, TASK)
            return {"id": TASK, "state": "succeeded", "event_seq": len(rows)}

        def events(self, owner, task, cursor):
            assert (owner, task) == (OWNER, TASK)
            cursors.append(cursor)
            return rows[cursor:cursor + 200]

    def private_actor(request: Request):
        assert request.url.path.endswith("/events/stream")
        return Actor(OWNER)

    def forbid_evidence(*_args, **_kwargs):
        pytest.fail("SSE in-memory control attempted an evidence read")

    service = SyntheticTaskService()
    evidence = SimpleNamespace(read=forbid_evidence, authorize=forbid_evidence, admit_scope=forbid_evidence)
    app = FastAPI()
    app.include_router(product_agent_router(service, MemoryService(service.database),
        SimpleNamespace(private_actor=private_actor), evidence))
    headers = {} if last_event_id is None else {"Last-Event-ID": last_event_id}
    with TestClient(app) as client:
        response = client.get(f"/api/agent/tasks/{TASK}/events/stream", params={"after_seq": after_seq}, headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    start = max(after_seq, int(last_event_id or 0))
    ids = [int(line.removeprefix("id: ")) for line in response.text.splitlines() if line.startswith("id: ")]
    assert ids == list(range(start + 1, 452))
    assert cursors == list(range(start, 451, 200))
    assert response.text.count("event: settled") == 1
    assert response.text.index("id: 451\n") < response.text.index("event: settled")
