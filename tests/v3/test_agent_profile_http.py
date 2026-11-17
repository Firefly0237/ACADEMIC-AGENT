"""Real FastAPI HTTP with an explicit database that forbids all SQL.

These controls verify capability authentication/composition only. Real
transaction admission, replay and current-source authorization are separate.
"""
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from citequest.agents_v3.api import build_router
from citequest.agents_v3.contracts import RuntimeLimits
from citequest.agents_v3.database import TaskService
from citequest.agents_v3.memory import MemoryService
from citequest.agents_v3.product_profiles import NewTaskProfile, TaskProfilePolicy
from citequest.v2.domain import Actor


class NoSQLDatabase:
    def connect(self, *args, **kwargs):
        raise AssertionError("capabilities must not connect to SQL")


def client(policy, authenticated=True):
    database = NoSQLDatabase()
    tasks = TaskService(database, RuntimeLimits(mode="bounded"), new_task_profile_policy=policy)

    def actor():
        if not authenticated:
            raise HTTPException(401, detail={"code": "authentication_required"})
        return Actor(uuid4())

    def no_evidence(*args):
        raise AssertionError("capabilities must not read evidence")

    app = FastAPI()
    app.include_router(build_router(tasks, MemoryService(database), actor, no_evidence, no_evidence))
    return TestClient(app)


@pytest.mark.parametrize("policy,mode,count", [
    (None, "legacy_compatibility", 5),
    (TaskProfilePolicy(release_id="explicit-http-fixture", profiles=[]), "configured_release", 0),
    (TaskProfilePolicy(release_id="explicit-http-fixture", profiles=[
        NewTaskProfile(kind="qa", profile_id="qa-v4", validation_state="development")]), "configured_release", 1),
])
def test_actual_http_capabilities_are_authenticated_and_do_not_dispatch(policy, mode, count):
    with client(policy) as session:
        response = session.get("/api/agent/capabilities")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    body = response.json()
    assert body["mode"] == mode and len(body["profiles"]) == count
    assert "dsn" not in response.text and "api_key" not in response.text


def test_actual_http_capabilities_reject_unauthenticated_request():
    with client(None, authenticated=False) as session:
        response = session.get("/api/agent/capabilities")
    assert response.status_code == 401
