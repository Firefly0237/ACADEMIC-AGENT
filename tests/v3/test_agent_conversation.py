"""Conversation policy with explicit SQL/source doubles; not a PG proof."""
import copy
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from citequest.agents_v3.contracts import AgentError
from citequest.agents_v3.conversation import VERSION, ConversationReader
from citequest.v2.domain import Actor


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class ExplicitDatabase:
    catalog_scope_policy = None

    def __init__(self, task, commands, records):
        self.task, self.commands, self.records = task, commands, records
        self.reads = 0

    @contextmanager
    def connect(self):
        self.reads += 1
        yield self

    def execute(self, query, params=()):
        if query == "SET TRANSACTION READ ONLY":
            return Rows([])
        if "JOIN cq_app.projects" in query:
            return Rows([self.task] if params == (self.task["id"], self.task["owner_id"]) else [])
        if "FROM cq_app.agent_commands" in query:
            task_id, owner, after, count = params
            assert task_id == self.task["id"] and owner == self.task["owner_id"]
            return Rows([row for row in self.commands if row["seq"] > after and row["kind"] != "cancel"][:count])
        if "FROM cq_app.agent_events" in query:
            return Rows([{"payload": row} for row in self.records if row["command_id"] in params[1]])
        raise AssertionError("unexpected IO in conversation fixture")


def setup(monkeypatch, count=6):
    from citequest.agents_v3 import user_skills
    monkeypatch.setattr(user_skills, "validate_task_skills", lambda *args: None)
    owner, project, identifier = uuid4(), uuid4(), uuid4()
    task = {"id": identifier, "owner_id": owner, "project_id": project, "kind": "qa",
            "revision": count, "brief": {"question": "current question"},
            "result": {"text": "CURRENT RESULT MUST NOT BECOME A HISTORICAL ANSWER"}}
    commands, records = [], []
    for i in range(1, count + 1):
        command = uuid4()
        question = "Explicit user fixture " + str(i)
        commands.append({"id": command, "seq": i, "kind": "start" if i == 1 else "followup",
            "payload": {"brief": {"question": question}} if i == 1 else {"message": question},
            "state": "done", "created_at": "2026-11-06T00:00:00Z"})
        answer = {"text": "Explicit retained answer " + str(i), "publication_status": "validated_ai_review",
            "context_manifest": {"memory_refs": [], "hard_memory_refs": [],
                                 "evidence_refs": [{"ref_id": "issued:" + str(i)}]},
            "citations": [{"ref_id": "issued:" + str(i)}]}
        records.append({"version": VERSION, "command_id": str(command), "command_seq": i,
            "brief_revision": i, "brief": {"question": question}, "answer": answer,
            "state": "succeeded", "error_code": None})
    database = ExplicitDatabase(task, commands, records)
    service = SimpleNamespace(database=database, public=lambda value, **kwargs: copy.deepcopy(value))
    memory = SimpleNamespace(current=lambda *args: [], validate_result_context=lambda *args: None)
    return ConversationReader(service, memory, lambda *args: None), Actor(owner), identifier, database


def test_six_turns_paginate_without_four_turn_truncation_or_current_answer_substitution(monkeypatch):
    reader, actor, identifier, _ = setup(monkeypatch)
    first = reader.page(actor, identifier, limit=3)
    second = reader.page(actor, identifier, after_seq=first["next_after_seq"], limit=3)
    assert [row["command_seq"] for row in first["items"] + second["items"]] == list(range(1, 7))
    assert first["next_after_seq"] == 3 and second["next_after_seq"] is None
    assert second["items"][-1]["answer"]["text"] == "Explicit retained answer 6"
    assert all(row["answer_availability"] == "available" for row in first["items"])


def test_legacy_answers_without_paired_brief_are_explicitly_not_retained(monkeypatch):
    reader, actor, identifier, database = setup(monkeypatch)
    database.records.clear()
    page = reader.page(actor, identifier)
    assert len(page["items"]) == 6
    assert all(row["answer"] is None and row["answer_availability"] == "not_retained" for row in page["items"])


def test_current_evidence_revocation_hides_historical_answer_on_each_read(monkeypatch):
    reader, actor, identifier, _ = setup(monkeypatch, count=1)
    assert reader.page(actor, identifier)["items"][0]["answer"] is not None
    def deny(*args):
        raise AgentError("explicit_source_revocation_fixture", 403)
    reader.reader.authorize_evidence = deny
    row = reader.page(actor, identifier)["items"][0]
    assert row["answer"] is None and row["answer_availability"] == "evidence_unavailable_or_access_revoked"
    assert row["user_text"] == "Explicit user fixture 1"


def test_each_answer_is_projected_with_its_original_brief_not_latest_question(monkeypatch):
    reader, actor, identifier, _ = setup(monkeypatch, count=2)
    seen = []
    reader.reader.memory.validate_result_context = lambda task, result: seen.append(task["brief"]["question"])
    reader.page(actor, identifier)
    assert seen == ["Explicit user fixture 1", "Explicit user fixture 2"]


def test_wrong_owner_and_ambiguous_or_changed_bindings_fail_closed(monkeypatch):
    reader, actor, identifier, database = setup(monkeypatch, count=2)
    with pytest.raises(AgentError) as error:
        reader.page(Actor(uuid4()), identifier)
    assert error.value.status == 404
    database.records.append(copy.deepcopy(database.records[0]))
    database.records[1]["brief"]["question"] = "a different user turn"
    rows = reader.page(actor, identifier)["items"]
    assert [row["answer_availability"] for row in rows] == ["ambiguous_turn_record", "invalid_turn_binding"]
    assert all(row["answer"] is None for row in rows)


@pytest.mark.parametrize("after,limit", [(-1, 20), (True, 20), (0, 0), (0, 51)])
def test_invalid_page_bounds_never_read_storage(monkeypatch, after, limit):
    reader, actor, identifier, database = setup(monkeypatch)
    with pytest.raises(AgentError):
        reader.page(actor, identifier, after_seq=after, limit=limit)
    assert database.reads == 0


def test_real_http_conversation_serializes_page_and_private_cache_policy(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from citequest.agents_v3.api import build_router

    reader, actor, identifier, database = setup(monkeypatch)
    app = FastAPI()
    app.include_router(build_router(reader.service, reader.reader.memory, lambda: actor, lambda *args: None,
                                    lambda *args: None))
    with TestClient(app) as client:
        response = client.get(f"/api/agent/tasks/{identifier}/conversation?limit=2")
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store" and response.headers["Vary"] == "Cookie"
        assert response.json()["next_after_seq"] == 2
        before = database.reads
        assert client.get(f"/api/agent/tasks/{identifier}/conversation?limit=51").status_code == 422
        assert database.reads == before


def test_real_http_conversation_requires_authentication_before_storage(monkeypatch):
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient

    from citequest.agents_v3.api import build_router

    reader, _, identifier, database = setup(monkeypatch)
    def unauthenticated():
        raise HTTPException(401, detail={"code": "authentication_required"})
    app = FastAPI()
    app.include_router(build_router(reader.service, reader.reader.memory, unauthenticated,
                                    lambda *args: None, lambda *args: None))
    with TestClient(app) as client:
        assert client.get(f"/api/agent/tasks/{identifier}/conversation").status_code == 401
    assert database.reads == 0
