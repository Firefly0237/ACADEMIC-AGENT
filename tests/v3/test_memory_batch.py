"""Exercise the official Store's batched reads without a database service."""
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest
from langgraph.store.postgres import PostgresStore

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.memory import MemoryService


@pytest.mark.parametrize("scenario", ["current", "missing", "changed", "expired", "deleted", "empty"])
def test_current_memory_uses_one_store_query_and_preserves_freshness_guards(monkeypatch, scenario):
    owner, project = UUID(int=1), UUID(int=2)
    now = datetime(2026, 11, 8, tzinfo=UTC)
    rows, stored = [], []
    for index in range(0 if scenario == "empty" else 100):
        identifier = UUID(int=index + 10)
        value = {"kind": "constraint", "value": f"Synthetic constraint {index}",
                 "source_refs": [], "revision": 1, "confirmed_by": str(owner),
                 "expires_at": (now - timedelta(seconds=1)).isoformat()
                 if scenario == "expired" and index == 0 else None}
        rows.append({"id": identifier, "revision": 1, "value_hash": digest(value)})
        if scenario == "missing" and index == 0:
            continue
        stored.append({"key": str(identifier) + ":1", "value": {**value,
            **({"value": "Changed after the control row was read"} if scenario == "changed" and index == 0 else {})},
            "created_at": now, "updated_at": now})

    query_calls, memory_connections, authorizations = [], [], []

    class Business:
        def execute(self, sql, parameters=()):
            if sql.startswith("SELECT 1 FROM cq_app.projects"):
                assert parameters == (project, owner)
                authorizations.append(parameters)
                return SimpleNamespace(fetchone=lambda: {"authorized": True})
            if sql.startswith("SELECT * FROM cq_app.agent_memories"):
                return SimpleNamespace(fetchall=lambda: rows)
            if sql == "SELECT clock_timestamp() AS now":
                return SimpleNamespace(fetchone=lambda: {"now": now})
            if sql.startswith("SELECT id,revision FROM cq_app.agent_memories"):
                return rows[1:] if scenario == "deleted" else rows
            pytest.fail("Unexpected business SQL")

    class Cursor:
        def execute(self, sql, parameters):
            query_calls.append((sql, parameters))
            return self

        def fetchall(self):
            # PostgreSQL does not promise row ordering for this query. The
            # official adapter must associate each item with its requested key.
            return list(reversed(stored))

    @contextmanager
    def cursor(_store, *, pipeline=False):
        assert pipeline
        yield Cursor()

    @contextmanager
    def connect(role="business", *, autocommit=False):
        if role == "memory":
            assert autocommit
            memory_connections.append(role)
            yield object()
        else:
            assert role == "business"
            yield Business()

    monkeypatch.setattr(PostgresStore, "_cursor", cursor)
    service = MemoryService(SimpleNamespace(connect=connect))
    if scenario in {"missing", "changed"}:
        with pytest.raises(AgentError, match="memory_materialization_unavailable") as caught:
            service.current(owner, project)
        assert caught.value.status == 503
        assert len(authorizations) == 1
    else:
        result = service.current(owner, project)
        expected = rows[1:] if scenario in {"expired", "deleted"} else rows
        assert [item["memory_id"] for item in result] == [str(row["id"]) for row in expected]
        assert len(authorizations) == 2
    assert len(memory_connections) == len(query_calls) == (0 if scenario == "empty" else 1)
    if query_calls:
        assert query_calls[0][1][0] == [str(row["id"]) + ":1" for row in rows]
