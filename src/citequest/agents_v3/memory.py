"""Current-version control rows + official Store, without automatic memory writes."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from uuid import UUID, uuid4

from langgraph.store.base import GetOp
from langgraph.store.postgres import PostgresStore
from psycopg.types.json import Jsonb

from .contracts import AgentError, MemoryDelete, MemoryWrite, digest
from .database import Database


def namespace(owner: UUID, project: UUID) -> tuple[str, ...]:
    return ("memory-v1", "user", str(owner), "project", str(project))


class MemoryService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def validate_result_context(self, task: dict, result: dict) -> None:
        """Use the dispatch contract again at publication/read, including pending hard writes."""
        self.validate_wire_context(task, {"input": [{"content": json.dumps({
            "context_manifest": result.get("context_manifest", {})}, ensure_ascii=False)}]})

    def validate_wire_context(self, task: dict, payload: dict) -> None:
        """Invalidate an assembled context before sending, including while it waited for a slot."""
        try:
            prompt = json.loads(payload["input"][0]["content"])
            manifest = prompt["context_manifest"]
            included = {(str(item["id"]), item["revision"]) for item in manifest["memory_refs"]}
            hard = {(str(item["id"]), item["revision"]) for item in manifest["hard_memory_refs"]}
        except (KeyError, IndexError, TypeError, ValueError):
            raise AgentError("context_memory_manifest_missing") from None
        current = [item for item in self.current(task["owner_id"], task["project_id"]) if not item.get("source_refs")]
        active = {(item["memory_id"], item["revision"]) for item in current}
        hard_now = {(item["memory_id"], item["revision"]) for item in current if item["kind"] != "preference"}
        if not included.issubset(active) or hard != hard_now:
            raise AgentError("memory_changed_before_dispatch")
        # A just-confirmed constraint must not be ignored while its Store outbox is pending.
        with self.database.connect() as db:
            self._authorize(db, task["owner_id"], task["project_id"])
            pending = db.execute("""SELECT 1 FROM cq_app.agent_memories WHERE owner_id=%s AND project_id=%s
                AND state='pending' AND kind<>'preference' AND source_refs='[]'::jsonb LIMIT 1""",
                (task["owner_id"], task["project_id"])).fetchone()
        if pending:
            raise AgentError("confirmed_memory_materialization_pending")

    @staticmethod
    def _authorize(db, owner: UUID, project: UUID) -> None:
        if not db.execute("SELECT 1 FROM cq_app.projects WHERE id=%s AND owner_id=%s", (project, owner)).fetchone():
            raise AgentError("project_not_found", 404)

    def write(self, owner: UUID, project: UUID, memory_id: UUID, request: MemoryWrite, *, delete: bool = False) -> dict:
        payload = {**request.model_dump(mode="json"), "delete": delete, "memory_id": str(memory_id), "project_id": str(project)}
        payload_hash = digest(payload)
        with self.database.connect() as db:
            self._authorize(db, owner, project)
            # A project lock also serializes creation of a previously absent memory key.
            db.execute("SELECT id FROM cq_app.projects WHERE id=%s AND owner_id=%s FOR UPDATE", (project, owner))
            existing = db.execute("SELECT * FROM cq_app.agent_memory_outbox WHERE owner_id=%s AND operation_id=%s", (owner, request.operation_id)).fetchone()
            if existing:
                if existing["payload_hash"] != payload_hash:
                    raise AgentError("idempotency_payload_conflict")
                return {"id": memory_id, "revision": existing["revision"], "state": existing["state"]}
            old = db.execute("SELECT * FROM cq_app.agent_memories WHERE owner_id=%s AND project_id=%s AND id=%s FOR UPDATE", (owner, project, memory_id)).fetchone()
            if (old["revision"] if old else 0) != request.expected_revision:
                raise AgentError("memory_revision_conflict")
            revision = request.expected_revision + 1
            value = {"kind": request.kind, "value": request.value, "source_refs": request.source_refs,
                "revision": revision, "confirmed_by": str(owner),
                "expires_at": request.expires_at.isoformat() if request.expires_at else None}
            db.execute("""INSERT INTO cq_app.agent_memories(owner_id,project_id,id,revision,kind,value_hash,state,source_refs,deletion_epoch)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(owner_id,project_id,id) DO UPDATE
                SET revision=EXCLUDED.revision,kind=EXCLUDED.kind,value_hash=EXCLUDED.value_hash,state=EXCLUDED.state,
                source_refs=EXCLUDED.source_refs,deletion_epoch=EXCLUDED.deletion_epoch,updated_at=clock_timestamp()""",
                (owner, project, memory_id, revision, request.kind, digest(value), "deleted" if delete else "pending",
                 Jsonb(request.source_refs), (old["deletion_epoch"] if old else 0) + int(delete)))
            db.execute("INSERT INTO cq_app.agent_memory_outbox(id,owner_id,project_id,memory_id,revision,operation_id,payload_hash,operation,payload) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (uuid4(), owner, project, memory_id, revision, request.operation_id, payload_hash, "delete" if delete else "put", Jsonb(value if not delete else {})))
        return {"id": memory_id, "revision": revision, "state": "deleted" if delete else "pending"}

    def delete(self, owner: UUID, project: UUID, memory_id: UUID, request: MemoryDelete) -> dict:
        with self.database.connect() as db:
            self._authorize(db, owner, project)
            row = db.execute("SELECT kind FROM cq_app.agent_memories WHERE owner_id=%s AND project_id=%s AND id=%s", (owner, project, memory_id)).fetchone()
            if row is None:
                raise AgentError("memory_not_found", 404)
        return self.write(owner, project, memory_id, MemoryWrite(operation_id=request.operation_id,
            expected_revision=request.expected_revision, kind=row["kind"], value="[delete]"), delete=True)

    def flush(self, limit: int = 20) -> int:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise AgentError("memory_flush_limit_invalid")
        with self.database.connect() as db:
            rows = db.execute("SELECT * FROM cq_app.agent_memory_outbox WHERE state='pending' ORDER BY id LIMIT %s", (limit,)).fetchall()
        completed = 0
        with self.database.connect("memory", autocommit=True) as connection:
            store = PostgresStore(connection)
            for row in rows:
                key_lock = int.from_bytes(hashlib.sha256(("memory:" + str(row["memory_id"])).encode()).digest()[:8], "big", signed=True)
                if not connection.execute("SELECT pg_try_advisory_lock(%s) AS ok", (key_lock,)).fetchone()["ok"]:
                    continue
                try:
                    with self.database.connect() as db:
                        current = db.execute("SELECT * FROM cq_app.agent_memories WHERE owner_id=%s AND project_id=%s AND id=%s", (row["owner_id"], row["project_id"], row["memory_id"])).fetchone()
                        pending = db.execute("SELECT * FROM cq_app.agent_memory_outbox WHERE id=%s AND state='pending'", (row["id"],)).fetchone()
                    if pending is None:
                        continue
                    ns = namespace(row["owner_id"], row["project_id"])
                    if row["operation"] == "put" and current["state"] != "deleted" and current["revision"] == row["revision"]:
                        key = str(row["memory_id"]) + ":" + str(row["revision"])
                        old = store.get(ns, key)
                        if old is not None and digest(old.value) != digest(row["payload"]):
                            raise AgentError("memory_version_hash_conflict")
                        store.put(ns, key, row["payload"])
                    elif row["operation"] == "delete":
                        # Serialized on the same physical Store session, including stale outbox replay.
                        for version in range(1, row["revision"]):
                            store.delete(ns, str(row["memory_id"]) + ":" + str(version))
                    with self.database.connect() as db:
                        db.execute("UPDATE cq_app.agent_memory_outbox SET state='done',payload='{}' WHERE id=%s AND state='pending'", (row["id"],))
                        if row["operation"] == "put":
                            db.execute("UPDATE cq_app.agent_memories SET state='active' WHERE owner_id=%s AND project_id=%s AND id=%s AND revision=%s AND state='pending'", (row["owner_id"], row["project_id"], row["memory_id"], row["revision"]))
                    completed += 1
                finally:
                    connection.execute("SELECT pg_advisory_unlock(%s)", (key_lock,))
        return completed

    def current(self, owner: UUID, project: UUID) -> list[dict]:
        with self.database.connect() as db:
            self._authorize(db, owner, project)
            rows = db.execute("SELECT * FROM cq_app.agent_memories WHERE owner_id=%s AND project_id=%s AND state='active' ORDER BY id LIMIT 101", (owner, project)).fetchall()
            if len(rows) > 100:
                raise AgentError("memory_catalog_limit_requires_cleanup", 413)
        values = []
        if rows:
            with self.database.connect("memory", autocommit=True) as connection:
                store = PostgresStore(connection)
                # The official adapter groups keys in one namespace into one
                # SQL query and restores operation order after reading the rows.
                items = store.batch([GetOp(namespace(owner, project), str(row["id"]) + ":" + str(row["revision"]))
                    for row in rows])
                for row, item in zip(rows, items, strict=True):
                    if item is None or digest(item.value) != row["value_hash"]:
                        raise AgentError("memory_materialization_unavailable", 503)
                    values.append({"memory_id": str(row["id"]), "revision": row["revision"], **item.value})
        # Recheck after Store IO: a concurrent correction/delete must not inject a stale value.
        with self.database.connect() as db:
            self._authorize(db, owner, project)
            database_now = db.execute("SELECT clock_timestamp() AS now").fetchone()["now"]
            active = {(str(r["id"]), r["revision"]) for r in db.execute("SELECT id,revision FROM cq_app.agent_memories WHERE owner_id=%s AND project_id=%s AND state='active'", (owner, project))}
        return [value for value in values if (value["memory_id"], value["revision"]) in active
            and (value.get("expires_at") is None or datetime.fromisoformat(value["expires_at"]) > database_now)]
