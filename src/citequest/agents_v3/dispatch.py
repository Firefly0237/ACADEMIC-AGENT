"""Versioned PostgreSQL resource admission; LangGraph remains the sole workflow.

No provider call or financial mutation occurs here. Model attempts still enter
through Execution.reserve and its existing campaign/owner/project/Task locks.
"""
from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from .contracts import AgentError, digest


class DispatchPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    version: Literal["cq.agent-dispatch-fifo/1"] = "cq.agent-dispatch-fifo/1"
    model_concurrency: Literal[1] = 1
    research_active: Literal[1] = 1
    research_queued: Literal[2] = 2
    research_queue_seconds: Literal[300] = 300
    short_kinds: tuple[Literal["qa"]] = ("qa",)
    long_kinds: tuple[Literal["summary"], Literal["compare"], Literal["stance"], Literal["research"]] = ("summary", "compare", "stance", "research")

    def fingerprint(self) -> str:
        return digest(self.model_dump(mode="json"))


def active_policy(db, configured: DispatchPolicy | None) -> DispatchPolicy | None:
    exists = db.execute("SELECT to_regclass('cq_app.agent_dispatch_control') AS name").fetchone()["name"]
    if exists is None:
        if configured is not None:
            raise AgentError("dispatch_schema_not_installed", 503)
        return None
    row = db.execute("SELECT enabled,policy,policy_sha256 FROM cq_app.agent_dispatch_control WHERE singleton").fetchone()
    if not row or not row["enabled"]:
        if configured is not None:
            raise AgentError("dispatch_policy_not_enabled", 503)
        return None
    if (configured is None or row["policy_sha256"] != configured.fingerprint()
            or row["policy"] != configured.model_dump(mode="json")):
        raise AgentError("dispatch_policy_binding_required", 503)
    return configured


def assert_checkpoint_binding(db, checkpoint) -> None:
    """Prove the actual sessions share one PostgreSQL database and cluster.

    DSN host/name comparisons cannot establish this behind proxies. A random
    short-lived checkpoint session lock must be visible (and unacquirable) on
    the business connection used to inspect the ownership locks.
    """
    name = db.execute("SELECT current_database() AS name").fetchone()["name"]
    if checkpoint.execute("SELECT current_database() AS name").fetchone()["name"] != name:
        raise AgentError("dispatch_checkpoint_database_binding_required", 503)
    key = int.from_bytes(uuid4().bytes[:8], "big", signed=True)
    acquired = checkpoint.execute("SELECT pg_try_advisory_lock(%s) AS ok", (key,)).fetchone()["ok"]
    if not acquired:
        raise AgentError("dispatch_checkpoint_binding_probe_busy", 503)
    try:
        if db.execute("SELECT pg_try_advisory_xact_lock(%s) AS ok", (key,)).fetchone()["ok"]:
            raise AgentError("dispatch_checkpoint_database_binding_required", 503)
    finally:
        checkpoint.execute("SELECT pg_advisory_unlock(%s)", (key,))


def queue_seconds(db, task_id) -> float:
    return float(db.execute("SELECT coalesce(sum(queue_seconds),0) AS seconds FROM cq_app.agent_command_queues WHERE task_id=%s", (task_id,)).fetchone()["seconds"])


def target_elapsed(db, task: dict) -> float:
    return float(task["elapsed"]) + (queue_seconds(db, task["id"]) if task["kind"] == "qa" else 0.0)


def check_target(db, task: dict, policy: DispatchPolicy | None) -> None:
    if policy is not None and task["kind"] == "qa" and target_elapsed(db, task) >= task["profile"]["active_seconds"]:
        raise AgentError("target_deadline_exhausted")


def research_admit(db, kind: str, policy: DispatchPolicy | None) -> None:
    if policy is not None and kind == "research":
        count = db.execute("SELECT count(*) AS count FROM cq_app.agent_tasks WHERE kind='research' AND state='queued'").fetchone()["count"]
        if count >= policy.research_queued:
            raise AgentError("research_queue_capacity", 429)


def can_claim(db, task: dict, policy: DispatchPolicy | None) -> bool:
    if policy is None or task["kind"] != "research":
        return True
    count = db.execute("SELECT count(*) AS count FROM cq_app.agent_tasks WHERE kind='research' AND state IN ('running','reconciling') AND id<>%s", (task["id"],)).fetchone()["count"]
    return count < policy.research_active


def record_queue_once(db, command: dict, task: dict, policy: DispatchPolicy | None) -> dict | None:
    if policy is None:
        return None
    # A running command recovery must not create a fresh queued interval. The
    # first policy-aware claim supplies this append-only receipt exactly once.
    return db.execute("""WITH observed AS MATERIALIZED(SELECT clock_timestamp() AS at)
        INSERT INTO cq_app.agent_command_queues(command_id,task_id,queued_at,claimed_at,queue_seconds)
        SELECT c.id,c.task_id,c.created_at,o.at,greatest(0,extract(epoch FROM o.at-c.created_at))
        FROM cq_app.agent_commands c CROSS JOIN observed o WHERE c.id=%s AND c.task_id=%s
        ON CONFLICT(command_id) DO NOTHING RETURNING *""", (command["id"], task["id"])).fetchone()


def command_wait(db, command: dict) -> float:
    value = db.execute("""SELECT CASE WHEN EXISTS(SELECT 1 FROM cq_app.agent_command_queues q WHERE q.command_id=c.id)
        THEN 0 ELSE greatest(0,extract(epoch FROM clock_timestamp()-c.created_at)) END AS seconds
        FROM cq_app.agent_commands c WHERE c.id=%s""", (command["id"],)).fetchone()
    return float(value["seconds"])


def expire_queued(db, policy: DispatchPolicy | None) -> list[dict]:
    if policy is None:
        return []
    # The caller owns the common wallet lock. No active/unknown attempt is
    # refunded or changed; this applies only to unclaimed queued commands.
    rows = db.execute("""SELECT t.id,c.id AS command_id,t.kind FROM cq_app.agent_tasks t
        JOIN cq_app.agent_commands c ON c.task_id=t.id AND c.state='queued'
        WHERE t.state='queued' AND NOT EXISTS(SELECT 1 FROM cq_app.agent_attempts a
          WHERE a.task_id=t.id AND a.status IN ('reserved','sent','unknown'))
        AND ((t.kind='research' AND clock_timestamp()-c.created_at>=interval '300 seconds')
          OR (t.kind='qa' AND t.active_seconds+coalesce((SELECT sum(q.queue_seconds) FROM cq_app.agent_command_queues q WHERE q.task_id=t.id),0)
             +CASE WHEN EXISTS(SELECT 1 FROM cq_app.agent_command_queues current_q WHERE current_q.command_id=c.id)
                 THEN 0 ELSE greatest(0,extract(epoch FROM clock_timestamp()-c.created_at)) END
              >= (t.profile->>'active_seconds')::double precision))
        ORDER BY c.created_at,c.seq FOR UPDATE OF t,c""").fetchall()
    for row in rows:
        code = "research_queue_deadline" if row["kind"] == "research" else "target_queue_deadline"
        # Expired requests must expose their actual wait, rather than looking
        # like zero-latency failures merely because no worker was admitted.
        record_queue_once(db, {"id": row["command_id"]}, {"id": row["id"]}, policy)
        db.execute("UPDATE cq_app.agent_tasks SET state='failed',error_code=%s WHERE id=%s", (code, row["id"]))
        db.execute("UPDATE cq_app.agent_commands SET state='rejected' WHERE id=%s", (row["command_id"],))
        row["error_code"] = code
    return rows


def enqueue(db, task: dict, command: dict, run_id, epoch: int, logical_key: str, request: dict) -> dict:
    if not logical_key.startswith(str(command["id"]) + ":"):
        raise AgentError("model_waiter_command_identity_invalid")
    old = db.execute("SELECT * FROM cq_app.agent_model_waiters WHERE task_id=%s AND logical_key=%s FOR UPDATE", (task["id"], logical_key)).fetchone()
    hashed = digest(request)
    if old:
        if old["request_hash"] != hashed or old["command_id"] != command["id"]:
            raise AgentError("model_waiter_replay_request_changed")
        if old["state"] != "waiting":
            raise AgentError("model_waiter_not_waiting")
        if old["epoch"] > epoch:
            raise AgentError("model_waiter_owner_changed")
        return db.execute("UPDATE cq_app.agent_model_waiters SET run_id=%s,epoch=%s WHERE id=%s RETURNING *", (run_id, epoch, old["id"])).fetchone()
    remaining = task["profile"]["active_seconds"] - target_elapsed(db, task)
    if remaining <= 0:
        raise AgentError("target_deadline_exhausted")
    return db.execute("""INSERT INTO cq_app.agent_model_waiters(task_id,command_id,run_id,epoch,logical_key,request_hash,expires_at)
        VALUES(%s,%s,%s,%s,%s,%s,clock_timestamp()+%s*interval '1 second') RETURNING *""",
        (task["id"], command["id"], run_id, epoch, logical_key, hashed, remaining)).fetchone()


def require_turn(db, task: dict, logical_key: str, request: dict, run_id, epoch: int) -> dict:
    row = db.execute("SELECT * FROM cq_app.agent_model_waiters WHERE task_id=%s AND logical_key=%s FOR UPDATE", (task["id"], logical_key)).fetchone()
    if not row or row["state"] != "waiting" or row["run_id"] != run_id or row["epoch"] != epoch or row["request_hash"] != digest(request):
        raise AgentError("model_waiter_current_owner_required")
    head = db.execute("SELECT cq_app.agent_dispatch_head() AS id").fetchone()["id"]
    if head != row["id"]:
        raise AgentError("global_model_dispatch_busy", 429)
    return row


def close_waiters(db, task_id, run_id, epoch: int, reason: str) -> None:
    # No attempts, cash or usage are touched. Exact run/epoch plus the current
    # Task owner fence prevents a late worker cancelling a successor's ticket.
    db.execute("""UPDATE cq_app.agent_model_waiters w SET state='cancelled',reason=%s
        WHERE w.task_id=%s AND w.run_id=%s AND w.epoch=%s AND w.state='waiting'
          AND EXISTS(SELECT 1 FROM cq_app.agent_tasks t WHERE t.id=w.task_id AND t.epoch=w.epoch)""",
        (reason, task_id, run_id, epoch))


def public_timing(db, task: dict, policy: DispatchPolicy | None) -> dict | None:
    if policy is None:
        return None
    # One SQL snapshot for persisted + still-unmeasured current command. This
    # is a read projection only; no new queue receipt or budget debit occurs.
    timing = db.execute("""WITH observed AS MATERIALIZED(SELECT clock_timestamp() AS at)
        SELECT coalesce((SELECT sum(q.queue_seconds) FROM cq_app.agent_command_queues q WHERE q.task_id=t.id),0) AS persisted,
          CASE WHEN t.state='queued' THEN coalesce((SELECT greatest(0,extract(epoch FROM o.at-c.created_at))
           FROM cq_app.agent_commands c WHERE c.task_id=t.id AND c.seq=t.command_seq AND c.state='queued'
           AND NOT EXISTS(SELECT 1 FROM cq_app.agent_command_queues q WHERE q.command_id=c.id)),0) ELSE 0 END AS current,
          t.active_seconds+CASE WHEN t.running_since IS NULL THEN 0 ELSE greatest(0,extract(epoch FROM o.at-t.running_since)) END AS execution
        FROM cq_app.agent_tasks t CROSS JOIN observed o WHERE t.id=%s""", (task["id"],)).fetchone()
    persisted, current, execution = (float(timing[key]) for key in ("persisted", "current", "execution"))
    queued = persisted + current
    return {"policy_version": policy.version, "queue_seconds": queued, "persisted_queue_seconds": persisted,
            "current_queue_seconds": current, "execution_seconds": execution,
            "target_total_seconds": execution + queued, "human_waiting_input_excluded": True}
