"""Bounded checkpoint retention using the official PostgresSaver.

Run with ``python -m citequest.agents_v3.maintenance --config api.json``.
The default is a read-only preview; ``--apply`` explicitly enables deletion.
Only failed/cancelled tasks can expire. Successful and partial tasks retain
their checkpoints because the product accepts follow-up commands for them.
Task records, conversation events, attempts, budgets and memories are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID

from langgraph.checkpoint.postgres import PostgresSaver

from .api_config import AgentAPIConfig
from .contracts import AgentError
from .database import Database

TERMINAL_STATES = ("failed", "cancelled")
DEFAULT_RETENTION_DAYS = 30

# Inspect only checkpoint metadata. Union includes orphan blobs/writes left by
# older runtimes; all deletion still goes through the supported Saver API.
THREAD_PAGE_SQL = """
SELECT thread_id FROM (
    (SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id>%s
     ORDER BY thread_id LIMIT %s)
    UNION
    (SELECT DISTINCT thread_id FROM checkpoint_blobs WHERE thread_id>%s
     ORDER BY thread_id LIMIT %s)
    UNION
    (SELECT DISTINCT thread_id FROM checkpoint_writes WHERE thread_id>%s
     ORDER BY thread_id LIMIT %s)
) threads ORDER BY thread_id LIMIT %s
"""

ELIGIBLE_SQL = """
SELECT t.id,t.state,t.updated_at FROM cq_app.agent_tasks t
WHERE t.id=ANY(%s::uuid[]) AND t.state=ANY(%s::text[])
  AND t.updated_at<=clock_timestamp()-(%s * interval '1 day')
  AND t.running_since IS NULL AND t.lease_until IS NULL
  AND NOT (t.profile ? 'development_campaign')
  AND NOT EXISTS (SELECT 1 FROM cq_app.agent_commands c
                  WHERE c.task_id=t.id AND c.state IN ('queued','running'))
  AND NOT EXISTS (SELECT 1 FROM cq_app.agent_runs r
                  WHERE r.task_id=t.id AND r.ended_at IS NULL)
  AND NOT EXISTS (SELECT 1 FROM cq_app.agent_attempts a
                  WHERE a.task_id=t.id AND a.status IN ('reserved','sent','unknown'))
ORDER BY t.id
"""


def _task_id(thread_id: str) -> UUID | None:
    try:
        value = UUID(thread_id)
    except (ValueError, TypeError, AttributeError):
        return None
    # The graph uses canonical str(task_id), not arbitrary equivalent spellings.
    return value if str(value) == thread_id else None


def prune_checkpoints(
    database: Database,
    *,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    batch_size: int = 200,
    after_thread_id: str = "",
    apply: bool = False,
) -> dict:
    """Inspect one stable page and optionally prune its eligible task threads.

    Continue a sweep with ``next_after_thread_id``. A new sweep starts with an
    empty cursor. Metadata scans are bounded; unknown/non-Task threads are never
    deleted. Repeating an applied page is safe because delete_thread is idempotent.
    """
    if type(retention_days) is not int or not 1 <= retention_days <= 36500:
        raise AgentError("checkpoint_retention_days_invalid", 400)
    if type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise AgentError("checkpoint_retention_batch_invalid", 400)
    if not isinstance(after_thread_id, str) or len(after_thread_id) > 255 or "\x00" in after_thread_id:
        raise AgentError("checkpoint_retention_cursor_invalid", 400)
    if type(apply) is not bool:
        raise AgentError("checkpoint_retention_apply_invalid", 400)
    if database.development_campaign_policy is not None or database.development_dispatch_policy is not None:
        raise AgentError("checkpoint_retention_requires_product_configuration", 400)

    take = batch_size + 1
    with database.connect("checkpoint") as checkpoint:
        checkpoint.execute("SET TRANSACTION READ ONLY")
        page = checkpoint.execute(THREAD_PAGE_SQL, (
            after_thread_id, take, after_thread_id, take, after_thread_id, take, take,
        )).fetchall()
    threads = [row["thread_id"] for row in page[:batch_size]]
    task_ids = [value for thread_id in threads if (value := _task_id(thread_id)) is not None]
    candidates = []
    if task_ids:
        with database.connect() as business:
            business.execute("SET TRANSACTION READ ONLY")
            candidates = business.execute(
                ELIGIBLE_SQL, (task_ids, list(TERMINAL_STATES), retention_days),
            ).fetchall()

    outcomes = []
    for row in candidates:
        identifier = row["id"]
        outcome = {"task_id": str(identifier), "state": row["state"], "status": "eligible"}
        if apply:
            # Same physical-session lock as claim(): lease expiry alone cannot
            # authorize deleting a worker's Saver state.
            key = int.from_bytes(hashlib.sha256(str(identifier).encode()).digest()[:8], "big", signed=True)
            with database.connect("checkpoint", autocommit=True) as checkpoint:
                acquired = checkpoint.execute(
                    "SELECT pg_try_advisory_lock(%s) AS acquired", (key,),
                ).fetchone()["acquired"]
                if not acquired:
                    outcomes.append({**outcome, "status": "skipped_active_owner"})
                    continue
                try:
                    # Lock the task row before repeating every admission guard.
                    # Commands cannot make this task runnable during deletion.
                    with database.connect() as business:
                        business.execute("SET LOCAL lock_timeout='2s'")
                        business.execute("SELECT id FROM cq_app.agent_tasks WHERE id=%s FOR UPDATE", (identifier,))
                        eligible = business.execute(
                            ELIGIBLE_SQL, ([identifier], list(TERMINAL_STATES), retention_days),
                        ).fetchone()
                        if eligible is None:
                            outcome["status"] = "skipped_changed"
                        else:
                            # All Saver-owned checkpoint tables are deleted in
                            # one transaction; no business table is modified.
                            with checkpoint.transaction():
                                checkpoint.execute("SET LOCAL statement_timeout='10s'")
                                PostgresSaver(checkpoint).delete_thread(str(identifier))
                            outcome["status"] = "pruned"
                finally:
                    checkpoint.execute("SELECT pg_advisory_unlock(%s)", (key,))
        outcomes.append(outcome)

    return {
        "mode": "apply" if apply else "dry_run",
        "retention_days": retention_days,
        "terminal_states": list(TERMINAL_STATES),
        "scanned_threads": len(threads),
        "eligible_threads": len(candidates),
        "pruned_threads": sum(item["status"] == "pruned" for item in outcomes),
        "next_after_thread_id": threads[-1] if len(page) > batch_size else None,
        "items": outcomes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Existing Agent API configuration")
    parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--after-thread-id", default="", help="Continuation cursor from the preceding page")
    parser.add_argument("--apply", action="store_true", help="Delete eligible checkpoint threads; default is dry-run")
    args = parser.parse_args()
    try:
        config = AgentAPIConfig.model_validate_json(args.config.read_bytes(), strict=True)
        result = prune_checkpoints(config.database(), retention_days=args.retention_days,
            batch_size=args.batch_size, after_thread_id=args.after_thread_id, apply=args.apply)
    except Exception as error:
        # Connection/configuration exception messages can contain private DSNs.
        result = {"status": "failed", "error_code": error.code if isinstance(error, AgentError)
                  else "checkpoint_maintenance_failed", "automatic_retry": False}
        print(json.dumps(result), flush=True)  # noqa: T201 -- explicit operational CLI output
        return 1
    print(json.dumps(result), flush=True)  # noqa: T201 -- no prompts, sources, credentials or private configuration
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
