"""Explicit owner/task-only export in one read-only repeatable-read snapshot."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from uuid import UUID

from .recorded import Protector, project_command


def export_recorded(dsn_file: Path, owner_id: UUID, task_ids: list[UUID], protector: Protector) -> dict:
    if not 1 <= len(task_ids) <= 50 or len(set(task_ids)) != len(task_ids):
        raise ValueError("one_to_fifty_explicit_distinct_owned_task_ids_required")
    if dsn_file.is_symlink() or not dsn_file.is_file() or dsn_file.stat().st_size > 65536:
        raise ValueError("bounded_private_database_dsn_file_required")
    import psycopg
    from psycopg.rows import dict_row

    dsn = dsn_file.read_text(encoding="utf8").strip().replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(dsn, row_factory=dict_row, connect_timeout=5, prepare_threshold=0) as db:
        db.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        db.execute("SET LOCAL statement_timeout = '15s'")
        owned = db.execute("SELECT id FROM cq_app.agent_tasks WHERE owner_id=%s AND id=ANY(%s)",
            (owner_id, task_ids)).fetchall()
        if {row["id"] for row in owned} != set(task_ids):
            raise ValueError("explicit_evaluation_task_missing_or_not_owned")
        commands = db.execute("""SELECT c.id,c.task_id,c.seq,c.state,c.created_at
            FROM cq_app.agent_commands c JOIN cq_app.agent_tasks t ON t.id=c.task_id
            WHERE t.owner_id=%s AND c.owner_id=%s AND t.id=ANY(%s) AND c.kind<>'cancel'
            ORDER BY c.task_id,c.seq LIMIT 10001""", (owner_id, owner_id, task_ids)).fetchall()
        runs = db.execute("""SELECT r.id,r.command_id,r.task_id,r.state,r.started_at,r.ended_at
            FROM cq_app.agent_runs r JOIN cq_app.agent_tasks t ON t.id=r.task_id
            WHERE t.owner_id=%s AND t.id=ANY(%s) ORDER BY r.started_at,r.id LIMIT 50001""",
            (owner_id, task_ids)).fetchall()
        attempts = db.execute("""SELECT a.id,a.task_id,r.command_id,a.kind,a.status,a.reserved_ncny,a.cost_ncny,
                a.usage,a.error_code,a.sent_at,a.settled_at,
                CASE WHEN a.kind='tool' THEN a.request->>'name' END AS tool_name,
                CASE WHEN a.kind='tool' THEN a.request->'arguments' END AS tool_arguments
            FROM cq_app.agent_attempts a JOIN cq_app.agent_runs r ON r.id=a.run_id AND r.task_id=a.task_id
            JOIN cq_app.agent_tasks t ON t.id=a.task_id
            WHERE t.owner_id=%s AND t.id=ANY(%s) LIMIT 50001""", (owner_id, task_ids)).fetchall()
        reserved = db.execute("""SELECT e.task_id,e.seq,e.payload->>'attempt_id' AS attempt_id
            FROM cq_app.agent_events e JOIN cq_app.agent_tasks t ON t.id=e.task_id
            WHERE t.owner_id=%s AND t.id=ANY(%s) AND e.event_type='attempt_reserved'
            ORDER BY e.task_id,e.seq LIMIT 50001""", (owner_id, task_ids)).fetchall()
        completions = db.execute("""SELECT e.task_id,e.seq,e.created_at,e.payload->>'command_id' AS command_id,
                (e.payload->>'command_seq')::bigint AS command_seq,e.payload->>'state' AS state,
                e.payload->'answer' AS answer
            FROM cq_app.agent_events e JOIN cq_app.agent_tasks t ON t.id=e.task_id
            WHERE t.owner_id=%s AND t.id=ANY(%s) AND e.event_type='conversation_turn_completed'
            ORDER BY e.task_id,e.seq LIMIT 10001""", (owner_id, task_ids)).fetchall()
        queues = db.execute("""SELECT q.command_id,q.queue_seconds FROM cq_app.agent_command_queues q
            JOIN cq_app.agent_tasks t ON t.id=q.task_id
            WHERE t.owner_id=%s AND t.id=ANY(%s) LIMIT 10001""", (owner_id, task_ids)).fetchall()
        if len(commands) > 10000 or len(runs) > 50000 or len(attempts) > 50000 or len(reserved) > 50000 or len(completions) > 10000 or len(queues) > 10000:
            raise ValueError("evaluation_export_scope_too_large_choose_fewer_tasks")
        cited = {item.get("ref_id") for row in completions if isinstance(row.get("answer"), dict)
            for item in row["answer"].get("citations", []) if isinstance(item, dict) and isinstance(item.get("ref_id"), str)}
        evidence = db.execute("""SELECT e.task_id,e.ref_id,e.payload->>'text' AS text
            FROM cq_app.agent_evidence_refs e JOIN cq_app.agent_tasks t ON t.id=e.task_id
            WHERE t.owner_id=%s AND t.id=ANY(%s) AND e.ref_id=ANY(%s) LIMIT 100001""",
            (owner_id, task_ids, sorted(cited))).fetchall() if cited else []
        if len(evidence) > 100000:
            raise ValueError("evaluation_citation_scope_too_large")
        queue_by_command = {row["command_id"]: row for row in queues}
        attempt_by_command, run_by_command = defaultdict(list), defaultdict(list)
        reservation_by_attempt, completion_by_command = defaultdict(list), defaultdict(list)
        evidence_by_task = defaultdict(dict)
        for row in attempts:
            attempt_by_command[row["command_id"]].append(row)
        for row in runs:
            run_by_command[row["command_id"]].append(row)
        for row in reserved:
            reservation_by_attempt[(row["task_id"], row["attempt_id"])].append(row)
        for row in completions:
            completion_by_command[(row["task_id"], row["command_id"])].append(row)
        for row in evidence:
            evidence_by_task[row["task_id"]][row["ref_id"]] = {"text": row["text"]}
        records = []
        for command in commands:
            selected = attempt_by_command[command["id"]]
            records.append(project_command(command, selected,
                [event for row in selected for event in reservation_by_attempt[(command["task_id"], str(row["id"]))]],
                completion_by_command[(command["task_id"], str(command["id"]))], evidence_by_task[command["task_id"]],
                queue_by_command.get(command["id"]), protector,
                runs=run_by_command[command["id"]]))
        return protector.seal(records)
