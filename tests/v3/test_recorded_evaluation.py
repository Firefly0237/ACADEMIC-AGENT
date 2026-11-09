"""Synthetic pure adapter checks; no official evaluator or real benchmark run."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from citequest.evaluation.recorded import (
    CASE_VERSION, Protector, canonical, evaluate_records, project_command, tool_messages, validate_cases,
)


def sample(*, state="done", status="settled", answer=True):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    command = {"id": UUID(int=1), "task_id": UUID(int=2), "seq": 1, "state": state, "created_at": now}
    attempts = [{"id": UUID(int=value), "kind": "tool", "status": status, "reserved_ncny": 0,
        "cost_ncny": 0 if status == "settled" else None, "sent_at": now + timedelta(seconds=value),
        "settled_at": now + timedelta(seconds=value + 1) if status == "settled" else None,
        "tool_name": "search_evidence", "tool_arguments": {"query": "private query", "headers": {"key": "private secret"}},
        "usage": None, "error_code": None} for value in (3, 4)]
    events = [{"attempt_id": str(UUID(int=3)), "seq": 20}, {"attempt_id": str(UUID(int=4)), "seq": 10}]
    completion = [{"command_id": str(command["id"]), "command_seq": 1, "state": "succeeded",
        "created_at": now + timedelta(seconds=8), "answer": {"text": "private answer",
            "publication_status": "validated_ai_review", "citations": [{"ref_id": "private:reference", "quote": "private literal"}]}}] if answer else []
    return command, attempts, events, completion, {"private:reference": {"text": "This is private literal evidence"}}


def test_projection_hides_raw_data_and_preserves_real_reservation_order():
    protector = Protector(b"x" * 32)
    command, attempts, events, completion, evidence = sample()
    value = project_command(command, attempts, events, completion, evidence, {"queue_seconds": 1}, protector)
    encoded = canonical(value).decode()
    for private in ("private query", "private secret", "private answer", "private literal", "private:reference", str(command["id"]), "headers"):
        assert private not in encoded
    assert [item["reservation_seq"] for item in value["tool_calls"]] == [10, 20]
    messages = tool_messages(value["tool_calls"])
    assert len(messages) == 2 and all(len(item["tool_calls"]) == 1 for item in messages)
    assert value["citations"]["historical_literal"] == "matched"
    assert value["citations"]["semantic_support"] == "not_checked"
    assert value["citations"]["current_source_authorization"] == "not_checked"
    assert value["latency"]["command_wall_seconds"] == 8


def test_unknown_cost_stays_null_and_unsent_tools_are_not_trajectory():
    protector = Protector(b"x" * 32)
    command, attempts, events, completion, evidence = sample(status="unknown")
    attempts[1].update(status="not_sent", sent_at=None, cost_ncny=0)
    value = project_command(command, attempts, events, completion, evidence, None, protector)
    assert len(value["tool_calls"]) == 1
    assert value["cost"]["unknown_attempts"] == 1
    assert value["cost"]["actual_total_cash_ncny"] is None
    assert value["cost"]["infrastructure_cash_ncny"] is None


def test_rejected_and_failed_runs_are_observed_without_answer():
    protector = Protector(b"x" * 32)
    command, _, _, _, _ = sample(state="rejected", answer=False)
    rejected = project_command(command, [], [], [], {}, None, protector)
    assert rejected["outcome_observed"] and rejected["result"] == {"command_outcome": "rejected"}
    assert "publication_status" not in rejected["result"]
    command["state"] = "done"
    run = {"state": "failed", "started_at": command["created_at"], "ended_at": command["created_at"] + timedelta(seconds=1)}
    failed = project_command(command, [], [], [], {}, None, protector, runs=[run])
    assert failed["outcome_observed"] and failed["command_outcome"] == "terminal_failure"


def test_missing_or_synthetic_cases_return_no_data_without_loading_evaluators(monkeypatch):
    protector = Protector(b"x" * 32)
    case = {"command_hmac": protector.token("command", "missing"), "source": "actual", "expected_tool_calls": [],
        "expected_result": {"publication_status": "validated_ai_review"}, "require_trajectory_match": True,
        "require_literal_citations": False, "exclude_if": []}
    cases = validate_cases({"schema_version": CASE_VERSION, "key_id": protector.key_id, "cases": [case]}, protector.key_id)
    import citequest.evaluation.recorded as recorded
    monkeypatch.setattr(recorded.importlib.metadata, "version", lambda *_: pytest.fail("loaded unavailable evaluator for no-data case"))
    value = evaluate_records([], cases)
    assert value["status"] == "no_data" and value["success_rate"] is None
    assert value["coverage"]["gold_without_record"] == 1


def test_signed_export_rejects_modified_observations():
    protector = Protector(b"x" * 32)
    signed = protector.seal([{"command_hmac": protector.token("command", "sample")}])
    assert protector.verify(signed) == signed["records"]
    signed["records"][0]["result"] = {"publication_status": "validated_ai_review"}
    with pytest.raises(ValueError, match="integrity"):
        protector.verify(signed)
