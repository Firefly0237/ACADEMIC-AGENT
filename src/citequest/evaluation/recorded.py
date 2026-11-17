"""Pure, HMAC-protected projection and official deterministic evaluators."""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import math
import re
from datetime import datetime

VERSION = "citequest-recorded-agent-evaluation/1"
CASE_VERSION = "citequest-agent-evaluation-cases/1"
TOKEN = re.compile(r"^[a-z]+_[a-f0-9]{64}$")
RESULT_FIELDS = frozenset({"publication_status", "status", "completion_status", "request_coverage",
    "evidence_status", "school_format_ready", "submission_ready", "claim_count", "citation_count",
    "school_section_count", "review_finding_count", "adopted_findings_resolved", "result_hmac", "command_outcome"})
PUBLIC_VALUES = frozenset({"validated_ai_review", "not_evaluated", "rejected", "complete", "partial", "none",
    "supported", "insufficient", "school_document_ready", "research_review_ready", "research_review_partial",
    "research_revision_partial", "experiment_design_ready", "structured_plan_ready", "failed",
    "partial_request_coverage", "partial_named_source_coverage", "bounded_windows_evaluated",
    "school_draft_ready_export_not_yet_verified", "partial_research_workflow", "structured_research_plan_ready",
    "partial_research_planning", "experiment_design_ready_current_evidence_only", "completed", "rejected",
    "terminal_failure", "cancelled", "pending", "waiting_input"})
ATTEMPT_STATES = frozenset({"reserved", "sent", "settled", "unknown", "not_sent"})
COMMAND_STATES = frozenset({"queued", "running", "done", "rejected"})
TASK_STATES = frozenset({"queued", "running", "waiting_input", "succeeded", "partial", "failed", "cancelled", "reconciling"})


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Protector:
    def __init__(self, key: bytes) -> None:
        if not 32 <= len(key) <= 4096:
            raise ValueError("private_hmac_key_32_to_4096_bytes_required")
        self._key = key
        self.key_id = self.token("key", "citequest-recorded-evaluation-key/1")

    def token(self, namespace: str, value: object) -> str:
        if re.fullmatch(r"[a-z]+", namespace) is None:
            raise ValueError("evaluation_hmac_namespace_invalid")
        return namespace + "_" + hmac.new(self._key, namespace.encode() + b"\x00" + canonical(value), hashlib.sha256).hexdigest()

    def tool(self, name: str, arguments: dict) -> dict:
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise ValueError("actual_named_tool_arguments_required")
        return {"tool_hmac": self.token("tool", name), "arguments_hmac": self.token("arguments", arguments)}

    def seal(self, records: list[dict]) -> dict:
        body = {"schema_version": VERSION, "key_id": self.key_id, "record_source": "postgres_repeatable_read",
            "records": records}
        return {**body, "integrity_hmac": self.token("export", body)}

    def verify(self, value: dict) -> list[dict]:
        if set(value) != {"schema_version", "key_id", "record_source", "records", "integrity_hmac"}:
            raise ValueError("recorded_export_contract_invalid")
        body = {key: item for key, item in value.items() if key != "integrity_hmac"}
        if (body["schema_version"] != VERSION or body["key_id"] != self.key_id
            or body["record_source"] != "postgres_repeatable_read" or not isinstance(body["records"], list)
            or not isinstance(value["integrity_hmac"], str)
            or not hmac.compare_digest(value["integrity_hmac"], self.token("export", body))):
            raise ValueError("recorded_export_integrity_or_private_key_mismatch")
        return body["records"]


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("invalid_evaluation_" + name)
    return value


def elapsed(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    seconds = (end - start).total_seconds()
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("recorded_timestamp_order_invalid")
    return seconds


def result_projection(answer: dict | None, protector: Protector) -> dict | None:
    if answer is None:
        return None
    if not isinstance(answer, dict):
        raise ValueError("recorded_answer_object_required")
    value = {key: answer[key] if answer[key] in PUBLIC_VALUES else protector.token("value", answer[key])
        for key in ("publication_status", "status", "completion_status", "request_coverage", "evidence_status")
        if key in answer and isinstance(answer[key], str)}
    for key in ("school_format_ready", "submission_ready"):
        if type(answer.get(key)) is bool:
            value[key] = answer[key]
    for key, source in (("claim_count", "claims"), ("citation_count", "citations")):
        if isinstance(answer.get(source), list):
            value[key] = len(answer[source])
    if isinstance(answer.get("school_document"), dict) and isinstance(answer["school_document"].get("sections"), list):
        value["school_section_count"] = len(answer["school_document"]["sections"])
    reports = answer.get("review_report", {}).get("role_reports") if isinstance(answer.get("review_report"), dict) else None
    if isinstance(reports, list):
        value["review_finding_count"] = sum(len(item.get("findings", [])) for item in reports if isinstance(item, dict))
    if isinstance(answer.get("revision_outcome"), dict):
        value["adopted_findings_resolved"] = answer["revision_outcome"].get("status") == "adopted_findings_resolved"
    value["result_hmac"] = protector.token("result", answer)
    return value


def citation_observations(answer: dict | None, evidence: dict[str, dict], protector: Protector) -> dict:
    citations = answer.get("citations") if isinstance(answer, dict) else None
    items = []
    for citation in citations if isinstance(citations, list) else []:
        if not isinstance(citation, dict) or not isinstance(citation.get("ref_id"), str):
            raise ValueError("recorded_citation_identity_invalid")
        reference, quote = citation["ref_id"], citation.get("quote")
        pinned = evidence.get(reference)
        text = pinned.get("text") if isinstance(pinned, dict) else None
        state = ("not_checked" if not isinstance(quote, str) or not quote or not isinstance(text, str)
            else "matched" if quote in text else "mismatch")
        items.append({"reference_hmac": protector.token("reference", reference),
            "quote_hmac": protector.token("quote", quote), "historical_literal": state})
    states = [item["historical_literal"] for item in items]
    return {"items": items, "historical_literal": ("no_data" if not states else
        "mismatch" if "mismatch" in states else "not_checked" if "not_checked" in states else "matched"),
        "semantic_support": "not_checked", "current_source_authorization": "not_checked"}


def project_command(
    command: dict, attempts: list[dict], reserved_events: list[dict], completion_events: list[dict],
    evidence: dict[str, dict], queue: dict | None, protector: Protector, *, runs: list[dict] | None = None,
) -> dict:
    """Observe only exact command-bound records; never use the latest Task answer."""
    if command["state"] not in COMMAND_STATES:
        raise ValueError("recorded_command_state_invalid")
    completions = [item for item in completion_events if item["command_id"] == str(command["id"])]
    if len(completions) > 1:
        raise ValueError("recorded_command_completion_ambiguous")
    completion = completions[0] if completions else None
    if completion and (completion["command_seq"] != command["seq"] or completion["state"] not in TASK_STATES):
        raise ValueError("recorded_command_completion_binding_invalid")
    runs = runs or []
    if any(item["state"] not in TASK_STATES | {"superseded"} for item in runs):
        raise ValueError("recorded_run_state_invalid")
    last_run = runs[-1] if runs else None
    terminal_state = completion["state"] if completion else (
        last_run["state"] if command["state"] == "done" and last_run and last_run["ended_at"] is not None else None)
    outcome = ("rejected" if command["state"] == "rejected" else {
        "succeeded": "completed", "failed": "terminal_failure", "cancelled": "cancelled",
        "partial": "partial", "waiting_input": "waiting_input"}.get(terminal_state, "pending"))
    observed_outcome = completion is not None or outcome in {"rejected", "terminal_failure", "cancelled"}
    event_positions = {}
    for event in reserved_events:
        identifier = event["attempt_id"]
        if identifier in event_positions:
            raise ValueError("recorded_attempt_reservation_ambiguous")
        event_positions[identifier] = _integer(event["seq"], "event_seq")
    observed, tools, known, reserved, unknown = [], [], 0, 0, 0
    for attempt in attempts:
        identifier = str(attempt["id"])
        if (identifier not in event_positions or attempt["status"] not in ATTEMPT_STATES
            or attempt["kind"] not in {"model", "tool"}):
            raise ValueError("recorded_attempt_reservation_missing_or_state_invalid")
        cash = _integer(attempt["reserved_ncny"], "reservation")
        known_cash = attempt["cost_ncny"]
        if known_cash is not None:
            known += _integer(known_cash, "known_cash")
        if attempt["status"] in {"reserved", "sent", "unknown"}:
            reserved += cash
        sent = attempt["sent_at"] is not None
        if sent and (attempt["status"] in {"sent", "unknown"} or known_cash is None):
            unknown += 1
        usage = attempt.get("usage")
        valid_usage = (isinstance(usage, dict) and all(type(usage.get(key)) is int and usage[key] >= 0
            for key in ("input", "cached", "output")) and usage["cached"] <= usage["input"])
        if sent and attempt["kind"] == "model" and not valid_usage:
            unknown += int(attempt["status"] not in {"sent", "unknown"} and known_cash is not None)
        entry = {"attempt_hmac": protector.token("attempt", identifier), "reservation_seq": event_positions[identifier],
            "kind": attempt["kind"], "status": attempt["status"], "sent": sent,
            "duration_seconds": elapsed(attempt["sent_at"], attempt["settled_at"]),
            "known_cash_ncny": known_cash, "reservation_ncny": cash if attempt["status"] in {"reserved", "sent", "unknown"} else 0,
            "usage": {key: usage[key] for key in ("input", "cached", "output")} if valid_usage else None,
            "error_hmac": protector.token("error", attempt["error_code"]) if attempt.get("error_code") else None}
        observed.append(entry)
        if sent and attempt["kind"] == "tool":
            tools.append({**protector.tool(attempt["tool_name"], attempt["tool_arguments"]),
                "attempt_hmac": entry["attempt_hmac"], "reservation_seq": entry["reservation_seq"]})
    observed.sort(key=lambda item: item["reservation_seq"])
    tools.sort(key=lambda item: item["reservation_seq"])
    answer = completion.get("answer") if completion else None
    result = {**(result_projection(answer, protector) or {}), "command_outcome": outcome}
    if queue is not None and (not math.isfinite(queue["queue_seconds"]) or queue["queue_seconds"] < 0):
        raise ValueError("recorded_queue_duration_invalid")
    return {"command_hmac": protector.token("command", str(command["id"])),
        "task_hmac": protector.token("task", str(command["task_id"])), "command_seq": command["seq"],
        "command_state": command["state"], "completion_observed": completion is not None,
        "outcome_observed": observed_outcome, "command_outcome": outcome, "terminal_state": terminal_state,
        "run_states": [{"state": row["state"], "ended": row["ended_at"] is not None,
            "duration_seconds": elapsed(row["started_at"], row["ended_at"])} for row in runs],
        "tool_calls": tools, "attempts": observed,
        "result": result, "citations": citation_observations(answer, evidence, protector),
        "latency": {"queue_seconds": float(queue["queue_seconds"]) if queue else None,
            "command_wall_seconds": elapsed(command["created_at"], completion["created_at"]) if completion else None,
            "attempt_duration_scope": "sent_to_settled_includes_transport_and_accounting"},
        "cost": {"known_cash_ncny": known, "reserved_cash_ncny": reserved, "unknown_attempts": unknown,
            "actual_total_cash_ncny": None if unknown or reserved else known, "infrastructure_cash_ncny": None}}


def tool_messages(calls: list[dict]) -> list[dict]:
    # Strict match permits reordering tool_calls within one message. One real
    # attempt per assistant message therefore preserves physical attempt order.
    return [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": item["tool_hmac"], "arguments": json.dumps({"arguments_hmac": item["arguments_hmac"]})}}]}
        for item in calls]


def validate_cases(value: dict, key_id: str) -> list[dict]:
    if set(value) != {"schema_version", "key_id", "cases"} or value["schema_version"] != CASE_VERSION or value["key_id"] != key_id:
        raise ValueError("evaluation_cases_version_or_private_key_mismatch")
    if not isinstance(value["cases"], list) or not 1 <= len(value["cases"]) <= 1000:
        raise ValueError("bounded_explicit_evaluation_cases_required")
    seen = set()
    for case in value["cases"]:
        if (not isinstance(case, dict) or set(case) != {"command_hmac", "source", "expected_tool_calls", "expected_result",
            "require_trajectory_match", "require_literal_citations", "exclude_if"} or case["source"] not in {"actual", "synthetic"}
            or not isinstance(case["command_hmac"], str) or TOKEN.fullmatch(case["command_hmac"]) is None
            or not case["command_hmac"].startswith("command_") or case["command_hmac"] in seen):
            raise ValueError("evaluation_case_binding_invalid")
        seen.add(case["command_hmac"])
        expected = case["expected_result"]
        if (not isinstance(expected, dict) or not expected or set(expected) - RESULT_FIELDS
            or any(type(flag) is not bool for flag in (case["require_trajectory_match"], case["require_literal_citations"]))
            or not isinstance(case["expected_tool_calls"], list) or len(case["expected_tool_calls"]) > 512
            or not isinstance(case["exclude_if"], list) or len(case["exclude_if"]) > 3
            or len(set(case["exclude_if"])) != len(case["exclude_if"])
            or set(case["exclude_if"]) - {"cancelled", "unknown_outcome", "missing_citation_evidence"}):
            raise ValueError("explicit_result_projection_and_criteria_required")
        for name, item in expected.items():
            if name.endswith("_count"):
                _integer(item, "gold_count")
            elif name in {"school_format_ready", "submission_ready", "adopted_findings_resolved"}:
                if type(item) is not bool:
                    raise ValueError("evaluation_gold_boolean_required")
            elif name == "result_hmac":
                if not isinstance(item, str) or TOKEN.fullmatch(item) is None or not item.startswith("result_"):
                    raise ValueError("evaluation_gold_result_hmac_required")
            elif not isinstance(item, str) or (item not in PUBLIC_VALUES
                and (TOKEN.fullmatch(item) is None or not item.startswith("value_"))):
                raise ValueError("evaluation_gold_value_must_be_public_enum_or_hmac")
        for call in case["expected_tool_calls"]:
            if (not isinstance(call, dict) or set(call) != {"tool_hmac", "arguments_hmac"}
                or any(not isinstance(call[key], str) or TOKEN.fullmatch(call[key]) is None
                    or not call[key].startswith(prefix) for key, prefix in (("tool_hmac", "tool_"), ("arguments_hmac", "arguments_")))):
                raise ValueError("evaluation_gold_tool_hmac_required")
    return value["cases"]


def evaluate_records(records: list[dict], cases: list[dict]) -> dict:
    by_command = {record["command_hmac"]: record for record in records}
    if len(by_command) != len(records):
        raise ValueError("recorded_command_identity_ambiguous")
    exclusions = {}
    for case in cases:
        record = by_command.get(case["command_hmac"])
        conditions = ({"cancelled"} if record and record["command_outcome"] == "cancelled" else set())
        if record and record["cost"]["unknown_attempts"]:
            conditions.add("unknown_outcome")
        if record and case["require_literal_citations"] and record["citations"]["historical_literal"] in {"no_data", "not_checked"}:
            conditions.add("missing_citation_evidence")
        exclusions[case["command_hmac"]] = sorted(conditions & set(case["exclude_if"]))
    ready = [case for case in cases if case["source"] == "actual" and case["command_hmac"] in by_command
        and by_command[case["command_hmac"]]["outcome_observed"] and not exclusions[case["command_hmac"]]]
    rows = []
    evaluator = exact = context = None
    if ready:
        for package, version in (("agentevals", "0.0.9"), ("openevals", "0.2.2")):
            if importlib.metadata.version(package) != version:
                raise ValueError("locked_official_evaluation_package_required")
        from langsmith import tracing_context
        context = tracing_context
        with context(enabled=False):
            from agentevals.trajectory.match import create_trajectory_match_evaluator
            from openevals.exact import exact_match
            evaluator = create_trajectory_match_evaluator(trajectory_match_mode="strict", tool_args_match_mode="exact")
            exact = exact_match
    for case in cases:
        record = by_command.get(case["command_hmac"])
        excluded = exclusions[case["command_hmac"]]
        if excluded and case["source"] == "actual":
            rows.append({"command_hmac": case["command_hmac"], "status": "excluded", "success": None,
                "trajectory_match": None, "business_result_match": None, "exclusion_reasons": excluded})
            continue
        if case not in ready:
            rows.append({"command_hmac": case["command_hmac"], "status": "no_data", "success": None,
                "trajectory_match": None, "business_result_match": None,
                "reason": "synthetic_case_or_missing_recorded_terminal_outcome"})
            continue
        with context(enabled=False):
            trajectory = evaluator(outputs=tool_messages(record["tool_calls"]), reference_outputs=tool_messages(case["expected_tool_calls"]))
            actual = {key: (record["result"] or {}).get(key, {"missing": True}) for key in case["expected_result"]}
            business = exact(outputs=actual, reference_outputs=case["expected_result"])
        if type(trajectory.get("score")) is not bool or type(business.get("score")) is not bool:
            raise ValueError("official_evaluator_boolean_score_required")
        literal = record["citations"]["historical_literal"]
        unknown = record["cost"]["unknown_attempts"] > 0
        citation_unknown = case["require_literal_citations"] and literal in {"no_data", "not_checked"}
        incomplete = unknown or citation_unknown
        success = (business["score"] and (trajectory["score"] or not case["require_trajectory_match"])
            and (literal == "matched" or not case["require_literal_citations"]))
        rows.append({"command_hmac": case["command_hmac"], "status": "incomplete" if incomplete else "evaluated",
            "success": None if incomplete else success, "trajectory_match": trajectory["score"],
            "business_result_match": business["score"], "citations": record["citations"],
            "latency": record["latency"], "cost": record["cost"]})
    scored = [row for row in rows if type(row["success"]) is bool]
    return {"schema_version": VERSION, "status": "evaluated" if scored else "no_data", "cases": rows,
        "scored_cases": len(scored), "unscored_cases": len(rows) - len(scored),
        "success_rate": sum(row["success"] for row in scored) / len(scored) if scored else None,
        "coverage": {"recorded_commands": len(records), "explicit_cases": len(cases),
            "recorded_without_gold": sum(key not in {case["command_hmac"] for case in cases} for key in by_command),
            "gold_without_record": sum(case["command_hmac"] not in by_command for case in cases),
            "excluded_cases": sum(row["status"] == "excluded" for row in rows),
            "no_data_cases": sum(row["status"] == "no_data" for row in rows),
            "incomplete_cases": sum(row["status"] == "incomplete" for row in rows),
            "observed_rejected_commands": sum(row["command_outcome"] == "rejected" for row in records),
            "observed_failure_commands": sum(row["command_outcome"] == "terminal_failure" for row in records),
            "observed_cancelled_commands": sum(row["command_outcome"] == "cancelled" for row in records)},
        "trajectory_method": "agentevals0.0.9_strict_exact_one_actual_attempt_per_message",
        "business_method": "openevals0.2.2_exact_explicit_projection", "judge_model_requests": 0,
        "semantic_support": "not_checked", "current_source_authorization": "not_checked"}
