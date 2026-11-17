"""Pure transactional ledger doubles, with real wire-contract classification."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from citequest.agents_v3 import database as database_module
from citequest.agents_v3.contracts import AgentError, canonical, digest
from citequest.agents_v3.database import Execution
from citequest.agents_v3.provider_contract import wire_contract_violation
from citequest.agents_v3.strict_chat import compile_review_request


def _request():
    return {"model": "deepseek-flash", "reasoning": {"effort": "none"}, "truncation": "disabled",
        "instructions": "Synthetic strict-wire ledger fixture", "max_output_tokens": 100,
        "input": [{"role": "user", "content": "Synthetic fixture"}],
        "tools": [{"type": "function", "name": "submit_review", "parameters": {"type": "object",
            "properties": {"verdict": {"type": "string", "enum": ["accepted", "rejected"]}},
            "required": ["verdict"], "additionalProperties": False}}],
        "tool_choice": {"type": "function", "name": "submit_review"}}


def _response():
    return {"_cq_review_envelope_gate": {"protocol": "chat-strict1", "accepted": True, "error_code": None},
        "object": "chat.completion", "usage": {"prompt_tokens": 10, "completion_tokens": 4,
            "total_tokens": 14, "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 8},
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant",
            "content": None, "tool_calls": [{"id": "synthetic-call", "type": "function",
                "function": {"name": "submit_review", "arguments": '{"verdict":"accepted"}}'}}]}}]}


class Ledger:
    def __init__(self, monkeypatch, protocol="chat-strict1"):
        self.execution = object.__new__(Execution)
        self.execution.task_id, self.execution.run_id, self.execution.epoch = uuid4(), uuid4(), 2
        self.execution.command = {"id": uuid4()}
        self.task = {"kind": "qa", "profile": {"profile_id": "qa-v6", "qa_execution_protocol": "grounded3",
            "review_transport_protocol": "function1", "review_api_protocol": "chat-strict1", "request_bytes": 131072},
            "model_calls": 3, "known_ncny": 100, "reserved_ncny": 0, "input_tokens": 30, "output_tokens": 12}
        self.parents, self.wires, self.events = {}, {}, []
        self.task["profile"]["review_api_protocol"] = protocol
        self.connections, self.unresolved = 0, False
        self.execution.database = SimpleNamespace(connect=self.connect)
        monkeypatch.setattr(self.execution, "lock_task", lambda _db: self.task)
        monkeypatch.setattr(database_module, "event", self.event)

    def parent(self, revision=0):
        key = str(self.execution.command["id"]) + f":qa5:review:{revision}"
        request, response = _request(), _response()
        protocol = self.task["profile"]["review_api_protocol"]
        response["_cq_review_envelope_gate"]["protocol"] = protocol
        row = {"id": uuid4(), "task_id": self.execution.task_id, "kind": "model", "logical_key": key,
            "run_id": uuid4(), "epoch": 1, "status": "settled", "request": request,
            "request_hash": digest(request), "response": response,
            "usage": {"input": 10, "cached": 2, "output": 4}, "error_code": None}
        self.parents[key] = row
        compiled = compile_review_request(request, fixture=True, protocol=protocol)
        self.wires[str(row["id"])] = {"attempt_id": str(row["id"]), **compiled.receipt,
            "actual_wire_bytes": len(canonical(compiled.wire)), "actual_wire_sha256": digest(canonical(compiled.wire))}
        violation = wire_contract_violation(request, response, protocol=protocol)
        assert violation is not None
        return key, request, violation

    @contextmanager
    def connect(self):
        self.connections += 1
        yield self

    def execute(self, statement, parameters):
        assert statement.lstrip().startswith("SELECT"), "recovery must not edit attempts, wallets or Task counters"
        rows = []
        if "count(*)" in statement:
            kind = "protocol_recovery_reserved" if "protocol_recovery_reserved" in statement else "content_revision_reserved"
            rows = [{"n": sum(row["event_type"] == kind for row in self.events)}]
        elif "event_type='model_wire_bound'" in statement:
            value = self.wires.get(parameters[1])
            rows = [{"payload": copy.deepcopy(value)}] if value is not None else []
        elif "event_type='protocol_recovery_reserved'" in statement:
            rows = [{"payload": copy.deepcopy(row["payload"])} for row in self.events
                    if row["event_type"] == "protocol_recovery_reserved"
                    and ("payload->>'parent_logical_key'=%s" not in statement
                         or row["payload"]["parent_logical_key"] == parameters[1])]
        elif "logical_key=%s" in statement:
            value = self.parents.get(parameters[1])
            rows = [copy.deepcopy(value)] if value is not None else []
        elif "status IN ('reserved','sent','unknown')" in statement:
            rows = [{"unresolved": True}] if self.unresolved else []
        else:
            raise AssertionError("unexpected fake SQL")
        return SimpleNamespace(fetchone=lambda: rows[0] if rows else None, fetchall=lambda: rows)

    def event(self, connection, task_id, event_type, payload):
        assert connection is self and task_id == self.execution.task_id
        assert event_type in {"provider_contract_violation", "protocol_recovery_reserved"}
        self.events.append({"event_type": event_type, "payload": copy.deepcopy(payload)})


def test_task_and_parent_bounds_with_idempotent_cold_reconstruction(monkeypatch):
    ledger = Ledger(monkeypatch)
    first = ledger.parent(0)
    expected = first[0] + ":wire-recovery:1"
    assert ledger.execution.reserve_protocol_recovery(*first) == expected
    assert ledger.execution.reserve_protocol_recovery(*copy.deepcopy(first)) == expected
    assert len(ledger.events) == 2 and ledger.execution.protocol_recovery_count() == 1
    second = ledger.parent(1)
    assert ledger.execution.reserve_protocol_recovery(*second) == second[0] + ":wire-recovery:1"
    assert ledger.execution.protocol_recovery_count() == 2
    with pytest.raises(AgentError, match="protocol_recovery_limit"):
        ledger.execution.reserve_protocol_recovery(*ledger.parent(2))
    with pytest.raises(AgentError, match="protocol_recovery_request_invalid"):
        ledger.execution.reserve_protocol_recovery(expected, first[1], first[2])
    assert len(ledger.events) == 4
    ledger.events[1]["payload"]["request_hash"] = "f" * 64
    with pytest.raises(AgentError, match="protocol_recovery_replay_changed"):
        ledger.execution.reserve_protocol_recovery(*first)


def test_parent_command_request_violation_profile_and_wire_binding_cannot_change(monkeypatch):
    ledger = Ledger(monkeypatch)
    key, request, violation = ledger.parent()
    with pytest.raises(AgentError, match="protocol_recovery_request_invalid"):
        ledger.execution.reserve_protocol_recovery(str(uuid4()) + ":qa5:review:0", request, violation)
    assert ledger.connections == 0
    for change in ({"category": []}, {"arguments_sha256": "wrong"}, {"private_text": "must not become feedback"}):
        with pytest.raises(AgentError, match="protocol_recovery_request_invalid"):
            ledger.execution.reserve_protocol_recovery(key, request, {**violation, **change})
    with pytest.raises(AgentError, match="protocol_recovery_parent_request_changed"):
        ledger.execution.reserve_protocol_recovery(key, {**request, "instructions": "Changed"}, violation)
    with pytest.raises(AgentError, match="protocol_recovery_violation_not_verified"):
        ledger.execution.reserve_protocol_recovery(key, request, {**violation, "arguments_sha256": "c" * 64})
    ledger.task["profile"]["qa_execution_protocol"] = "grounded2"
    with pytest.raises(AgentError, match="protocol_recovery_profile_not_enabled"):
        ledger.execution.reserve_protocol_recovery(key, request, violation)
    ledger.task["profile"]["qa_execution_protocol"] = "grounded3"
    ledger.wires[str(ledger.parents[key]["id"])]["wire_schema_sha256"] = "d" * 64
    with pytest.raises(AgentError, match="protocol_recovery_wire_binding_invalid"):
        ledger.execution.reserve_protocol_recovery(key, request, violation)
    ledger.parents[key]["request_hash"] = "e" * 64
    with pytest.raises(AgentError, match="protocol_recovery_parent_request_changed"):
        ledger.execution.reserve_protocol_recovery(key, request, violation)
    assert not ledger.events


def test_unknown_incomplete_usage_gate_http_refusal_and_unresolved_do_not_recover(monkeypatch):
    for change in ({"status": "unknown"}, {"status": "sent"}, {"usage": None}, {"error_code": "provider_usage_missing"},
            {"usage": {"input": 10, "cached": 11, "output": 4}}, {"usage": {"input": 10, "cached": 2, "output": 5}}):
        ledger = Ledger(monkeypatch)
        key, request, violation = ledger.parent()
        ledger.parents[key].update(change)
        with pytest.raises(AgentError, match="protocol_recovery_requires_known_receipt"):
            ledger.execution.reserve_protocol_recovery(key, request, violation)
        assert not ledger.events
    for error_code in ("chat_strict_thinking_output_forbidden", "provider_http_error_with_known_usage"):
        ledger = Ledger(monkeypatch)
        key, request, violation = ledger.parent()
        ledger.parents[key]["response"]["_cq_review_envelope_gate"].update(accepted=False, error_code=error_code)
        with pytest.raises(AgentError, match="protocol_recovery_violation_not_verified"):
            ledger.execution.reserve_protocol_recovery(key, request, violation)
        assert not ledger.events
    ledger = Ledger(monkeypatch)
    key, request, violation = ledger.parent()
    ledger.parents[key]["response"]["choices"][0]["message"]["refusal"] = "Synthetic refusal"
    with pytest.raises(AgentError, match="protocol_recovery_violation_not_verified"):
        ledger.execution.reserve_protocol_recovery(key, request, violation)
    ledger.parents[key]["response"]["choices"][0]["message"].pop("refusal")
    ledger.unresolved = True
    with pytest.raises(AgentError, match="attempt_outcome_unresolved"):
        ledger.execution.reserve_protocol_recovery(key, request, violation)
    assert not ledger.events
    ledger = Ledger(monkeypatch)
    key, request, violation = ledger.parent()
    ledger.parents[key]["response"].pop("_cq_review_envelope_gate")
    with pytest.raises(AgentError, match="protocol_recovery_violation_not_verified"):
        ledger.execution.reserve_protocol_recovery(key, request, violation)

    def deny_access(_db):
        raise AgentError("project_access_revoked", 403)

    monkeypatch.setattr(ledger.execution, "lock_task", deny_access)
    with pytest.raises(AgentError, match="project_access_revoked"):
        ledger.execution.reserve_protocol_recovery(key, request, violation)
    assert not ledger.events


def test_only_safe_events_are_added_without_rewriting_usage_or_spending_counters(monkeypatch):
    ledger = Ledger(monkeypatch)
    key, request, violation = ledger.parent()
    ledger.events = [{"event_type": "content_revision_reserved", "payload": {"ordinal": ordinal, "limit": 2}}
                     for ordinal in (1, 2)]
    content_before = copy.deepcopy(ledger.events)
    task_before, parent_before = copy.deepcopy(ledger.task), copy.deepcopy(ledger.parents[key])
    assert ledger.execution.reserve_protocol_recovery(key, request, violation) == key + ":wire-recovery:1"
    assert ledger.task == task_before and ledger.parents[key] == parent_before
    assert ledger.events[:2] == content_before and ledger.execution.content_revision_count() == 2
    assert ledger.execution.protocol_recovery_count() == 1
    assert [row["event_type"] for row in ledger.events[2:]] == ["provider_contract_violation", "protocol_recovery_reserved"]
    for row in ledger.events[2:]:
        payload = row["payload"]
        assert payload["parent_attempt_id"] == str(parent_before["id"])
        assert payload["canonical_request_sha256"] == digest(request)
        assert payload["violation"] == violation
        assert not {"request", "response", "arguments", "instructions", "source_text"} & payload.keys()
    assert ledger.events[-1]["payload"]["ordinal"] == 1
    assert ledger.events[-1]["payload"]["limit"] == 2
    assert ledger.events[-1]["payload"]["per_parent_limit"] == 1


@pytest.mark.parametrize("protocol", ["chat-strict1", "chat-strict2"])
def test_effective_source_uses_only_bound_valid_known_child_and_never_falls_back(monkeypatch, protocol):
    ledger = Ledger(monkeypatch, protocol=protocol)
    key, request, violation = ledger.parent()
    assert ledger.execution.effective_model_source_key(key) == key
    author_key = str(ledger.execution.command["id"]) + ":qa5:author:0"
    assert ledger.execution.effective_model_source_key(author_key) == author_key
    child_key = ledger.execution.reserve_protocol_recovery(key, request, violation)
    child = copy.deepcopy(ledger.parents[key])
    child.update(id=uuid4(), logical_key=child_key)
    child["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"verdict":"rejected"}'
    ledger.parents[child_key] = copy.deepcopy(child)
    parent_id, child_id = str(ledger.parents[key]["id"]), str(child["id"])
    ledger.wires[child_id] = {**copy.deepcopy(ledger.wires[parent_id]), "attempt_id": child_id}
    before = copy.deepcopy((ledger.events, ledger.parents, ledger.task))
    assert ledger.execution.effective_model_source_key(key) == child_key
    assert (ledger.events, ledger.parents, ledger.task) == before
    wires_before = copy.deepcopy(ledger.wires)
    for target, change in ((parent_id, None), (child_id, None),
            (parent_id, {"full_schema_sha256": "c" * 64}), (child_id, {"wire_request_sha256": "d" * 64}),
            (child_id, {"actual_wire_bytes": 0}), (child_id, {"actual_wire_bytes": 131073}),
            (parent_id, {"actual_wire_sha256": "invalid"}), (child_id, {"endpoint": "https://unexpected.invalid"})):
        ledger.wires = copy.deepcopy(wires_before)
        if change is None:
            ledger.wires.pop(target)
        else:
            ledger.wires[target].update(change)
        with pytest.raises(AgentError, match="protocol_recovery_source_binding_invalid"):
            ledger.execution.effective_model_source_key(key)
    ledger.wires = wires_before
    for change in ({"status": "unknown"}, {"status": "sent"}, {"usage": None}, {"error_code": "provider_usage_missing"}):
        ledger.parents[child_key] = {**copy.deepcopy(child), **change}
        with pytest.raises(AgentError, match="protocol_recovery_source_requires_known_receipt"):
            ledger.execution.effective_model_source_key(key)
    ledger.parents[child_key] = copy.deepcopy(child)
    ledger.parents[child_key]["request"]["instructions"] = "Changed canonical child"
    ledger.parents[child_key]["request_hash"] = digest(ledger.parents[child_key]["request"])
    with pytest.raises(AgentError, match="protocol_recovery_source_binding_invalid"):
        ledger.execution.effective_model_source_key(key)
    ledger.parents[child_key] = copy.deepcopy(child)
    ledger.parents[child_key]["response"]["_cq_review_envelope_gate"]["accepted"] = False
    ledger.parents[child_key]["response"]["_cq_review_envelope_gate"]["error_code"] = "chat_strict_thinking_output_forbidden"
    with pytest.raises(AgentError, match="protocol_recovery_source_requires_known_receipt"):
        ledger.execution.effective_model_source_key(key)
    ledger.parents[child_key] = copy.deepcopy(child)
    ledger.parents[child_key]["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] += "}"
    with pytest.raises(AgentError, match="protocol_recovery_source_binding_invalid"):
        ledger.execution.effective_model_source_key(key)
    ledger.parents[child_key] = copy.deepcopy(child)
    ledger.events[-1]["payload"]["request_hash"] = "f" * 64
    with pytest.raises(AgentError, match="protocol_recovery_source_binding_invalid"):
        ledger.execution.effective_model_source_key(key)
    assert len(ledger.events) == 2
