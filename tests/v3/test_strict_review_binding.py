"""Pure Execution receipt controls. Connections/events are explicit doubles."""
import copy
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from citequest.agents_v3 import database as database_module
from citequest.agents_v3.contracts import AgentError, canonical, digest
from citequest.agents_v3.database import Execution
from citequest.agents_v3.strict_chat import compile_review_request


def request():
    return {"model": "deepseek-flash", "reasoning": {"effort": "none"}, "truncation": "disabled",
        "instructions": "Explicit binding fixture", "max_output_tokens": 100,
        "input": [{"role": "user", "content": "Explicit binding fixture"}],
        "tools": [{"type": "function", "name": "submit_review", "parameters": {"type": "object",
            "properties": {"verdict": {"type": "string", "enum": ["accepted", "rejected"]}},
            "required": ["verdict"], "additionalProperties": False}}],
        "tool_choice": {"type": "function", "name": "submit_review"}}


def chat_response():
    return {"_cq_review_envelope_gate": {"protocol": "chat-strict1", "accepted": True, "error_code": None},
        "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "tool_calls",
        "message": {"role": "assistant", "content": None, "tool_calls": [{"type": "function",
            "id": "synthetic-submission", "function": {"name": "submit_review", "arguments": '{"verdict":"accepted"}'}}]}}]}


class Binding:
    def __init__(self, monkeypatch, *, strict=True):
        self.execution = object.__new__(Execution)
        self.execution.task_id, self.execution.run_id, self.execution.epoch = uuid4(), uuid4(), 3
        self.execution.command = {"id": uuid4()}
        self.key = str(self.execution.command["id"]) + ":qa5:review:0"
        self.task = {"profile": {"profile_id": "qa-v6", "qa_execution_protocol": "grounded1",
            "review_transport_protocol": "function1", "request_bytes": 131072}}
        if strict:
            self.task["profile"]["review_api_protocol"] = "chat-strict1"
        payload = request()
        self.attempt = {"id": uuid4(), "task_id": self.execution.task_id, "kind": "model",
            "run_id": self.execution.run_id, "epoch": self.execution.epoch, "status": "reserved",
            "logical_key": self.key, "request": payload, "request_hash": digest(payload),
            "usage": {"input": 10, "cached": 2, "output": 4}, "error_code": None,
            "response": chat_response() if strict else {"status": "completed", "output": []}}
        compiled = compile_review_request(payload, fixture=True)
        self.receipt = {**compiled.receipt, "actual_wire_bytes": len(canonical(compiled.wire)),
            "actual_wire_sha256": digest(canonical(compiled.wire))}
        self.events, self.connections, self.statements = [], 0, []
        self.execution.database = SimpleNamespace(connect=self.connect)
        monkeypatch.setattr(self.execution, "lock_task", lambda _db: self.task)
        monkeypatch.setattr(database_module, "event", self.event)

    @contextmanager
    def connect(self):
        self.connections += 1
        yield self

    def execute(self, statement, parameters):
        self.statements.append(statement)
        assert statement.lstrip().startswith("SELECT"), "the tested entry must not write another ledger table"
        if "cq_app.agent_events" in statement:
            row = {"payload": copy.deepcopy(self.events[0])} if self.events else None
        elif "cq_app.agent_attempts" in statement:
            expected = ((self.execution.task_id, self.key) if "logical_key=%s" in statement
                else (self.attempt["id"], self.execution.task_id))
            row = copy.deepcopy(self.attempt) if parameters == expected else None
        else:
            raise AssertionError("unexpected fake SQL")
        return SimpleNamespace(fetchone=lambda: row)

    def event(self, connection, task_id, event_type, payload):
        assert connection is self and task_id == self.execution.task_id and event_type == "model_wire_bound"
        self.events.append(copy.deepcopy(payload))


def test_legacy_and_strict_settled_requests_reconstruct_without_transport(monkeypatch):
    for strict in (False, True):
        binding = Binding(monkeypatch, strict=strict)
        binding.attempt["status"] = "settled"
        before = copy.deepcopy(binding.attempt)
        result = binding.execution.settled_model_request(binding.key)
        assert result == {"request": before["request"], "request_hash": before["request_hash"]}
        assert binding.attempt == before and not binding.events
    binding.attempt["request"]["tools"] = []  # Ordinary roles in a strict Task still use Responses.
    binding.attempt["request_hash"] = digest(binding.attempt["request"])
    assert binding.execution.settled_model_request(binding.key) is None


def test_unsettled_malformed_or_changed_requests_cannot_become_replay_hints(monkeypatch):
    for change in ({"status": "sent"}, {"status": "settled", "usage": None},
                   {"status": "settled", "usage": {"input": 1, "cached": 2, "output": 0}}):
        binding = Binding(monkeypatch)
        binding.attempt.update(change)
        assert binding.execution.settled_model_request(binding.key) is None
    binding = Binding(monkeypatch)
    binding.attempt.update(status="settled", request_hash="a" * 64)
    with pytest.raises(AgentError, match="settled_request_hash_invalid"):
        binding.execution.settled_model_request(binding.key)
    binding = Binding(monkeypatch)
    with pytest.raises(AgentError, match="settled_request_command_identity_invalid"):
        binding.execution.settled_model_request(str(uuid4()) + ":qa5:review:0")
    assert binding.connections == 0
    for gate in (None, {"protocol": "chat-strict1", "accepted": False, "error_code": "chat_strict_thinking_output_forbidden"},
                 {"protocol": "chat-strict1", "accepted": 1, "error_code": None}):
        binding = Binding(monkeypatch)
        binding.attempt["status"] = "settled"
        if gate is None:
            binding.attempt["response"].pop("_cq_review_envelope_gate")
        else:
            binding.attempt["response"]["_cq_review_envelope_gate"] = gate
        with pytest.raises(AgentError):
            binding.execution.settled_model_request(binding.key)
        assert not binding.events


def test_wire_metadata_shape_hash_syntax_and_record_size_reject_before_connect(monkeypatch):
    changes = ({"actual_wire_bytes": 0}, {"actual_wire_bytes": True}, {"actual_wire_sha256": "not-a-hash"},
        {"endpoint": "https://unexpected.invalid/chat/completions"}, {"local_only_constraints": {}},
        {"removed_annotations": ["x" * 65537]}, {"unexpected": "synthetic"})
    for change in changes:
        binding = Binding(monkeypatch)
        with pytest.raises(AgentError, match="model_wire_receipt_invalid"):
            binding.execution.observe_model_wire(binding.attempt["id"], {**binding.receipt, **change})
        assert binding.connections == 0 and not binding.events


def test_wire_receipt_rejects_foreign_owner_hash_profile_and_wire_budget(monkeypatch):
    changes = ({"run_id": uuid4()}, {"epoch": 4}, {"kind": "tool"}, {"status": "settled"})
    for change in changes:
        binding = Binding(monkeypatch)
        binding.attempt.update(change)
        with pytest.raises(AgentError, match="model_wire_receipt_identity_invalid"):
            binding.execution.observe_model_wire(binding.attempt["id"], binding.receipt)
        assert not binding.events
    for change in ({"canonical_request_sha256": "b" * 64}, {"actual_wire_bytes": 131073}):
        binding = Binding(monkeypatch)
        with pytest.raises(AgentError, match="model_wire_receipt_identity_invalid"):
            binding.execution.observe_model_wire(binding.attempt["id"], {**binding.receipt, **change})
        assert not binding.events
    binding = Binding(monkeypatch, strict=False)
    with pytest.raises(AgentError, match="model_wire_receipt_identity_invalid"):
        binding.execution.observe_model_wire(binding.attempt["id"], binding.receipt)
    with pytest.raises(AgentError, match="model_wire_receipt_identity_invalid"):
        binding.execution.observe_model_wire(uuid4(), binding.receipt)
    assert not binding.events


def test_same_wire_receipt_is_idempotent_and_changed_value_is_rejected(monkeypatch):
    binding = Binding(monkeypatch)
    before = copy.deepcopy(binding.attempt)
    first = binding.execution.observe_model_wire(binding.attempt["id"], binding.receipt)
    second = binding.execution.observe_model_wire(binding.attempt["id"], copy.deepcopy(binding.receipt))
    assert first == second == {"attempt_id": str(binding.attempt["id"]), **binding.receipt}
    assert len(binding.events) == 1 and binding.attempt == before
    with pytest.raises(AgentError, match="model_wire_receipt_replay_changed"):
        binding.execution.observe_model_wire(binding.attempt["id"], {**binding.receipt, "actual_wire_sha256": "c" * 64})
    assert len(binding.events) == 1 and binding.attempt == before
