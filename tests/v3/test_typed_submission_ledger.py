"""Two focused controls using the existing explicit transactional ledger double."""
import copy
from types import SimpleNamespace
from uuid import uuid4

import pytest

from citequest.agents_v3 import database as database_module
from citequest.agents_v3.catalog_scope import POLICY, graph_version, graph_versions
from citequest.agents_v3.contracts import AgentError, canonical, digest
from citequest.agents_v3.product_profiles import NewTaskProfile, TaskProfilePolicy
from citequest.agents_v3.provider_contract import wire_contract_violation
from citequest.agents_v3.typed_submission import VERSION, compile_submission, recorded_submission
from tests.v3.test_protocol_recovery_ledger import Ledger
from tests.v3.test_typed_submission import ROLE_FUNCTIONS, chat, profile, request


def parent(ledger, role, ordinal=0):
    key = str(ledger.execution.command["id"]) + f":qa5:{role}:{ordinal}"
    payload = request(role)
    valid = ('{"clarification":null,"category":null}' if role == "plan" else '{"answer":"value"}')
    raw = chat(ROLE_FUNCTIONS[role], valid + "}")
    response = recorded_submission(ledger.task["profile"], payload, key, raw, copy.deepcopy(raw), 200)
    row = {"id": uuid4(), "task_id": ledger.execution.task_id, "kind": "model", "logical_key": key,
        "run_id": uuid4(), "epoch": 1, "status": "settled", "request": payload,
        "request_hash": digest(payload), "response": response,
        "usage": {"input": 10, "cached": 2, "output": 4}, "error_code": None}
    ledger.parents[key] = row
    compiled = compile_submission(ledger.task["profile"], payload, key, fixture=True)
    ledger.wires[str(row["id"])] = {"attempt_id": str(row["id"]), **compiled.receipt,
        "actual_wire_bytes": len(canonical(compiled.wire)), "actual_wire_sha256": digest(canonical(compiled.wire))}
    violation = wire_contract_violation(payload, response, profile=ledger.task["profile"], logical_key=key)
    assert violation is not None
    return key, payload, violation, valid


def test_all_roles_share_two_recoveries_and_author_repair_uses_only_the_verified_child(monkeypatch):
    ledger = Ledger(monkeypatch, "chat-strict2")
    ledger.task["profile"].update(profile())
    counters = {key: value for key, value in ledger.task.items() if key != "profile"}
    planned = parent(ledger, "plan")
    planned_child = ledger.execution.reserve_protocol_recovery(*planned[:3])
    assert ledger.execution.reserve_protocol_recovery(*planned[:3]) == planned_child
    authored = parent(ledger, "author")
    child_key = ledger.execution.reserve_protocol_recovery(*authored[:3])
    assert ledger.execution.protocol_recovery_count() == 2
    with pytest.raises(AgentError, match="protocol_recovery_limit"):
        ledger.execution.reserve_protocol_recovery(*parent(ledger, "review")[:3])
    with pytest.raises(AgentError, match="protocol_recovery_source_requires_known_receipt"):
        ledger.execution.effective_model_source_key(authored[0])
    original = ledger.parents[authored[0]]
    child = {**copy.deepcopy(original), "id": uuid4(), "logical_key": child_key}
    raw = chat("submit_answer", authored[3])
    child["response"] = recorded_submission(ledger.task["profile"], authored[1], child_key, raw, copy.deepcopy(raw), 200)
    ledger.parents[child_key] = child
    ledger.wires[str(child["id"])] = {**copy.deepcopy(ledger.wires[str(original["id"])]), "attempt_id": str(child["id"])}
    assert ledger.execution.effective_model_source_key(authored[0]) == child_key
    assert ledger.execution.settled_model_request(child_key)["request"] == authored[1]
    ledger.wires[str(child["id"])]["role"] = "plan"
    with pytest.raises(AgentError, match="protocol_recovery_source_binding_invalid"):
        ledger.execution.effective_model_source_key(authored[0])
    assert {key: value for key, value in ledger.task.items() if key != "profile"} == counters


def test_explicit_graph30_and_actual_wire_receipt_bind_the_issued_planner_role(monkeypatch):
    selected = NewTaskProfile(kind="qa", profile_id="qa-v7", validation_state="development",
        review_transport_protocol="function1", qa_execution_protocol="grounded3", review_api_protocol="chat-strict2",
        structured_output_protocol=VERSION)
    policy = TaskProfilePolicy(release_id="pure-typed-submission", profiles=[selected])
    assert policy.runtime_metadata("qa", "qa-v7")["structured_output_protocol"] == VERSION
    version = graph_version("qa", "qa-v7", POLICY, "function1", "grounded3", "chat-strict2", VERSION)
    assert version.startswith("citequest-agent-graph/30-") and version in graph_versions(POLICY)
    assert graph_version("qa", "qa-v7", POLICY, "function1", "grounded3", "chat-strict2").startswith("citequest-agent-graph/29-")
    ledger = Ledger(monkeypatch, "chat-strict2")
    ledger.task["profile"].update(profile())
    key, payload, _, _ = parent(ledger, "plan")
    attempt = ledger.parents[key]
    attempt.update(status="reserved", run_id=ledger.execution.run_id, epoch=ledger.execution.epoch)
    original_execute = ledger.execute

    def execute(statement, parameters):
        if "SELECT * FROM cq_app.agent_attempts WHERE id=%s AND task_id=%s" in statement:
            return SimpleNamespace(fetchone=lambda: copy.deepcopy(attempt))
        return original_execute(statement, parameters)

    monkeypatch.setattr(ledger, "execute", execute)
    monkeypatch.setattr(database_module, "event", lambda _db, _task, kind, value:
        ledger.events.append({"event_type": kind, "payload": copy.deepcopy(value)}))
    receipt = {name: value for name, value in ledger.wires[str(attempt["id"])].items() if name != "attempt_id"}
    ledger.wires.pop(str(attempt["id"]))
    result = ledger.execution.observe_model_wire(attempt["id"], receipt)
    assert result["role"] == "plan" and result["function_name"] == "submit_plan"
    with pytest.raises(AgentError, match="model_wire_receipt_identity_invalid"):
        ledger.execution.observe_model_wire(attempt["id"], {**receipt, "role": "author"})
