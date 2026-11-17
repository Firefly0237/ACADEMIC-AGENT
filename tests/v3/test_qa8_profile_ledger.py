"""QA8 admission and ledger identity controls using the explicit SQL double."""
import copy
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3 import database as database_module
from citequest.agents_v3.catalog_scope import POLICY, graph_version, graph_versions, readable_runtime
from citequest.agents_v3.contracts import AgentError, PROFILES, TaskCreate, canonical, digest
from citequest.agents_v3.product_profiles import NewTaskProfile, QA8_PROTOCOL_METADATA, TaskProfilePolicy
from citequest.agents_v3.provider_contract import wire_contract_violation
from citequest.agents_v3.typed_submission import compile_submission, recorded_submission
from tests.v3.test_json_submission import chat, request
from tests.v3.test_protocol_recovery_ledger import Ledger


def test_qa9_reasoning_capacity_is_separate_and_reuses_the_anchored_workflow():
    from citequest.agents_v3.qa_v8 import QA8
    from citequest.agents_v3.qa_v9 import QA9

    current = PROFILES["qa-v9"].model_dump(mode="json")
    aggregate = current["input_tokens"] * current["input_miss_ncny"] + current["output_tokens"] * current["output_ncny"]
    reserve = current["context_window"] * current["input_miss_ncny"] + current["per_call_output"] * current["output_ncny"]
    assert aggregate == 35306496000 and reserve == 11206656000
    assert aggregate + reserve == 46513152000 < current["cash_ncny"] == 50000000000
    assert current["output_tokens"] == current["model_calls"] * current["per_call_output"] == 1179648
    assert (current["per_call_output"], current["active_seconds"], current["review_read_seconds"]) == (65536, 1800, 300)
    previous = PROFILES["qa-v8"].model_dump(mode="json")
    assert (previous["per_call_output"], previous["active_seconds"], previous["review_read_seconds"]) == (8192, 900, 180)
    assert previous["output_tokens"] == 147456 and previous["cash_ncny"] == 20000000000
    selected = NewTaskProfile(kind="qa", profile_id="qa-v9", validation_state="development", **QA8_PROTOCOL_METADATA)
    policy = TaskProfilePolicy(release_id="qa9-capacity-control", profiles=[selected])
    body = TaskCreate.model_validate_json(json.dumps({"project_id": str(uuid4()), "operation_id": str(uuid4()),
        "profile_id": "qa-v9", "brief": {"question": "What does the current evidence support?"}}))
    policy.admit(body)
    version = graph_version("qa", "qa-v9", POLICY, "json1", "grounded3", "chat-json2", "chat-json2", "jsonpatch1", "references1")
    assert version in graph_versions(POLICY)
    assert readable_runtime({"kind": "qa", "profile": {**current, **QA8_PROTOCOL_METADATA,
        "catalog_scope_policy": POLICY}, "graph_version": version})
    assert QA9.profile_id == "qa-v9" and QA8.profile_id == "qa-v8"
    assert issubclass(QA9, QA8)
    assert QA9.reasoning_policy is QA8.reasoning_policy


def test_qa8_explicit_profile_resources_and_old_budget_serialization():
    selected = NewTaskProfile(kind="qa", profile_id="qa-v8", validation_state="development", **QA8_PROTOCOL_METADATA)
    policy = TaskProfilePolicy(release_id="qa8-pure-control", profiles=[selected])
    metadata = policy.runtime_metadata("qa", "qa-v8")
    assert metadata == QA8_PROTOCOL_METADATA
    current = PROFILES["qa-v8"].model_dump(mode="json")
    aggregate = current["input_tokens"] * current["input_miss_ncny"] + current["output_tokens"] * current["output_ncny"]
    reserve = current["context_window"] * current["input_miss_ncny"] + current["per_call_output"] * current["output_ncny"]
    assert aggregate == 7437312000 and reserve == 9658368000
    assert aggregate + reserve == 17095680000 < current["cash_ncny"] == 20000000000
    assert 17 * current["per_call_output"] < current["output_tokens"]
    assert current["request_bytes"] == 262144 and current["review_read_seconds"] == 180
    previous = PROFILES["qa-v7"].model_dump(mode="json")
    assert previous["request_bytes"] == 131072 and previous["per_call_output"] == 4096
    assert previous["active_seconds"] == 450 and previous["cash_ncny"] == 15000000000
    assert "review_read_seconds" not in previous
    body = TaskCreate.model_validate_json(json.dumps({"project_id": str(uuid4()), "operation_id": str(uuid4()),
        "profile_id": "qa-v8", "kind": "qa", "brief": {"question": "What does this paper establish?"}}))
    policy.admit(body)
    version = graph_version("qa", "qa-v8", POLICY, "json1", "grounded3", "chat-json2", "chat-json2", "jsonpatch1", "references1")
    assert version.startswith("citequest-agent-graph/34-") and version in graph_versions(POLICY)
    assert readable_runtime({"kind": "qa", "profile": {**current, **metadata, "catalog_scope_policy": POLICY}, "graph_version": version})
    for key in QA8_PROTOCOL_METADATA:
        with pytest.raises(ValidationError):
            NewTaskProfile(kind="qa", profile_id="qa-v8", validation_state="development",
                **{**QA8_PROTOCOL_METADATA, key: None})
    with pytest.raises(AgentError):
        graph_version("qa", "qa-v8", None, "json1", "grounded3", "chat-json2", "chat-json2", "jsonpatch1", "references1")


def test_qa8_known_recovery_replay_and_exact_endpoint_identity(monkeypatch):
    ledger = Ledger(monkeypatch, "chat-json2")
    ledger.task["profile"] = {**PROFILES["qa-v8"].model_dump(mode="json"), **QA8_PROTOCOL_METADATA}
    baseline = {key: copy.deepcopy(value) for key, value in ledger.task.items() if key != "profile"}
    key = str(ledger.execution.command["id"]) + ":qa5:author:0"
    payload = request("author")
    raw = chat()
    raw["choices"][0]["message"]["content"] += "}"
    stored = recorded_submission(ledger.task["profile"], payload, key, raw, copy.deepcopy(raw), 200)
    parent = {"id": uuid4(), "task_id": ledger.execution.task_id, "kind": "model", "logical_key": key,
        "run_id": ledger.execution.run_id, "epoch": ledger.execution.epoch, "status": "settled",
        "request": payload, "request_hash": digest(payload), "response": stored,
        "usage": {"input": 10, "cached": 2, "output": 10}, "error_code": None}
    ledger.parents[key] = parent
    compiled = compile_submission(ledger.task["profile"], payload, key, fixture=True)
    receipt = {**compiled.receipt, "actual_wire_bytes": len(canonical(compiled.wire)),
        "actual_wire_sha256": digest(canonical(compiled.wire))}
    ledger.wires[str(parent["id"])] = {"attempt_id": str(parent["id"]), **receipt}
    violation = wire_contract_violation(payload, stored, profile=ledger.task["profile"], logical_key=key)
    assert violation["category"] == "invalid_json"
    child_key = ledger.execution.reserve_protocol_recovery(key, payload, violation)
    assert ledger.execution.reserve_protocol_recovery(key, payload, violation) == child_key
    good = chat()
    child = {**copy.deepcopy(parent), "id": uuid4(), "logical_key": child_key,
        "response": recorded_submission(ledger.task["profile"], payload, child_key, good, copy.deepcopy(good), 200)}
    ledger.parents[child_key] = child
    ledger.wires[str(child["id"])] = {"attempt_id": str(child["id"]), **receipt}
    assert ledger.execution.effective_model_source_key(key) == child_key
    assert ledger.execution.settled_model_request(child_key)["request_hash"] == digest(payload)
    ledger.wires[str(child["id"])]["endpoint"] = "https://s08-provider.invalid/beta/chat/completions"
    with pytest.raises(AgentError, match="protocol_recovery_source_binding_invalid"):
        ledger.execution.effective_model_source_key(key)

    parent["status"] = "reserved"
    original_execute = ledger.execute
    def execute(statement, parameters):
        if "SELECT * FROM cq_app.agent_attempts WHERE id=%s AND task_id=%s" in statement:
            return SimpleNamespace(fetchone=lambda: copy.deepcopy(parent))
        return original_execute(statement, parameters)
    monkeypatch.setattr(ledger, "execute", execute)
    monkeypatch.setattr(database_module, "event", lambda _db, _task, kind, value:
        ledger.events.append({"event_type": kind, "payload": copy.deepcopy(value)}))
    ledger.wires.pop(str(parent["id"]))
    assert ledger.execution.observe_model_wire(parent["id"], receipt)["protocol"] == "chat-json2"
    with pytest.raises(AgentError, match="model_wire_receipt_identity_invalid"):
        ledger.execution.observe_model_wire(parent["id"], {**receipt, "endpoint": "https://s08-provider.invalid/beta/chat/completions"})
    assert {name: value for name, value in ledger.task.items() if name != "profile"} == baseline
