"""One approved origin command may supply prior receipts; every other boundary stays fixed."""
import copy
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.database import Execution
from citequest.research_v3.handoff_contracts import bind_handoff, collect_handoff_audit


class ReceiptDatabase:
    def __init__(self):
        self.execution = Execution.__new__(Execution)
        self.execution.task_id = uuid4()
        self.execution.database = self
        self.origin = uuid4()
        self.task = {"owner_id": uuid4(), "kind": "research", "profile": {"profile_id": "research-v6"},
            "brief": {"execution_plan_protocol": "approval1"}}
        self.plan = {"protocol": "approval1", "task_id": str(self.execution.task_id),
            "origin_command_id": str(self.origin)}
        self.execution.command = {"id": uuid4(), "kind": "resume", "seq": 3,
            "payload": {"plan_decision": {"action": "approve", "plan_sha256": digest(self.plan)}}}
        self.origin_row = {"id": self.origin, "task_id": self.execution.task_id,
            "owner_id": self.task["owner_id"], "seq": 1, "state": "done", "kind": "start"}
        self.execution.lock_task = lambda _db: self.task
        self.reads = []
        self.rows = {}
        for identifier in (self.origin, self.execution.command["id"]):
            key = str(identifier) + ":research:synthetic:planner"
            request = {"text": {"format": {"name": "citequest_research_outline_planner"}}, "input": "fixture"}
            self.rows[key] = {"id": uuid4(), "logical_key": key, "status": "settled", "error_code": None,
                "request": request, "request_hash": digest(request), "response": {"status": "completed"},
                "usage": {"input": 20, "cached": 2, "output": 8}}

    @property
    def keys(self):
        return list(self.rows)

    @contextmanager
    def connect(self):
        yield self

    def execute(self, query, params=()):
        assert query.strip().startswith("SELECT"), "receipt reads cannot mutate commands or resources"
        self.reads.append((query, params))
        if "FROM cq_app.agent_commands" in query:
            row = self.origin_row
            match = ((row["id"], row["task_id"], row["owner_id"]) == params[:3]
                and row["state"] == "done" and row["kind"] != "cancel" and row["seq"] < params[3])
            rows = [{"id": row["id"]}] if match else []
        elif "FROM cq_app.agent_attempts" in query:
            assert params[0] == self.execution.task_id
            rows = [self.rows[key] for key in params[1] if key in self.rows]
        else:
            raise AssertionError("Unexpected receipt SQL fixture")
        return SimpleNamespace(fetchone=lambda: rows[0] if rows else None, fetchall=lambda: rows)

    def read(self):
        return self.execution.settled_model_receipts(self.keys, approved_plan=self.plan)


def test_approved_resume_reads_exactly_one_prior_origin_with_current_task_filter():
    database = ReceiptDatabase()
    receipts = database.read()
    assert set(receipts) == set(database.keys)
    assert all(value["usage"] == {"input": 20, "cached": 2, "output": 8} for value in receipts.values())
    assert all("request" not in value and "response" not in value for value in receipts.values())
    current = database.keys[1]
    assert database.execution.settled_model_receipts([current])[current] == receipts[current]
    with pytest.raises(AgentError, match="handoff_receipt_command_identity_invalid"):
        database.execution.settled_model_receipts([database.keys[0]])


@pytest.mark.parametrize("change", ["task", "marker", "protocol", "approve", "hash", "command", "missing_decision"])
def test_approval_cannot_authorize_other_tasks_or_unapproved_plans(change):
    database = ReceiptDatabase()
    if change == "task":
        database.plan["task_id"] = str(uuid4())
        database.execution.command["payload"]["plan_decision"]["plan_sha256"] = digest(database.plan)
    elif change == "marker":
        database.task["brief"].clear()
    elif change == "protocol":
        database.plan["protocol"] = "other"
        database.execution.command["payload"]["plan_decision"]["plan_sha256"] = digest(database.plan)
    elif change == "approve":
        database.execution.command["payload"]["plan_decision"].update(action="adjust_queries", queries={"RQ1": "change"})
    elif change == "hash":
        database.execution.command["payload"]["plan_decision"]["plan_sha256"] = "0" * 64
    elif change == "command":
        database.execution.command["kind"] = "followup"
    else:
        database.execution.command["payload"].clear()
    with pytest.raises(AgentError, match="handoff_receipt_plan_approval_invalid"):
        database.read()
    assert database.reads == []


@pytest.mark.parametrize("change", [{"task_id": uuid4()}, {"owner_id": uuid4()}, {"id": uuid4()},
    {"state": "queued"}, {"kind": "cancel"}, {"seq": 3}, {"seq": 4}])
def test_origin_must_be_an_earlier_completed_owned_command_of_the_same_task(change):
    database = ReceiptDatabase()
    database.origin_row.update(change)
    with pytest.raises(AgentError, match="handoff_receipt_plan_origin_invalid"):
        database.read()
    assert len(database.reads) == 1


def test_approved_origin_does_not_admit_third_command_or_unsettled_attempt():
    database = ReceiptDatabase()
    with pytest.raises(AgentError, match="handoff_receipt_command_identity_invalid"):
        database.execution.settled_model_receipts([str(uuid4()) + ":planner"], approved_plan=database.plan)
    database.rows[database.keys[0]]["status"] = "unknown"
    with pytest.raises(AgentError, match="handoff_model_receipt_unavailable"):
        database.read()


def test_handoff_audit_records_each_actual_origin_and_rejects_implicit_or_mixed_origin():
    database = ReceiptDatabase()
    receipts = database.read()
    artifact = {"goal": "Synthetic bounded goal"}
    entries = [bind_handoff({"role": "planner", "artifact": artifact, "sha256": digest(artifact)},
        {key: receipts[key]["schema_name"]}, {"brief": "a" * 64}) for key in database.keys]
    identity = {"task_id": str(database.execution.task_id), "command_id": str(database.execution.command["id"]),
        "brief_sha256": "a" * 64, "profile_sha256": "b" * 64}
    with pytest.raises(AgentError, match="research_handoff_receipt_identity_invalid"):
        collect_handoff_audit(entries, receipts, **identity)
    result = collect_handoff_audit(entries, receipts, prior_command_id=str(database.origin), **identity)
    assert [entry["command_id"] for entry in result["handoffs"]] == [str(database.origin), identity["command_id"]]
    mixed = copy.deepcopy(entries[:1])
    mixed[0]["audit_binding"]["sources"][database.keys[1]] = receipts[database.keys[1]]["schema_name"]
    with pytest.raises(AgentError, match="research_handoff_receipt_identity_ambiguous"):
        collect_handoff_audit(mixed, receipts, prior_command_id=str(database.origin), **identity)
