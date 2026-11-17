"""Approval inputs and command admission, with explicit SQL doubles and no services."""
import copy
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import (
    PROFILES,
    AgentError,
    PlanDecision,
    ResearchBrief,
    RuntimeLimits,
    TaskCommand,
    TaskCreate,
    canonical,
    digest,
)


def task_request(*, protocol="approval1", profile="research-v6", action="plan"):
    brief = {"kind": "research", "question": "Design a bounded experiment.", "research_action": action,
        "research_brief": {"id": uuid4(), "revision": 1, "sha256": "b" * 64}}
    if protocol is not None:
        brief["execution_plan_protocol"] = protocol
    return {"project_id": uuid4(), "operation_id": uuid4(), "kind": "research", "profile_id": profile,
        "brief": brief}


def decision(**updates):
    return {"plan_sha256": "a" * 64, "action": "approve", **updates}


def test_opt_in_protocol_is_limited_to_supported_tasks_and_omitted_from_legacy_inputs():
    old = TaskCreate.model_validate(task_request(protocol=None))
    assert "execution_plan_protocol" not in old.brief.model_dump(mode="json")
    for action in ("plan", "experiment"):
        new = TaskCreate.model_validate(task_request(action=action))
        assert new.brief.execution_plan_protocol == "approval1"
        assert TaskCreate.model_validate_json(canonical(new.model_dump(mode="json")), strict=True) == new
    with pytest.raises(ValidationError):
        TaskCreate.model_validate(task_request(profile="research-v5"))
    with pytest.raises(ValidationError):
        TaskCreate.model_validate(task_request(protocol="approval2"))
    with pytest.raises(ValidationError):
        TaskCreate.model_validate(task_request(action="suggest"))
    with pytest.raises(ValidationError):
        ResearchBrief(question="A bounded question.", execution_plan_protocol="approval1")


@pytest.mark.parametrize("updates", [
    {"plan_sha256": "A" * 64}, {"plan_sha256": "a" * 63},
    {"queries": {"RQ1": "changed query"}}, {"action": "adjust_queries"},
    {"action": "adjust_queries", "queries": {"unknown": "query"}},
    {"action": "adjust_queries", "queries": {"RQ1": " "}},
    {"action": "adjust_queries", "queries": {"RQ1": "\x00query"}},
    {"action": "adjust_queries", "queries": {"RQ1": "x" * 2001}},
    {"action": "adjust_queries", "queries": {f"RQ{i}": "query" for i in range(1, 10)}},
])
def test_decisions_reject_ambiguous_or_nonstorable_adjustments(updates):
    with pytest.raises(ValidationError):
        PlanDecision.model_validate(decision(**updates))


def test_query_adjustments_preserve_exact_text_and_decisions_are_isolated_resume_commands():
    value = decision(action="adjust_queries", queries={"RQ1": "  bounded query\nwith conditions  "})
    accepted = TaskCommand(operation_id=uuid4(), expected_revision=2, kind="resume", plan_decision=value)
    assert accepted.plan_decision.queries == value["queries"]
    assert accepted.message == ""
    assert "plan_decision" not in TaskCommand(operation_id=uuid4(), expected_revision=2,
        kind="resume", message="Continue.").model_dump(mode="json")
    for changes in ({"kind": "followup"}, {"kind": "cancel"}, {"reference_turn_seq": 1},
            {"selected_skills": []}, {"research_action": "plan"},
            {"research_brief": {"id": uuid4(), "revision": 1, "sha256": "b" * 64}}):
        with pytest.raises(ValidationError):
            TaskCommand.model_validate({**accepted.model_dump(mode="json"), **changes})
    with pytest.raises(ValidationError):
        TaskCommand(operation_id=uuid4(), expected_revision=2, kind="resume", execution_plan_protocol="approval1")


class CommandDatabase:
    """Exercise the command transaction boundary without claiming PostgreSQL coverage."""
    catalog_scope_policy = None
    dispatch_policy = None

    def __init__(self, monkeypatch):
        from citequest.agents_v3 import database as module
        from citequest.agents_v3.execution_plan import INTERRUPT_KIND, build_execution_plan

        request = TaskCreate.model_validate(task_request())
        profile = PROFILES[request.profile_id].model_dump(mode="json")
        self.task = {"id": uuid4(), "owner_id": uuid4(), "project_id": request.project_id,
            "kind": request.kind, "brief": request.brief.model_dump(mode="json"),
            "profile": profile, "profile_hash": digest(profile), "revision": 3,
            "state": "waiting_input", "cancellation_requested": False, "command_seq": 1}
        identity = request.brief.research_brief.sha256
        plan = build_execution_plan(self.task, {"intake": {"brief_sha256": identity},
            "execution_plan_origin_command_id": str(uuid4()),
            "outline": {"brief_sha256": identity, "questions": [{"id": "RQ1", "question": "Synthetic target",
                "retrieval_query": "retrieval", "experiment_ids": ["E1"]}]}, "research_capacity": {}})
        self.plan_hash = digest(plan)
        self.task["result"] = {"interrupts": [{"kind": INTERRUPT_KIND, "plan": plan, "plan_sha256": self.plan_hash}]}
        self.commands, self.events, self.admissions = [], [], []
        self.unresolved = False
        monkeypatch.setattr(module, "wallet_admit", lambda *args, **kwargs: self.admissions.append("admitted"))
        monkeypatch.setattr(module, "event", lambda _db, _id, kind, payload: self.events.append((kind, payload)))
        self.service = module.TaskService(self, RuntimeLimits(mode="bounded"))
        self.service.public = lambda task, **kwargs: copy.deepcopy(task)

    @contextmanager
    def connect(self):
        yield self

    def execute(self, query, params=()):
        rows = []
        if "JOIN cq_app.projects" in query:
            rows = [self.task] if params == (self.task["id"], self.task["owner_id"]) else []
        elif "pg_advisory_xact_lock" in query:
            pass
        elif "to_regclass('cq_app.agent_dispatch_control')" in query:
            rows = [{"name": None}]
        elif "SELECT * FROM cq_app.agent_commands" in query:
            rows = [row for row in self.commands if row["owner_id"] == params[0] and row["operation_id"] == params[1]]
        elif "FROM cq_app.agent_attempts" in query:
            rows = [{"unresolved": True}] if self.unresolved else []
        elif query.startswith("UPDATE cq_app.agent_tasks SET brief="):
            self.task.update(brief=params[0].obj, revision=self.task["revision"] + 1, state="queued", result=None)
        elif query.startswith("UPDATE cq_app.agent_tasks SET command_seq="):
            self.task["command_seq"] += 1
            rows = [{"command_seq": self.task["command_seq"]}]
        elif query.startswith("INSERT INTO cq_app.agent_commands"):
            self.commands.append(dict(zip(("id", "task_id", "owner_id", "operation_id", "seq", "kind",
                "payload", "payload_hash", "state"), params, strict=True)))
        elif query.startswith("UPDATE cq_app.agent_tasks SET cancellation_requested="):
            self.task["cancellation_requested"] = True
        elif query.startswith("UPDATE cq_app.agent_tasks SET state='cancelled'"):
            self.task["state"] = "cancelled"
        else:
            raise AssertionError("Unexpected SQL in explicit command fixture: " + query)
        return SimpleNamespace(fetchone=lambda: rows[0] if rows else None, fetchall=lambda: rows)

    def submit(self, **updates):
        request = TaskCommand.model_validate({"operation_id": uuid4(), "expected_revision": self.task["revision"],
            "kind": "resume", "plan_decision": decision(plan_sha256=self.plan_hash), **updates})
        self.service.command(self.task["owner_id"], self.task["id"], request)
        return request


def test_pending_approval_rejects_free_text_before_budget_or_command_mutation(monkeypatch):
    database = CommandDatabase(monkeypatch)
    with pytest.raises(AgentError, match="execution_plan_decision_required"):
        database.submit(plan_decision=None, message="Continue without a typed decision.")
    assert database.task["state"] == "waiting_input"
    assert database.commands == database.admissions == []


def test_approval_preserves_goal_and_records_decision_with_idempotent_replay(monkeypatch):
    database = CommandDatabase(monkeypatch)
    original = copy.deepcopy(database.task["brief"])
    request = database.submit()
    assert database.task["state"] == "queued" and database.task["revision"] == 4
    assert database.task["brief"] == original
    assert database.commands[0]["payload"].obj["plan_decision"] == request.plan_decision.model_dump(mode="json")
    assert database.events[-1][1]["plan_sha256"] == database.plan_hash
    assert database.events[-1][1]["plan_action"] == "approve"
    database.service.command(database.task["owner_id"], database.task["id"], request)
    assert len(database.commands) == len(database.admissions) == 1


def test_stale_revision_unknown_attempt_and_wrong_owner_keep_approval_waiting(monkeypatch):
    database = CommandDatabase(monkeypatch)
    with pytest.raises(AgentError, match="agent_revision_conflict"):
        database.submit(expected_revision=2)
    database.unresolved = True
    with pytest.raises(AgentError, match="agent_attempt_unresolved"):
        database.submit()
    with pytest.raises(AgentError, match="agent_task_not_found"):
        database.service.command(uuid4(), database.task["id"], TaskCommand(operation_id=uuid4(),
            expected_revision=3, kind="resume", plan_decision=decision(plan_sha256=database.plan_hash)))
    assert database.task["state"] == "waiting_input"
    assert database.commands == database.admissions == []


def test_cancellation_does_not_require_a_plan_decision(monkeypatch):
    database = CommandDatabase(monkeypatch)
    database.submit(kind="cancel", plan_decision=None)
    assert database.task["state"] == "cancelled"
    assert database.commands[0]["kind"] == "cancel" and database.admissions == []
