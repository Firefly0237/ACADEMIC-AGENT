"""Pure protocol model; not actual PostgreSQL/Milvus/source-admission evidence."""

from __future__ import annotations

import copy
from uuid import uuid4

import pytest

from citequest.index_v2.publication import AliasTransport, PublicationCoordinator


class Fixture:
    def __init__(self):
        self.old, self.new = uuid4(), uuid4()
        self.old_name, self.new_name = "cq_v2_old_fixture", "cq_v2_new_fixture"
        self.alias = "cq_v2_test_protocol_alias"
        self.aliases = {self.alias: self.old_name}
        self.sql = {"active": self.old, "epoch": 7, "pending": None, "intent": None}
        self.in_transaction = False
        self.calls = []
        self.timeout_after_mutation = False
        self.source_ready = True

    def tx(self, action):
        assert not self.in_transaction
        before = copy.deepcopy(self.sql)
        self.in_transaction = True
        try:
            return action(self)
        except BaseException:
            self.sql = before
            raise
        finally:
            self.in_transaction = False

    def plan(self):
        row = self.sql["intent"]
        return {
            "state": row["state"],
            "alias_name": self.alias,
            "target_collection": self.new_name,
            "previous_collection": self.old_name,
            "intent_id": str(row["id"]),
        }

    def publish_intent(self, generation_id, intent_id, *, expected_epoch):
        if not self.source_ready or generation_id != self.new:
            raise ValueError("source gate refused")
        prior = self.sql["intent"]
        if prior:
            if prior["id"] != intent_id or prior["epoch"] != expected_epoch:
                raise ValueError("intent conflict")
            return self.plan()
        if self.sql["epoch"] != expected_epoch or self.sql["pending"]:
            raise ValueError("epoch conflict")
        self.sql["intent"] = {"id": intent_id, "state": "prepared", "epoch": expected_epoch}
        self.sql["pending"] = intent_id
        return self.plan()

    def record_alias(self, intent_id, receipt):
        assert self.sql["pending"] == intent_id and receipt["observed_collection"] == self.new_name
        self.sql["intent"]["state"] = "alias_confirmed"

    def commit_publish(self, intent_id):
        assert self.sql["intent"]["state"] == "alias_confirmed" and self.sql["pending"] == intent_id
        self.sql["active"], self.sql["epoch"], self.sql["pending"] = self.new, 8, None
        self.sql["intent"]["state"] = "committed"
        return self.plan()

    def recovery_target(self, intent_id):
        assert self.sql["intent"]["id"] == intent_id
        committed = self.sql["intent"]["state"] == "committed"
        return {
            **self.plan(),
            "action": "repair_committed_alias" if committed else "rollback_uncommitted_alias",
            "authoritative_collection": self.new_name if committed else self.old_name,
        }

    def acknowledge_rollback(self, intent_id, actual_collection):
        assert actual_collection == self.old_name and self.sql["active"] == self.old and self.sql["epoch"] == 7
        self.sql["intent"]["state"], self.sql["pending"] = "rolled_back", None
        return self.plan()

    def call(self, method, **kwargs):
        assert not self.in_transaction, "network/alias RPC held a SQL transaction"
        self.calls.append((method, kwargs))
        alias = kwargs.get("alias")
        if method == "list_aliases":
            return {"aliases": list(self.aliases), "db_name": "default"}
        if method == "describe_alias":
            return {"collection_name": self.aliases[alias]}
        if method in {"create_alias", "alter_alias"}:
            self.aliases[alias] = kwargs["collection_name"]
        elif method == "drop_alias":
            self.aliases.pop(alias, None)
        else:
            raise AssertionError("unsupported fixture RPC")
        if self.timeout_after_mutation:
            self.timeout_after_mutation = False
            raise TimeoutError("synthetic error AFTER alias mutation")

    def coordinator(self):
        return PublicationCoordinator(self.tx, AliasTransport(self))


@pytest.mark.parametrize(
    "point",
    [
        "after_durable_intent_before_alias",
        "after_alias_before_sql_ack",
        "after_alias_ack_before_sql_commit",
        "after_sql_commit",
    ],
)
def test_each_crash_boundary_recovers_alias_to_the_sql_authority(point):
    fixture = Fixture()
    intent = uuid4()

    def crash(current):
        if current == point:
            raise RuntimeError("explicit synthetic crash")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        fixture.coordinator().publish(fixture.new, intent, expected_epoch=7, failpoint=crash)
    # The alias may already point at the new collection. Reads still capture
    # the SQL pointer, which changes only in the one durable CAS transaction.
    expected = fixture.new if point == "after_sql_commit" else fixture.old
    assert fixture.sql["active"] == expected
    fixture.coordinator().recover(intent)
    assert fixture.aliases[fixture.alias] == (fixture.new_name if expected == fixture.new else fixture.old_name)
    assert fixture.sql["active"] == expected and fixture.sql["pending"] is None


def test_unknown_alias_mutation_result_is_read_back_not_assumed_unwritten():
    fixture = Fixture()
    fixture.timeout_after_mutation = True
    result = fixture.coordinator().publish(fixture.new, uuid4(), expected_epoch=7)
    assert result["alias_receipt"]["mutation_error"]["type"] == "TimeoutError"
    assert fixture.sql["active"] == fixture.new and fixture.aliases[fixture.alias] == fixture.new_name


def test_committed_replay_repairs_drift_without_second_epoch_or_write_effect():
    fixture = Fixture()
    intent = uuid4()
    fixture.coordinator().publish(fixture.new, intent, expected_epoch=7)
    fixture.aliases[fixture.alias] = fixture.old_name
    result = fixture.coordinator().publish(fixture.new, intent, expected_epoch=7)
    assert result["status"] == "already_committed_sql_authoritative"
    assert fixture.sql["epoch"] == 8 and fixture.aliases[fixture.alias] == fixture.new_name


@pytest.mark.parametrize("reason", ["source_not_ready", "stale_epoch"])
def test_failed_admission_has_no_alias_side_effect(reason):
    fixture = Fixture()
    if reason == "source_not_ready":
        fixture.source_ready = False
    with pytest.raises(ValueError):
        fixture.coordinator().publish(fixture.new, uuid4(), expected_epoch=6 if reason == "stale_epoch" else 7)
    assert fixture.calls == [] and fixture.sql["active"] == fixture.old


def test_rolled_back_intent_cannot_be_reopened_by_replay():
    fixture = Fixture()
    intent = uuid4()

    def crash(point):
        if point == "after_alias_before_sql_ack":
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError):
        fixture.coordinator().publish(fixture.new, intent, expected_epoch=7, failpoint=crash)
    fixture.coordinator().recover(intent)
    calls = len(fixture.calls)
    with pytest.raises(ValueError, match="cannot_be_reopened"):
        fixture.coordinator().publish(fixture.new, intent, expected_epoch=7)
    assert len(fixture.calls) == calls and fixture.sql["active"] == fixture.old
