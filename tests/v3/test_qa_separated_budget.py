"""One resource/replay control with an explicit execution accounting double."""
import copy
from types import SimpleNamespace
from uuid import uuid4

import pytest

from citequest.agents_v3.contracts import AgentError, PROFILES, canonical, digest
from citequest.agents_v3.product_profiles import QA8_PROTOCOL_METADATA
from citequest.agents_v3.qa_separated import QASeparated


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"], "additionalProperties": False}
PROMPT = {"context_manifest": {}, "explicit_fixture": "resource accounting only"}


def node(task, *, command, settled=None):
    value = QASeparated.__new__(QASeparated)
    receipts = {} if settled is None else settled
    value.graph = SimpleNamespace(execution=SimpleNamespace(
        snapshot=lambda: copy.deepcopy(task), command={"id": command},
        settled_model_request=lambda key: receipts.get(key),
        content_revision_count=lambda: 0, protocol_recovery_count=lambda: 0))
    return value


def test_third_turn_capacity_preserves_two_repairs_two_recoveries_and_distinct_replay_keys():
    profile = {**PROFILES["qa-v9"].model_dump(), **QA8_PROTOCOL_METADATA,
        "qa_content_protocol": "isolated1"}
    assert 3 * 4 + 2 * 2 + 2 == profile["model_calls"] == 18
    cap = profile["per_call_output"]
    task = {"kind": "qa", "profile": profile, "model_calls": 8, "output_tokens": 8 * cap}
    command = uuid4()
    current = node(task, command=command)
    state = {"qa5": {"revision": 0, "reviewing_goals": False, "lock": None, "gap_only_lock": None}}

    # Two ordinary four-call turns have completed. Each stage in turn three
    # must still admit both unused two-call repairs and both protocol recoveries.
    for used, role, goals, ordinary_remaining in (
        (8, "plan", False, 4), (9, "plan", True, 3),
        (10, "author", False, 2), (11, "review", False, 1),
    ):
        task.update(model_calls=used, output_tokens=used * cap)
        state["qa5"]["reviewing_goals"] = goals
        assert current.required_calls(state, role) == ordinary_remaining
        request = current.request(state, role, copy.deepcopy(SCHEMA), copy.deepcopy(PROMPT))
        assert request["max_output_tokens"] == cap

    task.update(model_calls=9, output_tokens=8 * cap)
    state["qa5"]["reviewing_goals"] = False
    with pytest.raises(AgentError, match="qa5_complete_remaining_protocol_model_budget"):
        current.request(state, "plan", copy.deepcopy(SCHEMA), copy.deepcopy(PROMPT))
    task.update(model_calls=8, output_tokens=8 * cap + 1)
    with pytest.raises(AgentError, match="qa5_complete_remaining_protocol_output_budget"):
        current.request(state, "plan", copy.deepcopy(SCHEMA), copy.deepcopy(PROMPT))

    planner_keys = [current.key({"revision": index, "reviewing_goals": False}, "plan") for index in range(3)]
    goal_keys = [current.key({"revision": index, "reviewing_goals": True}, "plan") for index in range(3)]
    assert len(set(planner_keys + goal_keys)) == 6
    assert [int(key.rsplit(":", 1)[1]) for key in planner_keys] == [0, 2, 4]
    assert [int(key.rsplit(":", 1)[1]) for key in goal_keys] == [1, 3, 5]
    assert current.key(state["qa5"], "author") not in planner_keys + goal_keys
    assert current.key(state["qa5"], "review") not in planner_keys + goal_keys

    task.update(model_calls=9, output_tokens=9 * cap)
    state["qa5"]["reviewing_goals"] = True
    key = current.key(state["qa5"], "plan")
    first = current.request(state, "plan", copy.deepcopy(SCHEMA), copy.deepcopy(PROMPT))
    # A settled replay reconstructs its original request even after capacity is
    # exhausted. It neither reserves another call nor changes its role identity.
    task.update(model_calls=18, output_tokens=profile["output_tokens"])
    recovered = node(task, command=command, settled={key: {"request_hash": digest(first)}})
    assert recovered.key(state["qa5"], "plan") == key
    replay = recovered.request(state, "plan", copy.deepcopy(SCHEMA), copy.deepcopy(PROMPT))
    assert canonical(replay) == canonical(first)
