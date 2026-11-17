"""Approval tasks are dispatchable only by workers registering their protocol version."""
from __future__ import annotations

from unittest.mock import Mock
from uuid import uuid4

import pytest

from citequest.agents_v3.catalog_scope import (
    EXECUTION_APPROVAL_SUFFIX,
    POLICY,
    SUFFIX,
    current_runtime,
    graph_version,
    graph_versions,
    readable_runtime,
)
from citequest.agents_v3.contracts import (
    CURRENT_GRAPH_VERSIONS,
    PROFILES,
    AgentError,
    RuntimeLimits,
    TaskCreate,
    graph_version_for_kind,
)
from citequest.agents_v3.database import TaskService


def task(policy, *, marker="approval1"):
    profile = PROFILES["research-v6"].model_dump(mode="json")
    if policy is not None:
        profile["catalog_scope_policy"] = policy
    return {"kind": "research", "profile": profile,
        "brief": {"execution_plan_protocol": marker} if marker is not None else {},
        "graph_version": graph_version("research", "research-v6", policy, execution_plan=marker)}


@pytest.mark.parametrize("policy", [None, POLICY])
def test_approval_version_keeps_legacy_versions_and_catalog_suffix_unchanged(policy):
    plain = graph_version("research", "research-v6", policy)
    approved = graph_version("research", "research-v6", policy, execution_plan="approval1")
    assert plain == graph_version_for_kind("research", "research-v6") + (SUFFIX if policy else "")
    assert approved != plain
    assert approved == graph_version_for_kind("research", "research-v6") + EXECUTION_APPROVAL_SUFFIX + (SUFFIX if policy else "")
    registered = graph_versions(policy)
    assert plain in registered and approved in registered
    assert approved not in {version + (SUFFIX if policy else "") for version in CURRENT_GRAPH_VERSIONS}
    assert all(version + (SUFFIX if policy else "") in registered for version in CURRENT_GRAPH_VERSIONS)
    if policy is not None:
        assert all(version.endswith(SUFFIX) for version in registered)
    else:
        assert not approved.endswith(SUFFIX)


@pytest.mark.parametrize("policy", [None, POLICY])
def test_read_and_dispatch_identity_bind_the_frozen_brief_protocol(policy):
    approved, legacy = task(policy), task(policy, marker=None)
    assert current_runtime(approved, policy) and readable_runtime(approved)
    assert current_runtime(legacy, policy) and readable_runtime(legacy)
    assert not current_runtime({**approved, "graph_version": legacy["graph_version"]}, policy)
    assert not readable_runtime({**approved, "graph_version": legacy["graph_version"]})
    assert not current_runtime({**legacy, "graph_version": approved["graph_version"]}, policy)
    assert not readable_runtime({**legacy, "graph_version": approved["graph_version"]})
    assert not current_runtime(approved, POLICY if policy is None else None)


@pytest.mark.parametrize("kind,profile,marker", [
    ("research", "research-v6", "approval2"), ("research", "research-v5", "approval1"),
    ("research", "research-review-v1", "approval1"), ("qa", "qa-v1", "approval1"),
    ("summary", "workset-v7", "approval1"), ("stance", "stance-v3", "approval1"),
])
def test_unknown_or_incompatible_execution_protocol_is_rejected(kind, profile, marker):
    with pytest.raises(AgentError, match="task_execution_plan_protocol_not_supported"):
        graph_version(kind, profile, POLICY, execution_plan=marker)


def test_task_creation_selects_the_approval_version_before_opening_a_transaction(monkeypatch):
    from citequest.agents_v3 import database as module

    class StopBeforeDatabase(Exception):
        pass

    database = Mock(catalog_scope_policy=POLICY, development_dispatch_policy=None,
        development_campaign_policy=None)
    database.connect.side_effect = StopBeforeDatabase
    service = TaskService(database, RuntimeLimits(mode="bounded"))
    original = graph_version
    selected = []

    def observed(*args, **kwargs):
        selected.append(original(*args, **kwargs))
        return selected[-1]

    monkeypatch.setattr(module.catalog_scope, "graph_version", observed)
    request = TaskCreate(project_id=uuid4(), operation_id=uuid4(), kind="research", profile_id="research-v6",
        brief={"kind": "research", "question": "A synthetic bounded objective",
            "research_action": "experiment", "execution_plan_protocol": "approval1",
            "research_brief": {"id": uuid4(), "revision": 1, "sha256": "a" * 64}})
    with pytest.raises(StopBeforeDatabase):
        service.create(uuid4(), request)
    assert selected == [original("research", "research-v6", POLICY, execution_plan="approval1")]
