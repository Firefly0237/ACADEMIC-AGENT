"""Three pure metadata/version controls; no SQL, model, HTTP or containers."""
import copy
import json

import pytest
from pydantic import ValidationError
from tests.v3.test_development_campaign import bound_task, campaign_capsule, store

from citequest.agents_v3.catalog_scope import (
    POLICY,
    REVIEW_SUFFIX,
    SUFFIX,
    current_runtime,
    graph_version,
    graph_versions,
    readable_runtime,
)
from citequest.agents_v3.contracts import PROFILES, AgentError, digest, graph_version_for_kind
from citequest.agents_v3.development_campaign import DevelopmentCampaign, approval_body, load_campaign
from citequest.agents_v3.product_profiles import NewTaskProfile, TaskProfilePolicy, public_profile_policy


def test_absent_or_none_transport_retains_exact_legacy_policy_shape_and_hash():
    raw = {"release_id": "EXPLICIT_TRANSPORT_CONTROL", "profiles": [
        {"kind": "qa", "profile_id": "qa-v5", "validation_state": "development"}]}
    old_shape = {"schema_version": "new-task-profile-policy/1", **copy.deepcopy(raw)}
    absent = TaskProfilePolicy.model_validate(raw)
    explicit_none = TaskProfilePolicy.model_validate({**raw, "profiles": [{**raw["profiles"][0], "review_transport_protocol": None}]})
    assert absent.model_dump(mode="json") == explicit_none.model_dump(mode="json") == old_shape
    value = {**old_shape, "profile_hashes": {"qa-v5": digest(PROFILES["qa-v5"].model_dump(mode="json"))}}
    assert public_profile_policy(absent) == {**value, "mode": "configured_release", "policy_sha256": digest(value)}
    assert absent.runtime_metadata("qa", "qa-v5") == {}


def test_only_shared_combined_review_gets_an_explicit_immutable_graph_suffix():
    for kind, profile in (("qa", "qa-v5"), ("stance", "stance-v2")):
        entry = NewTaskProfile(kind=kind, profile_id=profile, validation_state="development", review_transport_protocol="function1")
        policy = TaskProfilePolicy(release_id="EXPLICIT_TRANSPORT_CONTROL", profiles=[entry])
        plain = {"kind": kind, "profile": {**PROFILES[profile].model_dump(), "catalog_scope_policy": POLICY},
            "graph_version": graph_version(kind, profile, POLICY)}
        changed = copy.deepcopy(plain)
        changed["profile"].update(policy.runtime_metadata(kind, profile))
        changed["graph_version"] = graph_version(kind, profile, POLICY, "function1")
        assert changed["graph_version"] == graph_version_for_kind(kind, profile) + REVIEW_SUFFIX + SUFFIX
        assert changed["graph_version"].endswith(SUFFIX)
        assert changed["graph_version"] in graph_versions(POLICY)
        assert current_runtime(plain, POLICY) and readable_runtime(plain)
        assert current_runtime(changed, POLICY) and readable_runtime(changed)
        assert digest(changed["profile"]) != digest(plain["profile"])
        changed["graph_version"] = plain["graph_version"]
        assert not current_runtime(changed, POLICY) and not readable_runtime(changed)
    for kind, profile in (("summary", "workset-v6"), ("research", "research-v3"), ("qa", "qa-v4")):
        with pytest.raises(ValidationError):
            NewTaskProfile(kind=kind, profile_id=profile, validation_state="development", review_transport_protocol="function1")
    with pytest.raises(AgentError):
        graph_version("qa", "qa-v5", None, "function1")


def test_campaign_requires_same_transport_and_graph_but_old_release_stays_readable(tmp_path):
    path, checksum, raw = campaign_capsule(tmp_path / "old")
    old = load_campaign(path, checksum)
    old_task = bound_task(old)
    old.require_task(old_task)
    changed = copy.deepcopy(raw)
    for row in changed["task_profile_policy"]["profiles"]:
        if row["kind"] in {"qa", "stance"}:
            row["review_transport_protocol"] = "function1"
    policy = TaskProfilePolicy.model_validate(changed["task_profile_policy"], strict=True)
    changed["task_profile_policy_sha256"] = public_profile_policy(policy)["policy_sha256"]
    approval = tmp_path / "new-approval.json"
    changed["root_approval"] = {"path": str(approval), "sha256": "0" * 64}
    candidate = DevelopmentCampaign.model_validate_json(json.dumps(changed), strict=True)
    changed["root_approval"]["sha256"] = store(approval, approval_body(candidate))
    new_path = tmp_path / "new-campaign.json"
    new = load_campaign(new_path, store(new_path, changed))
    current = bound_task(new)
    with pytest.raises(AgentError):
        new.require_task(current)  # Missing immutable transport field/suffix.
    current["profile"].update(policy.runtime_metadata("qa", "qa-v5"))
    current["profile_hash"] = digest(current["profile"])
    current["graph_version"] = graph_version("qa", "qa-v5", POLICY, "function1")
    new.require_task(current)
    with pytest.raises(AgentError):
        new.require_task(old_task)  # New release cannot dispatch an old flat Task.
    old.require_task(old_task)
    assert old_task["profile"].get("review_transport_protocol") is None
