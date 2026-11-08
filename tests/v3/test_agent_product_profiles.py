"""New-task release negotiation; no SQL, credentials or model dispatch."""
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import AgentError, TaskCreate
from citequest.agents_v3.product_profiles import NewTaskProfile, TaskProfilePolicy, public_profile_policy


def body(profile="qa-v1"):
    return TaskCreate(project_id=uuid4(), operation_id=uuid4(), profile_id=profile,
                      brief={"question": "What evidence is available?"})


def test_configured_release_rejects_old_profile_without_remapping_request():
    policy = TaskProfilePolicy(release_id="explicit-development-fixture", profiles=[
        NewTaskProfile(kind="qa", profile_id="qa-v4", validation_state="development")])
    request = body()
    with pytest.raises(AgentError) as exc:
        policy.admit(request)
    assert exc.value.code == "task_profile_refresh_required" and exc.value.status == 409
    assert request.profile_id == "qa-v1"
    policy.admit(body("qa-v4"))


def test_no_enabled_kind_is_explicitly_unavailable():
    with pytest.raises(AgentError) as exc:
        TaskProfilePolicy(release_id="quality-gates-pending", profiles=[]).admit(body())
    assert exc.value.status == 503 and exc.value.code == "task_kind_not_enabled_for_new_tasks"


@pytest.mark.parametrize("profiles", [
    [{"kind": "qa", "profile_id": "qa-invented", "validation_state": "development"}],
    [{"kind": "qa", "profile_id": "workset-v5", "validation_state": "development"}],
    [{"kind": "qa", "profile_id": "qa-v4", "validation_state": "development"}] * 2,
])
def test_unknown_or_ambiguous_release_fails_before_runtime(profiles):
    with pytest.raises(ValidationError):
        TaskProfilePolicy(release_id="explicit-fixture", profiles=profiles)


def test_legacy_capabilities_describe_compatibility_without_quality_upgrade():
    value = public_profile_policy(None)
    assert value["mode"] == "legacy_compatibility"
    assert {item["kind"]: item["profile_id"] for item in value["profiles"]}["summary"] == "workset-v2"
    assert all(item["validation_state"] == "development" for item in value["profiles"])
    assert len(value["policy_sha256"]) == 64
    assert not {"business_dsn", "api_key_file", "provider", "model_commands"} & value.keys()


def test_release_fingerprint_changes_on_explicit_selection_and_status():
    policy = TaskProfilePolicy(release_id="explicit-fixture", profiles=[
        NewTaskProfile(kind="qa", profile_id="qa-v4", validation_state="development")])
    public = public_profile_policy(policy)
    assert public["mode"] == "configured_release"
    assert public == public_profile_policy(policy)
    changed = policy.model_copy(update={"profiles": [NewTaskProfile(kind="qa", profile_id="qa-v4", validation_state="accepted")]})
    assert public_profile_policy(changed)["policy_sha256"] != public["policy_sha256"]
