"""Optional research review keeps default generation and immutable source distinct."""
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import PROFILES, AgentError, ReviewSource, TaskCommand, TaskCreate, canonical
from citequest.agents_v3.product_profiles import NewTaskProfile, TaskProfilePolicy


def source():
    return ReviewSource(source_task_id=uuid4(), source_command_id=uuid4(), command_seq=2,
                        event_seq=42, task_revision=2, document_sha256="a" * 64, context_sha256="b" * 64)


def request():
    return TaskCreate(project_id=uuid4(), operation_id=uuid4(), kind="research", profile_id="research-review-v1",
        brief={"kind": "research", "question": "Review the selected synthetic version.", "research_action": "review",
               "review_source": source(), "research_brief": {"id": uuid4(), "revision": 1, "sha256": "c" * 64}})


def policy(review=False):
    return TaskProfilePolicy(release_id="synthetic-policy", profiles=[NewTaskProfile(
        kind="research", profile_id="research-v6", validation_state="development")],
        research_review_profile=(NewTaskProfile(kind="research", profile_id="research-review-v1",
            validation_state="development") if review else None))


def test_auxiliary_review_is_explicit_and_does_not_replace_research_default():
    body = request()
    old = policy()
    assert "research_review_profile" not in old.model_dump(mode="json")
    with pytest.raises(AgentError, match="task_kind_not_enabled"):
        old.admit(body)
    enabled = policy(True)
    enabled.admit(body)
    assert enabled.profiles[0].profile_id == "research-v6"
    assert TaskCreate.model_validate_json(canonical(body.model_dump(mode="json")), strict=True) == body


def test_review_action_and_source_cannot_be_used_by_an_ordinary_profile():
    body = request().model_dump(mode="json")
    with pytest.raises(ValidationError):
        TaskCreate.model_validate({**body, "profile_id": "research-v6"})
    body["brief"].pop("review_source")
    with pytest.raises(ValidationError):
        TaskCreate.model_validate(body)
    with pytest.raises(ValidationError):
        TaskCommand(operation_id=uuid4(), expected_revision=1, kind="followup", review_source=source())


def test_declared_review_budget_covers_all_eight_full_window_flash_calls():
    profile = PROFILES["research-review-v1"]
    assert (profile.model_calls, profile.tool_calls, profile.per_call_output, profile.active_seconds) == (8, 80, 4096, 900)
    worst_call = profile.context_window * profile.input_miss_ncny + profile.per_call_output * profile.output_ncny
    assert profile.model_calls * worst_call <= profile.cash_ncny
    assert profile.input_tokens == profile.model_calls * profile.context_window
    assert profile.output_tokens == profile.model_calls * profile.per_call_output
