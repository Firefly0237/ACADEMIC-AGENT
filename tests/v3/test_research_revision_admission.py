"""Selected findings and their original report are fixed before revision."""
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import (
    PROFILES, RevisionRequest, ReviewReportSource, ReviewSource, TaskCreate, canonical,
)
from citequest.agents_v3.product_profiles import NewTaskProfile, TaskProfilePolicy


def request():
    manuscript = ReviewSource(source_task_id=uuid4(), source_command_id=uuid4(), command_seq=1,
        event_seq=20, task_revision=1, document_sha256="a" * 64, context_sha256="b" * 64)
    report = ReviewReportSource(task_id=uuid4(), command_id=uuid4(), command_seq=1, event_seq=30,
        task_revision=1, report_sha256="c" * 64, context_sha256="d" * 64)
    return TaskCreate(project_id=uuid4(), operation_id=uuid4(), kind="research", profile_id="research-revision-v1",
        brief={"kind": "research", "question": "Revise selected synthetic findings.", "research_action": "revise",
               "review_source": manuscript, "revision_request": RevisionRequest(report=report,
                    finding_keys=["evidence:E1", "methods:M2"]),
               "research_brief": {"id": uuid4(), "revision": 1, "sha256": "e" * 64}})


def test_revision_is_an_explicit_auxiliary_action_with_fixed_source_and_report():
    body = request()
    enabled = TaskProfilePolicy(release_id="synthetic-revision", profiles=[NewTaskProfile(
        kind="research", profile_id="research-v6", validation_state="development")],
        research_revision_profile=NewTaskProfile(kind="research", profile_id="research-revision-v1",
                                                validation_state="development"))
    enabled.admit(body)
    assert enabled.profiles[0].profile_id == "research-v6"
    assert TaskCreate.model_validate_json(canonical(body.model_dump(mode="json")), strict=True) == body
    with pytest.raises(ValidationError):
        TaskCreate.model_validate({**body.model_dump(mode="json"), "profile_id": "research-v6"})


def test_duplicate_or_mismatched_finding_keys_are_not_silently_dropped():
    revision = request().brief.revision_request
    for keys in (["evidence:E1", "evidence:E1"], ["evidence:M1"], [], ["unknown:X1"]):
        with pytest.raises(ValidationError):
            RevisionRequest(report=revision.report, finding_keys=keys)


def test_revision_budget_covers_declared_full_window_calls_and_retains_parent_limits():
    profile = PROFILES["research-revision-v1"]
    assert (profile.model_calls, profile.tool_calls, profile.per_call_output) == (18, 128, 2048)
    worst_call = profile.context_window * profile.input_miss_ncny + profile.per_call_output * profile.output_ncny
    assert profile.model_calls * worst_call <= profile.cash_ncny
    assert profile.output_tokens == profile.model_calls * profile.per_call_output
    assert PROFILES["research-v6"].model_calls == 64
