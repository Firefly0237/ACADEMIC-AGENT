"""Research-intake user flows; all inputs are explicit synthetic user statements."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.research_v3.brief import (
    REQUIRED,
    BriefCreate,
    BriefDocument,
    BriefResolution,
    BriefTurn,
    FieldPatch,
    apply_turn,
    create_document,
    planning_payload,
    readiness,
    resolve,
)
from citequest.v2.domain import DomainError


def turn(document: BriefDocument, *patches: FieldPatch) -> BriefDocument:
    return apply_turn(
        document,
        BriefTurn(operation_id=uuid4(), expected_revision=document.revision, message_id=uuid4(), patches=list(patches)),
    )


def empty() -> BriefDocument:
    return create_document(BriefCreate(operation_id=uuid4(), project_id=uuid4(), message_id=uuid4()))


def complete() -> BriefDocument:
    return turn(empty(), *(FieldPatch(field=field, value="User-confirmed " + field) for field in REQUIRED))


def test_missing_intake_waits_without_model_or_false_school_readiness() -> None:
    value = readiness(empty())
    assert value["state"] == "needs_input"
    assert value["model_calls_required"] == 0
    assert value["missing_fields"] == list(REQUIRED)
    assert not value["school_format_ready"]


def test_user_can_start_with_direction_then_supply_remaining_information() -> None:
    document = turn(empty(), FieldPatch(field="topic", value="A reproducible retrieval experiment"))
    assert "topic" not in readiness(document)["missing_fields"]
    document = turn(
        document, *(FieldPatch(field=field, value="Confirmed " + field) for field in REQUIRED if field != "topic")
    )
    handoff = planning_payload(document)
    assert handoff["research_status"] == "proposed_not_executed"
    assert handoff["source_admission_required"]
    assert not handoff["school_format_ready"]


def test_agent_suggestion_never_fills_a_confirmed_fact_automatically() -> None:
    document = turn(empty(), FieldPatch(field="methods", intent="suggest", value="Suggested method"))
    assert "methods" not in document.facts
    assert "methods" in readiness(document)["missing_fields"]
    proposal = document.proposals[0]
    document = resolve(
        document,
        BriefResolution(
            operation_id=uuid4(),
            expected_revision=document.revision,
            message_id=uuid4(),
            proposal_id=proposal.id,
            decision="use_candidate",
        ),
    )
    assert document.facts["methods"].value == "Suggested method"


def test_contradictory_turn_preserves_both_values_and_blocks_planning() -> None:
    document = complete()
    old = document.facts["resources"]
    document = turn(document, FieldPatch(field="resources", value="Changed budget and hardware"))
    assert document.facts["resources"] == old
    assert document.proposals[0].kind == "conflict"
    assert readiness(document)["open_conflicts"]
    with pytest.raises(DomainError, match="needs_input"):
        planning_payload(document)


def test_explicit_user_revision_does_not_require_redundant_approval() -> None:
    document = turn(complete(), FieldPatch(field="resources", intent="revise", value="Only one CPU machine"))
    assert document.facts["resources"].value == "Only one CPU machine"
    assert not readiness(document)["open_conflicts"]


def test_accepting_one_conflict_supersedes_other_candidates_for_the_field() -> None:
    document = turn(complete(), FieldPatch(field="resources", value="Candidate A"))
    document = turn(document, FieldPatch(field="resources", value="Candidate B"))
    first = document.proposals[0]
    document = resolve(
        document,
        BriefResolution(
            operation_id=uuid4(),
            expected_revision=document.revision,
            message_id=uuid4(),
            proposal_id=first.id,
            decision="use_candidate",
        ),
    )
    assert [p.state for p in document.proposals] == ["accepted", "superseded"]
    assert readiness(document)["state"] == "ready_for_planning"


def test_same_value_from_another_turn_does_not_make_existing_proposal_stale() -> None:
    document = complete()
    old = document.facts["resources"]
    document = turn(document, FieldPatch(field="resources", value="Candidate"))
    document = turn(document, FieldPatch(field="resources", value=old.value))
    document = resolve(
        document,
        BriefResolution(
            operation_id=uuid4(),
            expected_revision=document.revision,
            message_id=uuid4(),
            proposal_id=document.proposals[0].id,
            decision="keep_current",
        ),
    )
    assert document.facts["resources"] == old


def test_preliminary_work_can_be_explicitly_absent_without_invention() -> None:
    document = turn(
        complete(), FieldPatch(field="preliminary_work", intent="revise", value="No work has been performed")
    )
    document = turn(document, FieldPatch(field="preliminary_work", intent="unknown", value="Not provided"))
    document = resolve(
        document,
        BriefResolution(
            operation_id=uuid4(),
            expected_revision=document.revision,
            message_id=uuid4(),
            proposal_id=document.proposals[0].id,
            decision="use_candidate",
        ),
    )
    payload = planning_payload(document)
    assert payload["explicit_unknowns"]["preliminary_work"] == "Not provided"
    assert "preliminary_work" not in payload["user_confirmed"]


def test_unknown_resources_still_need_input() -> None:
    document = turn(empty(), FieldPatch(field="resources", intent="unknown", value="Unknown"))
    assert "resources" in readiness(document)["missing_fields"]


def test_stale_revision_cannot_overwrite_a_newer_turn() -> None:
    document = complete()
    with pytest.raises(DomainError, match="revision_conflict"):
        apply_turn(
            document,
            BriefTurn(
                operation_id=uuid4(),
                expected_revision=document.revision - 1,
                message_id=uuid4(),
                patches=[FieldPatch(field="topic", intent="revise", value="Old browser tab")],
            ),
        )


def test_clearing_a_field_invalidates_readiness_and_open_candidates() -> None:
    document = turn(complete(), FieldPatch(field="resources", value="Alternative"))
    document = turn(document, FieldPatch(field="resources", intent="clear"))
    assert "resources" in readiness(document)["missing_fields"]
    assert document.proposals[0].state == "superseded"


def test_duplicate_fields_and_oversized_drafts_are_not_silently_trimmed() -> None:
    with pytest.raises(DomainError, match="duplicate_brief_field"):
        turn(empty(), FieldPatch(field="topic", value="A"), FieldPatch(field="topic", value="B"))
    with pytest.raises(ValidationError):
        turn(empty(), *(FieldPatch(field=field, value="字" * 8000) for field in REQUIRED))


def test_resolved_conflict_cannot_be_applied_again_as_a_new_operation() -> None:
    document = turn(complete(), FieldPatch(field="resources", value="Alternative"))
    identifier = document.proposals[0].id
    document = resolve(
        document,
        BriefResolution(
            operation_id=uuid4(),
            expected_revision=document.revision,
            message_id=uuid4(),
            proposal_id=identifier,
            decision="keep_current",
        ),
    )
    with pytest.raises(DomainError, match="not_open"):
        resolve(
            document,
            BriefResolution(
                operation_id=uuid4(),
                expected_revision=document.revision,
                message_id=uuid4(),
                proposal_id=identifier,
                decision="use_candidate",
            ),
        )


def test_unrecognized_field_cannot_enter_the_brief() -> None:
    with pytest.raises(ValidationError):
        FieldPatch.model_validate({"field": "approved_spending", "value": "unlimited"})
