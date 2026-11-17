"""School selection is revisioned; old JSON and create replays stay stable."""
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import canonical, digest
from citequest.research_v3.brief import BriefCreate, BriefDocument, BriefTurn, apply_turn, create_document


LEGACY = "sjtu-scs-2024-master-cn-capacity-demo/3"
CURRENT = "sjtu-scs-2025-master-cn-capacity-demo/1"


def request():
    return BriefCreate(operation_id=uuid4(), project_id=uuid4(), message_id=uuid4())


def test_legacy_documents_and_omitted_create_field_preserve_serialization():
    command = request()
    assert "school_profile_id" not in command.model_dump(mode="json")
    legacy = BriefDocument(id=uuid4(), project_id=command.project_id, revision=1)
    original = legacy.model_dump(mode="json")
    assert "school_profile_id" not in original
    assert BriefDocument.model_validate_json(canonical(original), strict=True).model_dump(mode="json") == original
    current = create_document(command)
    assert current.school_profile_id == CURRENT
    assert "school_profile_id" not in command.model_dump(mode="json")


def test_explicit_profile_change_makes_new_revision_and_retains_previous_document():
    original = create_document(request())
    original_sha = digest(original.model_dump(mode="json"))
    changed = apply_turn(original, BriefTurn(operation_id=uuid4(), message_id=uuid4(),
        expected_revision=original.revision, school_profile_id=LEGACY))
    assert changed.revision == original.revision + 1 and changed.school_profile_id == LEGACY
    assert original.school_profile_id == CURRENT and digest(original.model_dump(mode="json")) == original_sha
    assert changed.facts == original.facts and changed.proposals == original.proposals
    assert digest(changed.model_dump(mode="json")) != original_sha


def test_unknown_school_profile_is_rejected_at_intake():
    with pytest.raises(ValidationError):
        BriefCreate(operation_id=uuid4(), project_id=uuid4(), message_id=uuid4(), school_profile_id="unknown/1")
