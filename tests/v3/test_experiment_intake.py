"""Action-specific intake retains unknown facts, source selection and version."""
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import PROFILES, TaskCreate, digest, graph_version_for_kind
from citequest.research_v3.brief import (
    EXPERIMENT_REQUIRED, BriefCreate, Fact, Proposal, create_document, planning_payload, readiness,
)
from citequest.v2.domain import DomainError


def intake():
    return create_document(BriefCreate.model_validate({
        "operation_id": str(uuid4()), "project_id": str(uuid4()), "message_id": str(uuid4()),
        "patches": [{"field": field, "intent": "unknown" if field == "data_sources" else "set",
                     "value": "尚未确定" if field == "data_sources" else "Synthetic proposed research"}
                    for field in EXPERIMENT_REQUIRED],
        "selected_sources": [{"kind": "material_file", "id": str(uuid4())}],
    }))


def test_experiment_handoff_preserves_unknown_and_does_not_require_school_bibliography():
    document = intake()
    before = digest(document.model_dump(mode="json"))
    assert readiness(document)["experiment_readiness"]["state"] == "ready_for_experiment"
    payload = planning_payload(document, action="experiment")
    assert payload["explicit_unknowns"]["data_sources"] == "尚未确定"
    assert "data_sources" not in payload["user_confirmed"]
    assert payload["selected_sources"] == [item.model_dump(mode="json") for item in document.selected_sources]
    assert payload["brief_sha256"] == before == digest(document.model_dump(mode="json"))
    with pytest.raises(DomainError, match="research_material_bibliography_required"):
        planning_payload(document)


def test_open_conflicts_block_experiment_instead_of_auto_selecting_a_fact():
    document = intake()
    conflict = Proposal(id=uuid4(), field="topic", kind="conflict",
                        candidate=Fact(value="Conflicting topic", status="confirmed", message_id=uuid4()))
    document = document.model_copy(update={"proposals": [conflict]})
    assert readiness(document)["experiment_readiness"]["open_conflicts"] == [str(conflict.id)]
    with pytest.raises(DomainError, match="research_brief_needs_input"):
        planning_payload(document, action="experiment")


def test_new_experiment_action_requires_explicit_new_graph_and_same_budget():
    document = intake()
    request = {"project_id": str(document.project_id), "operation_id": str(uuid4()), "kind": "research",
               "profile_id": "research-v6", "brief": {"kind": "research", "research_action": "experiment",
               "research_brief": {"id": str(document.id), "revision": document.revision,
                                  "sha256": digest(document.model_dump(mode="json"))}}}
    TaskCreate.model_validate(request)
    with pytest.raises(ValidationError):
        TaskCreate.model_validate({**request, "profile_id": "research-v5"})
    assert graph_version_for_kind("research", "research-v6") != graph_version_for_kind("research", "research-v5")
    old, new = (PROFILES[key].model_dump(exclude={"profile_id"}) for key in ("research-v5", "research-v6"))
    assert old == new
