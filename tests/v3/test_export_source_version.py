"""Exact export source selection never falls back to the current draft."""
from uuid import uuid4

import pytest

from citequest.agents_v3.contracts import ReviewSource
from citequest.exports_v3.contracts import ExportCreate
from citequest.exports_v3.service import ExportProfile, ExportService
from citequest.v2.domain import Actor, DomainError


def binding(task_id):
    return ReviewSource(source_task_id=task_id, source_command_id=uuid4(), command_seq=1,
        event_seq=2, task_revision=1, document_sha256="a" * 64, context_sha256="b" * 64)


def service(calls, *, version=True):
    def current(actor, task_id):
        calls.append(("current", task_id))
        return {"id": task_id, "kind": "qa", "state": "succeeded"}

    def exact(actor, task_id, source):
        calls.append(("exact", source))
        return {"id": task_id, "kind": "qa", "state": "succeeded"}

    return ExportService(None, current, None,
        ExportProfile("test-export", "test-fonts", "c" * 64), None,
        authorize_version=exact if version else None)


def test_legacy_export_serialization_stays_unchanged():
    request = ExportCreate(operation_id=uuid4(), formats=["docx"])
    assert "source_binding" not in request.model_dump(mode="json")


def test_pinned_expected_job_uses_only_exact_source_authorizer():
    calls, task_id = [], uuid4()
    pin = binding(task_id)
    # Stop after source selection; this test does not claim document/render QA.
    with pytest.raises(DomainError, match="reviewed_school_task_required"):
        service(calls).document(Actor(uuid4()), task_id, {"source_binding": pin.model_dump(mode="json")})
    assert calls == [("exact", pin)]


def test_unpinned_legacy_job_uses_original_authorizer():
    calls, task_id = [], uuid4()
    with pytest.raises(DomainError, match="reviewed_school_task_required"):
        service(calls).document(Actor(uuid4()), task_id, {"source_binding": None})
    assert calls == [("current", task_id)]


@pytest.mark.parametrize("mode", ["missing_authorizer", "different_task", "different_version"])
def test_invalid_exact_binding_never_dispatches_current_authorizer(mode):
    calls, task_id = [], uuid4()
    pin = binding(task_id)
    target = service(calls, version=mode != "missing_authorizer")
    if mode == "different_task":
        task_id = uuid4()
    expected = {"source_binding": pin.model_dump(mode="json")}
    arguments = {"source_binding": pin.model_copy(update={"event_seq": 3})} if mode == "different_version" else {}
    with pytest.raises(DomainError, match="export_(exact_source_authorizer_required|source_version_conflict)"):
        target.document(Actor(uuid4()), task_id, expected, **arguments)
    assert calls == []
