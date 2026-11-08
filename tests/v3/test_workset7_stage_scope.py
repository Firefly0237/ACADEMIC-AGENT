"""One scope control with explicit manifest/context fixtures; no provider call."""
import copy
from types import SimpleNamespace

from citequest.agents_v3.workset_v5_claims import DirectedClaimFlow
from citequest.agents_v3.workset_v6 import Workset6Sources
from citequest.agents_v3.workset_v7 import Workset7Sources
from citequest.research_v3.quote_spans import QuoteRegistry
from citequest.research_v3.workset_v5_contracts import SelectorSummaryV5, SummaryV5
from tests.v3.test_quote_spans_v3 import source, wire


def test_workset7_local_assignment_preserves_global_availability_and_full_source_review(monkeypatch):
    evidence = [source("Complete synthetic local source.")]
    documents = [SimpleNamespace(document_id=f"D0{i}", source_version_sha256=str(i) * 64,
        coverage_kind="native_literal", available_codepoints=30,
        selection=SimpleNamespace(model_dump=lambda number=i: {"kind": "native_representation", "id": str(number) * 64}),
        windows=[SimpleNamespace(window_id=f"window-{i}")]) for i in (1, 2)]
    owner = Workset7Sources.__new__(Workset7Sources)
    owner.manifest = lambda _data: SimpleNamespace(documents=documents)  # Explicit server-manifest fixture.
    anchor = {"typed_requirements": {"dialogue_reference": None}}
    data = {"anchor": anchor, "windows": {"window-1": {}, "window-2": {}}}
    job = owner.make_job(data, job_id="D01:final_review", role="reduce", evidence=evidence,
        contexts=evidence, task_data={"document_id": "D01"}, review_required=True)
    assignment = job["coverage_assignment"]
    assert assignment == job["task_data"]["coverage_assignment"]
    assert assignment["stage"] == "document_contribution" and assignment["document_ids"] == ["D01"]
    assert [row["read_windows"] for row in assignment["workset_documents"]] == [1, 1]
    assert [row["in_current_stage_evidence"] for row in assignment["workset_documents"]] == [True, False]
    assert all(row["complete_pdf_claimed"] is False for row in assignment["workset_documents"])
    atomic = {"claim_id": "C1", "question": "Compare the assigned method dimension.",
        "requirement_ids": ["R1"], "document_ids": ["D01", "D02"]}
    comparison = owner.make_job(data, job_id="atomic:C1", role="atomic_author", evidence=evidence,
        contexts=evidence, atomic=atomic)
    assert comparison["coverage_assignment"]["stage"] == "atomic_comparison"
    assert comparison["coverage_assignment"]["question"] == atomic["question"]
    legacy = Workset6Sources.make_job(owner, data, job_id="legacy", role="map", evidence=evidence,
        contexts=evidence, task_data={"document_id": "D01"})
    assert "coverage_assignment" not in legacy and "coverage_assignment" not in legacy["task_data"]

    original = {"context_manifest": {"source_scope": [
        {"selection": {"id": "one"}, "state": "bounded_literal_windows"},
        {"selection": {"id": "two"}, "state": "uncovered"}]}}
    monkeypatch.setattr(Workset6Sources, "project", lambda *_args, **_kwargs: (
        copy.deepcopy(original), copy.deepcopy(original["context_manifest"])))
    projected, manifest = owner.project(anchor, evidence=evidence)
    assert [row["state"] for row in manifest["source_scope"]] == ["bounded_literal_windows", "not_in_current_stage"]
    assert original["context_manifest"]["source_scope"][1]["state"] == "uncovered"
    assert "coverage_assignment" not in projected  # Global roles keep global requirements.

    initial = QuoteRegistry(evidence).scope(evidence)
    candidate = initial.expand(SelectorSummaryV5.model_validate(wire("Q1"), strict=True), SummaryV5).model_dump()
    flow = DirectedClaimFlow.__new__(DirectedClaimFlow)
    flow.scope = lambda state: QuoteRegistry(evidence).scope(evidence, candidates=[state["candidate"]])
    captured = []
    flow.send = lambda state, role, schema, payload, **kwargs: (captured.append((role, payload, schema)) or "{}", {})
    state = {"job": job, "candidate": candidate, "target": "author", "feedback": None}
    flow.review(state)
    role, payload, schema = captured[0]
    assert role == "document_review" and payload["coverage_assignment"] == assignment
    assert payload["review_grid"]["request_coverage_required"] is True
    assert set(payload["review_grid"]["claim_quote_ids"]) == {"C1"}
    assert "claims" in schema["properties"] and payload["untrusted_evidence_data"]
    author_schema, author_payload = flow.author_parts(state)
    assert author_payload["coverage_assignment"] == assignment and "claims" in author_schema["properties"]
    assert job["evidence"] == evidence and job["contexts"] == evidence
