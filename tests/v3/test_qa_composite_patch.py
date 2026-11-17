"""Two pure composite-repair controls; protocol/model objects are explicit doubles."""
import copy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from citequest.agents_v3 import qa_v7
from citequest.agents_v3.answer_patch import AnswerPatch
from citequest.agents_v3.contracts import AgentError, canonical, digest
from citequest.agents_v3.qa_separated import QASeparated, VERSION
from citequest.agents_v3.qa_v7 import QA7
from citequest.research_v3.contracts import Strict
from citequest.research_v3.typed_feedback import RepairLock


class Claim(Strict):
    id: str
    statement: str


class Answer(Strict):
    claims: list[Claim]
    gaps: list[str]
    presentation: dict


def baseline():
    return {"claims": [{"id": "C1", "statement": "Unaffected proposition."},
        {"id": "C5", "statement": "Proposed work, not measured results."}],
        "gaps": ["Synthetic rejected gap."],
        "presentation": {"format": "prose", "paragraphs": [["C1", "C5"]]}}


def submission(patch, operations):
    return canonical({"baseline_sha256": patch.baseline_sha256, "operations": operations}).decode()


def test_composite_patch_requires_each_reviewed_field_without_changing_legacy_contract():
    original = baseline()
    claim = {"op": "replace", "path": "/claims/1", "value": {
        "id": "C5", "statement": "The source explicitly reports unimplemented proposed work."}}
    gaps = {"op": "replace", "path": "/gaps", "value": []}
    legacy = AnswerPatch(original, Answer, ["C5"])
    explicit_default = AnswerPatch(original, Answer, ["C5"], required_fields=())
    assert legacy.model.model_json_schema() == explicit_default.model.model_json_schema()
    assert legacy.contract() == explicit_default.contract()
    assert legacy.contract()["required_paths_exactly_once"] == ["/claims/1"]
    assert legacy.contract()["rule"] == (
        "Replace each affected claim exactly once. Preserve its ID. Other advertised fields are optional replacements. "
        "The server retains all other original values and independently validates the complete assembled answer.")
    assert legacy.apply(submission(legacy, [claim]))[0]["gaps"] == original["gaps"]

    patch = AnswerPatch(original, Answer, ["C5"], required_fields=("gaps",))
    assert patch.contract()["required_paths_exactly_once"] == ["/claims/1", "/gaps"]
    with pytest.raises((ValidationError, AgentError)):
        patch.apply(submission(patch, [claim]))
    merged, receipt = patch.apply(submission(patch, [claim, gaps]))
    assert merged["gaps"] == [] and receipt["paths"] == ["/claims/1", "/gaps"]
    assert canonical(merged["claims"][0]) == canonical(original["claims"][0])
    RepairLock.bind(original, ["C5"]).verify(merged)
    Answer.model_validate(merged, strict=True)
    assert original == baseline()
    for operations in ([claim, gaps, gaps], [claim, {**gaps, "path": "/budget"}]):
        with pytest.raises((ValidationError, AgentError)):
            patch.apply(submission(patch, operations))
    with pytest.raises(ValidationError):
        patch.apply(canonical({"baseline_sha256": "0" * 64, "operations": [claim, gaps]}).decode())
    for fields in (("gaps", "gaps"), ("claims",), ("budget",), ("/gaps",)):
        with pytest.raises(AgentError, match="qa_answer_patch_required_fields_invalid"):
            AnswerPatch(original, Answer, ["C5"], required_fields=fields)


def test_separated_repair_uses_only_matching_server_assessment_and_wires_required_paths(monkeypatch):
    original = baseline()
    candidate = {"synthetic_current_requirements": ["R1", "R2"]}
    binding = {"protocol": "citequest-qa-answer-audit/1", "separated_review_protocol": VERSION,
        "compact_answer_sha256": digest(original), "gap_ids": ["G1"],
        "requirements_candidate_sha256": digest(candidate)}
    assessment = {"binding": copy.deepcopy(binding), "gaps_rejected": ["G1"],
        "requirement_coverage": {"R1": {"status": "fulfilled", "locations": ["C1"], "explanation": "Answered."},
            "R2": {"status": "format_mismatch", "locations": ["presentation"], "explanation": "Wrong paragraphs."}}}
    state = {"qa5": {"lock": RepairLock.bind(original, ["C5"]).__dict__, "gap_only_lock": None,
        "answer_audit": assessment, "answer_audit_binding": binding, "candidate": candidate}}
    workflow = QASeparated.__new__(QASeparated)
    assert workflow.required_repair_fields(state) == ("gaps", "presentation")
    assert QA7.required_repair_fields(workflow, state) == ()
    workflow.answer_model_for = lambda _state: Answer  # Explicit server-schema double.
    workflow.graph = SimpleNamespace(execution=SimpleNamespace(snapshot=lambda: {}))
    monkeypatch.setattr(qa_v7, "answer_revision_protocol", lambda _snapshot: "jsonpatch1")  # No network/DB.
    assert workflow.answer_patch(state).required_paths == ["/claims/1", "/gaps", "/presentation"]

    claim_format = copy.deepcopy(state)
    claim_format["qa5"]["answer_audit"]["requirement_coverage"]["R2"]["locations"] = ["C1"]
    assert workflow.required_repair_fields(claim_format) == ("gaps",)
    structural = {"qa5": {"lock": state["qa5"]["lock"], "feedback": {"schema_version": "citequest-typed-feedback/1",
        "issues": [{"location": ["presentation"], "code": "missing"}]}}}
    assert workflow.required_repair_fields(structural) == ()  # No candidate-bound source review.
    for stale in ("assessment", "baseline", "requirements"):
        changed = copy.deepcopy(state)
        if stale == "assessment":
            changed["qa5"]["answer_audit"]["binding"]["compact_answer_sha256"] = "0" * 64
        elif stale == "baseline":
            changed["qa5"]["lock"]["baseline_json"] = canonical({**original, "gaps": []}).decode()
        else:
            changed["qa5"]["candidate"] = {"synthetic_current_requirements": ["R3"]}
        with pytest.raises(AgentError, match="qa_composite_repair_binding_changed"):
            workflow.required_repair_fields(changed)
