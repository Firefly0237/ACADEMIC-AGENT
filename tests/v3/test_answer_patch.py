"""Three pure RFC6902 controls; graph and parent node are explicit doubles."""
import copy
from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import Field, ValidationError

from citequest.agents_v3.answer_patch import AnswerPatch
from citequest.agents_v3.contracts import AgentError, canonical, digest
from citequest.agents_v3.qa_audited_dialogue import GapOnlyRepairLock
from citequest.agents_v3.qa_v7 import PATCH_AUTHOR_POLICY, QA7
from citequest.agents_v3.qa_verified_dialogue import QADialogueVerified
from citequest.research_v3.contracts import Strict
from citequest.research_v3.typed_feedback import RepairLock, strict_object


class Claim(Strict):
    id: str
    statement: str = Field(min_length=1, max_length=200)


class Answer(Strict):
    claims: list[Claim] = Field(min_length=1, max_length=8)
    gaps: list[str] = Field(max_length=8)
    evidence_status: Literal["supported", "insufficient"]


def baseline():
    return {"claims": [{"id": "C1", "statement": "Preserve this proposition."},
        {"id": "C7", "statement": "The proposition needing source-based correction."}],
        "gaps": ["A bounded limitation."], "evidence_status": "supported"}


def submission(patch, operations):
    return canonical({"baseline_sha256": patch.baseline_sha256, "operations": operations}).decode()


def replacement():
    return {"op": "replace", "path": "/claims/1",
        "value": {"id": "C7", "statement": "A corrected proposition requiring independent review."}}


def test_patch_is_deterministic_preserves_unaffected_values_and_original_locks():
    original = baseline()
    patch = AnswerPatch(original, Answer, ["C7"])
    raw = submission(patch, [replacement()])
    merged, receipt = patch.apply(raw)
    assert patch.apply(raw) == (merged, receipt)
    assert original == baseline() and merged["claims"][0] == original["claims"][0]
    assert merged["gaps"] == original["gaps"] and merged["claims"][1] == replacement()["value"]
    RepairLock.bind(original, ["C7"]).verify(merged)
    Answer.model_validate(merged, strict=True)
    assert receipt == {"protocol": "jsonpatch1", "baseline_sha256": digest(original),
        "patch_sha256": digest(strict_object(raw)), "assembled_candidate_sha256": digest(merged),
        "paths": ["/claims/1"]}
    assert "/claims/0" not in patch.allowed_paths and patch.required_paths == ["/claims/1"]
    assert "anyOf" in canonical(patch.model.model_json_schema()).decode()
    gaps = AnswerPatch(original, Answer, [], gap_only=True)
    revised, _ = gaps.apply(submission(gaps, [{"op": "replace", "path": "/gaps", "value": []}]))
    GapOnlyRepairLock.bind(original).verify(revised)
    assert gaps.allowed_paths == gaps.required_paths == ["/gaps"]


def test_patch_rejects_unauthorized_paths_operations_identity_hash_and_duplicates():
    original = baseline()
    patch = AnswerPatch(original, Answer, ["C7"])
    forbidden = [
        [{**replacement(), "path": "/claims/0"}],
        [{**replacement(), "op": "add"}],
        [{**replacement(), "op": "remove"}],
        [{**replacement(), "op": "move", "from": "/claims/0"}],
        [{**replacement(), "op": "copy", "from": "/claims/0"}],
        [{**replacement(), "value": {"id": "C1", "statement": "Renamed."}}],
        [{"op": "replace", "path": "/gaps", "value": []}],
        [replacement(), replacement()],
    ]
    for operations in forbidden:
        with pytest.raises((ValidationError, AgentError)):
            patch.apply(submission(patch, operations))
        assert original == baseline()
    with pytest.raises(ValidationError):
        patch.apply(canonical({"baseline_sha256": "0" * 64, "operations": [replacement()]}).decode())
    gaps = AnswerPatch(original, Answer, [], gap_only=True)
    with pytest.raises(ValidationError):
        gaps.apply(submission(gaps, [replacement()]))


def test_qa7_routes_patch_and_delegates_complete_assembled_candidate_to_original_validator(monkeypatch):
    original = baseline()
    workflow = object.__new__(QA7)
    workflow.answer_model = Answer
    workflow.graph = SimpleNamespace(execution=SimpleNamespace(snapshot=lambda: {"kind": "qa", "profile": {
        "profile_id": "qa-v7", "qa_execution_protocol": "grounded3",
        "review_api_protocol": "chat-strict2", "review_transport_protocol": "function1",
        "structured_output_protocol": "chat-functions1", "answer_revision_protocol": "jsonpatch1"}}))
    workflow.prompt = lambda _state: ({"context_manifest": {"hard_memory_refs": []}}, {})
    workflow.scope = lambda _state: SimpleNamespace(contexts=[], binding={}, warnings=[],
        schema=lambda model: model.model_json_schema())
    state = {"qa5": {"lock": RepairLock.bind(original, ["C7"]).__dict__, "gap_only_lock": None,
        "feedback": {"error_codes": ["source_review_rejected"]}, "original_propositions": {},
        "author_raw": canonical(original).decode()}}
    calls = []

    def send(_state, role, schema, prompt):
        assert role == "author" and "operations" in schema["properties"]
        assert prompt["untrusted_previous_candidate"] == original
        assert workflow.role_policy(_state, role) == PATCH_AUTHOR_POLICY
        return submission(workflow.answer_patch(_state), [replacement()]), {"hard_memory_refs": []}

    def original_validator(_self, assembled_state):
        candidate = strict_object(assembled_state["qa5"]["author_raw"])
        RepairLock(**assembled_state["qa5"]["lock"]).verify(candidate)
        Answer.model_validate(candidate, strict=True)
        calls.append(copy.deepcopy(assembled_state))
        return {"qa5": assembled_state["qa5"], "answer": candidate}

    workflow.send = send
    monkeypatch.setattr(QADialogueVerified, "validate_author", original_validator)
    generated = workflow.author(state)
    assert generated["qa5"]["author_raw"] == state["qa5"]["author_raw"]
    assert generated["qa5"]["answer_patch_pending"]
    result = workflow.validate_author({**state, **generated})
    assert len(calls) == 1 and not result["qa5"]["answer_patch_pending"]
    assert result["answer"]["claims"][1] == replacement()["value"]
    record = result["qa5"]["answer_patch_records"][0]
    assert record["baseline"] == original and record["assembled_candidate"] == result["answer"]
    assert result["qa5"]["manifest"]["answer_patch"]["assembled_candidate_sha256"] == digest(result["answer"])
    assert state["qa5"]["author_raw"] == canonical(original).decode()
