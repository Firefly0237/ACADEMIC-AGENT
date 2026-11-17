"""Minimal new-content entry controls with explicit synthetic graph doubles."""
import copy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from citequest.agents_v3.answer_patch import AnswerPatch
from citequest.agents_v3.contracts import PROFILES, canonical
from citequest.agents_v3.product_profiles import QA8_PROTOCOL_METADATA
from citequest.agents_v3.qa_v5 import QA5
from citequest.agents_v3.qa_v8 import QA8
from citequest.agents_v3.qa_v9 import QA9
from tests.v3.test_qa_anchored_audit import anchored_fixture


def node(profile):
    value = QA9.__new__(QA9)
    value.graph = SimpleNamespace(execution=SimpleNamespace(snapshot=lambda: {
        "kind": "qa", "profile": profile}))
    return value


def test_legacy_none_uses_original_models_and_policies_without_a_catalog():
    profile = {**PROFILES["qa-v9"].model_dump(), **QA8_PROTOCOL_METADATA}
    current = node(profile)
    assert current.answer_model_for({}) is QA8.answer_model
    state = {"qa5": {"lock": None, "gap_only_lock": None}}
    for role in ("plan", "author", "review"):
        assert current.role_policy(state, role) == QA8.role_policy(current, state, role)
    assert QA5.answer_model_for(current, {}) is QA8.answer_model


def test_free_gap_is_rejected_at_author_entry_and_scoped_patch_uses_the_same_catalog():
    _audit, _opinions, candidate, grid, _base = anchored_fixture()
    current = node({**PROFILES["qa-v9"].model_dump(), **QA8_PROTOCOL_METADATA,
        "qa_content_protocol": "anchored2"})
    answer = {"kind": "qa", "requested_direction": "neutral", "claims": [{
        "id": "C1", "statement": "Synthetic unresolved proposition.", "status": "insufficient",
        "edges": [], "limitations": ["Synthetic evidence boundary."]}],
        "gaps": ["Invented source absence."], "retrieval_scope": "Synthetic scope.",
        "evidence_status": "insufficient", "presentation": {"format": "prose", "paragraphs": [["C1"]]}}
    data = {"candidate": candidate.model_dump(), "turn": candidate.user_turn.model_dump(),
        "author_raw": canonical(answer).decode(), "original_propositions": None,
        "lock": None, "gap_only_lock": None, "revision": 0}
    current.key = lambda _data, role: "synthetic-command:qa5:" + role + ":0"
    result = current.validate_author({"qa5": data, "evidence": []})
    assert "answer" not in result
    assert result["qa5"]["target"] == "author"
    assert result["qa5"]["feedback"]["error_codes"] == ["literal_error"]
    assert result["qa5"]["author_raw"] == data["author_raw"]

    state = {"qa5": data}
    model = current.answer_model_for(state)
    patch = AnswerPatch(answer, model, [], gap_only=True)
    invalid = {"baseline_sha256": patch.baseline_sha256, "operations": [
        {"op": "replace", "path": "/gaps", "value": ["Invented source absence."]}]}
    with pytest.raises(ValidationError):
        patch.apply(canonical(invalid).decode())
    allowed = current.gap_catalog(state).projection["typed_options"][0]["text"]
    valid = copy.deepcopy(invalid)
    valid["operations"][0]["value"] = [allowed]
    merged, _receipt = patch.apply(canonical(valid).decode())
    assert merged == {**answer, "gaps": [allowed]}
    assert current.gap_catalog(state).rejections(merged) == []
