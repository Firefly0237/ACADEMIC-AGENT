"""Two pure catalog controls. No model, source truth, SQL or HTTP is simulated."""
import copy

import pytest
from pydantic import Field, ValidationError

from citequest.agents_v3.contracts import digest
from citequest.agents_v3.qa_gap_catalog import GapCatalog
from citequest.research_v3.contracts import Strict
from citequest.research_v3.current_requirements import RequirementsCandidate


class SyntheticAnswer(Strict):
    gaps: list[str] = Field(max_length=2)
    unchanged: str


def candidate(*, format_only=False):
    long_text = "甲" * 4500
    text = long_text + "\n两段中文\n旧关注"
    turn_sha = digest(text)
    later = "撤回旧关注"

    def requirement(identifier, category, start, content, *, retired=False):
        return {"id": identifier, "category": category,
            "spans": [{"turn_id": "U1", "start": start + offset,
                "end": start + min(offset + 1500, len(content)), "text": content[offset:offset + 1500],
                "turn_sha256": turn_sha} for offset in range(0, len(content), 1500)],
            "state": "withdrawn" if retired else "active", "introduced_turn_id": "U1",
            "retired_turn_id": "U2" if retired else None, "replacement_ids": []}

    requirements = [requirement("R1", "format" if format_only else "question", 0, long_text),
        requirement("R2", "format", 4501, "两段中文"),
        requirement("R3", "focus", 4506, "旧关注", retired=True)]
    return RequirementsCandidate.model_validate({
        "base_sha256": "a" * 64, "delta_sha256": "b" * 64,
        "user_turn": {"turn_id": "U2", "text": later},
        "projected": {"authority_sha256": "a" * 64, "revision": 2,
            "turns": [{"turn_id": "U1", "text_sha256": turn_sha}, {"turn_id": "U2", "text_sha256": digest(later)}],
            "requirements": requirements, "accepted": False},
        "required_review_ids": ["R1", "R2", "R3"], "clarification": None}, strict=True)


def test_catalog_preserves_complete_spans_and_rejects_free_or_retired_gaps():
    original = candidate()
    catalog = GapCatalog(original)
    options = catalog.projection["typed_options"]
    assert [(item["requirement_id"], item["kind"]) for item in options] == [
        ("R1", "unconfirmed"), ("R1", "layout_uncertain")]
    complete_spans = "\n".join(span.text for span in original.projected.requirements[0].spans)
    assert all(complete_spans in item["text"] for item in options)
    assert len(complete_spans.replace("\n", "")) == 4500
    model = catalog.answer_model(SyntheticAnswer)
    value = {"gaps": [options[0]["text"]], "unchanged": "Synthetic untouched value."}
    assert model.model_validate(value, strict=True).model_dump() == value
    assert model.model_json_schema()["properties"]["gaps"]["maxItems"] == 2
    assert GapCatalog(original.model_copy(deep=True)).binding == catalog.binding
    wrong = copy.deepcopy(value)
    wrong["gaps"].append("Source contains no measurements.")
    assert catalog.rejections(wrong) == ["G2"]
    with pytest.raises(ValidationError):
        model.model_validate(wrong, strict=True)
    assert SyntheticAnswer.model_validate(wrong, strict=True).model_dump() == wrong
    assert catalog.rejections(value) == []


def test_format_only_catalog_allows_only_an_empty_original_string_list():
    catalog = GapCatalog(candidate(format_only=True))
    model = catalog.answer_model(SyntheticAnswer)
    assert catalog.projection["typed_options"] == []
    assert model.model_json_schema()["properties"]["gaps"]["maxItems"] == 0
    assert model.model_validate({"gaps": [], "unchanged": "Synthetic value."}, strict=True).gaps == []
    with pytest.raises(ValidationError):
        model.model_validate({"gaps": ["Invented source absence."], "unchanged": "Synthetic value."}, strict=True)
    assert catalog.rejections({"gaps": ["Invented source absence."]}) == ["G1"]
