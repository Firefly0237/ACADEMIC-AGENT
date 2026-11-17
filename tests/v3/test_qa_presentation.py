"""Two pure presentation controls. Statements and evidence gaps are synthetic."""
import copy

import pytest
from pydantic import ValidationError

from citequest.agents_v3.qa_presentation import PresentedAnswer, PresentedSelectorAnswer, render_presented_answer
from citequest.research_v3.typed_feedback import validation_feedback


def candidate():
    return {"kind": "qa", "requested_direction": "neutral", "claims": [
        {"id": identifier, "statement": statement, "status": "insufficient", "edges": [],
            "limitations": ["Synthetic evidence gap"]}
        for identifier, statement in (("C1", "第一项条件待核。"), ("C2", r"第二项条件为 $\alpha = 0.1$，证据待核。"),
            ("C3", "第三项条件待核。"))], "gaps": ["Synthetic evidence gap"],
        "retrieval_scope": "Synthetic bounded input only", "evidence_status": "insufficient",
        "presentation": {"format": "prose", "paragraphs": [["C2", "C1"], ["C3"]]}}


def test_paragraphs_partition_all_claims_and_keep_original_source_validators():
    value = candidate()
    for model in (PresentedSelectorAnswer, PresentedAnswer):
        assert model.model_validate(value, strict=True).presentation.paragraphs == [["C2", "C1"], ["C3"]]
        for paragraphs in ([["C1", "C2"]], [["C1", "C2"], ["C1", "C3"]], [["C1", "C2"], ["C9"]], [[]]):
            broken = {**value, "presentation": {"format": "prose", "paragraphs": paragraphs}}
            with pytest.raises(ValidationError):
                model.model_validate(broken, strict=True)
        broken = copy.deepcopy(value)
        broken["claims"][0]["status"] = "supported"
        with pytest.raises(ValidationError):
            model.model_validate(broken, strict=True)
        with pytest.raises(ValidationError):
            model.model_validate({**value, "presentation": {"paragraphs": [["C1", "C2", "C3"]]}}, strict=True)
    broken = {**value, "presentation": {"format": "prose", "paragraphs": [["C1", "C2"]]}}
    with pytest.raises(ValidationError) as error:
        PresentedSelectorAnswer.model_validate(broken, strict=True)
    feedback = validation_feedback(error.value, broken)
    assert feedback["repairable"] is True
    assert feedback["error_codes"] == ["qa_presentation_claim_partition"]
    assert feedback["affected_claim_ids"] == ["C1", "C2", "C3"]  # Existing root-error repair semantics.


def test_render_two_prose_paragraphs_and_preserve_block_math_and_code_verbatim():
    value = candidate()
    answer = PresentedAnswer.model_validate(value, strict=True)
    rendered = render_presented_answer(answer)
    expected = f"[C2] {value['claims'][1]['statement']} [C1] {value['claims'][0]['statement']}\n\n[C3] {value['claims'][2]['statement']}"
    assert rendered == expected and len(rendered.split("\n\n")) == 2
    assert answer.model_dump()["claims"] == value["claims"]
    block = "数学条件待核。\n\n$$\\alpha = 0.1$$\n\n```python\nresult = 1\n```"
    value["claims"][0]["statement"] = block
    with pytest.raises(ValidationError) as error:
        PresentedSelectorAnswer.model_validate(value, strict=True)
    assert validation_feedback(error.value, value)["error_codes"] == ["qa_presentation_inline_statement"]
    value["presentation"]["format"] = "blocks"
    selected = PresentedSelectorAnswer.model_validate(value, strict=True)
    expanded = PresentedAnswer.model_validate(selected.model_dump(), strict=True)
    assert expanded.claims[0].statement == block
    expected_blocks = "\n\n".join(f"[{identifier}]\n\n{next(c['statement'] for c in value['claims'] if c['id'] == identifier)}"
        for paragraph in value["presentation"]["paragraphs"] for identifier in paragraph)
    assert render_presented_answer(expanded) == expected_blocks and block in expected_blocks
