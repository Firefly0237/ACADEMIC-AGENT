"""Pure selector/lifecycle controls. Review opinions here are synthetic fixtures."""
import copy
import json

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import digest
from citequest.agents_v3.qa_dialogue import QADialogueGrounded
from citequest.agents_v3.qa_v5 import QA5
from citequest.agents_v3.qa_v6 import QA6
from citequest.agents_v3.requirements_turn_units import resolve_selected_plan, selected_plan_model, turn_units
from citequest.research_v3.current_requirements import RequirementReview, UserTurn, accept_review, initial_state, propose_delta
from citequest.research_v3.typed_feedback import strict_object


def unit_id(turn, literal):
    return next(item["unit_id"] for item in turn_units(turn)["units"] if item["text"] == literal)


def wire_plan(state, turn, operations, retained=()):
    return {"requirement_delta": {"version": "citequest-current-requirements/1",
        "parent_sha256": digest(state.model_dump(mode="json")), "authority_sha256": state.authority_sha256,
        "turn_id": turn.turn_id, "turn_sha256": digest(turn.text), "retain_ids": list(retained),
        "operations": operations, "intent": "clear", "clarification": None}, "action": "retrieve",
        "queries": [{"query": "method comparison parameter setting", "limit": 3}], "existing_ref_ids": []}


def fixture_accept(previous, candidate):
    review = RequirementReview(candidate_sha256=digest(candidate.model_dump(mode="json")),
        user_turn_sha256=digest(candidate.user_turn.text),
        opinions=[{"requirement_id": identifier, "verdict": "correct", "rationale": "synthetic control opinion"}
            for identifier in candidate.required_review_ids], current_turn_coverage="complete", decision="accepted")
    return accept_review(candidate, review, current_state=previous)


def test_location_catalog_preserves_unicode_full_input_and_repeated_positions():
    turn = UserTurn(turn_id="U1", text="只看甲，保留α=0.1。\n只看甲，证据不足就说明。🙂")
    units = turn_units(turn)
    assert units == turn_units(turn) and units["semantic_interpretation"] is False
    covered = set()
    repeated = []
    for item in units["units"]:
        assert item["text"] == turn.text[item["start"]:item["end"]]
        assert 0 < item["end"] - item["start"] <= 1500
        covered.update(range(item["start"], item["end"]))
        if item["text"] == "只看甲，":
            repeated.append((item["unit_id"], item["start"]))
    assert covered == set(range(len(turn.text)))
    assert len(repeated) == 2 and repeated[0] != repeated[1]


def test_selected_ids_resolve_exact_spans_and_retain_original_plan_validators():
    state = initial_state("a" * 64)
    turn = UserTurn(turn_id="U1", text="比较方法甲与乙？说明参数条件，保留负面结果。")
    operations = [{"action": "add", "target_ids": [], "category": category, "spans": [unit_id(turn, text)]}
        for category, text in (("question", "比较方法甲与乙？"), ("detail", "说明参数条件，"), ("constraint", "保留负面结果。"))]
    wire = wire_plan(state, turn, operations)
    plan = resolve_selected_plan(json.dumps(wire), state, turn)
    candidate = propose_delta(state, turn, plan.requirement_delta)
    assert candidate.required_review_ids == ["R1", "R2", "R3"]
    assert [item.spans[0].text for item in candidate.projected.requirements] == ["比较方法甲与乙？", "说明参数条件，", "保留负面结果。"]
    wrong = copy.deepcopy(wire)
    wrong["requirement_delta"]["operations"][0]["spans"] = ["U2.S1"]
    with pytest.raises(ValidationError):
        resolve_selected_plan(json.dumps(wrong), state, turn)
    wrong = copy.deepcopy(wire)
    wrong["requirement_delta"]["parent_sha256"] = "b" * 64
    with pytest.raises(ValidationError):
        resolve_selected_plan(json.dumps(wrong), state, turn)
    wrong = copy.deepcopy(wire)
    wrong["queries"][0]["limit"] = 21
    with pytest.raises(ValidationError):
        resolve_selected_plan(json.dumps(wrong), state, turn)
    schema = selected_plan_model(state, turn).model_json_schema()
    operation = next(value for key, value in schema["$defs"].items() if key.startswith("SelectedRequirementOperation_"))
    assert operation["properties"]["spans"]["items"]["type"] == "string"
    assert set(operation["properties"]["spans"]["items"]["enum"]) == {item["unit_id"] for item in turn_units(turn)["units"]}


@pytest.mark.parametrize("workflow_type", [QA5, QA6])
def test_legacy_parse_hook_is_identical_to_the_original_expression(workflow_type):
    accepted = initial_state("a" * 64)
    turn = UserTurn(turn_id="U1", text="比较方法甲与乙。")
    wire = wire_plan(accepted, turn, [{"action": "add", "target_ids": [], "category": "question",
        "spans": [unit_id(turn, turn.text)]}])
    normalized = resolve_selected_plan(json.dumps(wire), accepted, turn)
    state = {"requirements_state": accepted.model_dump(), "qa5": {"turn": turn.model_dump(),
        "plan_raw": normalized.model_dump_json()}}
    workflow = object.__new__(workflow_type)
    original = workflow.plan_output_model(state).model_validate(strict_object(state["qa5"]["plan_raw"]), strict=True)
    assert workflow.parse_plan(state).model_dump() == original.model_dump()


def test_three_turn_locations_keep_constraints_and_replace_questions_without_resurrection():
    accepted = initial_state("a" * 64)
    first = UserTurn(turn_id="U1", text="比较方法甲与乙？保留负面结果。")
    operations = [{"action": "add", "target_ids": [], "category": category, "spans": [unit_id(first, text)]}
        for category, text in (("question", "比较方法甲与乙？"), ("constraint", "保留负面结果。"))]
    second = UserTurn(turn_id="U2", text="只看方法甲。")
    third = UserTurn(turn_id="U3", text="刚才结论的适用条件是什么？")
    for turn, ops, retained in (
        (first, operations, []),
        (second, [{"action": "supersede", "target_ids": ["R1"], "category": "question",
            "spans": [unit_id(second, second.text)]}], ["R2"]),
        (third, [{"action": "supersede", "target_ids": ["R3"], "category": "question",
            "spans": [unit_id(third, third.text)]}], ["R2"]),
    ):
        plan = resolve_selected_plan(json.dumps(wire_plan(accepted, turn, ops, retained)), accepted, turn)
        candidate = propose_delta(accepted, turn, plan.requirement_delta)
        accepted = fixture_accept(accepted, candidate)
    assert accepted.revision == 3
    assert [(item.id, item.state) for item in accepted.requirements] == [
        ("R1", "superseded"), ("R2", "active"), ("R3", "superseded"), ("R4", "active")]
    assert accepted.requirements[1].spans[0].text == "保留负面结果。"
    assert accepted.requirements[-1].spans[0].text == "刚才结论的适用条件是什么？"


def test_review_descriptions_preserve_full_grid_and_independent_negative_opinions():
    from tests.v3.test_qa5_protocol_binding import fixture

    _model, grid, _identities, value = fixture()
    state = {"qa5": {"candidate": {}, "turn": {"text": "Synthetic current instruction"}}}
    workflow = object.__new__(QADialogueGrounded)
    model = workflow.review_model(state, grid)
    value.update(candidate_sha256=digest({}), user_turn_sha256=digest(state["qa5"]["turn"]["text"]),
        decision="needs_revision", current_turn_coverage="incomplete", request_coverage="partial",
        uncovered_requests=["Synthetic actual missing answer detail"])
    assert model.model_validate(value, strict=True).decision == "needs_revision"
    missing_claim = copy.deepcopy(value)
    missing_claim["claims"].pop(next(iter(grid.identities)))
    with pytest.raises(ValidationError):
        model.model_validate(missing_claim, strict=True)
    with pytest.raises(ValidationError):
        model.model_validate({**value, "uncovered_requests": []}, strict=True)
