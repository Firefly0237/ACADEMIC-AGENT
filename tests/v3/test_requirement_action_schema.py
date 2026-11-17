"""One pure control for online action shapes and the recorded legacy schema."""
import copy

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import canonical, digest
from citequest.agents_v3.requirements_turn_units import (
    resolve_selected_plan, selected_plan_model, turn_units,
)
from citequest.agents_v3.structured_output import strict_output_schema
from citequest.research_v3.current_requirements import RequirementsState, UserTurn, initial_state


def test_online_actions_reject_invented_targets_and_keep_the_recorded_legacy_schema():
    # This fingerprint is the actual R14 plan:0 advertised full schema. Its
    # authority hash is a public contract identity, not a credential or grant.
    accepted = initial_state("1889fac33b82f9152bf616f02facb1ac6c472e3a4ccae0d4dd7d9ebcb2c9c96d")
    turn = UserTurn(turn_id="U1", text="本文对 Word2vec 与 fastText 的增强词处理结果是否一致？说明改善量的适用设置，保留不改善的结果。")
    legacy = selected_plan_model(accepted, turn)
    assert digest(strict_output_schema(legacy.model_json_schema())) == (
        "396e0a78964b06054b7d294cd368772b3fc86aa92e9373cff4ba31aac83facf4")
    assert selected_plan_model(accepted, turn, constrain_actions=False).model_json_schema() == legacy.model_json_schema()
    unit = turn_units(turn)["units"][0]["unit_id"]
    plan = {"requirement_delta": {"parent_sha256": digest(accepted.model_dump(mode="json")),
        "authority_sha256": accepted.authority_sha256, "turn_id": turn.turn_id,
        "turn_sha256": digest(turn.text), "retain_ids": [], "intent": "clear", "clarification": None,
        "operations": [{"action": "add", "target_ids": [], "category": "question", "spans": [unit]}]},
        "action": "retrieve", "queries": [{"query": "current source evidence", "limit": 1}], "existing_ref_ids": []}
    online = selected_plan_model(accepted, turn, constrain_actions=True)
    assert online.model_validate(plan, strict=True).requirement_delta.operations[0].target_ids == []
    parsed = resolve_selected_plan(canonical(plan).decode(), accepted, turn, constrain_actions=True)
    assert parsed.requirement_delta.operations[0].spans[0].turn_id == turn.turn_id
    wrong = copy.deepcopy(plan)
    wrong["requirement_delta"]["operations"][0]["target_ids"] = ["R1"]
    with pytest.raises(ValidationError) as captured:
        online.model_validate(wrong, strict=True)
    assert captured.value.errors(include_input=False)[0]["type"] == "too_long"
    assert plan["requirement_delta"]["operations"][0]["target_ids"] == []

    previous = UserTurn(turn_id="U1", text="Current requirement. Retired requirement.")
    span = {"turn_id": previous.turn_id, "start": 0, "end": len(previous.text),
        "text": previous.text, "turn_sha256": digest(previous.text)}
    retirement = UserTurn(turn_id="U2", text="Withdraw the second requirement.")
    current = RequirementsState.model_validate({"authority_sha256": "a" * 64, "revision": 2,
        "turns": [{"turn_id": item.turn_id, "text_sha256": digest(item.text)} for item in (previous, retirement)],
        "requirements": [
            {"id": "R1", "category": "question", "spans": [span], "state": "active",
                "introduced_turn_id": "U1", "replacement_ids": []},
            {"id": "R2", "category": "question", "spans": [span], "state": "withdrawn",
                "introduced_turn_id": "U1", "retired_turn_id": "U2", "replacement_ids": []}],
        "accepted": True, "review_sha256": "b" * 64}, strict=True)
    later = UserTurn(turn_id="U3", text="Change the current requirement.")
    revised = copy.deepcopy(plan)
    revised["requirement_delta"].update(parent_sha256=digest(current.model_dump(mode="json")),
        authority_sha256=current.authority_sha256, turn_id=later.turn_id, turn_sha256=digest(later.text))
    operation = revised["requirement_delta"]["operations"][0]
    operation.update(action="supersede", target_ids=["R1"], spans=[turn_units(later)["units"][0]["unit_id"]])
    bounded = selected_plan_model(current, later, constrain_actions=True)
    assert bounded.model_validate(revised, strict=True).requirement_delta.operations[0].target_ids == ["R1"]
    for invalid_targets in (["R2"], ["R9"], []):
        wrong = copy.deepcopy(revised)
        wrong["requirement_delta"]["operations"][0]["target_ids"] = invalid_targets
        with pytest.raises(ValidationError):
            bounded.model_validate(wrong, strict=True)
    operation.update(action="withdraw", category=None)
    assert bounded.model_validate(revised, strict=True).requirement_delta.operations[0].category is None
