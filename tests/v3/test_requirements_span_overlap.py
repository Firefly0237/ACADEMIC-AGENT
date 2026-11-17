"""One pure transition control; old stored spans keep their original contract."""
import copy

import pytest

from citequest.agents_v3.contracts import AgentError, canonical, digest
from citequest.research_v3.current_requirements import (
    RequirementDelta, RequirementsState, UserTurn, propose_delta,
)


def test_new_operation_rejects_overlaps_but_keeps_adjacent_shared_and_saved_spans():
    old_text = "保留验证范围。"
    old_sha = digest(old_text)
    saved = {"version": "citequest-current-requirements/1", "authority_sha256": "a" * 64,
        "revision": 1, "turns": [{"turn_id": "U1", "text_sha256": old_sha}],
        "accepted": True, "review_sha256": "b" * 64, "requirements": [{
            "id": "R1", "category": "constraint", "state": "active", "introduced_turn_id": "U1",
            "retired_turn_id": None, "replacement_ids": [], "spans": [{
                "turn_id": "U1", "start": 0, "end": end, "text": old_text[:end],
                "turn_sha256": old_sha} for end in (3, len(old_text))]}]}
    frozen = canonical(saved)
    parent = RequirementsState.model_validate(saved, strict=True)
    assert canonical(parent.model_dump()) == frozen  # Existing overlap still loads unchanged.
    turn = UserTurn(turn_id="U2", text="补充指标并保留条件。")

    def operation(ranges):
        return {"action": "add", "target_ids": [], "category": "detail", "spans": [
            {"turn_id": turn.turn_id, "start": start, "end": end} for start, end in ranges]}

    def delta(operations):
        return RequirementDelta.model_validate({"parent_sha256": digest(parent.model_dump()),
            "authority_sha256": parent.authority_sha256, "turn_id": turn.turn_id,
            "turn_sha256": digest(turn.text), "retain_ids": ["R1"],
            "operations": operations, "intent": "clear", "clarification": None}, strict=True)

    # Neighbouring spans and the same contextual span in a separate operation
    # remain legal; this gate does not claim to judge semantic duplication.
    valid = delta([operation([(0, 4), (4, len(turn.text))]), operation([(0, 4)])])
    valid_before = copy.deepcopy(valid.model_dump())
    candidate = propose_delta(parent, turn, valid)
    assert [item.id for item in candidate.projected.requirements] == ["R1", "R2", "R3"]
    assert candidate.projected.requirements[0].model_dump() == saved["requirements"][0]
    assert len(candidate.projected.requirements[1].spans) == 2
    assert candidate.projected.requirements[2].spans[0].text == turn.text[:4]
    for ranges in ([(0, 4), (0, 4)], [(0, 4), (0, len(turn.text))], [(0, 5), (4, len(turn.text))]):
        with pytest.raises(AgentError, match="requirements_change_spans_overlap"):
            propose_delta(parent, turn, delta([operation(ranges)]))
    assert valid.model_dump() == valid_before
    assert canonical(parent.model_dump()) == frozen and canonical(saved) == frozen
