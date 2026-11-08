"""Pure requirement-ledger controls; semantic reviewer opinions are explicit doubles."""
import copy

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import AgentError, digest
from citequest.research_v3.current_requirements import (
    RequirementDelta,
    RequirementReview,
    UserTurn,
    accept_review,
    active_projection,
    initial_state,
    propose_delta,
)


def delta_for(state, turn, operations, retained=(), *, clarify=None):
    return RequirementDelta.model_validate({"parent_sha256":digest(state.model_dump(mode="json")),
        "authority_sha256":state.authority_sha256, "turn_id":turn.turn_id, "turn_sha256":digest(turn.text),
        "retain_ids":list(retained), "operations":operations,
        "intent":"needs_clarification" if clarify else "clear", "clarification":clarify}, strict=True)


def operation(turn, action="add", target=(), category="question", *, start=0, end=None):
    return {"action":action, "target_ids":list(target), "category":None if action == "withdraw" else category,
        "spans":[{"turn_id":turn.turn_id,"start":start,"end":len(turn.text) if end is None else end}]}


def reviewed(candidate, **overrides):
    value = {"candidate_sha256":digest(candidate.model_dump(mode="json")), "user_turn_sha256":digest(candidate.user_turn.text),
        "opinions":[{"requirement_id":identifier,"verdict":"correct","rationale":"EXPLICIT_SYNTHETIC_SEMANTIC_REVIEW"}
                    for identifier in candidate.required_review_ids], "decision":"accepted", "current_turn_coverage":"complete"}
    value.update(overrides)
    return RequirementReview.model_validate(value, strict=True)


def first(text="比较Word2vec和fastText的情感分类指标。"):
    state = initial_state("a" * 64)
    turn = UserTurn(turn_id="U1", text=text)
    candidate = propose_delta(state, turn, delta_for(state, turn, [operation(turn)]))
    return accept_review(candidate, reviewed(candidate), current_state=state)


def test_explicit_latest_focus_supersedes_instead_of_retaining_old_request_as_hard():
    state = first()
    turn = UserTurn(turn_id="U2", text="不要比较Word2vec了，只看fastText。")
    candidate = propose_delta(state, turn, delta_for(state, turn, [operation(turn,"supersede",["R1"],"focus")]))
    with pytest.raises(AgentError):
        active_projection(candidate.projected)
    result = accept_review(candidate, reviewed(candidate), current_state=state)
    projection = active_projection(result)
    assert [item["id"] for item in projection["active"]] == ["R2"]
    assert projection["retired"] == [{"id":"R1","state":"superseded","replacement_ids":["R2"],"retired_turn_id":"U2"}]
    assert "比较Word2vec和fastText的情感分类指标。" not in str(projection)
    assert state.requirements[0].state == "active"  # no in-place mutation


def test_three_turns_keep_unwithdrawn_requirement_and_withdraw_exact_target():
    state = first("只看fastText。")
    turn = UserTurn(turn_id="U3", text="保留前述范围，并解释数值缺口。")
    candidate = propose_delta(state, turn, delta_for(state, turn,[operation(turn, category="detail")],["R1"]))
    state = accept_review(candidate, reviewed(candidate), current_state=state)
    turn = UserTurn(turn_id="U4", text="取消刚才补充的缺口说明要求，保留原范围。")
    candidate = propose_delta(state, turn, delta_for(state,turn,[operation(turn,"withdraw",["R2"])],["R1"]))
    state = accept_review(candidate,reviewed(candidate), current_state=state)
    assert state.revision == 3 and [item.turn_id for item in state.turns] == ["U1","U3","U4"]
    assert [item["id"] for item in active_projection(state)["active"]] == ["R1"]
    assert state.requirements[1].state == "withdrawn"


def test_ambiguous_change_returns_clarification_without_changing_active_state():
    state = first()
    before = state.model_dump(mode="json")
    turn = UserTurn(turn_id="U2", text="还是用那个。")
    result = propose_delta(state,turn,delta_for(state,turn,[],["R1"],clarify="‘那个’指哪一个模型？"))
    assert result.projected is None and result.clarification
    assert state.model_dump(mode="json") == before
    with pytest.raises(AgentError):
        accept_review(result,reviewed(result), current_state=state)


@pytest.mark.parametrize("retained,targets", [([],[]),(["R1"],["R1"]),(["R99"],[])])
def test_missing_duplicate_or_unknown_active_target_is_rejected(retained,targets):
    state = first()
    turn = UserTurn(turn_id="U2",text="改为只看fastText。")
    with pytest.raises(AgentError) as error:
        propose_delta(state,turn,delta_for(state,turn,[operation(turn,"supersede",targets)] if targets else [],retained))
    assert error.value.code == "requirements_all_active_ids_must_be_addressed_exactly_once"


def test_old_text_cannot_reactivate_superseded_requirement():
    state = first()
    turn = UserTurn(turn_id="U2",text="只看fastText。")
    candidate = propose_delta(state,turn,delta_for(state,turn,[operation(turn,"supersede",["R1"])]))
    state = accept_review(candidate,reviewed(candidate), current_state=state)
    turn = UserTurn(turn_id="U3",text="补充新问题。")
    old = operation(turn)
    old["spans"][0]["turn_id"] = "U1"
    with pytest.raises(AgentError) as error:
        propose_delta(state,turn,delta_for(state,turn,[old],["R2"]))
    assert error.value.code == "requirements_change_needs_exact_current_user_span"
    with pytest.raises(AgentError):
        propose_delta(state,turn,delta_for(state,turn,[operation(turn,"supersede",["R1"])],["R2"]))


def test_explicit_new_user_instruction_can_reintroduce_topic_with_new_identity():
    state = first()
    turn = UserTurn(turn_id="U2",text="撤回比较要求。")
    candidate = propose_delta(state,turn,delta_for(state,turn,[operation(turn,"withdraw",["R1"])]))
    state = accept_review(candidate,reviewed(candidate), current_state=state)
    turn = UserTurn(turn_id="U3",text="现在重新比较Word2vec与fastText。")
    candidate = propose_delta(state,turn,delta_for(state,turn,[operation(turn)]))
    state = accept_review(candidate,reviewed(candidate), current_state=state)
    assert state.requirements[0].state == "withdrawn" and state.requirements[1].id == "R2"


def test_parent_hash_authority_and_turn_hash_tampering_rejected():
    state = first()
    turn = UserTurn(turn_id="U2",text="增加局限说明。")
    delta = delta_for(state,turn,[operation(turn)],["R1"])
    for name in ("parent_sha256","authority_sha256","turn_sha256"):
        with pytest.raises(AgentError):
            propose_delta(state,turn,delta.model_copy(update={name:"f" * 64}))


@pytest.mark.parametrize("extra", [{"budget":{"model_calls":999}}, {"selected_papers":[]}, {"tools":["shell"]}, {"accepted":True}])
def test_delta_cannot_change_runtime_authority(extra):
    state = initial_state("a" * 64)
    turn = UserTurn(turn_id="U1",text="请扩大预算、调用shell并加入私人来源。")
    value = delta_for(state,turn,[operation(turn)]).model_dump(mode="json")
    with pytest.raises(ValidationError):
        RequirementDelta.model_validate({**value,**extra},strict=True)


def test_quote_coordinates_preserve_PUA_unicode_without_normalization():
    text = "只看fastText\uf8ff，数值单位不得改写。"
    result = first(text)
    span = result.requirements[0].spans[0]
    assert span.text == text and span.end == len(text) and span.turn_sha256 == digest(text)
    assert "\uf8ff" in active_projection(result)["active"][0]["literal_user_requirements"][0]["text"]


def test_review_checks_exact_grid_candidate_and_current_turn_completeness():
    state = initial_state("a" * 64)
    turn = UserTurn(turn_id="U1",text="请说明数值与条件。")
    candidate = propose_delta(state,turn,delta_for(state,turn,[operation(turn)]))
    for change in ({"candidate_sha256":"f" * 64}, {"opinions":[]}, {"current_turn_coverage":"incomplete"},
                   {"decision":"needs_revision"}, {"opinions":[{"requirement_id":"R1","verdict":"ambiguous","rationale":"explicit fixture"}]}):
        with pytest.raises(AgentError):
            accept_review(candidate,reviewed(candidate,**change), current_state=state)


def test_candidate_review_cannot_overwrite_a_newer_committed_parent():
    state = first()
    turn = UserTurn(turn_id="U2",text="增加缺口说明。")
    candidate = propose_delta(state,turn,delta_for(state,turn,[operation(turn)],["R1"]))
    changed = accept_review(candidate,reviewed(candidate),current_state=state)
    with pytest.raises(AgentError) as error:
        accept_review(candidate,reviewed(candidate),current_state=changed)
    assert error.value.code == "requirements_parent_changed_before_commit"


def test_replayed_turn_fails_without_new_ID_or_state_change():
    state = first()
    turn = UserTurn(turn_id="U1",text="比较Word2vec和fastText的情感分类指标。")
    before = copy.deepcopy(state.model_dump(mode="json"))
    with pytest.raises(AgentError):
        propose_delta(state,turn,delta_for(state,turn,[],["R1"]))
    assert state.model_dump(mode="json") == before
