"""Server-issued user-text locations, separate from understanding their intent.

The planner selects literal IDs. Existing requirement transitions still resolve
and independently review the resulting exact spans. No input text is rewritten.
"""
from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, create_model

from citequest.research_v3.current_requirements import Category, RequirementDelta, RequirementOperation, RequirementsState, UserTurn
from citequest.research_v3.qa_v5_contracts import QAPlanV5
from citequest.research_v3.requirements_identity import bound_identity_model
from citequest.research_v3.typed_feedback import strict_object

from .contracts import AgentError, digest

VERSION = "citequest-user-turn-locations/1"
BOUNDARY = re.compile(r"[，。！？；\n!?;]|(?<!\d),(?!\d)|(?<!\d)\.(?!\d)")


def turn_units(turn: UserTurn) -> dict:
    """Offer exact clause locations and complete 1500-codepoint input blocks.

    Punctuation determines location choices, not semantic requirements. Include
    the complete turn in blocks as well so a clause split cannot erase context.
    Repeated identical text keeps distinct positional IDs. Existing full-wire
    admission bounds the entire mandatory catalog before any model dispatch.
    """
    ranges = {(start, min(len(turn.text), start + 1500)) for start in range(0, len(turn.text), 1500)}
    start = 0
    for end in [*(match.end() for match in BOUNDARY.finditer(turn.text)), len(turn.text)]:
        ranges.update((offset, min(end, offset + 1500)) for offset in range(start, end, 1500))
        start = end
    units = [{"unit_id": f"{turn.turn_id}.S{index}", "turn_id": turn.turn_id,
        "start": start, "end": end, "text": turn.text[start:end]}
        for index, (start, end) in enumerate(sorted(ranges), start=1)]
    body = {"version": VERSION, "turn_id": turn.turn_id, "turn_sha256": digest(turn.text), "units": units}
    return {**body, "catalog_sha256": digest(body), "semantic_interpretation": False}


def selected_plan_model(accepted: RequirementsState, turn: UserTurn, *,
                        constrain_actions: bool = False) -> type[BaseModel]:
    """Retain all base validators while replacing offsets with issued enums."""
    if not accepted.accepted:
        raise AgentError("requirement_identity_server_state_required")
    catalog = turn_units(turn)
    unit_id = Literal[tuple(item["unit_id"] for item in catalog["units"])]
    suffix = digest({"catalog": catalog["catalog_sha256"], "parent": digest(accepted.model_dump(mode="json"))})[:16]
    if constrain_actions:
        suffix += "_actions1"
    operation = create_model("SelectedRequirementOperation_" + suffix, __base__=RequirementOperation,
        spans=(list[unit_id], Field(min_length=1, max_length=3,
            description="Exact issued unit IDs. The server resolves their original user-text positions.")))
    bound = bound_identity_model(RequirementDelta, parent_sha256=digest(accepted.model_dump(mode="json")),
        authority_sha256=accepted.authority_sha256, turn_id=turn.turn_id, turn_sha256=digest(turn.text))
    fields = {"operations": (list[operation], Field(max_length=16))}
    if constrain_actions:
        active_ids = tuple(item.id for item in accepted.requirements if item.state == "active")
        add = create_model("AddRequirement_" + suffix, __base__=operation,
            action=(Literal["add"], ...), target_ids=(list[str], Field(max_length=0)),
            category=(Category, ...))
        action_type = add
        retained_type = str
        if active_ids:
            target_type = Literal[active_ids]
            targets = (list[target_type], Field(min_length=1, max_length=min(8, len(active_ids))))
            supersede = create_model("SupersedeRequirement_" + suffix, __base__=operation,
                action=(Literal["supersede"], ...), target_ids=targets, category=(Category, ...))
            withdraw = create_model("WithdrawRequirement_" + suffix, __base__=operation,
                action=(Literal["withdraw"], ...), target_ids=targets, category=(type(None), ...))
            action_type = Annotated[Union[add, supersede, withdraw], Field(discriminator="action")]
            retained_type = target_type
        fields = {"operations": (list[action_type], Field(max_length=16)),
            "retain_ids": (list[retained_type], Field(max_length=min(24, len(active_ids))))}
    delta = create_model("SelectedRequirementDelta_" + suffix, __base__=bound, **fields)
    return create_model("SelectedQAPlan_" + suffix, __base__=QAPlanV5, requirement_delta=(delta, ...))


def resolve_selected_plan(raw: str, accepted: RequirementsState, turn: UserTurn, *,
                          constrain_actions: bool = False) -> QAPlanV5:
    """Validate wire selectors, then reuse the complete original plan contract."""
    parsed = selected_plan_model(accepted, turn, constrain_actions=constrain_actions).model_validate(
        strict_object(raw), strict=True)
    value = parsed.model_dump(mode="json")
    by_id = {item["unit_id"]: item for item in turn_units(turn)["units"]}
    for operation in value["requirement_delta"]["operations"]:
        operation["spans"] = [{key: by_id[unit_id][key] for key in ("turn_id", "start", "end")}
            for unit_id in operation["spans"]]
    return QAPlanV5.model_validate(value, strict=True)
