"""Server-known requirement identities as schema Literals, never model opinions."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, create_model

from citequest.agents_v3.contracts import AgentError, digest

from .current_requirements import RequirementDelta, RequirementReview, RequirementsState, UserTurn

_SHA_IDENTITIES = frozenset({"parent_sha256", "authority_sha256", "turn_sha256", "candidate_sha256",
    "user_turn_sha256", "brief_sha256", "content_sha256"})


def bound_identity_model(base: type[BaseModel], **identities: str) -> type[BaseModel]:
    """Bind only explicit server control identities, preserving base validators.

    This does not bind model opinions, evidence choices, conflict positions or
    generated text. A caller must obtain each value from its authorized state.
    Legacy callers keep their original model rather than invoking this helper.
    """
    if not identities or not isinstance(base, type) or not issubclass(base, BaseModel):
        raise AgentError("identity_binding_model_required")
    for field, value in identities.items():
        pattern = r"U[1-9][0-9]{0,5}" if field == "turn_id" else r"[a-f0-9]{64}"
        if (field not in _SHA_IDENTITIES | {"turn_id"} or field not in base.model_fields
                or type(value) is not str or re.fullmatch(pattern, value) is None):
            raise AgentError("identity_binding_server_field_required")
    suffix = digest({"base": base.__name__, "identities": identities})[:20]
    return create_model("Bound" + base.__name__ + "_" + suffix, __base__=base,
        **{field: (Literal[value], ...) for field, value in identities.items()})


def bound_requirement_plan_model(base: type[BaseModel], accepted: RequirementsState, turn: UserTurn) -> type[BaseModel]:
    if not accepted.accepted or "requirement_delta" not in base.model_fields:
        raise AgentError("requirement_identity_server_state_required")
    parent = digest(accepted.model_dump(mode="json"))
    turn_sha = digest(turn.text)
    delta = bound_identity_model(RequirementDelta, parent_sha256=parent,
        authority_sha256=accepted.authority_sha256, turn_id=turn.turn_id, turn_sha256=turn_sha)
    return create_model("Bound" + base.__name__ + "_" + parent[:10] + turn_sha[:10],
        __base__=base, requirement_delta=(delta, ...))


def bound_requirement_review_model(candidate_sha: str, turn_sha: str) -> type[BaseModel]:
    if any(type(value) is not str or re.fullmatch(r"[a-f0-9]{64}", value) is None for value in (candidate_sha, turn_sha)):
        raise AgentError("requirement_identity_server_hash_required")
    return bound_identity_model(RequirementReview, candidate_sha256=candidate_sha, user_turn_sha256=turn_sha)
