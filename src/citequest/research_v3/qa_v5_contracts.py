"""QA5 typed planning and combined goal/source review; no execution authority."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, create_model, field_validator, model_validator

from citequest.agents_v3.contracts import AgentError, digest

from .contracts import Strict
from .current_requirements import RequirementDelta, RequirementReview
from .quote_spans import SelectorAnswer
from .workset_v5_contracts import SelectorClaimV5

if TYPE_CHECKING:
    from .review_grid import ReviewGrid

REVIEW_TRANSPORT_PROTOCOL = "flat1"


def result_limits(selected: list[dict]) -> list[int]:
    """Explicit allocation limits of the existing source service contracts.

    Native paper chunks support20 overall; each existing private material
    service query supports3 per document. Never execute then clip a plan.
    """
    if not selected:
        return list(range(1,21))
    supported = []
    for total in range(len(selected),21):
        quotient,remainder = divmod(total,len(selected))
        if all(item["kind"] != "material_file" or quotient+int(index<remainder) <= 3
               for index,item in enumerate(selected)):
            supported.append(total)
    return supported


class RetrievalIntent(Strict):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(ge=1, le=20)

    @field_validator("query")
    @classmethod
    def valid_search_text(cls,value: str) -> str:
        # Same public search shape is checked for EVERY planned query before
        # executing the first MCP call. Native token admission still applies.
        from citequest.agents_v3.contracts import validate_user_text
        from citequest.search_v3.schemas import SearchRequest
        validate_user_text(value)
        SearchRequest(q=value)
        return value


class QAPlanV5(Strict):
    requirement_delta: RequirementDelta
    action: Literal["retrieve", "verify_existing", "clarify"]
    queries: list[RetrievalIntent] = Field(max_length=2)
    existing_ref_ids: list[str] = Field(max_length=20)

    @model_validator(mode="after")
    def bounded_action(self) -> QAPlanV5:
        if len(set(self.existing_ref_ids)) != len(self.existing_ref_ids):
            raise ValueError("existing evidence refs must be unique")
        if self.action == "clarify":
            if self.queries or self.existing_ref_ids or self.requirement_delta.intent != "needs_clarification":
                raise ValueError("clarification cannot perform a retrieval or claim evidence reuse")
        elif self.requirement_delta.intent != "clear":
            raise ValueError("ambiguous user changes cannot perform provisional source work")
        elif self.action == "retrieve":
            if not self.queries or sum(item.limit for item in self.queries) + len(self.existing_ref_ids) > 20:
                raise ValueError("all planned queries and reused evidence share20 result slots")
        elif self.queries or not self.existing_ref_ids:
            raise ValueError("evidence reuse needs existing issued refs and no hidden query")
        return self


class SelectorAnswerV5(SelectorAnswer):
    kind: Literal["qa"] = "qa"
    requested_direction: Literal["neutral"] = "neutral"
    claims: list[SelectorClaimV5] = Field(min_length=1, max_length=8)


def combined_review_model(grid: ReviewGrid) -> type[BaseModel]:
    """One root from both strict contracts; retain dynamic C/Q types verbatim."""
    if set(RequirementReview.model_fields) & set(grid.model.model_fields):
        raise AgentError("qa5_review_root_field_collision")
    return create_model("QA5FlatReview_" + grid.binding["binding_sha256"][:16],
        __base__=(RequirementReview, grid.model))


def bound_combined_review_model(grid: ReviewGrid, exact_requirement_candidate_sha: str,
                                exact_turn_sha: str) -> type[BaseModel]:
    """Bind server control identities while retaining every independent opinion.

    These are RequirementsCandidate and current user-text identities, NEVER the
    answer or ReviewGrid candidate hash. Both inputs come from validated graph
    state, not model output. Legacy combined_review_model stays byte-equivalent;
    only explicitly versioned callers select this stricter schema.
    """
    if any(type(value) is not str or re.fullmatch(r"[a-f0-9]{64}",value) is None
           for value in (exact_requirement_candidate_sha,exact_turn_sha)):
        raise AgentError("qa_bound_review_server_identity_invalid")
    binding = digest({"grid":grid.binding["binding_sha256"],"requirement_candidate":exact_requirement_candidate_sha,
                      "current_user_turn":exact_turn_sha})
    return create_model("QABoundFlatReview_" + binding[:16], __base__=combined_review_model(grid),
        candidate_sha256=(Literal[exact_requirement_candidate_sha], ...),
        user_turn_sha256=(Literal[exact_turn_sha], ...))
