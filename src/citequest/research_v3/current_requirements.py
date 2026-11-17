"""Typed current user requirements, with exact input spans and tombstones.

Pure state transitions for official LangGraph nodes/checkpoints. No provider,
SQL, scheduler, retry, authority change or semantic-accuracy claim lives here.
The caller must load server-owned state and bind independently settled review;
these schemas are data shapes, not model-supplied authorization capabilities.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from citequest.agents_v3.contracts import AgentError, digest, validate_user_text

from .contracts import Strict

VERSION = "citequest-current-requirements/1"
SHA = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
RID = Annotated[str, Field(pattern=r"^R[1-9][0-9]{0,3}$")]
TID = Annotated[str, Field(pattern=r"^U[1-9][0-9]{0,5}$")]
Category = Literal["question", "focus", "detail", "format", "constraint"]


class UserTurn(Strict):
    turn_id: TID
    text: str = Field(min_length=1, max_length=8000)

    @model_validator(mode="after")
    def valid_text(self) -> UserTurn:
        validate_user_text(self.text)
        return self


class TurnRef(Strict):
    turn_id: TID
    text_sha256: SHA


class InputSpan(Strict):
    turn_id: TID
    start: int = Field(ge=0, le=7999)
    end: int = Field(ge=1, le=8000)

    @model_validator(mode="after")
    def ordered(self) -> InputSpan:
        if not 0 < self.end - self.start <= 1500:
            raise ValueError("requirement input span must have1..1500 Unicode codepoints")
        return self


class ResolvedSpan(InputSpan):
    text: str = Field(min_length=1, max_length=1500)
    turn_sha256: SHA


class Requirement(Strict):
    id: RID
    category: Category
    spans: list[ResolvedSpan] = Field(min_length=1, max_length=3)
    state: Literal["active", "superseded", "withdrawn"]
    introduced_turn_id: TID
    retired_turn_id: TID | None = None
    replacement_ids: list[RID] = Field(max_length=1)

    @model_validator(mode="after")
    def lifecycle(self) -> Requirement:
        if self.state == "active" and (self.retired_turn_id is not None or self.replacement_ids):
            raise ValueError("active requirement cannot have a retirement")
        if self.state != "active" and self.retired_turn_id is None:
            raise ValueError("retired requirement needs an actual later user turn")
        if (self.state == "superseded") != bool(self.replacement_ids):
            raise ValueError("only superseded requirements name their replacement")
        return self


class RequirementsState(Strict):
    version: Literal["citequest-current-requirements/1"] = VERSION
    authority_sha256: SHA
    revision: int = Field(ge=0)
    turns: list[TurnRef] = Field(max_length=128)
    requirements: list[Requirement] = Field(max_length=128)
    accepted: bool
    review_sha256: SHA | None = None

    @model_validator(mode="after")
    def unique_ledger(self) -> RequirementsState:
        ids = [item.id for item in self.requirements]
        if ids != ["R" + str(i + 1) for i in range(len(ids))]:
            raise ValueError("requirement identities must be the full server-issued ordered ledger")
        turn_ids = [item.turn_id for item in self.turns]
        if len(set(turn_ids)) != len(turn_ids) or [int(i[1:]) for i in turn_ids] != sorted(int(i[1:]) for i in turn_ids):
            raise ValueError("user turns must retain unique increasing command sequence")
        turn_map = {item.turn_id: item.text_sha256 for item in self.turns}
        if self.revision != len(self.turns):
            raise ValueError("each interpreted user turn must advance the projection exactly once")
        if len([item for item in self.requirements if item.state == "active"]) > 24:
            raise ValueError("current requirement capacity exceeded; do not silently delete older requirements")
        for item in self.requirements:
            if item.introduced_turn_id not in turn_map or (item.retired_turn_id and item.retired_turn_id not in turn_map):
                raise ValueError("requirement lifecycle turn missing from ledger")
            if any(span.turn_id not in turn_map or span.turn_sha256 != turn_map[span.turn_id]
                   or span.turn_id != item.introduced_turn_id or span.end - span.start != len(span.text) for span in item.spans):
                raise ValueError("requirement spans must remain bound to their exact user turn")
            if any(target not in ids for target in item.replacement_ids):
                raise ValueError("superseded replacement missing from ledger")
            if item.retired_turn_id and int(item.retired_turn_id[1:]) <= int(item.introduced_turn_id[1:]):
                raise ValueError("requirement retirement must be later than its creation")
            if any(int(target[1:]) <= int(item.id[1:]) for target in item.replacement_ids):
                raise ValueError("superseded replacement must have a fresh later identity")
        if self.accepted and self.revision > 0 and self.review_sha256 is None:
            raise ValueError("accepted nonempty projection needs an independently bound review receipt")
        return self


class RequirementOperation(Strict):
    action: Literal["add", "supersede", "withdraw"]
    target_ids: list[RID] = Field(max_length=8)
    category: Category | None
    spans: list[InputSpan] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def shape(self) -> RequirementOperation:
        if len(set(self.target_ids)) != len(self.target_ids):
            raise ValueError("requirement targets must be unique")
        if (self.action == "add") != (not self.target_ids):
            raise ValueError("add has no old target; supersede/withdraw require old targets")
        if (self.action == "withdraw") != (self.category is None):
            raise ValueError("only a withdrawal lacks a new requirement category")
        return self


class RequirementDelta(Strict):
    version: Literal["citequest-current-requirements/1"] = VERSION
    parent_sha256: SHA
    authority_sha256: SHA
    turn_id: TID
    turn_sha256: SHA
    retain_ids: list[RID] = Field(max_length=24)
    operations: list[RequirementOperation] = Field(max_length=16)
    intent: Literal["clear", "needs_clarification"]
    clarification: str | None = Field(max_length=500)

    @model_validator(mode="after")
    def clarity(self) -> RequirementDelta:
        if (self.intent == "needs_clarification") != bool(self.clarification):
            raise ValueError("ambiguous intent requires an explicit clarification")
        if self.intent == "needs_clarification" and self.operations:
            raise ValueError("ambiguous intent cannot apply requirement mutations")
        return self


class RequirementsCandidate(Strict):
    base_sha256: SHA
    delta_sha256: SHA
    user_turn: UserTurn
    projected: RequirementsState | None
    required_review_ids: list[RID] = Field(max_length=48)
    clarification: str | None


class RequirementOpinion(Strict):
    requirement_id: RID
    verdict: Literal["correct", "incorrect", "ambiguous"]
    rationale: str = Field(min_length=1, max_length=500)


class RequirementReview(Strict):
    candidate_sha256: SHA
    user_turn_sha256: SHA
    opinions: list[RequirementOpinion] = Field(max_length=48)
    current_turn_coverage: Literal["complete", "incomplete", "ambiguous"]
    decision: Literal["accepted", "needs_revision", "needs_clarification"]


def initial_state(authority_sha256: str) -> RequirementsState:
    return RequirementsState(authority_sha256=authority_sha256, revision=0, turns=[], requirements=[], accepted=True)


def propose_delta(state: RequirementsState, turn: UserTurn, delta: RequirementDelta) -> RequirementsCandidate:
    """Propose a deterministic delta; never silently promote it to current state.

    Every old active requirement is explicitly retained or targeted once.
    Superseded/withdrawn records can never be targets; reintroducing a topic
    requires a new literal instruction from the current user turn and new ID.
    """
    base = digest(state.model_dump(mode="json"))
    if not state.accepted or delta.parent_sha256 != base or delta.authority_sha256 != state.authority_sha256:
        raise AgentError("requirements_parent_or_authority_changed")
    turn_sha = digest(turn.text)
    if delta.turn_id != turn.turn_id or delta.turn_sha256 != turn_sha:
        raise AgentError("requirements_current_turn_binding_mismatch")
    if state.turns and int(turn.turn_id[1:]) <= int(state.turns[-1].turn_id[1:]):
        raise AgentError("requirements_turn_already_processed_or_out_of_order")
    active = {item.id for item in state.requirements if item.state == "active"}
    touched = [target for operation in delta.operations for target in operation.target_ids]
    addresses = delta.retain_ids + touched
    if len(addresses) != len(set(addresses)) or set(addresses) != active:
        raise AgentError("requirements_all_active_ids_must_be_addressed_exactly_once")
    resolved = []
    for operation in delta.operations:
        spans = []
        for span in operation.spans:
            if span.turn_id != turn.turn_id or not 0 <= span.start < span.end <= len(turn.text):
                raise AgentError("requirements_change_needs_exact_current_user_span")
            spans.append(ResolvedSpan(**span.model_dump(), text=turn.text[span.start:span.end], turn_sha256=turn_sha))
        # One operation must not count the same input codepoint twice. This
        # checks new changes only; saved ledgers and independent operations
        # retain their original spans, including shared contextual wording.
        ordered = sorted(spans, key=lambda span: (span.start, span.end))
        if any(right.start < left.end for left, right in zip(ordered, ordered[1:])):
            raise AgentError("requirements_change_spans_overlap")
        resolved.append((operation, spans))
    if delta.intent == "needs_clarification":
        return RequirementsCandidate(base_sha256=base, delta_sha256=digest(delta.model_dump(mode="json")), user_turn=turn,
            projected=None, required_review_ids=[], clarification=delta.clarification)
    if state.revision == 0 and not delta.operations:
        raise AgentError("requirements_initial_user_request_not_represented")
    records = {item.id: item.model_copy(deep=True) for item in state.requirements}
    for operation, spans in resolved:
        replacement = None
        if operation.action != "withdraw":
            replacement = "R" + str(len(records) + 1)
            records[replacement] = Requirement(id=replacement, category=operation.category, spans=spans, state="active",
                                               introduced_turn_id=turn.turn_id, replacement_ids=[])
        for target in operation.target_ids:
            item = records[target]
            records[target] = item.model_copy(update={"state": "superseded" if replacement else "withdrawn",
                "retired_turn_id": turn.turn_id, "replacement_ids": [replacement] if replacement else []})
    proposed = RequirementsState(authority_sha256=state.authority_sha256, revision=state.revision + 1,
        turns=[*state.turns, TurnRef(turn_id=turn.turn_id, text_sha256=turn_sha)], requirements=list(records.values()), accepted=False)
    review_ids = [item.id for item in proposed.requirements if item.state == "active" or item.id in touched]
    return RequirementsCandidate(base_sha256=base, delta_sha256=digest(delta.model_dump(mode="json")), user_turn=turn,
        projected=proposed, required_review_ids=review_ids, clarification=None)


def accept_review(candidate: RequirementsCandidate, review: RequirementReview, *, current_state: RequirementsState) -> RequirementsState:
    """Bind an independent review of this exact candidate, not a vote on facts.

    The calling Graph must first prove the review was produced through its
    independent role with known settled usage; a model-supplied Review object
    alone is not a trusted receipt. Semantic review remains an AI weak opinion.
    """
    if candidate.projected is None:
        raise AgentError("requirements_clarification_required_no_projection")
    if digest(current_state.model_dump(mode="json")) != candidate.base_sha256:
        raise AgentError("requirements_parent_changed_before_commit")
    if review.candidate_sha256 != digest(candidate.model_dump(mode="json")) or review.user_turn_sha256 != digest(candidate.user_turn.text):
        raise AgentError("requirements_review_bound_to_another_candidate")
    identities = [item.requirement_id for item in review.opinions]
    if len(identities) != len(set(identities)) or set(identities) != set(candidate.required_review_ids):
        raise AgentError("requirements_review_identity_grid_incomplete")
    if (review.decision != "accepted" or review.current_turn_coverage != "complete"
            or any(item.verdict != "correct" for item in review.opinions)):
        raise AgentError("requirements_projection_not_accepted")
    value = {**candidate.projected.model_dump(mode="json"), "accepted": True,
             "review_sha256": digest(review.model_dump(mode="json"))}
    return RequirementsState.model_validate(value, strict=True)


def active_projection(state: RequirementsState) -> dict:
    """Current goals only; retired prose is omitted, tombstone identities remain."""
    if not state.accepted:
        raise AgentError("requirements_unreviewed_projection_not_current")
    return {"version": VERSION, "revision": state.revision, "sha256": digest(state.model_dump(mode="json")),
        "authority_sha256": state.authority_sha256,
        "active": [{"id": item.id, "category": item.category, "literal_user_requirements": [span.model_dump() for span in item.spans]}
                   for item in state.requirements if item.state == "active"],
        "retired": [{"id": item.id, "state": item.state, "replacement_ids": item.replacement_ids,
                     "retired_turn_id": item.retired_turn_id} for item in state.requirements if item.state != "active"],
        "raw_turn_audit_refs": [item.model_dump() for item in state.turns]}
