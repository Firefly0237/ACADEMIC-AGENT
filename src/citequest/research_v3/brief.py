"""Versioned research intake: user facts, suggestions and conflicts stay distinct.

This module does not call a model, store credentials, or infer that a quoted paper
supports a fact. Persistence/authorization are supplied by ``brief_service``.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID, uuid5

from pydantic import Field, StringConstraints, ValidationError, field_validator, model_validator

from citequest.agents_v3.contracts import EvidenceSelection, Strict, digest, validate_user_text
from citequest.v2.domain import DomainError

ID = Annotated[UUID, Field(strict=False)]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
BriefField = Literal[
    "topic",
    "purpose",
    "research_questions",
    "literature_scope",
    "data_sources",
    "methods",
    "baselines",
    "metrics",
    "statistical_plan",
    "resources",
    "timeline",
    "risks",
    "stop_conditions",
    "preliminary_work",
    "expected_outputs",
    "innovation_hypotheses",
]
FIELD_ORDER: tuple[BriefField, ...] = (
    "topic",
    "purpose",
    "research_questions",
    "literature_scope",
    "data_sources",
    "methods",
    "baselines",
    "metrics",
    "statistical_plan",
    "resources",
    "timeline",
    "risks",
    "stop_conditions",
    "preliminary_work",
    "expected_outputs",
    "innovation_hypotheses",
)
QUESTIONS: dict[BriefField, str] = {
    "topic": "希望研究的具体方向或拟定题目是什么？",
    "purpose": "希望解决什么问题，研究意义和适用范围是什么？",
    "research_questions": "哪些可检验的研究问题需要回答？",
    "literature_scope": "检索学科、主题、时间或文献范围有哪些约束？",
    "data_sources": "拟用什么数据，是否公开可取得，有何许可或隐私约束？",
    "methods": "拟采用哪些方法，哪些只是待比较的候选？",
    "baselines": "需要与哪些可复现的基线或对照比较？",
    "metrics": "用哪些指标和单位判断效果及失败？",
    "statistical_plan": "拟如何重复实验、估计不确定性或实施统计比较？",
    "resources": "可用算力、经费、人力和访问条件有哪些？",
    "timeline": "研究起止及主要里程碑如何安排？",
    "risks": "有哪些可行性、数据、伦理或实现风险？",
    "stop_conditions": "达到何种条件应停止或调整实验？",
    "preliminary_work": "已有相关积累或成果是什么？没有或未提供可以明确说明。",
    "expected_outputs": "预期交付什么可检验的产物？",
    "innovation_hypotheses": "有哪些待证实的创新假设？不要表述为已经证实的创新。",
}
REQUIRED: tuple[BriefField, ...] = (
    "topic",
    "purpose",
    "research_questions",
    "data_sources",
    "methods",
    "baselines",
    "metrics",
    "resources",
    "timeline",
    "risks",
    "stop_conditions",
    "preliminary_work",
)


class Fact(Strict):
    value: Text
    status: Literal["confirmed", "explicit_unknown"]
    message_id: ID

    @field_validator("value")
    @classmethod
    def valid_user_value(cls, value: str) -> str:
        return validate_user_text(value)


class Proposal(Strict):
    id: ID
    field: BriefField
    candidate: Fact
    kind: Literal["conflict", "suggestion"]
    prior_value_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    state: Literal["open", "accepted", "discarded", "superseded"] = "open"


class BriefDocument(Strict):
    schema_version: Literal["citequest-research-intake/1"] = "citequest-research-intake/1"
    id: ID
    project_id: ID
    revision: int = Field(ge=1)
    facts: dict[BriefField, Fact] = Field(default_factory=dict, max_length=16)
    proposals: list[Proposal] = Field(default_factory=list, max_length=64)
    selected_sources: list[EvidenceSelection] = Field(default_factory=list, max_length=20)
    draft_source_ids: list[ID] = Field(default_factory=list, max_length=4, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def bounded(self) -> BriefDocument:
        keys = [(source.kind, source.id) for source in self.selected_sources]
        if len(keys) != len(set(keys)):
            raise ValueError("selected sources must be unique")
        material_ids = {source.id for source in self.selected_sources if source.kind == "material_file"}
        if len(set(self.draft_source_ids)) != len(self.draft_source_ids) or any(str(value) not in material_ids for value in self.draft_source_ids):
            raise ValueError("draft markers must be unique selected material file IDs")
        if len({proposal.id for proposal in self.proposals}) != len(self.proposals):
            raise ValueError("proposal ids must be unique")
        if len(self.model_dump_json().encode("utf-8")) > 65536:
            raise ValueError("brief exceeds 64 KiB; attach a draft as a material instead")
        return self


class FieldPatch(Strict):
    field: BriefField
    intent: Literal["set", "revise", "suggest", "unknown", "clear"] = "set"
    value: Text | None = None

    @field_validator("value")
    @classmethod
    def valid_user_value(cls, value: str | None) -> str | None:
        return validate_user_text(value) if value is not None else None

    @model_validator(mode="after")
    def valid_patch(self) -> FieldPatch:
        if (self.intent == "clear") != (self.value is None):
            raise ValueError("clear omits value; all other intents require a value")
        return self


class BriefCreate(Strict):
    operation_id: ID
    project_id: ID
    message_id: ID
    patches: list[FieldPatch] = Field(default_factory=list, max_length=16)
    selected_sources: list[EvidenceSelection] = Field(default_factory=list, max_length=20)
    draft_source_ids: list[ID] = Field(default_factory=list, max_length=4, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def valid_sources(self) -> BriefCreate:
        sources = [(item.kind, item.id) for item in self.selected_sources]
        if len(sources) != len(set(sources)) or len(self.draft_source_ids) != len(set(self.draft_source_ids)):
            raise ValueError("duplicate source or draft marker")
        if any(("material_file", str(value)) not in sources for value in self.draft_source_ids):
            raise ValueError("draft markers must identify selected material files")
        return self


class BriefTurn(Strict):
    operation_id: ID
    expected_revision: int = Field(ge=1)
    message_id: ID
    patches: list[FieldPatch] = Field(default_factory=list, max_length=16)
    selected_sources: list[EvidenceSelection] | None = Field(default=None, max_length=20)
    draft_source_ids: list[ID] | None = Field(default=None, max_length=4, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def nonempty(self) -> BriefTurn:
        if not self.patches and self.selected_sources is None and self.draft_source_ids is None:
            raise ValueError("empty brief turn")
        if self.draft_source_ids is not None and len(set(self.draft_source_ids)) != len(self.draft_source_ids):
            raise ValueError("duplicate draft marker")
        if self.selected_sources is not None and len({(item.kind, item.id) for item in self.selected_sources}) != len(self.selected_sources):
            raise ValueError("duplicate selected source")
        return self


class BriefResolution(Strict):
    operation_id: ID
    expected_revision: int = Field(ge=1)
    message_id: ID
    proposal_id: ID
    decision: Literal["keep_current", "use_candidate", "custom"]
    value: Text | None = None

    @model_validator(mode="after")
    def custom_value(self) -> BriefResolution:
        if (self.decision == "custom") != (self.value is not None):
            raise ValueError("only a custom resolution includes a value")
        return self


class BriefLifecycle(Strict):
    operation_id: ID
    expected_revision: int = Field(ge=1)
    deleted: bool


def _update(document: BriefDocument, **changes: object) -> BriefDocument:
    try:
        return BriefDocument.model_validate({**document.model_dump(), **changes})
    except ValidationError:
        raise DomainError("invalid_research_brief_update", 422) from None


def apply_turn(document: BriefDocument, command: BriefTurn) -> BriefDocument:
    """Apply a user-origin turn; adapters must force agent content to ``suggest``."""
    if command.expected_revision != document.revision:
        raise DomainError("brief_revision_conflict")
    if len({patch.field for patch in command.patches}) != len(command.patches):
        raise DomainError("duplicate_brief_field", 400)
    facts, proposals = dict(document.facts), list(document.proposals)
    for patch in command.patches:
        if patch.intent == "clear":
            facts.pop(patch.field, None)
            proposals = [
                p.model_copy(update={"state": "superseded"}) if p.field == patch.field and p.state == "open" else p
                for p in proposals
            ]
            continue
        candidate = Fact(
            value=patch.value,
            status="explicit_unknown" if patch.intent == "unknown" else "confirmed",
            message_id=command.message_id,
        )
        prior = facts.get(patch.field)
        if patch.intent == "suggest" or (
            prior is not None
            and patch.intent != "revise"
            and (prior.value, prior.status) != (candidate.value, candidate.status)
        ):
            proposals.append(
                Proposal(
                    id=uuid5(command.operation_id, patch.field),
                    field=patch.field,
                    candidate=candidate,
                    kind="suggestion" if patch.intent == "suggest" else "conflict",
                    prior_value_sha256=digest(prior.model_dump(mode="json")) if prior else None,
                )
            )
        else:
            if prior is None or (prior.value, prior.status) != (candidate.value, candidate.status):
                facts[patch.field] = candidate
            if patch.intent == "revise":
                proposals = [
                    p.model_copy(update={"state": "superseded"}) if p.field == patch.field and p.state == "open" else p
                    for p in proposals
                ]
    return _update(
        document,
        revision=document.revision + 1,
        facts=facts,
        proposals=proposals,
        selected_sources=document.selected_sources if command.selected_sources is None else command.selected_sources,
        draft_source_ids=document.draft_source_ids if command.draft_source_ids is None else command.draft_source_ids,
    )


def create_document(request: BriefCreate) -> BriefDocument:
    base = BriefDocument(
        id=uuid5(request.project_id, str(request.operation_id)), project_id=request.project_id, revision=1
    )
    if not request.patches and not request.selected_sources and not request.draft_source_ids:
        return base
    value = apply_turn(
        base,
        BriefTurn(
            operation_id=request.operation_id,
            expected_revision=1,
            message_id=request.message_id,
            patches=request.patches,
            selected_sources=request.selected_sources,
            draft_source_ids=request.draft_source_ids,
        ),
    )
    return _update(value, revision=1)


def resolve(document: BriefDocument, command: BriefResolution) -> BriefDocument:
    if command.expected_revision != document.revision:
        raise DomainError("brief_revision_conflict")
    proposal = next((p for p in document.proposals if p.id == command.proposal_id and p.state == "open"), None)
    if proposal is None:
        raise DomainError("brief_proposal_not_open", 404)
    current = document.facts.get(proposal.field)
    if (digest(current.model_dump(mode="json")) if current else None) != proposal.prior_value_sha256:
        raise DomainError("brief_proposal_stale")
    facts = dict(document.facts)
    if command.decision != "keep_current":
        facts[proposal.field] = Fact(
            value=command.value if command.decision == "custom" else proposal.candidate.value,
            status="confirmed" if command.decision == "custom" else proposal.candidate.status,
            message_id=command.message_id,
        )
    proposals = []
    for item in document.proposals:
        if item.id == proposal.id:
            item = item.model_copy(update={"state": "discarded" if command.decision == "keep_current" else "accepted"})
        elif item.field == proposal.field and item.state == "open" and command.decision != "keep_current":
            item = item.model_copy(update={"state": "superseded"})
        proposals.append(item)
    return _update(document, revision=document.revision + 1, facts=facts, proposals=proposals)


def readiness(document: BriefDocument) -> dict:
    missing = [
        field
        for field in REQUIRED
        if field not in document.facts or (document.facts[field].status != "confirmed" and field != "preliminary_work")
    ]
    conflicts = [str(p.id) for p in document.proposals if p.kind == "conflict" and p.state == "open"]
    return {
        "state": "needs_input" if missing or conflicts else "ready_for_planning",
        "missing_fields": missing,
        "questions": [{"field": field, "question": QUESTIONS[field]} for field in missing],
        "open_conflicts": conflicts,
        "model_calls_required": 0,
        "school_format_ready": False,
    }


def planning_payload(document: BriefDocument) -> dict:
    """A deterministic handoff; current source admission is a separate mandatory gate."""
    assessment = readiness(document)
    if assessment["state"] != "ready_for_planning":
        raise DomainError("research_brief_needs_input", 422)
    return {
        "schema_version": "citequest-research-planning-input/1",
        "brief_id": str(document.id),
        "brief_revision": document.revision,
        "brief_sha256": digest(document.model_dump(mode="json")),
        "user_confirmed": {key: fact.value for key, fact in document.facts.items() if fact.status == "confirmed"},
        "explicit_unknowns": {
            key: fact.value for key, fact in document.facts.items() if fact.status == "explicit_unknown"
        },
        "selected_sources": [source.model_dump(mode="json") for source in document.selected_sources],
        **({"draft_source_ids": [str(value) for value in document.draft_source_ids]} if document.draft_source_ids else {}),
        "research_status": "proposed_not_executed",
        "source_admission_required": True,
        "school_format_ready": False,
    }
