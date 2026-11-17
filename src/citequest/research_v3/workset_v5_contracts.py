"""New Workset5 contracts with stable local cross-field error codes.

No old Claim/SelectorClaim validator is modified or made more permissive.
These schemas are a prepared candidate, not a registered product profile.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from .contracts import Strict
from .quote_spans import SelectorEdge, SpanEdge
from .workset_contracts import DocumentCoverage, DocumentManifest, WorksetAnswer

BoundedText = Annotated[str, Field(min_length=1, max_length=500)]


class SpanEdgeV5(SpanEdge):
    conditions: list[BoundedText] = Field(max_length=6)


class SelectorEdgeV5(SelectorEdge):
    conditions: list[BoundedText] = Field(max_length=6)


class ClaimV5(Strict):
    id: str = Field(pattern=r"^C[1-9][0-9]?$", max_length=3)
    statement: str = Field(min_length=1, max_length=1500)
    status: Literal["supported", "contradicted", "mixed", "insufficient"]
    edges: list[SpanEdgeV5] = Field(max_length=5)
    limitations: list[BoundedText] = Field(max_length=6)

    @model_validator(mode="after")
    def justified(self):
        relations = {edge.relation for edge in self.edges}
        if self.status == "supported" and "supports" not in relations:
            raise PydanticCustomError("claim_supported_requires_supports", "A supported claim requires a supports edge")
        if self.status == "contradicted" and "contradicts" not in relations:
            raise PydanticCustomError("claim_contradicted_requires_contradicts", "A contradicted claim requires a contradicts edge")
        if self.status == "mixed" and not {"supports", "contradicts"} <= relations:
            raise PydanticCustomError("claim_mixed_requires_both_relations", "A mixed claim requires both source relations")
        if self.status == "insufficient" and not self.limitations:
            raise PydanticCustomError("claim_insufficient_requires_gap", "An insufficient claim requires an explicit gap")
        return self


class SelectorClaimV5(ClaimV5):
    edges: list[SelectorEdgeV5] = Field(max_length=5)


class SummaryV5(Strict):
    claims: list[ClaimV5] = Field(min_length=1, max_length=8)
    gaps: list[BoundedText] = Field(max_length=8)
    evidence_status: Literal["supported", "insufficient"]

    @model_validator(mode="after")
    def evidence_consistency(self):
        if len({claim.id for claim in self.claims}) != len(self.claims):
            raise PydanticCustomError("summary_claim_ids_not_unique", "Claim IDs must be unique")
        if self.evidence_status == "supported" and all(claim.status == "insufficient" for claim in self.claims):
            raise PydanticCustomError("summary_supported_requires_supported_claim", "Supported evidence_status requires an evidenced claim")
        return self


class SelectorSummaryV5(SummaryV5):
    claims: list[SelectorClaimV5] = Field(min_length=1, max_length=8)


class ComparisonQuestionV5(Strict):
    question: str = Field(min_length=1, max_length=700)
    requirement_ids: list[Annotated[str, Field(pattern=r"^R[1-9][0-9]{0,2}$")]] = Field(min_length=1, max_length=12)
    document_ids: list[Annotated[str, Field(pattern=r"^D(0[1-9]|10)$")]] = Field(min_length=2, max_length=5)
    evidence_route_ids: list[Annotated[str, Field(pattern=r"^E[1-9][0-9]{0,3}$")]] = Field(min_length=2, max_length=5)


class ComparisonPlanV5(Strict):
    # The server assigns stable C1..C12 only after accepting a plan. Model prose
    # never gets authority to create a source identity or current requirement.
    questions: list[ComparisonQuestionV5] = Field(min_length=1, max_length=12)
    unplanned_requirement_ids: list[Annotated[str, Field(pattern=r"^R[1-9][0-9]{0,2}$")]] = Field(max_length=12)
    limitations: list[BoundedText] = Field(max_length=8)


class PlanReviewV5(Strict):
    all_current_requirements_addressed: bool
    all_routes_within_current_scope: bool
    all_questions_atomic: bool
    omissions: list[BoundedText] = Field(max_length=12)


class GlobalCoverageV5(Strict):
    request_coverage: Literal["complete", "partial"]
    uncovered_requirement_ids: list[Annotated[str, Field(pattern=r"^R[1-9][0-9]{0,2}$")]] = Field(max_length=12)
    missing_document_ids: list[Annotated[str, Field(pattern=r"^D(0[1-9]|10)$")]] = Field(max_length=10)
    conflicting_claim_ids: list[Annotated[str, Field(pattern=r"^C[1-9][0-9]?$")]] = Field(max_length=12)
    gaps: list[BoundedText] = Field(max_length=12)


class DocumentResultV5(Strict):
    document: DocumentManifest
    coverage: DocumentCoverage
    summary: SummaryV5 | None
    review_status: Literal["ai_review_validated", "quality_rejected", "not_assessed"]


class WorksetAnswerV5(WorksetAnswer):
    schema_version: Literal["citequest-workset-result/5"] = "citequest-workset-result/5"
    documents: list[DocumentResultV5] = Field(min_length=1, max_length=10)
    comparisons: list[ClaimV5] = Field(max_length=12)
