"""Fixed requirement/gap opinions for the explicit grounded2 answer audit.

The models constrain identities and consistency. Semantic verdicts remain
independent AI opinions. Only validated user spans supply public omissions.
"""
from __future__ import annotations

import copy
import re
from typing import Literal

from pydantic import BaseModel, Field, create_model, model_validator

from citequest.research_v3.contracts import Strict
from citequest.research_v3.current_requirements import RequirementsCandidate

from .contracts import AgentError, digest

VERSION = "citequest-qa-answer-audit/1"
REPAIR_STATUSES = frozenset({"missing_response", "format_mismatch"})

AUDIT_POLICY = """
<answer_audit>
Judge every supplied active requirement against its exact original user spans
and the actual current answer. Fill every fixed requirement and gap property.
Use fulfilled when the requested response is provided. A request to identify
missing information can be fulfilled by accurately identifying that information.
Use bounded_evidence_gap when a requested substantive answer remains unavailable
within the actual evidence boundary and the answer states that limit faithfully.
Use missing_response for an actual omitted response and format_mismatch for an
actual user-specified presentation requirement that the answer fails to meet.
Choose locations from the current C/G/presentation identifiers. Fulfilled and
bounded_evidence_gap each require an actual location. Keep explanations concise.
Every non-fulfilled R ID belongs in uncovered_requests in supplied R order.
request_coverage is complete exactly when every active requirement is fulfilled.
These wire omissions contain R IDs only; the server projects the original text.

Independently judge every gap against the sources, reviewed claims, exact user
scope and observed retrieval boundary. Use faithful, contradictory or
unsubstantiated according to the actual content. Preserve distinctions between
a bounded retrieval observation and a universal absence, a stated research goal
and completed experimental measurements. A gap can contain a factual error even
when every claim passes. Retain genuine negative verdicts and explain them.
Keep source entailment, requirement interpretation and answer coverage separate.
Additional labels, headings and background become mandatory only through the
actual user instructions or factual accuracy of the requested response.
The supplied hashes identify this answer and requirement candidate. Opinions
authorize no new sources, requirements, tools or resources.
</answer_audit>
"""


class RequirementCoverage(Strict):
    status: Literal["fulfilled", "bounded_evidence_gap", "missing_response", "format_mismatch"]
    locations: list[str]
    explanation: str = Field(min_length=1, max_length=120)


class GapReview(Strict):
    verdict: Literal["faithful", "contradictory", "unsubstantiated"]
    explanation: str = Field(min_length=1, max_length=120)


class AnswerAudit:
    """Extend a facts-only combined review without changing its original model."""

    def __init__(self, base_review_model: type[BaseModel], grid, candidate: RequirementsCandidate) -> None:
        if {"request_coverage", "uncovered_requests"} & set(base_review_model.model_fields):
            raise AgentError("qa_answer_audit_facts_only_base_required")
        if candidate.projected is None or candidate.clarification is not None:
            raise AgentError("qa_answer_audit_projected_requirements_required")
        if any(re.fullmatch(r"[a-f0-9]{64}", str(grid.binding.get(key, ""))) is None
               for key in ("candidate_sha256", "binding_sha256")):
            raise AgentError("qa_answer_audit_grid_binding_invalid")

        self.requirements = [item.model_copy(deep=True) for item in candidate.projected.requirements if item.state == "active"]
        self.requirement_ids = [item.id for item in self.requirements]
        self.answer = copy.deepcopy(grid.candidate)
        self.claim_ids = list(grid.identities)
        self.gap_ids = ["G" + str(index + 1) for index in range(len(self.answer["gaps"]))]
        locations = self.claim_ids + self.gap_ids
        if "presentation" in self.answer:
            locations.append("presentation")
        self.binding = {
            "protocol": VERSION,
            "answer_candidate_sha256": grid.binding["candidate_sha256"],
            "compact_answer_sha256": digest(self.answer),
            "requirements_candidate_sha256": digest(candidate.model_dump()),
            "projected_requirements_sha256": digest(candidate.projected.model_dump()),
            "review_grid_binding_sha256": grid.binding["binding_sha256"],
            "requirement_ids": list(self.requirement_ids),
            "gap_ids": list(self.gap_ids),
            "locations": list(locations),
        }
        self.binding["binding_sha256"] = digest(self.binding)
        suffix = self.binding["binding_sha256"][:16]
        location_type = Literal[tuple(locations)]
        row_model = create_model(
            "RequirementCoverage_" + suffix,
            __base__=RequirementCoverage,
            locations=(list[location_type], Field(max_length=len(locations))),
        )
        coverage_model = create_model(
            "ActiveRequirements_" + suffix,
            __base__=Strict,
            **{identifier: (row_model, ...) for identifier in self.requirement_ids},
        )
        gap_model = create_model(
            "CurrentGaps_" + suffix,
            __base__=Strict,
            **{identifier: (GapReview, ...) for identifier in self.gap_ids},
        )
        requirement_id_type = Literal[tuple(self.requirement_ids)] if self.requirement_ids else str
        expected_ids = tuple(self.requirement_ids)

        @model_validator(mode="after")
        def consistent_answer_coverage(value):
            coverage = value.requirement_coverage.model_dump()
            incomplete = [identifier for identifier in expected_ids if coverage[identifier]["status"] != "fulfilled"]
            if value.uncovered_requests != incomplete:
                raise ValueError("wire omissions must equal the ordered non-fulfilled active requirement IDs")
            if value.request_coverage != ("partial" if incomplete else "complete"):
                raise ValueError("global answer coverage must agree with every requirement opinion")
            for row in coverage.values():
                if len(set(row["locations"])) != len(row["locations"]):
                    raise ValueError("answer audit locations must be unique")
                if row["status"] in {"fulfilled", "bounded_evidence_gap"} and not row["locations"]:
                    raise ValueError("an addressed requirement needs an actual answer location")
                if not row["explanation"].strip():
                    raise ValueError("requirement explanation must contain text")
            if any(not row["explanation"].strip() for row in value.gap_review.model_dump().values()):
                raise ValueError("gap explanation must contain text")
            return value

        self.model = create_model(
            "AuditedAnswerReview_" + suffix,
            __base__=base_review_model,
            __validators__={"consistent_answer_coverage": consistent_answer_coverage},
            candidate_sha256=(Literal[self.binding["requirements_candidate_sha256"]], ...),
            answer_candidate_sha256=(Literal[self.binding["answer_candidate_sha256"]], ...),
            requirement_coverage=(coverage_model, ...),
            gap_review=(gap_model, ...),
            request_coverage=(Literal["complete", "partial"], ...),
            uncovered_requests=(list[requirement_id_type], Field(max_length=len(self.requirement_ids))),
        )
        self.projection = {
            "binding": copy.deepcopy(self.binding),
            "requirements": [
                {"id": item.id, "category": item.category,
                 "original_user_spans": [span.model_dump() for span in item.spans]}
                for item in self.requirements
            ],
            "gaps": [{"id": identifier, "text": text}
                     for identifier, text in zip(self.gap_ids, self.answer["gaps"])],
            "allowed_locations": list(locations),
            "wire_omission_format": "ordered active R IDs only; all non-fulfilled statuses including bounded_evidence_gap",
            "verdict_authority": "independent AI opinion; structural consistency is validated separately",
        }

    def assess(self, parsed: BaseModel | dict) -> dict:
        """Project validated opinions without manufacturing a semantic verdict."""
        raw = parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
        value = self.model.model_validate(raw, strict=True).model_dump()
        coverage = value["requirement_coverage"]
        incomplete = value["uncovered_requests"]
        repair_locations = {
            location for row in coverage.values() if row["status"] in REPAIR_STATUSES
            for location in row["locations"]
        }
        bad_gaps = [identifier for identifier in self.gap_ids
                    if value["gap_review"][identifier]["verdict"] != "faithful"]
        return {
            "binding": copy.deepcopy(self.binding),
            "requirement_coverage": coverage,
            "gap_review": value["gap_review"],
            "request_coverage": value["request_coverage"],
            "wire_uncovered_requirement_ids": list(incomplete),
            "incomplete_requirement_ids": list(incomplete),
            "uncovered_requests": [
                "[" + item.id + "] " + "\n".join(span.text for span in item.spans)
                + "\n" + coverage[item.id]["explanation"]
                for item in self.requirements if item.id in incomplete
            ],
            "repair_claim_ids": [identifier for identifier in self.claim_ids if identifier in repair_locations],
            "bad_gap_ids": bad_gaps,
            "gaps_rejected": list(bad_gaps),
        }
