"""Versioned school-document contract; byte/semantic/export gates are separate."""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal

from pydantic import Field, model_validator

from citequest.agents_v3.contracts import Strict, canonical, digest

TEXT = Annotated[str, Field(min_length=1, max_length=5000)]
SHA = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
SOURCE_URL = "https://cs.sjtu.edu.cn/en/Graduate_Announcements/1086.html"
SOURCE_URL_2025 = "https://www.cs.sjtu.edu.cn/bsyjs_notice/1931.html"
LEGACY_PROFILE_ID = "sjtu-scs-2024-master-cn-capacity-demo/3"
CURRENT_PROFILE_ID = "sjtu-scs-2025-master-cn-capacity-demo/1"
LEGACY_SCOPE = "2024_grade_SJTU_SCS_master_public_template_not_universal_current_school_policy"
CURRENT_SCOPE = "2025_grade_SJTU_SCS_master_public_template_not_universal_current_school_policy"
TEMPLATE_SHA = "6bd1b4efb2106451686f15c406b9ae03b586939eca0d81604f20ab2c75a2da8d"
CSL_SHA = "0ca3d2c7c881cd98cdd81098b5349f5921edf430498aa9a5a41918621cf7cc45"
TOKEN = re.compile(r"\[@(S[1-9][0-9]*)\]")
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def han(text: str) -> int:
    return len(HAN.findall(text))


class SchoolProfile(Strict):
    id: Literal[LEGACY_PROFILE_ID, CURRENT_PROFILE_ID] = LEGACY_PROFILE_ID
    document_mode: Literal["capacity_demo"] = "capacity_demo"
    source_url: Literal[SOURCE_URL, SOURCE_URL_2025] = SOURCE_URL
    template_sha256: Literal[TEMPLATE_SHA] = TEMPLATE_SHA
    csl_sha256: Literal[CSL_SHA] = CSL_SHA
    scope: Literal[LEGACY_SCOPE, CURRENT_SCOPE] = LEGACY_SCOPE
    literature_minimum_han: Literal[5000] = 5000
    demonstration_minimum_han: Literal[8000] = 8000
    demonstration_maximum_han: Literal[12000] = 12000
    demonstration_length_is_school_rule: Literal[False] = False
    maximum_writing_model_calls: Literal[36] = 36
    per_call_output_maximum: Literal[2048] = 2048
    maximum_target_repairs: Literal[2] = 2
    maximum_blocks_per_section: Literal[3] = 3
    section_wire_overhead_bytes: Literal[600] = 600
    cross_review_mode: Literal["all14_full_literal_sections"] = "all14_full_literal_sections"

    @model_validator(mode="after")
    def registered_source(self) -> SchoolProfile:
        expected = (SOURCE_URL, LEGACY_SCOPE) if self.id == LEGACY_PROFILE_ID else (SOURCE_URL_2025, CURRENT_SCOPE)
        if (self.source_url, self.scope) != expected:
            raise ValueError("school_profile_source_and_cohort_mismatch")
        return self


class SectionSpec(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    group: Literal["review", "objectives", "methods", "innovation", "schedule", "prior"]
    title: str = Field(min_length=1, max_length=150)
    purpose: TEXT
    minimum_han: int = Field(ge=0, le=1500)
    maximum_han: int = Field(ge=1, le=1500)
    required_plan_ids: list[str] = Field(max_length=24)
    user_fields: list[str] = Field(max_length=16)
    allowed_claim_ids: list[str] = Field(max_length=64)
    allowed_source_aliases: list[str] = Field(max_length=20)


class SchoolBlock(Strict):
    id: str = Field(pattern=r"^P[1-9][0-9]?$", max_length=3)
    kind: Literal["evidence", "proposal", "user_fact", "limitation"]
    text: TEXT
    claim_ids: list[str] = Field(max_length=10)
    source_aliases: list[str] = Field(max_length=10)
    user_fields: list[str] = Field(max_length=16)

    @model_validator(mode="after")
    def accountable(self) -> SchoolBlock:
        if self.kind == "evidence" and (not self.claim_ids or not self.source_aliases):
            raise ValueError("every scientific assertion needs admitted claims and inline sources")
        if self.kind == "user_fact" and not self.user_fields:
            raise ValueError("user facts need explicit confirmed Brief fields")
        if self.kind != "evidence" and self.claim_ids:
            raise ValueError("a proposal or unknown must not be presented as a verified paper finding")
        if self.kind != "evidence" and self.source_aliases:
            raise ValueError("put sourced scientific rationale in a separate evidence block")
        if len(set(self.source_aliases)) != len(self.source_aliases):
            raise ValueError("duplicate block source aliases")
        return self


class SchoolSection(Strict):
    section_id: str
    brief_sha256: SHA
    blocks: list[SchoolBlock] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_blocks(self) -> SchoolSection:
        if len({item.id for item in self.blocks}) != len(self.blocks):
            raise ValueError("duplicate block ids")
        return self

    def body(self) -> str:
        return "\n\n".join(item.text for item in self.blocks)


class BlockReview(Strict):
    block_id: str
    verdict: Literal["passed", "needs_revision", "needs_input"]
    claim_ids: list[str] = Field(max_length=10)
    source_aliases: list[str] = Field(max_length=10)
    quantities_conditions_and_negation_preserved: bool
    explanation: str = Field(min_length=1, max_length=1000)


class Commitment(Strict):
    block_id: str
    kind: Literal["question", "method", "resource", "timeline", "term", "numerical", "limitation"]
    statement: str = Field(min_length=1, max_length=500)
    plan_ids: list[str] = Field(max_length=24)
    numeric_literals: list[str] = Field(max_length=30)


class SectionReview(Strict):
    section_id: str
    content_sha256: SHA
    blocks: list[BlockReview] = Field(min_length=1, max_length=8)
    covered_plan_ids: list[str] = Field(max_length=24)
    commitments: list[Commitment] = Field(min_length=1, max_length=16)
    fabricated_completed_work: bool
    source_instruction_injection: bool
    unapproved_resource_or_identity: bool


class CrossReviewItem(Strict):
    section_id: str
    verdict: Literal["passed", "needs_revision", "needs_input"]
    reason: str = Field(min_length=1, max_length=1000)


class CrossReview(Strict):
    content_sha256: SHA
    sections: list[CrossReviewItem] = Field(min_length=1, max_length=20)
    all_plan_ids: list[str] = Field(max_length=24)
    conflicts: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(max_length=16)


class RequestCoverageItem(Strict):
    field: str
    state: Literal["covered", "explicit_unknown", "uncovered", "conflict"]
    section_ids: list[str] = Field(max_length=14)
    reason: str = Field(min_length=1, max_length=600)


class RequestCoverage(Strict):
    content_sha256: SHA
    requests: list[RequestCoverageItem] = Field(min_length=1, max_length=16)
    school_sections_complete: bool
    performed_experiments_invented: bool


class ReferenceMetadataSource(Strict):
    provenance: Literal["current_authorized_source_metadata", "user_confirmed"]
    source_ref: dict
    file_sha256: SHA | None = Field(default=None, exclude_if=lambda value: value is None)
    representation_id: SHA | None = Field(default=None, exclude_if=lambda value: value is None)
    bibliography_sha256: SHA | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def confirmed_material_pin(self) -> ReferenceMetadataSource:
        if self.provenance == "user_confirmed" and (
            self.source_ref.get("kind") != "material_file" or not self.file_sha256
            or not self.representation_id or not self.bibliography_sha256
        ):
            raise ValueError("user-confirmed metadata must retain its exact material and bibliography pins")
        return self


class ReferenceAdmission(Strict):
    status: Literal["admitted"] = "admitted"
    provenance: Literal["current_authorized_metadata", "explicit_reference_fixture"]
    context_sha256: SHA
    references: list[dict] = Field(min_length=1, max_length=20)
    metadata_sources: dict[str, ReferenceMetadataSource] | None = Field(default=None, max_length=20,
        exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def metadata_matches_aliases(self) -> ReferenceAdmission:
        if self.metadata_sources and set(self.metadata_sources) != {item.get("id") for item in self.references}:
            raise ValueError("metadata provenance must identify every bibliography alias")
        return self


class SchoolDocument(Strict):
    schema_version: Literal["citequest-school-document/3", "citequest-school-document/4"] = "citequest-school-document/3"
    profile: SchoolProfile
    brief_sha256: SHA
    reviewed_plan_sha256: SHA
    title: str = Field(min_length=1, max_length=300)
    sections: list[SchoolSection] = Field(min_length=14, max_length=14)
    section_specs: list[SectionSpec] = Field(min_length=14, max_length=14)
    source_aliases: dict[str, list[str]] = Field(min_length=1, max_length=20)
    section_reviews: list[SectionReview] = Field(min_length=14, max_length=14)
    cross_review: CrossReview
    request_coverage: RequestCoverage
    reference_admission: ReferenceAdmission
    context_manifest: dict
    performed_experiments: Literal[False] = False
    submission_ready: Literal[False] = False

    @model_validator(mode="after")
    def profile_version(self) -> SchoolDocument:
        expected = "citequest-school-document/3" if self.profile.id == LEGACY_PROFILE_ID else "citequest-school-document/4"
        if self.schema_version != expected:
            raise ValueError("school_document_profile_version_mismatch")
        return self

    def content_hash(self) -> str:
        return digest([item.model_dump(mode="json") for item in self.sections])


def section_wire_limit(spec: SectionSpec) -> int:
    # The full structured section, including escaped metadata and text, must fit
    # this explicit capacity profile. Minimum Han and evidence gates stay intact.
    return 3 * spec.maximum_han + SchoolProfile().section_wire_overhead_bytes


def section_wire_bytes(section: SchoolSection | dict) -> int:
    value = section.model_dump(mode="json") if isinstance(section, SchoolSection) else section
    # Inner canonical prompt JSON is itself a string in the actual HTTP JSON.
    # Include that second escaping layer; UTF8 text length alone is insufficient.
    return len(json.dumps(canonical(value).decode(), ensure_ascii=False, separators=(",", ":")).encode()) - 2


def section_issues(
    section: SchoolSection, spec: SectionSpec, brief_sha256: str, claim_sources: dict[str, set[str]]
) -> list[str]:
    """Deterministic checks do not claim semantic entailment."""
    issues = []
    if section.section_id != spec.id or section.brief_sha256 != brief_sha256:
        issues.append("section_or_brief_identity_mismatch")
    count = han(section.body())
    if not spec.minimum_han <= count <= spec.maximum_han:
        issues.append(f"han_out_of_bounds:{count}:{spec.minimum_han}:{spec.maximum_han}")
    if len(section.blocks) > SchoolProfile().maximum_blocks_per_section:
        issues.append("capacity_profile_maximum_three_blocks")
    if section_wire_bytes(section) > section_wire_limit(spec):
        issues.append("capacity_profile_section_serialized_wire_bytes_exceeded")
    for block in section.blocks:
        if any((ord(char) < 32 and char not in "\n\r\t") or 0xD800 <= ord(char) <= 0xDFFF for char in block.text):
            issues.append(block.id + ":invalid_text_control_or_surrogate")
        tokens = TOKEN.findall(block.text)
        # Every '[' used for a source must have the precise issued syntax.
        residue = TOKEN.sub("", block.text)
        if re.search(r"[\[［]\s*(?:[@＠]|[EeSsＥｅＳｓ]\d)", residue):
            issues.append(block.id + ":malformed_source_citation")
        if set(tokens) != set(block.source_aliases) or not set(tokens) <= set(spec.allowed_source_aliases):
            issues.append(block.id + ":inline_citation_scope_mismatch")
        if not set(block.claim_ids) <= set(spec.allowed_claim_ids):
            issues.append(block.id + ":unknown_scientific_claim")
        if any(not set(block.source_aliases) & claim_sources.get(claim, set()) for claim in block.claim_ids):
            issues.append(block.id + ":claim_source_relation_missing")
        admitted_aliases = set().union(*(claim_sources.get(claim, set()) for claim in block.claim_ids))
        if block.kind == "evidence" and not set(block.source_aliases) <= admitted_aliases:
            issues.append(block.id + ":citation_has_no_admitted_claim_edge")
        if not set(block.user_fields) <= set(spec.user_fields):
            issues.append(block.id + ":user_fact_scope_mismatch")
    return issues


def review_issues(section: SchoolSection, spec: SectionSpec, review: SectionReview) -> list[str]:
    problems = []
    if review.content_sha256 != digest(section.model_dump(mode="json")) or review.section_id != spec.id:
        problems.append("review_content_identity_mismatch")
    index = {item.block_id: item for item in review.blocks}
    if len(index) != len(review.blocks) or set(index) != {block.id for block in section.blocks}:
        problems.append("review_must_cover_every_full_block_once")
    for block in section.blocks:
        item = index.get(block.id)
        if item and (
            item.verdict != "passed"
            or not item.quantities_conditions_and_negation_preserved
            or set(item.claim_ids) != set(block.claim_ids)
            or set(item.source_aliases) != set(block.source_aliases)
        ):
            problems.append(block.id + ":independent_content_review_failed")
    if set(spec.required_plan_ids) != set(review.covered_plan_ids) or len(set(review.covered_plan_ids)) != len(
        review.covered_plan_ids
    ):
        problems.append("required_plan_coverage_incomplete")
    if any(
        getattr(review, key)
        for key in ("fabricated_completed_work", "source_instruction_injection", "unapproved_resource_or_identity")
    ):
        problems.append("section_safety_review_failed")
    for item in review.commitments:
        if item.block_id not in index:
            problems.append("cross_section_commitment_not_in_full_section")
            continue
        block = next(block for block in section.blocks if block.id == item.block_id)
        if item.statement not in block.text or not set(item.plan_ids) <= set(spec.required_plan_ids):
            problems.append("cross_section_commitment_must_quote_current_block_and_known_plan_ids")
        if any(value not in item.statement for value in item.numeric_literals):
            problems.append("cross_section_numeric_commitment_not_literal_in_its_excerpt")
    for block in section.blocks:
        commitments = [item for item in review.commitments if item.block_id == block.id]
        if not commitments:
            problems.append(block.id + ":cross_section_commitment_missing")
        numbers = set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:%|‰)?", TOKEN.sub("", block.text)))
        represented = {value for item in commitments for value in item.numeric_literals}
        if numbers != represented:
            problems.append(block.id + ":cross_section_numeric_ledger_incomplete")
    return problems
