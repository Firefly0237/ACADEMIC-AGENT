"""Strict product inputs and the server-owned bounded runtime profiles."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def validate_user_text(value: str) -> str:
    """Reject non-storable user text before accounting or provider admission.

    Do not normalize scientific source strings or silently replace characters.
    This guard is used only by explicit user command/intake fields.
    """
    if "\x00" in value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise ValueError("user text contains NUL or an unpaired Unicode surrogate")
    return value


class Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


ID = Annotated[UUID, Field(strict=False)]
Text = Annotated[str, Field(min_length=1, max_length=8000)]
DIALOGUE_REFERENCE_PROFILE_IDS = frozenset({"qa-v6", "qa-v7", "qa-v8", "qa-v9", "stance-v3", "workset-v7"})


class RuntimeLimits(Strict):
    """Trusted lifetime development policy; no caller input, hidden reset or daily window."""
    mode: Literal["bounded", "development_aggregate_unlimited"]
    owner_cash_ncny: int = Field(default=100000000000, ge=1, le=100000000000)
    project_cash_ncny: int = Field(default=500000000000, ge=1, le=500000000000)
    owner_active_tasks: int = Field(default=2, ge=1, le=2)
    owner_total_tasks: int = Field(default=200, ge=1, le=200)
    # Optional database-wide development campaign. Product owner/project limits remain effective.
    campaign_cash_ncny: int | None = Field(default=None, ge=1, le=500000000000)
    campaign_prior_reserved_ncny: int = Field(default=0, ge=0, le=500000000000)
    model_dispatch_concurrency: int | None = Field(default=None, ge=1, le=1)

    @model_validator(mode="after")
    def bounded_campaign(self) -> RuntimeLimits:
        if self.campaign_cash_ncny is None:
            if self.campaign_prior_reserved_ncny:
                raise ValueError("carry-over reservation requires a campaign cap")
        elif self.mode != "bounded" or self.campaign_prior_reserved_ncny > self.campaign_cash_ncny:
            raise ValueError("campaign must be bounded and retain prior unknown liability")
        return self


class BudgetProfile(Strict):
    profile_id: str
    model_alias: str = "deepseek-flash"
    model_version: str = "DeepSeek-V4.1-Flash-1017"
    model_calls: int = Field(ge=1, le=64)
    tool_calls: int = Field(ge=1, le=128)
    input_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=1)
    per_call_output: int = Field(ge=1, le=65536)
    active_seconds: int = Field(ge=1, le=1800)
    cash_ncny: int = Field(ge=1)
    request_bytes: int = Field(ge=1, le=262144)
    review_read_seconds: int | None = Field(default=None, ge=1, le=300, exclude_if=lambda value: value is None)
    context_window: int = 1048576
    input_miss_ncny: int = 2000
    input_cached_ncny: int = 40
    output_ncny: int = 8000


PROFILES = {"qa-v1": BudgetProfile(profile_id="qa-v1", model_calls=6, tool_calls=8,
    input_tokens=24000, output_tokens=6000, per_call_output=1500, active_seconds=120,
    cash_ncny=15000000000, request_bytes=24576),
    "qa-v2": BudgetProfile(profile_id="qa-v2", model_calls=6, tool_calls=28,
    input_tokens=96000, output_tokens=12288, per_call_output=3072, active_seconds=120,
    cash_ncny=15000000000, request_bytes=49152),
    "qa-v3": BudgetProfile(profile_id="qa-v3", model_calls=6, tool_calls=28,
    input_tokens=96000, output_tokens=16384, per_call_output=4096, active_seconds=120,
    cash_ncny=15000000000, request_bytes=49152),
    "qa-v4": BudgetProfile(profile_id="qa-v4", model_calls=10, tool_calls=28,
    input_tokens=192000, output_tokens=30720, per_call_output=4096, active_seconds=120,
    cash_ncny=15000000000, request_bytes=49152),
    "qa-v5": BudgetProfile(profile_id="qa-v5", model_calls=18, tool_calls=64,
    input_tokens=384000, output_tokens=73728, per_call_output=4096, active_seconds=120,
    cash_ncny=15000000000, request_bytes=131072),
    "qa-v6": BudgetProfile(profile_id="qa-v6", model_calls=18, tool_calls=64,
    input_tokens=384000, output_tokens=73728, per_call_output=4096, active_seconds=450,
    cash_ncny=15000000000, request_bytes=131072),
    "qa-v9": BudgetProfile(profile_id="qa-v9", model_alias="deepseek-v4-pro",
        model_version="DeepSeek-V4-Pro-0813", model_calls=18, tool_calls=64,
        input_tokens=384000, output_tokens=1179648, per_call_output=65536, active_seconds=1800,
        cash_ncny=50000000000, request_bytes=262144, review_read_seconds=300, context_window=1048576,
        input_miss_ncny=9000, input_cached_ncny=300, output_ncny=27000),
    "qa-v8": BudgetProfile(profile_id="qa-v8", model_alias="deepseek-v4-pro",
        model_version="DeepSeek-V4-Pro-0813", model_calls=18, tool_calls=64,
        input_tokens=384000, output_tokens=147456, per_call_output=8192, active_seconds=900,
        cash_ncny=20000000000, request_bytes=262144, review_read_seconds=180, context_window=1048576,
        input_miss_ncny=9000, input_cached_ncny=300, output_ncny=27000),
    "qa-v7": BudgetProfile(profile_id="qa-v7", model_alias="deepseek-v4-pro",
        model_version="DeepSeek-V4-Pro-0813", model_calls=18, tool_calls=64,
        input_tokens=384000, output_tokens=73728, per_call_output=4096, active_seconds=450,
        cash_ncny=15000000000, request_bytes=131072, context_window=1048576,
        input_miss_ncny=9000, input_cached_ncny=300, output_ncny=27000),
    "stance-v1-r2": BudgetProfile(profile_id="stance-v1-r2", model_calls=20, tool_calls=24,
    input_tokens=90000, output_tokens=24000, per_call_output=1500, active_seconds=450,
    cash_ncny=25000000000, request_bytes=24576),
    "stance-v2": BudgetProfile(profile_id="stance-v2", model_calls=24, tool_calls=64,
    input_tokens=576000, output_tokens=98304, per_call_output=4096, active_seconds=450,
    cash_ncny=25000000000, request_bytes=131072),
    "stance-v3": BudgetProfile(profile_id="stance-v3", model_calls=24, tool_calls=64,
    input_tokens=576000, output_tokens=98304, per_call_output=4096, active_seconds=450,
    cash_ncny=25000000000, request_bytes=131072),
    "workset-v1": BudgetProfile(profile_id="workset-v1", model_calls=64, tool_calls=64,
    input_tokens=256000, output_tokens=48000, per_call_output=1500, active_seconds=600,
    cash_ncny=50000000000, request_bytes=24576),
    "workset-v2": BudgetProfile(profile_id="workset-v2", model_calls=64, tool_calls=64,
    input_tokens=512000, output_tokens=96000, per_call_output=4096, active_seconds=600,
    cash_ncny=50000000000, request_bytes=49152),
    "workset-v3": BudgetProfile(profile_id="workset-v3", model_calls=64, tool_calls=64,
    input_tokens=512000, output_tokens=96000, per_call_output=4096, active_seconds=600,
    cash_ncny=50000000000, request_bytes=49152),
    "workset-v4": BudgetProfile(profile_id="workset-v4", model_calls=64, tool_calls=64,
    input_tokens=512000, output_tokens=96000, per_call_output=4096, active_seconds=600,
    cash_ncny=50000000000, request_bytes=131072),
    "workset-v5": BudgetProfile(profile_id="workset-v5", model_calls=64, tool_calls=64,
    input_tokens=1024000, output_tokens=160000, per_call_output=4096, active_seconds=600,
    cash_ncny=50000000000, request_bytes=131072),
    "workset-v6": BudgetProfile(profile_id="workset-v6", model_calls=64, tool_calls=64,
    input_tokens=1024000, output_tokens=160000, per_call_output=4096, active_seconds=600,
    cash_ncny=50000000000, request_bytes=131072),
    "workset-v7": BudgetProfile(profile_id="workset-v7", model_calls=64, tool_calls=64,
    input_tokens=1024000, output_tokens=160000, per_call_output=4096, active_seconds=600,
    cash_ncny=50000000000, request_bytes=131072),
    "research-v1": BudgetProfile(profile_id="research-v1", model_calls=64, tool_calls=64,
    input_tokens=512000, output_tokens=96000, per_call_output=2048, active_seconds=1800,
    cash_ncny=80000000000, request_bytes=65536),
    "research-v2": BudgetProfile(profile_id="research-v2", model_calls=64, tool_calls=64,
    input_tokens=512000, output_tokens=96000, per_call_output=2048, active_seconds=1800,
    cash_ncny=80000000000, request_bytes=131072),
    "research-v3": BudgetProfile(profile_id="research-v3", model_calls=64, tool_calls=64,
    input_tokens=512000, output_tokens=96000, per_call_output=2048, active_seconds=1800,
    cash_ncny=80000000000, request_bytes=131072),
    "research-v4": BudgetProfile(profile_id="research-v4", model_calls=64, tool_calls=64,
    input_tokens=512000, output_tokens=96000, per_call_output=2048, active_seconds=1800,
    cash_ncny=80000000000, request_bytes=131072),
    "research-v5": BudgetProfile(profile_id="research-v5", model_calls=64, tool_calls=80,
    input_tokens=512000, output_tokens=131072, per_call_output=2048, active_seconds=1800,
    cash_ncny=80000000000, request_bytes=131072)}
PROFILES["research-v6"] = PROFILES["research-v5"].model_copy(update={"profile_id": "research-v6"})
PROFILES["research-review-v1"] = BudgetProfile(profile_id="research-review-v1", model_calls=8,
    tool_calls=80, input_tokens=8388608, output_tokens=32768, per_call_output=4096,
    active_seconds=900, cash_ncny=20000000000, request_bytes=262144)
PROFILES["research-revision-v1"] = BudgetProfile(profile_id="research-revision-v1", model_calls=18,
    tool_calls=128, input_tokens=18874368, output_tokens=36864, per_call_output=2048,
    active_seconds=1800, cash_ncny=40000000000, request_bytes=262144)
LEGACY_GRAPH_VERSION = "citequest-agent-graph/5-observed-source-workset"
GRAPH_VERSION = "citequest-agent-graph/8-current-pinned-skills-and-context"
RESEARCH_GRAPH_VERSION = "citequest-agent-graph/6-versioned-research-and-school-document"
WORKSET_V2_GRAPH_VERSION = "citequest-agent-graph/7-compact-source-wire-workset"
WORKSET_V3_GRAPH_VERSION = "citequest-agent-graph/9-server-selected-quote-spans"
QA_V2_GRAPH_VERSION = "citequest-agent-graph/10-bounded-sequential-tool-batches"
RESEARCH_V2_GRAPH_VERSION = "citequest-agent-graph/11-full-research-review-wire-capacity"
QA_V3_GRAPH_VERSION = "citequest-agent-graph/12-fixed-identity-per-quote-review-grid"
QA_V4_GRAPH_VERSION = "citequest-agent-graph/13-durable-typed-feedback-qa"
WORKSET_V4_GRAPH_VERSION = "citequest-agent-graph/14-full-workset-request-capacity"
WORKSET_V5_GRAPH_VERSION = "citequest-agent-graph/15-grouped-source-atomic-workset"
QA_V5_GRAPH_VERSION = "citequest-agent-graph/16-typed-current-requirements-and-chunk-QA"
RESEARCH_V3_GRAPH_VERSION = "citequest-agent-graph/17-server-Q-literature-and-shared-revision"
STANCE_V2_GRAPH_VERSION = "citequest-agent-graph/18-current-requirements-and-atomic-stance-plan"
WORKSET_V6_GRAPH_VERSION = "citequest-agent-graph/19-typed-current-requirements-and-grouped-workset"
QA_V6_GRAPH_VERSION = "citequest-agent-graph/20-multiturn-QA-bound-goal-review"
QA_GROUNDED_GRAPH_VERSION = "citequest-agent-graph/26-grounded-multiturn-QA"
QA_AUDITED_GRAPH_VERSION = "citequest-agent-graph/27-requirement-and-gap-audited-QA"
QA_VERIFIED_GRAPH_VERSION = "citequest-agent-graph/28-quantitative-QA-and-bounded-protocol-recovery"
QA_PRO_GRAPH_VERSION = "citequest-agent-graph/29-registered-Pro-grounded-QA"
QA_STRUCTURED_GRAPH_VERSION = "citequest-agent-graph/30-typed-submission-for-all-QA-roles"
QA_PATCH_GRAPH_VERSION = "citequest-agent-graph/31-scoped-answer-patch-QA"
QA_JSON_REVIEW_GRAPH_VERSION = "citequest-agent-graph/32-explicit-JSON-review-QA"
QA_REFERENCE_CONTEXT_GRAPH_VERSION = "citequest-agent-graph/33-reference-context-QA"
QA_ANCHORED_GRAPH_VERSION = "citequest-agent-graph/34-grounded-grader-and-reasoning-QA"
QA_DECOMPOSED_GRAPH_VERSION = "citequest-agent-graph/35-decomposed-setting-and-bounded-gaps-QA"
QA_ONLINE_GRAPH_VERSION = "citequest-agent-graph/36-online-grounded-review-QA"
QA_SEPARATED_GRAPH_VERSION = "citequest-agent-graph/37-isolated-goal-and-source-review-QA"
STANCE_V3_GRAPH_VERSION = "citequest-agent-graph/21-atomic-stance-bound-goal-review"
WORKSET_V7_GRAPH_VERSION = "citequest-agent-graph/22-grouped-workset-bound-goal-review"
RESEARCH_V4_GRAPH_VERSION = "citequest-agent-graph/23-research-host-bound-control-identities"
RESEARCH_V5_GRAPH_VERSION = "citequest-agent-graph/25-research-reviewed-outline-bound-input-gaps"
RESEARCH_REVIEW_GRAPH_VERSION = "citequest-agent-graph/39-version-bound-research-review"
RESEARCH_REVISION_GRAPH_VERSION = "citequest-agent-graph/40-finding-bound-targeted-research-revision"
RESEARCH_V6_GRAPH_VERSION = "citequest-agent-graph/38-standalone-reviewed-experiment-design"
CURRENT_GRAPH_VERSIONS = (RESEARCH_REVISION_GRAPH_VERSION, RESEARCH_REVIEW_GRAPH_VERSION, RESEARCH_V6_GRAPH_VERSION, GRAPH_VERSION, RESEARCH_GRAPH_VERSION, WORKSET_V2_GRAPH_VERSION, WORKSET_V3_GRAPH_VERSION, QA_V2_GRAPH_VERSION, RESEARCH_V2_GRAPH_VERSION, QA_V3_GRAPH_VERSION, QA_V4_GRAPH_VERSION, WORKSET_V4_GRAPH_VERSION, WORKSET_V5_GRAPH_VERSION, QA_V5_GRAPH_VERSION, RESEARCH_V3_GRAPH_VERSION, STANCE_V2_GRAPH_VERSION, WORKSET_V6_GRAPH_VERSION, QA_V6_GRAPH_VERSION, STANCE_V3_GRAPH_VERSION, WORKSET_V7_GRAPH_VERSION, RESEARCH_V4_GRAPH_VERSION, RESEARCH_V5_GRAPH_VERSION)
POLICY_VERSION = "citequest-agent-policy/1"


def graph_version_for_kind(kind: str, profile_id: str | None = None) -> str:
    if kind == "research" and profile_id == "research-review-v1":
        return RESEARCH_REVIEW_GRAPH_VERSION
    if kind == "research" and profile_id == "research-revision-v1":
        return RESEARCH_REVISION_GRAPH_VERSION
    if kind == "qa" and profile_id in {"qa-v8", "qa-v9"}:
        return QA_ANCHORED_GRAPH_VERSION
    if kind == "qa" and profile_id == "qa-v7":
        return QA_PRO_GRAPH_VERSION
    if kind == "qa" and profile_id == "qa-v6":
        return QA_V6_GRAPH_VERSION
    if kind == "stance" and profile_id == "stance-v3":
        return STANCE_V3_GRAPH_VERSION
    if kind in {"summary", "compare"} and profile_id == "workset-v7":
        return WORKSET_V7_GRAPH_VERSION
    if kind == "stance" and profile_id == "stance-v2":
        return STANCE_V2_GRAPH_VERSION
    if kind in {"summary", "compare"} and profile_id == "workset-v6":
        return WORKSET_V6_GRAPH_VERSION
    if kind in {"summary", "compare"} and profile_id == "workset-v5":
        return WORKSET_V5_GRAPH_VERSION
    if kind == "research":
        return {"research-v2":RESEARCH_V2_GRAPH_VERSION,"research-v3":RESEARCH_V3_GRAPH_VERSION,"research-v4":RESEARCH_V4_GRAPH_VERSION,"research-v5":RESEARCH_V5_GRAPH_VERSION,"research-v6":RESEARCH_V6_GRAPH_VERSION}.get(profile_id,RESEARCH_GRAPH_VERSION)
    if kind == "qa" and profile_id in {"qa-v2", "qa-v3", "qa-v4", "qa-v5"}:
        return {"qa-v2": QA_V2_GRAPH_VERSION, "qa-v3": QA_V3_GRAPH_VERSION, "qa-v4": QA_V4_GRAPH_VERSION, "qa-v5": QA_V5_GRAPH_VERSION}[profile_id]
    if kind in {"summary", "compare"} and profile_id in {"workset-v3", "workset-v4"}:
        return WORKSET_V4_GRAPH_VERSION if profile_id == "workset-v4" else WORKSET_V3_GRAPH_VERSION
    return WORKSET_V2_GRAPH_VERSION if kind in {"summary", "compare"} and profile_id == "workset-v2" else GRAPH_VERSION


class VersionedBriefRef(Strict):
    id: ID
    revision: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ReviewSource(Strict):
    source_task_id: ID
    source_command_id: ID
    command_seq: int = Field(ge=1)
    event_seq: int = Field(ge=1)
    task_revision: int = Field(ge=1)
    document_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    context_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ReviewReportSource(Strict):
    task_id: ID
    command_id: ID
    command_seq: int = Field(ge=1)
    event_seq: int = Field(ge=1)
    task_revision: int = Field(ge=1)
    report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    context_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class RevisionRequest(Strict):
    report: ReviewReportSource
    finding_keys: list[Annotated[str, Field(pattern=r"^(evidence:E|methods:M|structure:S)[1-8]$")]] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_findings(self):
        if len(set(self.finding_keys)) != len(self.finding_keys):
            raise ValueError("selected findings must be unique")
        return self


class SkillPin(Strict):
    skill_id: ID
    version_id: ID
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    head_revision: int = Field(ge=1)


class EvidenceSelection(Strict):
    kind: Literal["metadata_projection", "native_representation", "material_file"]
    id: Annotated[str, Field(min_length=1, max_length=64)]

    @field_validator("id")
    @classmethod
    def canonical_typed_id(cls, value: str, info):
        if info.data.get("kind") == "native_representation":
            if re.fullmatch(r"[a-f0-9]{64}", value) is None:
                raise ValueError("native representation requires its SHA256 ID")
            return value
        return str(UUID(value))


class ResearchBrief(Strict):
    schema_version: Literal["citequest-agent-brief/1"] = "citequest-agent-brief/1"
    question: Annotated[str, Field(max_length=8000)] = ""
    reference_turn_seq: int | None = Field(default=None, ge=1, le=9007199254740991,
        exclude_if=lambda value: value is None)
    kind: Literal["qa", "summary", "compare", "stance", "research"] = "qa"
    requested_direction: Literal["neutral", "support", "refute"] = "neutral"
    objective: Annotated[str, Field(max_length=2000)] = "Answer the research question using permitted evidence."
    catalog: Literal["application_papers", "standard_metadata"] = "application_papers"
    selected_papers: list[EvidenceSelection] = Field(default_factory=list, max_length=20)
    constraints: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(default_factory=list, max_length=12)
    unknowns: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(default_factory=list, max_length=12)
    research_brief: VersionedBriefRef | None = Field(default=None, exclude_if=lambda value: value is None)
    selected_skills: list[SkillPin] = Field(default_factory=list, max_length=3, exclude_if=lambda value: not value)
    review_source: ReviewSource | None = Field(default=None, exclude_if=lambda value: value is None)
    revision_request: RevisionRequest | None = Field(default=None, exclude_if=lambda value: value is None)
    research_action: Literal["plan", "suggest", "extract", "experiment", "review", "revise"] = Field(default="plan", exclude_if=lambda value: value == "plan")
    research_help_fields: list[Annotated[str, Field(min_length=1, max_length=40)]] = Field(default_factory=list,
        max_length=3, exclude_if=lambda value: not value)
    research_draft_files: list[ID] = Field(default_factory=list, max_length=4, exclude_if=lambda value: not value)
    research_extract_fields: list[Annotated[str, Field(min_length=1, max_length=40)]] = Field(default_factory=list,
        max_length=16, exclude_if=lambda value: not value)

    @field_validator("question", "objective")
    @classmethod
    def valid_user_text(cls, value: str) -> str:
        return validate_user_text(value)

    @field_validator("constraints", "unknowns")
    @classmethod
    def valid_user_text_list(cls, values: list[str]) -> list[str]:
        return [validate_user_text(value) for value in values]

    @model_validator(mode="after")
    def direction_matches_kind(self):
        if (self.kind == "research") != (self.research_brief is not None):
            raise ValueError("research tasks require an authorized versioned research Brief reference")
        if (self.research_action in {"review", "revise"}) != (self.review_source is not None):
            raise ValueError("research review requires one immutable source version")
        if self.research_action in {"review", "revise"} and self.kind != "research":
            raise ValueError("research review requires a research task")
        if (self.research_action == "revise") != (self.revision_request is not None):
            raise ValueError("targeted revision requires a fixed review and selected findings")
        if self.research_action == "experiment" and self.kind != "research":
            raise ValueError("experiment design requires a research task")
        if self.research_action == "suggest":
            from citequest.research_v3.brief import FIELD_ORDER
            if self.kind != "research" or not self.research_help_fields or any(field not in FIELD_ORDER for field in self.research_help_fields):
                raise ValueError("research suggestions require one to three registered intake fields")
            if len(set(self.research_help_fields)) != len(self.research_help_fields):
                raise ValueError("research help fields must be unique")
        elif self.research_help_fields:
            raise ValueError("planning cannot include a suggestion field request")
        if self.research_draft_files:
            materials = {item.id for item in self.selected_papers if item.kind == "material_file"}
            if self.kind != "research" or len(set(self.research_draft_files)) != len(self.research_draft_files) or any(str(value) not in materials for value in self.research_draft_files):
                raise ValueError("research drafts must be unique selected material file IDs")
        if self.research_action == "extract":
            from citequest.research_v3.brief import FIELD_ORDER
            if self.kind != "research" or not self.research_draft_files or not self.research_extract_fields or any(field not in FIELD_ORDER for field in self.research_extract_fields):
                raise ValueError("draft extraction requires selected drafts and registered fields")
            if len(set(self.research_extract_fields)) != len(self.research_extract_fields):
                raise ValueError("draft extraction fields must be unique")
        elif self.research_extract_fields:
            raise ValueError("only draft extraction may request extraction fields")
        if (self.kind == "stance") == (self.requested_direction == "neutral"):
            raise ValueError("stance requires explicit support/refute; other kinds require neutral")
        selected = [(item.kind, item.id) for item in self.selected_papers]
        if len({item.skill_id for item in self.selected_skills}) != len(self.selected_skills):
            raise ValueError("selected skill pins must be unique")
        if len(set(selected)) != len(selected):
            raise ValueError("selected source locators must be unique")
        if self.kind in {"summary", "compare"} and len(selected) > 10:
            raise ValueError("summary/compare may use at most ten selected documents")
        return self


class TaskCreate(Strict):
    project_id: ID
    operation_id: ID
    profile_id: Literal["qa-v1", "qa-v2", "qa-v3", "qa-v4", "qa-v5", "qa-v6", "qa-v7", "qa-v8", "qa-v9", "stance-v1-r2", "stance-v2", "stance-v3", "workset-v1", "workset-v2", "workset-v3", "workset-v4", "workset-v5", "workset-v6", "workset-v7", "research-v1", "research-v2", "research-v3", "research-v4", "research-v5", "research-v6", "research-review-v1", "research-revision-v1"] = "qa-v1"
    kind: Literal["qa", "summary", "compare", "stance", "research"] = "qa"
    brief: ResearchBrief

    @model_validator(mode="after")
    def same_kind(self):
        from citequest.platform_v3.input_text import request_text
        request_text(self.brief.question, allow_empty=self.kind == "research")
        if (self.profile_id == "research-revision-v1") != (self.brief.research_action == "revise"):
            raise ValueError("targeted revision action requires its dedicated profile")
        if (self.profile_id == "research-review-v1") != (self.brief.research_action == "review"):
            raise ValueError("research review action requires its dedicated profile")
        if self.brief.research_action == "experiment" and (self.kind != "research" or self.profile_id != "research-v6"):
            raise ValueError("standalone experiment design requires the versioned research-v6 workflow")
        if self.kind != self.brief.kind:
            raise ValueError("Task and Brief kind must match")
        if self.brief.reference_turn_seq is not None:
            raise ValueError("a new task cannot reference a prior task turn")
        expected = "research-v1" if self.kind == "research" else ("stance-v1-r2" if self.kind == "stance" else ("workset-v1" if self.kind in {"summary", "compare"} else "qa-v1"))
        if self.profile_id != expected and not (self.kind in {"summary", "compare"} and self.profile_id in {"workset-v2", "workset-v3", "workset-v4", "workset-v5", "workset-v6", "workset-v7"}) and not (self.kind == "qa" and self.profile_id in {"qa-v2", "qa-v3", "qa-v4", "qa-v5", "qa-v6", "qa-v7", "qa-v8", "qa-v9"}) and not (self.kind == "research" and self.profile_id in {"research-v2", "research-v3", "research-v4", "research-v5", "research-v6", "research-review-v1", "research-revision-v1"}) and not (self.kind == "stance" and self.profile_id in {"stance-v2", "stance-v3"}):
            raise ValueError("profile does not match the trusted task-kind policy")
        selected = [(item.kind, item.id) for item in self.brief.selected_papers]
        if len(set(selected)) != len(selected):
            raise ValueError("selected source locators must be unique")
        if self.kind in {"summary", "compare"} and len(selected) > 10:
            raise ValueError("summary/compare may use at most ten selected documents")
        return self


class TaskCommand(Strict):
    operation_id: ID
    expected_revision: int = Field(ge=1)
    kind: Literal["resume", "followup", "cancel"]
    message: Annotated[str, Field(max_length=8000)] = ""
    reference_turn_seq: int | None = Field(default=None, ge=1, le=9007199254740991,
        exclude_if=lambda value: value is None)
    requested_direction: Literal["support", "refute"] | None = None
    research_brief: VersionedBriefRef | None = Field(default=None, exclude_if=lambda value: value is None)
    selected_skills: list[SkillPin] | None = Field(default=None, max_length=3, exclude_if=lambda value: value is None)
    research_action: Literal["plan", "suggest", "extract", "experiment", "review", "revise"] | None = Field(default=None, exclude_if=lambda value: value is None)
    research_help_fields: list[Annotated[str, Field(min_length=1, max_length=40)]] | None = Field(default=None,
        max_length=3, exclude_if=lambda value: value is None)
    research_extract_fields: list[Annotated[str, Field(min_length=1, max_length=40)]] | None = Field(default=None,
        max_length=16, exclude_if=lambda value: value is None)

    @field_validator("message")
    @classmethod
    def valid_user_message(cls, value: str) -> str:
        from citequest.platform_v3.input_text import request_text
        return request_text(value, allow_empty=True)

    @model_validator(mode="after")
    def direction_command(self):
        if self.kind == "cancel" and self.reference_turn_seq is not None:
            raise ValueError("cancel cannot select a prior answer")
        if self.selected_skills is not None and len({item.skill_id for item in self.selected_skills}) != len(self.selected_skills):
            raise ValueError("selected skill pins must be unique")
        if self.kind == "cancel" and (self.requested_direction is not None or self.research_brief is not None or self.selected_skills is not None):
            raise ValueError("cancel cannot change a direction or research Brief")
        if self.kind == "cancel" and (self.research_action is not None or self.research_help_fields is not None or self.research_extract_fields is not None):
            raise ValueError("cancel cannot request research assistance")
        return self


class MemoryWrite(Strict):
    operation_id: ID
    expected_revision: int = Field(ge=0)
    kind: Literal["preference", "decision", "constraint"]
    value: Text
    source_refs: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(default_factory=list, max_length=20)
    expires_at: AwareDatetime | None = None

    @field_validator("value")
    @classmethod
    def valid_memory_text(cls, value: str) -> str:
        return validate_user_text(value)


class MemoryDelete(Strict):
    operation_id: ID
    expected_revision: int = Field(ge=1)


class Citation(Strict):
    ref_id: Annotated[str, Field(min_length=1, max_length=300)]
    quote: Annotated[str, Field(min_length=1, max_length=1200)]


class Answer(Strict):
    text: Annotated[str, Field(min_length=1, max_length=12000)]
    citations: list[Citation] = Field(max_length=12)
    gaps: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(max_length=12)
    evidence_status: Literal["supported", "insufficient"]


class AgentError(RuntimeError):
    def __init__(self, code: str, status: int = 409) -> None:
        self.code, self.status = code, status
        super().__init__(code)


def redact(value: object) -> object:
    if isinstance(value, dict):
        if value.get("type") in {"reasoning", "reasoning_text"}:
            return {"type": value["type"], "internal_content_recorded": False}
        return {k: redact(v) for k, v in value.items() if k.lower() not in
            {"authorization", "headers", "api_key", "deepseek_api_key", "reasoning_content", "encrypted_content"}}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value
