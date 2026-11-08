"""Checkpointable school-writing child graph on its parent's Execution and wire.

No Task, provider, wallet, model client or independent run window is created.
Compile this as an inherited-checkpointer subgraph in the parent's research node;
standalone controls may supply an explicit in-memory checkpointer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from citequest.research_v3.contracts import ResearchAnswer, Review, validate_literal, validate_review
from citequest.research_v3.research_plan import ResearchPlan, validate_plan_context
from citequest.research_v3.school_contracts import (
    CrossReview,
    ReferenceAdmission,
    RequestCoverage,
    SchoolDocument,
    SchoolProfile,
    SchoolSection,
    SectionReview,
    SectionSpec,
    han,
    review_issues,
    section_issues,
    section_wire_bytes,
    section_wire_limit,
)
from citequest.research_v3.school_prompts import CROSS_REVIEWER, REQUEST_REVIEWER, SECTION_REVIEWER, WRITER

from .contracts import AgentError, digest


class SchoolState(TypedDict, total=False):
    upstream: dict
    intake: dict
    hard_memory_refs: list[dict]
    evidence: list[dict]
    specs: list[dict]
    aliases: dict[str, list[str]]
    claims: dict[str, dict]
    claim_sources: dict[str, list[str]]
    reference_admission: dict
    sections: dict[str, dict]
    section_model_keys: dict[str, str]
    reviews: dict[str, dict]
    review_history: list[dict]
    index: int
    repairs: int
    writing_start_model_calls: int
    issues: list[str]
    cross_review: dict
    request_coverage: dict
    repair_target: str
    return_after_repair: str
    next: str
    result: dict
    context_preflight: list[dict]


def make_specs(upstream: dict, claims: dict[str, dict], aliases: dict[str, list[str]]) -> list[SectionSpec]:
    plan = ResearchPlan.model_validate(upstream["plan"], strict=True)
    rq = [item.id for item in plan.questions]
    experiments = [item.id for item in plan.experiments]
    fields = list(upstream["research_intake"]["user_confirmed"])
    layouts = [
        (
            "review_scope",
            "review",
            "研究背景与问题边界",
            "Bound the actual problem, source coverage and significance without exhaustive claims.",
            750,
            950,
            rq,
        ),
        (
            "review_taxonomy",
            "review",
            "相关技术与研究脉络",
            "Compare technical approaches and their assumptions using the admitted evidence.",
            750,
            950,
            rq,
        ),
        (
            "review_baselines",
            "review",
            "已有方法与比较条件",
            "Analyze verified baselines and conditions; retain unavailable details as gaps.",
            750,
            950,
            rq,
        ),
        (
            "review_evaluation",
            "review",
            "评价方法与可比性",
            "Discuss actual tasks, evaluation context, units and limits of cross-paper comparison.",
            750,
            950,
            rq,
        ),
        (
            "review_support",
            "review",
            "已有证据的贡献",
            "Explain scoped supported findings and what each finding does not establish.",
            750,
            950,
            rq,
        ),
        (
            "review_limits",
            "review",
            "分歧与适用局限",
            "Preserve counterevidence, negation, uncertainty, untested conditions and source gaps.",
            750,
            950,
            rq,
        ),
        (
            "review_gap",
            "review",
            "研究空白与问题衔接",
            "Connect scoped literature limitations to the reviewed proposed questions, without asserting universal absence.",
            750,
            950,
            rq,
        ),
        (
            "objectives",
            "objectives",
            "研究目标与关键问题",
            "Explain each reviewed RQ, its motivation, outputs and falsifiable question.",
            650,
            850,
            rq,
        ),
        (
            "methods_design",
            "methods",
            "研究方法与实验设计",
            "Describe proposed data/access, method/variables, baselines and controls for every Experiment.",
            650,
            850,
            experiments,
        ),
        (
            "methods_evaluation",
            "methods",
            "评价指标与统计方案",
            "Describe proposed units, repetitions, uncertainty and stopping rules for every Experiment.",
            650,
            850,
            experiments,
        ),
        (
            "feasibility",
            "methods",
            "可行性与风险边界",
            "Explain confirmed resource ceilings, obtainable inputs, risks and stop conditions; no assumed infrastructure.",
            500,
            700,
            experiments,
        ),
        (
            "innovation",
            "innovation",
            "创新假设与验证方式",
            "Present reviewed innovation candidates as hypotheses and their disconfirmation conditions, never proven novelty.",
            350,
            500,
            rq,
        ),
        (
            "schedule",
            "schedule",
            "进度安排与预期成果",
            "Give a proposed schedule and outputs within confirmed constraints, not invented calendar dates or achievements.",
            500,
            650,
            experiments,
        ),
        (
            "prior",
            "prior",
            "相关积累与前期工作",
            "Report only confirmed prior work; if unknown say it is unprovided. Do not fill the section with invented experience.",
            0,
            350,
            [],
        ),
    ]
    return [
        SectionSpec(
            id=i,
            group=g,
            title=t,
            purpose=p,
            minimum_han=lo,
            maximum_han=hi,
            required_plan_ids=ids,
            user_fields=fields,
            allowed_claim_ids=list(claims),
            allowed_source_aliases=list(aliases),
        )
        for i, g, t, p, lo, hi, ids in layouts
    ]


class SchoolDocumentWorkflow:
    NODES = (
        "prepare",
        "write_section",
        "review_section",
        "decide_section",
        "repair_section",
        "review_cross",
        "review_requests",
        "publish",
    )

    def __init__(
        self,
        research_boundary: Any,
        *,
        checkpointer: Any = None,
        reference_preflight: Callable[[dict], dict] | None = None,
        allow_fixture_references: bool = False,
    ) -> None:
        self.boundary = research_boundary
        self.host = research_boundary.host
        self.profile = SchoolProfile()
        self.reference_preflight = reference_preflight
        self.allow_fixture_references = allow_fixture_references
        builder = StateGraph(SchoolState)
        for name in self.NODES:
            builder.add_node(name, self.observe(name, getattr(self, name)))
            builder.add_conditional_edges(name, lambda state: state["next"], [*self.NODES, END])
        builder.add_edge(START, "prepare")
        self.workflow = builder.compile(checkpointer=checkpointer)

    def observe(self, name: str, node: Any) -> Any:
        def observed(state: dict) -> dict:
            self.host.execution.progress("school_" + name, "started")
            if state.get("intake"):
                if self.boundary.synchronized_input().model_dump(mode="json") != state["intake"]:
                    return self.partial(state, "versioned_brief_changed_requires_research_review")
                memories = [
                    {"id": item["memory_id"], "revision": item["revision"]} for item in self.boundary.hard_memories()
                ]
                if memories != state["hard_memory_refs"]:
                    return self.partial(state, "hard_memory_changed_requires_research_review")
            try:
                result = node(state)
            except (ValidationError, ValueError) as error:
                # Paid output remains recorded by the shared provider. No retry,
                # truncation, fake JSON repair or unreviewed publication occurs.
                result = self.partial(state, "school_structured_or_content_contract_failed:" + type(error).__name__)
            self.host.execution.progress("school_" + name, "completed")
            return result

        return observed

    def task(self) -> dict:
        return self.host.execution.snapshot()

    def output_model(self, state: dict, model: Any, data: dict) -> Any:
        # Exact same bound model is used by wire preflight and paid validation.
        if not getattr(self.boundary, "strict_identity_binding", False):
            return model
        if model is SchoolSection:
            return self.boundary.identity_model(model, brief_sha256=state["intake"]["brief_sha256"])
        if model in {SectionReview, CrossReview, RequestCoverage}:
            return self.boundary.identity_model(model, content_sha256=data["content_sha256"])
        return model

    def partial(self, state: dict, reason: str) -> dict:
        from .research_progress import school_partial_progress

        upstream = state.get("upstream", {})
        progress = school_partial_progress(upstream, state)
        return {
            "next": END,
            "result": {
                **upstream,
                "status": "partial",
                "school_format_ready": False,
                "completion_status": "partial_school_document",
                "publication_status": "not_evaluated",
                "text": "学校研究计划草案尚未通过全部检查。",
                "school_writing": {
                    "reason": reason,
                    "completed_section_ids": list(state.get("sections", {})),
                    "section_count_required": 14,
                    "repairs": state.get("repairs", 0),
                },
                "unpublished_draft_available_to_authorized_owner": bool(state.get("sections")),
                "quality": {"school_document": "not_passed", "semantic_review": "AI_not_human_calibrated"},
                **({"research_progress": progress} if progress is not None else {}),
            },
        }

    def remaining(self, state: dict) -> int:
        current = self.task()
        stage = self.profile.maximum_writing_model_calls - (current["model_calls"] - state["writing_start_model_calls"])
        return min(stage, current["profile"]["model_calls"] - current["model_calls"])

    def prepare(self, state: dict) -> dict:
        upstream = state["upstream"]
        if (
            upstream.get("status") != "structured_plan_ready"
            or upstream.get("publication_status") != "validated_ai_review"
            or upstream.get("performed_experiments") is not False
        ):
            return self.partial(state, "reviewed_S12_plan_required")
        intake = self.boundary.synchronized_input().model_dump(mode="json")
        if intake != upstream["research_intake"]:
            return self.partial(state, "upstream_research_intake_is_not_current")
        task = self.task()
        references = upstream["evidence_ledger"]
        minimum_calls = 14 * 2 + 2
        if task["profile"].get("profile_id") == "research-v5":
            # These are the same shared credits reserved by outline admission,
            # not a new allowance for School. A cross-section repair costs a
            # writer, section reviewer and replacement cross review.
            minimum_calls += 3 * max(0, 2 - self.host.execution.content_revision_count())
        if task["profile"]["model_calls"] - task["model_calls"] < minimum_calls:
            return self.partial(state, "remaining_same_task_model_calls_below_complete_school_reserve"
                if task["profile"].get("profile_id") == "research-v5" else "remaining_same_task_model_calls_below_30")
        if task["profile"]["tool_calls"] - task["tool_calls"] < 2 * len(references):
            return self.partial(state, "remaining_same_task_tools_below_initial_and_final_source_readmission")
        if task["profile"]["output_tokens"] - task["output_tokens"] < minimum_calls * 2048:
            return self.partial(state, "remaining_same_task_output_below_reserved_full_writing_window")
        prepared = {
            **state,
            "intake": intake,
            "hard_memory_refs": upstream["context_manifest"]["hard_memory_refs"],
            "sections": {},
            "section_model_keys": {},
            "reviews": {},
            "review_history": [],
            "index": 0,
            "repairs": 0,
            "writing_start_model_calls": task["model_calls"],
        }
        evidence = []
        for entry in references:
            current = self.boundary.tool(
                prepared, "reviewer", "read_evidence", {"ref_id": entry["ref_id"]}, "school:admit:" + entry["ref_id"]
            )
            if any(current.get(key) != entry.get(key) for key in ("source_sha256", "unit_text_sha256", "source_ref")):
                return self.partial(prepared, "upstream_evidence_changed_before_writing")
            evidence.append(current)
        grouped: dict[str, list[str]] = {}
        aliases, by_ref = {}, {}
        for item in evidence:
            grouped.setdefault(digest(item["source_ref"]), []).append(item["ref_id"])
        for number, key in enumerate(sorted(grouped), 1):
            alias = "S" + str(number)
            aliases[alias] = sorted(grouped[key])
            by_ref.update(dict.fromkeys(grouped[key], alias))
        claims, claim_sources = {}, {}
        for ledger in upstream["literature_ledger"]:
            if task["profile"].get("profile_id") in {"research-v3", "research-v4", "research-v5"}:
                from citequest.research_v3.quote_spans import ResearchAnswerV3
                from citequest.research_v3.review_grid import RequestReview
                answer = ResearchAnswerV3.model_validate(ledger["answer"], strict=True)
                review = RequestReview.model_validate(ledger["review"], strict=True)
                if review.request_coverage != "complete":
                    return self.partial(prepared, "upstream_literature_request_coverage_partial")
            else:
                answer = ResearchAnswer.model_validate(ledger["answer"], strict=True)
                review = Review.model_validate(ledger["review"], strict=True)
            validate_literal(answer, evidence)
            validate_review(answer, review)
            if ledger["claim_ids"] != [ledger["research_question_id"] + ":" + claim.id for claim in answer.claims]:
                raise ValueError("upstream claim ledger identity mismatch")
            for key, claim in zip(ledger["claim_ids"], answer.claims, strict=True):
                claims[key] = claim.model_dump(mode="json")
                claim_sources[key] = sorted({by_ref[edge.ref_id] for edge in claim.edges})
        plan = ResearchPlan.model_validate(upstream["plan"], strict=True)
        validate_plan_context(plan, intake["brief_sha256"], set(claims), set(by_ref))
        specs = make_specs(upstream, claims, aliases)
        prepared.update(
            {
                "evidence": evidence,
                "claims": claims,
                "claim_sources": claim_sources,
                "aliases": aliases,
                "specs": [item.model_dump(mode="json") for item in specs],
            }
        )
        if self.reference_preflight is None:
            return self.partial(prepared, "server_reference_preflight_required_before_writing")
        from citequest.research_v3.school_export import reference_context

        try:
            reference_admission = ReferenceAdmission.model_validate(self.reference_preflight(prepared), strict=True)
        except ValueError as error:
            return self.partial(prepared, "bibliography_admission_failed_before_writing:" + str(error)[:500])
        if (
            reference_admission.context_sha256 != digest(reference_context(prepared))
            or (reference_admission.provenance != "current_authorized_metadata" and not self.allow_fixture_references)
            or {item.get("id") for item in reference_admission.references} != set(aliases)
            or len(reference_admission.references) != len(aliases)
        ):
            return self.partial(prepared, "bibliography_admission_identity_or_provenance_mismatch")
        prepared["reference_admission"] = reference_admission.model_dump(mode="json")
        # Complete prompts are checked before the first generation. No source,
        # section or user field is dropped in order to make the request fit.
        try:
            for spec in specs:
                self.boundary.payload(
                    prepared,
                    "school_writer",
                    self.output_model(prepared, SchoolSection, self.section_data(prepared, spec)),
                    "citequest_school_section",
                    WRITER,
                    self.section_data(prepared, spec),
                )
            prepared["context_preflight"] = self.full_review_preflight(prepared)
        except AgentError as error:
            if error.code == "required_context_exceeds_wire_limit":
                return self.partial(
                    prepared,
                    "maximum_full_document_review_context_exceeds_wire_limit_before_writing; narrow_confirmed_scope_or_sources",
                )
            raise
        return {**prepared, "next": "write_section"}

    def full_review_preflight(self, state: dict) -> list[dict]:
        """Probe exact encoders at the maximum enforceable section byte bounds."""
        from .context import wire_bytes

        sections, reviews, measurements = {}, {}, []
        for value in state["specs"]:
            spec = SectionSpec.model_validate(value, strict=True)
            candidate = {
                "section_id": spec.id,
                "brief_sha256": state["intake"]["brief_sha256"],
                "blocks": [
                    {
                        "id": "P1",
                        "kind": "proposal",
                        "text": "界" * spec.maximum_han,
                        "claim_ids": [],
                        "source_aliases": [],
                        "user_fields": [],
                    }
                ],
            }
            padding = section_wire_limit(spec) - section_wire_bytes(candidate)
            if padding < 0:
                raise ValueError("minimum_section_shape_exceeds_profile_wire_bound")
            candidate["blocks"][0]["text"] += "A" * padding
            if section_wire_bytes(candidate) != section_wire_limit(spec):
                raise ValueError("exact_section_wire_bound_probe_failed")
            sections[spec.id] = candidate
            reviews[spec.id] = {
                "section_id": spec.id,
                "content_sha256": digest(candidate),
                "covered_plan_ids": spec.required_plan_ids,
            }
            payload = self.boundary.payload(
                state,
                "school_independent_reviewer",
                self.output_model(state, SectionReview, {"content_sha256": digest(candidate)}),
                "citequest_school_section_review",
                SECTION_REVIEWER,
                {**self.section_data(state, spec), "candidate": candidate, "content_sha256": digest(candidate)},
            )
            measurements.append(
                {
                    "role": "school_independent_reviewer",
                    "section_id": spec.id,
                    "request_bytes": wire_bytes(payload),
                    "section_wire_bytes_bound": section_wire_limit(spec),
                }
            )
        maximal = {**state, "sections": sections, "reviews": reviews}
        data = self.cross_data(maximal)
        for role, model, name, policy, additions in (
            ("school_cross_reviewer", CrossReview, "citequest_school_cross_review", CROSS_REVIEWER, {}),
            (
                "school_request_auditor",
                RequestCoverage,
                "citequest_school_request_coverage",
                REQUEST_REVIEWER,
                {
                    "section_obligations": self.request_obligations(state),
                    "confirmed_requests": state["intake"]["user_confirmed"],
                    "explicit_unknowns": state["intake"]["explicit_unknowns"],
                },
            ),
        ):
            payload = self.boundary.payload(state, role, self.output_model(state, model, data), name, policy, {**data, **additions})
            measurements.append(
                {
                    "role": role,
                    "request_bytes": wire_bytes(payload),
                    "all14_section_wire_bytes_bound": sum(
                        section_wire_limit(SectionSpec.model_validate(v, strict=True)) for v in state["specs"]
                    ),
                }
            )
        return measurements

    def spec(self, state: dict, identifier: str | None = None) -> SectionSpec:
        return SectionSpec.model_validate(
            next(item for item in state["specs"] if item["id"] == identifier)
            if identifier
            else state["specs"][state["index"]],
            strict=True,
        )

    def section_data(self, state: dict, spec: SectionSpec) -> dict:
        plan = state["upstream"]["plan"]
        return {
            "assignment": spec.model_dump(mode="json"),
            "output_admission": {
                "maximum_blocks": self.profile.maximum_blocks_per_section,
                "full_structured_section_wire_bytes_maximum": section_wire_limit(spec),
                "measurement": "canonical section JSON escaped once more as HTTP prompt string; preserve all Han/evidence requirements",
            },
            "reviewed_claim_ledger": state["claims"],
            "source_aliases": state["aliases"],
            "untrusted_complete_evidence_windows": state["evidence"],
            "reviewed_plan": {
                "questions": plan["questions"],
                "experiments": plan["experiments"],
                "innovation_hypotheses": plan["innovation_hypotheses"],
            },
            "epistemic_rule": "Only supplied claims are scientific facts; new design choices are explicit proposals.",
        }

    def complete(self, state: dict, role: str, model: Any, name: str, policy: str, data: dict, key: str) -> Any:
        settled = (self.host.execution.settled_model_request(self.boundary.model_key(state, "school:" + key))
            if self.task()["profile"].get("profile_id") in {"research-v3", "research-v4", "research-v5"} else None)
        if self.remaining(state) <= 0 and settled is None:
            raise AgentError("school_same_task_writing_call_capacity_exhausted")
        return self.boundary.complete(state, role, self.output_model(state, model, data), name, policy, data, "school:" + key)

    def write_section(self, state: dict) -> dict:
        spec = self.spec(state)
        value = self.complete(
            state,
            "school_writer",
            SchoolSection,
            "citequest_school_section",
            WRITER,
            self.section_data(state, spec),
            spec.id + ":initial",
        )
        return {"sections": {**state["sections"], spec.id: value.model_dump(mode="json")},
            "section_model_keys": {**state.get("section_model_keys", {}), spec.id:
                self.boundary.model_key(state, "school:" + spec.id + ":initial")}
                if self.task()["profile"].get("profile_id") in {"research-v3", "research-v4", "research-v5"} else state.get("section_model_keys", {}),
            "next": "review_section"}

    def review_section(self, state: dict) -> dict:
        spec = self.spec(state, state.get("repair_target"))
        candidate = SchoolSection.model_validate(state["sections"][spec.id], strict=True)
        if (
            section_wire_bytes(candidate) > section_wire_limit(spec)
            or len(candidate.blocks) > self.profile.maximum_blocks_per_section
        ):
            return self.schedule_repair(
                state,
                spec.id,
                ["section_exceeds_explicit_capacity_wire_or_block_bound"],
                state.get("return_after_repair") or "advance",
            )
        review = self.complete(
            state,
            "school_independent_reviewer",
            SectionReview,
            "citequest_school_section_review",
            SECTION_REVIEWER,
            {
                **self.section_data(state, spec),
                "candidate": candidate.model_dump(mode="json"),
                "content_sha256": digest(candidate.model_dump(mode="json")),
            },
            spec.id + ":review:" + digest(candidate.model_dump(mode="json")),
        )
        return {
            "reviews": {**state["reviews"], spec.id: review.model_dump(mode="json")},
            "review_history": [*state["review_history"], review.model_dump(mode="json")],
            "next": "decide_section",
        }

    def decide_section(self, state: dict) -> dict:
        spec = self.spec(state, state.get("repair_target"))
        section = SchoolSection.model_validate(state["sections"][spec.id], strict=True)
        review = SectionReview.model_validate(state["reviews"][spec.id], strict=True)
        issues = section_issues(
            section,
            spec,
            state["intake"]["brief_sha256"],
            {key: set(value) for key, value in state["claim_sources"].items()},
        )
        issues += review_issues(section, spec, review)
        if spec.group == "review" and not any(item.kind == "evidence" for item in section.blocks):
            issues.append("literature_subsection_has_no_evidence_grounded_content")
        if any(item.verdict == "needs_input" for item in review.blocks):
            return self.partial(state, "section_requires_real_input:" + spec.id)
        if issues:
            return self.schedule_repair(state, spec.id, issues, state.get("return_after_repair") or "advance")
        if state.get("return_after_repair") == "review_cross":
            return {"repair_target": "", "return_after_repair": "", "next": "review_cross"}
        index = state["index"] + 1
        return {
            "repair_target": "",
            "return_after_repair": "",
            "index": index,
            "next": "write_section" if index < len(state["specs"]) else "review_cross",
        }

    def schedule_repair(self, state: dict, target: str, issues: list[str], after: str) -> dict:
        remaining_sections = max(0, len(state["specs"]) - state["index"] - 1) if after == "advance" else 0
        required = 2 + 2 * remaining_sections + 2
        if state["repairs"] >= self.profile.maximum_target_repairs or self.remaining(state) < required:
            return self.partial(state, "targeted_revision_exceeds_remaining_same_task_capacity:" + target)
        if self.task()["profile"].get("profile_id") in {"research-v3", "research-v4", "research-v5"} and self.host.execution.content_revision_count() >= 2:
            return self.partial(state, "research_shared_content_revision_limit:" + target)
        return {"repair_target": target, "issues": issues, "return_after_repair": after, "next": "repair_section"}

    def repair_section(self, state: dict) -> dict:
        spec = self.spec(state, state["repair_target"])
        # The explicit failed paragraph set is supplied; a whole old long draft
        # is never a hidden anchor and no deterministic padding/trimming occurs.
        data = {
            **self.section_data(state, spec),
            "current_candidate": state["sections"][spec.id],
            "independent_findings": state["reviews"].get(
                spec.id, {"scope": "deterministic byte admission failed; no model review dispatched"}
            ),
            "failed_checks": state["issues"],
            "repair_scope": "Only this section; preserve required claims and proposal status.",
        }
        revision = state["repairs"] + 1
        if self.task()["profile"].get("profile_id") in {"research-v3", "research-v4", "research-v5"}:
            current = state["sections"][spec.id]
            review = state["reviews"].get(spec.id)
            source_key = (self.boundary.model_key(state, "school:" + spec.id + ":review:" + digest(current))
                if review and review["content_sha256"] == digest(current)
                else state.get("section_model_keys", {}).get(spec.id))
            if not source_key:
                raise AgentError("research_school_revision_source_receipt_missing")
            try:
                self.host.execution.reserve_content_revision(
                    self.boundary.model_key(state, "school:" + spec.id + ":credit:" + str(revision)),
                    phase="research_school_content_revision", error_codes=["research_school_section_review_rejected"],
                    source_logical_key=source_key)
            except AgentError as error:
                if error.code != "content_revision_limit":
                    raise
                return self.partial(state, "research_shared_content_revision_limit:" + spec.id)
        value = self.complete(
            state,
            "school_targeted_reviser",
            SchoolSection,
            "citequest_school_section",
            WRITER,
            data,
            spec.id + ":repair:" + str(revision),
        )
        return {
            "repairs": revision,
            "sections": {**state["sections"], spec.id: value.model_dump(mode="json")},
            "section_model_keys": {**state.get("section_model_keys", {}), spec.id:
                self.boundary.model_key(state, "school:" + spec.id + ":repair:" + str(revision))}
                if self.task()["profile"].get("profile_id") in {"research-v3", "research-v4", "research-v5"} else state.get("section_model_keys", {}),
            "next": "review_section",
        }

    def ordered(self, state: dict) -> list[dict]:
        return [state["sections"][item["id"]] for item in state["specs"]]

    def cross_data(self, state: dict) -> dict:
        identifiers = [item["id"] for item in state["specs"]]
        if (
            len(identifiers) != 14
            or len(set(identifiers)) != 14
            or set(state["sections"]) != set(identifiers)
            or set(state["reviews"]) != set(identifiers)
        ):
            raise ValueError("cross_review_requires_all14_sections_and_current_reviews")
        for identifier in identifiers:
            review = state["reviews"][identifier]
            if review["section_id"] != identifier or review["content_sha256"] != digest(state["sections"][identifier]):
                raise ValueError("cross_review_section_changed_after_full_text_review")
        return {
            "content_sha256": digest(self.ordered(state)),
            "reviewed_plan": state["upstream"]["plan"],
            "full_reviewed_sections": self.ordered(state),
            "full_text_review_status": [
                {
                    "section_id": key,
                    "content_sha256": item["content_sha256"],
                    "covered_plan_ids": item["covered_plan_ids"],
                }
                for key in identifiers
                for item in [state["reviews"][key]]
            ],
        }

    def review_cross(self, state: dict) -> dict:
        sections = [SchoolSection.model_validate(item, strict=True) for item in self.ordered(state)]
        review_han = sum(
            han(item.body()) for item, spec in zip(sections, state["specs"], strict=True) if spec["group"] == "review"
        )
        total_han = sum(han(item.body()) for item in sections)
        if review_han < 5000 or not 8000 <= total_han <= 12000:
            return self.partial(state, f"whole_document_length_gate:{review_han}:{total_han}")
        texts = [block.text for item in sections for block in item.blocks]
        if len(set(texts)) != len(texts):
            return self.partial(state, "exact_repeated_paragraph_across_document")
        review = self.complete(
            state,
            "school_cross_reviewer",
            CrossReview,
            "citequest_school_cross_review",
            CROSS_REVIEWER,
            self.cross_data(state),
            "cross:" + digest(self.ordered(state)),
        )
        identifiers = {item["id"] for item in state["specs"]}
        plan = state["upstream"]["plan"]
        plan_ids = {item["id"] for item in [*plan["questions"], *plan["experiments"]]}
        if (
            review.content_sha256 != digest(self.ordered(state))
            or len(review.sections) != len(identifiers)
            or {item.section_id for item in review.sections} != identifiers
            or set(review.all_plan_ids) != plan_ids
        ):
            raise ValueError("cross review must cover every section and RQ/experiment at current content hash")
        failed = [item for item in review.sections if item.verdict != "passed"]
        if review.conflicts and not failed:
            raise ValueError("cross review conflicts cannot have all-passed section verdicts")
        if any(item.verdict == "needs_input" for item in failed):
            return self.partial(state, "cross_section_conflict_requires_user_input")
        if failed:
            if len(failed) != 1:
                return self.partial(state, "multiple_cross_section_targets_exceed_bounded_repair_plan")
            scheduled = self.schedule_repair(
                state, failed[0].section_id, [failed[0].reason, *review.conflicts], "review_cross"
            )
            return {**scheduled, "cross_review": review.model_dump(mode="json")}
        return {"cross_review": review.model_dump(mode="json"), "next": "review_requests"}

    def review_requests(self, state: dict) -> dict:
        data = {
            **self.cross_data(state),
            "section_obligations": self.request_obligations(state),
            "confirmed_requests": state["intake"]["user_confirmed"],
            "explicit_unknowns": state["intake"]["explicit_unknowns"],
        }
        value = self.complete(
            state,
            "school_request_auditor",
            RequestCoverage,
            "citequest_school_request_coverage",
            REQUEST_REVIEWER,
            data,
            "request-coverage:" + digest(self.ordered(state)),
        )
        required = set(state["intake"]["user_confirmed"]) | set(state["intake"]["explicit_unknowns"])
        if (
            value.content_sha256 != digest(self.ordered(state))
            or len(value.requests) != len(required)
            or {item.field for item in value.requests} != required
        ):
            raise ValueError("request coverage must name every current user field once")
        section_ids = {item["id"] for item in state["specs"]}
        for item in value.requests:
            expected = "explicit_unknown" if item.field in state["intake"]["explicit_unknowns"] else "covered"
            if item.state != expected or not item.section_ids or not set(item.section_ids) <= section_ids:
                return self.partial(state, "user_request_not_fully_covered:" + item.field)
        if not value.school_sections_complete or value.performed_experiments_invented:
            return self.partial(state, "school_category_or_completed_work_coverage_failed")
        return {"request_coverage": value.model_dump(mode="json"), "next": "publish"}

    @staticmethod
    def request_obligations(state: dict) -> list[dict]:
        # Preserve every assigned purpose, user field and RQ/experiment mapping.
        # Admission-only source allowlists/length counters were already checked
        # against the full section and are not duplicated fourteen times here.
        keys = ("id", "group", "title", "purpose", "required_plan_ids", "user_fields")
        return [{key: item[key] for key in keys} for item in state["specs"]]

    def publish(self, state: dict) -> dict:
        for entry in state["evidence"]:
            current = self.boundary.tool(
                state, "reviewer", "read_evidence", {"ref_id": entry["ref_id"]}, "school:publish:" + entry["ref_id"]
            )
            if any(current.get(key) != entry.get(key) for key in ("source_sha256", "unit_text_sha256", "source_ref")):
                raise AgentError("school_source_changed_before_publication")
        self.host.wait(self.host.tools.check_scope())
        if self.boundary.synchronized_input().model_dump(mode="json") != state["intake"]:
            return self.partial(state, "versioned_brief_changed_before_school_publication")
        current_memories = [
            {"id": item["memory_id"], "revision": item["revision"]} for item in self.boundary.hard_memories()
        ]
        if current_memories != state["hard_memory_refs"]:
            return self.partial(state, "hard_memory_changed_before_school_publication")
        context = {
            **state["upstream"]["context_manifest"],
            "workflow_version": "citequest-school-document-workflow/5-bound-control-identity-full14"
                if getattr(self.boundary, "strict_identity_binding", False)
                else "citequest-school-document-workflow/4-server-Q-shared-revisions-full14"
                if self.task()["profile"].get("profile_id") == "research-v3" else "citequest-school-document-workflow/3-full14",
            "school_review_context_preflight": state["context_preflight"],
            "reviewed_plan_sha256": digest(state["upstream"]["plan"]),
            "school_profile_sha256": digest(self.profile.model_dump(mode="json")),
            "document_content_sha256": digest(self.ordered(state)),
            "source_aliases": state["aliases"],
        }
        document = SchoolDocument(
            profile=self.profile,
            brief_sha256=state["intake"]["brief_sha256"],
            reviewed_plan_sha256=digest(state["upstream"]["plan"]),
            title=state["intake"]["user_confirmed"]["topic"],
            sections=[SchoolSection.model_validate(item, strict=True) for item in self.ordered(state)],
            section_specs=[SectionSpec.model_validate(item, strict=True) for item in state["specs"]],
            source_aliases=state["aliases"],
            section_reviews=[
                SectionReview.model_validate(state["reviews"][item["id"]], strict=True) for item in state["specs"]
            ],
            cross_review=CrossReview.model_validate(state["cross_review"], strict=True),
            request_coverage=RequestCoverage.model_validate(state["request_coverage"], strict=True),
            reference_admission=ReferenceAdmission.model_validate(state["reference_admission"], strict=True),
            context_manifest=context,
        )
        text = (
            "# "
            + document.title
            + "\n\n"
            + "\n\n".join(
                "## " + spec.title + "\n\n" + section.body()
                for spec, section in zip(document.section_specs, document.sections, strict=True)
            )
        )
        return {
            "next": END,
            "result": {
                **state["upstream"],
                "status": "school_document_ready",
                "school_format_ready": True,
                "submission_ready": False,
                "publication_status": "validated_ai_review",
                "completion_status": "school_draft_ready_export_not_yet_verified",
                "text": text,
                "school_document": document.model_dump(mode="json"),
                "context_manifest": context,
                "quality": {
                    "format": "validated",
                    "length": "validated",
                    "literal_sources": "validated",
                    "semantic": "AI_review_not_human_calibrated",
                    "export": "not_run",
                },
                **({"document_notice": "依据已选上交2024级计算机硕士公开模板生成的研究计划草稿。8000–12000汉字为本项目演示规模；不代表学校通用现行要求、学校审批或可直接提交。",
                    "export_options": {"formats": ["docx", "html", "pdf"],
                        "create_endpoint": "/api/agent/tasks/" + str(self.host.execution.task_id) + "/exports",
                        "processing": "existing_authorized_export_queue", "artifacts_generated": False}}
                    if self.task()["profile"].get("profile_id") == "research-v5" else {}),
            },
        }

    def run(self, upstream: dict, config: dict | None = None) -> dict:
        """Use within the parent node so LangGraph inherits its Saver namespace."""
        return self.workflow.invoke({"upstream": upstream}, config, durability="sync")
