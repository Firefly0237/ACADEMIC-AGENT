"""Exact completed research drafts with live dependency authorization.

Private completion events and immutable intake revisions establish identity.
They never substitute for current source, memory, Skill or project authority.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from psycopg.types.json import Jsonb

from citequest.research_v3.brief import BriefDocument
from citequest.research_v3.school_contracts import SchoolDocument
from citequest.research_v3.school_export import validate_export_document
from citequest.v2.domain import Actor

from .contracts import AgentError, ResearchBrief, ReviewSource, ReviewReportSource, RevisionRequest, canonical, digest
from .database import TaskService, owned_task
from .research_runtime import load_document, load_document_revision, planning_input

if TYPE_CHECKING:
    from .database import Execution
    from .memory import MemoryService


@dataclass(frozen=True)
class HistoricalResearchSource:
    task: dict
    result: dict
    brief: BriefDocument
    binding: dict

    @property
    def binding_brief_sha256(self) -> str:
        return self.task["brief"]["research_brief"]["sha256"]


@dataclass(frozen=True)
class HistoricalReviewReport:
    task: dict
    result: dict
    report: dict
    binding: dict


@dataclass(frozen=True)
class RevisionInputs:
    source: HistoricalResearchSource
    report: dict
    report_binding: dict
    selected_findings: list[dict]
    target_section_ids: list[str]


def _source(value: ReviewSource | dict) -> ReviewSource:
    return value if isinstance(value, ReviewSource) else ReviewSource.model_validate_json(canonical(value), strict=True)


def source_ref_ids(result: dict) -> list[str]:
    """Only references bound to this exact published draft may be inherited."""
    try:
        manifest = result["context_manifest"]
        entries = manifest["evidence_refs"]
        refs = [item["ref_id"] for item in entries]
        ledger = result["evidence_ledger"]
        by_ref = {item["ref_id"]: item for item in ledger}
        if (not refs or len(refs) > 64 or len(set(refs)) != len(refs)
                or len(by_ref) != len(ledger) or set(refs) != set(by_ref)):
            raise ValueError("source evidence identity")
        for item in entries:
            if any(item.get(key) != by_ref[item["ref_id"]].get(key)
                   for key in ("source_sha256", "unit_text_sha256", "source_ref")):
                raise ValueError("source evidence pin")
        aliases = result["school_document"]["source_aliases"]
        if any(not values or not set(values) <= set(refs) for values in aliases.values()):
            raise ValueError("bibliography evidence scope")
        if any(item["ref_id"] not in by_ref for item in result.get("citations", [])):
            raise ValueError("citation evidence scope")
        return sorted(refs)
    except (KeyError, TypeError, ValueError):
        raise AgentError("research_source_evidence_binding_invalid", 409) from None


def _snapshot(db, owner: UUID, source_task_id: UUID, command_seq: int,
              *, project: UUID | None = None, expected: ReviewSource | None = None) -> HistoricalResearchSource:
    task = owned_task(db, owner, source_task_id)
    if task["kind"] != "research" or (project is not None and task["project_id"] != project):
        raise AgentError("research_source_project_or_kind_mismatch", 409)
    if digest(task["profile"]) != task["profile_hash"]:
        raise AgentError("research_source_profile_integrity_failure", 409)
    command = db.execute("""SELECT id,seq,kind,payload,state FROM cq_app.agent_commands
        WHERE task_id=%s AND owner_id=%s AND seq=%s AND kind<>'cancel'""",
        (source_task_id, owner, command_seq)).fetchone()
    if command is None or command["state"] != "done":
        raise AgentError("research_source_turn_not_completed", 409)
    records = db.execute("""SELECT seq,payload FROM cq_app.agent_events WHERE task_id=%s
        AND event_type='conversation_turn_completed' AND payload->>'command_id'=%s
        ORDER BY seq LIMIT 2""", (source_task_id, str(command["id"]))).fetchall()
    if len(records) != 1:
        raise AgentError("research_source_completion_missing_or_ambiguous", 409)
    event, record = records[0], records[0]["payload"]
    try:
        raw_brief, result = record["brief"], record["answer"]
        question = command["payload"].get("brief", {}).get("question", "") if command["kind"] == "start" else command["payload"].get("message", "")
        if (record["version"] != "citequest-conversation-turn/1"
                or record["command_id"] != str(command["id"]) or record["command_seq"] != command_seq
                or record["state"] != "succeeded" or record.get("error_code") is not None
                or raw_brief["question"] != question or not isinstance(result, dict)
                or result.get("status") != "school_document_ready"
                or result.get("publication_status") != "validated_ai_review"):
            raise ValueError("completion binding")
        brief = ResearchBrief.model_validate_json(canonical(raw_brief), strict=True)
        if brief.kind != "research" or brief.research_brief is None:
            raise ValueError("research intake identity")
        document = SchoolDocument.model_validate_json(canonical(result["school_document"]), strict=True)
        validate_export_document(document)
        manifest = result["context_manifest"]
        if (document.context_manifest != manifest or document.brief_sha256 != brief.research_brief.sha256
                or manifest.get("graph_version") != task["graph_version"]
                or manifest.get("profile_hash") != task["profile_hash"]
                or manifest.get("brief_id") != str(brief.research_brief.id)
                or manifest.get("brief_revision") != brief.research_brief.revision
                or manifest.get("brief_sha256") != brief.research_brief.sha256
                or manifest.get("source_scope") != raw_brief.get("selected_papers", [])
                or manifest.get("document_content_sha256") != document.content_hash()
                or manifest.get("reviewed_plan_sha256") != digest(result["plan"])
                or document.reviewed_plan_sha256 != digest(result["plan"])):
            raise ValueError("school document context")
        binding = ReviewSource(source_task_id=source_task_id, source_command_id=command["id"],
            command_seq=command_seq, event_seq=event["seq"], task_revision=record["brief_revision"],
            document_sha256=digest(result["school_document"]), context_sha256=digest(manifest))
        if expected is not None and binding != expected:
            raise ValueError("source version changed")
    except (KeyError, TypeError, ValueError):
        raise AgentError("research_source_completion_binding_invalid", 409) from None
    archived, sha256 = load_document_revision(db, owner, task["project_id"], brief.research_brief)
    if (archived.selected_sources != brief.selected_papers or archived.draft_source_ids != brief.research_draft_files
            or result.get("research_intake") != planning_input(archived, sha256)):
        raise AgentError("research_source_intake_binding_invalid", 409)
    # A corrected or removed confirmation must not authenticate an old citation.
    head, _ = load_document(db, owner, task["project_id"], archived.id)
    for item in (document.reference_admission.metadata_sources or {}).values():
        if item.provenance == "user_confirmed":
            identifier = item.source_ref["id"]
            current = (head.material_bibliography or {}).get(identifier)
            old = (archived.material_bibliography or {}).get(identifier)
            if (old is None or current is None or old != current
                    or digest(old.model_dump(mode="json")) != item.bibliography_sha256
                    or old.file_sha256 != item.file_sha256 or old.representation_id != item.representation_id
                    or identifier in {str(value) for value in head.draft_source_ids}):
                raise AgentError("research_source_bibliography_changed", 409)
    source_ref_ids(result)
    historical = {**task, "brief": raw_brief, "revision": record["brief_revision"],
                  "state": record["state"], "command_seq": command_seq, "result": result}
    return HistoricalResearchSource(historical, result, archived, binding.model_dump(mode="json"))


def review_admit(db, owner: UUID, project: UUID, brief: ResearchBrief) -> None:
    """DB-only create guard; API and worker separately check live dependencies."""
    if brief.kind != "research" or brief.research_action != "review" or brief.review_source is None:
        raise AgentError("research_review_source_required", 422)
    source = _source(brief.review_source)
    snapshot = _snapshot(db, owner, source.source_task_id, source.command_seq, project=project, expected=source)
    original = ResearchBrief.model_validate_json(canonical(snapshot.task["brief"]), strict=True)
    if (brief.research_brief != original.research_brief or brief.selected_papers != original.selected_papers
            or brief.research_draft_files != original.research_draft_files
            or brief.selected_skills != original.selected_skills):
        raise AgentError("research_review_scope_must_match_source", 409)


def _report_source(value: ReviewReportSource | dict) -> ReviewReportSource:
    return value if isinstance(value, ReviewReportSource) else ReviewReportSource.model_validate_json(canonical(value), strict=True)


def _report_snapshot(db, owner: UUID, task_id: UUID, command_seq: int, *,
                     project: UUID | None = None, expected: ReviewReportSource | None = None) -> HistoricalReviewReport:
    from citequest.research_v3.review_contracts import ReviewReport, validate_aggregation, validate_role_report
    task = owned_task(db, owner, task_id)
    if (task["kind"] != "research" or task["profile"].get("profile_id") != "research-review-v1"
            or digest(task["profile"]) != task["profile_hash"]
            or (project is not None and task["project_id"] != project)):
        raise AgentError("research_review_report_task_identity_invalid", 409)
    command = db.execute("""SELECT id,seq,kind,payload,state FROM cq_app.agent_commands
        WHERE task_id=%s AND owner_id=%s AND seq=%s AND kind<>'cancel'""", (task_id, owner, command_seq)).fetchone()
    if command is None or command["state"] != "done":
        raise AgentError("research_review_report_turn_not_completed", 409)
    records = db.execute("""SELECT seq,payload FROM cq_app.agent_events WHERE task_id=%s
        AND event_type='conversation_turn_completed' AND payload->>'command_id'=%s
        ORDER BY seq LIMIT 2""", (task_id, str(command["id"]))).fetchall()
    if len(records) != 1:
        raise AgentError("research_review_report_completion_missing_or_ambiguous", 409)
    event, record = records[0], records[0]["payload"]
    try:
        brief = ResearchBrief.model_validate_json(canonical(record["brief"]), strict=True)
        result = record["answer"]
        question = command["payload"].get("brief", {}).get("question", "") if command["kind"] == "start" else command["payload"].get("message", "")
        if (record["version"] != "citequest-conversation-turn/1" or record["command_id"] != str(command["id"])
                or record["command_seq"] != command_seq or record["state"] != "succeeded"
                or record.get("error_code") is not None or brief.question != question
                or brief.research_action != "review" or brief.review_source is None
                or not isinstance(result, dict) or result.get("status") != "research_review_ready"
                or result.get("publication_status") != "validated_ai_review"):
            raise ValueError("review completion")
        report = ReviewReport.model_validate_json(canonical(result["review_report"]), strict=True)
        report_value = report.model_dump(mode="json")
        manifest = result["context_manifest"]
        binding = ReviewReportSource(task_id=task_id, command_id=command["id"], command_seq=command_seq,
            event_seq=event["seq"], task_revision=record["brief_revision"],
            report_sha256=digest(result["review_report"]), context_sha256=digest(manifest))
        source_binding = brief.review_source.model_dump(mode="json")
        if (expected is not None and binding != expected) or (
                manifest.get("review_source") != source_binding
                or manifest.get("graph_version") != task["graph_version"]
                or manifest.get("profile_hash") != task["profile_hash"]
                or report.source_document_sha256 != brief.review_source.document_sha256
                or report.source_binding_sha256 != digest(source_binding)
                or result.get("review_report_sha256") != binding.report_sha256
                or manifest.get("review_report_sha256") != binding.report_sha256
                or digest(report_value) != binding.report_sha256):
            raise ValueError("report version")
    except (KeyError, TypeError, ValueError):
        raise AgentError("research_review_report_binding_invalid", 409) from None
    source = _snapshot(db, owner, brief.review_source.source_task_id, brief.review_source.command_seq,
        project=task["project_id"], expected=brief.review_source)
    if (brief.research_brief != ResearchBrief.model_validate_json(canonical(source.task["brief"]), strict=True).research_brief
            or any(manifest.get(key) != source.result["context_manifest"].get(key)
                   for key in ("brief_id", "brief_revision", "brief_sha256", "source_scope"))
            or {item["ref_id"]: item for item in manifest.get("evidence_refs", [])}
                != {item["ref_id"]: item for item in source.result["context_manifest"]["evidence_refs"]}
            or len(manifest.get("evidence_refs", [])) != len(source.result["context_manifest"]["evidence_refs"])):
        raise AgentError("research_review_report_context_changed", 409)
    document = source.result["school_document"]
    hashes = {item["section_id"]: digest(item) for item in document["sections"]}
    texts = {item["section_id"]: "\n\n".join(block["text"] for block in item["blocks"]) for item in document["sections"]}
    all_claims = {identifier: claim for ledger in source.result["literature_ledger"]
        for identifier, claim in zip(ledger["claim_ids"], ledger["answer"]["claims"], strict=True)}
    for role in report.role_reports:
        sections = document["sections"]
        if role.role == "methods":
            relevant = {item["id"] for item in document["section_specs"] if item["group"] in {"methods", "objectives"}}
            sections = [item for item in sections if item["section_id"] in relevant]
        selected = ({claim for item in sections for block in item["blocks"] for claim in block["claim_ids"]}
            if role.role != "structure" else set())
        claim_quotes, quotes = {}, {}
        for identifier in selected:
            if identifier not in all_claims:
                raise AgentError("research_review_report_claim_missing", 409)
            by_ref = {}
            for edge in all_claims[identifier]["edges"]:
                by_ref.setdefault(edge["ref_id"], []).append(edge["quote"])
                quotes.setdefault(edge["ref_id"], []).append(edge["quote"])
            claim_quotes[identifier] = by_ref
        validate_role_report(role, role=role.role, document_sha256=source.binding["document_sha256"],
            section_sha256s=hashes, section_texts=texts, claim_ids=selected, quotes=quotes, claim_quotes=claim_quotes)
    if report.section_sha256s != hashes:
        raise AgentError("research_review_report_section_versions_changed", 409)
    validate_aggregation(report.aggregation, report.role_reports,
        document_sha256=source.binding["document_sha256"], section_sha256s=hashes)
    historical = {**task, "brief": record["brief"], "revision": record["brief_revision"],
        "state": record["state"], "command_seq": command_seq, "result": result}
    return HistoricalReviewReport(historical, result, report_value, binding.model_dump(mode="json"))


def _revision_inputs(source: HistoricalResearchSource, report: HistoricalReviewReport,
                     request: RevisionRequest) -> RevisionInputs:
    from citequest.research_v3.revision_contracts import select_revision_findings
    if report.result["context_manifest"].get("review_source") != source.binding:
        raise AgentError("research_revision_report_for_another_manuscript", 409)
    order = [item["section_id"] for item in source.result["school_document"]["sections"]]
    selected, target_sections = select_revision_findings(report.report, request.finding_keys, order)
    return RevisionInputs(source, report.report, report.binding, selected, target_sections)


def revision_admit(db, owner: UUID, project: UUID, brief: ResearchBrief) -> None:
    if (brief.kind != "research" or brief.research_action != "revise"
            or brief.review_source is None or brief.revision_request is None):
        raise AgentError("research_revision_source_and_findings_required", 422)
    binding = brief.review_source
    source = _snapshot(db, owner, binding.source_task_id, binding.command_seq, project=project, expected=binding)
    report_binding = brief.revision_request.report
    report = _report_snapshot(db, owner, report_binding.task_id, report_binding.command_seq,
        project=project, expected=report_binding)
    _revision_inputs(source, report, brief.revision_request)
    original = ResearchBrief.model_validate_json(canonical(source.task["brief"]), strict=True)
    if (brief.research_brief != original.research_brief or brief.selected_papers != original.selected_papers
            or brief.research_draft_files != original.research_draft_files or brief.selected_skills != original.selected_skills):
        raise AgentError("research_revision_scope_must_match_source", 409)


class HistoricalResearchReader:
    def __init__(self, service: TaskService, memory: MemoryService,
                 authorize_evidence: Callable[[Actor, UUID, list[str]], None]) -> None:
        self.service, self.memory, self.authorize_evidence = service, memory, authorize_evidence

    def _authorize(self, actor: Actor, source: HistoricalResearchSource) -> None:
        from .result_projection import authorize_result_dependencies
        authorize_result_dependencies(self.service, self.memory, self.authorize_evidence,
            actor, UUID(source.binding["source_task_id"]), source.task, source.result)

    def bind(self, actor: Actor, source_task_id: UUID, command_seq: int, *,
             expected_document_sha256: str | None = None) -> dict:
        source = self.read_turn(actor, source_task_id, command_seq)
        if expected_document_sha256 is not None and source.binding["document_sha256"] != expected_document_sha256:
            raise AgentError("research_source_document_changed", 409)
        return source.binding

    def read_turn(self, actor: Actor, task_id: UUID, command_seq: int) -> HistoricalResearchSource:
        if type(command_seq) is not int or not 1 <= command_seq <= 9007199254740991:
            raise AgentError("invalid_research_source_command_seq", 400)
        with self.service.database.connect() as db:
            source = _snapshot(db, actor.owner_id, task_id, command_seq)
        self._authorize(actor, source)
        return source

    def read(self, actor: Actor, source: ReviewSource | dict) -> HistoricalResearchSource:
        binding = _source(source)
        with self.service.database.connect() as db:
            snapshot = _snapshot(db, actor.owner_id, binding.source_task_id, binding.command_seq, expected=binding)
        self._authorize(actor, snapshot)
        # Source IO is outside SQL locks. Recheck exact identity after that IO.
        with self.service.database.connect() as db:
            current = _snapshot(db, actor.owner_id, binding.source_task_id, binding.command_seq, expected=binding)
        if canonical(current.result) != canonical(snapshot.result):
            raise AgentError("research_source_changed_during_authorization", 409)
        return snapshot

    def prepare_child(self, execution: Execution, source: ReviewSource | dict) -> HistoricalResearchSource:
        initial = execution.snapshot()
        actor = Actor(initial["owner_id"])
        snapshot = self.read(actor, source)
        self._inherit(execution, snapshot, "research-review-v1", review_admit)
        return snapshot

    def _inherit(self, execution: Execution, snapshot: HistoricalResearchSource,
                 profile_id: str, admit: Callable) -> None:
        actor = Actor(execution.snapshot()["owner_id"])
        refs = source_ref_ids(snapshot.result)
        with execution.database.connect() as db:
            child = execution.lock_task(db)
            if (child["profile"].get("profile_id") != profile_id or child["state"] != "running"
                    or child["project_id"] != snapshot.task["project_id"]
                    or child["brief"].get("review_source") != snapshot.binding):
                raise AgentError("research_review_child_identity_changed", 409)
            admit(db, actor.owner_id, child["project_id"], ResearchBrief.model_validate_json(canonical(child["brief"]), strict=True))
            from .catalog_scope import CatalogPins
            parent = owned_task(db, actor.owner_id, UUID(snapshot.binding["source_task_id"]))
            CatalogPins._rows(db, parent)
            parent_pins = db.execute("SELECT * FROM cq_app.agent_catalog_pins WHERE task_id=%s",
                (parent["id"],)).fetchall()
            needed_catalogs = {"application_papers" if item["source_ref"]["kind"] == "native_representation"
                else "standard_metadata" for item in snapshot.result["context_manifest"]["evidence_refs"]
                if item.get("source_ref", {}).get("kind") in {"native_representation", "metadata_projection"}}
            if not needed_catalogs <= {pin["catalog"] for pin in parent_pins}:
                raise AgentError("research_review_parent_catalog_pin_required", 409)
            for pin in parent_pins:
                db.execute("""INSERT INTO cq_app.agent_catalog_pins
                    (task_id,catalog,owner_id,generation_id,epoch,binding,binding_canonical,binding_sha256)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                    (execution.task_id, pin["catalog"], actor.owner_id, pin["generation_id"], pin["epoch"],
                     Jsonb(pin["binding"]), pin["binding_canonical"], pin["binding_sha256"]))
                copied = db.execute("SELECT * FROM cq_app.agent_catalog_pins WHERE task_id=%s AND catalog=%s",
                    (execution.task_id, pin["catalog"])).fetchone()
                if copied is None or any(copied[key] != pin[key] for key in
                        ("owner_id", "generation_id", "epoch", "binding", "binding_canonical", "binding_sha256")):
                    raise AgentError("research_review_child_catalog_pin_conflict", 409)
            child_catalogs = db.execute("SELECT catalog FROM cq_app.agent_catalog_pins WHERE task_id=%s",
                (execution.task_id,)).fetchall()
            if {row["catalog"] for row in child_catalogs} != {pin["catalog"] for pin in parent_pins}:
                raise AgentError("research_review_child_catalog_scope_conflict", 409)
            records = db.execute("SELECT * FROM cq_app.agent_evidence_refs WHERE task_id=%s AND ref_id=ANY(%s)",
                (parent["id"], refs)).fetchall()
            if {row["ref_id"] for row in records} != set(refs):
                raise AgentError("research_review_parent_evidence_missing", 409)
            expected_refs = {item["ref_id"]: item for item in snapshot.result["context_manifest"]["evidence_refs"]}
            for row in records:
                expected_ref = expected_refs[row["ref_id"]]
                if (row["source_hash"] != row["payload"].get("source_sha256")
                        or row["source_ref"] != row["payload"].get("source_ref")
                        or any(row["payload"].get(key) != expected_ref.get(key)
                               for key in ("source_sha256", "unit_text_sha256", "source_ref"))):
                    raise AgentError("research_review_parent_evidence_pin_changed", 409)
                db.execute("""INSERT INTO cq_app.agent_evidence_refs(task_id,ref_id,source_hash,source_ref,payload)
                    VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                    (execution.task_id, row["ref_id"], row["source_hash"], Jsonb(row["source_ref"]), Jsonb(row["payload"])))
                copied = db.execute("SELECT source_hash,source_ref,payload FROM cq_app.agent_evidence_refs WHERE task_id=%s AND ref_id=%s",
                    (execution.task_id, row["ref_id"])).fetchone()
                if copied is None or any(copied[key] != row[key] for key in ("source_hash", "source_ref", "payload")):
                    raise AgentError("research_review_child_evidence_conflict", 409)
            child_refs = db.execute("SELECT ref_id FROM cq_app.agent_evidence_refs WHERE task_id=%s",
                (execution.task_id,)).fetchall()
            if {row["ref_id"] for row in child_refs} != set(refs):
                raise AgentError("research_review_child_evidence_scope_conflict", 409)

    def bind_report(self, actor: Actor, task_id: UUID, command_seq: int) -> dict:
        if type(command_seq) is not int or not 1 <= command_seq <= 9007199254740991:
            raise AgentError("invalid_research_report_command_seq", 400)
        with self.service.database.connect() as db:
            report = _report_snapshot(db, actor.owner_id, task_id, command_seq)
        return self.read_report(actor, report.binding).binding

    def read_report(self, actor: Actor, source: ReviewReportSource | dict) -> HistoricalReviewReport:
        binding = _report_source(source)
        with self.service.database.connect() as db:
            report = _report_snapshot(db, actor.owner_id, binding.task_id, binding.command_seq, expected=binding)
        self.read(actor, report.result["context_manifest"]["review_source"])
        from .result_projection import authorize_result_dependencies
        authorize_result_dependencies(self.service, self.memory, self.authorize_evidence,
            actor, binding.task_id, report.task, report.result)
        with self.service.database.connect() as db:
            after = _report_snapshot(db, actor.owner_id, binding.task_id, binding.command_seq, expected=binding)
        if canonical(after.result) != canonical(report.result):
            raise AgentError("research_review_report_changed_during_authorization", 409)
        return report

    def read_revision(self, actor: Actor, manuscript_source: ReviewSource | dict,
                      revision_request: RevisionRequest | dict) -> RevisionInputs:
        request = (revision_request if isinstance(revision_request, RevisionRequest)
            else RevisionRequest.model_validate_json(canonical(revision_request), strict=True))
        source = self.read(actor, manuscript_source)
        report = self.read_report(actor, request.report)
        if report.task["project_id"] != source.task["project_id"]:
            raise AgentError("research_revision_report_project_mismatch", 409)
        return _revision_inputs(source, report, request)

    def prepare_revision_child(self, execution: Execution, manuscript_source: ReviewSource | dict,
                               revision_request: RevisionRequest | dict) -> RevisionInputs:
        actor = Actor(execution.snapshot()["owner_id"])
        inputs = self.read_revision(actor, manuscript_source, revision_request)
        self._inherit(execution, inputs.source, "research-revision-v1", revision_admit)
        return inputs

    def versions(self, actor: Actor, task_id: UUID, *, after_command_seq: int = 0, limit: int = 20) -> dict:
        if (type(after_command_seq) is not int or after_command_seq < 0
                or type(limit) is not int or not 1 <= limit <= 20):
            raise AgentError("invalid_research_version_cursor_or_limit", 400)
        with self.service.database.connect() as db:
            owned_task(db, actor.owner_id, task_id)
            commands = db.execute("""SELECT c.id,c.seq,c.state,c.created_at FROM cq_app.agent_commands c
                WHERE c.task_id=%s AND c.owner_id=%s AND c.kind<>'cancel' AND c.state='done' AND c.seq>%s
                AND EXISTS(SELECT 1 FROM cq_app.agent_events e WHERE e.task_id=%s
                    AND e.event_type='conversation_turn_completed' AND e.payload->>'command_id'=c.id::text
                    AND e.payload->'answer'->>'status'='school_document_ready')
                ORDER BY c.seq LIMIT %s""", (task_id, actor.owner_id, after_command_seq, task_id, limit + 1)).fetchall()
        items = []
        for command in commands[:limit]:
            row = {"command_seq": command["seq"], "created_at": command["created_at"], "state": command["state"],
                "title": None, "document_sha256": None, "source_binding": None, "availability": "source_unavailable"}
            try:
                with self.service.database.connect() as db:
                    source = _snapshot(db, actor.owner_id, task_id, command["seq"])
                row.update(title=source.result["school_document"]["title"],
                    document_sha256=source.binding["document_sha256"])
                self._authorize(actor, source)
                row.update(source_binding=source.binding, availability="available")
            except (AgentError, ValueError, TypeError, KeyError):
                pass
            items.append(row)
        return {"items": items, "next_after_command_seq": items[-1]["command_seq"]
            if len(commands) > limit and items else None}
