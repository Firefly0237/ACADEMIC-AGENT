"""Business authority and physical-session ownership, independent of graph history."""
from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .contracts import AgentError, DIALOGUE_REFERENCE_PROFILE_IDS, PROFILES, RuntimeLimits, TaskCommand, TaskCreate, canonical, digest, redact
from .capabilities import free_read_attempt
from . import dispatch as dispatch_control
from . import catalog_scope
from .product_profiles import TaskProfilePolicy, is_profile_implemented
from . import development_campaign as campaign_control

if TYPE_CHECKING:
    from .development_dispatch import DevelopmentDispatchPolicy
    from .development_dispatch_v2 import DevelopmentDispatchPolicyV2


@dataclass(frozen=True)
class Database:
    business_dsn: str
    checkpoint_dsn: str
    memory_dsn: str
    dispatch_policy: dispatch_control.DispatchPolicy | None = None
    worker_lane: str | None = None
    catalog_scope_policy: catalog_scope.CatalogScopePolicy | None = None
    development_dispatch_policy: DevelopmentDispatchPolicy | DevelopmentDispatchPolicyV2 | None = None
    development_campaign_policy: campaign_control.DevelopmentCampaign | None = None

    def __post_init__(self) -> None:
        if self.development_dispatch_policy is not None and self.development_campaign_policy is not None:
            raise AgentError("development_policy_modes_are_mutually_exclusive")

    @classmethod
    def from_files(cls, business: Path, checkpoint: Path, memory: Path, *,
                   dispatch_policy_file: Path | None = None, worker_lane: str | None = None,
                   catalog_scope_policy: catalog_scope.CatalogScopePolicy | None = None,
                   development_dispatch_policy: DevelopmentDispatchPolicy | DevelopmentDispatchPolicyV2 | None = None,
                   development_campaign_policy: campaign_control.DevelopmentCampaign | None = None) -> Database:
        return cls(*(p.read_text(encoding="utf-8").strip().replace("postgresql+psycopg://", "postgresql://", 1)
                     for p in (business, checkpoint, memory)),
                   dispatch_policy=dispatch_control.DispatchPolicy.model_validate_json(dispatch_policy_file.read_bytes())
                        if dispatch_policy_file else None, worker_lane=worker_lane, catalog_scope_policy=catalog_scope_policy,
                   development_dispatch_policy=development_dispatch_policy,
                   development_campaign_policy=development_campaign_policy)

    def connect(self, role: str = "business", *, autocommit: bool = False) -> psycopg.Connection:
        dsn = {"business": self.business_dsn, "checkpoint": self.checkpoint_dsn, "memory": self.memory_dsn}[role]
        options = {"checkpoint": "-c search_path=cq_agent_checkpoint", "memory": "-c search_path=cq_agent_memory"}.get(role, "")
        return psycopg.connect(dsn, row_factory=dict_row, autocommit=autocommit,
            prepare_threshold=0, connect_timeout=5, options=options)


def event(db: psycopg.Connection, task_id: UUID, kind: str, payload: dict) -> None:
    seq = db.execute("UPDATE cq_app.agent_tasks SET event_seq=event_seq+1,updated_at=GREATEST(updated_at,clock_timestamp()) WHERE id=%s RETURNING event_seq", (task_id,)).fetchone()["event_seq"]
    db.execute("INSERT INTO cq_app.agent_events(task_id,seq,event_type,payload) VALUES(%s,%s,%s,%s)",
        (task_id, seq, kind, Jsonb(payload)))


def public_event(row: dict) -> dict:
    """Private turn snapshots support authorized readers, never the public SSE.

    Keep sequence/cursor metadata so removing private body does not create gaps.
    Every future property of this private event is excluded by default.
    """
    if row["event_type"] != "conversation_turn_completed":
        return row
    allowed = ("version","command_seq","brief_revision","state","error_code")
    return {**row,"payload":{key:row["payload"][key] for key in allowed if key in row["payload"]}}


def owned_task(db: psycopg.Connection, owner: UUID, task_id: UUID, *, lock: bool = False) -> dict:
    row = db.execute("SELECT t.* FROM cq_app.agent_tasks t JOIN cq_app.projects p ON (p.id,p.owner_id)=(t.project_id,t.owner_id) JOIN cq_app.owners o ON o.id=t.owner_id WHERE t.id=%s AND t.owner_id=%s" + (" FOR UPDATE OF t" if lock else ""), (task_id, owner)).fetchone()
    if row is None:
        raise AgentError("agent_task_not_found", 404)
    return row


def wallet_locks(db: psycopg.Connection, owner: UUID, project: UUID) -> None:
    # A common short transaction lock prevents concurrent owners/projects spending one campaign twice.
    # Ordering is campaign -> owner -> project -> Task; no network call holds these locks.
    # These are transaction locks, not the physical Saver session ownership lock.
    for value in ("agent-campaign-wallet", "agent-owner-wallet:" + str(owner), "agent-project-wallet:" + str(project)):
        key = int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big", signed=True)
        db.execute("SELECT pg_advisory_xact_lock(%s)", (key,))


def task_wallet_locks(db: psycopg.Connection, task_id: UUID) -> dict:
    identity = db.execute("SELECT owner_id,project_id FROM cq_app.agent_tasks WHERE id=%s", (task_id,)).fetchone()
    if identity is None:
        raise AgentError("agent_task_not_found", 404)
    wallet_locks(db, identity["owner_id"], identity["project_id"])
    return identity


def wallet_admit(db: psycopg.Connection, limits: RuntimeLimits, owner: UUID, project: UUID,
                 *, reservation: int = 0, new_task: bool = False, activate_task: bool = False,
                 development_active_policy: DevelopmentDispatchPolicyV2 | None = None,
                 development_active_target: dict | None = None,
                 development_active_operation_id: UUID | None = None,
                 development_active_command_kind: str | None = None,
                 development_campaign: campaign_control.DevelopmentCampaign | None = None) -> None:
    owner_state = db.execute("""SELECT count(*) AS tasks,
        count(*) FILTER(WHERE state IN ('queued','running','reconciling')) AS active,
        coalesce(sum(known_ncny+reserved_ncny),0) AS cash FROM cq_app.agent_tasks WHERE owner_id=%s""", (owner,)).fetchone()
    if new_task and owner_state["tasks"] >= limits.owner_total_tasks:
        raise AgentError("owner_lifetime_task_limit")
    active = owner_state["active"]
    if development_campaign is not None:
        if development_active_policy is not None or not (new_task or activate_task):
            raise AgentError("development_campaign_active_projection_not_admissible")
        if dispatch_control.active_policy(db, None) is not None:
            raise AgentError("development_campaign_FIFO_binding_not_adopted", 503)
        active = development_campaign.active_tasks(db, limits, owner, project)
    if development_active_policy is not None:
        if (not (new_task or activate_task) or development_active_target is None
                or development_active_operation_id is None
                or dispatch_control.active_policy(db, None) is not None):
            raise AgentError("development_dispatch_active_projection_not_admissible")
        from .development_dispatch_v2 import project_active
        active = project_active(db, limits, owner, project, development_active_target,
            development_active_operation_id, development_active_command_kind, development_active_policy)
    if (new_task or activate_task) and active >= limits.owner_active_tasks:
        raise AgentError("owner_active_task_limit", 429)
    if limits.mode == "bounded":
        if limits.campaign_cash_ncny is not None:
            campaign_cash = db.execute(
                "SELECT coalesce(sum(known_ncny+reserved_ncny),0) AS cash FROM cq_app.agent_tasks"
            ).fetchone()["cash"]
            if int(campaign_cash) + limits.campaign_prior_reserved_ncny + reservation > limits.campaign_cash_ncny:
                raise AgentError("development_campaign_cash_limit", 429)
        if int(owner_state["cash"]) + reservation > limits.owner_cash_ncny:
            raise AgentError("owner_lifetime_cash_limit", 429)
        project_cash = db.execute("SELECT coalesce(sum(known_ncny+reserved_ncny),0) AS cash FROM cq_app.agent_tasks WHERE project_id=%s", (project,)).fetchone()["cash"]
        if int(project_cash) + reservation > limits.project_cash_ncny:
            raise AgentError("project_lifetime_cash_limit", 429)


def same_limits_policy(saved: object, limits: RuntimeLimits) -> bool:
    """Normalize absent optional fields without rewriting an immutable historical profile."""
    try:
        return RuntimeLimits.model_validate(saved, strict=True).model_dump() == limits.model_dump()
    except (TypeError, ValueError):
        return False


class TaskService:
    def __init__(self, database: Database, limits: RuntimeLimits, *,
                 new_task_profile_policy: TaskProfilePolicy | None = None) -> None:
        self.database, self.limits = database, RuntimeLimits.model_validate(limits.model_dump(), strict=True)
        self.new_task_profile_policy = (TaskProfilePolicy.model_validate_json(new_task_profile_policy.model_dump_json())
                                        if new_task_profile_policy is not None else None)
        campaign = getattr(database, "development_campaign_policy", None)
        if campaign is not None:
            if self.new_task_profile_policy is None:
                raise AgentError("development_campaign_profile_policy_required", 503)
            campaign.require_configuration(self.limits, database.catalog_scope_policy, self.new_task_profile_policy)

    def create(self, owner: UUID, request: TaskCreate) -> dict:
        payload = request.model_dump(mode="json")
        request_hash = digest(payload)
        development_policy = getattr(self.database, "development_dispatch_policy", None)
        active_policy = development_policy if getattr(development_policy, "version", None) == "citequest-development-named-unknown-dispatch/2" else None
        campaign = getattr(self.database, "development_campaign_policy", None)
        profile = {**PROFILES[request.profile_id].model_dump(), "aggregate_policy": self.limits.model_dump()}
        if self.new_task_profile_policy is not None:
            profile.update(self.new_task_profile_policy.runtime_metadata(request.kind, request.profile_id))
        if self.database.catalog_scope_policy is not None:
            profile['catalog_scope_policy'] = self.database.catalog_scope_policy
        if campaign is not None:
            profile[campaign_control.BINDING_KEY] = campaign.task_binding()
        graph_version = catalog_scope.graph_version(request.kind, request.profile_id, self.database.catalog_scope_policy,
            profile.get("review_transport_protocol"), profile.get("qa_execution_protocol"), profile.get("review_api_protocol"), profile.get("structured_output_protocol"), profile.get("answer_revision_protocol"), profile.get("qa_context_protocol"), profile.get("qa_content_protocol"), profile.get("evidence_navigation_protocol"), profile.get("retrieval_rerank_protocol"), profile.get("rerank_config_sha256"), profile.get("qa_retrieval_protocol"))
        preview = ({"id": active_policy.target.task_id, "owner_id": owner, "project_id": request.project_id,
            "operation_id": request.operation_id, "create_hash": request_hash, "kind": request.kind,
            "profile": profile, "profile_hash": digest(profile), "graph_version": graph_version} if active_policy is not None else None)
        with self.database.connect() as db:
            wallet_locks(db, owner, request.project_id)
            if active_policy is not None:
                from .development_dispatch_v2 import require_identity
                require_identity(preview, active_policy)
            if not db.execute("SELECT 1 FROM cq_app.projects WHERE id=%s AND owner_id=%s", (request.project_id, owner)).fetchone():
                raise AgentError("project_not_found", 404)
            existing = db.execute("SELECT * FROM cq_app.agent_tasks WHERE owner_id=%s AND operation_id=%s", (owner, request.operation_id)).fetchone()
            if existing:
                if existing["create_hash"] != request_hash:
                    raise AgentError("idempotency_payload_conflict")
                if active_policy is not None:
                    require_identity(existing, active_policy)
                return self.public(existing, catalog_scope_policy=self.database.catalog_scope_policy,
                    development_campaign_policy=campaign)
            # A replay of a committed original request remains readable after
            # a release changes. Only genuinely new Tasks use the new policy.
            if not is_profile_implemented(request.profile_id):
                raise AgentError("task_profile_implementation_unavailable", 503)
            if request.profile_id in {"qa-v5", "qa-v6", "qa-v7", "qa-v8", "qa-v9"} and self.database.catalog_scope_policy != catalog_scope.POLICY:
                raise AgentError("qa5_requires_task_catalog_snapshot_policy", 503)
            if self.new_task_profile_policy is not None:
                self.new_task_profile_policy.admit(request)
            elif request.profile_id in {"research-review-v1", "research-revision-v1"}:
                raise AgentError("research_auxiliary_action_not_enabled", 503)
            if request.profile_id in {"research-review-v1", "research-revision-v1"}:
                # wallet_locks above serializes admission across clients and
                # refreshes. Original operation replays already returned above.
                report = (Jsonb(request.brief.revision_request.report.model_dump(mode="json"))
                          if request.brief.revision_request is not None else None)
                unresolved = db.execute("""SELECT 1 FROM cq_app.agent_tasks t
                    WHERE t.owner_id=%s AND t.project_id=%s AND t.profile->>'profile_id'=%s
                      AND t.brief->'review_source'=%s::jsonb
                      AND (%s::jsonb IS NULL OR t.brief->'revision_request'->'report'=%s::jsonb)
                      AND (t.state IN ('queued','running','waiting_input','reconciling') OR EXISTS (
                        SELECT 1 FROM cq_app.agent_attempts a WHERE a.task_id=t.id
                          AND a.status IN ('reserved','sent','unknown'))) LIMIT 1""",
                    (owner, request.project_id, request.profile_id,
                     Jsonb(request.brief.review_source.model_dump(mode="json")), report, report)).fetchone()
                if unresolved:
                    raise AgentError("research_source_action_already_pending_or_unknown", 409)
            if request.profile_id in {"qa-v7", "qa-v8", "qa-v9"}:
                from .product_profiles import qa_execution_protocol
                qa_execution_protocol({"kind": request.kind, "profile": profile})
            if campaign is not None:
                campaign.require_owner(owner, request.project_id)
            catalog_scope.require_schema(db, self.database.catalog_scope_policy)
            policy = dispatch_control.active_policy(db, self.database.dispatch_policy)
            for expired in dispatch_control.expire_queued(db, policy):
                event(db, expired["id"], "queue_deadline_expired", {"command_id": str(expired["command_id"]), "error_code": expired["error_code"]})
            dispatch_control.research_admit(db, request.kind, policy)
            if request.profile_id == "research-review-v1":
                from .research_history import review_admit
                review_admit(db, owner, request.project_id, request.brief)
            elif request.profile_id == "research-revision-v1":
                from .research_history import revision_admit
                revision_admit(db, owner, request.project_id, request.brief)
            else:
                from .research_runtime import admit_research
                admit_research(db, owner, request.project_id, request.brief)
            if request.brief.selected_skills:
                from .user_skills import activate_for_task
                activate_for_task(self.database, {"owner_id": owner, "project_id": request.project_id,
                    "brief": request.brief.model_dump(mode="json")})
            task_id, command_id = (active_policy.target.task_id if active_policy is not None else uuid4()), uuid4()
            wallet_admit(db, self.limits, owner, request.project_id, new_task=True,
                development_active_policy=active_policy, development_active_target=preview,
                development_active_operation_id=request.operation_id, development_active_command_kind="start",
                development_campaign=campaign)
            if active_policy is not None and db.execute("SELECT 1 FROM cq_app.agent_tasks WHERE id=%s", (task_id,)).fetchone():
                raise AgentError("development_dispatch_prebound_Task_ID_already_used")
            inserted = db.execute("""INSERT INTO cq_app.agent_tasks(id,owner_id,project_id,operation_id,create_hash,kind,brief,profile,profile_hash,graph_version)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(owner_id,operation_id) DO NOTHING RETURNING *""",
                (task_id, owner, request.project_id, request.operation_id, request_hash, request.kind,
                 Jsonb(request.brief.model_dump(mode="json")), Jsonb(profile), digest(profile), graph_version)).fetchone()
            if inserted is None:
                existing = db.execute("SELECT * FROM cq_app.agent_tasks WHERE owner_id=%s AND operation_id=%s", (owner, request.operation_id)).fetchone()
                if existing["create_hash"] != request_hash:
                    raise AgentError("idempotency_payload_conflict")
                if active_policy is not None:
                    require_identity(existing, active_policy)
                return self.public(existing, catalog_scope_policy=self.database.catalog_scope_policy,
                    development_campaign_policy=campaign)
            db.execute("INSERT INTO cq_app.agent_commands(id,task_id,owner_id,operation_id,seq,kind,payload,payload_hash) VALUES(%s,%s,%s,%s,1,'start',%s,%s)",
                (command_id, task_id, owner, request.operation_id, Jsonb(payload), request_hash))
            accepted_event = {"state": "queued", "revision": 1, "profile_id": request.profile_id}
            if active_policy is not None:
                accepted_event["development_dispatch"] = active_policy.public_binding()
            if campaign is not None:
                accepted_event["development_campaign"] = campaign.public_binding()
            event(db, task_id, "task_accepted", accepted_event)
            return self.public(owned_task(db, owner, task_id), catalog_scope_policy=self.database.catalog_scope_policy,
                development_campaign_policy=campaign)

    def command(self, owner: UUID, task_id: UUID, request: TaskCommand) -> dict:
        payload, request_hash = request.model_dump(mode="json"), digest(request.model_dump(mode="json"))
        development_policy = getattr(self.database, "development_dispatch_policy", None)
        active_policy = development_policy if getattr(development_policy, "version", None) == "citequest-development-named-unknown-dispatch/2" else None
        campaign = getattr(self.database, "development_campaign_policy", None)
        if active_policy is not None:
            active_policy.require_task(task_id)
        with self.database.connect() as db:
            identity = owned_task(db, owner, task_id)
            wallet_locks(db, owner, identity["project_id"])
            task = owned_task(db, owner, task_id, lock=True)
            if active_policy is not None:
                from .development_dispatch_v2 import require_identity
                require_identity(task, active_policy)
            existing = db.execute("SELECT * FROM cq_app.agent_commands WHERE owner_id=%s AND operation_id=%s", (owner, request.operation_id)).fetchone()
            if existing:
                if existing["task_id"] != task_id or existing["payload_hash"] != request_hash:
                    raise AgentError("idempotency_payload_conflict")
                return self.public(task, catalog_scope_policy=self.database.catalog_scope_policy,
                    development_campaign_policy=campaign)
            if campaign is not None and task_id in campaign.historical_task_ids:
                raise AgentError("development_campaign_historical_Task_readonly", 409)
            if request.kind != "cancel":
                campaign_control.require_runtime_task(task, campaign)
            if task["revision"] != request.expected_revision:
                raise AgentError("agent_revision_conflict")
            if request.requested_direction is not None and task["kind"] != "stance":
                raise AgentError("requested_direction_requires_stance", 400)
            if request.research_brief is not None and task["kind"] != "research":
                raise AgentError("research_brief_requires_research_task", 400)
            if (request.research_action is not None or request.research_help_fields is not None or request.research_extract_fields is not None) and task["kind"] != "research":
                raise AgentError("research_action_requires_research_task", 400)
            if request.research_action == "experiment" and task["profile"].get("profile_id") != "research-v6":
                raise AgentError("experiment_design_requires_research_v6", 400)
            review_task = task["profile"].get("profile_id") in {"research-review-v1", "research-revision-v1"}
            fixed_action = "review" if task["profile"].get("profile_id") == "research-review-v1" else "revise"
            if review_task and request.kind != "cancel" and (
                    request.research_action not in {None, fixed_action} or
                    (request.selected_skills is not None and [item.model_dump(mode="json") for item in request.selected_skills]
                     != task["brief"].get("selected_skills", [])) or
                    (request.research_brief is not None and
                     request.research_brief.model_dump(mode="json") != task["brief"]["research_brief"])):
                raise AgentError("research_review_source_is_immutable", 409)
            if request.kind == "cancel":
                db.execute("UPDATE cq_app.agent_tasks SET cancellation_requested=true WHERE id=%s", (task_id,))
                if task["state"] != "running":
                    db.execute("UPDATE cq_app.agent_tasks SET state='cancelled' WHERE id=%s", (task_id,))
                policy = dispatch_control.active_policy(db, self.database.dispatch_policy)
                if policy is not None:
                    db.execute("UPDATE cq_app.agent_model_waiters SET state='cancelled',reason='task_cancellation_requested' WHERE task_id=%s AND state='waiting'", (task_id,))
            else:
                catalog_scope.require_runtime(task, self.database.catalog_scope_policy)
                if self.database.catalog_scope_policy is not None and not catalog_scope.current_runtime(task, self.database.catalog_scope_policy):
                    raise AgentError('task_catalog_pinned_runtime_required', 503)
                allowed = {"resume": {"waiting_input"}, "followup": {"succeeded", "partial"}}[request.kind]
                if task["state"] not in allowed or not request.message.strip() or task["cancellation_requested"]:
                    raise AgentError("agent_command_not_admissible")
                if db.execute("SELECT 1 FROM cq_app.agent_attempts WHERE task_id=%s AND status IN ('reserved','sent','unknown')", (task_id,)).fetchone():
                    raise AgentError("agent_attempt_unresolved")
                if request.reference_turn_seq is not None:
                    if task["profile"]["profile_id"] not in DIALOGUE_REFERENCE_PROFILE_IDS:
                        raise AgentError("task_profile_does_not_support_turn_references", 400)
                    previous_command = db.execute("""SELECT id FROM cq_app.agent_commands
                        WHERE task_id=%s AND owner_id=%s AND seq=%s AND kind <> 'cancel' AND state='done'""",
                        (task_id, owner, request.reference_turn_seq)).fetchone()
                    if previous_command is None:
                        raise AgentError("reference_turn_not_completed", 422)
                wallet_admit(db, self.limits, owner, task["project_id"], activate_task=True,
                    development_active_policy=active_policy, development_active_target=task,
                    development_active_operation_id=request.operation_id, development_active_command_kind=request.kind,
                    development_campaign=campaign)
                policy = dispatch_control.active_policy(db, self.database.dispatch_policy)
                for expired in dispatch_control.expire_queued(db, policy):
                    event(db, expired["id"], "queue_deadline_expired", {"command_id": str(expired["command_id"]), "error_code": expired["error_code"]})
                dispatch_control.research_admit(db, task["kind"], policy)
                brief = {**task["brief"], "question": request.message}
                # The selector applies to one submitted turn, never to a later follow-up by inheritance.
                brief.pop("reference_turn_seq", None)
                if request.reference_turn_seq is not None:
                    brief["reference_turn_seq"] = request.reference_turn_seq
                if request.requested_direction is not None:
                    brief["requested_direction"] = request.requested_direction
                if request.research_brief is not None and not review_task:
                    from .research_runtime import bind_reference
                    document = bind_reference(db, owner, task["project_id"], request.research_brief)
                    if str(document.id) != task["brief"]["research_brief"]["id"]:
                        raise AgentError("research_brief_identity_cannot_change", 409)
                    brief.update(research_brief=request.research_brief.model_dump(mode="json"),
                        selected_papers=[item.model_dump(mode="json") for item in document.selected_sources],
                        research_draft_files=[str(value) for value in document.draft_source_ids])
                if request.selected_skills is not None:
                    brief["selected_skills"] = [item.model_dump(mode="json") for item in request.selected_skills]
                    from .user_skills import activate_for_task
                    activate_for_task(self.database, {**task, "brief": brief})
                if request.research_action is not None:
                    brief["research_action"] = request.research_action
                    if request.research_action != "suggest":
                        brief.pop("research_help_fields", None)
                    if request.research_action != "extract":
                        brief.pop("research_extract_fields", None)
                if request.research_help_fields is not None:
                    brief["research_help_fields"] = request.research_help_fields
                if request.research_extract_fields is not None:
                    brief["research_extract_fields"] = request.research_extract_fields
                from .contracts import ResearchBrief
                try:
                    brief = ResearchBrief.model_validate_json(canonical(brief), strict=True).model_dump(mode="json")
                except ValueError:
                    raise AgentError("invalid_task_brief_update", 422) from None
                db.execute("UPDATE cq_app.agent_tasks SET brief=%s,revision=revision+1,state='queued',result=NULL WHERE id=%s", (Jsonb(brief), task_id))
            seq = db.execute("UPDATE cq_app.agent_tasks SET command_seq=command_seq+1 WHERE id=%s RETURNING command_seq", (task_id,)).fetchone()["command_seq"]
            db.execute("INSERT INTO cq_app.agent_commands(id,task_id,owner_id,operation_id,seq,kind,payload,payload_hash,state) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (uuid4(), task_id, owner, request.operation_id, seq, request.kind, Jsonb(payload), request_hash,
                 "done" if request.kind == "cancel" else "queued"))
            command_event = {"kind": request.kind, "command_seq": seq}
            if active_policy is not None:
                command_event["development_dispatch"] = active_policy.public_binding()
            if campaign is not None and task["profile"].get(campaign_control.BINDING_KEY) == campaign.task_binding():
                command_event["development_campaign"] = campaign.public_binding()
            event(db, task_id, "cancellation_requested" if request.kind == "cancel" else "command_accepted", command_event)
            return self.public(owned_task(db, owner, task_id), catalog_scope_policy=self.database.catalog_scope_policy,
                development_campaign_policy=campaign)

    @staticmethod
    def public(task: dict, *, catalog_scope_policy: catalog_scope.CatalogScopePolicy | None = None,
               development_campaign_policy: campaign_control.DevelopmentCampaign | None = None) -> dict:
        value = {key: task[key] for key in ("id", "project_id", "kind", "brief", "profile", "revision", "state", "model_calls", "tool_calls", "input_tokens", "output_tokens", "known_ncny", "reserved_ncny", "active_seconds", "result", "error_code", "event_seq", "cancellation_requested")}
        value["command_seq"] = task["command_seq"]
        value.update(runtime_version=task["graph_version"], runtime_availability="current" if
            catalog_scope.readable_runtime(task)
            else "pinned_legacy_runtime_required")
        if catalog_scope_policy is not None or catalog_scope.task_policy(task) is not None:
            value['dispatch_runtime_availability'] = ('current' if catalog_scope.current_runtime(task, catalog_scope_policy)
                else 'different_catalog_policy_worker_required')
        if development_campaign_policy is not None or task["profile"].get(campaign_control.BINDING_KEY) is not None:
            try:
                campaign_control.require_runtime_task(task, development_campaign_policy)
            except AgentError:
                value["dispatch_runtime_availability"] = "different_development_campaign_worker_required"
        if task["result"] and "interrupts" not in task["result"] and task["result"].get("publication_status") != "validated_ai_review":
            value.update(result=None, result_availability="quality_rejected")
            if task["result"].get("publication_status") == "not_evaluated":
                value.update(result_availability="partial_no_evaluated_claims", coverage_report={key: task["result"].get(key)
                    for key in ("claim_plan", "claim_coverage", "coverage_gaps", "completion_status")})
        return value

    def get(self, owner: UUID, task_id: UUID) -> dict:
        with self.database.connect() as db:
            task = owned_task(db, owner, task_id)
            value = self.public(task, catalog_scope_policy=self.database.catalog_scope_policy,
                development_campaign_policy=getattr(self.database, "development_campaign_policy", None))
            policy = dispatch_control.active_policy(db, self.database.dispatch_policy)
            timing = dispatch_control.public_timing(db, task, policy)
            if timing is not None:
                value["dispatch_timing"] = timing
            return value

    def events(self, owner: UUID, task_id: UUID, after: int = 0) -> list[dict]:
        if type(after) is not int or after < 0:
            raise AgentError("invalid_event_cursor", 400)
        with self.database.connect() as db:
            owned_task(db, owner, task_id)
            rows = db.execute("SELECT seq,event_type,payload,created_at FROM cq_app.agent_events WHERE task_id=%s AND seq>%s ORDER BY seq LIMIT 200", (task_id, after)).fetchall()
            return [public_event(row) for row in rows]


class Execution:
    """One owner epoch; a closed/lost physical saver session is never replaced."""
    def __init__(self, database: Database, task_id: UUID, command: dict, run_id: UUID,
                 epoch: int, worker_id: str, saver_connection: psycopg.Connection, limits: RuntimeLimits) -> None:
        self.database, self.task_id, self.command, self.run_id = database, task_id, command, run_id
        self.epoch, self.worker_id, self.saver_connection = epoch, worker_id, saver_connection
        self.invalid = False
        self.limits = limits

    def lock_task(self, db: psycopg.Connection, *, dispatch: bool = True) -> dict:
        if self.invalid or self.saver_connection.closed or self.saver_connection.broken:
            self.invalid = True
            raise AgentError("execution_session_lost")
        # This SELECT uses the same physical connection as Saver. A reconnect is never attempted.
        try:
            self.saver_connection.execute("SELECT 1")
        except psycopg.Error:
            self.invalid = True
            raise AgentError("execution_session_lost") from None
        task_wallet_locks(db, self.task_id)
        task = db.execute("SELECT t.*,clock_timestamp() AS database_now FROM cq_app.agent_tasks t WHERE id=%s FOR UPDATE", (self.task_id,)).fetchone()
        if not task or task["epoch"] != self.epoch or task["worker_id"] != self.worker_id:
            raise AgentError("execution_owner_changed")
        if not db.execute("SELECT 1 FROM cq_app.projects WHERE id=%s AND owner_id=%s", (task["project_id"], task["owner_id"])).fetchone():
            raise AgentError("project_access_revoked", 403)
        elapsed = task["active_seconds"] + (max(0.0, (task["database_now"] - task["running_since"]).total_seconds()) if task["running_since"] else 0)
        task["elapsed"] = elapsed
        if dispatch:
            campaign_control.require_runtime_task(task, getattr(self.database, "development_campaign_policy", None))
            catalog_scope.require_runtime(task, self.database.catalog_scope_policy)
            if task["database_now"] < task["updated_at"]:
                raise AgentError("database_clock_regression")
            if task["state"] != "running" or task["cancellation_requested"]:
                raise AgentError("task_not_dispatchable")
            if elapsed >= task["profile"]["active_seconds"]:
                raise AgentError("active_budget_exhausted")
            dispatch_control.check_target(db, task, dispatch_control.active_policy(db, self.database.dispatch_policy))
            wallet_admit(db, self.limits, task["owner_id"], task["project_id"])
            db.execute("UPDATE cq_app.agent_tasks SET lease_until=clock_timestamp()+interval '15 seconds',updated_at=GREATEST(updated_at,clock_timestamp()) WHERE id=%s", (self.task_id,))
        return task

    def snapshot(self) -> dict:
        with self.database.connect() as db:
            return self.lock_task(db)

    def scope_event(self, event_type: str, payload: dict) -> None:
        if event_type not in {"scope_check_started", "scope_checked", "scope_check_failed", "tool_plan_bound"}:
            raise AgentError("invalid_scope_event_type")
        with self.database.connect() as db:
            self.lock_task(db)
            event(db, self.task_id, event_type, payload)

    def progress(self, phase: str, state: str) -> None:
        """Persist public execution phases, never a model's private reasoning or source text."""
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", phase) or state not in {"started", "completed", "waiting", "failed"}:
            raise AgentError("invalid_workflow_progress")
        with self.database.connect() as db:
            self.lock_task(db, dispatch=state == "started")
            event(db, self.task_id, "workflow_progress", {"phase": phase, "state": state, "run_id": str(self.run_id)})

    def content_revision_count(self) -> int:
        with self.database.connect() as db:
            self.lock_task(db)
            return int(db.execute("""SELECT count(*) AS n FROM cq_app.agent_events
                WHERE task_id=%s AND event_type='content_revision_reserved'""",
                (self.task_id,)).fetchone()["n"])

    def protocol_recovery_count(self) -> int:
        """Count durable protocol reservations independently of content repairs."""
        with self.database.connect() as db:
            self.lock_task(db)
            return int(db.execute("""SELECT count(*) AS n FROM cq_app.agent_events
                WHERE task_id=%s AND event_type='protocol_recovery_reserved'""",
                (self.task_id,)).fetchone()["n"])

    def reserve_protocol_recovery(self, parent_key: str, canonical_request: dict, violation: dict) -> str:
        """Reserve one child identity for a proven strict-wire contract failure.

        This event grants no physical dispatch or budget. The child still uses
        reserve/mark_sent/settle and all current authority and wallet checks.
        Parent receipts and the two content-revision credits remain unchanged.
        A known parent from an earlier run of this command may be inspected by
        the current lock_task owner, including after checkpoint recovery.
        """
        from .provider_contract import wire_contract_violation
        from .strict_chat import PROTOCOL_COMPILERS, usage_counts_chat
        from .typed_submission import COMPILER, VERSION, compile_submission, submission_role
        from .json_review import COMPILER as JSON_COMPILER, VERSION as JSON_VERSION, JSON2_COMPILER, JSON2_VERSION
        compilers = {**PROTOCOL_COMPILERS, VERSION: COMPILER, JSON_VERSION: JSON_COMPILER, JSON2_VERSION: JSON2_COMPILER}

        expected_fields = {"code", "category", "protocol", "compiler", "wire_schema_sha256",
            "arguments_sha256", "arguments_utf8_bytes"}
        if (not isinstance(parent_key, str) or len(parent_key) > 256
                or re.fullmatch(re.escape(str(self.command["id"])) + r":qa5:(plan|author|review):(0|[1-9][0-9]*)", parent_key) is None
                or not isinstance(canonical_request, dict)
                or not isinstance(violation, dict) or set(violation) != expected_fields
                or violation["code"] not in {"strict_review_wire_contract_violation", "json_object_wire_contract_violation"}
                or not isinstance(violation["category"], str)
                or violation["category"] not in {"invalid_json", "duplicate_property", "nonfinite_json", "wire_schema"}
                or type(violation["protocol"]) is not str or violation["protocol"] not in compilers
                or violation["compiler"] != compilers[violation["protocol"]]
                or any(not isinstance(violation[key], str) or re.fullmatch(r"[a-f0-9]{64}", violation[key]) is None
                    for key in ("wire_schema_sha256", "arguments_sha256"))
                or type(violation["arguments_utf8_bytes"]) is not int or violation["arguments_utf8_bytes"] < 1):
            raise AgentError("protocol_recovery_request_invalid")
        request_hash = digest(canonical_request)
        child_key = parent_key + ":wire-recovery:1"
        with self.database.connect() as db:
            task = self.lock_task(db)
            profile = task["profile"]
            role = submission_role(profile, canonical_request, parent_key)
            expected_protocol = (compile_submission(profile, canonical_request, parent_key).receipt["protocol"]
                if role is not None else profile.get("review_api_protocol"))
            if (role is None and ":qa5:review:" not in parent_key):
                raise AgentError("protocol_recovery_profile_not_enabled")
            if (task["kind"] != "qa" or profile.get("profile_id") not in {"qa-v6", "qa-v7", "qa-v8", "qa-v9"}
                    or profile.get("qa_execution_protocol") != "grounded3"
                    or profile.get("review_transport_protocol") not in {"function1", "json1"}
                    or profile.get("review_api_protocol") not in {"chat-strict1", "chat-strict2", "chat-json1", "chat-json2"}
                    or violation["protocol"] != expected_protocol):
                raise AgentError("protocol_recovery_profile_not_enabled")
            parent = db.execute("""SELECT * FROM cq_app.agent_attempts
                WHERE task_id=%s AND logical_key=%s AND kind='model' FOR UPDATE""",
                (self.task_id, parent_key)).fetchone()
            if (not parent or parent["kind"] != "model" or parent["logical_key"] != parent_key
                    or parent["status"] != "settled" or parent["error_code"] is not None
                    or not isinstance(parent["usage"], dict) or not isinstance(parent["response"], dict)
                    or any(type(parent["usage"].get(key)) is not int or parent["usage"][key] < 0
                        for key in ("input", "cached", "output"))
                    or parent["usage"]["cached"] > parent["usage"]["input"]):
                raise AgentError("protocol_recovery_requires_known_receipt")
            if (not isinstance(parent["request"], dict) or digest(parent["request"]) != parent["request_hash"]
                    or parent["request_hash"] != request_hash):
                raise AgentError("protocol_recovery_parent_request_changed")
            try:
                usage = usage_counts_chat(parent["response"])
            except AgentError:
                raise AgentError("protocol_recovery_requires_known_receipt") from None
            if usage != {key: parent["usage"][key] for key in ("input", "cached", "output")}:
                raise AgentError("protocol_recovery_requires_known_receipt")
            verified = wire_contract_violation(parent["request"], parent["response"], protocol=profile["review_api_protocol"], profile=profile, logical_key=parent_key)
            if verified is None or verified != violation:
                raise AgentError("protocol_recovery_violation_not_verified")
            wire = db.execute("""SELECT payload FROM cq_app.agent_events WHERE task_id=%s
                AND event_type='model_wire_bound' AND payload->>'attempt_id'=%s ORDER BY seq LIMIT 1""",
                (self.task_id, str(parent["id"]))).fetchone()
            if (not wire or wire["payload"].get("canonical_request_sha256") != request_hash
                    or wire["payload"].get("protocol") != verified["protocol"]
                    or wire["payload"].get("compiler") != verified["compiler"]
                    or wire["payload"].get("wire_schema_sha256") != verified["wire_schema_sha256"]):
                raise AgentError("protocol_recovery_wire_binding_invalid")
            identity = {"protocol": "citequest-protocol-recovery/1", "parent_logical_key": parent_key,
                "parent_attempt_id": str(parent["id"]), "canonical_request_sha256": request_hash,
                "child_logical_key": child_key, "violation": verified}
            identity_hash = digest(identity)
            rows = db.execute("""SELECT payload FROM cq_app.agent_events
                WHERE task_id=%s AND event_type='protocol_recovery_reserved' ORDER BY seq""",
                (self.task_id,)).fetchall()
            for row in rows:
                old = row["payload"]
                if old["parent_logical_key"] == parent_key:
                    if old["request_hash"] != identity_hash or old["child_logical_key"] != child_key:
                        raise AgentError("protocol_recovery_replay_changed")
                    return child_key
            if len(rows) >= 2:
                raise AgentError("protocol_recovery_limit")
            if db.execute("""SELECT 1 FROM cq_app.agent_attempts WHERE task_id=%s
                    AND status IN ('reserved','sent','unknown')""", (self.task_id,)).fetchone():
                raise AgentError("attempt_outcome_unresolved")
            receipt = {**identity, "request_hash": identity_hash, "ordinal": len(rows) + 1,
                "limit": 2, "per_parent_limit": 1, "run_id": str(self.run_id), "epoch": self.epoch}
            event(db, self.task_id, "provider_contract_violation", {
                **identity, "run_id": str(self.run_id), "epoch": self.epoch})
            event(db, self.task_id, "protocol_recovery_reserved", receipt)
            return child_key

    def effective_model_source_key(self, base_key: str) -> str:
        """Resolve a grounded3 content opinion to its actual known model receipt.

        Recovery never makes the malformed parent a source of valid opinions.
        A bound but unresolved or invalid child is a hard stop, not permission
        to fall back to that parent. This lookup writes no ledger records.
        """
        from .provider_contract import wire_contract_violation
        from .strict_chat import FIXTURE_ENDPOINT, LIVE_ENDPOINT, usage_counts_chat
        from .typed_submission import compile_submission, normalize_submission
        from .json_review import FIXTURE_ENDPOINT as JSON_FIXTURE, LIVE_ENDPOINT as JSON_LIVE, VERSION as JSON_VERSION, JSON2_VERSION

        if (not isinstance(base_key, str) or len(base_key) > 256
                or re.fullmatch(re.escape(str(self.command["id"])) + r":qa5:(plan|author|review):(0|[1-9][0-9]*)", base_key) is None):
            raise AgentError("protocol_recovery_source_identity_invalid")
        with self.database.connect() as db:
            task = self.lock_task(db)
            profile = task["profile"]
            if (task["kind"] != "qa" or profile.get("profile_id") not in {"qa-v6", "qa-v7", "qa-v8", "qa-v9"}
                    or profile.get("qa_execution_protocol") != "grounded3"
                    or profile.get("review_transport_protocol") not in {"function1", "json1"}
                    or profile.get("review_api_protocol") not in {"chat-strict1", "chat-strict2", "chat-json1", "chat-json2"}):
                raise AgentError("protocol_recovery_profile_not_enabled")
            rows = db.execute("""SELECT payload FROM cq_app.agent_events WHERE task_id=%s
                AND event_type='protocol_recovery_reserved' AND payload->>'parent_logical_key'=%s ORDER BY seq""",
                (self.task_id, base_key)).fetchall()
            if not rows:
                return base_key
            if len(rows) != 1 or (profile.get("structured_output_protocol") is None and ":qa5:review:" not in base_key):
                raise AgentError("protocol_recovery_source_binding_invalid")
            receipt = rows[0]["payload"]
            child_key = base_key + ":wire-recovery:1"
            if (receipt.get("protocol") != "citequest-protocol-recovery/1"
                    or receipt.get("parent_logical_key") != base_key or receipt.get("child_logical_key") != child_key):
                raise AgentError("protocol_recovery_source_binding_invalid")
            attempts = []
            for key in (base_key, child_key):
                row = db.execute("""SELECT * FROM cq_app.agent_attempts
                    WHERE task_id=%s AND logical_key=%s AND kind='model' FOR UPDATE""",
                    (self.task_id, key)).fetchone()
                if (not row or row["kind"] != "model" or row["logical_key"] != key
                        or row["status"] != "settled" or row["error_code"] is not None
                        or not isinstance(row["usage"], dict) or not isinstance(row["request"], dict)
                        or not isinstance(row["response"], dict)
                        or any(type(row["usage"].get(name)) is not int or row["usage"][name] < 0
                            for name in ("input", "cached", "output"))
                        or row["usage"]["cached"] > row["usage"]["input"]):
                    raise AgentError("protocol_recovery_source_requires_known_receipt")
                if digest(row["request"]) != row["request_hash"]:
                    raise AgentError("protocol_recovery_source_binding_invalid")
                try:
                    normalize_submission(profile, row["request"], key, row["response"])
                    usage = usage_counts_chat(row["response"])
                except AgentError:
                    raise AgentError("protocol_recovery_source_requires_known_receipt") from None
                if usage != {name: row["usage"][name] for name in ("input", "cached", "output")}:
                    raise AgentError("protocol_recovery_source_requires_known_receipt")
                wire = db.execute("""SELECT payload FROM cq_app.agent_events WHERE task_id=%s
                    AND event_type='model_wire_bound' AND payload->>'attempt_id'=%s ORDER BY seq LIMIT 1""",
                    (self.task_id, str(row["id"]))).fetchone()
                metadata = wire["payload"] if wire else {}
                compiled = compile_submission(profile, row["request"], key,
                    fixture=metadata.get("endpoint") in {FIXTURE_ENDPOINT, JSON_FIXTURE}).receipt
                if compiled["protocol"] in {JSON_VERSION, JSON2_VERSION}:
                    role_fields = ("structured_output_protocol", "role", "schema_name", "schema_enforcement")
                else:
                    role_fields = (("structured_output_protocol", "role", "function_name", "schema_name", "nullable_type_contract")
                        if profile.get("structured_output_protocol") is not None else ())
                # JSONB can reorder object properties and therefore compiler
                # annotation lists. Compare canonical identities, not those
                # diagnostic list orders. Actual SDK bytes are unavailable
                # here; their original positive size and hash remain evidence.
                if (not isinstance(metadata, dict) or metadata.get("attempt_id") != str(row["id"])
                        or not isinstance(metadata.get("endpoint"), str)
                        or metadata["endpoint"] not in {LIVE_ENDPOINT, FIXTURE_ENDPOINT, JSON_LIVE, JSON_FIXTURE}
                        or any(metadata.get(name) != compiled[name] for name in (
                            "protocol", "compiler", "endpoint", "canonical_request_sha256", "full_schema_sha256",
                            "wire_schema_sha256", "wire_request_sha256", *role_fields))
                        or type(metadata.get("actual_wire_bytes")) is not int
                        or not 0 < metadata["actual_wire_bytes"] <= profile["request_bytes"]
                        or not isinstance(metadata.get("actual_wire_sha256"), str)
                        or re.fullmatch(r"[a-f0-9]{64}", metadata["actual_wire_sha256"]) is None):
                    raise AgentError("protocol_recovery_source_binding_invalid")
                attempts.append(row)
            parent, child = attempts
            violation = wire_contract_violation(parent["request"], parent["response"], protocol=profile["review_api_protocol"], profile=profile, logical_key=base_key)
            identity = {"protocol": "citequest-protocol-recovery/1", "parent_logical_key": base_key,
                "parent_attempt_id": str(parent["id"]), "canonical_request_sha256": parent["request_hash"],
                "child_logical_key": child_key, "violation": violation}
            if (violation is None or receipt.get("violation") != violation
                    or any(receipt.get(key) != value for key, value in identity.items())
                    or receipt.get("request_hash") != digest(identity)
                    or child["request_hash"] != parent["request_hash"]
                    or wire_contract_violation(child["request"], child["response"], protocol=profile["review_api_protocol"], profile=profile, logical_key=child_key) is not None):
                raise AgentError("protocol_recovery_source_binding_invalid")
            return child_key

    def reserve_content_revision(self, logical_key: str, *, phase: str,
                                 error_codes: list[str], source_logical_key: str | None = None) -> dict:
        """Durably bound known-content repairs across all commands of one Task.

        This internal capability is not a provider retry. The model receipt must
        already have settled with real usage. A reservation is never refunded;
        idempotent replay returns the same ordinal after checkpoint recovery.
        """
        if (not isinstance(logical_key, str) or not 1 <= len(logical_key) <= 256
                or re.fullmatch(r"[A-Za-z0-9:._/-]+", logical_key) is None
                or not isinstance(phase, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", phase) is None
                or not isinstance(error_codes, list) or not 1 <= len(error_codes) <= 8
                or any(not isinstance(code, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) is None for code in error_codes)):
            raise AgentError("invalid_content_revision_request")
        identity = {"logical_key": logical_key, "phase": phase,"error_codes": sorted(set(error_codes))}
        if source_logical_key is not None:
            if (not isinstance(source_logical_key,str) or not 1 <= len(source_logical_key) <= 256
                    or not source_logical_key.startswith(str(self.command["id"])+":")
                    or re.fullmatch(r"[A-Za-z0-9:._/-]+",source_logical_key) is None):
                raise AgentError("content_revision_source_identity_invalid")
            identity["source_logical_key"] = source_logical_key
        request_hash = digest(identity)
        with self.database.connect() as db:
            task = self.lock_task(db)
            if task["profile"].get("profile_id") not in {"workset-v3", "workset-v4", "workset-v5", "workset-v6", "workset-v7", "qa-v4", "qa-v5", "qa-v6", "qa-v7", "qa-v8", "qa-v9", "research-v3", "research-v4", "research-v5", "research-v6", "research-revision-v1", "stance-v2", "stance-v3"}:
                raise AgentError("content_revision_profile_not_enabled")
            rows = db.execute("""SELECT payload FROM cq_app.agent_events
                WHERE task_id=%s AND event_type='content_revision_reserved' ORDER BY seq""",
                (self.task_id,)).fetchall()
            for row in rows:
                old = row["payload"]
                if old["logical_key"] == logical_key:
                    if old["request_hash"] != request_hash:
                        raise AgentError("content_revision_replay_changed")
                    return {key: old[key] for key in ("logical_key", "ordinal", "limit")}
            if len(rows) >= 2:
                raise AgentError("content_revision_limit")
            if db.execute("""SELECT 1 FROM cq_app.agent_attempts WHERE task_id=%s
                    AND status IN ('reserved','sent','unknown')""", (self.task_id,)).fetchone():
                raise AgentError("attempt_outcome_unresolved")
            receipt = db.execute("""SELECT id,status,usage,error_code FROM cq_app.agent_attempts
                WHERE task_id=%s AND kind='model'""" + (" AND logical_key=%s" if source_logical_key is not None else "") +
                " ORDER BY created_at DESC,id DESC LIMIT 1",
                (self.task_id,source_logical_key) if source_logical_key is not None else (self.task_id,)).fetchone()
            if not receipt or receipt["status"] != "settled" or receipt["usage"] is None or receipt["error_code"]:
                raise AgentError("content_revision_requires_known_receipt")
            value = {"logical_key": logical_key, "ordinal": len(rows) + 1, "limit": 2}
            event(db, self.task_id, "content_revision_reserved", {**value,
                "request_hash": request_hash, "phase": phase, "error_codes": sorted(set(error_codes)),
                "source_attempt_id": str(receipt["id"]), "run_id": str(self.run_id), "epoch": self.epoch,
                **({"source_logical_key":source_logical_key} if source_logical_key is not None else {})})
            return value

    def settled_model_request(self, logical_key: str) -> dict | None:
        """Current-owner reconstruction hint for one already known model receipt.

        This does not dispatch, reserve, refund or authorize a changed request.
        The caller must reconstruct the complete current payload and compare its
        hash; the ordinary provider boundary still rechecks current authority.
        """
        if not isinstance(logical_key, str) or not logical_key.startswith(str(self.command["id"]) + ":"):
            raise AgentError("settled_request_command_identity_invalid")
        with self.database.connect() as db:
            task = self.lock_task(db)
            row = db.execute("""SELECT request,request_hash,usage,response,error_code,status FROM cq_app.agent_attempts
                WHERE task_id=%s AND logical_key=%s AND kind='model'""", (self.task_id, logical_key)).fetchone()
            if not row or row["status"] != "settled" or row["error_code"] is not None:
                return None
            usage = row["usage"]
            if (not isinstance(usage, dict)
                    or any(type(usage.get(key)) is not int or usage[key] < 0 for key in ("input", "cached", "output"))
                    or usage["cached"] > usage["input"] or not isinstance(row["response"], dict)):
                return None
            if not isinstance(row["request"], dict) or digest(row["request"]) != row["request_hash"]:
                raise AgentError("settled_request_hash_invalid")
            completed = row["response"].get("status") == "completed"
            if task["profile"].get("review_api_protocol") is not None:
                from .strict_chat import is_strict_review
                from .typed_submission import normalize_submission, submission_role
                role = submission_role(task["profile"], row["request"], logical_key)
                if role is not None or is_strict_review(task["profile"], row["request"]):
                    normalize_submission(task["profile"], row["request"], logical_key, row["response"])
                    completed = True
            if not completed:
                return None
            return {"request": row["request"], "request_hash": row["request_hash"]}

    def settled_tool_keys(self, logical_keys: list[str]) -> set[str]:
        """Count known charged replay keys without authorizing cached sources."""
        if (not isinstance(logical_keys, list) or len(logical_keys) > 128
                or any(not isinstance(key, str) or not 1 <= len(key) <= 256
                       or not key.startswith(str(self.command["id"]) + ":") for key in logical_keys)):
            raise AgentError("settled_tool_command_identity_invalid")
        keys = list(dict.fromkeys(logical_keys))
        if not keys:
            return set()
        with self.database.connect() as db:
            self.lock_task(db)
            rows = db.execute("""SELECT logical_key FROM cq_app.agent_attempts
                WHERE task_id=%s AND kind='tool' AND logical_key=ANY(%s) AND status='settled'
                AND error_code IS NULL AND jsonb_typeof(response)='object'""", (self.task_id, keys)).fetchall()
            return {row["logical_key"] for row in rows}

    def settled_model_receipts(self, logical_keys: list[str]) -> dict[str, dict]:
        """One current-owner read for immutable handoff links in this command."""
        if (not isinstance(logical_keys, list) or len(logical_keys) > 128
                or any(not isinstance(key, str) or not 1 <= len(key) <= 256
                       or not key.startswith(str(self.command["id"]) + ":") for key in logical_keys)):
            raise AgentError("handoff_receipt_command_identity_invalid")
        keys = list(dict.fromkeys(logical_keys))
        if not keys:
            return {}
        from .model_receipts import model_receipt
        with self.database.connect() as db:
            task = self.lock_task(db)
            rows = db.execute("""SELECT id,logical_key,request,request_hash,response,usage,status,error_code
                FROM cq_app.agent_attempts WHERE task_id=%s AND kind='model' AND logical_key=ANY(%s)""",
                (self.task_id, keys)).fetchall()
            if {row["logical_key"] for row in rows} != set(keys):
                raise AgentError("handoff_model_receipt_not_found")
            return {row["logical_key"]: model_receipt(row, row["logical_key"], task["profile"]) for row in rows}

    def observe_model_wire(self, attempt_id: UUID, metadata: dict) -> dict:
        """Bind the trusted strict-review adapter and actual SDK bytes before send."""
        from .strict_chat import LENGTH_VERSION, PROTOCOL_COMPILERS
        from .typed_submission import COMPILER, VERSION, compile_submission
        from .json_review import COMPILER as JSON_COMPILER, VERSION as JSON_VERSION, JSON2_COMPILER, JSON2_VERSION

        compilers = {**PROTOCOL_COMPILERS, VERSION: COMPILER, JSON_VERSION: JSON_COMPILER, JSON2_VERSION: JSON2_COMPILER}
        role_fields = {"structured_output_protocol", "role", "function_name", "schema_name", "nullable_type_contract"}

        hashes = ("canonical_request_sha256", "wire_request_sha256", "full_schema_sha256",
            "wire_schema_sha256", "actual_wire_sha256")
        expected = {"protocol", "compiler", "endpoint", "local_only_constraints", "removed_annotations",
            "actual_wire_bytes", *hashes}
        protocol = metadata.get("protocol") if isinstance(metadata, dict) else None
        if protocol in {JSON_VERSION, JSON2_VERSION}:
            role_fields = {"structured_output_protocol", "role", "schema_name", "schema_enforcement"}
        if protocol in {LENGTH_VERSION, VERSION, JSON_VERSION, JSON2_VERSION}:
            expected.add("transformed_constraints")
        if protocol in {VERSION, JSON_VERSION, JSON2_VERSION}:
            expected.update(role_fields)
        if (not isinstance(metadata, dict) or set(metadata) != expected
                or type(protocol) is not str or protocol not in compilers
                or metadata["compiler"] != compilers[protocol]
                or metadata["endpoint"] not in {"https://api.deepseek.com/beta/chat/completions",
                    "https://s08-provider.invalid/beta/chat/completions", "https://api.deepseek.com/chat/completions",
                    "https://s08-provider.invalid/chat/completions"}
                or any(not isinstance(metadata[key], str) or re.fullmatch(r"[a-f0-9]{64}", metadata[key]) is None for key in hashes)
                or type(metadata["actual_wire_bytes"]) is not int or metadata["actual_wire_bytes"] <= 0
                or not isinstance(metadata["local_only_constraints"], list) or not isinstance(metadata["removed_annotations"], list)
                or (protocol in {LENGTH_VERSION, VERSION, JSON_VERSION, JSON2_VERSION} and not isinstance(metadata["transformed_constraints"], list))
                or len(canonical(metadata)) > 65536):
            raise AgentError("model_wire_receipt_invalid")
        with self.database.connect() as db:
            task = self.lock_task(db)
            from .strict_chat import is_strict_review
            attempt = db.execute("SELECT * FROM cq_app.agent_attempts WHERE id=%s AND task_id=%s FOR UPDATE",
                (attempt_id, self.task_id)).fetchone()
            profile = task["profile"]
            typed = profile.get("structured_output_protocol") in {VERSION, JSON2_VERSION}
            if typed:
                if not attempt or protocol not in {VERSION, JSON_VERSION, JSON2_VERSION}:
                    raise AgentError("model_wire_receipt_identity_invalid")
                compiled = compile_submission(profile, attempt["request"], attempt["logical_key"],
                    fixture=metadata["endpoint"] in {"https://s08-provider.invalid/beta/chat/completions", "https://s08-provider.invalid/chat/completions"}).receipt
                identities = {"protocol", "compiler", "endpoint", "canonical_request_sha256", "full_schema_sha256",
                    "wire_schema_sha256", "wire_request_sha256", *role_fields}
                if any(metadata.get(name) != compiled[name] for name in identities):
                    raise AgentError("model_wire_receipt_identity_invalid")
            elif protocol != profile.get("review_api_protocol") or not attempt or not is_strict_review(profile, attempt["request"]):
                raise AgentError("model_wire_receipt_identity_invalid")
            if (not attempt or attempt["kind"] != "model" or attempt["run_id"] != self.run_id or attempt["epoch"] != self.epoch
                    or attempt["status"] != "reserved"
                    or metadata["canonical_request_sha256"] != attempt["request_hash"]
                    or metadata["actual_wire_bytes"] > task["profile"]["request_bytes"]):
                raise AgentError("model_wire_receipt_identity_invalid")
            old = db.execute("SELECT payload FROM cq_app.agent_events WHERE task_id=%s AND event_type='model_wire_bound' AND payload->>'attempt_id'=%s ORDER BY seq LIMIT 1",
                (self.task_id, str(attempt_id))).fetchone()
            payload = {"attempt_id": str(attempt_id), **metadata}
            if old:
                if old["payload"] != payload:
                    raise AgentError("model_wire_receipt_replay_changed")
                return old["payload"]
            event(db, self.task_id, "model_wire_bound", payload)
            return payload

    def enqueue_model(self, logical_key: str, request: dict) -> None:
        """Commit a zero-cost ticket before attempting the separately atomic slot."""
        with self.database.connect() as db:
            task = self.lock_task(db)
            policy = dispatch_control.active_policy(db, self.database.dispatch_policy)
            if policy is None:
                return
            old = db.execute("SELECT request_hash,status,kind FROM cq_app.agent_attempts WHERE task_id=%s AND logical_key=%s", (self.task_id, logical_key)).fetchone()
            if old:
                if old["request_hash"] != digest(request) or old["kind"] != "model":
                    raise AgentError("attempt_replay_request_changed")
                # Known replay and a definitely unsent, already reserved attempt
                # retain their original accounting/slot. Neither re-enqueues.
                if old["status"] in {"settled", "reserved"}:
                    return
                raise AgentError("attempt_outcome_unresolved")
            profile = task["profile"]
            output_limit = request.get("max_output_tokens")
            if (type(output_limit) is not int or not 1 <= output_limit <= min(profile["per_call_output"], profile["output_tokens"]-task["output_tokens"])
                    or task["model_calls"] >= profile["model_calls"] or task["input_tokens"] >= profile["input_tokens"]):
                raise AgentError("model_budget_exhausted")
            if len(canonical(request)) > profile["request_bytes"]:
                raise AgentError("context_byte_limit")
            dispatch_control.enqueue(db, task, self.command, self.run_id, self.epoch, logical_key, request)

    def cancel_model_waiter(self, reason: str) -> None:
        with self.database.connect() as db:
            task_wallet_locks(db, self.task_id)
            policy = dispatch_control.active_policy(db, self.database.dispatch_policy)
            if policy is not None:
                dispatch_control.close_waiters(db, self.task_id, self.run_id, self.epoch, reason)

    def development_occupied(self, db: psycopg.Connection, task: dict,
                             request: dict, logical_key: str, *, current_attempt_id: UUID | None = None) -> list[dict] | None:
        """Recheck the narrow development guard at every possible HTTP entry.

        Caller holds the ordinary campaign/owner/project/Task locks. A reserved
        replay does not buy another reservation, but it also must not retain a
        stale exception after another unknown or an authority change appears.
        """
        campaign = getattr(self.database, "development_campaign_policy", None)
        if campaign is not None:
            campaign.require_configuration(self.limits, self.database.catalog_scope_policy)
            if dispatch_control.active_policy(db, self.database.dispatch_policy) is not None:
                raise AgentError("development_campaign_FIFO_binding_not_adopted", 503)
            return campaign.model_occupied(db, task, self.command, request, logical_key,
                current_attempt_id=current_attempt_id)
        policy = getattr(self.database, "development_dispatch_policy", None)
        if policy is None:
            return None
        if (dispatch_control.active_policy(db, self.database.dispatch_policy) is not None
                or self.limits.mode != "bounded" or self.limits.model_dispatch_concurrency != 1
                or (policy.version == "citequest-development-named-unknown-dispatch/2" and self.limits.owner_active_tasks != 2)
                or self.limits.campaign_cash_ncny != policy.campaign_cash_ncny
                or self.limits.campaign_prior_reserved_ncny != policy.campaign_prior_reserved_ncny):
            raise AgentError("development_dispatch_runtime_policy_mismatch")
        from .development_dispatch import validate_occupied
        return validate_occupied(db, task, request, logical_key, self.command, policy,
                                 current_attempt_id=current_attempt_id)

    def reserved_dispatch_guard(self, db: psycopg.Connection, task: dict, attempt: dict) -> None:
        """A definitely-unsent reservation never freezes an old dispatch grant."""
        occupied = self.development_occupied(db, task, attempt["request"], attempt["logical_key"],
                                            current_attempt_id=attempt["id"])
        policy = dispatch_control.active_policy(db, self.database.dispatch_policy)
        limit = policy.model_concurrency if policy is not None else self.limits.model_dispatch_concurrency
        if occupied is None:
            if limit is None:
                return
            occupied = db.execute("""SELECT id,status FROM cq_app.agent_attempts
                WHERE kind='model' AND status IN ('reserved','sent','unknown') AND id<>%s""", (attempt["id"],)).fetchall()
        if any(row["status"] == "unknown" for row in occupied):
            raise AgentError("global_model_dispatch_unknown")
        if len(occupied) >= limit:
            raise AgentError("global_model_dispatch_busy", 429)

    def reserve(self, logical_key: str, request: dict, *, kind: str = "model") -> dict:
        if kind == "model":
            self.enqueue_model(logical_key, request)
        with self.database.connect() as db:
            task = self.lock_task(db)
            old = db.execute("SELECT * FROM cq_app.agent_attempts WHERE task_id=%s AND logical_key=%s FOR UPDATE", (self.task_id, logical_key)).fetchone()
            if old:
                if old["request_hash"] != digest(request) or old["kind"] != kind:
                    raise AgentError("attempt_replay_request_changed")
                if old["status"] == "settled":
                    if old["error_code"]:
                        raise AgentError(old["error_code"])
                    return old
                if old["status"] != "reserved":
                    raise AgentError("attempt_outcome_unresolved")
                if kind == "model":
                    self.reserved_dispatch_guard(db, task, old)
                # Reserved is definitely before dispatch: mark_sent commits before HTTP/MCP starts.
                db.execute("UPDATE cq_app.agent_attempts SET run_id=%s,epoch=%s WHERE id=%s", (self.run_id, self.epoch, old["id"]))
                return {**old, "run_id": self.run_id, "epoch": self.epoch}
            if db.execute("SELECT 1 FROM cq_app.agent_attempts WHERE task_id=%s AND status IN ('reserved','sent','unknown')", (self.task_id,)).fetchone():
                raise AgentError("attempt_outcome_unresolved")
            profile = task["profile"]
            if kind == "model":
                policy = dispatch_control.active_policy(db, self.database.dispatch_policy)
                output_limit = request.get("max_output_tokens")
                if type(output_limit) is not int or not 1 <= output_limit <= min(profile["per_call_output"], profile["output_tokens"] - task["output_tokens"]):
                    raise AgentError("output_budget_exhausted")
                if len(canonical(request)) > profile["request_bytes"]:
                    raise AgentError("context_byte_limit")
                reservation = profile["context_window"] * profile["input_miss_ncny"] + output_limit * profile["output_ncny"]
                wallet_admit(db, self.limits, task["owner_id"], task["project_id"], reservation=reservation)
                if (task["model_calls"] >= profile["model_calls"] or task["input_tokens"] >= profile["input_tokens"]
                    or task["known_ncny"] + task["reserved_ncny"] + reservation > profile["cash_ncny"]):
                    raise AgentError("model_budget_exhausted")
                if (self.limits.model_dispatch_concurrency is not None or policy is not None
                        or getattr(self.database, "development_dispatch_policy", None) is not None
                        or getattr(self.database, "development_campaign_policy", None) is not None):
                    # The global wallet lock serializes check+insert. Durable unresolved rows hold
                    # the slot across network I/O, worker death and unknown outcomes.
                    occupied = self.development_occupied(db, task, request, logical_key)
                    if occupied is None:
                        occupied = db.execute("""SELECT status FROM cq_app.agent_attempts
                            WHERE kind='model' AND status IN ('reserved','sent','unknown')""").fetchall()
                    if any(row["status"] == "unknown" for row in occupied):
                        raise AgentError("global_model_dispatch_unknown")
                    if len(occupied) >= (policy.model_concurrency if policy is not None else self.limits.model_dispatch_concurrency):
                        raise AgentError("global_model_dispatch_busy", 429)
                if policy is not None:
                    dispatch_control.require_turn(db, task, logical_key, request, self.run_id, self.epoch)
                column = "model_calls"
            else:
                output_limit, reservation, column = 0, 0, "tool_calls"
                if task["tool_calls"] >= profile["tool_calls"]:
                    raise AgentError("tool_budget_exhausted")
            attempt_id = uuid4()
            row = db.execute("""INSERT INTO cq_app.agent_attempts(id,task_id,run_id,epoch,logical_key,kind,request_hash,request,reserved_ncny,output_limit)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                (attempt_id, self.task_id, self.run_id, self.epoch, logical_key, kind, digest(request), Jsonb(request), reservation, output_limit)).fetchone()
            db.execute(f"UPDATE cq_app.agent_tasks SET {column}={column}+1,reserved_ncny=reserved_ncny+%s WHERE id=%s", (reservation, self.task_id))
            reserved_event = {"attempt_id": str(attempt_id), "kind": kind, "reserved_ncny": reservation,
                "provider_metadata_sha256": getattr(self, "provider_metadata_sha256", None) if kind == "model" else None}
            development_policy = getattr(self.database, "development_dispatch_policy", None)
            if kind == "model" and development_policy is not None:
                reserved_event["development_dispatch"] = development_policy.public_binding()
            campaign = getattr(self.database, "development_campaign_policy", None)
            if kind == "model" and campaign is not None:
                reserved_event["development_campaign"] = campaign.public_binding()
            event(db, self.task_id, "attempt_reserved", reserved_event)
            return row

    def mark_sent(self, attempt_id: UUID) -> float:
        with self.database.connect() as db:
            task = self.lock_task(db)
            attempt = db.execute("SELECT * FROM cq_app.agent_attempts WHERE id=%s AND task_id=%s AND epoch=%s AND status='reserved' FOR UPDATE",
                (attempt_id,self.task_id,self.epoch)).fetchone()
            if attempt is None:
                raise AgentError("attempt_not_reserved")
            if attempt["kind"] == "model":
                self.reserved_dispatch_guard(db, task, attempt)
            changed = db.execute("UPDATE cq_app.agent_attempts SET status='sent',sent_at=clock_timestamp() WHERE id=%s AND task_id=%s AND epoch=%s AND status='reserved' RETURNING id", (attempt_id, self.task_id, self.epoch)).fetchone()
            if not changed:
                raise AgentError("attempt_not_reserved")
            policy = dispatch_control.active_policy(db, self.database.dispatch_policy)
            # The provider and MCP callers use this value as their physical
            # I/O deadline. QA's pre-claim queue must consume that same window,
            # not merely be checked again after a late response arrives.
            elapsed = dispatch_control.target_elapsed(db, task) if policy is not None else task["elapsed"]
            return task["profile"]["active_seconds"] - elapsed

    def settle(self, attempt_id: UUID, response: dict, usage: dict | None) -> str | None:
        # Accounting may accept authentic late receipts without granting the old run graph rights.
        # The provider adapter owns this capability; it is never an HTTP caller-controlled endpoint.
        with self.database.connect() as db:
            task_wallet_locks(db, self.task_id)
            task = db.execute("SELECT * FROM cq_app.agent_tasks WHERE id=%s FOR UPDATE", (self.task_id,)).fetchone()
            attempt = db.execute("SELECT * FROM cq_app.agent_attempts WHERE id=%s AND task_id=%s FOR UPDATE", (attempt_id, self.task_id)).fetchone()
            if not attempt:
                raise AgentError("attempt_not_found")
            sanitized = redact(response)
            if attempt["status"] == "settled":
                if attempt["usage"] != usage or attempt["response"] != sanitized:
                    raise AgentError("receipt_conflict")
                return attempt["error_code"]
            if attempt["status"] not in {"sent", "unknown"}:
                raise AgentError("receipt_without_dispatch")
            if attempt["response"] is not None and attempt["response"] != sanitized:
                raise AgentError("receipt_conflict")
            cost, error = 0, None
            input_tokens, output_tokens = task["input_tokens"], task["output_tokens"]
            if attempt["kind"] == "model":
                if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input", "cached", "output")) or usage["cached"] > usage["input"]:
                    raise AgentError("malformed_usage")
                p = task["profile"]
                cost = (usage["input"] - usage["cached"]) * p["input_miss_ncny"] + usage["cached"] * p["input_cached_ncny"] + usage["output"] * p["output_ncny"]
                input_tokens += usage["input"]
                output_tokens += usage["output"]
                if input_tokens > p["input_tokens"] or output_tokens > p["output_tokens"] or usage["output"] > attempt["output_limit"] or cost > attempt["reserved_ncny"]:
                    error = "observed_usage_budget_exceeded"
            db.execute("UPDATE cq_app.agent_attempts SET status='settled',response=%s,usage=%s,cost_ncny=%s,error_code=%s,settled_at=clock_timestamp() WHERE id=%s", (Jsonb(sanitized), Jsonb(usage) if usage is not None else None, cost, error, attempt_id))
            db.execute("UPDATE cq_app.agent_tasks SET input_tokens=%s,output_tokens=%s,known_ncny=known_ncny+%s,reserved_ncny=reserved_ncny-%s WHERE id=%s", (input_tokens, output_tokens, cost, attempt["reserved_ncny"], self.task_id))
            if error:
                db.execute("UPDATE cq_app.agent_tasks SET error_code=%s WHERE id=%s", (error, self.task_id))
            event(db, self.task_id, "attempt_settled", {"attempt_id": str(attempt_id), "kind": attempt["kind"], "usage": usage, "cost_ncny": cost, "error_code": error,
                "source_reads": sanitized.get("source_reads") if attempt["kind"] == "tool" else None,
                "late_receipt": task["epoch"] != self.epoch or attempt["status"] == "unknown"})
            return error

    def observe_model_http_receipt(self, attempt_id: UUID, response: dict, metadata: dict) -> dict:
        """Persist a completed HTTP observation before any usage interpretation.

        Transport-only capability: no settlement, refund, usage inference,
        status/epoch advancement or graph permission is granted. Authentic late
        bodies may be observed after ownership changes, like accounting receipts.
        Public events contain ONLY the fixed transport metadata, never body text.
        """
        fields = {"http_status","body_sha256","body_bytes","json_type","usage_field_state"}
        # Five fields: usage_field_state encodes existence/type without exposing
        # arbitrary provider error strings, headers or generated/source content.
        if (not isinstance(metadata,dict) or set(metadata) != fields
                or type(metadata["http_status"]) is not int or not 100 <= metadata["http_status"] <= 599
                or type(metadata["body_bytes"]) is not int or not 0 <= metadata["body_bytes"] <= 4*1024*1024
                or not isinstance(metadata["body_sha256"],str) or re.fullmatch(r"[a-f0-9]{64}",metadata["body_sha256"]) is None
                or not isinstance(metadata["json_type"],str) or not isinstance(metadata["usage_field_state"],str)
                or metadata["json_type"] not in {"object","array","string","number","boolean","null","invalid_json"}
                or metadata["usage_field_state"] not in {"object","missing","null","non_object","not_applicable"}
                or not isinstance(response,dict)):
            raise AgentError("model_http_receipt_metadata_invalid")
        if metadata["json_type"] == "object":
            actual = ("missing" if "usage" not in response else "null" if response["usage"] is None
                else "object" if isinstance(response["usage"],dict) else "non_object")
            if metadata["usage_field_state"] != actual:
                raise AgentError("model_http_receipt_usage_shape_mismatch")
        elif metadata["json_type"] == "invalid_json":
            if metadata["usage_field_state"] != "not_applicable" or response != {"_cq_non_json_body_recorded":False}:
                raise AgentError("model_http_receipt_non_json_body_forbidden")
        else:
            if metadata["usage_field_state"] != "not_applicable" or set(response) != {"_cq_received_json_value"}:
                raise AgentError("model_http_receipt_json_envelope_invalid")
            value = response["_cq_received_json_value"]
            kind = ("null" if value is None else "boolean" if type(value) is bool else
                "number" if type(value) in {int,float} else "string" if isinstance(value,str) else
                "array" if isinstance(value,list) else "object")
            if kind != metadata["json_type"]:
                raise AgentError("model_http_receipt_json_envelope_invalid")
        sanitized = redact(response)
        try:
            if len(canonical(sanitized)) > 4*1024*1024+256:
                raise AgentError("model_http_receipt_json_byte_limit")
        except (TypeError,ValueError,UnicodeError):
            raise AgentError("model_http_receipt_json_invalid") from None
        payload = {"attempt_id":str(attempt_id),**metadata}
        with self.database.connect() as db:
            task_wallet_locks(db,self.task_id)
            db.execute("SELECT id FROM cq_app.agent_tasks WHERE id=%s FOR UPDATE",(self.task_id,)).fetchone()
            attempt = db.execute("SELECT * FROM cq_app.agent_attempts WHERE id=%s AND task_id=%s FOR UPDATE",(attempt_id,self.task_id)).fetchone()
            if (not attempt or attempt["kind"] != "model" or attempt["run_id"] != self.run_id
                    or attempt["epoch"] != self.epoch):
                raise AgentError("model_http_receipt_attempt_identity_invalid")
            previous = db.execute("""SELECT payload FROM cq_app.agent_events WHERE task_id=%s
                AND event_type='provider_http_response_observed' AND payload->>'attempt_id'=%s""",
                (self.task_id,str(attempt_id))).fetchall()
            if previous:
                if len(previous) != 1 or previous[0]["payload"] != payload or attempt["response"] != sanitized:
                    raise AgentError("receipt_conflict")
                return {"attempt_id":str(attempt_id),"already_recorded":True}
            if attempt["status"] not in {"sent","unknown"} or attempt["sent_at"] is None:
                raise AgentError("receipt_without_dispatch")
            if attempt["response"] is not None and attempt["response"] != sanitized:
                raise AgentError("receipt_conflict")
            db.execute("UPDATE cq_app.agent_attempts SET response=%s WHERE id=%s AND task_id=%s",
                (Jsonb(sanitized),attempt_id,self.task_id))
            event(db,self.task_id,"provider_http_response_observed",payload)
            return {"attempt_id":str(attempt_id),"already_recorded":False}

    def not_sent(self, attempt_id: UUID, error_code: str, *, before_wire: bool) -> bool:
        """Trusted transport-only proof of no HTTP entry; never fabricate provider usage.

        This can settle accounting after session loss, but cannot refund an attempt
        that has been adopted by a different run/epoch or advance either graph.
        """
        if before_wire is not True:
            raise AgentError("not_sent_requires_transport_before_wire_proof")
        with self.database.connect() as db:
            task_wallet_locks(db, self.task_id)
            attempt = db.execute("SELECT * FROM cq_app.agent_attempts WHERE id=%s AND task_id=%s FOR UPDATE",
                (attempt_id, self.task_id)).fetchone()
            if attempt is None or attempt["kind"] != "model":
                raise AgentError("not_sent_attempt_not_found")
            if attempt["run_id"] != self.run_id or attempt["epoch"] != self.epoch:
                return False
            if attempt["status"] == "not_sent":
                return True
            if attempt["status"] not in {"reserved", "sent", "unknown"}:
                raise AgentError("not_sent_attempt_already_settled")
            db.execute("UPDATE cq_app.agent_attempts SET status='not_sent',error_code=%s,cost_ncny=0 WHERE id=%s",
                (error_code, attempt_id))
            db.execute("UPDATE cq_app.agent_tasks SET reserved_ncny=reserved_ncny-%s WHERE id=%s",
                (attempt["reserved_ncny"], self.task_id))
            event(db, self.task_id, "attempt_not_sent", {"attempt_id": str(attempt_id), "error_code": error_code,
                "physical_HTTP_started": False, "usage": None, "reservation_released_ncny": attempt["reserved_ncny"],
                "model_attempt_counter_retained": True})
            return True

    def observe_tool_receipt(self, attempt_id: UUID, response: dict) -> None:
        """Retain a partial tool receipt without settling unknown computation.

        This adapter-only observation accepts a late response from the original
        physical attempt. It grants no graph, source, retry or accounting rights.
        """
        if not isinstance(response, dict):
            raise AgentError("tool_receipt_object_required")
        sanitized = redact(response)
        if len(canonical(sanitized)) > 1048576:
            raise AgentError("tool_receipt_too_large")
        with self.database.connect() as db:
            task_wallet_locks(db, self.task_id)
            db.execute("SELECT id FROM cq_app.agent_tasks WHERE id=%s FOR UPDATE", (self.task_id,)).fetchone()
            attempt = db.execute("SELECT * FROM cq_app.agent_attempts WHERE id=%s AND task_id=%s FOR UPDATE",
                                 (attempt_id, self.task_id)).fetchone()
            if (not attempt or attempt["kind"] != "tool" or attempt["run_id"] != self.run_id
                    or attempt["epoch"] != self.epoch or attempt["sent_at"] is None
                    or attempt["status"] not in {"sent", "unknown"}):
                raise AgentError("tool_receipt_attempt_identity_invalid")
            if attempt["response"] is not None:
                if attempt["response"] != sanitized:
                    raise AgentError("receipt_conflict")
                return
            db.execute("UPDATE cq_app.agent_attempts SET response=%s WHERE id=%s AND task_id=%s",
                       (Jsonb(sanitized), attempt_id, self.task_id))
            event(db, self.task_id, "tool_partial_receipt_observed", {
                "attempt_id": str(attempt_id), "receipt_sha256": digest(sanitized),
                "outcome": "observed_not_settled"})

    def unknown(self, attempt_id: UUID, error_code: str) -> None:
        with self.database.connect() as db:
            task_wallet_locks(db, self.task_id)
            db.execute("SELECT * FROM cq_app.agent_tasks WHERE id=%s FOR UPDATE", (self.task_id,)).fetchone()
            changed = db.execute("UPDATE cq_app.agent_attempts SET status='unknown',error_code=%s WHERE id=%s AND task_id=%s AND status IN ('reserved','sent') RETURNING *", (error_code, attempt_id, self.task_id)).fetchone()
            if changed:
                if free_read_attempt(changed):
                    db.execute("UPDATE cq_app.agent_attempts SET cost_ncny=0 WHERE id=%s", (attempt_id,))
                    db.execute("""UPDATE cq_app.agent_tasks SET state='failed',error_code='failed_read_outcome_unknown',
                        active_seconds=active_seconds+CASE WHEN running_since IS NULL THEN 0 ELSE greatest(0,extract(epoch FROM clock_timestamp()-running_since)) END,
                        running_since=NULL,lease_until=NULL WHERE id=%s""", (self.task_id,))
                    db.execute("UPDATE cq_app.agent_commands SET state='done' WHERE task_id=%s AND state='running'", (self.task_id,))
                    db.execute("UPDATE cq_app.agent_runs SET state='failed',ended_at=clock_timestamp() WHERE task_id=%s AND state='running'", (self.task_id,))
                else:
                    db.execute("UPDATE cq_app.agent_tasks SET state='reconciling',error_code='attempt_outcome_unknown' WHERE id=%s", (self.task_id,))
                event(db, self.task_id, "attempt_unknown", {"attempt_id": str(attempt_id), "error_code": error_code,
                    "reservation_retained": True, "zero_cash_read_terminal": free_read_attempt(changed), "automatic_retry": False})

    def finish(self, state: str, *, result: dict | None = None, checkpoint_id: str | None = None, error: str | None = None) -> None:
        with self.database.connect() as db:
            task = self.lock_task(db, dispatch=False)
            run = db.execute("SELECT ended_at FROM cq_app.agent_runs WHERE id=%s AND task_id=%s AND epoch=%s FOR UPDATE",
                (self.run_id, self.task_id, self.epoch)).fetchone()
            if run is None:
                raise AgentError("execution_run_not_found")
            # The graph commits its outcome before transport cleanup. Cleanup
            # failures cannot replace that committed result or duplicate history.
            if run["ended_at"] is not None:
                return
            if task["cancellation_requested"]:
                state, result = "cancelled", None
            unresolved = db.execute("SELECT * FROM cq_app.agent_attempts WHERE task_id=%s AND status IN ('sent','unknown')", (self.task_id,)).fetchall()
            if unresolved:
                if all(free_read_attempt(attempt) for attempt in unresolved):
                    state, result, error = "failed", None, "failed_read_outcome_unknown"
                else:
                    state, result = "reconciling", None
            if task["elapsed"] > task["profile"]["active_seconds"] and state not in {"reconciling", "cancelled"}:
                # Final publication is bounded too: receipts settle first, then this gate.
                state, result, error = "failed", None, "failed_deadline"
            policy = dispatch_control.active_policy(db, self.database.dispatch_policy)
            if policy is not None:
                if task["kind"] == "qa" and dispatch_control.target_elapsed(db, task) > task["profile"]["active_seconds"] and state not in {"reconciling", "cancelled"}:
                    state, result, error = "failed", None, "failed_target_deadline"
                dispatch_control.close_waiters(db, self.task_id, self.run_id, self.epoch, "execution_" + state)
            history = task["history"]
            if result and state in {"succeeded", "partial"} and result.get("publication_status") == "validated_ai_review":
                history = (history + [{"question": task["brief"]["question"], "answer": result, "brief_revision": task["revision"]}])[-4:]
            db.execute("UPDATE cq_app.agent_tasks SET state=%s,result=%s,error_code=%s,active_seconds=%s,running_since=CASE WHEN %s='reconciling' THEN clock_timestamp() ELSE NULL END,lease_until=NULL,accepted_checkpoint=%s,history=%s WHERE id=%s", (state, Jsonb(result) if result is not None else None, error, task["elapsed"], state, checkpoint_id, Jsonb(history), self.task_id))
            db.execute("UPDATE cq_app.agent_runs SET state=%s,checkpoint_id=%s,ended_at=clock_timestamp() WHERE id=%s AND epoch=%s", (state, checkpoint_id, self.run_id, self.epoch))
            db.execute("UPDATE cq_app.agent_commands SET state=%s WHERE id=%s", ("running" if state == "reconciling" else "done", self.command["id"]))
            if state not in {"reconciling","cancelled"} and self.command["kind"] != "cancel":
                event(db,self.task_id,"conversation_turn_completed",{"version":"citequest-conversation-turn/1",
                    "command_id":str(self.command["id"]),"command_seq":self.command["seq"],
                    "brief_revision":task["revision"],"brief":task["brief"],"answer":result,
                    "state":state,"error_code":error})
            event(db, self.task_id, "task_" + state, {"state": state, "revision": task["revision"], "error_code": error})


@contextmanager
def claim(database: Database, worker_id: str, *, limits: RuntimeLimits,
          task_id: UUID | None = None) -> Iterator[Execution | None]:
    """Lease expiry is a recovery hint; only the physical session lock grants ownership."""
    development_policy = getattr(database, "development_dispatch_policy", None)
    if development_policy is not None:
        # The direct claim entry must be as narrow as Worker.run_once. A policy
        # is never a capability to consume the ordinary product queue.
        development_policy.require_task(task_id)
    campaign = getattr(database, "development_campaign_policy", None)
    if campaign is not None:
        campaign.require_configuration(limits, database.catalog_scope_policy)
        if task_id in campaign.historical_task_ids:
            raise AgentError("development_campaign_historical_Task_readonly", 409)
    with database.connect() as db:
        catalog_scope.require_schema(db, database.catalog_scope_policy)
        policy = dispatch_control.active_policy(db, database.dispatch_policy)
        kinds = None
        if campaign is not None:
            if policy is not None:
                raise AgentError("development_campaign_FIFO_binding_not_adopted", 503)
            wallet_locks(db, campaign.owner_id, campaign.project_id)
            campaign.occupied(db)
        if policy is not None:
            if database.worker_lane not in {"qa", "long"}:
                raise AgentError("dispatch_worker_lane_required", 503)
            with database.connect("checkpoint", autocommit=True) as checkpoint:
                dispatch_control.assert_checkpoint_binding(db, checkpoint)
            kinds = list(policy.short_kinds if database.worker_lane == "qa" else policy.long_kinds)
            db.execute("SELECT pg_advisory_xact_lock(%s)", (7122828494682946860,))
            for expired in dispatch_control.expire_queued(db, policy):
                event(db, expired["id"], "queue_deadline_expired", {"command_id": str(expired["command_id"]), "error_code": expired["error_code"]})
        candidate_sql = """SELECT c.* FROM cq_app.agent_commands c JOIN cq_app.agent_tasks t ON t.id=c.task_id
            WHERE ((c.state='queued' AND t.state='queued') OR (c.state='running' AND t.state='running' AND t.lease_until<clock_timestamp())
            OR (c.state='running' AND t.state='reconciling' AND NOT EXISTS
                (SELECT 1 FROM cq_app.agent_attempts a WHERE a.task_id=t.id AND a.status IN ('reserved','sent','unknown'))))
            AND (%s::uuid IS NULL OR t.id=%s)
            AND t.graph_version=ANY(%s)
            AND (%s::text[] IS NULL OR t.kind=ANY(%s::text[]))
            AND (NOT %s OR cq_app.agent_dispatch_task_unlocked(t.id))
            AND (NOT %s OR t.kind<>'research' OR NOT EXISTS(SELECT 1 FROM cq_app.agent_tasks other
                 WHERE other.kind='research' AND other.id<>t.id AND other.state IN ('running','reconciling')))
            ORDER BY c.created_at,c.seq LIMIT 16""" if policy is not None else """SELECT c.* FROM cq_app.agent_commands c JOIN cq_app.agent_tasks t ON t.id=c.task_id
            WHERE ((c.state='queued' AND t.state='queued') OR (c.state='running' AND t.state='running' AND t.lease_until<clock_timestamp())
            OR (c.state='running' AND t.state='reconciling' AND NOT EXISTS
                (SELECT 1 FROM cq_app.agent_attempts a WHERE a.task_id=t.id AND a.status IN ('reserved','sent','unknown'))))
            AND (%s::uuid IS NULL OR t.id=%s) AND t.graph_version=ANY(%s)
            ORDER BY c.created_at,c.seq LIMIT 16"""
        candidate_params = ((task_id, task_id, list(catalog_scope.graph_versions(database.catalog_scope_policy)), kinds, kinds, True, True)
            if policy is not None else (task_id, task_id, list(catalog_scope.graph_versions(database.catalog_scope_policy))))
        scope_sql, scope_params = (campaign.queue_predicate() if campaign is not None else
            (" AND NOT(t.profile ? 'development_campaign')", ()))
        candidate_sql = candidate_sql.replace("ORDER BY c.created_at,c.seq LIMIT 16", scope_sql + " ORDER BY c.created_at,c.seq LIMIT 16")
        candidates = db.execute(candidate_sql, (*candidate_params, *scope_params)).fetchall()
    for command in candidates:
        session = database.connect("checkpoint", autocommit=True)
        key = int.from_bytes(hashlib.sha256(str(command["task_id"]).encode()).digest()[:8], "big", signed=True)
        if not session.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (key,)).fetchone()["acquired"]:
            session.close()
            continue
        execution = None
        try:
            with database.connect() as db:
                task_wallet_locks(db, command["task_id"])
                task = db.execute("SELECT *,clock_timestamp() AS database_now FROM cq_app.agent_tasks WHERE id=%s FOR UPDATE", (command["task_id"],)).fetchone()
                current = db.execute("SELECT * FROM cq_app.agent_commands WHERE id=%s FOR UPDATE", (command["id"],)).fetchone()
                if current["state"] not in {"queued", "running"} or task["state"] not in {"queued", "running", "reconciling"}:
                    continue
                campaign_control.require_runtime_task(task, campaign)
                if campaign is not None:
                    campaign.occupied(db)
                policy = dispatch_control.active_policy(db, database.dispatch_policy)
                if policy is not None:
                    # Bind the actual physical owner session as well as the
                    # earlier candidate-query preflight (proxy-safe).
                    dispatch_control.assert_checkpoint_binding(db, session)
                if not dispatch_control.can_claim(db, task, policy):
                    continue
                policy_error = None
                if (not catalog_scope.current_runtime(task, database.catalog_scope_policy)
                        or digest(task["profile"]) != task["profile_hash"]):
                    policy_error = "unsupported_runtime_or_profile"
                elif not same_limits_policy(task["profile"].get("aggregate_policy"), limits):
                    policy_error = "aggregate_policy_changed_requires_explicit_migration"
                if policy_error:
                    unresolved = db.execute("SELECT * FROM cq_app.agent_attempts WHERE task_id=%s AND status IN ('reserved','sent','unknown')", (task["id"],)).fetchall()
                    blocking = any(not free_read_attempt(attempt) for attempt in unresolved)
                    db.execute("UPDATE cq_app.agent_tasks SET state=%s,error_code=%s WHERE id=%s", ("reconciling" if blocking else "failed", policy_error, task["id"]))
                    db.execute("UPDATE cq_app.agent_commands SET state='rejected' WHERE id=%s", (command["id"],))
                    event(db, task["id"], "runtime_recovery_refused", {"error_code": policy_error, "reservation_retained": bool(unresolved)})
                    continue
                try:
                    wallet_admit(db, limits, task["owner_id"], task["project_id"])
                except AgentError as exc:
                    db.execute("UPDATE cq_app.agent_tasks SET state='failed',error_code=%s WHERE id=%s", (exc.code, task["id"]))
                    db.execute("UPDATE cq_app.agent_commands SET state='rejected' WHERE id=%s", (command["id"],))
                    event(db, task["id"], "aggregate_budget_refused", {"error_code": exc.code})
                    continue
                if task["database_now"] < task["updated_at"]:
                    db.execute("UPDATE cq_app.agent_tasks SET state='failed',error_code='database_clock_regression' WHERE id=%s", (task["id"],))
                    db.execute("UPDATE cq_app.agent_commands SET state='rejected' WHERE id=%s", (command["id"],))
                    continue
                unresolved = db.execute("SELECT * FROM cq_app.agent_attempts WHERE task_id=%s AND status IN ('sent','unknown')", (task["id"],)).fetchall()
                if unresolved:
                    db.execute("UPDATE cq_app.agent_attempts SET status='unknown',error_code='recovered_dispatch_without_receipt' WHERE task_id=%s AND status='sent'", (task["id"],))
                    if all(free_read_attempt(attempt) for attempt in unresolved):
                        db.execute("UPDATE cq_app.agent_attempts SET cost_ncny=0 WHERE task_id=%s AND status='unknown'", (task["id"],))
                        db.execute("""UPDATE cq_app.agent_tasks SET state='failed',error_code='failed_read_outcome_unknown',
                            active_seconds=active_seconds+CASE WHEN running_since IS NULL THEN 0 ELSE greatest(0,extract(epoch FROM clock_timestamp()-running_since)) END,
                            running_since=NULL,lease_until=NULL WHERE id=%s""", (task["id"],))
                        db.execute("UPDATE cq_app.agent_commands SET state='done' WHERE id=%s", (command["id"],))
                        db.execute("UPDATE cq_app.agent_runs SET state='failed',ended_at=clock_timestamp() WHERE task_id=%s AND state='running'", (task["id"],))
                        event(db, task["id"], "recovery_read_outcome_unknown_terminal", {"known_cash_ncny": 0, "automatic_retry": False})
                    else:
                        db.execute("UPDATE cq_app.agent_tasks SET state='reconciling',error_code='attempt_outcome_unknown' WHERE id=%s", (task["id"],))
                        event(db, task["id"], "recovery_requires_reconciliation", {"reservation_retained": True})
                    continue
                active = task["active_seconds"] + (max(0, (task["database_now"] - task["running_since"]).total_seconds()) if task["running_since"] else 0)
                if policy is not None:
                    waiting = dispatch_control.command_wait(db, current)
                    queued_total = dispatch_control.queue_seconds(db, task["id"])
                    if ((task["kind"] == "qa" and active+queued_total+waiting >= task["profile"]["active_seconds"])
                            or task["kind"] == "research" and waiting >= policy.research_queue_seconds):
                        code = "research_queue_deadline" if task["kind"] == "research" else "target_queue_deadline"
                        dispatch_control.record_queue_once(db, current, task, policy)
                        db.execute("UPDATE cq_app.agent_tasks SET state='failed',error_code=%s WHERE id=%s", (code, task["id"]))
                        db.execute("UPDATE cq_app.agent_commands SET state='rejected' WHERE id=%s", (current["id"],))
                        event(db, task["id"], "queue_deadline_expired", {"command_id": str(current["id"]), "error_code": code})
                        continue
                    receipt = dispatch_control.record_queue_once(db, current, task, policy)
                    if receipt is not None:
                        event(db, task["id"], "command_queue_measured", {"command_id": str(current["id"]),
                            "queue_seconds": float(receipt["queue_seconds"]), "active_seconds_unchanged": True})
                if active >= task["profile"]["active_seconds"] or task["cancellation_requested"]:
                    db.execute("UPDATE cq_app.agent_tasks SET state=%s,error_code=%s,active_seconds=%s,running_since=NULL WHERE id=%s", ("cancelled" if task["cancellation_requested"] else "failed", "active_budget_exhausted", active, task["id"]))
                    db.execute("UPDATE cq_app.agent_commands SET state='rejected' WHERE id=%s", (command["id"],))
                    continue
                run_id, epoch = uuid4(), task["epoch"] + 1
                db.execute("UPDATE cq_app.agent_runs SET state='superseded',ended_at=clock_timestamp() WHERE task_id=%s AND state='running'", (task["id"],))
                db.execute("UPDATE cq_app.agent_tasks SET epoch=%s,worker_id=%s,state='running',active_seconds=%s,running_since=clock_timestamp(),lease_until=clock_timestamp()+interval '15 seconds' WHERE id=%s", (epoch, worker_id, active, task["id"]))
                db.execute("UPDATE cq_app.agent_commands SET state='running' WHERE id=%s", (command["id"],))
                db.execute("INSERT INTO cq_app.agent_runs(id,task_id,command_id,epoch,worker_id) VALUES(%s,%s,%s,%s,%s)", (run_id, task["id"], command["id"], epoch, worker_id))
                event(db, task["id"], "run_started", {"run_id": str(run_id), "epoch": epoch, "command_id": str(command["id"])})
                execution = Execution(database, task["id"], current, run_id, epoch, worker_id, session, limits)
            yield execution
            return
        finally:
            if execution:
                execution.invalid = True
            # The caller must finish and drain the graph/Saver before leaving this context.
            session.close()
    yield None
