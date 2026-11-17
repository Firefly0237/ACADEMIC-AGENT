"""Trusted development release scope, not product quarantine or a model retry.

One reviewed campaign admits normal server-created Tasks and multi-turn commands.
Versions 1/2 exclude exactly two proved stopped historical clients from activity.
Version 3 can explicitly approve additional frozen stopped-client cases. Their
full reservations and every database Task remain in the ordinary wallet.
No network, Docker, migration, model or accounting mutation occurs here.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, PrivateAttr, model_validator

from .contracts import PROFILES, AgentError, RuntimeLimits, Strict, digest
from .development_dispatch import SEMANTICS, SHA, Artifact, validate_stopped_client
from .development_dispatch_v2 import HistoricalCase, historical_occupied_for_scope, validate_historical_proofs
from .product_profiles import TaskProfilePolicy, public_profile_policy

VERSION = "citequest-development-campaign/1"
VERSION_V2 = "citequest-development-campaign/2"
VERSION_V3 = "citequest-development-campaign/3"
BINDING_KEY = "development_campaign"
APPROVAL_VERSION = "citequest-development-campaign-approval/1"
ALLOWED_PAIRS = {("qa", "qa-v5"), ("summary", "workset-v6"), ("compare", "workset-v6"),
                 ("stance", "stance-v2"), ("research", "research-v3")}
ALLOWED_PAIRS_V2 = frozenset({("qa", "qa-v6"), ("summary", "workset-v7"), ("compare", "workset-v7"),
    ("stance", "stance-v3"), ("research", "research-v4")})
ALLOWED_PAIRS_PRO = (ALLOWED_PAIRS_V2 - {("qa", "qa-v6")}) | {("qa", "qa-v7")}
ALLOWED_PAIRS_ANCHORED = (ALLOWED_PAIRS_V2 - {("qa", "qa-v6")}) | {("qa", "qa-v8")}
ALLOWED_PAIRS_REASONING = (ALLOWED_PAIRS_V2 - {("qa", "qa-v6")}) | {("qa", "qa-v9")}
ALLOWED_PAIRS_CORE = (ALLOWED_PAIRS_REASONING - {("research", "research-v4")}) | {("research", "research-v5")}


def pairs_for_version(version: str) -> set | frozenset:
    if version == VERSION:
        return ALLOWED_PAIRS
    if version in {VERSION_V2, VERSION_V3}:
        return ALLOWED_PAIRS_V2
    raise AgentError("development_campaign_version_not_supported", 503)


class DatabaseIdentity(Strict):
    database_name: str = Field(min_length=1, max_length=100)
    database_oid: int = Field(ge=1, le=4294967295)
    # Root observes the cluster identity in the signed read-only ledger artifact.
    # Runtime roles do not acquire pg_monitor privileges just for this feature.
    postgres_system_identifier: str = Field(pattern=r"^[0-9]{1,20}$")
    readonly_identity_proof: Artifact


class DevelopmentCampaign(Strict):
    version: Literal[VERSION, VERSION_V2, VERSION_V3] = VERSION
    campaign_id: UUID
    scope: Literal["isolated_control", "development_main"]
    owner_id: UUID
    project_id: UUID
    database: DatabaseIdentity
    limits: RuntimeLimits
    catalog_scope_policy: Literal["task-catalog-snapshot/1"] = "task-catalog-snapshot/1"
    task_profile_policy: TaskProfilePolicy
    task_profile_policy_sha256: SHA
    historical: tuple[HistoricalCase, HistoricalCase]
    quarantined_cases: list[HistoricalCase] = Field(default_factory=list, exclude_if=lambda value: not value)
    semantics: Literal[SEMANTICS] = SEMANTICS
    root_approval: Artifact
    _file_sha256: str | None = PrivateAttr(default=None)
    _proofs: dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def frozen_scope(self) -> DevelopmentCampaign:
        if ((self.scope == "development_main" and self.database.database_name != "citequest_s03")
                or (self.scope == "isolated_control" and not self.database.database_name.startswith("citequest_s08_fixture"))):
            raise ValueError("explicit matching Main/fixture database required")
        outcomes = [case.outcome for case in self.historical]
        if (len({case.task_id for case in outcomes}) != 2 or len({case.attempt_id for case in outcomes}) != 2
                or any(case.owner_id != self.owner_id for case in outcomes)):
            raise ValueError("exactly two distinct stopped historical outcomes of the frozen owner required")
        if self.version != VERSION_V3 and self.quarantined_cases:
            raise ValueError("additional stopped-client isolation requires explicit campaign version3")
        approved = [case.outcome for case in (*self.historical, *self.quarantined_cases)]
        if (len({case.task_id for case in approved}) != len(approved)
                or len({case.attempt_id for case in approved}) != len(approved)
                or any(case.owner_id != self.owner_id for case in approved)):
            raise ValueError("every approved isolation must have a distinct Task/attempt and the frozen owner")
        limits = self.limits
        if (limits.mode != "bounded" or limits.campaign_cash_ncny != 500000000000
                or limits.campaign_prior_reserved_ncny != 4212736000 or limits.model_dispatch_concurrency != 1
                or limits.owner_active_tasks != 2 or limits.owner_cash_ncny != 100000000000
                or limits.project_cash_ncny != 500000000000):
            raise ValueError("original bounded campaign, liability, owner/project caps and known-client slot required")
        pairs = {(item.kind, item.profile_id) for item in self.task_profile_policy.profiles}
        registered_pairs = (pairs == pairs_for_version(self.version)
            or (self.version == VERSION_V3 and pairs in (ALLOWED_PAIRS_PRO, ALLOWED_PAIRS_ANCHORED, ALLOWED_PAIRS_REASONING, ALLOWED_PAIRS_CORE)))
        if not registered_pairs or any(item.validation_state != "development" for item in self.task_profile_policy.profiles):
            raise ValueError("exactly the five reviewed development kind/profile pairs required")
        if self.version in {VERSION_V2, VERSION_V3} and any(item.review_transport_protocol != "function1"
                and not (self.version == VERSION_V3 and item.kind == "qa"
                    and (item.profile_id, item.review_api_protocol) in {("qa-v7", "chat-json1"), ("qa-v8", "chat-json2"), ("qa-v9", "chat-json2")}
                    and item.review_transport_protocol == "json1")
                for item in self.task_profile_policy.profiles if item.kind in {"qa", "stance"}):
            raise ValueError("bound-goal QA/stance require the explicitly registered review transport")
        if public_profile_policy(self.task_profile_policy)["policy_sha256"] != self.task_profile_policy_sha256:
            raise ValueError("frozen profile policy/resources changed")
        return self

    @property
    def source_sha256(self) -> str:
        if self._file_sha256 is None:
            raise AgentError("development_campaign_must_be_loaded_from_trusted_file")
        return self._file_sha256

    @property
    def historical_task_ids(self) -> list[UUID]:
        return [case.outcome.task_id for case in (*self.historical, *self.quarantined_cases)]

    def task_binding(self) -> dict:
        return {"version": self.version, "campaign_id": str(self.campaign_id), "policy_sha256": self.source_sha256}

    def public_binding(self) -> dict:
        return {**self.task_binding(), "semantics": SEMANTICS, "known_client_concurrency": 1,
            "owner_active_tasks_limit": 2, "historical_outcomes": len(self.historical_task_ids),
            "remote_physical_concurrency_asserted": False, "unknown_financial_record_changed": False,
            "quality_acceptance_asserted": False}

    def require_configuration(self, limits: RuntimeLimits, policy: str | None,
                              profiles: TaskProfilePolicy | None = None) -> None:
        _ = self.source_sha256
        if limits.model_dump() != self.limits.model_dump() or policy != self.catalog_scope_policy:
            raise AgentError("development_campaign_runtime_policy_mismatch", 503)
        if profiles is not None and profiles.model_dump() != self.task_profile_policy.model_dump():
            raise AgentError("development_campaign_profile_policy_mismatch", 503)

    def require_worker(self, task_id: UUID | None, mode: str) -> None:
        _ = self.source_sha256
        if mode != ("live" if self.scope == "development_main" else "fixture"):
            raise AgentError("development_campaign_worker_mode_mismatch")
        if task_id in self.historical_task_ids:
            raise AgentError("development_campaign_historical_Task_readonly", 409)

    def require_owner(self, owner: UUID, project: UUID) -> None:
        if (owner, project) != (self.owner_id, self.project_id):
            raise AgentError("development_campaign_owner_project_not_enabled", 403)

    def require_task(self, task: dict) -> None:
        from . import catalog_scope

        self.require_owner(task["owner_id"], task["project_id"])
        profile = task["profile"]
        pair = (task["kind"], profile.get("profile_id"))
        if task["id"] in self.historical_task_ids or profile.get(BINDING_KEY) != self.task_binding():
            raise AgentError("development_campaign_Task_binding_required", 409)
        if pair not in {(item.kind, item.profile_id) for item in self.task_profile_policy.profiles}:
            raise AgentError("development_campaign_kind_profile_not_enabled", 409)
        expected = {**PROFILES[pair[1]].model_dump(mode="json"),
            "aggregate_policy": self.limits.model_dump(mode="json"),
            "catalog_scope_policy": self.catalog_scope_policy, BINDING_KEY: self.task_binding()}
        expected.update(self.task_profile_policy.runtime_metadata(*pair))
        if (profile != expected or digest(profile) != task["profile_hash"]
                or task["graph_version"] != catalog_scope.graph_version(*pair, self.catalog_scope_policy,
                    profile.get("review_transport_protocol"), profile.get("qa_execution_protocol"), profile.get("review_api_protocol"), profile.get("structured_output_protocol"), profile.get("answer_revision_protocol"), profile.get("qa_context_protocol"), profile.get("qa_content_protocol"), profile.get("evidence_navigation_protocol"))):
            raise AgentError("development_campaign_Task_profile_changed", 409)

    def occupied(self, db: Any) -> list[dict]:
        """Under the wallet lock, any unknown outside the exact proofs stops."""
        _ = self.source_sha256
        identity = db.execute("SELECT current_database() AS name,oid FROM pg_database WHERE datname=current_database()").fetchone()
        if not identity or (identity["name"], int(identity["oid"])) != (self.database.database_name, self.database.database_oid):
            raise AgentError("development_campaign_database_identity_changed", 503)
        return historical_occupied_for_scope(db, database_name=self.database.database_name,
            project_id=self.project_id, historical=(*self.historical, *self.quarantined_cases), proofs_loaded=bool(self._proofs))

    def active_tasks(self, db: Any, limits: RuntimeLimits, owner: UUID, project: UUID) -> int:
        self.require_configuration(limits, self.catalog_scope_policy)
        self.require_owner(owner, project)
        self.occupied(db)
        # Only exact, explicitly approved stopped Tasks are excluded from activity.
        # Old money and other unbound/queued Tasks still occupy their full limits.
        return int(db.execute("""SELECT count(*) AS active FROM cq_app.agent_tasks WHERE owner_id=%s
            AND state IN ('queued','running','reconciling') AND NOT(id=ANY(%s::uuid[]))""",
            (owner, self.historical_task_ids)).fetchone()["active"])

    def model_occupied(self, db: Any, task: dict, command: dict, request: dict, logical_key: str,
                       *, current_attempt_id: UUID | None = None) -> list[dict]:
        self.require_task(task)
        if (command.get("task_id") != task["id"] or command.get("seq") != task["command_seq"]
                or command.get("kind") not in {"start", "resume", "followup"}
                or not logical_key.startswith(str(command.get("id")) + ":")
                or request.get("model") != task["profile"]["model_alias"]):
            raise AgentError("development_campaign_command_or_model_binding_changed")
        initial = db.execute("""SELECT operation_id,payload,payload_hash FROM cq_app.agent_commands
            WHERE task_id=%s AND seq=1 AND kind='start'""", (task["id"],)).fetchone()
        if (not initial or initial["operation_id"] != task["operation_id"]
                or initial["payload_hash"] != task["create_hash"] or digest(initial["payload"]) != task["create_hash"]):
            raise AgentError("development_campaign_initial_payload_changed")
        occupied = self.occupied(db)
        if current_attempt_id is not None:
            current = [row for row in occupied if row["id"] == current_attempt_id]
            if (len(current) != 1 or current[0]["task_id"] != task["id"] or current[0]["status"] != "reserved"
                    or current[0]["request_hash"] != digest(request)):
                raise AgentError("development_campaign_current_reservation_changed")
        return [row for row in occupied if row["id"] != current_attempt_id]

    def queue_predicate(self) -> tuple[str, tuple]:
        """Scope is in SQL before ORDER BY/LIMIT; never filter a truncated queue."""
        pairs = sorted((item.kind, item.profile_id) for item in self.task_profile_policy.profiles)
        clause = " AND t.owner_id=%s AND t.project_id=%s AND t.profile->'development_campaign'=%s::jsonb AND ("
        clause += " OR ".join("(t.kind=%s AND t.profile->>'profile_id'=%s)" for _pair in pairs) + ")"
        params = (self.owner_id, self.project_id, json.dumps(self.task_binding()),
                  *(value for pair in pairs for value in pair))
        return clause, params


def approval_body(policy: DevelopmentCampaign) -> dict:
    return {"version": APPROVAL_VERSION, "decision": "approve_scoped_development_campaign",
        "campaign_spec_sha256": digest(policy.model_dump(mode="json", exclude={"root_approval"})),
        "approved_by": "Root" if policy.scope == "development_main" else "EXPLICIT_ISOLATED_CONTROL",
        "Main_dispatch_authorized": policy.scope == "development_main",
        "old_Task_resume_authorized": False, "old_unknown_financial_change_authorized": False,
        "new_unknown_stops_admission_claim_and_wire": True,
        "per_Task_approval_required": False, "remote_physical_concurrency_asserted": False}


def validate_quarantined_proofs(policy: DevelopmentCampaign) -> None:
    """Check each separately approved capture against its exact unknown prefix.

    The original two proofs retain their original two-unknown observation. New
    cases have their own capture and stopped-worker identity, never an automatic
    growing allowlist. Root approval separately binds the entire ordered spec.
    """
    prefix = [str(case.outcome.attempt_id) for case in policy.historical]
    for case in policy.quarantined_cases:
        stop, ledger = case.stopped_client_proof.read(), case.ledger_proof.read()
        validate_stopped_client(stop)
        outcome = case.outcome.model_dump(mode="json")
        prefix.append(str(case.outcome.attempt_id))
        if (any(ledger.get(key) != value for key, value in outcome.items())
                or ledger.get("approved_unknown_attempt_ids") != prefix
                or ledger.get("global_unknown_models") != len(prefix)
                or ledger.get("status") != "unknown" or "usage" not in ledger or ledger["usage"] is not None
                or "cost_ncny" not in ledger or ledger["cost_ncny"] is not None or ledger.get("task_state") != "reconciling"
                or ledger.get("project_id") != str(policy.project_id)
                or ledger.get("response_missing") is not (case.outcome.response_sha256 is None)
                or digest(ledger.get("profile")) != case.outcome.profile_hash):
            raise AgentError("development_campaign_quarantine_ledger_changed")
        if (any(stop.get(key) != outcome[key] for key in ("task_id", "attempt_id", "run_id"))
                or not isinstance(ledger.get("worker_id"), str) or not ledger["worker_id"]
                or stop.get("worker_id") != ledger["worker_id"]):
            raise AgentError("development_campaign_quarantine_client_binding_changed")
        try:
            started, finished, sent, observed = (datetime.fromisoformat(value.replace("Z", "+00:00"))
                for value in (stop["state"]["StartedAt"], stop["state"]["FinishedAt"],
                              ledger["attempt_sent_at"], ledger["observed_at"]))
            if any(value.tzinfo is None for value in (started, finished, sent, observed)) or not started <= sent <= finished <= observed:
                raise ValueError
        except (KeyError, TypeError, AttributeError, ValueError):
            raise AgentError("development_campaign_quarantine_capture_time_invalid") from None


def load_campaign(path, expected_sha256: str) -> DevelopmentCampaign:
    if path.is_symlink():
        raise AgentError("development_campaign_symlink_forbidden")
    value = Artifact(path=str(path.resolve()), sha256=expected_sha256).read()
    policy = DevelopmentCampaign.model_validate_json(json.dumps(value), strict=True)
    validate_historical_proofs(policy.historical)
    validate_quarantined_proofs(policy)
    identity = policy.database.readonly_identity_proof.read()
    if identity != {"database_name": policy.database.database_name, "database_oid": policy.database.database_oid,
                    "postgres_system_identifier": policy.database.postgres_system_identifier}:
        raise AgentError("development_campaign_database_proof_changed")
    approval = policy.root_approval.read()
    if approval != approval_body(policy):
        raise AgentError("development_campaign_Root_approval_binding_changed")
    policy._file_sha256 = expected_sha256
    policy._proofs = {"approval": approval, "database_identity": identity}
    return policy


def require_runtime_task(task: dict, campaign: DevelopmentCampaign | None) -> None:
    if campaign is not None:
        campaign.require_task(task)
    elif task["profile"].get(BINDING_KEY) is not None:
        raise AgentError("development_campaign_bound_worker_required", 503)
