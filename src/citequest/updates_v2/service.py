"""Short-transaction source facts, checkpoints and monotone lineage quarantine."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid5
from xml.etree.ElementTree import ParseError

from defusedxml.common import DefusedXmlException

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from citequest.data_v2.canonical import CanonicalResolver
from citequest.materials_v2.reference_resolution import parse_crossref
from citequest.materials_v2.references import plan_reference
from citequest.updates_v2 import models as m
from citequest.updates_v2.protocols import (
    digest,
    parse_atom,
    parse_oai,
    reconciliation_state,
    timestamp,
    validate_discovery,
)
from citequest.updates_v2.schemas import ProtocolLocator, WindowInput
from citequest.updates_v2.requests import validate_request
from citequest.updates_v2.transport import update_url, provider
from citequest.v2 import models as core
from citequest.v2.domain import Actor, DomainError, IdentifierInput, WorkInput
from citequest.v2.services import Catalog
from citequest.v2.storage import CAS, StoredBlob
from citequest.v2.storage_config import app_storage

ROOT = Path(__file__).resolve().parents[3]


class UpdateService:
    def __init__(self, db: Session, actor: Actor, store: CAS, *, managed_scope_factory: Callable | None = None) -> None:
        app_storage(ROOT).assert_store(store)
        self.db, self.actor, self.store = db, actor, store
        self.catalog = Catalog(db, actor)
        self.managed_scope_factory = managed_scope_factory

    def window(self, window_id: UUID, *, writing: bool = False) -> m.UpdateWindow:
        row = self.db.get(m.UpdateWindow, window_id, with_for_update=writing, populate_existing=writing)
        if row is None or row.owner_id != self.actor.owner_id:
            raise DomainError("source_window_not_found", 404)
        self.catalog.lock_owner()
        if row.security_domain == "public":
            self.catalog.require_publisher()
        return row

    def create_window(self, request: WindowInput) -> dict:
        self.catalog.lock_owner()
        self.catalog.require_publisher()
        config = request.model_dump(mode="json", exclude={"client_operation_id"})
        if request.mode == "production":
            if self.managed_scope_factory is None or request.scope_origin_id is None:
                raise DomainError("production_managed_scope_factory_required", 409)
            proof = self.managed_scope_factory(self.db, request.source_namespace, request.scope_origin_id)
            if proof["managed_ids"] != request.managed_ids:
                raise DomainError("production_managed_scope_membership_mismatch", 409)
            config["managed_scope_proof"] = proof
        domain = self.actor.security_domain if request.mode != "production" else "public"
        scope = digest({key: value for key, value in config.items() if key not in {"lower_bound", "upper_bound"}} | {"security_domain": domain})

        def effect() -> dict:
            identifier = uuid5(self.actor.owner_id, "source-window:" + str(request.client_operation_id))
            self.db.add(m.UpdateWindow(id=identifier, owner_id=self.actor.owner_id, security_domain=domain,
                        source_namespace=request.source_namespace, lower_bound=request.lower_bound, upper_bound=request.upper_bound,
                        scope_sha256=scope, managed_ids=request.managed_ids, config=config, config_sha256=digest(config)))
            return {"window_id": str(identifier), "scope_sha256": scope, "state": "queued", "status_current_confirmed": False}

        return self.catalog.operation(request.client_operation_id, {"operation": "create_source_update_window", "request": config}, effect)

    def reserve_fetch(self, window_id: UUID, request: dict) -> dict:
        row = self.window(window_id, writing=True)
        config = WindowInput.model_validate_json(json.dumps({**{key: value for key, value in row.config.items() if key in WindowInput.model_fields}, "client_operation_id": str(UUID(int=0))}))
        validate_request(config, request, row.cursor)
        if row.cursor.get("pending_request"):
            raise DomainError("source_pending_attempt_requires_reconciliation_not_blind_retry", 409)
        if row.state in {"observed", "failed", "cancelled", "partial"}:
            raise DomainError("source_window_terminal", 409)
        if row.network_calls >= config.maximum_network_calls:
            raise DomainError("source_window_network_call_limit", 409)
        if request["kind"] == "atom_page" and self.db.scalar(select(m.UpdatePage.id).where(m.UpdatePage.window_id == window_id, m.UpdatePage.phase == "atom_page").offset(config.maximum_pages - 1).limit(1)):
            raise DomainError("source_discovery_page_limit", 409)
        now = datetime.now(UTC)
        started = row.cursor.get("budget_started_at")
        if started is None:
            started = now.isoformat()
        elapsed = max(row.active_seconds, (now - timestamp(started)).total_seconds())
        if elapsed >= config.maximum_active_seconds:
            raise DomainError("source_window_absolute_budget_exhausted", 409)
        sequence = row.page_count + 1
        row.network_calls += 1
        row.active_seconds = elapsed
        row.state = "running"
        row.cursor = {**row.cursor, "budget_started_at": started, "pending_request": request,
                      "pending_sequence": sequence, "budget_clock": "wall_elapsed_including_restart_downtime_conservative"}
        return {"sequence": sequence, "remaining_seconds": config.maximum_active_seconds - elapsed, "request_sha256": digest(request),
                'maximum_physical_hops': min(4, config.maximum_network_calls-row.network_calls+1)}

    def _blob(self, blob: StoredBlob | None) -> bytes:
        if blob is None:
            return b""
        data = self.store.read(blob)
        existing = self.db.get(core.Blob, (blob.security_domain, blob.sha256))
        if existing is None:
            self.db.add(core.Blob(security_domain=blob.security_domain, sha256=blob.sha256, size_bytes=blob.size_bytes))
            self.db.flush()
        elif existing.size_bytes != blob.size_bytes:
            raise DomainError("source_response_blob_conflict", 409)
        return data

    @staticmethod
    def _decode(data: bytes, request: dict, source: str) -> dict:
        if request["kind"] in {"atom_page", "atom_head_recheck"}:
            return parse_atom(data, expected_start=request["start"], maximum_records=request["page_size"])
        if request["kind"] == "oai_managed":
            return parse_oai(data, expected_identifier=request["native_id"])
        if request["kind"] == "oai_list":
            return parse_oai(data, expected_verb="ListRecords")
        if request["kind"] == "crossref_known":
            metadata = parse_crossref(data, plan_reference(request["native_id"]), request["url"])
            record = {**metadata, "native_id": request["native_id"], "source_revision": hashlib.sha256(data).hexdigest(),
                      "record_ordinal": 0, "state": "observed_metadata", "updated": None, "status_asof": None, "protocol": "Crossref"}
            record["record_sha256"] = digest({key: value for key, value in record.items() if key != "record_ordinal"})
            return {"protocol": "Crossref", "records": [record]}
        raise DomainError("source_request_kind_unsupported", 422)

    def append_page(self, window_id: UUID, sequence: int, request: dict, blob: StoredBlob | None, response: dict) -> dict:
        row = self.window(window_id, writing=True)
        expected = self.db.scalar(select(m.UpdatePage).where(m.UpdatePage.window_id == window_id, m.UpdatePage.sequence == sequence))
        if expected:
            if expected.request_sha256 != digest(request) or expected.raw_sha256 != (blob.sha256 if blob else None):
                raise DomainError("source_page_replay_conflict", 409)
            return {"page_id": str(expected.id), "replayed": True}
        if sequence != row.page_count + 1 or row.cursor.get("pending_sequence") != sequence or digest(row.cursor.get("pending_request")) != digest(request):
            raise DomainError("source_fetch_not_reserved_or_wrong_sequence", 409)
        data = self._blob(blob)
        additional_hops = max(0, len(response.get('network_receipts') or []) - 1)
        if row.network_calls + additional_hops > row.config['maximum_network_calls']:
            raise DomainError('source_physical_HTTP_hop_count_exceeds_reserved_budget', 409)
        row.network_calls += additional_hops
        error, parsed = response.get("error"), None
        accepted = response.get("status") == 200 and response.get("complete") is True and error is None
        mime = str(response.get("content_type", "")).split(";", 1)[0].lower().strip()
        allowed_mime = {"application/json", "application/vnd.crossref-api-message+json"} if row.source_namespace == "crossref" else {"application/atom+xml", "application/xml", "text/xml"}
        if accepted and mime not in allowed_mime:
            accepted, error = False, "source_metadata_MIME_rejected"
        if accepted and row.config["mode"] != "synthetic_fixture":
            update_url(request["url"])
            receipts = response.get("network_receipts") or []
            if not 1 <= len(receipts) <= 4 or receipts[0]['url'] != request['url']:
                raise DomainError("source_response_has_no_settled_exact_network_receipt", 409)
            seen = set()
            for number, receipt in enumerate(receipts):
                url = update_url(receipt['url'])
                gate = receipt.get('gate')
                attempt = self.db.get(m.NetworkAttempt, UUID(gate['attempt_id'])) if gate else None
                outcome = (attempt.receipt or {}).get('outcome', {}) if attempt else {}
                if (attempt is None or attempt.id in seen or attempt.state != 'settled' or attempt.url != url
                        or provider(url) != provider(request['url']) or outcome.get('tls_hostname') != urlsplit(url).hostname
                        or outcome.get('tls_certificate_verified') is not True or attempt.receipt.get('source_window_id') != str(window_id)
                        or attempt.receipt.get('source_sequence') != sequence):
                    raise DomainError('source_response_has_no_settled_exact_network_receipt', 409)
                seen.add(attempt.id)
                if number + 1 < len(receipts):
                    if outcome.get('status') not in {301, 302, 303, 307, 308} or receipt.get('redirect_to') != receipts[number+1]['url']:
                        raise DomainError('source_network_redirect_chain_mismatch', 409)
                elif outcome.get('status') != 200 or outcome.get('body_sha256') != (blob.sha256 if blob else None) or response.get('url') != url:
                    raise DomainError('source_final_network_body_or_URL_mismatch', 409)
        if accepted:
            try:
                parsed = self._decode(data, request, row.source_namespace)
            except (ValueError, TypeError, ParseError, DefusedXmlException) as exc:
                error = "source_schema_invalid:" + type(exc).__name__
        page = m.UpdatePage(id=uuid5(window_id, "page:" + str(sequence)), window_id=window_id, sequence=sequence,
                    phase=request["kind"], request_sha256=digest(request), request=request,
                    raw_security_domain=blob.security_domain if blob else row.security_domain, raw_sha256=blob.sha256 if blob else None,
                    body_complete=response.get("complete") is True, receipt=response, parsed=parsed)
        self.db.add(page)
        self.db.flush()
        records = (parsed or {}).get("records", [])
        if request["kind"] == "oai_list":
            from datetime import date
            from .oai_cursor import state_for
            config = WindowInput.model_validate_json(json.dumps({**{key: value for key, value in row.config.items() if key in WindowInput.model_fields}, "client_operation_id": str(UUID(int=0))}))
            fixed = state_for(config, row.cursor)
            for record in records:
                stamp = record["datestamp"]
                day = date.fromisoformat(stamp) if len(stamp) == 10 else timestamp(stamp).astimezone(UTC).date()
                if not fixed.lower <= day <= fixed.upper:
                    raise DomainError("OAI_record_outside_fixed_day_window", 409)
        if parsed and parsed.get("error") == "idDoesNotExist":
            records = [{"native_id": request["native_id"], "state": "quarantine_missing", "source_revision": None,
                        "record_ordinal": 0, "title": None, "abstract": None, "year": None,
                        "record_sha256": digest({"error": "idDoesNotExist", "native_id": request["native_id"]}), "status_asof": None}]
        # Head rechecks are integrity observations, not duplicate source facts.
        if request["kind"] == "atom_head_recheck":
            records = []
        if row.record_count + len(records) > row.config["maximum_records"]:
            raise DomainError("source_window_record_cap", 409)
        for record in records:
            ordinal = record["record_ordinal"]
            is_error = parsed.get("error") == "idDoesNotExist"
            locator = ProtocolLocator(kind="oai_error" if is_error else "atom_entry" if request["kind"] == "atom_page" else "oai_record" if request["kind"] in {"oai_managed", "oai_list"} else "crossref_work",
                      raw_response_sha256=blob.sha256, record_ordinal=None if is_error else ordinal, identifier=record["native_id"], source_url=request["url"],
                      coordinate_system="parsed_XML_or_JSON_record_identifier_and_ordinal_not_PDF_or_byte_offsets").model_dump(mode="json")
            metadata = {**record, "title": record.get("title"), "abstract": record.get("abstract"), "year": record.get("year")}
            observation_id = uuid5(page.id, "record:" + str(ordinal))
            work_id = None
            observation_state = record["state"]
            in_window = request["kind"] != "atom_page" or row.lower_bound <= timestamp(record["updated"]) <= row.upper_bound
            if not in_window:
                observation_state = "fence_only_observation"
            if request["kind"] == "oai_managed":
                prior_atom = self.db.scalars(select(m.UpdateObservation).where(m.UpdateObservation.window_id == window_id,
                    m.UpdateObservation.native_id == record["native_id"])).all()
                candidate_atom = next((item.metadata_payload for item in reversed(prior_atom) if item.metadata_payload.get("protocol") == "Atom"), None)
                reconciled = reconciliation_state(candidate_atom, parsed)
                if reconciled.startswith("quarantine_"):
                    observation_state = reconciled
            if in_window and metadata.get("title") and 0 < len(metadata["title"].strip()) <= 500 and not observation_state.startswith("quarantine_"):
                authors = metadata.get("authors")
                if authors is not None and (not authors or any(type(a) is not str or not a.strip() or len(a) > 500 for a in authors)):
                    authors = None
                namespace = "arxiv" if row.source_namespace == "arxiv" else "doi"
                candidate = WorkInput(title=metadata["title"], authors=authors, year=metadata.get("year"), identifiers=[IdentifierInput(namespace=namespace, raw=record["native_id"])],
                                      client_operation_id=uuid5(observation_id, "canonical-work"))
                work_id = CanonicalResolver(self.catalog).resolve(row.source_namespace, record["native_id"], candidate, public=row.security_domain == "public").id
            projection_sha = digest({"locator": locator, "record_sha256": record["record_sha256"], "metadata_payload": metadata})
            observation = m.UpdateObservation(id=observation_id, window_id=window_id, page_id=page.id, security_domain=row.security_domain,
                      source_namespace=row.source_namespace, native_id=record["native_id"], source_revision=record.get("source_revision"), record_ordinal=ordinal,
                      record_sha256=record["record_sha256"], projection_sha256=projection_sha, locator=locator, metadata_payload=metadata, state=observation_state,
                      metadata_updated_at=timestamp(record["updated"]) if record.get("updated") and "T" in record["updated"] else None,
                      verified_status_asof=None, work_id=work_id)
            self.db.add(observation)
            self.db.flush()
            if observation.state.startswith("quarantine_"):
                self.db.execute(insert(m.SourceDenial).values(security_domain=row.security_domain, source_namespace=row.source_namespace,
                                native_id=observation.native_id, source_revision="*", observation_id=observation_id,
                                reason=observation.state, active=True, proof={"raw_response_sha256": blob.sha256, "scope": "this_source_only", "global_Work_retraction_claimed": False})
                                .on_conflict_do_nothing())
            envelope = {"schema_version": "cq.outbox.payload/1", "entity_type": "SourceUpdateObservation", "entity_schema": "cq.protocol-metadata/1",
                        "entity_id": str(observation_id), "entity_revision": 1, "security_domain": row.security_domain,
                        "operation": "quarantine" if observation.state.startswith("quarantine_") else "stage",
                        "data": {"observation_id": str(observation_id), "native_id": observation.native_id, "source_namespace": row.source_namespace,
                                 "projection_sha256": projection_sha, "state": observation.state, "publication_eligible": False}}
            self.db.add(core.Outbox(entity_id=str(observation_id), entity_revision=1, entity_type="SourceUpdateObservation", security_domain=row.security_domain,
                        operation=envelope["operation"], schema_version="cq.outbox.payload/1", payload=envelope, payload_sha256=digest(envelope),
                        idempotency_key="SourceUpdateObservation:" + str(observation_id) + ":1:" + envelope["operation"]))
        row.page_count += 1
        row.record_count += len(records)
        row.active_seconds = max(row.active_seconds, (datetime.now(UTC) - timestamp(row.cursor["budget_started_at"])).total_seconds())
        if request["kind"] == "oai_list" and parsed is not None and error is None and accepted:
            from .oai_cursor import state_for
            config = WindowInput.model_validate_json(json.dumps({**{key: value for key, value in row.config.items() if key in WindowInput.model_fields}, "client_operation_id": str(UUID(int=0))}))
            state = state_for(config, row.cursor)
            state.accept(parsed)
            if len(state.records) > config.maximum_records:
                raise DomainError("OAI_distinct_record_cap", 409)
            row.cursor = {**row.cursor, "oai": state.cursor()}
        row.cursor = {**row.cursor, "pending_request": None, "pending_sequence": None}
        allowed_errors = {None, "idDoesNotExist"} | ({"badResumptionToken", "noRecordsMatch"} if request["kind"] == "oai_list" else set())
        if error or not accepted or (parsed and parsed.get("error") not in allowed_errors):
            row.state, row.error = "partial", {"code": error or "source_HTTP_or_protocol_error", "status": response.get("status"), "parsed_error": (parsed or {}).get("error")}
        return {"page_id": str(page.id), "records": len(records), "state": row.state, "replayed": False}

    def seal(self, window_id: UUID) -> dict:
        row = self.window(window_id, writing=True)
        existing = self.db.scalar(select(m.UpdateSeal).where(m.UpdateSeal.window_id == window_id))
        if existing:
            return {"seal_id": str(existing.id), "publication_eligible": False, "replayed": True}
        if row.state != "running" or row.cursor.get("pending_request"):
            raise DomainError("incomplete_source_window_cannot_seal", 409)
        pages = list(self.db.scalars(select(m.UpdatePage).where(m.UpdatePage.window_id == window_id).order_by(m.UpdatePage.sequence)))
        if not pages or any(not p.body_complete or p.parsed is None for p in pages):
            raise DomainError("incomplete_source_pages", 409)
        if row.config.get("protocol_profile") == "oai-fixed-window/1":
            from .oai_enumeration import OAIEnumeration
            state = OAIEnumeration.restore(row.cursor.get("oai", {}))
            if not state.finished or state.awaiting_response:
                raise DomainError("complete_OAI_fixed_window_required", 409)
            complete_pages = [p for p in pages if p.phase == "oai_list" and p.request.get("oai_epoch") == state.restarts]
            if not complete_pages:
                raise DomainError("OAI_successful_epoch_pages_required", 409)
            eligible_ids = set(state.records) | set(row.managed_ids)
            managed = {p.request["native_id"] for p in pages if p.phase == "oai_managed"}
        elif row.source_namespace == "arxiv":
            discovery = [p.parsed for p in pages if p.phase == "atom_page"]
            head = [p.parsed for p in pages if p.phase == "atom_head_recheck"]
            if len(head) != 1:
                raise DomainError("source_head_fence_missing", 409)
            discovered = validate_discovery(discovery, head[0], lower=row.lower_bound, upper=row.upper_bound, page_size=row.config["page_size"], overlap=row.config["overlap"])
            eligible_ids = {record["native_id"] for record in discovered} | set(row.managed_ids)
            managed = {p.request["native_id"] for p in pages if p.phase == "oai_managed"}
        else:
            managed = {p.request["native_id"] for p in pages if p.phase == "crossref_known"}
            eligible_ids = set(row.managed_ids)
        if managed != set(row.managed_ids):
            raise DomainError("frozen_managed_scope_not_fully_accounted", 409)
        observations = list(self.db.scalars(select(m.UpdateObservation).join(m.UpdatePage, m.UpdateObservation.page_id == m.UpdatePage.id)
                       .where(m.UpdateObservation.window_id == window_id).order_by(m.UpdatePage.sequence, m.UpdateObservation.record_ordinal)))
        denied = set(self.db.scalars(select(m.SourceDenial.native_id).where(m.SourceDenial.security_domain == row.security_domain,
                     m.SourceDenial.source_namespace == row.source_namespace, m.SourceDenial.active.is_(True))))
        current = {}
        for obs in observations:
            if row.config.get("protocol_profile") == "oai-fixed-window/1":
                page = next(p for p in pages if p.id == obs.page_id)
                if page.phase == "oai_list" and page.request.get("oai_epoch") != state.restarts:
                    continue
            if (obs.native_id in eligible_ids and obs.native_id not in denied and obs.work_id
                    and obs.metadata_payload.get("title") and obs.state != "fence_only_observation" and not obs.state.startswith("quarantine_")):
                key = (obs.source_namespace, obs.native_id)
                # Atom retains explicit versions; OAI dates/metadata never overwrite it by arrival order.
                if key not in current or obs.metadata_payload.get("protocol") == "Atom":
                    current[key] = obs
        members = [{"source_namespace": ns, "native_id": native, "observation_id": str(obs.id), "projection_sha256": obs.projection_sha256}
                   for (ns, native), obs in sorted(current.items())]
        membership_sha = digest(members)
        payload = {"window_id": str(window_id), "scope_sha256": row.scope_sha256, "membership_sha256": membership_sha,
                   "lower_bound": row.lower_bound.isoformat(), "upper_bound": row.upper_bound.isoformat(),
                   "managed_ids": row.managed_ids, "state": "observed_window_complete", "status_asof_proofs": 0,
                   "current_status_confirmed": False, "publication_eligible": False, "complete_global_or_latest_sync_claimed": False}
        payload["membership_scope"] = "this_bounded_window_observations_not_full_corpus"
        if row.config.get("protocol_profile") == "oai-fixed-window/1":
            payload.update(protocol_profile="oai-fixed-window/1", observed_watermark=row.upper_bound.isoformat(),
                successful_OAI_epoch=state.restarts, OAI_requests_cumulative=state.requests)
        seal_id = uuid5(window_id, "observed-seal")
        self.db.add(m.UpdateSeal(id=seal_id, window_id=window_id, security_domain=row.security_domain, source_namespace=row.source_namespace,
                    scope_sha256=row.scope_sha256, membership_sha256=membership_sha, payload=payload, seal_sha256=digest(payload), publication_eligible=False))
        self.db.flush()
        for member in members:
            self.db.add(m.SealMember(seal_id=seal_id, source_namespace=member["source_namespace"], native_id=member["native_id"],
                                    observation_id=UUID(member["observation_id"]), projection_sha256=member["projection_sha256"]))
        checkpoint = self.db.get(m.Checkpoint, (row.security_domain, row.source_namespace, row.scope_sha256, "discovery"), with_for_update=True)
        if checkpoint and timestamp(checkpoint.value["upper_bound"]) > row.upper_bound:
            raise DomainError("source_checkpoint_cannot_move_backwards", 409)
        value = {"upper_bound": row.upper_bound.isoformat(), "observed_window_complete": True, "current_status_confirmed": False}
        if checkpoint:
            checkpoint.seal_id, checkpoint.value = seal_id, value
        else:
            self.db.add(m.Checkpoint(security_domain=row.security_domain, source_namespace=row.source_namespace, scope_sha256=row.scope_sha256,
                                    kind="discovery", seal_id=seal_id, value=value))
        row.state, row.phase = "observed", "finished"
        envelope = {"schema_version": "cq.outbox.payload/1", "entity_type": "SourceUpdateSeal", "entity_schema": "cq.update-seal/1",
                    "entity_id": str(seal_id), "entity_revision": 1, "security_domain": row.security_domain, "operation": "stage",
                    "data": {"seal_id": str(seal_id), "seal_sha256": digest(payload), "membership_sha256": membership_sha, "publication_eligible": False}}
        self.db.add(core.Outbox(entity_id=str(seal_id), entity_revision=1, entity_type="SourceUpdateSeal", security_domain=row.security_domain,
                    operation="stage", schema_version="cq.outbox.payload/1", payload=envelope, payload_sha256=digest(envelope),
                    idempotency_key="SourceUpdateSeal:" + str(seal_id) + ":1:stage"))
        return {"seal_id": str(seal_id), "observed_window_complete": True, "publication_eligible": False,
                "managed_status_checkpoint_advanced": False, "index_publication_changed": False, "members": len(members), "replayed": False}
