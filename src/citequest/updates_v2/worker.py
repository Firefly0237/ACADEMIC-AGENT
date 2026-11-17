"""Bounded restartable ETL: reserve, HTTP outside SQL, durable receipt, commit."""

from __future__ import annotations

import json
import os
from io import BytesIO
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from filelock import Timeout
from sqlalchemy import select

from citequest.updates_v2.context import WindowFetchContext
from citequest.updates_v2 import models as m
from citequest.updates_v2.protocols import timestamp
from citequest.updates_v2.requests import atom_request, managed_request
from citequest.updates_v2.schemas import WindowInput
from citequest.updates_v2.service import UpdateService
from citequest.updates_v2.transport import SourceFetcher
from sqlalchemy import func
from citequest.v2.domain import Actor, DomainError
from citequest.v2.storage import StoredBlob
from citequest.v2.storage_config import app_storage
from .locks import LINUX_PROFILE, WINDOWS_PROFILE, native_lock


class UpdateWorker:
    def __init__(self, factory, actor: Actor, fetcher: SourceFetcher, runtime: Path, *, managed_scope_factory=None,
                 source_gate_profile: str = WINDOWS_PROFILE) -> None:
        self.factory, self.actor, self.fetcher = factory, actor, fetcher
        self.runtime = runtime.resolve()
        self.store = app_storage().cas()
        self.managed_scope_factory = managed_scope_factory
        self.source_gate_profile = source_gate_profile
        self.stop_requested = lambda: False
        self.running = False
        (self.runtime / "window-locks").mkdir(parents=True, exist_ok=True)
        (self.runtime / "receipts").mkdir(parents=True, exist_ok=True)

    def service(self, db) -> UpdateService:
        return UpdateService(db, self.actor, self.store, managed_scope_factory=self.managed_scope_factory)

    def _next(self, db, window_id: UUID) -> tuple[dict | None, dict]:
        row = self.service(db).window(window_id)
        config = WindowInput.model_validate_json(json.dumps({**{k: v for k, v in row.config.items() if k in WindowInput.model_fields}, "client_operation_id": str(UUID(int=0))}))
        pages = list(db.scalars(select(m.UpdatePage).where(m.UpdatePage.window_id == window_id).order_by(m.UpdatePage.sequence)))
        if config.protocol_profile == "oai-fixed-window/1":
            from .oai_cursor import next_request
            request, value = next_request(config, row.cursor, datetime.now(UTC))
            row.cursor = {**row.cursor, "oai": value}
            db.flush()
            if request is not None:
                return request, row.cursor
        elif config.source_namespace == "arxiv":
            discovery = [p for p in pages if p.phase == "atom_page"]
            tail = discovery[-1].parsed["records"] if discovery else []
            crossed = bool(discovery) and (len(tail) < config.page_size or any(timestamp(r["updated"]) < config.lower_bound for r in tail))
            if not crossed:
                return atom_request(config, len(discovery) * (config.page_size - config.overlap)), row.cursor
            if not any(p.phase == "atom_head_recheck" for p in pages):
                return atom_request(config, 0, head=True), row.cursor
        observed = {p.request.get("native_id") for p in pages if p.phase in {"oai_managed", "crossref_known"}}
        remaining = sorted(set(config.managed_ids) - observed)
        return (managed_request(config.source_namespace, remaining[0]) if remaining else None), row.cursor

    def _stop(self, window_id: UUID, code: str) -> dict:
        with self.factory.begin() as db:
            row = self.service(db).window(window_id, writing=True)
            if row.state not in {"observed", "failed", "cancelled", "partial"}:
                known_hops = db.scalar(select(func.count()).select_from(m.NetworkAttempt).where(m.NetworkAttempt.receipt['source_window_id'].as_string() == str(window_id)))
                row.network_calls = max(row.network_calls, known_hops)
                row.state, row.error = "partial", {"code": code, "at": datetime.now(UTC).isoformat(), "unknown_request_not_blindly_retried": True}
        return {"window_id": str(window_id), "state": "partial", "error": code, "publication_eligible": False}

    async def run(self, window_id: UUID) -> dict:
        lock = native_lock(self.runtime / "window-locks" / (str(window_id) + ".lock"), self.source_gate_profile)
        try:
            lock.acquire()
        except Timeout as exc:
            raise DomainError("source_window_worker_busy", 409) from exc
        try:
            self.running = True
            while True:
                if self.stop_requested():
                    return {"window_id": str(window_id), "state": "draining", "publication_eligible": False}
                with self.factory.begin() as db:
                    service = self.service(db)
                    row = service.window(window_id)
                    if row.state in {"partial", "failed", "cancelled"}:
                        return {"window_id": str(window_id), "state": row.state, "error": row.error, "publication_eligible": False}
                    if row.state == "observed":
                        return service.seal(window_id)
                    pending = row.cursor.get("pending_request")
                    if pending:
                        sequence, request = row.cursor["pending_sequence"], pending
                        reserved = None
                    else:
                        request, cursor = self._next(db, window_id)
                        if request is None:
                            return service.seal(window_id)
                        reserved = service.reserve_fetch(window_id, request)
                        sequence = reserved["sequence"]
                    domain = row.security_domain
                journal = self.runtime / "receipts" / f"{window_id}-{sequence}.json"
                if reserved is None:
                    if self.source_gate_profile == LINUX_PROFILE:
                        with self.factory() as db:
                            receipt = db.get(m.FetchReceipt, (window_id, sequence))
                            saved = receipt.payload if receipt is not None else None
                            if receipt is not None:
                                from .protocols import digest
                                if digest(saved) != receipt.payload_sha256 or json.loads(receipt.payload_canonical) != saved:
                                    return self._stop(window_id, "source_durable_receipt_hash_changed")
                    else:
                        saved = json.loads(journal.read_text(encoding="utf-8")) if journal.is_file() else None
                    if saved is None:
                        return self._stop(window_id, "pending_request_without_durable_result_unknown")
                else:
                    context = WindowFetchContext.start(min(30.0, reserved["remaining_seconds"]))
                    context.source_window_id, context.source_sequence = str(window_id), sequence
                    result = await self.fetcher.fetch_page(request["url"], maximum_bytes=2 * 1024 * 1024, context=context,
                                                          accept=b"application/json" if request["kind"] == "crossref_known" else b"application/atom+xml, application/xml, text/xml",
                                                          maximum_hops=reserved['maximum_physical_hops'])
                    blob = self.store.put(domain, BytesIO(result.body), 2 * 1024 * 1024) if result.body else None
                    saved = {"window_id": str(window_id), "sequence": sequence, "request": request,
                             "blob": {"security_domain": blob.security_domain, "sha256": blob.sha256, "size_bytes": blob.size_bytes} if blob else None,
                             "response": {"url": result.url, "status": result.status, "content_type": result.content_type,
                                          "complete": result.complete, "headers": result.headers, "error": result.error, "network_receipts": context.receipts}}
                    if self.source_gate_profile == LINUX_PROFILE:
                        from .protocols import digest
                        from citequest.agents_v3.contracts import canonical
                        with self.factory.begin() as db:
                            db.add(m.FetchReceipt(window_id=window_id, sequence=sequence, payload=saved,
                                payload_sha256=digest(saved), payload_canonical=canonical(saved).decode()))
                    else:
                        temporary = journal.with_suffix(".tmp")
                        with temporary.open("x", encoding="utf-8") as output:
                            json.dump(saved, output, ensure_ascii=False, sort_keys=True)
                            output.flush()
                            os.fsync(output.fileno())
                        os.replace(temporary, journal)
                if saved["window_id"] != str(window_id) or saved["sequence"] != sequence or saved["request"] != request:
                    return self._stop(window_id, "durable_source_result_reservation_mismatch")
                with self.factory.begin() as db:
                    outcome = self.service(db).append_page(window_id, sequence, request, StoredBlob(**saved["blob"]) if saved["blob"] else None, saved["response"])
                if outcome["state"] == "partial":
                    return {"window_id": str(window_id), **outcome, "publication_eligible": False}
        except (DomainError, ValueError) as exc:
            return self._stop(window_id, getattr(exc, "code", type(exc).__name__))
        finally:
            self.running = False
            lock.release()
