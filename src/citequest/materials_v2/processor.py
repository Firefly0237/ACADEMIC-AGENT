"""One PG-coordinated ingestion; native work occurs between short transactions."""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import or_, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from citequest.v2 import models as core
from citequest.v2.domain import Actor, DomainError
from citequest.v2.services import digest
from citequest.v2.storage import CAS, StoredBlob

from . import models as m
from .artifacts import Asset, capture_assets, register_assets
from .isolation import run_isolated
from .services import MaterialService, checked
from .text import MaterialError

if TYPE_CHECKING:
    from citequest.business_storage_v3.service import BusinessStorage

    from .execution_storage import MaterialExecutionStorage

SLOT = 735041993104


@dataclass(frozen=True)
class Claim:
    ingestion_id: UUID
    owner_id: UUID
    file_id: UUID
    material_version_id: UUID
    lease_id: UUID
    sha256: str
    security_domain: str
    size_bytes: int
    media_type: str
    parser_profile_sha256: str
    chunk_profile_sha256: str
    tokenizer_profile_sha256: str
    config_sha256: str


class MaterialProcessor:
    def __init__(
        self,
        factory: sessionmaker[Session],
        store: CAS | None,
        *,
        parser_python: Path,
        root: Path,
        artifact_root: Path,
        cached_docling_by_sha: dict[str, Path] | None = None,
        budget_seconds: float = 120,
        clock: Callable[[], float] | None = None,
        runner: Callable[..., dict] | None = None,
        parser_profile: Path | None = None,
        chunk_profile: Path | None = None,
        extra_configuration_files: tuple[str, ...] = (),
        business_storage: BusinessStorage | None = None,
        execution_id: UUID | None = None,
    ) -> None:
        self.factory, self.store, self.root = factory, store, root.resolve()
        self.business_storage = business_storage
        self.assigned_execution_id = execution_id
        if business_storage is not None:
            from citequest.materials_v3.private_spool_client import PrivateSpoolBrokerParser

            if (not isinstance(runner, PrivateSpoolBrokerParser) or not isinstance(execution_id, UUID)
                    or clock is not None or cached_docling_by_sha):
                raise ValueError("business_material_requires_fixed_private_runner_real_clock_without_cache")
        elif execution_id is not None:
            raise ValueError("legacy_material_execution_identity_is_not_overridden")
        elif store is None:
            raise ValueError("explicit_material_processor_storage_required")
        self.parser_python, self.artifact_root = parser_python, artifact_root.resolve()
        if not self.artifact_root.is_relative_to(self.root):
            raise ValueError("Owned artifact directory required")
        self.cache = cached_docling_by_sha or {}  # Explicit developer-only cache, never from HTTP.
        if (
            type(budget_seconds) not in {float, int}
            or not math.isfinite(budget_seconds)
            or not 0 < budget_seconds <= 120
        ):
            raise ValueError("S04 active ingestion budget may only be lowered from 120 seconds")
        self.budget_seconds, self.clock = float(budget_seconds), clock or time.monotonic
        # Trusted construction-time adapters only; HTTP inputs cannot select a runner/profile.
        self.runner = runner
        self.parser_profile = (parser_profile or self.root / "runtime/s04/worker-profile.json").resolve()
        self.chunk_profile = (chunk_profile or self.root / "runtime/s04/chunk-worker-profile.json").resolve()
        for path in (
            self.parser_profile,
            self.chunk_profile,
            *(self.root / name for name in extra_configuration_files),
        ):
            if not path.resolve().is_relative_to(self.root) or not path.is_file():
                raise ValueError("Owned immutable ingestion configuration required")
        self.parser_profile_key = self.parser_profile.relative_to(self.root).as_posix()
        self.chunk_profile_key = self.chunk_profile.relative_to(self.root).as_posix()
        self.extra_configuration_files = extra_configuration_files

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self.clock()
        if not math.isfinite(remaining) or remaining <= 0:
            raise MaterialError("PRODUCT_DEADLINE_EXCEEDED")
        return remaining

    def _config(self) -> dict:
        files = [
            self.parser_profile_key,
            self.chunk_profile_key,
            "runtime/s04/tokenizer-profile.json",
            "src/citequest/materials_v2/worker.py",
            "src/citequest/materials_v2/isolation.py",
            "src/citequest/materials_v2/provenance.py",
            "src/citequest/materials_v2/chunking.py",
            "src/citequest/materials_v2/artifacts.py",
            "src/citequest/materials_v2/processor.py",
            "runtime/v2/uv.lock",
            "runtime/s04-app/uv.lock",
            *self.extra_configuration_files,
        ]
        if self.business_storage is not None:
            files.extend([
                "src/citequest/materials_v2/execution_storage.py",
                "src/citequest/materials_v3/private_spool_client.py",
                "src/citequest/materials_v3/private_spool_protocol.py",
                "src/citequest/business_storage_v3/contracts.py",
                "src/citequest/business_storage_v3/service.py",
                "src/citequest/business_storage_v3/io_process.py",
                "src/citequest/business_storage_v3/io_pg.py",
                "src/citequest/business_storage_v3/io_pg_worker.py",
                "src/citequest/business_storage_v3/io_worker.py",
                "src/citequest/business_storage_v3/io_runtime.py",
                "src/citequest/business_storage_v3/io_contracts.py",
            ])
        value = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in files}
        if self.business_storage is not None:
            value["business_storage_binding_sha256"] = self.business_storage.binding.sha256
        return value

    def _claim(self, file_id: UUID | None, started_at: datetime, deadline: float | None = None) -> Claim | dict | None:
        config = self._config()
        with self.factory.begin() as db:
            if deadline is not None:
                milliseconds = str(max(1, int(self._remaining(deadline) * 1000)))
                db.execute(text("SELECT set_config('statement_timeout', :value, true)"), {"value": milliseconds})
                db.execute(text("SELECT set_config('lock_timeout', :value, true)"), {"value": milliseconds})
            query = select(m.MaterialIngestion).where(m.MaterialIngestion.state == "queued")
            if file_id:
                query = query.where(m.MaterialIngestion.file_id == file_id)
            row = db.scalar(query.order_by(m.MaterialIngestion.created_at, m.MaterialIngestion.id).limit(1))
            if row is None:
                return None
            service = MaterialService(db, Actor(row.owner_id), self.store)
            service.catalog.lock_owner()
            db.refresh(row)
            if row.state != "queued":
                return {"status": "not_claimed", "file_id": str(row.file_id), "ingestion_state": row.state}
            try:
                file = service.catalog.file(row.file_id)
            except DomainError:
                file = db.get(core.File, row.file_id)
                service._advance(row, file, "cancelled", "current_acl_or_file_revoked")
                return {"status": "processed", "file_id": str(row.file_id), "ingestion_state": "cancelled"}
            version = db.scalar(
                select(m.MaterialVersion).where(
                    m.MaterialVersion.owner_id == row.owner_id, m.MaterialVersion.file_id == file.id
                )
            )
            if version is None or version.source_sha256 != file.sha256:
                raise DomainError("material_version_source_mismatch", 409)
            blob = db.get(core.Blob, (file.security_domain, file.sha256))
            lease_id = self.assigned_execution_id or uuid4()
            if self.assigned_execution_id is not None and db.scalar(select(m.MaterialIngestion.id).where(
                    m.MaterialIngestion.execution_id == lease_id).limit(1)) is not None:
                raise DomainError("material_execution_identity_already_used", 409)
            row.lease_id, row.lease_expires_at = lease_id, started_at + timedelta(seconds=self.budget_seconds + 60)
            row.execution_id, row.execution_started_at = lease_id, started_at
            row.deadline_at, row.execution_budget_seconds = (
                started_at + timedelta(seconds=self.budget_seconds),
                self.budget_seconds,
            )
            row.execution_outcome = "running"
            row.profile_sha256 = digest(config)
            service._advance(row, file, "validating")
            return Claim(
                row.id,
                row.owner_id,
                file.id,
                version.id,
                lease_id,
                file.sha256,
                file.security_domain,
                blob.size_bytes,
                file.media_type,
                config[self.parser_profile_key],
                config[self.chunk_profile_key],
                config["runtime/s04/tokenizer-profile.json"],
                digest(config),
            )

    def _recover_expired(self, file_id: UUID | None) -> dict | None:
        """Archive an abandoned lease; never extend, reset, or auto-retry it.

        Called only while holding the same global PG advisory worker slot as
        normal execution. The original execution identity/deadline are retained.
        Filesystem bytes published before an interrupted SQL transaction may be
        unreferenced orphans; this method does not infer or republish a result.
        """
        with self.factory.begin() as db:
            query = select(m.MaterialIngestion).where(
                m.MaterialIngestion.state.in_(["validating", "parsing"]),
                m.MaterialIngestion.lease_expires_at <= datetime.now(UTC),
            )
            if file_id is not None:
                query = query.where(m.MaterialIngestion.file_id == file_id)
            row = db.scalar(query.order_by(m.MaterialIngestion.lease_expires_at, m.MaterialIngestion.id).limit(1))
            if row is None:
                return None
            service = MaterialService(db, Actor(row.owner_id), self.store)
            service.catalog.lock_owner()
            db.refresh(row)
            if (
                row.state not in {"validating", "parsing"}
                or row.lease_expires_at is None
                or row.lease_expires_at > datetime.now(UTC)
            ):
                return None
            file = db.get(core.File, row.file_id)
            cancelled = file.deleted_at is not None
            try:
                service.catalog.file(row.file_id)
            except DomainError:
                cancelled = True
            code = "EXPIRED_MATERIAL_LEASE_NO_AUTOMATIC_RETRY"
            service._advance(row, file, "cancelled" if cancelled else "failed", code)
            return {
                "status": "recovered_abandoned_execution",
                "file_id": str(file.id),
                "ingestion_state": row.state,
                "error_code": code,
                "execution_success": False,
                "automatic_retry": False,
            }

    def _active(self, db: Session, claim: Claim) -> tuple[MaterialService, m.MaterialIngestion, core.File]:
        service = MaterialService(db, Actor(claim.owner_id), self.store)
        service.catalog.lock_owner()
        file = service.catalog.file(claim.file_id)
        row = service._ingestion(claim.file_id)
        if (
            row.lease_id != claim.lease_id
            or row.lease_expires_at is None
            or row.lease_expires_at <= datetime.now(UTC)
            or file.sha256 != claim.sha256
            or file.security_domain != claim.security_domain
        ):
            raise DomainError("stale_material_lease_or_source", 409)
        return service, row, file

    def _parsing(self, claim: Claim) -> None:
        with self.factory.begin() as db:
            service, row, file = self._active(db, claim)
            service._advance(row, file, "parsing")

    def _cancelled(self, claim: Claim) -> bool:
        with self.factory.begin() as db:
            try:
                service = MaterialService(db, Actor(claim.owner_id), self.store)
                service.catalog.file(claim.file_id)
                row = service._ingestion(claim.file_id)
                return row.state == "cancelled" or row.lease_id != claim.lease_id
            except DomainError:
                return True

    def _fail(self, claim: Claim, code: str, started: float | None = None) -> dict:
        with self.factory.begin() as db:
            service = MaterialService(db, Actor(claim.owner_id), self.store)
            service.catalog.lock_owner()
            row = db.get(m.MaterialIngestion, claim.ingestion_id)
            if started is not None:
                row.active_elapsed_seconds = max(0, self.clock() - started)
            file = db.get(core.File, claim.file_id)
            if row.lease_id != claim.lease_id:
                if row.execution_id == claim.lease_id and row.state in {"ready", "partial"}:
                    row.execution_outcome = "uncertain"
                    row.error_code = "PUBLICATION_ACK_UNCERTAIN"
                    return {
                        "status": "uncertain",
                        "file_id": str(claim.file_id),
                        "ingestion_state": row.state,
                        "error_code": row.error_code,
                        "execution_success": False,
                    }
                return {
                    "status": "processed",
                    "file_id": str(claim.file_id),
                    "ingestion_state": row.state,
                    "error_code": row.error_code,
                    "execution_success": False,
                }
            cancelled = file.deleted_at is not None
            try:
                service.catalog.file(claim.file_id)
            except DomainError:
                cancelled = True
            service._advance(
                row, file, "cancelled" if cancelled or code == "PARSER_CANCELLED" else "failed", code[:100]
            )
            row.execution_outcome = "cancelled" if row.state == "cancelled" else "failed"
            if code == "PARSER_TERMINATION_UNCONFIRMED":
                row.execution_outcome = "uncertain"
            return {
                "status": "processed",
                "file_id": str(claim.file_id),
                "ingestion_state": row.state,
                "error_code": code,
                "execution_success": False,
                "execution_outcome": row.execution_outcome,
                "active_elapsed_seconds": row.active_elapsed_seconds,
            }

    def observe_failed_stop(self, file_id: UUID) -> dict:
        """Close resource uncertainty only after genuine fixed-request readback.

        The failed/cancelled File, original error/deadline/execution and outputs
        remain untouched. This trusted-worker method does not adopt any result.
        Active/abandoned parents and publication-ACK uncertainty are ineligible.
        """
        if self.business_storage is None:
            raise DomainError("private_parser_reconciliation_not_configured", 503)
        from citequest.materials_v3.private_spool_recovery import discover

        with self.factory.begin() as db:
            row = db.scalar(select(m.MaterialIngestion).where(m.MaterialIngestion.file_id == file_id))
            if (row is None or row.execution_id is None or row.execution_outcome != "uncertain"
                    or row.state not in {"failed", "cancelled"} or row.error_code != "PARSER_TERMINATION_UNCONFIRMED"):
                return {"status": "not_eligible", "publication_permitted": False}
            file = db.get(core.File, file_id)
            snapshot = (row.id, row.owner_id, row.execution_id, row.revision, file.sha256)
        observation = discover(self.root, target=(snapshot[0], snapshot[2]), include_released=True)
        rows = observation["items"]
        if (len(rows) != 1 or rows[0].get("source_sha256") != snapshot[4]
                or rows[0].get("stop_evidence_observed") is not True or rows[0].get("input_exists") is not False
                or rows[0].get("ownership_state") not in {
                    "input_released_after_stop_or_zero_dispatch", "input_released_after_late_stop"}):
            return {"status": "still_uncertain", "publication_permitted": False}
        with self.factory.begin() as db:
            MaterialService(db, Actor(snapshot[1]), None).catalog.lock_owner()
            row = db.get(m.MaterialIngestion, snapshot[0])
            file = db.get(core.File, file_id)
            if (row.execution_id != snapshot[2] or row.revision != snapshot[3] or file.sha256 != snapshot[4]
                    or row.execution_outcome != "uncertain" or row.error_code != "PARSER_TERMINATION_UNCONFIRMED"
                    or row.state not in {"failed", "cancelled"}):
                raise DomainError("private_parser_reconciliation_state_changed", 409)
            row.execution_outcome = "cancelled" if row.state == "cancelled" else "failed"
            row.revision += 1
            db.add(m.MaterialEvent(ingestion_id=row.id, seq=row.revision, owner_id=row.owner_id, state=row.state,
                occurred_at=db.execute(text("SELECT clock_timestamp()")).scalar_one(),
                error_code="PARSER_STOP_OBSERVED_AFTER_FAILURE"))
        return {"status": "resource_uncertainty_observed_closed", "original_failure_preserved": True,
                "publication_permitted": False}

    def _finish(
        self,
        claim: Claim,
        parsed: dict,
        chunks: list[dict],
        artifact: StoredBlob,
        deadline: float,
        assets: list[Asset] | None = None,
        execution_storage: MaterialExecutionStorage | None = None,
    ) -> dict:
        self._remaining(deadline)
        original = StoredBlob(claim.security_domain, claim.sha256, claim.size_bytes)
        (execution_storage or self.store).verify(original)
        if digest(self._config()) != claim.config_sha256:
            raise MaterialError("INGESTION_CONFIGURATION_CHANGED")
        representation_id = digest(
            {
                "owner_id": str(claim.owner_id),
                "file_id": str(claim.file_id),
                "material_version_id": str(claim.material_version_id),
                "source_sha256": claim.sha256,
                "config_sha256": claim.config_sha256,
                "source_view_id": parsed.get("representation_id"),
            }
        )
        missing = [f"{r['item_ref']}:{r['reason']}" for r in parsed.get("missing_ranges", [])]
        missing += [f"page:{page}:OCR_or_review_required" for page in parsed.get("pages_requiring_ocr_or_review", [])]
        coverage = "partial" if missing or parsed["state"] == "partial" else "text_full"
        if parsed.get("pages_requiring_ocr_or_review") and not chunks:
            coverage = (
                "image_only" if len(parsed["pages_requiring_ocr_or_review"]) == parsed.get("pages") else "partial"
            )
        payload = checked(
            "ParsedDocument",
            {
                "representation_id": representation_id,
                "file_id": str(claim.file_id),
                "file_sha256": claim.sha256,
                "parser_version": "citequest-material/1;"
                + json.dumps(parsed["parser_versions"], sort_keys=True, separators=(",", ":")),
                "normalizer_version": "docling-text-provenance-page-view/1"
                if claim.media_type == "application/pdf"
                else "ooxml-main-paragraph/1"
                if "wordprocessingml" in claim.media_type
                else "newlines-only/1",
                "config_sha256": claim.config_sha256,
                "coverage": coverage,
                "missing_ranges": missing,
                "requires_ocr": bool(parsed.get("pages_requiring_ocr_or_review")),
                "formula_warnings": parsed.get("formula_warnings", []),
                "table_warnings": parsed.get("table_warnings", []),
            },
            response=True,
        )
        with self.factory.begin() as db:
            db.execute(
                text("SELECT set_config('statement_timeout', :value, true)"),
                {"value": str(max(1, int(self._remaining(deadline) * 1000)))},
            )
            db.execute(
                text("SELECT set_config('lock_timeout', :value, true)"),
                {"value": str(max(1, int(self._remaining(deadline) * 1000)))},
            )
            service, row, file = self._active(db, claim)
            if execution_storage is not None:
                execution_storage.validate_in(db, original)
                execution_storage.validate_in(db, artifact)
            if row.state != "parsing":
                raise DomainError("validated_parser_phase_not_observed", 409)
            known = db.get(core.Blob, (artifact.security_domain, artifact.sha256))
            if known is None:
                db.add(
                    core.Blob(
                        security_domain=artifact.security_domain, sha256=artifact.sha256, size_bytes=artifact.size_bytes
                    )
                )
                db.flush()
            db.add(
                m.ParsedRepresentation(
                    id=representation_id,
                    owner_id=claim.owner_id,
                    file_id=claim.file_id,
                    material_version_id=claim.material_version_id,
                    security_domain=claim.security_domain,
                    source_sha256=claim.sha256,
                    config_sha256=claim.config_sha256,
                    artifact_sha256=artifact.sha256,
                    page_count=parsed.get("pages"),
                    payload=payload,
                )
            )
            db.flush()
            if assets:
                register_assets(
                    db, self.store, owner_id=claim.owner_id, representation_id=representation_id, assets=assets,
                    verify_in=execution_storage.validate_in if execution_storage is not None else None,
                )
            for ordinal, chunk in enumerate(chunks, 1):
                self._remaining(deadline)
                checked("Locator", chunk["locator"], response=True)
                if chunk["locator"]["file_sha256"] != claim.sha256:
                    raise DomainError("fragment_source_mismatch", 409)
                fragment_id = digest({"representation_id": representation_id, "ordinal": ordinal, "chunk": chunk})
                fragment = {
                    "fragment_id": fragment_id,
                    "ordinal": ordinal,
                    "material_version_id": str(claim.material_version_id),
                    **chunk,
                }
                db.add(
                    m.MaterialFragment(
                        id=fragment_id,
                        owner_id=claim.owner_id,
                        file_id=claim.file_id,
                        representation_id=representation_id,
                        ordinal=ordinal,
                        source_text=chunk["source_text"],
                        index_text=chunk["index_text"],
                        token_count=chunk["token_count"],
                        search_document=func_vector(chunk["index_text"]),
                        payload=fragment,
                    )
                )
            db.flush()
            self._remaining(deadline)
            row.representation_id = representation_id
            service._publish_formal(file, row)
            service._advance(
                row,
                file,
                "partial" if parsed["state"] == "partial" else "ready",
                parsed["code"] if parsed["state"] == "partial" else None,
            )
            row.execution_outcome = "uncertain"  # Only the post-commit observer can acknowledge success.
            self._remaining(deadline)
            return {
                "status": "processed",
                "file_id": str(file.id),
                "ingestion_state": row.state,
                "representation_id": representation_id,
                "fragments": len(chunks),
                "coverage": coverage,
                "missing_ranges": len(missing),
            }

    def run_next(self, *, file_id: UUID | None = None) -> dict:
        engine = self.factory.kw["bind"]
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as guard:
            if guard.execute(text("SELECT pg_try_advisory_lock(:slot)"), {"slot": SLOT}).scalar() is not True:
                return {"status": "busy"}
            try:
                if self.business_storage is not None:
                    with self.factory.begin() as db:
                        unresolved = db.scalar(select(m.MaterialIngestion.id).where(
                            or_(m.MaterialIngestion.execution_outcome == "uncertain",
                                m.MaterialIngestion.state.in_(["validating", "parsing"]))
                        ).limit(1))
                    if unresolved is not None:
                        return {"status": "blocked", "code": "material_physical_or_publication_reconciliation_required",
                                "execution_success": False}
                recovered = self._recover_expired(file_id)
                if recovered is not None:
                    return recovered
                started, started_at = self.clock(), datetime.now(UTC)
                deadline = started + self.budget_seconds
                claim = self._claim(file_id, started_at, deadline)
                if claim is None:
                    return {"status": "empty"}
                if isinstance(claim, dict):
                    return claim
                try:
                    original = StoredBlob(claim.security_domain, claim.sha256, claim.size_bytes)
                    execution_storage = None
                    if self.business_storage is not None:
                        from citequest.business_storage_v3.contracts import OperationDeadline

                        from .execution_storage import MaterialExecutionStorage

                        execution_storage = MaterialExecutionStorage(self.business_storage, Actor(claim.owner_id),
                            claim.lease_id, OperationDeadline(deadline), lambda db: self._active(db, claim))
                    object_store = execution_storage or self.store
                    if execution_storage is None:
                        self.store.verify(original)
                    output = self.artifact_root / str(claim.ingestion_id) / str(claim.lease_id)
                    request = {
                        "source_sha256": claim.sha256,
                        "media_type": claim.media_type,
                        "expected_profile_sha256": claim.parser_profile_sha256,
                    }
                    if claim.sha256 in self.cache:
                        cache = self.cache[claim.sha256]
                        request.update(
                            cached_docling_document=str(cache),
                            cached_docling_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                        )
                    source_context = (self.runner.source_context(execution_storage, original, ingestion_id=claim.ingestion_id)
                        if execution_storage is not None else nullcontext(self.store.path(claim.security_domain, claim.sha256)))
                    with source_context as source:
                        request["source_path"] = str(source)
                        receipt = (self.runner or run_isolated)(
                            request,
                            root=self.root,
                            profile_path=self.parser_profile,
                            output=output / "parse",
                            python=self.parser_python,
                            on_validated=lambda: self._parsing(claim),
                            cancel_requested=lambda: self._cancelled(claim),
                            deadline_seconds=min(120, self._remaining(deadline)),
                        )
                    parsed = receipt["result"]
                    if parsed["state"] not in {"ready", "partial"}:
                        return self._fail(claim, parsed["code"], started)
                    if not receipt["validation_phase_observed"] or parsed.get("source_sha256") != claim.sha256:
                        return self._fail(claim, "PARSER_SOURCE_OR_VALIDATION_EVENT_MISMATCH", started)
                    if self._cancelled(claim):
                        return self._fail(claim, "PARSER_CANCELLED", started)
                    self._remaining(deadline)
                    chunks = []
                    if parsed["blocks"]:
                        source = output / "chunk-input.json"
                        source.write_text(
                            json.dumps({"file_sha256": claim.sha256, "blocks": parsed["blocks"]}, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        tokenized = (self.runner or run_isolated)(
                            {
                                "operation": "chunk",
                                "source_path": str(source),
                                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                "expected_profile_sha256": claim.chunk_profile_sha256,
                                "expected_tokenizer_profile_sha256": claim.tokenizer_profile_sha256,
                            },
                            root=self.root,
                            profile_path=self.chunk_profile,
                            output=output / "chunk",
                            python=Path(sys.executable),
                            cancel_requested=lambda: self._cancelled(claim),
                            deadline_seconds=min(30, self._remaining(deadline)),
                        )
                        if tokenized["result"]["state"] != "ready":
                            return self._fail(claim, tokenized["result"]["code"], started)
                        chunks = tokenized["result"]["chunks"]
                        if (
                            tokenized["result"].get("original_file_sha256") != claim.sha256
                            or tokenized["result"].get("tokenizer_profile_sha256") != claim.tokenizer_profile_sha256
                        ):
                            return self._fail(claim, "CHUNKER_SOURCE_OR_PROFILE_MISMATCH", started)
                    self._remaining(deadline)
                    parsed["product_execution"] = {
                        "active_budget_seconds": self.budget_seconds,
                        "started_at": started_at.isoformat(),
                        "elapsed_before_publish": self.clock() - started,
                        "queue_time_excluded": True,
                        "parser_elapsed_seconds": receipt["elapsed_seconds"],
                        "parser_child_deadline": receipt["deadline_seconds"],
                    }
                    artifact_bytes = json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode()
                    artifact = object_store.put(claim.security_domain, io.BytesIO(artifact_bytes), 64 * 1024 * 1024)
                    assets = capture_assets(
                        output,
                        parsed,
                        original,
                        artifact,
                        object_store,
                        mode="captured_before_publish",
                        check_remaining=lambda: self._remaining(deadline),
                    )
                    result = self._finish(claim, parsed, chunks, artifact, deadline, assets, execution_storage)
                    elapsed = self.clock() - started
                    with self.factory.begin() as db:
                        row = db.get(m.MaterialIngestion, claim.ingestion_id)
                        if row.execution_id != claim.lease_id or row.representation_id != result["representation_id"]:
                            raise DomainError("publication_reconciliation_mismatch", 409)
                        row.active_elapsed_seconds = elapsed
                        row.execution_outcome = "within_deadline" if elapsed <= self.budget_seconds else "late"
                        if elapsed > self.budget_seconds:
                            row.error_code = "PUBLICATION_OBSERVED_AFTER_DEADLINE"
                    result.update(
                        execution_outcome="within_deadline" if elapsed <= self.budget_seconds else "late",
                        active_elapsed_seconds=elapsed,
                        active_budget_seconds=self.budget_seconds,
                        execution_success=elapsed <= self.budget_seconds and result["ingestion_state"] == "ready",
                    )
                    observed = self.clock() - started
                    if observed > self.budget_seconds:
                        with self.factory.begin() as db:
                            row = db.get(m.MaterialIngestion, claim.ingestion_id)
                            row.execution_outcome, row.active_elapsed_seconds = "late", observed
                            row.error_code = "PUBLICATION_OBSERVED_AFTER_DEADLINE"
                        result.update(
                            execution_outcome="late", active_elapsed_seconds=observed, execution_success=False
                        )
                    return result
                except (MaterialError, DomainError) as exc:
                    return self._fail(claim, exc.code, started)
                except ValueError:
                    return self._fail(claim, "SOURCE_OR_ARTIFACT_INTEGRITY_FAILED", started)
                except Exception as exc:
                    return self._fail(claim, "INGESTION_INTERNAL_" + type(exc).__name__.upper(), started)
            finally:
                guard.execute(text("SELECT pg_advisory_unlock(:slot)"), {"slot": SLOT})


def func_vector(value: str) -> ColumnElement:
    from sqlalchemy import func, literal_column

    return func.to_tsvector(literal_column("'simple'::regconfig"), value)
