"""Separate non-model export worker; no request handler runs LibreOffice."""

from __future__ import annotations

import hashlib
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from citequest.agents_v3.contracts import AgentError, digest
from citequest.research_v3.school_contracts import TOKEN, SchoolDocument
from citequest.v2.domain import Actor, DomainError

from .contracts import Format
from .service import ExportProfile, ExportService


@dataclass(frozen=True)
class RenderBundle:
    files: dict[Format, Path]
    receipt: dict


class RenderFailure(Exception):
    """Bounded private tool diagnostics; never an HTTP response or public URL."""

    def __init__(self, code: str, diagnostics: bytes) -> None:
        super().__init__(code)
        self.code, self.diagnostics = code, diagnostics[:65536]


class Renderer(Protocol):
    profile: ExportProfile

    def render(
        self,
        document: SchoolDocument,
        references: list[dict],
        formats: list[Format],
        output: Path,
        *,
        timeout_seconds: float,
    ) -> RenderBundle: ...


class ExportWorker:
    def __init__(
        self,
        service: ExportService,
        renderer: Renderer,
        *,
        worker_id: str,
        spool_dir: Path,
        admit_references: Callable[..., list[dict]],
    ) -> None:
        if renderer.profile != service.profile:
            raise ValueError("export_worker_toolchain_profile_mismatch")
        self.service, self.renderer, self.worker_id = service, renderer, worker_id
        self.spool_dir, self.admit_references = spool_dir.resolve(), admit_references
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        if spool_dir.is_symlink():
            raise ValueError("export_spool_symlink_rejected")

    def run_once(self) -> dict | None:
        self.service.ready()
        # Anchor before claim's round trip, conservatively including its
        # latency rather than extending the SQL deadline after it returns.
        started = time.monotonic()
        row = self.service.claim(self.worker_id)
        if row is None:
            return None
        actor = Actor(row["owner_id"])
        try:
            self.service.fence(row)
            _, document = self.service.document(actor, row["task_id"], row)
            references = (self.admit_references(actor, row["task_id"], document, source_binding=row["source_binding"])
                if row.get("source_binding") is not None else self.admit_references(actor, row["task_id"], document))
            expected = {item["id"]: item for item in document.reference_admission.references}
            used = {alias for section in document.sections for alias in TOKEN.findall(section.body())}
            if (
                not references
                or {item["id"] for item in references} != used
                or len({item["id"] for item in references}) != len(references)
                or any(item != expected.get(item["id"]) for item in references)
            ):
                raise DomainError("export_current_bibliography_changed", 409)
            with tempfile.TemporaryDirectory(prefix="export-" + str(row["id"]) + "-", dir=self.spool_dir) as staging:
                root = Path(staging)
                self.service.fence(row)
                remaining = 90 - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("export_job_deadline")
                bundle = self.renderer.render(
                    document, references, row["requested_formats"], root, timeout_seconds=remaining
                )
                if time.monotonic() - started >= 90:
                    raise TimeoutError("export_job_deadline")
                if set(bundle.files) != set(row["requested_formats"]):
                    raise ValueError("renderer_missing_requested_format")
                receipt = bundle.receipt
                if (
                    receipt.get("status") != "automatic_export_gates_passed"
                    or receipt.get("actual_model_requests") != 0
                    or receipt.get("document_sha256") != row["document_sha256"]
                    or receipt.get("toolchain_sha256") != self.service.profile.toolchain_sha256
                    or receipt.get("font_profile") != self.service.profile.font_profile
                ):
                    raise ValueError("trusted_renderer_receipt_contract_failed")
                self.service.document(actor, row["task_id"], row)
                self.service.fence(row)
                artifacts = {}
                for kind, path in bundle.files.items():
                    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                        raise ValueError("renderer_artifact_outside_private_staging")
                    if path.stat().st_size > 32 * 1024**2:
                        raise ValueError("renderer_artifact_exceeds_size_profile")
                    data = path.read_bytes()
                    recorded = receipt.get("artifacts", {}).get(kind, {})
                    if recorded != {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}:
                        raise ValueError("renderer_artifact_receipt_mismatch")
                    self.service.fence(row)
                    artifacts[kind] = self.service.store_bytes(actor, data)
                # The guard runs again after conversion/CAS and before SQL publish.
                self.service.document(actor, row["task_id"], row)
                if time.monotonic() - started >= 90:
                    raise TimeoutError("export_job_deadline")
                self.service.finish(
                    row,
                    artifacts,
                    {
                        **receipt,
                        "elapsed_seconds": time.monotonic() - started,
                        "export_model_cost": 0,
                        "task_budget_was_reset": False,
                    },
                )
            return {"id": str(row["id"]), "state": "succeeded", "actual_model_requests": 0}
        except RenderFailure as error:
            try:
                self.service.fence(row)
            except DomainError:
                return self.service.fail(row, error.code)
            diagnostics = error.diagnostics or b"renderer failed without diagnostics"
            try:
                audit = {"private_renderer_diagnostics": self.service.store_bytes(actor, diagnostics)}
            except Exception:
                # Keep the primary render failure even if private diagnostic
                # storage has an uncertain write. Do not retry the blob upload.
                audit = {"diagnostic_persistence": "not_verified", "diagnostic_sha256": digest(diagnostics)}
            return self.service.fail(row, error.code, audit=audit)
        except TimeoutError:
            code = "export_job_deadline_exceeded"
        except (AgentError, DomainError):
            code = "export_authority_or_execution_changed"
        except Exception:
            # Raw tool errors/paths/source text stay private; never return repr(exc).
            code = "export_renderer_or_artifact_contract_failed"
        return self.service.fail(row, code)
