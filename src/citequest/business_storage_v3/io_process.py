"""Linux-owned subprocess lifetime around pinned BlobStore IO.

Session advisory slots coordinate normal clients, not a claim of fleet hard
concurrency after unconfirmed host/kernel failure. Unknown writes remain subject
to the separate durable business admission barrier. No shell/preexec_fn is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

from citequest.platform_v3.blob_config import BlobStoreConfig
from citequest.platform_v3.blob_store import BlobIdentity, BlobReference
from citequest.v2.domain import DomainError

from .contracts import OperationDeadline, StorageBinding, parse_reference, reference_value
from .io_contracts import IOJob, IOProfile
from .io_pg import PGResourceConfig, PGResourceLease

_SPOOL_SLOTS = threading.BoundedSemaphore(2)


@dataclass
class _Started:
    process: subprocess.Popen
    operation: str
    identity: BlobIdentity | None
    reference: BlobReference | None


class _Lease:
    def __init__(self, executor: SupervisedBlobIO, deadline: OperationDeadline) -> None:
        self.executor, self.deadline = executor, deadline
        if not _SPOOL_SLOTS.acquire(timeout=deadline.remaining()):
            raise DomainError("business_blob_local_spool_busy", 429)
        self.temporary = None
        self.resource = None
        self.receipt = None
        self.closed = self.quarantined = False
        try:
            deadline.remaining()
            self.temporary = tempfile.TemporaryDirectory(prefix="cq-blob-io-", dir=executor.config.spool_dir)
            self.directory = Path(self.temporary.name)
            self.directory.chmod(0o700)
            (self.directory / "sdk-spool").mkdir(mode=0o700)
        except BaseException:
            _SPOOL_SLOTS.release()
            raise

    def acquire_physical_slot(self) -> None:
        if self.resource is not None:
            raise DomainError("business_blob_io_slot_already_owned", 409)
        self.deadline.remaining()
        if self.executor.resource_config is None:
            return  # Explicit component-only local coordination.
        keys = []
        for index in range(self.executor.profile.global_io_slots):
            raw = hashlib.sha256(("cq-blob-io/1:" + self.executor.binding.store_id + ":" +
                                  self.executor.binding.sha256 + ":" + str(index)).encode()).digest()
            key = int.from_bytes(raw[:8], "big", signed=True)
            keys.append(key)
        self.resource = PGResourceLease(self.executor.resource_config, self.deadline,
            self.executor._environment(self.directory), self.directory)
        self.receipt = self.executor.last_receipt = {"phase": "resource_acquisition", "go_attempted": False,
            "resource": self.resource.receipt}
        self.resource.acquire(keys)

    def release_physical_slot(self) -> None:
        if self.resource is not None:
            if not self.resource.close():
                self.quarantine(self.resource.process)
                raise DomainError("business_blob_resource_termination_unconfirmed_quarantined", 503)
            self.resource = None

    def assert_resource_alive(self) -> None:
        if self.resource is not None:
            self.resource.assert_alive()

    def close(self) -> None:
        if self.closed:
            return
        # An unconfirmed resource-owner exit retains the same local quota/spool
        # through quarantine. Do not run cleanup in an unconditional finally.
        self.release_physical_slot()
        self.closed = True
        try:
            if self.temporary is not None:
                self.temporary.cleanup()
        finally:
            _SPOOL_SLOTS.release()

    def quarantine(self, process: subprocess.Popen) -> None:
        self.quarantined = True

        def reap() -> None:
            process.wait()  # No new IO. Hold resources until real exit, not lease expiry.
            if self.receipt is not None:
                self.receipt.update(termination_confirmed=True, exit_code=process.returncode, quarantine_reaped=True)
            self.close()

        threading.Thread(target=reap, name="cq-blob-io-quarantine", daemon=True).start()


class PreparedIO:
    """One private spool and explicit physical phase; can bracket durable sent."""
    def __init__(self, executor: SupervisedBlobIO, lease: _Lease) -> None:
        self.executor, self.lease = executor, lease
        self.active = False
        self.used = False
        self.started = None

    def admit(self, identity: BlobIdentity, source: BinaryIO) -> None:
        if identity.size_bytes > self.executor.profile.maximum_object_bytes:
            raise DomainError("business_blob_size_exceeds_profile", 413)
        sha, size = hashlib.sha256(), 0
        with (self.lease.directory / "source.bin").open("xb") as target:
            while True:
                self.lease.deadline.remaining()
                part = source.read(65536)
                if not part:
                    break
                size += len(part)
                if size > identity.size_bytes:
                    raise DomainError("business_blob_upload_size_mismatch", 422)
                sha.update(part)
                target.write(part)
        if size != identity.size_bytes or sha.hexdigest() != identity.sha256:
            raise DomainError("business_blob_upload_identity_mismatch", 422)

    @contextmanager
    def physical(self) -> Iterator[PreparedIO]:
        self.lease.acquire_physical_slot()
        self.active = True
        try:
            yield self
        finally:
            self.active = False
            if self.started is not None and self.started.process.poll() is None:
                self.executor._terminate(self.started.process, self.lease)
            if not self.lease.quarantined:
                self.lease.release_physical_slot()

    def run(self, operation: str, *, identity: BlobIdentity | None = None,
            reference: BlobReference | None = None) -> dict:
        self.start(operation, identity=identity, reference=reference)
        return self.go()

    def start(self, operation: str, *, identity: BlobIdentity | None = None,
              reference: BlobReference | None = None) -> None:
        if not self.active:
            raise DomainError("business_blob_physical_slot_required", 503)
        if self.used:
            raise DomainError("business_blob_single_physical_job_only", 409)
        self.used = True
        self.started = self.executor._start(self.lease, operation, identity=identity, reference=reference)

    def go(self) -> dict:
        if not self.active or self.started is None:
            raise DomainError("business_blob_ready_child_required", 503)
        if self.lease.receipt["go_attempted"]:
            raise DomainError("business_blob_GO_is_not_retried", 409)
        return self.executor._go(self.lease, self.started)

    @contextmanager
    def verified_stream(self, reference: BlobReference) -> Iterator[BinaryIO]:
        path = self.lease.directory / "verified.bin"
        if path.is_symlink() or path.stat().st_size != reference.identity.size_bytes:
            raise DomainError("business_blob_io_verified_path_invalid", 503)
        with path.open("rb") as stream:
            sha, size = hashlib.sha256(), 0
            while part := stream.read(65536):
                self.lease.deadline.remaining()
                sha.update(part)
                size += len(part)
            if size != reference.identity.size_bytes or sha.hexdigest() != reference.identity.sha256:
                raise DomainError("business_blob_io_verified_bytes_invalid", 503)
            stream.seek(0)
            yield stream


class SupervisedBlobIO:
    def __init__(self, config: BlobStoreConfig, *, profile: IOProfile | None = None,
                 resource: PGResourceConfig | None = None, resource_connect: Callable | None = None) -> None:
        if sys.platform != "linux":
            raise DomainError("business_blob_supervisor_requires_linux", 503)
        self.config = BlobStoreConfig.model_validate(config.model_dump(mode="python"), strict=True)
        profile = profile or IOProfile(maximum_object_bytes=config.maximum_object_bytes)
        self.profile = IOProfile.model_validate(profile.model_dump(mode="python"), strict=True)
        if self.profile.maximum_object_bytes != config.maximum_object_bytes:
            raise DomainError("business_blob_io_profile_cap_mismatch", 503)
        if config.mode == "cloud" and any(os.environ.get(name) for name in
                ("STORAGE_EMULATOR_HOST", "FIREBASE_STORAGE_EMULATOR_HOST", "API_ENDPOINT_OVERRIDE")):
            raise DomainError("business_blob_ambient_endpoint_forbidden", 503)
        if resource_connect is not None:
            raise DomainError("business_blob_unbounded_resource_callback_rejected", 503)
        if resource is not None and not isinstance(resource, PGResourceConfig):
            raise ValueError("trusted_supervised_PG_resource_config_required")
        self.binding, self.resource_config = StorageBinding.configured(self.config), resource
        self.config.spool_dir.mkdir(parents=True, exist_ok=True)
        if self.config.spool_dir.is_symlink():
            raise DomainError("business_blob_spool_symlink_rejected", 503)
        self.last_receipt = None

    @contextmanager
    def operation(self, deadline: OperationDeadline, *, identity: BlobIdentity | None = None,
                  source: BinaryIO | None = None) -> Iterator[PreparedIO]:
        if deadline.remaining() > self.profile.maximum_active_seconds:
            raise DomainError("business_blob_io_deadline_exceeds_profile", 503)
        lease = _Lease(self, deadline)
        try:
            operation = PreparedIO(self, lease)
            if source is not None:
                if identity is None:
                    raise ValueError("source_requires_exact_identity")
                operation.admit(identity, source)
            yield operation
        finally:
            if not lease.quarantined:
                lease.close()

    @staticmethod
    def _environment(directory: Path) -> dict:
        # No inherited PG/model keys, credentials files, SDK endpoint overrides,
        # proxies, user sitecustomize or user-writable PYTHONPATH.
        return {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                "TMPDIR": str(directory / "sdk-spool")}

    def _command(self, job: Path) -> list[str]:
        return [sys.executable, "-m", "citequest.business_storage_v3.io_worker",
                "--parent-pid", str(os.getpid()), "--job", str(job)]

    def _terminate(self, process: subprocess.Popen, lease: _Lease) -> None:
        if process.stdin is not None:
            with suppress(BrokenPipeError, OSError):
                process.stdin.close()
        # A denied kill is not successful termination. Still wait for proof;
        # quarantine retains every lease if the process remains alive.
        with suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=self.profile.kill_confirmation_seconds)
        except subprocess.TimeoutExpired:
            if lease.receipt is not None:
                lease.receipt.update(termination_confirmed=False, quarantined=True)
            lease.quarantine(process)
            raise DomainError("business_blob_io_termination_unconfirmed_quarantined", 503) from None
        if lease.receipt is not None:
            lease.receipt.update(termination_confirmed=True, exit_code=process.returncode, killed_and_waited=True)

    def _start(self, lease: _Lease, operation: str, *, identity: BlobIdentity | None,
               reference: BlobReference | None) -> _Started:
        storage = self.config.model_copy(update={"spool_dir": lease.directory / "sdk-spool"})
        job = IOJob(operation=operation, expected_parent_pid=os.getpid(), expires_at=lease.deadline.expires_at,
                    profile=self.profile, storage=storage, identity=asdict(identity) if identity else None,
                    reference=reference_value(reference) if reference else None)
        raw = job.model_dump_json().encode()
        if len(raw) > self.profile.maximum_job_bytes:
            raise DomainError("business_blob_io_job_bytes_exceeded", 413)
        job_path = lease.directory / "job.json"
        job_path.write_bytes(raw)
        process = subprocess.Popen(self._command(job_path), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True, cwd=lease.directory,
            env=self._environment(lease.directory))
        receipt = {"pid": process.pid, "exit_code": None, "operation": operation,
                   "termination_confirmed": False, "go_attempted": False,
                   "normal_operation_pg_coordination": self.resource_config is not None,
                   "resource": lease.resource.receipt if lease.resource is not None else None,
                   "fleet_hard_physical_concurrency_after_host_failure": "not_verified"}
        lease.receipt = self.last_receipt = receipt
        ready_path = lease.directory / "ready.json"
        try:
            while not ready_path.is_file():
                if process.poll() is not None:
                    receipt.update(exit_code=process.returncode, termination_confirmed=True)
                    raise DomainError("business_blob_child_preflight_failed_no_GO", 503)
                time.sleep(min(0.01, lease.deadline.remaining()))
            if ready_path.is_symlink() or ready_path.stat().st_size > self.profile.maximum_result_bytes:
                raise DomainError("business_blob_io_ready_path_or_size_invalid", 503)
            ready = json.loads(ready_path.read_bytes())
            observation = ready.get("observations", {})
            receipt["ready_snapshot"] = {key: ready.get(key) for key in
                ("version", "pid", "parent_pid", "operation", "supervision", "profile", "observations")}
            if (ready.get("version") != "citequest-blob-io-ready/1" or ready.get("pid") != process.pid
                    or ready.get("parent_pid") != os.getpid() or ready.get("operation") != operation
                    or ready.get("supervision") != "linux-pdeathsig-kill" or ready.get("profile") != self.profile.version
                    or observation.get("go") is not False or observation.get("network_before_go") != 0
                    or observation.get("client_or_auth_before_go") != 0):
                raise DomainError("business_blob_io_ready_identity_or_zero_network_invalid", 503)
            lease.deadline.remaining()
            receipt.update(ready_observation=observation, child_limits=ready.get("limits"), ready_verified=True)
            return _Started(process, operation, identity, reference)
        except BaseException:
            self._terminate(process, lease)
            raise

    def _go(self, lease: _Lease, started: _Started) -> dict:
        process, operation, identity, reference = started.process, started.operation, started.identity, started.reference
        receipt = lease.receipt
        try:
            lease.deadline.remaining()
            lease.assert_resource_alive()
            receipt["go_attempted"] = True  # Broken GO pipe is not assumed to mean server saw nothing.
            process.stdin.write(b"GO\n")
            process.stdin.flush()
            process.stdin.close()
            while process.poll() is None:
                lease.assert_resource_alive()
                try:
                    process.wait(timeout=min(0.05, lease.deadline.remaining()))
                except subprocess.TimeoutExpired:
                    continue
            lease.assert_resource_alive()
        except subprocess.TimeoutExpired:
            self._terminate(process, lease)
            raise DomainError("business_blob_io_deadline_killed_and_waited", 408) from None
        except DomainError as exc:
            self._terminate(process, lease)
            if exc.code == "business_blob_deadline_exceeded":
                raise DomainError("business_blob_io_deadline_killed_and_waited", 408) from None
            raise
        except BaseException:
            self._terminate(process, lease)
            raise
        receipt.update(exit_code=process.returncode, termination_confirmed=True)
        if process.returncode != 0:
            raise DomainError("business_blob_io_child_failed", 503)
        lease.deadline.remaining()
        result_path = lease.directory / "result.json"
        if result_path.is_symlink() or not result_path.is_file() or result_path.stat().st_size > self.profile.maximum_result_bytes:
            raise DomainError("business_blob_io_result_path_or_size_invalid", 503)
        value = json.loads(result_path.read_bytes())
        if (set(value) != {"version", "operation", "pid", "parent_pid", "supervision", "profile", "reference", "limits", "observations"}
                or value.get("version") != "citequest-blob-io-result/1" or value.get("operation") != operation
                or value.get("pid") != process.pid or value.get("parent_pid") != os.getpid()
                or value.get("supervision") != "linux-pdeathsig-kill" or value.get("profile") != self.profile.version):
            raise DomainError("business_blob_io_result_identity_invalid", 503)
        if operation in {"put", "read"} and value["reference"] is None:
            raise DomainError("business_blob_io_required_reference_missing", 503)
        if operation != "probe" and value.get("reference") is not None:
            actual = parse_reference(value["reference"])
            expected = identity if identity else reference.identity
            self.binding.validate(actual, expected)
            if reference is not None and actual != reference:
                raise DomainError("business_blob_io_result_generation_changed", 409)
        lease.deadline.remaining()
        receipt["child_limits"] = value.get("limits")
        receipt["observations"] = value.get("observations")
        return value

    @contextmanager
    def open_verified(self, reference: BlobReference, *, deadline: OperationDeadline) -> Iterator[BinaryIO]:
        self.binding.validate(reference, reference.identity)
        with self.operation(deadline) as operation:
            with operation.physical():
                operation.run("read", reference=reference)
            with operation.verified_stream(reference) as stream:
                yield stream  # Same verified FD. PG slot already released; local spool quota remains held.

    @contextmanager
    def materialized(self, reference: BlobReference, *, deadline: OperationDeadline) -> Iterator[Path]:
        """Private path for fixed parser/FileResponse adapters, never a locator.

        The caller must own this context for the full ASGI/parser lifetime and
        recheck current business ACL after download. Release in finally even on
        disconnect. This primitive does not authorize a File/Work or cache one.
        """
        self.binding.validate(reference, reference.identity)
        with self.operation(deadline) as operation:
            with operation.physical():
                operation.run("read", reference=reference)
            with operation.verified_stream(reference):
                pass
            path = operation.lease.directory / "verified.bin"
            path.chmod(0o400)
            try:
                yield path
            finally:
                path.chmod(0o600)
