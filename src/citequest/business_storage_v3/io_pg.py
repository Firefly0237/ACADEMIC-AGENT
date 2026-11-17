"""Bounded Linux process owning a normal PostgreSQL advisory resource slot.

The trusted DSN crosses only a private stdin pipe. This is an IO adapter, not a
second scheduler. Parent/host uncertainty does not establish fleet physical2.
"""
from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from citequest.v2.domain import DomainError

from .contracts import OperationDeadline


@dataclass(frozen=True)
class PGResourceConfig:
    dsn: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.dsn, str) or not self.dsn or len(self.dsn.encode()) > 32768 or "\x00" in self.dsn:
            raise ValueError("bounded_trusted_resource_dsn_required")


class PGResourceLease:
    def __init__(self, config: PGResourceConfig, deadline: OperationDeadline, environment: dict,
                 directory: Path) -> None:
        deadline.remaining()
        self.deadline, self.acquired = deadline, False
        self.receipt = {"kind": "supervised_pg_session/1", "pid": None, "acquired": False,
                        "termination_confirmed": False, "fleet_hard_physical2": "not_verified"}
        self.process = subprocess.Popen([sys.executable, "-m", "citequest.business_storage_v3.io_pg_worker",
            "--parent-pid", str(os.getpid())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, close_fds=True, start_new_session=True, cwd=directory, env=environment)
        self.config = config
        self.receipt["pid"] = self.process.pid
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def acquire(self, keys: list[int]) -> None:
        raw = json.dumps({"version": "citequest-pg-resource-job/1", "parent": os.getpid(),
            "expires": self.deadline.expires_at, "keys": keys, "dsn": self.config.dsn},
            separators=(",", ":"), allow_nan=False).encode() + b"\n"
        # DSN never enters durable jobs, argv, result/receipt, stderr or logs.
        if len(raw) > 40000:
            raise DomainError("business_blob_resource_job_exceeds_cap", 503)
        offset = 0
        while offset < len(raw):
            self.assert_alive()
            _, writable, _ = select.select([], [self.process.stdin.fileno()], [], self.deadline.remaining())
            if writable:
                with suppress(BlockingIOError):
                    offset += os.write(self.process.stdin.fileno(), raw[offset:])
        response = bytearray()
        while b"\n" not in response:
            readable, _, _ = select.select([self.process.stdout.fileno()], [], [],
                                           min(0.01, self.deadline.remaining()))
            if readable:
                part = os.read(self.process.stdout.fileno(), 1025 - len(response))
                if not part or len(response) + len(part) > 1024:
                    raise DomainError("business_blob_resource_receipt_invalid", 503)
                response.extend(part)
            else:
                self.assert_alive()
        value = json.loads(response)
        if (set(value) != {"version", "parent", "pid", "status", "key", "backend_pid"}
                or value["version"] != "citequest-pg-resource-ready/1" or value["parent"] != os.getpid()
                or value["pid"] != self.process.pid):
            raise DomainError("business_blob_resource_receipt_invalid", 503)
        if value["status"] == "busy":
            raise DomainError("business_blob_coordinated_io_busy", 429)
        if value["status"] != "acquired" or value["key"] not in keys or type(value["backend_pid"]) is not int:
            raise DomainError("business_blob_resource_receipt_invalid", 503)
        self.deadline.remaining()
        self.acquired = True
        self.receipt.update(acquired=True, backend_pid=value["backend_pid"], key=value["key"])

    def assert_alive(self) -> None:
        if self.process.poll() is not None:
            self.receipt.update(termination_confirmed=True, exit_code=self.process.returncode)
            raise DomainError("business_blob_resource_process_exited", 503)

    def close(self) -> bool:
        """Confirm local process exit; a remote lock may release later."""
        if self.process.poll() is None:
            with suppress(OSError):
                if self.process.stdin is not None:
                    self.process.stdin.close()
            # Cleanup never grants a fresh deadline. At/after expiry kill now.
            remaining = max(0.0, self.deadline.expires_at - self.deadline.clock())
            try:
                self.process.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                with suppress(OSError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.receipt.update(termination_confirmed=False, quarantined=True)
                    return False
        self.receipt.update(termination_confirmed=True, exit_code=self.process.returncode)
        if self.process.stdout is not None:
            self.process.stdout.close()
        return True
