"""Private-input ownership around the existing fixed parser broker client.

Late stop receipts release resources only: they never turn a failed/deadline
execution into success. No new parsing window, retry or model call is created.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from citequest.materials_v2.execution_storage import MaterialExecutionStorage
from citequest.materials_v2.text import MaterialError
from citequest.v2.storage import StoredBlob

from .broker_client import BrokerParser
from .private_spool_protocol import TRANSPORT


def atomic_state(path: Path, value: dict) -> None:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > 8192:
        raise MaterialError("PRIVATE_PARSER_STATE_EXCEEDS_CAP")
    temporary = path.with_suffix(".pending")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def bound_stop_received(response: Path, receipt: Path, *, nonce: str, request_sha256: str,
                        parser_profile_sha256: str, parser_image_id: str) -> bool:
    """A stop receipt is not authorization to publish a late parser result."""
    try:
        if (response.is_symlink() or receipt.is_symlink() or response.stat().st_size > 8192
                or receipt.stat().st_size > 64 * 1024**2):
            return False
        value = json.loads(response.read_bytes())
        raw = receipt.read_bytes()
        result = json.loads(raw)
        return (value.get("nonce") == nonce and value.get("request_sha256") == request_sha256
            and value.get("receipt_sha256") == hashlib.sha256(raw).hexdigest()
            and result.get("profile_sha256") == parser_profile_sha256
            and result.get("image_id") == parser_image_id and result.get("worker_stopped") is True)
    except (OSError, ValueError, TypeError):
        return False


class PrivateSpoolBrokerParser(BrokerParser):
    source_transport = TRANSPORT

    def __init__(self, *, root: Path, exchange: Path, nonce: str, parser_image_id: str) -> None:
        super().__init__(root=root, exchange=exchange, nonce=nonce)
        if not parser_image_id.startswith("sha256:") or len(parser_image_id) != 71:
            raise ValueError("immutable_parser_image_required")
        self.parser_image_id = parser_image_id
        self.termination_confirmed = True
        self.request_dispatched = False
        self.quarantined = False
        self._record_value = None
        self._record_path = None
        self._request_sha = None
        self._output = None
        self._sequence = None

    def _record(self, state: str, **extra: object) -> None:
        if self._record_value is None or self._record_path is None:
            raise MaterialError("PRIVATE_PARSER_STATE_NOT_PREPARED")
        self._record_value.update(state=state, revision=self._record_value.get("revision", 0) + 1, **extra)
        journal = self._record_path.with_name(f"parser-request-events-{self._sequence}.jsonl")
        raw = json.dumps(self._record_value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        if journal.exists() and journal.stat().st_size + len(raw) > 65536:
            raise MaterialError("PRIVATE_PARSER_EVENT_JOURNAL_EXCEEDS_CAP")
        with journal.open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_state(self._record_path, self._record_value)

    def _observe_stop(self) -> bool:
        """Current fixed-request readback only; never trusts a new invocation."""
        if self._request_sha is None or self._output is None:
            return False
        response = self.exchange / f"response-{self._sequence}.json"
        receipt = self._output / "receipt.json"
        return bound_stop_received(response, receipt, nonce=self.nonce, request_sha256=self._request_sha,
            parser_profile_sha256=self._record_value["parser_profile_sha256"], parser_image_id=self.parser_image_id)

    def __call__(self, request: dict, **kwargs: object) -> dict:
        if self.quarantined:
            raise MaterialError("PRIVATE_PARSER_INSTANCE_QUARANTINED")
        output = kwargs.get("output")
        if not isinstance(output, Path):
            raise MaterialError("PRIVATE_PARSER_OWNED_OUTPUT_REQUIRED")
        output = output.resolve()
        artifact_root = self.root / "data/runtime-s04-linux/artifacts"
        if not output.is_relative_to(artifact_root) or len(output.relative_to(artifact_root).parts) != 3:
            raise MaterialError("PRIVATE_PARSER_OWNED_OUTPUT_REQUIRED")
        seq = self.sequence + 1
        self._output, self._sequence = output, seq
        self.termination_confirmed, self.request_dispatched = False, False
        self._request_sha = None
        output.parent.mkdir(parents=True, exist_ok=True)
        self._record_path = output.parent / f"parser-request-state-{seq}.json"
        if self._record_path.exists():
            raise MaterialError("PRIVATE_PARSER_REQUEST_STATE_EXISTS")
        self._record_value = {"version": "citequest-private-parser-request/1", "nonce": self.nonce,
            "sequence": seq, "ingestion_id": output.parent.parent.name, "execution_id": output.parent.name,
            "parser_image_id": self.parser_image_id, "parser_profile_sha256": request.get("expected_profile_sha256"),
            "request_sha256": None, "state": "dispatch_intent", "termination_confirmed": False,
            "publication_permitted": False, "no_new_execution_window": True,
            "parent_pid": os.getpid(), "exchange_relative": self.exchange.relative_to(self.root).as_posix(),
            "output_relative": output.relative_to(self.root).as_posix(), "source_sha256": request.get("source_sha256")}
        self._record("dispatch_intent")
        try:
            result = super().__call__(request, **kwargs)
            if result.get("image_id") != self.parser_image_id:
                raise MaterialError("PRIVATE_PARSER_IMAGE_CHANGED")
            published = self.exchange / f"request-{seq}.json"
            if not published.is_file() or published.is_symlink() or published.stat().st_size > 262144:
                raise MaterialError("PRIVATE_PARSER_PUBLISHED_REQUEST_INVALID")
            self._request_sha = hashlib.sha256(published.read_bytes()).hexdigest()
            self.termination_confirmed = True
            self.request_dispatched = True
            self._record("confirmed_stopped", termination_confirmed=True, request_sha256=self._request_sha)
            return result
        except BaseException as exc:
            raw_path = self.exchange / f"request-{seq}.json"
            if raw_path.is_file() and not raw_path.is_symlink() and raw_path.stat().st_size <= 262144:
                self._request_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                self.request_dispatched = True
                self._record("unknown", request_sha256=self._request_sha, failure_type=type(exc).__name__,
                    failure_code=getattr(exc, "code", None))
                (self.exchange / f"cancel-{seq}").touch(exist_ok=True)
                # Termination cleanup is not a renewed parser budget. Preserve
                # the original exception even when this late receipt arrives.
                until = time.monotonic() + 1.0
                while time.monotonic() < until and not self._observe_stop():
                    time.sleep(0.02)
                self.termination_confirmed = self._observe_stop()
            else:
                # The fixed client has returned and never published request;
                # its .pending file cannot be consumed by the fixed host.
                self.termination_confirmed = True
            if self.termination_confirmed:
                self._record("failed_but_stopped", termination_confirmed=True)
                raise
            self._record("unknown", termination_confirmed=False)
            self.quarantined = True
            raise MaterialError("PARSER_TERMINATION_UNCONFIRMED") from exc

    @contextmanager
    def source_context(self, execution: MaterialExecutionStorage, blob: StoredBlob,
                       *, ingestion_id: UUID) -> Iterator[Path]:
        if self.quarantined:
            raise MaterialError("PRIVATE_PARSER_INSTANCE_QUARANTINED")
        manager = execution.handoff(blob, root=self.root, ingestion_id=ingestion_id)
        source = manager.__enter__()
        ownership_path = source.parent.parent / "input-ownership.json"
        ownership = {"version": "citequest-private-input/1", "nonce": self.nonce,
            "parent_pid": os.getpid(), "ingestion_id": str(ingestion_id), "execution_id": source.parent.parent.name,
            "source_sha256": blob.sha256, "source_relative": source.relative_to(self.root).as_posix(),
            "state": "input_ready", "publication_permitted": False}
        try:
            if ownership_path.exists():
                raise MaterialError("PRIVATE_INPUT_OWNERSHIP_ALREADY_EXISTS")
            atomic_state(ownership_path, ownership)
        except BaseException:
            manager.__exit__(*sys.exc_info())
            raise
        self.termination_confirmed, self.request_dispatched = True, False
        original_exception = (None, None, None)
        try:
            yield source
        except BaseException:
            original_exception = sys.exc_info()
            raise
        finally:
            if self.termination_confirmed:
                manager.__exit__(*original_exception)
                ownership.update(state="input_released_after_stop_or_zero_dispatch",
                    request_dispatched=self.request_dispatched, termination_confirmed=True)
                atomic_state(ownership_path, ownership)
            else:
                self.quarantined = True
                self._record("unknown", input_quarantined=True)
                ownership.update(state="input_quarantined", termination_confirmed=False)
                atomic_state(ownership_path, ownership)

                def observe_and_close() -> None:
                    # No retry/dispatch. If the host died, retain resources
                    # until a genuine fixed-request stop receipt is observed.
                    while not self._observe_stop():
                        time.sleep(0.1)
                    error_type = None
                    try:
                        manager.__exit__(None, None, None)
                    except Exception as error:
                        # Current SQL authorization may now reject this failed
                        # lease; inner finally still closes FD/quota/input.
                        error_type = type(error).__name__
                    self._record("late_stop_cleanup_only", termination_confirmed=True,
                        input_quarantined=False, cleanup_guard_error_type=error_type)
                    ownership.update(state="input_released_after_late_stop", termination_confirmed=True,
                        cleanup_guard_error_type=error_type)
                    atomic_state(ownership_path, ownership)
                    self.termination_confirmed = True

                threading.Thread(target=observe_and_close, daemon=True,
                    name="cq-private-parser-quarantine").start()
