"""Nonprivileged PG worker exchanges one lease with a fixed developer broker."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

from citequest.materials_v2.text import MaterialError


class BrokerParser:
    def __init__(self, *, root: Path, exchange: Path, nonce: str) -> None:
        self.root, self.exchange, self.nonce = root.resolve(), exchange.resolve(), nonce
        if not self.exchange.is_relative_to(self.root / "data/runtime-s04-linux/exchanges"):
            raise ValueError("Owned broker exchange required")
        self.sequence = 0

    def __call__(
        self,
        request: dict,
        *,
        root: Path,
        profile_path: Path,
        output: Path,
        python: Path,
        on_validated: Callable[[], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        deadline_seconds: float | None = None,
    ) -> dict:
        del python
        if root.resolve() != self.root or self.sequence >= 2:
            raise MaterialError("BROKER_ROOT_OR_REQUEST_COUNT_REJECTED")
        started = time.monotonic()
        profile_raw = profile_path.read_bytes()
        profile_sha = hashlib.sha256(profile_raw).hexdigest()
        profile = json.loads(profile_raw)
        if profile_sha != request.get("expected_profile_sha256") or profile["platform"] != "linux":
            raise MaterialError("BROKER_PROFILE_BINDING_MISMATCH")
        if type(deadline_seconds) not in {int, float} or not 0 < deadline_seconds <= profile["deadline_seconds"]:
            raise MaterialError("BROKER_REMAINING_DEADLINE_REJECTED")
        self.sequence += 1
        seq = self.sequence
        message = {
            "schema_version": "cq.parser-broker-request/1",
            "nonce": self.nonce,
            "sequence": seq,
            "source_relative": Path(request["source_path"]).resolve().relative_to(self.root).as_posix(),
            "output_relative": output.resolve().relative_to(self.root).as_posix(),
            "profile_relative": profile_path.resolve().relative_to(self.root).as_posix(),
            "profile_sha256": profile_sha,
            "deadline_seconds": deadline_seconds,
            "request": {k: v for k, v in request.items() if k != "source_path"},
        }
        raw = json.dumps(message, ensure_ascii=False, allow_nan=False).encode()
        sha = hashlib.sha256(raw).hexdigest()
        target = self.exchange / f"request-{seq}.json"
        if target.exists():
            raise MaterialError("BROKER_REQUEST_ALREADY_CONSUMED")
        pending = target.with_suffix(".pending")
        pending.write_bytes(raw)
        pending.rename(target)
        cancelled = self.exchange / f"cancel-{seq}"
        response = self.exchange / f"response-{seq}.json"
        phase = self.exchange / f"phase-{seq}.json"
        observed = False
        try:
            while True:
                if phase.exists() and not observed:
                    item = json.loads(phase.read_bytes())
                    if item != {"nonce": self.nonce, "request_sha256": sha, "profile_sha256": profile_sha}:
                        raise MaterialError("BROKER_PHASE_BINDING_MISMATCH")
                    observed = True
                    if on_validated:
                        on_validated()
                if cancel_requested and cancel_requested():
                    cancelled.touch(exist_ok=True)
                if response.exists():
                    if response.is_symlink() or response.stat().st_size > 64 * 1024**2:
                        raise MaterialError("BROKER_RESPONSE_LIMIT_EXCEEDED")
                    value = json.loads(response.read_bytes())
                    if value["nonce"] != self.nonce or value["request_sha256"] != sha:
                        raise MaterialError("BROKER_RESPONSE_BINDING_MISMATCH")
                    receipt = output / "receipt.json"
                    data = receipt.read_bytes()
                    if hashlib.sha256(data).hexdigest() != value["receipt_sha256"]:
                        raise MaterialError("BROKER_RECEIPT_CHANGED")
                    result = json.loads(data)
                    if result["profile_sha256"] != profile_sha or result["worker_stopped"] is not True:
                        raise MaterialError("BROKER_INCOMPLETE_WORKER_RECEIPT")
                    if time.monotonic() - started > deadline_seconds:
                        raise MaterialError("PARSER_DEADLINE_EXCEEDED")
                    return result
                if time.monotonic() - started > deadline_seconds:
                    raise MaterialError("PARSER_DEADLINE_EXCEEDED")
                time.sleep(0.05)
        finally:
            if not response.exists():
                cancelled.touch(exist_ok=True)
