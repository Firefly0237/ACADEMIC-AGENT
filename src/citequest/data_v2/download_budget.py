"""Single-writer durable source-byte reservations, independent of model/API money."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4


class DownloadBudget:
    """Unknown/interrupted requests retain their worst-case byte reservation."""

    def __init__(self, root: Path, limit: int = 1024**3) -> None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("positive_integer_download_limit_required")
        root.mkdir(parents=True, exist_ok=True)
        self.path, self.lock = root / "requests.jsonl", root / "writer.lock"
        self.limit, self.reserved, self.settled = limit, {}, {}
        self.lock_fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            if self.path.exists():
                with self.path.open(encoding="utf-8") as stream:
                    records = [json.loads(line) for line in stream]
                if not records or records[0] != {"kind": "limit", "bytes": limit}:
                    raise ValueError("download_budget_limit_or_history_mismatch")
                for item in records[1:]:
                    if item["kind"] == "reserve" and item["id"] not in self.reserved and item["id"] not in self.settled:
                        value = item["bytes"]
                        if type(value) is not int or not 0 < value <= self.remaining:
                            raise ValueError("invalid_download_reservation_history")
                        self.reserved[item["id"]] = value
                    elif item["kind"] == "settle" and item["id"] in self.reserved:
                        value = item["bytes"]
                        if type(value) is not int or not 0 <= value <= self.reserved[item["id"]]:
                            raise ValueError("invalid_download_settlement_history")
                        self.reserved.pop(item["id"])
                        self.settled[item["id"]] = value
                    else:
                        raise ValueError("invalid_download_history_transition")
            else:
                self.append({"kind": "limit", "bytes": limit})
        except BaseException:
            self.close()
            raise

    def append(self, value: dict) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @property
    def remaining(self) -> int:
        return self.limit - sum(self.reserved.values()) - sum(self.settled.values())

    def reserve(self, maximum: int) -> tuple[str, int]:
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("positive_integer_reservation_required")
        allowed = min(maximum, self.remaining)
        if allowed <= 0:
            raise ValueError("download_budget_exhausted")
        key = uuid4().hex
        self.append({"kind": "reserve", "id": key, "bytes": allowed})
        self.reserved[key] = allowed
        return key, allowed

    def settle(self, key: str, actual: int) -> None:
        if key not in self.reserved or type(actual) is not int or not 0 <= actual <= self.reserved[key]:
            raise ValueError("download_actual_outside_reservation")
        self.append({"kind": "settle", "id": key, "bytes": actual})
        self.reserved.pop(key)
        self.settled[key] = actual

    def summary(self) -> dict:
        return {
            "limit": self.limit,
            "settled_bytes": sum(self.settled.values()),
            "unresolved_reserved_bytes": sum(self.reserved.values()),
            "remaining": self.remaining,
            "settled_requests": len(self.settled),
            "unresolved_requests": len(self.reserved),
        }

    def close(self) -> None:
        os.close(self.lock_fd)
        self.lock.unlink()
