"""Purpose-bound HMAC cursors; pagination uses persisted rank, never native offset."""
import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime
from uuid import UUID
from citequest.v2.domain import DomainError


class CursorSigner:
    def __init__(self, key: bytes):
        if type(key) is not bytes or len(key) < 32:
            raise ValueError("search_cursor_key_needs_at_least_32_bytes")
        self.key = key

    def encode(self, *, snapshot, after: int) -> str:
        payload = json.dumps({"v": 1, "purpose": "search-page", "snapshot": str(snapshot.id),
                              "owner": str(snapshot.owner_id), "generation": str(snapshot.generation_id),
                              "query": snapshot.query_sha256, "after": after,
                              "exp": int(snapshot.expires_at.timestamp())}, sort_keys=True, separators=(",", ":")).encode()
        signature = hmac.digest(self.key, payload, "sha256")
        return base64.urlsafe_b64encode(payload + signature).rstrip(b"=").decode("ascii")

    def decode(self, token: str, *, snapshot, owner_id: UUID) -> int:
        try:
            if type(token) is not str or len(token) > 1800:
                raise ValueError()
            raw = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
            payload, signature = raw[:-32], raw[-32:]
            if not hmac.compare_digest(hmac.digest(self.key, payload, "sha256"), signature):
                raise ValueError()
            data = json.loads(payload)
            after = data.pop("after")
            if type(after) is not int or not 0 <= after <= 200:
                raise ValueError()
            expected = {"v": 1, "purpose": "search-page", "snapshot": str(snapshot.id), "owner": str(owner_id),
                        "generation": str(snapshot.generation_id), "query": snapshot.query_sha256,
                        "exp": int(snapshot.expires_at.timestamp())}
            if data != expected or data["exp"] <= datetime.now(UTC).timestamp():
                raise ValueError()
            return after
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            raise DomainError("search_cursor_invalid_or_expired", 400) from None
