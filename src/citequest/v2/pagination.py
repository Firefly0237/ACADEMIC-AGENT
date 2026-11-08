"""Signed, expiring owner/query-bound keyset cursors; no offset scan."""

from __future__ import annotations

import base64
import hmac
import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, tuple_

from . import models as m
from .domain import DomainError
from .services import Catalog


class Pager:
    def __init__(self, key: bytes) -> None:
        if len(key) < 32:
            raise ValueError("cursor_signing_key_too_short")
        self.key = key

    def encode(self, value: dict) -> str:
        body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(body + hmac.digest(self.key, body, "sha256")).decode()

    def decode(self, cursor: str, identity: dict) -> dict:
        try:
            if len(cursor) > 4096:
                raise ValueError("large")
            raw = base64.b64decode(cursor, altchars=b"-_", validate=True)
            body, signature = raw[:-32], raw[-32:]
            if not hmac.compare_digest(hmac.digest(self.key, body, "sha256"), signature):
                raise ValueError("signature")
            value = json.loads(body)
            if value["identity"] != identity or value["expires"] <= datetime.now(UTC).timestamp():
                raise ValueError("identity_or_expired")
            UUID(value["id"])
            datetime.fromisoformat(value["created"])
            return value
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            raise DomainError("invalid_or_expired_cursor", 400) from None

    def page(self, catalog: Catalog, kind: str, limit: int = 20, cursor: str | None = None) -> tuple[list, str | None]:
        models = {"library": m.LibraryItem, "collections": m.Collection, "notes": m.Note, "history": m.History}
        if kind not in models or not 1 <= limit <= 50:
            raise DomainError("invalid_page", 400)
        model = models[kind]
        identity = {"owner": str(catalog.actor.owner_id), "kind": kind, "limit": limit}
        query = select(model).where(model.owner_id == catalog.actor.owner_id)
        if hasattr(model, "deleted_at"):
            query = query.where(model.deleted_at.is_(None))
        if kind in {"library", "notes"}:
            # Filter before LIMIT; no inaccessible IDs leak or shorten a page.
            from sqlalchemy import exists, or_

            if kind == "notes":
                query = query.join(m.LibraryItem, m.Note.library_item_id == m.LibraryItem.id).where(
                    m.LibraryItem.deleted_at.is_(None)
                )
            query = query.join(m.Work, m.LibraryItem.work_id == m.Work.id)
            access = exists(
                select(m.WorkAccess.work_id).where(
                    m.WorkAccess.work_id == m.Work.id,
                    m.WorkAccess.owner_id == catalog.actor.owner_id,
                    m.WorkAccess.revoked.is_(False),
                )
            )
            query = query.where(or_(m.Work.public.is_(True), access))
        if cursor:
            last = self.decode(cursor, identity)
            query = query.where(
                tuple_(model.created_at, model.id) < tuple_(datetime.fromisoformat(last["created"]), UUID(last["id"]))
            )
        rows = list(catalog.db.scalars(query.order_by(model.created_at.desc(), model.id.desc()).limit(limit + 1)))
        next_cursor = None
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor = self.encode(
                {
                    "identity": identity,
                    "created": last.created_at.isoformat(),
                    "id": str(last.id),
                    "expires": datetime.now(UTC).timestamp() + 900,
                }
            )
        return rows[:limit], next_cursor
