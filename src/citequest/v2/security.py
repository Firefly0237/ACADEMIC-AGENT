"""Opaque server-issued local sessions; owner is never accepted from HTTP input."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from .domain import Actor, DomainError
from .models import LocalSession, Owner


def token_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def issue_session(db: Session, actor: Actor, lifetime: timedelta = timedelta(hours=8)) -> tuple[str, str]:
    if db.get(Owner, actor.owner_id) is None or not timedelta(0) < lifetime <= timedelta(days=1):
        raise DomainError("invalid_local_session", 400)
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    db.add(
        LocalSession(
            owner_id=actor.owner_id,
            token_sha256=token_hash(token),
            csrf_sha256=token_hash(csrf),
            expires_at=datetime.now(UTC) + lifetime,
        )
    )
    return token, csrf


def authenticate(db: Session, token: str | None, *, csrf: str | None = None, mutation: bool = False) -> Actor:
    if token is None or not 32 <= len(token) <= 200:
        raise DomainError("session_required", 401)
    session = db.scalar(
        select(LocalSession).where(
            LocalSession.token_sha256 == token_hash(token),
            LocalSession.revoked.is_(False),
            LocalSession.expires_at > datetime.now(UTC),
        )
    )
    if session is None:
        raise DomainError("session_required", 401)
    if mutation and (csrf is None or not secrets.compare_digest(token_hash(csrf), session.csrf_sha256)):
        raise DomainError("csrf_required", 403)
    owner = db.get(Owner, session.owner_id)
    if owner is None:
        raise DomainError("session_required", 401)
    return Actor(session.owner_id, publisher=owner.can_publish)
