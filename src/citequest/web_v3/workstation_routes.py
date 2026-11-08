"""Composition root for the workstation; no duplicate identity or task scheduler."""

from collections.abc import Callable

from fastapi import APIRouter
from sqlalchemy.orm import Session, sessionmaker

from citequest.agents_v3.contracts import EvidenceSelection
from citequest.v2.domain import Actor
from citequest.v2.pagination import Pager
from citequest.workstation_v3.api import build_router as workstation_router

from .auth import AuthService


def build_router(
    factory: sessionmaker[Session],
    auth: AuthService,
    *,
    cursor_key: bytes,
    authorize_sources: Callable[[Actor, list[EvidenceSelection]], None],
    read_history_papers: Callable | None = None,
) -> APIRouter:
    """Reuse signed-session, Origin and CSRF checks for every workstation mutation."""
    return workstation_router(factory, auth.private_actor, Pager(cursor_key), authorize_sources,
                              read_history_papers=read_history_papers)
