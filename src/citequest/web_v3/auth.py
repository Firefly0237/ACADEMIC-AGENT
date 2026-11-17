"""Authlib OIDC authorization-code + S256 PKCE, PG opaque application sessions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlsplit

import httpx2
from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse
from joserfc.errors import JoseError
from sqlalchemy import func, select, text, update
from sqlalchemy.orm import sessionmaker
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from citequest.v2.domain import Actor, DomainError
from citequest.v2.models import LocalSession, Owner, Project
from citequest.v2.origin import OriginPolicy
from citequest.v2.security import authenticate, issue_session, token_hash

from .models import GuestOwner, OIDCIdentity


@dataclass(frozen=True)
class AuthConfig:
    policy: OriginPolicy
    issuer: str
    client_id: str
    client_secret: str
    session_secret: str
    metadata_url: str | None = None

    def __post_init__(self) -> None:
        parts = urlsplit(self.issuer)
        if (
            not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
            or self.issuer.endswith("/")
            or not self.client_id
            or len(self.client_secret) < 32
            or len(self.session_secret) < 32
        ):
            raise ValueError("explicit_oidc_configuration_required")
        if self.policy.development:
            if parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost"}:
                raise ValueError("development_oidc_must_be_loopback")
        elif parts.scheme != "https":
            raise ValueError("oidc_https_required")
        if self.metadata_url is not None:
            # Keycloak's documented dynamic backchannel; issuer/frontchannel
            # remain fixed. Only our named local Compose service is allowed.
            metadata = urlsplit(self.metadata_url)
            if (
                not self.policy.development
                or metadata.scheme != "http"
                or metadata.hostname != "keycloak"
                or metadata.port != 8080
                or metadata.username
                or metadata.password
                or metadata.query
                or metadata.fragment
                or metadata.path != parts.path + "/.well-known/openid-configuration"
            ):
                raise ValueError("unsupported_oidc_backchannel_configuration")


@dataclass(frozen=True)
class Principal:
    actor: Actor
    authenticated: bool


class AuthService:
    def __init__(self, factory: sessionmaker, config: AuthConfig) -> None:
        self.factory, self.config = factory, config

    def principal(self, request: Request, *, private: bool = False) -> Principal:
        mutation = request.method not in {"GET", "HEAD", "OPTIONS"}
        if mutation and request.headers.get("origin") != self.config.policy.origin:
            raise DomainError("same_origin_required", 403)
        with self.factory.begin() as db:
            actor = authenticate(
                db,
                request.cookies.get("citequest_session"),
                csrf=request.headers.get("x-csrf-token"),
                mutation=mutation,
            )
            is_guest = db.scalar(select(GuestOwner.id).where(GuestOwner.owner_id == actor.owner_id)) is not None
            if private and is_guest:
                raise DomainError("login_required", 401)
            return Principal(actor, not is_guest)

    def public_actor(self, request: Request) -> Actor:
        return self.principal(request).actor

    def private_actor(self, request: Request) -> Actor:
        return self.principal(request, private=True).actor

    def _cookies(self, response: Response, token: str, csrf: str) -> None:
        options = {"max_age": 28800, "path": "/", "secure": not self.config.policy.development, "samesite": "lax"}
        response.set_cookie("citequest_session", token, httponly=True, **options)
        response.set_cookie("citequest_csrf", csrf, httponly=False, **options)

    def session(self, request: Request) -> JSONResponse:
        token = request.cookies.get("citequest_session")
        csrf = request.cookies.get("citequest_csrf")
        with self.factory.begin() as db:
            try:
                actor = authenticate(db, token, csrf=csrf, mutation=True)
            except DomainError:
                # An invalid *present* session must not silently switch identity.
                if token:
                    response = JSONResponse({"code": "session_required"}, status_code=401)
                    response.delete_cookie("citequest_session", path="/")
                    response.delete_cookie("citequest_csrf", path="/")
                    return response
                db.execute(text("SELECT pg_advisory_xact_lock(73419307228001)"))
                recent = db.scalar(
                    select(func.count())
                    .select_from(GuestOwner)
                    .where(GuestOwner.created_at > func.now() - text("interval '1 minute'"))
                )
                if recent >= 120:
                    raise DomainError("anonymous_session_capacity_reached", 429) from None
                owner = Owner(name="Anonymous search", can_publish=False)
                db.add(owner)
                db.flush()
                db.add(GuestOwner(owner_id=owner.id))
                actor = Actor(owner.id, publisher=False)
                token, csrf = issue_session(db, actor, timedelta(hours=8))
            guest = db.scalar(select(GuestOwner.id).where(GuestOwner.owner_id == actor.owner_id)) is not None
            project = (
                None
                if guest
                else db.scalar(
                    select(Project).where(Project.owner_id == actor.owner_id).order_by(Project.created_at).limit(1)
                )
            )
            payload = {
                "authenticated": not guest,
                "owner_id": str(actor.owner_id),
                "csrf": csrf,
                "project_id": str(project.id) if project else None,
            }
        response = JSONResponse(payload)
        response.headers["Cache-Control"] = "no-store"
        self._cookies(response, token, csrf)
        return response

    def bind(self, claims: dict) -> tuple[str, str]:
        if (
            claims.get("iss") != self.config.issuer
            or not isinstance(claims.get("sub"), str)
            or not 1 <= len(claims["sub"]) <= 500
        ):
            raise DomainError("oidc_identity_invalid", 401)
        with self.factory.begin() as db:
            lock = int.from_bytes(
                hashlib.sha256((self.config.issuer + "\0" + claims["sub"]).encode()).digest()[:8], signed=True
            )
            db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock})
            identity = db.scalar(
                select(OIDCIdentity).where(
                    OIDCIdentity.issuer == self.config.issuer, OIDCIdentity.subject == claims["sub"]
                )
            )
            if identity is None:
                # Email is display data only; never an account-linking key.
                owner = Owner(
                    name=str(claims.get("name") or claims.get("preferred_username") or "Researcher")[:500],
                    can_publish=False,
                )
                db.add(owner)
                db.flush()
                identity = OIDCIdentity(owner_id=owner.id, issuer=self.config.issuer, subject=claims["sub"])
                db.add(identity)
                db.add(Project(owner_id=owner.id, title="我的研究工作区"))
            return issue_session(db, Actor(identity.owner_id, publisher=False))

    def router(self) -> APIRouter:
        router = APIRouter()
        oauth = OAuth()
        oauth.register(
            "research",
            client_id=self.config.client_id,
            client_secret=self.config.client_secret,
            server_metadata_url=self.config.metadata_url or self.config.issuer + "/.well-known/openid-configuration",
            client_kwargs={"scope": "openid profile", "code_challenge_method": "S256", "timeout": 15},
        )
        client = oauth.create_client("research")

        @router.get("/api/auth/session")
        async def session(request: Request) -> JSONResponse:
            return await run_in_threadpool(self.session, request)

        @router.get("/auth/login")
        async def login(request: Request) -> Response:
            request.session.clear()  # one outstanding authorization flow per browser
            return await client.authorize_redirect(request, self.config.policy.origin + "/auth/callback")

        @router.get("/auth/callback")
        async def callback(request: Request) -> Response:
            try:
                token = await client.authorize_access_token(request, leeway=30)
                claims = dict(token.get("userinfo") or {})
                app_token, csrf = await run_in_threadpool(self.bind, claims)
            except (OAuthError, JoseError, ValueError, KeyError, DomainError, httpx2.HTTPError):
                request.session.clear()
                return JSONResponse({"code": "oidc_login_failed"}, status_code=401)
            request.session.clear()
            response = RedirectResponse("/", status_code=303)
            response.headers["Cache-Control"] = "no-store"
            self._cookies(response, app_token, csrf)
            return response

        @router.post("/api/auth/logout")
        async def logout(request: Request) -> JSONResponse:
            self.principal(request)
            with self.factory.begin() as db:
                db.execute(
                    update(LocalSession)
                    .where(LocalSession.token_sha256 == token_hash(request.cookies["citequest_session"]))
                    .values(revoked=True)
                )
            response = JSONResponse({"logged_out": True})
            response.delete_cookie("citequest_session", path="/")
            response.delete_cookie("citequest_csrf", path="/")
            return response

        return router
