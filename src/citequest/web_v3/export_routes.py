"""Minimal composition seam; app/main retain routing and deployment ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter

from citequest.exports_v3.api import build_router
from citequest.exports_v3.product import ExportDomain

if TYPE_CHECKING:
    from citequest.web_v3.auth import AuthService


def product_export_router(domain: ExportDomain, auth: AuthService) -> APIRouter:
    """Use the same opaque-session + Origin/CSRF dependency as Agent writes.

    Build the domain with exports_v3.product.build_export_domain and current
    product services. Request handling enqueues only; no renderer/model runs in
    this router. Missing candidate migration is an explicit 503 on use.
    """
    return build_router(domain.service, auth.private_actor)
