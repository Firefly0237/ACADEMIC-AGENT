"""Offline plans for only the two approved reference adapters; never fetch here."""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import quote, unquote, urlsplit

from .text import MaterialError


@dataclass(frozen=True)
class ReferencePlan:
    adapter: str
    raw_input: str
    identifier: str
    metadata_url: str
    fulltext_candidate_url: str | None
    follow_redirects: bool = False
    requires_network_verification: bool = True
    rights_status: str = "unknown"
    can_qa: bool = False


def _doi(raw: str, original: str) -> ReferencePlan:
    value = re.sub(r"^doi:\s*", "", raw, flags=re.I).strip().lower()
    if not re.fullmatch(r"10\.\d{4,9}/\S+", value) or any(ord(c) < 32 for c in value) or "\\" in value or len(value) > 512:
        raise MaterialError("INVALID_DOI")
    return ReferencePlan("doi_crossref", original, value, "https://api.crossref.org/works/" + quote(value, safe=""), None)


def _decode_path(value: str) -> str:
    if re.search(r"%(?![0-9a-fA-F]{2})", value):
        raise MaterialError("INVALID_REFERENCE_URL_ENCODING")
    try:
        return unquote(value, errors="strict")
    except UnicodeDecodeError as exc:
        raise MaterialError("INVALID_REFERENCE_URL_ENCODING") from exc


def plan_reference(value: str) -> ReferencePlan:
    if type(value) is not str or not value.strip() or any(ord(c) < 32 for c in value) or len(value) > 4096:
        raise MaterialError("INVALID_REFERENCE")
    original, value = value, value.strip()
    if value.lower().startswith("doi:") or value.startswith("10."):
        return _doi(value, original)
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise MaterialError("INVALID_REFERENCE_URL") from exc
    if (parts.scheme != "https" or parts.username or parts.password or port not in (None, 443)
            or parts.query or parts.fragment):
        raise MaterialError("UNSUPPORTED_REFERENCE_URL")
    host = (parts.hostname or "").lower()
    if host in {"doi.org", "dx.doi.org"}:
        return _doi(_decode_path(parts.path.lstrip("/")), original)
    if host not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        raise MaterialError("UNSUPPORTED_REFERENCE_ADAPTER")
    path = _decode_path(parts.path)
    if not path.startswith(("/abs/", "/pdf/")):
        raise MaterialError("UNSUPPORTED_ARXIV_PATH")
    identifier = path[5:].removesuffix(".pdf")
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z][a-z.-]+/\d{7})(?:v[1-9]\d*)?", identifier):
        raise MaterialError("INVALID_ARXIV_ID")
    return ReferencePlan("arxiv", original, identifier,
                         "https://export.arxiv.org/api/query?id_list=" + quote(identifier, safe=""),
                         "https://arxiv.org/pdf/" + quote(identifier, safe="/") + ".pdf")


def validate_resolved_ips(addresses: list[str]) -> tuple[str, ...]:
    """For the future transport: a validated DNS list alone does not pin a socket."""
    if not addresses:
        raise MaterialError("NO_RESOLVED_ADDRESS")
    result = []
    for raw in addresses:
        if "%" in raw:
            raise MaterialError("INVALID_RESOLVED_ADDRESS")
        try:
            address = ipaddress.ip_address(raw)
        except ValueError as exc:
            raise MaterialError("INVALID_RESOLVED_ADDRESS") from exc
        actual = address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped else address
        if (not actual.is_global or actual.is_loopback or actual.is_private or actual.is_link_local
                or actual.is_multicast or actual.is_reserved or actual.is_unspecified):
            raise MaterialError("UNSAFE_RESOLVED_ADDRESS")
        result.append(str(address))
    return tuple(result)
