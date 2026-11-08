"""Official httpcore, pinned sockets, and a per-hop injectable shared gate."""

from __future__ import annotations

import hashlib
import ipaddress
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlsplit

import anyio
import httpcore
from citequest.materials_v2.text import MaterialError

from citequest.materials_v2.reference_fetch import (
    ALLOWED_HOSTS,
    FetchContext,
    FetchError,
    FetchResult,
    PinnedBackend,
    PinnedStream,
    allowed_url,
)

OAI_HOST = "oaipmh.arxiv.org"
SAFE_HEADERS = {b"date", b"age", b"etag", b"last-modified", b"retry-after", b"cache-control", b"content-type", b"content-length"}


def update_url(url: str, *, references: bool = False) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise FetchError("UPDATE_ENDPOINT_REJECTED") from exc
    if parsed.hostname != OAI_HOST:
        value = allowed_url(url)
        if not references and urlsplit(value).hostname in {"arxiv.org", "www.arxiv.org"}:
            raise FetchError("UPDATE_FULLTEXT_NOT_ALLOWED")
        return value
    if (references or parsed.scheme != "https" or parsed.path != "/oai" or port not in {None, 443}
            or parsed.username is not None or parsed.password is not None or parsed.fragment
            or "\\" in url or any(ord(c) < 33 or ord(c) == 127 for c in url) or len(url) > 8192):
        raise FetchError("UPDATE_OAI_ENDPOINT_REJECTED")
    return url


def provider(url: str) -> str:
    return "crossref" if urlsplit(url).hostname == "api.crossref.org" else "arxiv"


class UpdateBackend(PinnedBackend):
    async def connect_tcp(self, host: str, port: int, timeout: float | None = None,
                          local_address: str | None = None, socket_options: Any = None) -> httpcore.AsyncNetworkStream:
        if host not in ALLOWED_HOSTS | {OAI_HOST} or port != 443 or local_address is not None:
            raise FetchError("UPDATE_SOCKET_DESTINATION_REJECTED")
        addresses = await self.resolve(host)
        if not 1 <= len(addresses) <= 8:
            raise FetchError("UPDATE_DNS_ADDRESS_CAP_EXCEEDED")
        self.receipt.update(hostname=host, resolved_ips=list(addresses), dns_all_addresses_safe=True, address_connection_attempts=[])
        # Only TCP setup traverses the one validated DNS result. No TLS failure,
        # headers-sent failure, or HTTP status/response is replayed.
        last_error = None
        for selected in sorted(addresses, key=lambda value: ipaddress.ip_address(value).version):
            self.receipt["selected_ip"] = selected
            attempt = {"ip": selected, "HTTP_headers_sent": False}
            self.receipt["address_connection_attempts"].append(attempt)
            budget = min(3.0, self.context.remaining(), timeout or 3.0)
            try:
                with anyio.fail_after(budget):
                    stream = await self.delegate.connect_tcp(selected, 443, timeout=budget, socket_options=socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout, TimeoutError, OSError) as exc:
                attempt["error_type"] = type(exc).__name__
                last_error = exc
                continue
            peer = stream.get_extra_info("server_addr")
            if not peer or ipaddress.ip_address(peer[0]) != ipaddress.ip_address(selected):
                await stream.aclose()
                raise FetchError("UPDATE_CONNECTED_PEER_CHANGED")
            attempt["connected"] = True
            return PinnedStream(stream, hostname=host, address=selected, context=self.context, receipt=self.receipt)
        raise FetchError("UPDATE_ALL_VALIDATED_TCP_ADDRESSES_UNAVAILABLE") from last_error


@dataclass(frozen=True)
class PageResult:
    url: str
    status: int | None
    content_type: str
    body: bytes
    complete: bool
    headers: dict[str, str]
    error: str | None


class CanaryGate:
    """Isolated protocol probe only; production assembly rejects this gate."""
    production = False

    def __init__(self) -> None:
        self.last: dict[str, float] = {}
        self.slot = anyio.Lock()

    @asynccontextmanager
    async def permit(self, name: str, url: str, context: FetchContext) -> AsyncIterator[dict]:
        async with self.slot:
            wait = self.last.get(name, 0) + (3.1 if name == "arxiv" else 1.0) - time.monotonic()
            if wait > 0:
                with anyio.fail_after(context.remaining()):
                    await anyio.sleep(wait)
            self.last[name] = time.monotonic()
            receipt = {"gate_mode": "protocol_canary_local_only", "provider": name}
            yield receipt


class SourceFetcher:
    """No HTTP parser/proxy/retry; secure transport policy is independent of ETL."""

    def __init__(self, gate: Any, *, references: bool = False, require_production_gate: bool = True,
                 maximum_metadata_response_bytes: int = 2 * 1024 * 1024) -> None:
        if require_production_gate and not getattr(gate, "production", False):
            raise ValueError("shared_production_source_gate_required")
        if type(maximum_metadata_response_bytes) is not int or not 1 <= maximum_metadata_response_bytes <= 8 * 1024 * 1024:
            raise ValueError("explicit_metadata_response_cap_required")
        self.gate, self.references, self.maximum_metadata_response_bytes = gate, references, maximum_metadata_response_bytes

    async def fetch_page(self, url: str, *, maximum_bytes: int, context: FetchContext, accept: bytes, maximum_hops: int = 4) -> PageResult:
        if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= (50 * 1024 * 1024 if self.references else self.maximum_metadata_response_bytes):
            raise ValueError("source_response_cap_required")
        if type(maximum_hops) is not int or not 1 <= maximum_hops <= 4:
            raise ValueError('bounded_physical_HTTP_hops_required')
        original_provider = provider(update_url(url, references=self.references))
        for redirect in range(maximum_hops):
            update_url(url, references=self.references)
            current_provider = provider(url)
            if current_provider != original_provider:
                raise FetchError("SOURCE_CROSS_PROVIDER_REDIRECT_REJECTED")
            body, headers, status, content_type = bytearray(), {}, None, ""
            receipt: dict[str, Any] = {"url": url, "retrieved_at": datetime.now(UTC).isoformat(), "status": None,
                                       "body_bytes": 0, "body_complete": False, "body_sha256": None,
                                       "ambient_proxy_used": False, "automatic_retries": 0, "request_headers_sent": False}
            context.receipts.append(receipt)

            async def trace(event: str, info: dict, *, target: dict = receipt) -> None:
                if event.endswith("send_request_headers.complete"):
                    target["request_headers_sent"] = True

            try:
                async with self.gate.permit(current_provider, url, context) as gate_receipt:
                    receipt["gate"] = gate_receipt
                    backend = UpdateBackend(context, receipt)
                    with anyio.fail_after(context.remaining()):
                        async with httpcore.AsyncConnectionPool(network_backend=backend, max_connections=1,
                                                               max_keepalive_connections=0, retries=0, http1=True, http2=False) as pool:
                            timeout = min(15.0, context.remaining())
                            async with pool.stream("GET", url, headers=[(b"User-Agent", b"CiteQuestResearch/0.5 (bounded source updates)"),
                                                                        (b"Accept", accept), (b"Accept-Encoding", b"identity")],
                                                   extensions={"timeout": dict.fromkeys(("connect", "read", "write", "pool"), timeout), "trace": trace}) as response:
                                status = receipt["status"] = response.status
                                raw_headers = {key.lower(): value for key, value in response.headers}
                                headers = {key.decode("ascii"): value.decode("latin-1") for key, value in raw_headers.items() if key in SAFE_HEADERS}
                                receipt["headers"] = headers
                                if status in {301, 302, 303, 307, 308}:
                                    if redirect + 1 >= maximum_hops:
                                        raise FetchError('SOURCE_PHYSICAL_HTTP_HOP_BUDGET_EXHAUSTED')
                                    if context.redirects_used >= 3 or b"location" not in raw_headers:
                                        raise FetchError("SOURCE_REDIRECT_LIMIT_OR_LOCATION")
                                    context.redirects_used += 1
                                    next_url = update_url(urljoin(url, raw_headers[b"location"].decode("ascii")), references=self.references)
                                    if provider(next_url) != original_provider:
                                        raise FetchError("SOURCE_CROSS_PROVIDER_REDIRECT_REJECTED")
                                    receipt["redirect_to"] = next_url
                                    gate_receipt["outcome"] = {"status": status, "redirect_response": True, "body_not_consumed": True,
                                        'tls_hostname': receipt.get('tls_hostname'), 'tls_certificate_verified': receipt.get('tls_certificate_verified'),
                                        'connected_ip': receipt.get('connected_ip')}
                                    url = next_url
                                    continue
                                if raw_headers.get(b"content-encoding", b"identity").lower() != b"identity":
                                    raise FetchError("SOURCE_COMPRESSED_BODY_REJECTED")
                                length = raw_headers.get(b"content-length")
                                if length is not None and (not length.isdigit() or len(length) > 12 or int(length) > maximum_bytes):
                                    raise FetchError("SOURCE_RESPONSE_BYTES_EXCEEDED")
                                async for part in response.aiter_stream():
                                    if len(body) + len(part) > maximum_bytes:
                                        raise FetchError("SOURCE_RESPONSE_BYTES_EXCEEDED")
                                    body.extend(part)
                                    receipt["body_bytes"] = len(body)
                                content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
                                receipt.update(body_complete=True, body_sha256=hashlib.sha256(body).hexdigest())
                    gate_receipt["outcome"] = {"status": status, "body_complete": True, "body_bytes": len(body), "body_sha256": receipt["body_sha256"],
                                                "tls_hostname": receipt.get("tls_hostname"), "tls_certificate_verified": receipt.get("tls_certificate_verified"),
                                                "connected_ip": receipt.get("connected_ip")}
                return PageResult(url, status, content_type, bytes(body), True, headers, None)
            except (httpcore.NetworkError, httpcore.TimeoutException, httpcore.ProtocolError, TimeoutError, OSError, MaterialError) as exc:
                receipt.update(error_type=type(exc).__name__, error=str(exc)[:300])
                return PageResult(url, status, content_type, bytes(body), receipt["body_complete"], headers, getattr(exc, "code", type(exc).__name__))
        raise FetchError("SOURCE_REDIRECT_LIMIT_OR_LOCATION")

    async def fetch(self, url: str, *, maximum_bytes: int, context: FetchContext, accept: bytes) -> FetchResult:
        """S04-compatible injected reference reader, with the S05 shared gate."""
        result = await self.fetch_page(url, maximum_bytes=maximum_bytes, context=context, accept=accept)
        from citequest.materials_v2.reference_resolution import reference_contract
        public_fields = reference_contract()['$defs']['ReferenceResolution']['properties']['provenance']['properties']['network_receipts']['items']['properties']
        audit_error = False
        for receipt in context.receipts:
            if receipt.get('gate'):
                try:
                    await anyio.to_thread.run_sync(self.gate.record_reference_transport, receipt)
                except Exception:
                    audit_error = True
        # This is an explicit compatibility projection. The complete S05 trace
        # remains attached to its real PG attempt UUID/fence before projection.
        context.receipts[:] = [{key: value for key, value in ({**receipt, **({'error_detail': receipt['error']} if 'error' in receipt else {})}).items()
                                if key in public_fields} for receipt in context.receipts]
        if audit_error:
            raise FetchError('SOURCE_REFERENCE_TRANSPORT_AUDIT_UNAVAILABLE')
        if not result.complete or result.error:
            raise FetchError(result.error or "SOURCE_NETWORK_UNAVAILABLE")
        return FetchResult(result.url, result.status, result.content_type, result.body)
