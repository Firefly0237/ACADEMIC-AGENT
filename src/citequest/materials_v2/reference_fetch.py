"""Official httpcore transport with DNS-pinned sockets and unchanged TLS names.

No HTTP parser, proxy discovery, TLS bypass, Unix socket or generic URL fetcher.
Public interfaces: https://www.encode.io/httpcore/network-backends/
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
import ssl
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlsplit

import anyio
import httpcore

from .references import validate_resolved_ips
from .text import MaterialError

ALLOWED_HOSTS = frozenset({"api.crossref.org", "export.arxiv.org", "arxiv.org", "www.arxiv.org"})
USER_AGENT = b"CiteQuestResearch/0.4 (bounded academic reference validation)"


class FetchError(MaterialError):
    pass


def allowed_url(url: str) -> str:
    if (
        type(url) is not str
        or not url
        or len(url) > 4096
        or any(ord(c) < 33 or ord(c) == 127 for c in url)
        or "\\" in url
    ):
        raise FetchError("REFERENCE_URL_REJECTED")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise FetchError("REFERENCE_URL_REJECTED") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise FetchError("REFERENCE_URL_REJECTED")
    if parsed.hostname == "api.crossref.org" and (not parsed.path.startswith("/works/") or parsed.query):
        raise FetchError("REFERENCE_ENDPOINT_REJECTED")
    if parsed.hostname == "export.arxiv.org" and not (parsed.path == "/api/query" or parsed.path.startswith("/pdf/")):
        raise FetchError("REFERENCE_ENDPOINT_REJECTED")
    if parsed.hostname in {"arxiv.org", "www.arxiv.org"} and not parsed.path.startswith(("/pdf/", "/abs/")):
        raise FetchError("REFERENCE_ENDPOINT_REJECTED")
    return url


@dataclass
class FetchContext:
    deadline: float
    redirects_used: int = 0
    receipts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def start(cls, seconds: float = 30) -> FetchContext:
        if type(seconds) not in {float, int} or not 0 < seconds <= 30:
            raise ValueError("Reference total deadline cannot exceed30s")
        return cls(time.monotonic() + seconds)

    def remaining(self) -> float:
        seconds = self.deadline - time.monotonic()
        if seconds <= 0:
            raise FetchError("REFERENCE_TOTAL_DEADLINE_EXCEEDED")
        return seconds


class PinnedStream(httpcore.AsyncNetworkStream):
    def __init__(
        self, stream: httpcore.AsyncNetworkStream, *, hostname: str, address: str, context: FetchContext, receipt: dict
    ) -> None:
        self.stream, self.hostname, self.address = stream, hostname, address
        self.context, self.receipt = context, receipt

    def timeout(self, requested: float | None) -> float:
        remaining = self.context.remaining()
        return min(remaining, requested) if requested is not None else remaining

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        with anyio.fail_after(self.timeout(timeout)):
            return await self.stream.read(max_bytes, timeout=self.timeout(timeout))

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        with anyio.fail_after(self.timeout(timeout)):
            await self.stream.write(buffer, timeout=self.timeout(timeout))

    async def aclose(self) -> None:
        await self.stream.aclose()

    async def start_tls(
        self, ssl_context: ssl.SSLContext, server_hostname: str | None = None, timeout: float | None = None
    ) -> httpcore.AsyncNetworkStream:
        if (
            server_hostname != self.hostname
            or not ssl_context.check_hostname
            or ssl_context.verify_mode != ssl.CERT_REQUIRED
        ):
            await self.aclose()
            raise FetchError("REFERENCE_TLS_NAME_OR_VERIFICATION_REJECTED")
        with anyio.fail_after(self.timeout(timeout)):
            upgraded = await self.stream.start_tls(
                ssl_context, server_hostname=server_hostname, timeout=self.timeout(timeout)
            )
        peer = upgraded.get_extra_info("server_addr")
        if not peer or ipaddress.ip_address(peer[0]) != ipaddress.ip_address(self.address):
            await upgraded.aclose()
            raise FetchError("REFERENCE_CONNECTED_PEER_CHANGED")
        self.receipt.update(
            tls_hostname=server_hostname, tls_certificate_verified=True, connected_ip=str(ipaddress.ip_address(peer[0]))
        )
        return PinnedStream(
            upgraded, hostname=self.hostname, address=self.address, context=self.context, receipt=self.receipt
        )

    def get_extra_info(self, info: str) -> Any:
        return self.stream.get_extra_info(info)


class PinnedBackend(httpcore.AsyncNetworkBackend):
    def __init__(
        self, context: FetchContext, receipt: dict, *, delegate: httpcore.AsyncNetworkBackend | None = None
    ) -> None:
        self.context, self.receipt = context, receipt
        self.delegate = delegate or httpcore.AnyIOBackend()

    async def resolve(self, hostname: str) -> tuple[str, ...]:
        with anyio.fail_after(self.context.remaining()):
            records = await anyio.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        return validate_resolved_ips(list(dict.fromkeys(record[4][0] for record in records)))

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        if host not in ALLOWED_HOSTS or port != 443 or local_address is not None:
            raise FetchError("REFERENCE_SOCKET_DESTINATION_REJECTED")
        addresses = await self.resolve(host)
        # One selected numeric IP, never a second hostname lookup. Prefer IPv4
        # where both families exist; no automatic HTTP or connection retry.
        selected = sorted(addresses, key=lambda value: ipaddress.ip_address(value).version)[0]
        self.receipt.update(
            hostname=host, resolved_ips=list(addresses), dns_all_addresses_safe=True, selected_ip=selected
        )
        budget = min(self.context.remaining(), timeout) if timeout is not None else self.context.remaining()
        with anyio.fail_after(budget):
            stream = await self.delegate.connect_tcp(selected, 443, timeout=budget, socket_options=socket_options)
        peer = stream.get_extra_info("server_addr")
        if not peer or ipaddress.ip_address(peer[0]) != ipaddress.ip_address(selected):
            await stream.aclose()
            raise FetchError("REFERENCE_CONNECTED_PEER_CHANGED")
        return PinnedStream(stream, hostname=host, address=selected, context=self.context, receipt=self.receipt)

    async def connect_unix_socket(self, *args: Any, **kwargs: Any) -> httpcore.AsyncNetworkStream:
        raise FetchError("REFERENCE_UNIX_SOCKET_FORBIDDEN")

    async def sleep(self, seconds: float) -> None:
        with anyio.fail_after(self.context.remaining()):
            await anyio.sleep(seconds)


@dataclass(frozen=True)
class FetchResult:
    url: str
    status: int
    content_type: str
    body: bytes


class OfficialFetcher:
    """One process-local outbound slot; provider spacing spans requests/redirects."""

    def __init__(self) -> None:
        self.slot = anyio.CapacityLimiter(1)
        self.last_started: dict[str, float] = {}

    async def fetch(self, url: str, *, maximum_bytes: int, context: FetchContext, accept: bytes) -> FetchResult:
        borrower = object()
        try:
            self.slot.acquire_on_behalf_of_nowait(borrower)
        except anyio.WouldBlock as exc:
            raise FetchError("REFERENCE_NETWORK_BUSY") from exc
        try:
            return await self._fetch(url, maximum_bytes=maximum_bytes, context=context, accept=accept)
        finally:
            self.slot.release_on_behalf_of(borrower)

    async def _fetch(self, url: str, *, maximum_bytes: int, context: FetchContext, accept: bytes) -> FetchResult:
        if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= 50 * 1024 * 1024:
            raise ValueError("Invalid bounded fetch size")
        for _ in range(4):
            url = allowed_url(url)
            host = urlsplit(url).hostname
            provider = "arxiv" if host != "api.crossref.org" else "crossref"
            gap = 3 if provider == "arxiv" else 1
            wait = self.last_started.get(provider, 0) + gap - time.monotonic()
            if wait > 0:
                with anyio.fail_after(context.remaining()):
                    await anyio.sleep(wait)
            self.last_started[provider] = time.monotonic()
            receipt: dict[str, Any] = {
                "url": url,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "status": None,
                "body_complete": False,
                "body_bytes": 0,
                "body_sha256": None,
                "ambient_proxy_used": False,
                "automatic_retries": 0,
            }
            context.receipts.append(receipt)
            backend = PinnedBackend(context, receipt)
            try:
                with anyio.fail_after(context.remaining()):
                    async with httpcore.AsyncConnectionPool(
                        network_backend=backend,
                        max_connections=1,
                        max_keepalive_connections=0,
                        retries=0,
                        http1=True,
                        http2=False,
                    ) as pool:
                        timeout = min(10.0, context.remaining())
                        async with pool.stream(
                            "GET",
                            url,
                            headers=[
                                (b"User-Agent", USER_AGENT),
                                (b"Accept", accept),
                                (b"Accept-Encoding", b"identity"),
                            ],
                            extensions={
                                "timeout": {"connect": timeout, "read": timeout, "write": timeout, "pool": timeout}
                            },
                        ) as response:
                            receipt["status"] = response.status
                            headers = {name.lower(): value for name, value in response.headers}
                            if response.status in {301, 302, 303, 307, 308}:
                                if context.redirects_used >= 3 or b"location" not in headers:
                                    raise FetchError("REFERENCE_REDIRECT_LIMIT_OR_LOCATION")
                                context.redirects_used += 1
                                url = allowed_url(urljoin(url, headers[b"location"].decode("ascii")))
                                receipt["redirect_to"] = url
                                continue
                            if headers.get(b"content-encoding", b"identity").lower() != b"identity":
                                raise FetchError("REFERENCE_COMPRESSED_BODY_REJECTED")
                            length = headers.get(b"content-length")
                            if length is not None and (
                                not re.fullmatch(rb"[0-9]{1,12}", length) or int(length) > maximum_bytes
                            ):
                                raise FetchError("REFERENCE_RESPONSE_BYTES_EXCEEDED")
                            body = bytearray()
                            async for part in response.aiter_stream():
                                if len(body) + len(part) > maximum_bytes:
                                    raise FetchError("REFERENCE_RESPONSE_BYTES_EXCEEDED")
                                body.extend(part)
                                receipt["body_bytes"] = len(body)
                            receipt.update(body_complete=True, body_sha256=hashlib.sha256(body).hexdigest())
                            content_type = (
                                headers.get(b"content-type", b"")
                                .decode("ascii", errors="strict")
                                .split(";", 1)[0]
                                .strip()
                                .lower()
                            )
                            return FetchResult(url, response.status, content_type, bytes(body))
            except (
                httpcore.NetworkError,
                httpcore.TimeoutException,
                httpcore.ProtocolError,
                TimeoutError,
                OSError,
            ) as exc:
                receipt["error_type"] = type(exc).__name__
                receipt["error_detail"] = str(exc)[:300]
                code = (
                    "REFERENCE_TOTAL_DEADLINE_EXCEEDED"
                    if time.monotonic() >= context.deadline
                    else "REFERENCE_NETWORK_UNAVAILABLE"
                )
                raise FetchError(code) from exc
        raise FetchError("REFERENCE_REDIRECT_LIMIT_OR_LOCATION")
