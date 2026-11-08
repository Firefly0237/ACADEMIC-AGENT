"""Pure transport security controls, not claims of real Internet availability."""

from __future__ import annotations

import asyncio
import socket
import ssl
from contextlib import asynccontextmanager

import pytest

from citequest.materials_v2 import reference_fetch as fetch
from citequest.materials_v2.text import MaterialError


@pytest.mark.parametrize(
    "url",
    [
        "http://api.crossref.org/works/x",
        "https://127.0.0.1/works/x",
        "https://api.crossref.org.evil.test/works/x",
        "https://u:p@api.crossref.org/works/x",
        "https://api.crossref.org:444/works/x",
        "https://api.crossref.org/other",
        "https://api.crossref.org/works/x?y=z",
        "https://arxiv.org/admin",
        "https://arxiv.org/pdf/x#f",
    ],
)
def test_allowlist_rejects_unsafe_url_before_socket(url):
    with pytest.raises(MaterialError):
        fetch.allowed_url(url)


def test_all_dns_answers_must_be_global_before_delegate_connect(monkeypatch):
    class Delegate:
        async def connect_tcp(self, *args, **kwargs):
            raise AssertionError("unsafe destination connected")

    async def addresses(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in ["1.1.1.1", "127.0.0.1"]]

    monkeypatch.setattr(fetch.anyio, "getaddrinfo", addresses)

    async def run():
        with pytest.raises(MaterialError, match="UNSAFE_RESOLVED_ADDRESS"):
            await fetch.PinnedBackend(fetch.FetchContext.start(), {}, delegate=Delegate()).connect_tcp(
                "api.crossref.org", 443
            )

    asyncio.run(run())


def test_numeric_socket_and_original_tls_hostname_with_normal_verification(monkeypatch):
    class Stream:
        def get_extra_info(self, name):
            return ("1.1.1.1", 443)

        async def start_tls(self, context, server_hostname, timeout):
            assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
            assert server_hostname == "api.crossref.org"
            return self

        async def aclose(self):
            pass

    class Delegate:
        async def connect_tcp(self, host, port, **kwargs):
            assert host == "1.1.1.1" and port == 443
            return Stream()

    async def addresses(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))]

    monkeypatch.setattr(fetch.anyio, "getaddrinfo", addresses)

    async def run():
        receipt = {}
        stream = await fetch.PinnedBackend(fetch.FetchContext.start(), receipt, delegate=Delegate()).connect_tcp(
            "api.crossref.org", 443
        )
        await stream.start_tls(ssl.create_default_context(), server_hostname="api.crossref.org")
        assert receipt["connected_ip"] == "1.1.1.1" and receipt["tls_certificate_verified"] is True
        with pytest.raises(MaterialError, match="TLS_NAME"):
            await stream.start_tls(ssl.create_default_context(), server_hostname="evil.test")

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode,code",
    [
        ("declared", "BYTES_EXCEEDED"),
        ("stream", "BYTES_EXCEEDED"),
        ("compression", "COMPRESSED_BODY"),
        ("redirect_private", "URL_REJECTED"),
        ("redirect_limit", "REDIRECT_LIMIT"),
        ("protocol", "NETWORK_UNAVAILABLE"),
    ],
)
def test_actual_response_and_redirect_boundaries_no_retries(monkeypatch, mode, code):
    calls = []

    class Response:
        status = 302 if mode.startswith("redirect") else 200
        headers = {
            "declared": [(b"content-length", b"6")],
            "compression": [(b"content-encoding", b"gzip")],
            "redirect_private": [(b"location", b"https://127.0.0.1/x")],
            "redirect_limit": [(b"location", b"https://api.crossref.org/works/x")],
        }.get(mode, [])

        async def aiter_stream(self):
            yield b"123"
            if mode == "protocol":
                raise fetch.httpcore.RemoteProtocolError("synthetic incomplete chunked body")
            yield b"456"

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["retries"] == 0 and kwargs["max_connections"] == 1

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        @asynccontextmanager
        async def stream(self, *args, **kwargs):
            calls.append(args)
            assert (b"Accept-Encoding", b"identity") in kwargs["headers"]
            yield Response()

    async def sleep(seconds):
        pass

    monkeypatch.setattr(fetch.httpcore, "AsyncConnectionPool", Pool)
    monkeypatch.setattr(fetch.anyio, "sleep", sleep)

    async def run():
        context = fetch.FetchContext.start()
        with pytest.raises(MaterialError, match=code):
            await fetch.OfficialFetcher().fetch(
                "https://api.crossref.org/works/x",
                maximum_bytes=5,
                context=context,
                accept=b"application/json",
            )
        assert len(calls) == (4 if mode == "redirect_limit" else 1)
        if mode == "protocol":
            assert context.receipts[0]["error_type"] == "RemoteProtocolError"
            assert context.receipts[0]["body_complete"] is False and context.receipts[0]["body_sha256"] is None

    asyncio.run(run())
