"""Official httpcore parser with explicitly injected synthetic wire bytes."""

from contextlib import asynccontextmanager

import anyio
import httpcore
import pytest
from jsonschema import Draft202012Validator, FormatChecker

from citequest.materials_v2.reference_fetch import FetchContext, FetchError, PinnedStream
from citequest.materials_v2.text import MaterialError
from citequest.updates_v2 import transport


class Gate:
    production = True

    def __init__(self):
        self.entries = []
        self.audits = []

    def record_reference_transport(self, receipt):
        self.audits.append(dict(receipt))

    @asynccontextmanager
    async def permit(self, provider, url, context):
        receipt = {'provider': provider, 'url': url}
        self.entries.append(receipt)
        yield receipt


def fetch(monkeypatch, messages, *, context=None, maximum_bytes=100, maximum_hops=4):
    gate = Gate()
    source = iter(messages)
    monkeypatch.setattr(transport, 'UpdateBackend', lambda *args: httpcore.AsyncMockBackend([next(source)]))
    ctx = context or FetchContext.start(10)
    async def run():
        return await transport.SourceFetcher(gate).fetch_page('https://export.arxiv.org/api/query', maximum_bytes=maximum_bytes,
                                                              context=ctx, accept=b'application/xml', maximum_hops=maximum_hops)
    return anyio.run(run), ctx, gate


def test_boundaries_bytes_mime_and_shared_redirect_budget(monkeypatch):
    value, _, gate = fetch(monkeypatch, [b'HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\nContent-Length: 4\r\n\r\ndata'])
    assert value.complete and value.body == b'data' and gate.entries[0]['outcome']['body_bytes'] == 4
    ctx = FetchContext.start(10)
    ctx.redirects_used = 2  # e.g. metadata already used two, PDF shares the same context.
    redirect = b'HTTP/1.1 302 Found\r\nLocation: /api/query?next=1\r\nContent-Length: 0\r\n\r\n'
    value, ctx, gate = fetch(monkeypatch, [redirect, redirect], context=ctx)
    assert value.error == 'SOURCE_REDIRECT_LIMIT_OR_LOCATION' and ctx.redirects_used == 3 and len(gate.entries) == 2


def test_cross_provider_redirect_is_rejected_before_other_provider_gate(monkeypatch):
    value, _, gate = fetch(monkeypatch, [b'HTTP/1.1 302 Found\r\nLocation: https://api.crossref.org/works/10.9999/a\r\nContent-Length: 0\r\n\r\n'])
    assert value.error == 'SOURCE_CROSS_PROVIDER_REDIRECT_REJECTED' and len(gate.entries) == 1


def test_final_physical_hop_never_sends_another_redirect_request(monkeypatch):
    value, _, gate = fetch(monkeypatch, [b'HTTP/1.1 302 Found\r\nLocation: /api/query?next=1\r\nContent-Length: 0\r\n\r\n'], maximum_hops=1)
    assert value.error == 'SOURCE_PHYSICAL_HTTP_HOP_BUDGET_EXHAUSTED' and len(gate.entries) == 1


@pytest.mark.parametrize('message,code', [
    (b'HTTP/1.1 200 OK\r\nContent-Length: 101\r\n\r\n', 'SOURCE_RESPONSE_BYTES_EXCEEDED'),
    (b'HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: 0\r\n\r\n', 'SOURCE_COMPRESSED_BODY_REJECTED'),
    (b'HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nshort', 'RemoteProtocolError'),
])
def test_partial_or_bad_wire_is_never_success(monkeypatch, message, code):
    result, _, _ = fetch(monkeypatch, [message])
    assert not result.complete and result.error == code


def test_validated_address_setup_only_no_second_DNS_or_HTTP_replay():
    class Stream:
        def get_extra_info(self, name):
            return ('1.1.1.1', 443) if name == 'server_addr' else None

    class Delegate:
        def __init__(self):
            self.ips = []

        async def connect_tcp(self, ip, port, **kwargs):
            self.ips.append(ip)
            if len(self.ips) == 1:
                raise httpcore.ConnectError('explicit synthetic TCP failure before any headers')
            return Stream()

    async def run():
        delegate, receipt = Delegate(), {}
        backend = transport.UpdateBackend(FetchContext.start(10), receipt, delegate=delegate)
        async def fixed(host):
            return ('93.184.216.34', '1.1.1.1')
        backend.resolve = fixed
        result = await backend.connect_tcp('api.crossref.org', 443)
        assert isinstance(result, PinnedStream) and result.hostname == 'api.crossref.org' and result.address == '1.1.1.1'
        assert delegate.ips == ['93.184.216.34', '1.1.1.1'] and len(receipt['address_connection_attempts']) == 2
        assert all(row['HTTP_headers_sent'] is False for row in receipt['address_connection_attempts'])
        with pytest.raises(FetchError):
            await backend.connect_tcp('attacker.example', 443)
    anyio.run(run)


def test_mixed_public_private_DNS_rejected_before_connect(monkeypatch):
    async def mixed(*args, **kwargs):
        return [(None, None, None, None, ('93.184.216.34', 443)), (None, None, None, None, ('127.0.0.1', 443))]
    monkeypatch.setattr(anyio, 'getaddrinfo', mixed)
    async def run():
        with pytest.raises(MaterialError, match='UNSAFE_RESOLVED_ADDRESS'):
            await transport.UpdateBackend(FetchContext.start(10), {}).connect_tcp('api.crossref.org', 443)
    anyio.run(run)


def test_reference_v1_projection_preserves_full_new_transport_audit(monkeypatch):
    from citequest.materials_v2.reference_resolution import reference_contract
    gate = Gate()
    monkeypatch.setattr(transport, 'UpdateBackend', lambda *args: httpcore.AsyncMockBackend([b'HTTP/1.1 200 OK\r\nContent-Length: 4\r\nContent-Type: application/xml\r\n\r\ndata']))
    context = FetchContext.start(10)
    async def run():
        result = await transport.SourceFetcher(gate).fetch('https://export.arxiv.org/api/query', maximum_bytes=100, context=context, accept=b'application/xml')
        assert result.body == b'data'
    anyio.run(run)
    assert len(gate.audits) == 1 and gate.audits[0]['gate'] and gate.audits[0]['request_headers_sent']
    schema = reference_contract()['$defs']['ReferenceResolution']['properties']['provenance']['properties']['network_receipts']['items']
    assert Draft202012Validator(schema, format_checker=FormatChecker()).is_valid(context.receipts[0])
    assert 'gate' not in context.receipts[0] and 'headers' not in context.receipts[0]
