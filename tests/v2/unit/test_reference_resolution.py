"""Provider parsing/admission controls; responses below are synthetic fixtures."""

from __future__ import annotations

import asyncio
import json

import pytest
from defusedxml.common import DefusedXmlException

from citequest.materials_v2.reference_fetch import FetchResult
from citequest.materials_v2.reference_resolution import acquire, parse_arxiv, parse_crossref
from citequest.materials_v2.references import plan_reference
from citequest.v2.domain import DomainError
from citequest.v2.storage import CAS

DOI = "10.1234/test"
ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/1706.03762v1</id><title> Synthetic title </title><summary>Exact\nabstract</summary>
<author><name>A Person</name></author><published>2017-06-12T00:00:00Z</published></entry></feed>"""


def crossref():
    return {
        "status": "ok",
        "message-type": "work",
        "message": {
            "DOI": DOI,
            "title": ["Synthetic"],
            "author": [{"given": "First", "family": "Last"}, {}],
            "published": {"date-parts": [[2024, 1]]},
        },
    }


def test_crossref_null_unknowns_and_unmodified_provider_text():
    plan = plan_reference(DOI)
    result = parse_crossref(json.dumps(crossref()).encode(), plan, plan.metadata_url)
    assert result["authors"] == ["First Last", None] and result["year"] == 2024
    assert result["abstract"] is None and result["license_urls"] is None
    assert result["title"] == "Synthetic" and result["normalization_profile"] == "cq.reference-metadata/1"


@pytest.mark.parametrize("kind", ["identifier", "author", "date", "year_bool", "long_title", "license", "duplicate"])
def test_bad_crossref_is_not_guessed_or_truncated(kind):
    obj = crossref()
    message = obj["message"]
    if kind == "identifier":
        message["DOI"] = "10.9999/different"
    if kind == "author":
        message["author"] = "First Last"
    if kind == "date":
        message["published"]["date-parts"] = []
    if kind == "year_bool":
        message["published"]["date-parts"] = [[True]]
    if kind == "long_title":
        message["title"] = ["a" * 8193]
    if kind == "license":
        message["license"] = [{"URL": "javascript:alert(1)"}]
    raw = json.dumps(obj).encode() if kind != "duplicate" else b'{"status":"ok","status":"ok"}'
    with pytest.raises(ValueError):
        parse_crossref(raw, plan_reference(DOI), "https://api.crossref.org/works/10.1234%2Ftest")


def test_arxiv_version_and_exact_abstract_are_preserved():
    plan = plan_reference("https://arxiv.org/abs/1706.03762")
    result = parse_arxiv(ATOM, plan, plan.metadata_url)
    assert result["arxiv_id"] == "1706.03762v1" and result["abstract"] == "Exact\nabstract"
    assert result["title"] == " Synthetic title " and result["license_urls"] is None


@pytest.mark.parametrize(
    "data,identifier",
    [
        (ATOM, "1706.03762v2"),
        (ATOM.replace(b"1706.03762v1", b"1706.03763v1"), "1706.03762"),
        (ATOM.replace(b"1706.03762v1", b"1706.03762"), "1706.03762"),
        (ATOM.replace(b"</feed>", b"<entry/></feed>"), "1706.03762"),
        (ATOM.replace(b"1706.03762v1</id>", b"1706.03762v1?unexpected=1</id>"), "1706.03762"),
        (ATOM.replace(b"http://arxiv.org", b"http://user:secret@arxiv.org"), "1706.03762"),
    ],
)
def test_arxiv_wrong_or_missing_exact_version_rejected(data, identifier):
    plan = plan_reference("https://arxiv.org/abs/" + identifier)
    with pytest.raises(ValueError):
        parse_arxiv(data, plan, plan.metadata_url)


def test_arxiv_xml_entities_are_not_resolved():
    data = b'<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///C:/secret">]>' + ATOM
    plan = plan_reference("https://arxiv.org/abs/1706.03762")
    with pytest.raises(DefusedXmlException):
        parse_arxiv(data, plan, plan.metadata_url)


@pytest.mark.parametrize("pdf,expected", [(b"%PDF-1.7\nsynthetic", True), (b"<html>error</html>", False)])
def test_acquire_preserves_raw_metadata_and_only_queues_pdf_signature(tmp_path, pdf, expected):
    class Fetcher:
        async def fetch(self, url, **kwargs):
            if "/api/query?" in url:
                return FetchResult(url, 200, "application/atom+xml", ATOM)
            return FetchResult(url, 200, "application/pdf", pdf)

    plan = plan_reference("https://arxiv.org/abs/1706.03762")
    store = CAS(tmp_path / "cas")
    result = asyncio.run(acquire(plan, Fetcher(), store, "test:private"))
    assert store.read(result.metadata_blob) == ATOM
    assert (result.pdf is not None) == expected and result.metadata["arxiv_id"] == "1706.03762v1"
    if expected:
        assert store.read(result.pdf) == pdf
    else:
        assert result.reason == "reference_fulltext_pdf_signature_or_media_rejected"


def test_provider_deep_json_preserves_raw_but_becomes_structured_unavailable(tmp_path):
    data = b"[" * 20000 + b"]" * 20000

    class Fetcher:
        async def fetch(self, url, **kwargs):
            return FetchResult(url, 200, "application/json", data)

    plan = plan_reference(DOI)
    store = CAS(tmp_path / "cas")
    result = asyncio.run(acquire(plan, Fetcher(), store, "test:private"))
    assert result.metadata is None and result.pdf is None and store.read(result.metadata_blob) == data
    assert result.reason == "reference_metadata_invalid_recursionerror"


def test_cas_failure_is_a_storage_error_not_a_provider_verdict():
    class Fetcher:
        async def fetch(self, url, **kwargs):
            return FetchResult(url, 200, "application/json", json.dumps(crossref()).encode())

    class BrokenStore:
        def put(self, *args, **kwargs):
            raise OSError("synthetic storage unavailable")

    with pytest.raises(DomainError, match="reference_storage_unavailable"):
        asyncio.run(acquire(plan_reference(DOI), Fetcher(), BrokenStore(), "test:private"))
