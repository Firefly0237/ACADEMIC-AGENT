"""Deterministic CSL-JSON projection from currently authorized source metadata."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from .schemas import CSLItem

_TYPES = {
    "article": "article", "article-journal": "article-journal", "journal-article": "article-journal",
    "paper-conference": "paper-conference", "proceedings-article": "paper-conference", "inproceedings": "paper-conference",
    "book": "book", "chapter": "chapter", "book-chapter": "chapter", "incollection": "chapter",
    "thesis": "thesis", "dissertation": "thesis", "phdthesis": "thesis", "mastersthesis": "thesis",
    "report": "report", "techreport": "report", "manuscript": "manuscript", "dataset": "dataset", "document": "document",
}


def _text(value: object) -> str | None:
    return value if type(value) is str and value.strip() and not any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in value) else None


def _first(metadata: dict, *keys: str) -> str | None:
    return next((value for key in keys if (value := _text(metadata.get(key))) is not None), None)


def source_url(value: object) -> str | None:
    value = _text(value)
    if value is None:
        return None
    try:
        parsed = urlsplit(value)
        parsed.port  # Reject malformed/out-of-range explicit ports as well.
        if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password and not any(char.isspace() for char in value):
            return value
    except ValueError:
        pass
    return None


def project_csl(*, identifier: str, title: str | None, authors: list[str] | None, year: int | None,
                url: str | None, metadata: dict | None = None, verified_paper: bool = False) -> tuple[CSLItem, list[str]]:
    """No name splitting, publication-type inference, network or model completion.

    An already verified paper identity permits generic CSL ``article`` only.
    A journal/conference/book classification requires an explicit source field.
    Other records use generic ``document`` without asserting a publication kind.
    Missing publication classification remains visible even for a generic item.
    """
    metadata = metadata if type(metadata) is dict else {}
    explicit = _first(metadata, "type", "publication_type", "entry_type")
    item_type = _TYPES.get(explicit.casefold()) if explicit else None
    missing = []
    if item_type in {None, "article", "document"}:
        missing.append("publication_type")
    if item_type is None:
        item_type = "article" if verified_paper else "document"
    data = {"id": identifier}
    if item_type:
        data["type"] = item_type
    actual_title = _text(title)
    if actual_title:
        data["title"] = actual_title
    else:
        missing.append("title")
    if type(authors) is list and authors and all(_text(value) is not None for value in authors):
        data["author"] = [{"literal": value} for value in authors]
    else:
        missing.append("author")
    if type(year) is int and 1000 <= year <= 2200:
        data["issued"] = {"date-parts": [[year]]}
    else:
        missing.append("issued")
    doi = _first(metadata, "DOI", "doi")
    if doi and re.fullmatch(r"10\.\d{4,9}/\S+", doi, re.I):
        data["DOI"] = doi
    actual_url = source_url(url)
    if actual_url:
        data["URL"] = actual_url
    for target, keys in {
        "container-title": ("container-title", "container_title", "journal", "booktitle", "venue"),
        "volume": ("volume",), "issue": ("issue",), "page": ("page", "pages"),
        "publisher": ("publisher",),
    }.items():
        value = _first(metadata, *keys)
        if value:
            data[target] = value
    if item_type in {"article-journal", "paper-conference", "chapter"} and "container-title" not in data:
        missing.append("container-title")
    if item_type in {"book", "chapter", "report"} and "publisher" not in data:
        missing.append("publisher")
    return CSLItem.model_validate(data, strict=True), missing
