"""Validate original metadata and verify pinned citeproc's actual safe output.

Pre-escaping source values double-escapes literal ampersands in the pinned
citeproc. Preserve their values, use its real escaping, then check the DOM.
"""

from __future__ import annotations

import copy
from pathlib import Path
from urllib.parse import urlsplit


def literal_csl(references: list[dict]) -> list[dict]:
    result = copy.deepcopy(references)
    for item in result:
        if not isinstance(item.get("title"), str) or not isinstance(item.get("author"), list):
            raise ValueError("source_CSL_title_and_authors_required")
        for author in item["author"]:
            if set(author) != {"literal"} or not isinstance(author["literal"], str):
                raise ValueError("source_literal_author_names_required")
        if item.get("URL"):
            url = item["URL"]
            if not isinstance(url, str) or any(ord(c) < 32 for c in url):
                raise ValueError("invalid_bibliographic_source_URL")
            parsed = urlsplit(url)
            if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("invalid_bibliographic_source_URL")
    return result


def assert_safe_export_html(path: Path) -> dict:
    from lxml import html

    root = html.fromstring(path.read_bytes(), parser=html.HTMLParser(no_network=True))
    if root.xpath("//script|//img|//iframe|//object|//embed|//base|//form|//input|//button|//link"):
        raise ValueError("export_HTML_active_or_remote_resource_element")
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        if any(name.lower().startswith("on") or name.lower() == "style" for name in node.attrib):
            raise ValueError("export_HTML_active_or_unapproved_inline_attribute")
        if node.tag.lower() == "meta" and node.get("http-equiv", "").lower() == "refresh":
            raise ValueError("export_HTML_refresh_rejected")
        href = node.get("href")
        if href and not href.startswith("#"):
            parsed = urlsplit(href)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("export_HTML_unsafe_link_protocol")
    return {
        "status": "passed",
        "active_DOM_nodes": 0,
        "escaping": "actual pinned citeproc; metadata was not double-escaped",
    }
