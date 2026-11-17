"""Current-authority history headings; never reuse a previously visible title."""
from __future__ import annotations

from collections.abc import Callable

from pydantic import ValidationError

from citequest.search_v3.batch_headers import CurrentPaperHeader, locator_key
from citequest.search_v3.schemas import PaperLocator
from citequest.v2.domain import Actor


def project_history(rows: list[dict], actor: Actor, read_many: Callable | None) -> list[dict]:
    papers = {}
    targets = {}
    for item in rows:
        parts = item["target_id"].split(":")
        if len(parts) != 3 or parts[0] != "paper":
            continue
        try:
            paper = PaperLocator(kind=parts[1], id=parts[2])
        except (ValidationError, ValueError):
            continue
        key = locator_key(paper)
        papers[key] = paper
        targets[item["history_id"]] = key
    headers = {}
    if papers and read_many is not None:
        try:
            headers = read_many(actor, list(papers.values()))
            if not isinstance(headers, dict) or set(headers) != set(papers) or any(
                not isinstance(value, CurrentPaperHeader) or value.paper != papers[key]
                for key, value in headers.items()
            ):
                raise ValueError("history_current_header_contract_invalid")
        except Exception:
            # A failing batch is unavailable, never an excuse for a captured
            # title or scalar fallback with a different authorization boundary.
            headers = {}
    result = []
    for item in rows:
        key = targets.get(item["history_id"])
        if key is None:
            result.append(item)
            continue
        header = headers.get(key)
        available = header is not None and header.source_availability == "available"
        result.append({**item, "display_title": header.detail.title if available else None,
                       "source_availability": header.source_availability if header else "temporarily_unavailable"})
    return result
