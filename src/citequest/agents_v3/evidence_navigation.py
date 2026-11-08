"""Server-selected, bounded frontier for the optional real probability navigator."""
from collections import OrderedDict

from .contracts import AgentError, digest


def navigation_frontier(evidence: list[dict], maximum: int) -> list[str]:
    """Round-robin whole issued passages by document; never shorten source text."""
    if type(maximum) is not int or not 1 <= maximum <= 8:
        raise AgentError("decisionrag_frontier_limit_invalid")
    groups: dict[str, list[str]] = OrderedDict()
    seen = set()
    for item in evidence:
        ref = item.get("ref_id")
        if not isinstance(ref, str) or not ref or ref in seen:
            raise AgentError("decisionrag_frontier_reference_invalid")
        seen.add(ref)
        source = item.get("source_ref", {})
        document = ("material_file", item["file_id"]) if item.get("file_id") else (
            source.get("kind"), source.get("id"))
        groups.setdefault(digest(document), []).append(ref)
    selected = []
    for position in range(max((len(rows) for rows in groups.values()), default=0)):
        for rows in groups.values():
            if position < len(rows):
                selected.append(rows[position])
                if len(selected) == maximum:
                    return selected
    return selected
