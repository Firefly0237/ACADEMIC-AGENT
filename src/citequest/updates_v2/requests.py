"""Server-built requests for the frozen accepted metadata observation profile."""

from urllib.parse import quote, urlencode

from citequest.updates_v2.schemas import WindowInput


def atom_request(config: WindowInput, start: int, *, head: bool = False) -> dict:
    if type(start) is not int or start < 0 or start % (config.page_size - config.overlap):
        raise ValueError("fixed_Atom_offset_required")
    parameters = {"search_query": config.query, "sortBy": "lastUpdatedDate", "sortOrder": "descending",
                  "start": start, "max_results": config.page_size}
    return {"kind": "atom_head_recheck" if head else "atom_page", "start": start, "page_size": config.page_size,
            "url": "https://export.arxiv.org/api/query?" + urlencode(parameters)}


def managed_request(source: str, native_id: str) -> dict:
    if source == "arxiv":
        return {"kind": "oai_managed", "native_id": native_id,
                "url": "https://oaipmh.arxiv.org/oai?" + urlencode({"verb": "GetRecord", "identifier": "oai:arXiv.org:" + native_id, "metadataPrefix": "arXivRaw"})}
    return {"kind": "crossref_known", "native_id": native_id,
            "url": "https://api.crossref.org/works/" + quote(native_id, safe="")}


def validate_request(config: WindowInput, request: dict) -> None:
    kind = request.get("kind")
    if config.source_namespace == "arxiv" and kind in {"atom_page", "atom_head_recheck"}:
        expected = atom_request(config, request.get("start"), head=kind == "atom_head_recheck")
        if kind == "atom_head_recheck" and request.get("start") != 0:
            raise ValueError("head_recheck_must_use_zero_offset")
    elif request.get("native_id") in config.managed_ids:
        expected = managed_request(config.source_namespace, request["native_id"])
    else:
        raise ValueError("request_outside_frozen_profile")
    if request != expected:
        raise ValueError("request_differs_from_server_built_source_request")
