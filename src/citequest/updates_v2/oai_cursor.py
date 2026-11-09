"""Pure fixed-day OAI request/cursor contracts shared by reservation and commit."""
from datetime import timedelta
from urllib.parse import urlencode

from .oai_enumeration import OAIEnumeration


def state_for(config, cursor: dict) -> OAIEnumeration:
    state = (OAIEnumeration.restore(cursor["oai"]) if cursor.get("oai") else OAIEnumeration(
        config.lower_bound.date(), (config.upper_bound - timedelta(microseconds=1)).date(), config.maximum_pages))
    if (state.lower != config.lower_bound.date() or state.upper != (config.upper_bound - timedelta(microseconds=1)).date()
        or state.maximum_requests != config.maximum_pages):
        raise ValueError("OAI_cursor_changed_frozen_window")
    return state


def reserved_request(state: OAIEnumeration) -> dict:
    parameters = {"verb": "ListRecords", "resumptionToken": state.token} if state.token else {
        "verb": "ListRecords", "metadataPrefix": "arXiv", "set": "cs", "from": state.lower.isoformat(), "until": state.upper.isoformat()}
    return {"kind": "oai_list", "url": "https://oaipmh.arxiv.org/oai?" + urlencode(parameters),
        "oai_epoch": state.restarts, "oai_request": state.requests}


def next_request(config, cursor: dict, now):
    state = state_for(config, cursor)
    if state.finished:
        return None, state.cursor()
    url = state.request(now)
    request = reserved_request(state)
    if request["url"] != url:
        raise ValueError("OAI_reserved_URL_mismatch")
    return request, state.cursor()
