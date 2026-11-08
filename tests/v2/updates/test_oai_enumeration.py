"""Synthetic OAI token expiry/restart controls, never source availability evidence."""

from datetime import date
from urllib.parse import parse_qs, urlsplit

import pytest

from citequest.updates_v2.oai_enumeration import OAIEnumeration
from citequest.updates_v2.protocols import parse_oai, timestamp
from fixtures import oai


def test_bad_token_restarts_identical_fixed_window_not_watermark_or_budget():
    pager = OAIEnumeration(date(2026, 11, 1), date(2026, 11, 2), maximum_requests=4)
    now = timestamp('2026-11-01T00:00:00Z')
    original = pager.request(now)
    pager.accept(parse_oai(oai('9911.00001', verb='ListRecords', token='opaque + token'), expected_verb='ListRecords'))
    resumed = parse_qs(urlsplit(pager.request(now)).query)
    assert resumed == {'verb': ['ListRecords'], 'resumptionToken': ['opaque + token']}
    pager.accept(parse_oai(oai('9911.00001', error='badResumptionToken'), expected_verb='ListRecords'))
    assert pager.request(now) == original and pager.requests == 3 and pager.restarts == 1 and not pager.records
    pager.accept(parse_oai(oai('9911.00001', verb='ListRecords'), expected_verb='ListRecords'))
    assert pager.finished and len(pager.records) == 1 and pager.failures[0]['reason'] == 'badResumptionToken'
    with pytest.raises(ValueError):
        pager.request(now)


def test_expiration_fence_and_restart_caps_preserve_history():
    pager = OAIEnumeration(date(2026, 11, 1), date(2026, 11, 2), maximum_requests=4)
    pager.request(timestamp('2026-11-01T00:00:00Z'))
    pager.accept(parse_oai(oai('9911.00001', verb='ListRecords', token='old'), expected_verb='ListRecords'))
    url = pager.request(timestamp('2026-11-03T00:00:00Z'))
    assert 'from=2026-11-01' in url and pager.requests == 2 and pager.restarts == 1
    with pytest.raises(ValueError, match='restart_cap'):
        pager.accept({'error': 'badResumptionToken'})
    assert len(pager.failures) == 2 and pager.requests == 2
