"""Pure synthetic XML protocol controls; no real provider success claim."""

import copy
from uuid import uuid4

import pytest
from defusedxml.common import DefusedXmlException

from citequest.updates_v2.protocols import parse_atom, parse_oai, reconciliation_state, timestamp, validate_discovery
from citequest.updates_v2.requests import atom_request, managed_request, validate_request
from citequest.updates_v2.schemas import WindowInput
from citequest.updates_v2.transport import CanaryGate, SourceFetcher, update_url
from citequest.materials_v2.reference_fetch import FetchError
from fixtures import atom, oai

ROWS = [("9911.00001", "2026-11-03T00:00:00Z", 2), ("9911.00002", "2026-11-02T00:00:00Z", 1), ("9911.00003", "2026-10-31T00:00:00Z", 1)]


def pages():
    return [parse_atom(atom(ROWS[:2], total=3), expected_start=0, maximum_records=2),
            parse_atom(atom(ROWS[1:], start=1, total=3), expected_start=1, maximum_records=2)]


def validate(items, head=None):
    return validate_discovery(items, head or items[0], lower=timestamp("2026-11-01T00:00:00Z"), upper=timestamp("2026-11-04T00:00:00Z"), page_size=2, overlap=1)


def test_overlap_fixed_window_excludes_fence():
    assert [r['native_id'] for r in validate(pages())] == ["9911.00001", "9911.00002"]


@pytest.mark.parametrize('mutation', ['head', 'overlap', 'order', 'total', 'offset'])
def test_drift_never_advances(mutation):
    items, head = pages(), copy.deepcopy(pages()[0])
    if mutation == 'head':
        head['records'][0]['record_sha256'] = '0' * 64
    if mutation == 'overlap':
        items[1]['records'][0]['record_sha256'] = '0' * 64
    if mutation == 'order':
        items[1]['records'].reverse()
    if mutation == 'total':
        items[1]['total_results'] += 1
    if mutation == 'offset':
        items[1]['start'] += 1
    with pytest.raises(ValueError):
        validate(items, head)


def test_deleted_missing_and_unknown_are_different():
    assert parse_oai(oai('9911.00001', deleted=True))['records'][0]['state'] == 'quarantine_deleted_header'
    missing = parse_oai(oai('9911.00001', error='idDoesNotExist'))
    assert missing['deleted'] is False and missing['state'] == 'quarantine_missing'
    assert reconciliation_state(pages()[0]['records'][0], parse_oai(oai('9911.00001', version=2))) == 'unknown'


def test_expired_token_is_not_empty_success_or_freshness():
    first = parse_oai(oai('9911.00001', verb='ListRecords', token='opaque + token'), expected_verb='ListRecords')
    assert first['token'] == 'opaque + token' and first['status_asof'] is None
    expired = parse_oai(oai('9911.00001', error='badResumptionToken'), expected_verb='ListRecords')
    assert expired['error'] == 'badResumptionToken' and not expired['records']


def test_withdrawal_conflict_and_xml_entities():
    assert reconciliation_state(None, parse_oai(oai('9911.00001', comment='This paper has been withdrawn'))) == 'quarantine_withdrawal_signal'
    assert reconciliation_state(pages()[0]['records'][0], parse_oai(oai('9911.00001', version=1))) == 'quarantine_version_conflict'
    with pytest.raises(DefusedXmlException):
        parse_atom(b'<!DOCTYPE feed [<!ENTITY x "boom">]><feed>&x;</feed>', expected_start=0, maximum_records=2)


def test_server_request_scope_and_production_gate():
    config = WindowInput(source_namespace='arxiv', mode='synthetic_fixture', lower_bound=timestamp('2026-11-01T00:00:00Z'), upper_bound=timestamp('2026-11-02T00:00:00Z'), managed_ids=['9911.00001'], page_size=2, overlap=1, maximum_network_calls=8, maximum_active_seconds=120, client_operation_id=uuid4())
    validate_request(config, atom_request(config, 0))
    validate_request(config, managed_request('arxiv', '9911.00001'))
    with pytest.raises(ValueError):
        validate_request(config, managed_request('arxiv', '9911.00002'))
    with pytest.raises(ValueError, match='shared_production'):
        SourceFetcher(CanaryGate())
    for invalid in ['https://oaipmh.arxiv.org:bad/oai', 'https://oaipmh.arxiv.org/oai#x', 'https://127.0.0.1/oai']:
        with pytest.raises(FetchError):
            update_url(invalid)
