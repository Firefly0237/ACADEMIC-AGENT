"""Official OAI-PMH fixed-window token state, separate from adopted Atom route."""

from dataclasses import dataclass, field
from datetime import date, datetime
from urllib.parse import urlencode

from citequest.updates_v2.protocols import timestamp


@dataclass
class OAIEnumeration:
    lower: date
    upper: date
    maximum_requests: int
    maximum_restarts: int = 1
    requests: int = 0
    restarts: int = 0
    token: str | None = None
    token_expiration: str | None = None
    finished: bool = False
    awaiting_response: bool = False
    records: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)

    def __post_init__(self):
        if type(self.lower) is not date or type(self.upper) is not date or self.lower > self.upper or not 1 <= self.maximum_requests <= 100:
            raise ValueError('bounded_fixed_OAI_day_window_required')
        if type(self.maximum_restarts) is not int or not 0 <= self.maximum_restarts <= 2:
            raise ValueError('explicit_bounded_OAI_restarts_required')

    def _restart(self, reason):
        self.failures.append({'reason': reason, 'requests_retained': self.requests, 'discarded_uncommitted_records': len(self.records)})
        if self.restarts >= self.maximum_restarts:
            raise ValueError('OAI_restart_cap_reached_no_checkpoint')
        self.restarts += 1
        self.token, self.token_expiration, self.records = None, None, {}

    def request(self, now: datetime) -> str:
        if now.tzinfo is None or self.awaiting_response or self.finished:
            raise ValueError('OAI_request_state_invalid')
        if self.token_expiration and timestamp(self.token_expiration) <= now:
            self._restart('resumptionToken_expirationDate_elapsed')
        if self.requests >= self.maximum_requests:
            raise ValueError('OAI_request_cap_no_checkpoint')
        parameters = {'verb': 'ListRecords', 'resumptionToken': self.token} if self.token else {
            'verb': 'ListRecords', 'metadataPrefix': 'arXiv', 'set': 'cs', 'from': self.lower.isoformat(), 'until': self.upper.isoformat()}
        self.requests += 1
        self.awaiting_response = True
        return 'https://oaipmh.arxiv.org/oai?' + urlencode(parameters)

    def accept(self, parsed: dict):
        if not self.awaiting_response:
            raise ValueError('OAI_no_reserved_request')
        self.awaiting_response = False
        if parsed.get('error') == 'badResumptionToken':
            self._restart('badResumptionToken')
            return
        if parsed.get('error'):
            self.failures.append({'reason': parsed['error'], 'requests_retained': self.requests})
            raise ValueError('OAI_error_no_checkpoint')
        for record in parsed['records']:
            previous = self.records.get(record['native_id'])
            if previous and previous['record_sha256'] != record['record_sha256']:
                raise ValueError('OAI_same_ID_content_conflict_no_checkpoint')
            self.records[record['native_id']] = record
        token = parsed.get('token')
        if token is not None and (type(token) is not str or len(token) > 4096):
            raise ValueError('OAI_opaque_token_cap_exceeded')
        self.token, self.token_expiration = token or None, parsed.get('token_expiration')
        self.finished = self.token is None
