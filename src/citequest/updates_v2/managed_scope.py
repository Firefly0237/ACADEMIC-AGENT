"""Pinned candidate membership, rechecked against the existing identity registry.

This is an observation scope, not application-profile adoption or freshness proof.
"""

import hashlib
import json
from pathlib import Path
from uuid import UUID

from sqlalchemy import select, text

from citequest.data_v2.models import SourceSnapshot
from citequest.updates_v2.protocols import digest
from citequest.v2 import models as core
from citequest.v2.domain import DomainError
from citequest.v2.services import normalize_identifier

ROOT = Path(__file__).resolve().parents[3]
SCOPE = ROOT / 'reports/stages/v2/S05/data-execution/managed-candidate-scope/scope.json'
SCOPE_SHA = '455bb70180aad1d97275f2d0bda571c880928f681329898e6b831668a65e8858'


def candidate_managed_scope(db, source: str, origin_id: UUID) -> dict:
    raw = SCOPE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SCOPE_SHA:
        raise DomainError('managed_candidate_manifest_hash_changed', 409)
    manifest = json.loads(raw)
    if str(origin_id) != manifest['source_snapshot_id'] or source not in {'arxiv', 'crossref'}:
        raise DomainError('managed_candidate_origin_not_allowed', 409)
    snapshot = db.get(SourceSnapshot, origin_id)
    if (snapshot is None or snapshot.security_domain != 'public' or snapshot.status == 'disabled'
            or snapshot.source_manifest_sha256 != manifest['source_manifest_sha256']):
        raise DomainError('managed_candidate_source_snapshot_changed', 409)
    sql = SCOPE.with_name('selection.sql').read_bytes()
    if hashlib.sha256(sql).hexdigest() != manifest['query_sha256']:
        raise DomainError('managed_candidate_selector_changed', 409)
    selected = {str(row['source_row_id']): dict(row) for row in db.execute(text(sql.decode('utf-8')), {'snapshot_id': origin_id}).mappings()}
    if len(selected) != manifest['member_count'] or digest(manifest['members']) != manifest['members_sha256']:
        raise DomainError('managed_candidate_membership_changed', 409)
    actual_dois = {}
    for member in manifest['members']:
        actual = selected.get(member['source_row_id'])
        if actual is None or any(str(actual[key]) != str(member[key]) for key in ('native_id', 'work_id', 'work_version_id', 'representation_id', 'source_text_sha256')):
            raise DomainError('managed_candidate_identity_or_source_changed', 409)
        if actual['revision'] != member['resolution_revision'] or actual['proof_sha256'] != member['resolution_proof_sha256'] or digest(actual['proof']) != actual['proof_sha256']:
            raise DomainError('managed_candidate_resolution_proof_changed', 409)
        raw_doi = actual['proof'].get('attribution', {}).get('doi')
        actual_doi = normalize_identifier('doi', raw_doi) if raw_doi else None
        if actual_doi != member['verified_doi']:
            raise DomainError('managed_candidate_attributed_DOI_changed', 409)
        if actual_doi:
            actual_dois[actual_doi] = UUID(member['work_id'])
        if member['verified_arxiv_ids']:
            raise DomainError('managed_candidate_arxiv_policy_requires_new_reviewed_manifest', 409)
    identifiers = {row.normalized: row.work_id for row in db.scalars(select(core.Identifier).where(core.Identifier.security_domain == 'public', core.Identifier.namespace == 'doi', core.Identifier.normalized.in_(actual_dois)))}
    public_work_ids = set(db.scalars(select(core.Work.id).where(core.Work.id.in_(set(actual_dois.values())), core.Work.public.is_(True))))
    if identifiers != actual_dois or set(actual_dois.values()) != public_work_ids:
        raise DomainError('managed_candidate_strong_identifier_or_current_ACL_changed', 409)
    return {'managed_ids': sorted(actual_dois) if source == 'crossref' else [], 'scope_origin_id': manifest['scope_origin_id'],
            'source_snapshot_id': str(origin_id), 'scope_manifest_sha256': SCOPE_SHA, 'members_sha256': manifest['members_sha256'],
            'identity_candidates': 692, 'known_DOIs': len(actual_dois), 'verified_arxiv_ids': 0,
            'application_frozen': False, 'profile_adopted': False, 'scope_role': 'real_verified_identity_candidates_not_whole_future_application',
            'unverified_or_missing_identifiers_not_invented': True}
