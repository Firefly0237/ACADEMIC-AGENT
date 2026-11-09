"""ADR0035 server-owned Task catalog bindings; no model-supplied generations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb
import psycopg
from psycopg.rows import dict_row
from sqlalchemy import text

from citequest.search_v3.catalog_snapshot import CatalogSnapshot
from citequest.search_v3.schemas import PaperLocator
from citequest.v2.domain import Actor

from .contracts import AgentError, CURRENT_GRAPH_VERSIONS, QA_ANCHORED_GRAPH_VERSION, QA_AUDITED_GRAPH_VERSION, QA_DECOMPOSED_GRAPH_VERSION, QA_GROUNDED_GRAPH_VERSION, QA_JSON_REVIEW_GRAPH_VERSION, QA_ONLINE_GRAPH_VERSION, QA_PATCH_GRAPH_VERSION, QA_PRO_GRAPH_VERSION, QA_REFERENCE_CONTEXT_GRAPH_VERSION, QA_SEPARATED_GRAPH_VERSION, QA_STRUCTURED_GRAPH_VERSION, QA_VERIFIED_GRAPH_VERSION, canonical, digest, graph_version_for_kind
from .product_profiles import FUNCTION_REVIEW_PAIRS, answer_revision_protocol, evidence_navigation_protocol, qa_content_protocol, qa_context_protocol, qa_execution_protocol, qa_retrieval_protocol, require_qa8_protocol, retrieval_rerank_protocol, review_api_protocol, review_transport, structured_output_protocol

POLICY = 'task-catalog-snapshot/1'
SUFFIX = '+task-catalog-snapshot1'
REVIEW_SUFFIX = '+review-function1'
STRICT_REVIEW_SUFFIX = '+review-chatstrict1'
LENGTH_REVIEW_SUFFIX = '+review-chatstrict2'
JSON_REVIEW_SUFFIX = '+review-json1'
CatalogScopePolicy = Literal['task-catalog-snapshot/1']


@dataclass(frozen=True)
class BusinessCatalogDB:
    """MCP receives only a business secret reference, never Saver/Store roles."""
    dsn_file: Path
    catalog_scope_policy: CatalogScopePolicy | None

    def connect(self):
        dsn = self.dsn_file.read_text().strip().replace('postgresql+psycopg://', 'postgresql://', 1)
        return psycopg.connect(dsn, row_factory=dict_row, connect_timeout=5, prepare_threshold=0)


def graph_version(kind: str, profile_id: str | None, policy: str | None, transport: str | None = None,
                  qa_protocol: str | None = None, review_api: str | None = None,
                  structured_output: str | None = None, answer_revision: str | None = None,
                  qa_context: str | None = None, qa_content: str | None = None,
                  evidence_navigation: str | None = None, retrieval_rerank: str | None = None,
                  rerank_sha256: str | None = None, qa_retrieval: str | None = None) -> str:
    if policy not in {None, POLICY}:
        raise AgentError('unsupported_catalog_scope_policy', 503)
    if qa_content is not None and (qa_content not in {'anchored2', 'online1', 'isolated1'} or (kind, profile_id) != ('qa', 'qa-v9')):
        raise AgentError('task_qa_content_protocol_not_supported', 503)
    if evidence_navigation is not None and (evidence_navigation not in {'laya1', 'laya2'}
            or (kind, profile_id, qa_content, policy) != ('qa', 'qa-v9', 'isolated1', POLICY)):
        raise AgentError('task_evidence_navigation_not_supported', 503)
    if qa_retrieval is not None and (qa_retrieval != 'adaptive1'
            or (kind, profile_id, qa_content, policy) != ('qa', 'qa-v9', 'isolated1', POLICY)):
        raise AgentError('task_qa_retrieval_protocol_not_supported', 503)
    if retrieval_rerank is not None or rerank_sha256 is not None:
        retrieval_rerank_protocol({'kind': kind, 'profile': {'profile_id': profile_id,
            'review_transport_protocol': transport, 'qa_execution_protocol': qa_protocol,
            'review_api_protocol': review_api, 'structured_output_protocol': structured_output,
            'answer_revision_protocol': answer_revision, 'qa_context_protocol': qa_context,
            'qa_content_protocol': qa_content, 'retrieval_rerank_protocol': retrieval_rerank,
            'rerank_config_sha256': rerank_sha256}})
        if policy != POLICY:
            raise AgentError('rerank_task_catalog_policy_required', 503)
    if profile_id in {'qa-v8', 'qa-v9'}:
        require_qa8_protocol({'kind': kind, 'profile': {'profile_id': profile_id,
            'review_transport_protocol': transport, 'qa_execution_protocol': qa_protocol,
            'review_api_protocol': review_api, 'structured_output_protocol': structured_output,
            'answer_revision_protocol': answer_revision, 'qa_context_protocol': qa_context}})
        if policy != POLICY:
            raise AgentError('qa8_catalog_policy_required', 503)
        versions = {None: QA_ANCHORED_GRAPH_VERSION, 'anchored2': QA_DECOMPOSED_GRAPH_VERSION,
            'online1': QA_ONLINE_GRAPH_VERSION, 'isolated1': QA_SEPARATED_GRAPH_VERSION}
        version = versions[qa_content]
        navigation_suffix = '+evidence-' + evidence_navigation if evidence_navigation is not None else ''
        rerank_suffix = '+rerank-tei1' if retrieval_rerank is not None else ''
        retrieval_suffix = '+retrieval-adaptive1' if qa_retrieval is not None else ''
        return version + '+review-json1+review-chatjson2' + navigation_suffix + rerank_suffix + retrieval_suffix + SUFFIX
    allowed_transport = (transport == 'function1' and (kind, profile_id) in FUNCTION_REVIEW_PAIRS
        or transport == 'json1' and (kind, profile_id) == ('qa', 'qa-v7'))
    if transport is not None and (not allowed_transport or policy != POLICY):
        raise AgentError('review_transport_and_catalog_policy_not_supported', 503)
    if qa_protocol is not None and (qa_protocol not in {'grounded1', 'grounded2', 'grounded3'} or kind != 'qa' or profile_id not in {'qa-v6', 'qa-v7'}
            or policy != POLICY or transport not in {'function1', 'json1'}):
        raise AgentError('qa_execution_and_catalog_policy_not_supported', 503)
    if profile_id == 'qa-v7' and qa_protocol is not None and (qa_protocol != 'grounded3' or review_api not in {'chat-strict2', 'chat-json1'}):
        raise AgentError('qa_execution_and_catalog_policy_not_supported', 503)
    if review_api is not None and (review_api not in {'chat-strict1', 'chat-strict2', 'chat-json1'} or qa_protocol not in {'grounded1', 'grounded2', 'grounded3'}):
        raise AgentError('review_api_and_qa_execution_not_supported', 503)
    if qa_protocol in {'grounded2', 'grounded3'} and review_api not in {'chat-strict1', 'chat-strict2', 'chat-json1'}:
        raise AgentError('review_api_and_qa_execution_not_supported', 503)
    if review_api == 'chat-strict2' and qa_protocol != 'grounded3':
        raise AgentError('review_api_and_qa_execution_not_supported', 503)
    # The existing database pin guard requires the catalog identity suffix to
    # remain the final bytes. Keep transport explicit before that suffix.
    base = graph_version_for_kind(kind, profile_id)
    if qa_protocol == 'grounded1':
        base = QA_GROUNDED_GRAPH_VERSION
    elif qa_protocol == 'grounded2':
        base = QA_AUDITED_GRAPH_VERSION
    elif qa_protocol == 'grounded3':
        base = QA_PRO_GRAPH_VERSION if profile_id == 'qa-v7' else QA_VERIFIED_GRAPH_VERSION
    if structured_output is not None:
        if (structured_output != 'chat-functions1' or (kind, profile_id) != ('qa', 'qa-v7')
                or qa_protocol != 'grounded3' or review_api not in {'chat-strict2', 'chat-json1'}):
            raise AgentError('structured_output_and_qa_execution_not_supported', 503)
        base = QA_STRUCTURED_GRAPH_VERSION
    if answer_revision is not None:
        if answer_revision != 'jsonpatch1' or structured_output != 'chat-functions1':
            raise AgentError('answer_revision_and_qa_execution_not_supported', 503)
        base = QA_PATCH_GRAPH_VERSION
    if review_api == 'chat-json1' or transport == 'json1':
        if ((kind, profile_id) != ('qa', 'qa-v7') or qa_protocol != 'grounded3'
                or review_api != 'chat-json1' or transport != 'json1'
                or structured_output != 'chat-functions1' or answer_revision != 'jsonpatch1'):
            raise AgentError('json_review_and_qa_execution_not_supported', 503)
        base = QA_JSON_REVIEW_GRAPH_VERSION
    if qa_context is not None:
        if qa_context != 'references1' or base != QA_JSON_REVIEW_GRAPH_VERSION or policy != POLICY:
            raise AgentError('qa_context_and_execution_not_supported', 503)
        base = QA_REFERENCE_CONTEXT_GRAPH_VERSION
    api_suffix = {None: '', 'chat-strict1': STRICT_REVIEW_SUFFIX, 'chat-strict2': LENGTH_REVIEW_SUFFIX, 'chat-json1': '+review-chatjson1'}[review_api]
    transport_suffix = {None: '', 'function1': REVIEW_SUFFIX, 'json1': JSON_REVIEW_SUFFIX}[transport]
    return base + transport_suffix + api_suffix + (SUFFIX if policy is not None else '')


def graph_versions(policy: str | None) -> tuple[str, ...]:
    if policy not in {None, POLICY}:
        raise AgentError('unsupported_catalog_scope_policy', 503)
    old = tuple(version + (SUFFIX if policy is not None else '') for version in CURRENT_GRAPH_VERSIONS)
    reviewed = (tuple(graph_version(kind, profile, policy, 'function1') for kind, profile in sorted(FUNCTION_REVIEW_PAIRS) if profile not in {'qa-v7', 'qa-v8', 'qa-v9'})
        if policy == POLICY else ())
    grounded = (graph_version('qa', 'qa-v6', policy, 'function1', 'grounded1'),
        graph_version('qa', 'qa-v6', policy, 'function1', 'grounded1', 'chat-strict1'),
        graph_version('qa', 'qa-v6', policy, 'function1', 'grounded2', 'chat-strict1'),
        graph_version('qa', 'qa-v6', policy, 'function1', 'grounded3', 'chat-strict1'),
        graph_version('qa', 'qa-v6', policy, 'function1', 'grounded3', 'chat-strict2'),
        graph_version('qa', 'qa-v7', policy, 'function1', 'grounded3', 'chat-strict2'),
        graph_version('qa', 'qa-v7', policy, 'function1', 'grounded3', 'chat-strict2', 'chat-functions1'),
        graph_version('qa', 'qa-v7', policy, 'function1', 'grounded3', 'chat-strict2', 'chat-functions1', 'jsonpatch1'),
        graph_version('qa', 'qa-v7', policy, 'json1', 'grounded3', 'chat-json1', 'chat-functions1', 'jsonpatch1'),
        graph_version('qa', 'qa-v7', policy, 'json1', 'grounded3', 'chat-json1', 'chat-functions1', 'jsonpatch1', 'references1'),
        graph_version('qa', 'qa-v8', policy, 'json1', 'grounded3', 'chat-json2', 'chat-json2', 'jsonpatch1', 'references1'),
        graph_version('qa', 'qa-v9', policy, 'json1', 'grounded3', 'chat-json2', 'chat-json2', 'jsonpatch1', 'references1'),
        graph_version('qa', 'qa-v9', policy, 'json1', 'grounded3', 'chat-json2', 'chat-json2', 'jsonpatch1', 'references1', 'anchored2'),
        graph_version('qa', 'qa-v9', policy, 'json1', 'grounded3', 'chat-json2', 'chat-json2', 'jsonpatch1', 'references1', 'online1'),
        graph_version('qa', 'qa-v9', policy, 'json1', 'grounded3', 'chat-json2', 'chat-json2', 'jsonpatch1', 'references1', 'isolated1'),
        graph_version('qa', 'qa-v9', policy, 'json1', 'grounded3', 'chat-json2', 'chat-json2', 'jsonpatch1', 'references1', 'isolated1', 'laya1')) if policy == POLICY else ()
    reranked = tuple(graph_version('qa', 'qa-v9', policy, 'json1', 'grounded3', 'chat-json2',
        'chat-json2', 'jsonpatch1', 'references1', 'isolated1', navigation, 'tei-cross-encoder1', '0' * 64)
        for navigation in (None, 'laya1')) if policy == POLICY else ()
    adaptive = tuple(graph_version('qa', 'qa-v9', policy, 'json1', 'grounded3', 'chat-json2',
        'chat-json2', 'jsonpatch1', 'references1', 'isolated1', navigation, rerank,
        '0' * 64 if rerank else None, 'adaptive1')
        for navigation in (None, 'laya1') for rerank in (None, 'tei-cross-encoder1')) if policy == POLICY else ()
    batched = tuple(graph_version('qa', 'qa-v9', policy, 'json1', 'grounded3', 'chat-json2',
        'chat-json2', 'jsonpatch1', 'references1', 'isolated1', 'laya2', rerank,
        '0' * 64 if rerank else None, retrieval)
        for rerank in (None, 'tei-cross-encoder1') for retrieval in (None, 'adaptive1')) if policy == POLICY else ()
    return old + reviewed + grounded + reranked + adaptive + batched


def require_schema(db, policy: str | None) -> None:
    if policy is not None:
        if policy != POLICY or db.execute("SELECT to_regclass('cq_app.agent_catalog_pins') AS table_name").fetchone()['table_name'] is None:
            raise AgentError('task_catalog_snapshot_schema_or_policy_unavailable', 503)


def task_policy(task: dict) -> str | None:
    value = task['profile'].get('catalog_scope_policy')
    if value is not None and (not isinstance(value, str) or value != POLICY):
        raise AgentError('unsupported_task_catalog_scope_policy', 503)
    return value


def require_runtime(task: dict, configured: str | None) -> None:
    if task_policy(task) != configured:
        raise AgentError('task_catalog_runtime_policy_mismatch', 503)
    review_transport(task)
    qa_execution_protocol(task)
    review_api_protocol(task)
    structured_output_protocol(task)
    answer_revision_protocol(task)
    qa_context_protocol(task)
    qa_content_protocol(task)
    qa_retrieval_protocol(task)
    evidence_navigation_protocol(task)
    retrieval_rerank_protocol(task)


def current_runtime(task: dict, configured: str | None) -> bool:
    return (task_policy(task) == configured and task['graph_version'] ==
        graph_version(task['kind'], task['profile'].get('profile_id'), configured, review_transport(task), qa_execution_protocol(task), review_api_protocol(task), structured_output_protocol(task), answer_revision_protocol(task), qa_context_protocol(task), qa_content_protocol(task), evidence_navigation_protocol(task), retrieval_rerank_protocol(task), task['profile'].get('rerank_config_sha256'), qa_retrieval_protocol(task)))


def readable_runtime(task: dict) -> bool:
    # A reader can validate either known immutable policy. Dispatch lane/
    # creation settings do not decide whether a result may be projected/exported.
    return task['graph_version'] == graph_version(task['kind'], task['profile'].get('profile_id'), task_policy(task), review_transport(task), qa_execution_protocol(task), review_api_protocol(task), structured_output_protocol(task), answer_revision_protocol(task), qa_context_protocol(task), qa_content_protocol(task), evidence_navigation_protocol(task), retrieval_rerank_protocol(task), task['profile'].get('rerank_config_sha256'), qa_retrieval_protocol(task))


def required_scopes(brief: dict) -> dict[str, list[PaperLocator]]:
    if brief['kind'] in {'summary', 'compare'} or (brief['kind'] == 'research' and brief.get('research_action') in {'suggest', 'extract'}):
        return {}
    from citequest.research_v3.evidence import reference_selections
    selected = reference_selections(brief)
    papers = [PaperLocator(kind=item['kind'], id=item['id']) for item in selected if item['kind'] != 'material_file']
    if not papers:
        return {} if selected else {brief['catalog']: []}
    result = {}
    for paper in papers:
        catalog = 'application_papers' if paper.kind == 'native_representation' else 'standard_metadata'
        result.setdefault(catalog, []).append(paper)
    return result


class CatalogPins:
    def __init__(self, database, search) -> None:
        self.database, self.search = database, search

    def _same_database(self, db) -> None:
        if self.search is None:
            raise AgentError('task_catalog_search_bridge_unavailable', 503)
        name = db.execute('SELECT current_database() AS name').fetchone()['name']
        with self.search.factory.begin() as source:
            if source.execute(text('SELECT current_database()')).scalar_one() != name:
                raise AgentError('task_catalog_database_binding_required', 503)
            key = int.from_bytes(uuid4().bytes[:8], 'big', signed=True)
            source.execute(text('SELECT pg_advisory_xact_lock(:key)'), {'key': key})
            if db.execute('SELECT pg_try_advisory_xact_lock(%s) AS acquired', (key,)).fetchone()['acquired']:
                raise AgentError('task_catalog_database_binding_required', 503)

    def _task(self, db, actor: Actor, task_id: UUID, *, lock: bool = False, dispatch: bool = False) -> dict:
        from .database import owned_task
        task = owned_task(db, actor.owner_id, task_id, lock=lock)
        # Creation/dispatch configuration is not a read capability. GET/export
        # authorize an owned persisted Task under its known immutable policy,
        # even when their composition root does not create snapshot Tasks.
        if dispatch:
            require_runtime(task, self.database.catalog_scope_policy)
            if not current_runtime(task, self.database.catalog_scope_policy):
                raise AgentError('task_catalog_pinned_runtime_required', 503)
        if digest(task['profile']) != task['profile_hash']:
            raise AgentError('task_profile_integrity_failure', 503)
        require_schema(db, task_policy(task))
        return task

    @staticmethod
    def _rows(db, task: dict) -> dict[str, CatalogSnapshot]:
        result = {}
        for row in db.execute('SELECT * FROM cq_app.agent_catalog_pins WHERE task_id=%s', (task['id'],)).fetchall():
            if row['owner_id'] != task['owner_id'] or row['binding_sha256'] != digest(row['binding_canonical'].encode()):
                raise AgentError('task_catalog_pin_integrity_failure', 503)
            snapshot = CatalogSnapshot.model_validate_json(row['binding_canonical'], strict=True)
            if (canonical(json.loads(row['binding_canonical'])).decode() != row['binding_canonical']
                    or json.loads(row['binding_canonical']) != row['binding'] or snapshot.catalog != row['catalog']
                    or snapshot.generation_id != row['generation_id'] or snapshot.epoch != row['epoch']):
                raise AgentError('task_catalog_pin_integrity_failure', 503)
            result[snapshot.catalog] = snapshot
        return result

    def ensure(self, actor: Actor, task_id: UUID) -> dict[str, CatalogSnapshot]:
        # The Task row serializes first use across workers. No network/model
        # RPC is permitted inside this short SQL admission transaction.
        with self.database.connect() as db:
            task = self._task(db, actor, task_id, lock=True, dispatch=True)
            if task_policy(task) is None:
                return {}
            scopes = required_scopes(task['brief'])
            if not scopes:
                return {}
            if task['state'] != 'running' or task['cancellation_requested']:
                raise AgentError('task_not_dispatchable')
            self._same_database(db)
            pins = self._rows(db, task)
            for catalog, selected in scopes.items():
                if catalog in pins:
                    self.search.recheck_catalog_snapshot(actor, pins[catalog], selected)
            missing = {catalog: selected for catalog, selected in scopes.items() if catalog not in pins}
            if missing:
                # One statement captures all initially missing catalogs.
                # A later newly selected catalog has its own documented first
                # admission; it does not rewrite any already bound catalog.
                captured = self.search.capture_catalog_snapshots(actor, missing)
                if set(captured) != set(missing):
                    raise AgentError('task_catalog_capture_incomplete', 503)
                for catalog, snapshot in captured.items():
                    if not isinstance(snapshot, CatalogSnapshot) or snapshot.catalog != catalog:
                        raise AgentError('server_owned_catalog_snapshot_required', 503)
                    encoded = canonical(snapshot.model_dump(mode='json')).decode()
                    db.execute('''INSERT INTO cq_app.agent_catalog_pins
                        (task_id,catalog,owner_id,generation_id,epoch,binding,binding_canonical,binding_sha256)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s)''',
                        (task_id, catalog, actor.owner_id, snapshot.generation_id, snapshot.epoch,
                         Jsonb(snapshot.model_dump(mode='json')), encoded, digest(encoded.encode())))
                    pins[catalog] = snapshot
            return {catalog: pins[catalog] for catalog in scopes}

    def authorize_evidence(self, actor: Actor, task_id: UUID, evidence: list[dict]) -> None:
        with self.database.connect() as db:
            task = self._task(db, actor, task_id)
            if task_policy(task) is None:
                return
            papers = [item for item in evidence if item.get('evidence_type') != 'source_window'
                and item.get('source_ref', {}).get('kind') in {'metadata_projection', 'native_representation'}]
            if not papers:
                return  # Source-window/material permission remains its own bridge.
            self._same_database(db)
            pins = self._rows(db, task)
            grouped = {}
            for item in papers:
                source = item['source_ref']
                catalog = 'application_papers' if source['kind'] == 'native_representation' else 'standard_metadata'
                pin = pins.get(catalog)
                if pin is None or str(pin.generation_id) != str(item.get('generation_id')) or pin.epoch != item.get('epoch'):
                    raise AgentError('evidence_outside_task_catalog_snapshot', 409)
                locator = PaperLocator(kind=source['kind'], id=source['id'])
                grouped.setdefault(catalog, {})[(locator.kind, locator.id)] = locator
            for catalog, values in grouped.items():
                selected = list(values.values())
                for start in range(0, len(selected), 20):
                    self.search.recheck_catalog_snapshot(actor, pins[catalog], selected[start:start + 20])
