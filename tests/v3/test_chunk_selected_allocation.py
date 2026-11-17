"""Per-document query allocation only; every source/ACL/native result is a double."""
from uuid import uuid4

import pytest

from citequest.search_v3.catalog_snapshot import CatalogSnapshot
from citequest.search_v3.chunk_evidence import ChunkEvidencePage, ChunkEvidenceSearch
from citequest.search_v3.schemas import PaperLocator
from citequest.v2.domain import Actor, DomainError


def snapshot(catalog):
    hashes = dict.fromkeys(("manifest_sha256", "source_seal_sha256", "server_proof_sha256", "index_profile_sha256", "tokenizer_profile_sha256", "policy_sha256", "evaluation_sha256", "native_runtime_sha256"), "f" * 64)
    return CatalogSnapshot(catalog=catalog, generation_id=uuid4(), epoch=1, collection_name="explicit_fixture",
                           source_snapshot_id=uuid4(), **hashes)


def selected():
    return [PaperLocator(kind="native_representation", id=char * 64) for char in ("a", "b", "c")]


class FakeScope:
    def __init__(self):
        self.checks = []
        self.revoke_on_final = False

    def recheck_catalog_snapshot(self, actor, pin, papers):
        self.checks.append((pin.catalog, tuple(paper.id for paper in papers)))
        if self.revoke_on_final and len(self.checks) == 2:
            raise DomainError("selected_paper_outside_pinned_catalog_or_access", 404)


def setup():
    scope = FakeScope()
    service = ChunkEvidenceSearch(scope)
    calls = []

    def retrieve(actor, request, pin, *, limit):
        calls.append((request.selected_papers, request.catalog, request.candidate_limit, limit))
        return ChunkEvidencePage(retrieval_id=uuid4(), generation_id=pin.generation_id, epoch=pin.epoch,
                                 evidence=[], candidates=0, masked_before_delivery=0)

    service.retrieve = retrieve
    return scope, service, calls


def test_selected_documents_are_separately_scoped_before_native_queries():
    scope, service, calls = setup()
    papers = selected()
    pages = service.retrieve_selected(Actor(uuid4()), papers, "explicit query", {"application_papers": snapshot("application_papers")}, limit=8)
    assert [call[0] for call in calls] == [[paper] for paper in papers]
    assert [call[2:] for call in calls] == [(3, 3), (3, 3), (2, 2)]
    assert len(pages) == 3 and len(scope.checks) == 2 and scope.checks[0] == scope.checks[1]


def test_insufficient_capacity_is_rejected_before_any_source_or_query():
    scope, service, calls = setup()
    with pytest.raises(DomainError) as error:
        service.retrieve_selected(Actor(uuid4()), selected(), "query", {"application_papers": snapshot("application_papers")}, limit=2)
    assert error.value.code == "selected_chunk_retrieval_requires_capacity_for_every_document"
    assert not calls and not scope.checks


def test_final_scope_revocation_prevents_returning_earlier_receipts():
    scope, service, calls = setup()
    scope.revoke_on_final = True
    with pytest.raises(DomainError) as error:
        service.retrieve_selected(Actor(uuid4()), selected(), "query", {"application_papers": snapshot("application_papers")}, limit=8)
    assert error.value.code == "selected_paper_outside_pinned_catalog_or_access"
    assert len(calls) == 3 and len(scope.checks) == 2
