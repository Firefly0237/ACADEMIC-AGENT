"""One source/strong-identifier identity service for corpus and update adapters."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.dialects.postgresql import insert

from citequest.v2 import models as core
from citequest.v2.domain import DomainError, WorkInput
from citequest.v2.services import Catalog, normalize_identifier


class CanonicalResolver:
    """The caller owns Session.begin and a versioned source-fact operation.

    Work existence is an identity fact, never a rights/fulltext/QA grant. Source
    adapters supply actual protocol IDs, not title similarity or LLM proposals.
    This class preserves existing metadata; source-backed revisions are separate.
    """

    def __init__(self, catalog: Catalog) -> None:
        self.catalog, self.db = catalog, catalog.db

    def resolve(self, source: str, native_id: str, request: WorkInput, *, public: bool) -> core.Work:
        if not source.strip() or len(source) > 100 or not native_id.strip() or len(native_id) > 512:
            raise DomainError("invalid_canonical_source_identity", 422)
        self.catalog.lock_owner()
        if public:
            self.catalog.require_publisher()
        domain = "public" if public else self.catalog.actor.security_domain
        common = self.db.get(
            core.SourceIdentity, (domain, source, native_id), with_for_update=True, populate_existing=True
        )
        identifiers = [
            (item.namespace, item.raw, normalize_identifier(item.namespace, item.raw)) for item in request.identifiers
        ]
        if len({(namespace, normalized) for namespace, _raw, normalized in identifiers}) != len(identifiers):
            raise DomainError("duplicate_identifier", 400)
        if common is not None:
            work = self.catalog.work(common.work_id)
        else:
            result = self.catalog.create_work(request, public=public)
            work = self.catalog.work(UUID(result["work_id"]))
        if public and not work.public:
            raise DomainError("public_source_cannot_resolve_private_work", 409)
        # Existing common identity does not excuse a newly supplied DOI/arXiv
        # pointing to another Work. Private copies of the exact same public ID
        # retain the existing S03 isolation rule and are not published.
        for namespace, raw, normalized in identifiers:
            public_identifier = self.db.get(core.Identifier, ("public", namespace, normalized))
            local_identifier = self.db.get(core.Identifier, (domain, namespace, normalized))
            for identifier in (public_identifier, local_identifier):
                if identifier is not None and identifier.work_id != work.id:
                    is_private_copy = (
                        domain != "public"
                        and identifier.security_domain == domain
                        and public_identifier is not None
                        and public_identifier.work_id == work.id
                    )
                    if not is_private_copy:
                        raise DomainError("identifier_conflict", 409)
            if local_identifier is None:
                self.db.add(
                    core.Identifier(
                        security_domain=domain,
                        namespace=namespace,
                        normalized=normalized,
                        raw=raw,
                        work_id=work.id,
                        normalizer_version="identifier-v1",
                    )
                )
        self.db.flush()
        self.db.execute(
            insert(core.SourceIdentity)
            .values(
                security_domain=domain,
                source=source,
                native_id=native_id,
                work_id=work.id,
            )
            .on_conflict_do_nothing(index_elements=["security_domain", "source", "native_id"])
        )
        actual = self.db.get(core.SourceIdentity, (domain, source, native_id), populate_existing=True)
        if actual is None or actual.work_id != work.id:
            raise DomainError("source_identity_conflict", 409)
        return work
