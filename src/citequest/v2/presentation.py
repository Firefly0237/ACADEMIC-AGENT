"""Explicit public DTO mapping; database columns are never blindly exposed."""

from __future__ import annotations

from sqlalchemy import select

from . import models as m
from .services import Catalog


def work(catalog: Catalog, obj: m.Work, *, public_only: bool = False) -> dict:
    catalog.work(obj.id)
    domains = ["public"] if public_only else [catalog.actor.security_domain, "public"]
    identifiers = catalog.db.scalars(
        select(m.Identifier)
        .where(m.Identifier.work_id == obj.id, m.Identifier.security_domain.in_(domains))
        .order_by(m.Identifier.namespace, m.Identifier.normalized)
    )
    return {
        "work_id": str(obj.id),
        "title": obj.title,
        "authors": obj.authors,
        "year": obj.year,
        "metadata_revision": obj.metadata_revision,
        "field_provenance": obj.field_provenance,
        "canonical_redirect_work_id": str(obj.canonical_redirect_work_id) if obj.canonical_redirect_work_id else None,
        "identifiers": [
            {
                "namespace": i.namespace,
                "raw": i.raw,
                "normalized": i.normalized,
                "normalizer_version": i.normalizer_version,
            }
            for i in identifiers
        ],
    }


def item(catalog: Catalog, obj: m.LibraryItem) -> dict:
    catalog.item(obj.id, deleted=True)
    files = catalog.db.scalars(
        select(m.File.id)
        .join(m.ItemFile, m.ItemFile.file_id == m.File.id)
        .where(m.ItemFile.item_id == obj.id, m.File.owner_id == catalog.actor.owner_id, m.File.deleted_at.is_(None))
    )
    collections = catalog.db.scalars(
        select(m.Collection.id)
        .join(m.ItemCollection, m.ItemCollection.collection_id == m.Collection.id)
        .where(
            m.ItemCollection.item_id == obj.id,
            m.Collection.owner_id == catalog.actor.owner_id,
            m.Collection.deleted_at.is_(None),
        )
    )
    tags = catalog.db.scalars(select(m.ItemTag.tag).where(m.ItemTag.item_id == obj.id).order_by(m.ItemTag.tag))
    return {
        "library_item_id": str(obj.id),
        "owner_id": str(obj.owner_id),
        "work_id": str(obj.work_id),
        "file_ids": sorted(map(str, files)),
        "collection_ids": sorted(map(str, collections)),
        "tags": list(tags),
        "deleted_at": obj.deleted_at.isoformat() if obj.deleted_at else None,
        "revision": obj.revision,
    }


def collection(obj: m.Collection) -> dict:
    return {
        "collection_id": str(obj.id),
        "owner_id": str(obj.owner_id),
        "name": obj.name,
        "parent_id": str(obj.parent_id) if obj.parent_id else None,
        "revision": obj.revision,
    }


def note(obj: m.Note) -> dict:
    return {
        "note_id": str(obj.id),
        "owner_id": str(obj.owner_id),
        "library_item_id": str(obj.library_item_id),
        "body": obj.body,
        "evidence_ids": [],
        "revision": obj.revision,
    }


def history(obj: m.History) -> dict:
    return {
        "history_id": str(obj.id),
        "owner_id": str(obj.owner_id),
        "kind": obj.kind,
        "target_id": obj.target_id,
        "occurred_at": obj.created_at.isoformat(),
        "snapshot_id": str(obj.snapshot_id) if obj.snapshot_id else None,
    }


def file(catalog: Catalog, obj: m.File) -> dict:
    blob = catalog.db.get(m.Blob, (obj.security_domain, obj.sha256))
    if blob is None:
        raise RuntimeError("File has no referenced blob")
    return {
        "file_id": str(obj.id),
        "version_id": str(obj.version_id) if obj.version_id else None,
        "owner_id": str(obj.owner_id),
        "security_domain": obj.security_domain,
        "sha256": obj.sha256,
        "media_type": obj.media_type,
        "size_bytes": blob.size_bytes,
        "page_count": None,
        "rights_status": obj.rights_status,
        "outbound_policy": obj.outbound_policy,
        "status": obj.status,
        "can_preview": False,
        "can_qa": False,
        "can_download": False,
        "coverage": "metadata_only",
    }


def source_record(obj: m.SourceRecord) -> dict:
    return {
        "source": obj.source,
        "native_id": obj.native_id,
        "source_revision": obj.source_revision,
        "retrieved_at": obj.retrieved_at.isoformat(),
        "raw_sha256": obj.raw_sha256,
        "source_url": obj.source_url,
        "rights_status": obj.rights_status,
        "license": obj.license,
    }


def version(obj: m.WorkVersion) -> dict:
    return {
        "version_id": str(obj.id),
        "work_id": str(obj.work_id),
        "source_sha256": obj.source_sha256,
        "kind": obj.kind,
        "version_label": obj.version_label,
        "created_at": obj.created_at.isoformat(),
    }
