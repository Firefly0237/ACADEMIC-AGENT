"""Typed reference acquisition: metadata and queued bytes have distinct states.

Crossref metadata does not grant publisher download rights. An arXiv download is
an unverified user material, never an automatically trusted paper/source binding.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit
from uuid import NAMESPACE_URL, UUID, uuid5

import anyio
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
from jsonschema import Draft202012Validator, FormatChecker
from sqlalchemy import select
from sqlalchemy.orm import Session

from citequest.v2 import models as core
from citequest.v2.domain import Actor, DomainError
from citequest.v2.services import Catalog, digest
from citequest.v2.storage import CAS, StoredBlob

from .reference_fetch import FetchContext, OfficialFetcher
from .references import ReferencePlan
from .services import MaterialService
from .text import MaterialError

ROOT = Path(__file__).resolve().parents[3]
METADATA_LIMIT = 2 * 1024 * 1024
PDF_LIMIT = 50 * 1024 * 1024


@lru_cache(maxsize=1)
def reference_contract() -> dict:
    value = json.loads((ROOT / "contracts/v2-materials/reference.schema.json").read_text(encoding="utf-8"))
    base = json.loads((ROOT / "contracts/v2/domain.schema.json").read_text(encoding="utf-8"))
    value["$defs"] = {**base["$defs"], **value["$defs"]}
    return value


def checked(value: dict) -> dict:
    if not Draft202012Validator(reference_contract(), format_checker=FormatChecker()).is_valid(value):
        raise DomainError("reference_response_contract_failed", 500)
    file = value["file"]
    if (value["state"] == "queued") != (file is not None):
        raise DomainError("reference_state_contract_failed", 500)
    if file is not None and (
        file["rights_status"] != "unknown"
        or file["version_id"] is not None
        or file["can_qa"]
        or file["status"] != "queued"
    ):
        raise DomainError("reference_identity_contract_failed", 500)
    for match in value["matches"]:
        bindings = match.get("file_bindings", [])
        ids = [binding["file_id"] for binding in bindings]
        if len(ids) != len(set(ids)) or any(
            binding["reuse_eligible"]
            and (binding["binding_source"] != "trusted_source" or binding["identity_status"] != "hash_verified")
            for binding in bindings
        ):
            raise DomainError("reference_identity_contract_failed", 500)
        if {item["file_id"] for item in match["files"]} != {
            binding["file_id"] for binding in bindings if binding["reuse_eligible"]
        }:
            raise DomainError("reference_identity_contract_failed", 500)
    return value


def strict_pairs(items: list) -> dict:
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("duplicate_metadata_key")
        value[key] = item
    return value


def _string(value: Any, maximum: int) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError("invalid_metadata_string")
    value.encode("utf-8")
    return value


def _base(provider: str, identifier: str, url: str) -> dict:
    return {
        "provider": provider,
        "identifier": identifier,
        "title": None,
        "alternate_titles": [],
        "authors": None,
        "year": None,
        "year_source": None,
        "doi": None,
        "arxiv_id": None,
        "abstract": None,
        "abstract_format": "none",
        "license_urls": None,
        "source_url": url,
        "normalization_profile": "cq.reference-metadata/1",
    }


def parse_crossref(data: bytes, plan: ReferencePlan, url: str) -> dict:
    value = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=strict_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite_metadata")),
    )
    if value.get("status") != "ok" or value.get("message-type") != "work" or type(value.get("message")) is not dict:
        raise ValueError("invalid_crossref_envelope")
    work = value["message"]
    if type(work.get("DOI")) is not str or work["DOI"].lower() != plan.identifier:
        raise ValueError("crossref_identifier_mismatch")
    result = _base("crossref", plan.identifier, url)
    titles = work.get("title", [])
    if type(titles) is not list:
        raise ValueError("invalid_crossref_titles")
    titles = [_string(item, 8192) for item in titles]
    result.update(title=titles[0] if titles else None, alternate_titles=titles[1:], doi=plan.identifier)
    if "author" in work:
        if type(work["author"]) is not list or len(work["author"]) > 10000:
            raise ValueError("invalid_crossref_authors")
        result["authors"] = []
        for author in work["author"]:
            if type(author) is not dict:
                raise ValueError("invalid_crossref_author")
            names = [_string(author.get(key), 2048) for key in ("given", "family")]
            name = " ".join(part for part in names if part) or _string(author.get("name"), 2048)
            result["authors"].append(_string(name, 2048))
    for key in ("published", "published-print", "published-online", "issued"):
        if key in work:
            dates = work[key].get("date-parts") if type(work[key]) is dict else None
            if type(dates) is not list or not dates or type(dates[0]) is not list or not dates[0]:
                raise ValueError("invalid_crossref_date")
            year = dates[0][0]
            if type(year) is not int or not 1000 <= year <= 2200:
                raise ValueError("invalid_crossref_year")
            result.update(year=year, year_source=key)
            break
    if "abstract" in work:
        result.update(abstract=_string(work["abstract"], 1048576), abstract_format="jats_xml")
    if "license" in work:
        if type(work["license"]) is not list:
            raise ValueError("invalid_crossref_license")
        result["license_urls"] = []
        for license_item in work["license"]:
            if type(license_item) is not dict:
                raise ValueError("invalid_crossref_license")
            license_url = _string(license_item.get("URL"), 4096)
            if license_url is not None:
                if urlsplit(license_url).scheme not in {"https", "http"} or not urlsplit(license_url).hostname:
                    raise ValueError("invalid_license_url")
                result["license_urls"].append(license_url)
    return result


def parse_arxiv(data: bytes, plan: ReferencePlan, url: str) -> dict:
    root = ElementTree.fromstring(data, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        raise ValueError("invalid_arxiv_feed")
    entries = root.findall("a:entry", ns)
    if len(entries) != 1:
        raise ValueError("arxiv_exact_one_entry_required")
    entry = entries[0]
    identifier_url = entry.findtext("a:id", namespaces=ns)
    parts = urlsplit(identifier_url or "")
    if (
        parts.scheme not in {"http", "https"}
        or parts.hostname not in {"arxiv.org", "www.arxiv.org"}
        or parts.username is not None
        or parts.password is not None
        or parts.port not in {None, 80, 443}
        or parts.query
        or parts.fragment
        or not parts.path.startswith("/abs/")
    ):
        raise ValueError("invalid_arxiv_identifier")
    identifier = parts.path[5:]
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z][a-z.-]+/\d{7})v[1-9]\d*", identifier):
        raise ValueError("arxiv_version_required")
    if re.sub(r"v[1-9]\d*$", "", identifier) != re.sub(r"v[1-9]\d*$", "", plan.identifier):
        raise ValueError("arxiv_identifier_mismatch")
    if re.search(r"v[1-9]\d*$", plan.identifier) and identifier != plan.identifier:
        raise ValueError("arxiv_version_mismatch")
    result = _base("arxiv", identifier, url)
    title = _string(entry.findtext("a:title", namespaces=ns), 8192)
    abstract = _string(entry.findtext("a:summary", namespaces=ns), 1048576)
    published = entry.findtext("a:published", namespaces=ns)
    if not published or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", published):
        raise ValueError("invalid_arxiv_published_date")
    result.update(
        title=title,
        abstract=abstract,
        abstract_format="plain_text" if abstract is not None else "none",
        authors=[_string(author.findtext("a:name", namespaces=ns), 2048) for author in entry.findall("a:author", ns)],
        year=int(published[:4]),
        year_source="published",
        arxiv_id=identifier,
    )
    doi = entry.findtext("x:doi", namespaces=ns)
    result["doi"] = _string(doi, 512)
    license_url = entry.findtext("x:license", namespaces=ns)
    if license_url:
        if urlsplit(license_url).scheme not in {"http", "https"} or not urlsplit(license_url).hostname:
            raise ValueError("invalid_arxiv_license")
        result["license_urls"] = [_string(license_url, 4096)]
    return result


@dataclass(frozen=True)
class Acquisition:
    metadata: dict | None
    metadata_blob: StoredBlob | None
    pdf: StoredBlob | None
    pdf_url: str | None
    reason: str | None
    receipts: list[dict]


async def acquire(plan: ReferencePlan, fetcher: OfficialFetcher, store: CAS, domain: str) -> Acquisition:
    """No SQL; raw metadata bytes survive even a rejected provider projection."""
    context = FetchContext.start()
    metadata, raw, pdf, pdf_url, reason = None, None, None, None, None
    stage = "metadata"

    async def save(data: bytes, maximum: int) -> StoredBlob:
        try:
            return await anyio.to_thread.run_sync(store.put, domain, io.BytesIO(data), maximum)
        except (OSError, ValueError) as exc:
            raise DomainError("reference_storage_unavailable", 503) from exc

    try:
        result = await fetcher.fetch(
            plan.metadata_url,
            maximum_bytes=METADATA_LIMIT,
            context=context,
            accept=b"application/json" if plan.adapter == "doi_crossref" else b"application/atom+xml",
        )
        if result.body:
            raw = await save(result.body, METADATA_LIMIT)
        if result.status != 200:
            reason = "reference_metadata_http_" + str(result.status)
        elif plan.adapter == "doi_crossref":
            if result.content_type not in {"application/json", "application/vnd.crossref-api-message+json"}:
                raise ValueError("crossref_json_media_required")
            metadata = parse_crossref(result.body, plan, result.url)
            reason = "publisher_fulltext_not_in_supported_download_adapters"
        else:
            if result.content_type not in {"application/atom+xml", "text/xml", "application/xml"}:
                raise ValueError("arxiv_atom_media_required")
            metadata = parse_arxiv(result.body, plan, result.url)
            stage = "fulltext"
            # Current arXiv canonical PDF URLs omit .pdf; the old suffix incurs
            # a needless redirect and consumes the same bounded total window.
            pdf_url = "https://arxiv.org/pdf/" + quote(metadata["arxiv_id"], safe="/")
            document = await fetcher.fetch(pdf_url, maximum_bytes=PDF_LIMIT, context=context, accept=b"application/pdf")
            pdf_url = document.url
            if document.status != 200:
                reason = "reference_fulltext_http_" + str(document.status)
            elif document.content_type not in {
                "application/pdf",
                "application/octet-stream",
            } or not document.body.startswith(b"%PDF-"):
                reason = "reference_fulltext_pdf_signature_or_media_rejected"
            else:
                pdf = await save(document.body, PDF_LIMIT)
    except MaterialError as exc:
        reason = exc.code.lower()
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        UnicodeError,
        ElementTree.ParseError,
        DefusedXmlException,
        RecursionError,
    ) as exc:
        reason = "reference_" + stage + "_invalid_" + type(exc).__name__.lower()
    return Acquisition(metadata, raw, pdf, pdf_url, reason, context.receipts)


class ReferenceService:
    """Short SQL phases; replay always rechecks current referenced ACLs."""

    def __init__(self, db: Session, actor: Actor, store: CAS) -> None:
        self.db, self.actor, self.store = db, actor, store
        self.catalog = Catalog(db, actor)
        self.materials = MaterialService(db, actor, store)

    def recheck(self, body: dict, result: dict) -> dict:
        if body["project_id"]:
            self.catalog.project(UUID(body["project_id"]))
        for match in result["matches"]:
            self.catalog.work(UUID(match["work_id"]))
            bindings = {binding["file_id"]: binding for binding in match.get("file_bindings", [])}
            for binding in bindings.values():
                self.catalog.file(UUID(binding["file_id"]))
            for file in match["files"]:
                self.catalog.file(UUID(file["file_id"]))
                if (
                    file["file_id"] not in bindings
                    or not bindings[file["file_id"]]["reuse_eligible"]
                    or not self.binding(UUID(file["file_id"]))["reuse_eligible"]
                ):
                    raise DomainError("reference_identity_proof_unavailable", 409)
        if result["file"]:
            self.catalog.file(UUID(result["file"]["file_id"]))
        return checked(result)

    def binding(self, file_id: UUID) -> dict:
        try:
            grade = self.materials.get_ingestion(file_id)
        except DomainError as exc:
            if exc.status == 404:
                grade = {"binding_source": "none", "identity_status": "unverified", "formal_evidence_ready": False}
            else:
                raise
        eligible = (
            grade["binding_source"] == "trusted_source"
            and grade["identity_status"] == "hash_verified"
            and grade["formal_evidence_ready"] is True
        )
        return {
            "file_id": str(file_id),
            "binding_source": grade["binding_source"],
            "identity_status": grade["identity_status"],
            "reuse_eligible": eligible,
        }

    def preflight(self, key: UUID, payload: dict) -> dict | None:
        if payload["body"]["project_id"]:
            self.catalog.project(UUID(payload["body"]["project_id"]))
        old = self.db.get(core.Operation, (self.actor.owner_id, key))
        if old is None:
            return None
        if old.payload_sha256 != digest(payload) or old.result is None:
            raise DomainError("idempotency_conflict", 409)
        return self.recheck(payload["body"], old.result)

    def matches(self, plan: ReferencePlan | None) -> list[dict]:
        if plan is None:
            return []
        namespace = "doi" if plan.adapter == "doi_crossref" else "arxiv"
        identifiers = {plan.identifier}
        if namespace == "arxiv":
            identifiers.add(re.sub(r"v[1-9]\d*$", "", plan.identifier))
        work_ids = self.db.scalars(
            select(core.Identifier.work_id).where(
                core.Identifier.security_domain.in_(["public", self.actor.security_domain]),
                core.Identifier.namespace == namespace,
                core.Identifier.normalized.in_(identifiers),
            )
        ).all()
        result = []
        for work_id in dict.fromkeys(work_ids):
            try:
                work = self.catalog.work(work_id)
            except DomainError as exc:
                if exc.status == 404:
                    continue
                raise
            # A Work-level arXiv identifier, version_label or user association
            # cannot establish which arXiv revision these bytes represent.
            # Keep the Work candidate but do not short-circuit official exact
            # version acquisition without a dedicated trusted revision proof.
            files = (
                []
                if namespace == "arxiv"
                else self.db.scalars(
                    select(core.File.id)
                    .join(core.WorkVersion, core.File.version_id == core.WorkVersion.id)
                    .where(
                        core.WorkVersion.work_id == work_id,
                        core.File.owner_id == self.actor.owner_id,
                        core.File.deleted_at.is_(None),
                        core.File.status.in_(["ready", "partial"]),
                    )
                ).all()
            )
            bindings = [self.binding(file_id) for file_id in files]
            eligible = {UUID(binding["file_id"]) for binding in bindings if binding["reuse_eligible"]}
            result.append(
                {
                    "work_id": str(work_id),
                    "title": work.title,
                    "match_basis": "exact_doi_identifier" if namespace == "doi" else "exact_arxiv_identifier",
                    "files": [self.materials.get_file(file_id) for file_id in files if file_id in eligible],
                    "file_bindings": bindings,
                }
            )
        return result

    def finalize(self, key: UUID, payload: dict, plan: ReferencePlan | None, acquisition: Acquisition | None) -> dict:
        self.catalog.lock_owner()
        replay = self.preflight(key, payload)
        if replay is not None:
            return replay

        def effect() -> dict:
            matches = self.matches(plan)
            file, raw = None, acquisition.metadata_blob if acquisition else None
            if raw:
                prior = self.db.get(core.Blob, (raw.security_domain, raw.sha256))
                if prior is None:
                    self.db.add(
                        core.Blob(security_domain=raw.security_domain, sha256=raw.sha256, size_bytes=raw.size_bytes)
                    )
                elif prior.size_bytes != raw.size_bytes:
                    raise DomainError("metadata_blob_size_conflict", 500)
            existing = any(match["files"] for match in matches)
            if acquisition and acquisition.pdf and not existing:
                blob = acquisition.pdf
                ticket = self.materials.create_upload(
                    {
                        "file_name": "arxiv-reference.pdf",
                        "media_type": "application/pdf",
                        "size_bytes": blob.size_bytes,
                        "expected_sha256": blob.sha256,
                        "project_id": payload["body"]["project_id"],
                    },
                    key=uuid5(NAMESPACE_URL, str(self.actor.owner_id) + str(key) + ":reference-pdf"),
                )
                file = self.materials.put_upload_content(UUID(ticket["upload_id"]), blob)
            metadata = acquisition.metadata if acquisition else None
            reason = acquisition.reason if acquisition else "unsupported_reference_host_or_adapter"
            result = {
                "schema_version": "cq.reference-resolution/1",
                "operation_id": str(key),
                "request": {
                    "input_type": "doi" if payload["input_schema"] == "DoiImportInput" else "url",
                    "value": payload["body"]["doi"]
                    if payload["input_schema"] == "DoiImportInput"
                    else payload["body"]["url"],
                    "project_id": payload["body"]["project_id"],
                },
                "adapter": plan.adapter if plan else "unsupported",
                "identifier": plan.identifier if plan else None,
                "state": "queued"
                if file
                else "existing_material"
                if existing
                else "metadata_only"
                if metadata
                else "unavailable",
                "metadata": metadata,
                "matches": matches,
                "file": file,
                "fulltext": {
                    "status": "queued" if file else "existing_authorized_material" if existing else "unavailable",
                    "reason": None if file or existing else reason,
                    "source_url": acquisition.pdf_url if acquisition else None,
                    "new_material_rights_status": "unknown" if file else None,
                    "rights_verification_performed": False,
                },
                "provenance": {
                    "metadata_blob": {"sha256": raw.sha256, "size_bytes": raw.size_bytes} if raw else None,
                    "network_receipts": acquisition.receipts if acquisition else [],
                    "local_lookup_scope": "current_actor_and_public_identifiers",
                },
                "warnings": [
                    "Provider metadata and download availability do not verify rights or paper identity.",
                    "Queued material requires isolated parsing; Agent QA is not enabled.",
                ],
            }
            return checked(result)

        result = self.catalog.operation(key, payload, effect)
        return self.recheck(payload["body"], result)
