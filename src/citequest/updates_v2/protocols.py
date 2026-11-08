"""Namespace-aware official Atom/OAI codecs; no inferred deletion/freshness."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from urllib.parse import urlsplit
from xml.etree.ElementTree import Element

from defusedxml import ElementTree

ATOM = "http://www.w3.org/2005/Atom"
OAI = "http://www.openarchives.org/OAI/2.0/"
OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"
ARXIV = "http://arxiv.org/schemas/atom"
ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z][a-z.-]+/\d{7})(?:v[1-9]\d*)?\Z")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("aware_source_datetime_required")
    return result.astimezone(UTC)


def canonical_arxiv(identifier: str) -> tuple[str, int | None]:
    if type(identifier) is not str or not ARXIV_ID.fullmatch(identifier):
        raise ValueError("invalid_arxiv_identifier")
    version = re.search(r"v([1-9]\d*)$", identifier)
    return (identifier[:version.start()], int(version[1])) if version else (identifier, None)


def literal(node: Element, tag: str) -> str | None:
    child = node.find(tag)
    if child is None:
        return None
    if list(child):
        raise ValueError("unsupported_inner_XML_metadata_preserve_raw")
    return child.text or ""


def parse_atom(data: bytes, *, expected_start: int, maximum_records: int) -> dict:
    root = ElementTree.fromstring(data, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    if root.tag != "{" + ATOM + "}feed":
        raise ValueError("not_an_Atom_feed")
    raw_start = root.findtext("{" + OPENSEARCH + "}startIndex")
    if raw_start is None or not re.fullmatch(r"[0-9]+", raw_start) or int(raw_start) != expected_start:
        raise ValueError("Atom_start_index_mismatch")
    entries = root.findall("{" + ATOM + "}entry")
    totals = root.findtext("{" + OPENSEARCH + "}totalResults")
    if totals is None or not re.fullmatch(r"[0-9]+", totals):
        raise ValueError("Atom_total_results_missing")
    total = int(totals)
    if len(entries) > maximum_records:
        raise ValueError("Atom_page_record_cap")
    if len(entries) < min(maximum_records, max(0, total - expected_start)):
        raise ValueError("Atom_short_partial_page")
    records = []
    for ordinal, entry in enumerate(entries):
        url = literal(entry, "{" + ATOM + "}id") or ""
        parts = urlsplit(url)
        if parts.hostname not in {"arxiv.org", "www.arxiv.org"} or parts.scheme not in {"http", "https"} or not parts.path.startswith("/abs/"):
            raise ValueError("Atom_article_identifier_required")
        if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
            raise ValueError("Atom_identifier_components_rejected")
        identifier = parts.path[5:]
        native, version = canonical_arxiv(identifier)
        if version is None:
            raise ValueError("Atom_explicit_version_required")
        updated = literal(entry, "{" + ATOM + "}updated")
        published = literal(entry, "{" + ATOM + "}published")
        if not updated or not published:
            raise ValueError("Atom_content_dates_required")
        timestamp(updated)
        publication = timestamp(published)
        comment = literal(entry, "{" + ARXIV + "}comment")
        record = {"native_id": native, "source_revision": "v" + str(version), "version": version,
                  "title": literal(entry, "{" + ATOM + "}title"), "abstract": literal(entry, "{" + ATOM + "}summary"),
                  "authors": [literal(author, "{" + ATOM + "}name") for author in entry.findall("{" + ATOM + "}author")],
                  "year": publication.year, "published": published, "updated": updated, "comment": comment,
                  "categories": [c.get("term") for c in entry.findall("{" + ATOM + "}category")],
                  "state": "quarantine_withdrawal_signal" if re.search(r"\bwithdrawn\b", comment or "", re.I) else "observed_metadata",
                  "record_ordinal": ordinal, "source_url": url, "status_asof": None,
                  "license": None, "fulltext_rights": "unknown", "protocol": "Atom"}
        record["record_sha256"] = digest({key: value for key, value in record.items() if key != "record_ordinal"})
        records.append(record)
    return {"protocol": "Atom", "start": expected_start, "total_results": total, "records": records, "raw_sha256": hashlib.sha256(data).hexdigest()}


def parse_oai(data: bytes, *, expected_identifier: str | None = None, expected_verb: str = "GetRecord") -> dict:
    root = ElementTree.fromstring(data, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    if root.tag != "{" + OAI + "}OAI-PMH":
        raise ValueError("not_OAI_PMH")
    errors = root.findall("{" + OAI + "}error")
    if errors:
        if len(errors) != 1:
            raise ValueError("ambiguous_OAI_error")
        code = errors[0].get("code")
        return {"protocol": "OAI", "error": code, "records": [], "deleted": False,
                "state": "quarantine_missing" if code == "idDoesNotExist" else "protocol_error"}
    verb = root.find("{" + OAI + "}" + expected_verb)
    if verb is None:
        raise ValueError("OAI_verb_missing")
    records = []
    for ordinal, record in enumerate(verb.findall("{" + OAI + "}record")):
        header = record.find("{" + OAI + "}header")
        if header is None:
            raise ValueError("OAI_header_missing")
        identifier = header.findtext("{" + OAI + "}identifier")
        if not identifier or not identifier.startswith("oai:arXiv.org:"):
            raise ValueError("OAI_arxiv_ID_missing")
        native, _ = canonical_arxiv(identifier[14:])
        if expected_identifier is not None and native != canonical_arxiv(expected_identifier)[0]:
            raise ValueError("OAI_managed_identifier_mismatch")
        deleted = header.get("status") == "deleted"
        metadata = record.find("{" + OAI + "}metadata")
        if deleted:
            if metadata is not None:
                raise ValueError("deleted_header_must_not_contain_metadata")
            value = {"native_id": native, "state": "quarantine_deleted_header", "deleted": True, "status_asof": None}
        else:
            if metadata is None or len(metadata) != 1:
                raise ValueError("OAI_single_metadata_required")
            body = metadata[0]
            ns = body.tag[1:].split("}", 1)[0]
            if ns not in {"http://arxiv.org/OAI/arXiv/", "http://arxiv.org/OAI/arXivRaw/"}:
                raise ValueError("unsupported_arxiv_OAI_metadata")
            content_id = literal(body, "{" + ns + "}id")
            if content_id and canonical_arxiv(content_id)[0] != native:
                raise ValueError("OAI_content_identifier_mismatch")
            versions = []
            for child in body.findall("{" + ns + "}version"):
                raw_version = child.get("version", "")
                if not re.fullmatch(r"v[1-9]\d*", raw_version):
                    raise ValueError("OAI_Raw_version_invalid")
                versions.append(int(raw_version[1:]))
            comments = literal(body, "{" + ns + "}comments")
            value = {"native_id": native, "title": literal(body, "{" + ns + "}title"), "abstract": literal(body, "{" + ns + "}abstract"),
                     "comment": comments, "metadata_format": body.tag, "version": max(versions) if versions else None,
                     "created": literal(body, "{" + ns + "}created"), "updated": literal(body, "{" + ns + "}updated"),
                     "license_snapshot": literal(body, "{" + ns + "}license"), "license_current_verified": False,
                     "state": "quarantine_withdrawal_signal" if re.search(r"\bwithdrawn\b", comments or "", re.I) else "observed_metadata",
                     "deleted": False, "status_asof": None}
        value.update(protocol="OAI", record_ordinal=ordinal, datestamp=header.findtext("{" + OAI + "}datestamp"))
        value["record_sha256"] = digest({key: item for key, item in value.items() if key != "record_ordinal"})
        records.append(value)
    if expected_verb == "GetRecord" and len(records) != 1:
        raise ValueError("OAI_GetRecord_exactly_one_required")
    token = verb.find("{" + OAI + "}resumptionToken")
    return {"protocol": "OAI", "records": records, "token": token.text if token is not None else None,
            "token_expiration": token.get("expirationDate") if token is not None else None,
            "response_date": root.findtext("{" + OAI + "}responseDate"), "status_asof": None}


def validate_discovery(pages: list[dict], head_recheck: dict, *, lower: datetime, upper: datetime, page_size: int, overlap: int) -> list[dict]:
    if not pages or not 0 < overlap < page_size or lower >= upper:
        raise ValueError("discovery_profile_or_pages_invalid")
    def key(row: dict) -> tuple:
        return row["native_id"], row["source_revision"], row["record_sha256"]

    if pages[0].get("total_results") != head_recheck.get("total_results"):
        raise ValueError("Atom_total_results_fence_changed")
    if [key(r) for r in pages[0]["records"]] != [key(r) for r in head_recheck["records"]]:
        raise ValueError("Atom_head_fence_changed")
    unique = {}
    previous = None
    for number, page in enumerate(pages):
        if page["total_results"] != pages[0]["total_results"]:
            raise ValueError("Atom_page_total_results_changed")
        if page["start"] != number * (page_size - overlap):
            raise ValueError("Atom_offset_sequence_changed")
        rows = page["records"]
        times = [timestamp(row["updated"]) for row in rows]
        if times != sorted(times, reverse=True):
            raise ValueError("Atom_page_order_changed")
        if previous is not None:
            if [key(r) for r in previous[-overlap:]] != [key(r) for r in rows[:overlap]]:
                raise ValueError("Atom_overlap_changed")
            if rows and timestamp(rows[0]["updated"]) > timestamp(previous[-overlap]["updated"]):
                raise ValueError("Atom_global_order_changed")
        for row in rows:
            existing = unique.get(row["native_id"])
            if existing is not None and key(existing) != key(row):
                raise ValueError("Atom_same_ID_version_or_content_conflict")
            unique[row["native_id"]] = row
        previous = rows
    tail = pages[-1]["records"]
    if tail and len(tail) >= page_size and not any(timestamp(row["updated"]) < lower for row in tail):
        raise ValueError("Atom_lower_fence_not_crossed")
    return [row for row in unique.values() if lower <= timestamp(row["updated"]) <= upper]


def reconciliation_state(atom: dict | None, oai: dict) -> str:
    if oai.get("state") == "quarantine_missing":
        return "quarantine_missing"
    if oai.get("error"):
        return "unknown"
    record = oai["records"][0]
    if record["state"].startswith("quarantine_"):
        return record["state"]
    if atom and atom["state"].startswith("quarantine_"):
        return atom["state"]
    if atom and record.get("version") is not None and atom["version"] != record["version"]:
        return "quarantine_version_conflict"
    return "unknown"  # Matching versions/dates are not authoritative current-status proofs.
