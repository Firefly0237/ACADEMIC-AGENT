"""Deterministic full-available-source windows, independent of similarity top-k.

The caller must persist the server-issued manifest under its Task and must not
accept a replacement manifest/selection from model arguments. Every read and
batch verification repeats current source authorization; a manifest is not an
ACL cache or a claim of complete PDF/image extraction.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID
from sqlalchemy import select
from citequest.index_v2.rows_v2 import canonical_sha
from citequest.materials_v2.models import MaterialFragment
from citequest.search_v3.schemas import PaperLocator
from citequest.v2 import models as core
from citequest.v2.domain import Actor, DomainError
from .workset_contracts import DocumentManifest, SourceSelection, WindowDescriptor, WorksetManifest

MAX_WINDOW = 16384


def encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Unit:
    reference: dict
    text: str
    unit_sha256: str
    coordinate: str
    pin: dict
    start: int = 0
    end: int | None = None


@dataclass
class Document:
    document_id: str
    selection: SourceSelection
    title: str | None
    version: str
    coverage: str
    missing: list[dict]
    units: list[Unit]
    loaded_units: int
    loaded_bytes: int


class FullSourceWindows:
    def __init__(self, search, materials):
        self.search, self.materials = search, materials

    @staticmethod
    def _selection(value: dict) -> SourceSelection:
        selection = SourceSelection.model_validate(value)
        if selection.kind == "material_file":
            return SourceSelection(kind=selection.kind, id=str(UUID(selection.id)))
        locator = PaperLocator(kind=selection.kind, id=selection.id)
        return SourceSelection(kind=locator.kind, id=locator.id)

    def _load(self, actor: Actor, selection: SourceSelection, document_id: str) -> Document:
        if selection.kind != "material_file":
            with self.search.factory.begin() as db:
                source, snapshot = self.search._current_source(db, actor, PaperLocator(kind=selection.kind, id=selection.id))
                work = db.get(core.Work, source.work_id, populate_existing=True) if source.work_id else None
                title = work.title if work is not None and selection.kind == "native_representation" else source.original_source_dto.get("title")
                pin = {"source_snapshot_id": str(snapshot.id), "source_manifest_sha256": source.root.source_manifest_sha256,
                    "source_seal_sha256": source.root.source_seal_sha256, "projection_sha256": source.projection_sha256,
                    "source_pointer_sha256": source.source_pointer_sha256, "unit_text_sha256": source.unit_text_sha256,
                    "work_id": str(source.work_id) if source.work_id else None,
                    "work_version_id": str(source.work_version_id) if source.work_version_id else None,
                    "metadata_revision": work.metadata_revision if work is not None else None}
                text = source.literal_text
            coverage = "native_literal" if selection.kind == "native_representation" else "metadata_only"
            units = [Unit(selection.model_dump(), text, source.unit_text_sha256, source.coordinate_system, pin)] if text else []
            version = canonical_sha({"selection": selection.model_dump(), "pin": pin, "title": title, "coverage": coverage})
            return Document(document_id, selection, title, version, coverage, [], units, int(bool(text)), len(text.encode()))
        if self.materials is None:
            raise DomainError("material_source_not_configured", 503)
        with self.materials.factory.begin() as db:
            file, representation, parsed = self.materials._source(db, actor, UUID(selection.id))
            rows = list(db.scalars(select(MaterialFragment).where(MaterialFragment.owner_id == actor.owner_id,
                MaterialFragment.file_id == file.id, MaterialFragment.representation_id == representation.id)
                .order_by(MaterialFragment.ordinal, MaterialFragment.id).limit(32001)))
            if len(rows) > 32000:
                raise DomainError("workset_source_fragment_limit_exceeded", 413)
            pin = {"file_id": str(file.id), "file_revision": file.revision, "source_file_sha256": file.sha256,
                "representation_id": representation.id, "material_version_id": str(representation.material_version_id),
                "config_sha256": representation.config_sha256, "artifact_sha256": representation.artifact_sha256,
                "outbound_policy": file.outbound_policy, "rights_status": file.rights_status}
            missing = [value if isinstance(value, dict) else {"reason": value} for value in parsed["missing_ranges"]]
            if len(missing) > 500:
                raise DomainError("workset_missing_range_limit_exceeded", 413)
            coverage = "material_full_text" if parsed["coverage"] == "text_full" and not missing else "material_partial"
            groups: dict[str, list[tuple[int, int, MaterialFragment]]] = {}
            order: dict[str, int] = {}
            signatures, loaded_bytes = [], 0
            for row in rows:
                text = row.source_text
                if row.payload.get("source_text") != text or row.payload.get("source_text_sha256") != sha(text):
                    raise DomainError("workset_fragment_source_integrity_failure", 409)
                locator = row.payload.get("locator", {})
                start, end = locator.get("char_start"), locator.get("char_end")
                if type(start) is not int or type(end) is not int or not 0 <= start < end or end-start != len(text):
                    raise DomainError("workset_fragment_unicode_locator_invalid", 409)
                static_locator = {key: value for key, value in locator.items() if key not in {"char_start", "char_end", "line_1based"}}
                group = canonical_sha({"source_block_id": row.payload.get("source_block_id"), "locator": static_locator})
                groups.setdefault(group, []).append((start, end, row))
                order[group] = min(order.get(group, row.ordinal), row.ordinal)
                signatures.append({"id": row.id, "ordinal": row.ordinal, "text_sha256": sha(text), "locator_sha256": canonical_sha(locator)})
                loaded_bytes += len(text.encode())
            # Deduplicate only proven identical overlap inside the same actual
            # source block. Missing gaps are not filled with invented text.
            units = []
            for group in sorted(groups, key=lambda value: (order[value], value)):
                active_start, active_text = None, ""
                for start, end, row in sorted(groups[group], key=lambda value: (value[0], value[1], value[2].id)):
                    previous_end = active_start+len(active_text) if active_start is not None else start
                    overlap = max(0, min(end, previous_end)-start)
                    if overlap and row.source_text[:overlap] != active_text[start-active_start:start-active_start+overlap]:
                        raise DomainError("workset_overlapping_fragments_disagree", 409)
                    if end <= previous_end and active_start is not None:
                        continue
                    units.append(Unit({"kind": "material_fragment", "id": row.id}, row.source_text, sha(row.source_text),
                        "unicode_codepoints_in_material_fragment", {**pin, "source_block_sha256": group}, overlap, len(row.source_text)))
                    if active_start is None or start > previous_end:
                        active_start, active_text = start, row.source_text
                    else:
                        active_text += row.source_text[overlap:]
            version = canonical_sha({"selection": selection.model_dump(), "pin": pin, "fragments": signatures,
                "coverage": coverage, "missing": missing})
            return Document(document_id, selection, None, version, coverage, missing, units, len(rows), loaded_bytes)

    @staticmethod
    def _evidence(actor, document, unit, start, end):
        text = unit.text[start:end]
        value = {"evidence_type": "source_window", "document_id": document.document_id,
            "selection": document.selection.model_dump(), "source_version_sha256": document.version,
            "source_ref": unit.reference, "source_pin_sha256": canonical_sha(unit.pin),
            "unit_text_sha256": unit.unit_sha256, "source_sha256": sha(text), "text": text,
            "span_start": start, "span_end": end, "coordinate_system": unit.coordinate, "coverage_kind": document.coverage}
        identifier = canonical_sha({"owner": str(actor.owner_id), **{key: item for key, item in value.items() if key != "text"}})
        return {"ref_id": "source-window:"+identifier, **value}

    @staticmethod
    def _result(actor, document, evidence, ordinal):
        pin = canonical_sha({"source_version": document.version, "ref_ids": [item["ref_id"] for item in evidence]})
        identifier = canonical_sha({"owner": str(actor.owner_id), "document_id": document.document_id,
            "source_version": document.version, "ordinal": ordinal, "source_pin": pin})
        result = {"window_id": identifier, "document_id": document.document_id, "evidence": evidence,
            "source_reads": {"authorized_documents_loaded": 1, "literal_units_loaded": document.loaded_units,
                "literal_text_bytes_loaded": document.loaded_bytes,
                "evidence_text_bytes_returned": sum(len(item["text"].encode()) for item in evidence)}}
        descriptor = WindowDescriptor(window_id=identifier, ordinal=ordinal, source_parts=len(evidence),
            text_bytes=sum(len(item["text"].encode()) for item in evidence), encoded_result_bytes=len(encoded(result)), source_pin_sha256=pin)
        return descriptor, result

    def _windows(self, actor, document):
        windows, pieces = [], []
        def fits(candidate):
            if len(candidate) > 32:
                return False
            try:
                self._result(actor, document, candidate, len(windows)+1)
                return True
            except ValueError:
                return False
        def flush():
            if pieces:
                windows.append(self._result(actor, document, list(pieces), len(windows)+1))
                pieces.clear()
                if len(windows) > 1000:
                    raise DomainError("workset_source_window_limit_exceeded", 413)
        for unit in document.units:
            cursor, end = unit.start, len(unit.text) if unit.end is None else unit.end
            while cursor < end:
                if len(pieces) == 32:
                    flush()
                candidate = self._evidence(actor, document, unit, cursor, end)
                if fits([*pieces, candidate]):
                    pieces.append(candidate)
                    break
                low, high = 0, end-cursor
                while low < high:
                    middle = (low+high+1)//2
                    part = self._evidence(actor, document, unit, cursor, cursor+middle)
                    if fits([*pieces, part]): low = middle
                    else: high = middle-1
                if not low:
                    if not pieces:
                        raise DomainError("workset_source_metadata_cannot_fit_one_codepoint", 413)
                    flush()
                    continue
                pieces.append(self._evidence(actor, document, unit, cursor, cursor+low))
                cursor += low
                flush()
        flush()
        return windows

    @staticmethod
    def _manifest_id(actor, documents, complete):
        return canonical_sha({"owner": str(actor.owner_id), "schema_version": "citequest-workset-manifest/1",
            "documents": [document.model_dump(mode="json") for document in documents], "full_coverage_planned": complete, "window_max_encoded_bytes": MAX_WINDOW})

    def plan(self, actor: Actor, selections: list[dict]) -> WorksetManifest:
        if type(selections) is not list or not 1 <= len(selections) <= 10:
            raise DomainError("workset_requires_one_to_ten_selected_sources", 422)
        selected = [self._selection(value) for value in selections]
        if len({(value.kind, value.id) for value in selected}) != len(selected):
            raise DomainError("workset_selected_sources_must_be_unique", 422)
        documents = []
        for ordinal, selection in enumerate(selected, 1):
            current = self._load(actor, selection, f"D{ordinal:02}")
            windows = self._windows(actor, current)
            codepoints = sum((len(unit.text) if unit.end is None else unit.end)-unit.start for unit in current.units)
            size = sum(len(unit.text[unit.start:unit.end].encode()) for unit in current.units)
            if sum(sum(len(piece["text"]) for piece in result["evidence"]) for _, result in windows) != codepoints:
                raise DomainError("workset_window_union_not_lossless", 500)
            documents.append(DocumentManifest(document_id=current.document_id, selection=selection, title=current.title,
                source_version_sha256=current.version, coverage_kind=current.coverage, available_text_bytes=size,
                available_codepoints=codepoints, missing_ranges=current.missing, windows=[descriptor for descriptor, _ in windows]))
        complete = all(document.available_codepoints > 0 for document in documents)
        return WorksetManifest(manifest_id=self._manifest_id(actor, documents, complete), documents=documents, full_coverage_planned=complete)

    def read_window(self, actor: Actor, manifest: WorksetManifest, window_id: str) -> dict:
        if manifest.manifest_id != self._manifest_id(actor, manifest.documents, manifest.full_coverage_planned):
            raise DomainError("workset_manifest_owner_or_contents_changed", 403)
        matches = [(document, window) for document in manifest.documents for window in document.windows if window.window_id == window_id]
        if len(matches) != 1:
            raise DomainError("window_not_in_task_workset", 404)
        document, descriptor = matches[0]
        current = self._load(actor, document.selection, document.document_id)
        if current.version != document.source_version_sha256:
            raise DomainError("workset_source_version_changed", 409)
        windows = self._windows(actor, current)
        if descriptor.ordinal > len(windows) or windows[descriptor.ordinal-1][0] != descriptor:
            raise DomainError("workset_current_window_pin_changed", 409)
        return windows[descriptor.ordinal-1][1]

    def verify_pinned_batch(self, actor: Actor, evidence: list[dict]) -> dict:
        if type(evidence) is not list or not 1 <= len(evidence) <= 64:
            raise DomainError("workset_verify_batch_requires_one_to_64_refs", 422)
        cache, pins, sources_loaded, bytes_loaded, units_loaded = {}, [], 0, 0, 0
        for item in evidence:
            if not isinstance(item, dict) or item.get("evidence_type") != "source_window":
                raise DomainError("source_window_evidence_required", 422)
            selection = self._selection(item.get("selection"))
            key = (item.get("document_id"), selection.kind, selection.id)
            if key not in cache:
                current = self._load(actor, selection, item.get("document_id"))
                cache[key] = current
                sources_loaded += 1
                bytes_loaded += current.loaded_bytes
                units_loaded += current.loaded_units
            current = cache[key]
            if current.version != item.get("source_version_sha256"):
                raise DomainError("workset_pinned_source_version_changed", 409)
            candidates = [unit for unit in current.units if unit.reference == item.get("source_ref")]
            start, end = item.get("span_start"), item.get("span_end")
            valid = None
            for unit in candidates:
                unit_end = len(unit.text) if unit.end is None else unit.end
                if type(start) is int and type(end) is int and unit.start <= start < end <= unit_end:
                    expected = self._evidence(actor, current, unit, start, end)
                    if expected == item:
                        valid = expected
                        break
            if valid is None:
                raise DomainError("workset_pinned_literal_or_locator_changed", 409)
            pins.append({key: valid[key] for key in ("ref_id", "document_id", "source_version_sha256", "source_sha256", "unit_text_sha256")})
        return {"valid": True, "pins": pins, "source_reads": {"authorized_documents_loaded": sources_loaded,
            "literal_units_loaded": units_loaded, "literal_text_bytes_loaded": bytes_loaded, "refs_verified": len(pins)}}
