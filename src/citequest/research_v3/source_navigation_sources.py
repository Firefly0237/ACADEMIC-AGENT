"""Current-authority source loading for immutable structure navigation.

The derived index is reusable. Its pointer is never an access grant. Every
expansion rereads the selected source and compares the captured authority pin.
"""
from __future__ import annotations

import json
from uuid import UUID

from sqlalchemy import select

from citequest.data_v2 import models as data
from citequest.data_v2.native_codec import NativeRepresentationDraft
from citequest.materials_v2.models import MaterialFragment
from citequest.v2 import models as core
from citequest.v2.domain import DomainError
from citequest.v2.storage import StoredBlob

from .source_structure import StructureSource, StructureUnit, read_leaf
from .source_structure_cache import SourceStructureCache, native_structure_source
from .source_windows import FullSourceWindows, sha


def digest(value) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


class NavigationSources:
    def __init__(self, search, materials):
        self.search, self.materials = search, materials
        self.windows = FullSourceWindows(search, materials)

    def _current(self, actor, selection, document_id):
        selected = self.windows._selection(selection)
        document = self.windows._load(actor, selected, document_id)
        service = self.materials if selected.kind == "material_file" else self.search
        if service is None:
            raise DomainError("navigation_source_service_unavailable", 503)
        if selected.kind == "native_representation":
            with service.factory.begin() as db:
                native = db.get(data.NativeRepresentation, selected.id)
                if native is None:
                    raise DomainError("navigation_source_unavailable", 409)
                if (native.id != selected.id or native.security_domain not in {"public", actor.security_domain}
                        or native.storage_security_domain != native.security_domain):
                    raise DomainError("navigation_current_native_identity_changed", 409)
                blob = db.get(core.Blob, (native.storage_security_domain, native.draft_sha256))
                if blob is None:
                    raise DomainError("navigation_source_blob_unavailable", 409)
                draft = NativeRepresentationDraft.model_validate_json(service.store.read(
                    StoredBlob(blob.security_domain, blob.sha256, blob.size_bytes)))
                if (draft.representation_id != selected.id or native.id != selected.id
                        or native.source_text_sha256 != draft.source_text_sha256
                        or native.security_domain != native.storage_security_domain):
                    raise DomainError("navigation_current_native_identity_changed", 409)
                source = native_structure_source(draft, native.security_domain)
            if len(document.units) != 1 or document.units[0].text != draft.literal_text:
                raise DomainError("navigation_current_native_text_changed", 409)
        elif selected.kind == "material_file":
            units = []
            with service.factory.begin() as db:
                for unit in document.units:
                    fragment = db.scalar(select(MaterialFragment).where(
                        MaterialFragment.id == unit.reference["id"], MaterialFragment.owner_id == actor.owner_id,
                        MaterialFragment.file_id == UUID(selected.id)))
                    if fragment is None or fragment.source_text != unit.text:
                        raise DomainError("navigation_current_fragment_changed", 409)
                    units.append(StructureUnit(unit.reference, unit.text, unit.unit_sha256, unit.coordinate,
                        locator=fragment.payload.get("locator", {}),
                        native_provenance=fragment.payload.get("native_provenance", {}), start=unit.start, end=unit.end))
            representation = document.units[0].pin["representation_id"] if document.units else digest(selected.model_dump())
            source = StructureSource(selected.model_dump(), representation, representation, actor.security_domain,
                None, tuple(units), document.coverage, tuple(document.missing))
        else:
            from citequest.search_v3.schemas import PaperLocator
            with service.factory.begin() as db:
                current_source, _ = service._current_source(db, actor, PaperLocator(kind=selected.kind, id=selected.id))
                security_domain = current_source.root.security_domain
                if len(document.units) != 1 or document.units[0].text != current_source.literal_text:
                    raise DomainError("navigation_current_metadata_changed", 409)
            version = digest({"selection": selected.model_dump(), "units": [unit.unit_sha256 for unit in document.units]})
            source = StructureSource(selected.model_dump(), version, version,
                security_domain,
                None, tuple(StructureUnit(unit.reference, unit.text, unit.unit_sha256, unit.coordinate,
                    start=unit.start, end=unit.end) for unit in document.units), document.coverage, tuple(document.missing))
        return document, source, SourceStructureCache(service.factory, service.store)

    def capture(self, actor, task_id, selections, check_scope):
        if not 1 <= len(selections) <= 20 or len({(row["kind"], row["id"]) for row in selections}) != len(selections):
            raise DomainError("navigation_explicit_source_set_required", 422)
        records, loaded = [], {}
        for index, selection in enumerate(selections, 1):
            check_scope()
            document_id = f"N{index:02}"
            document, source, cache = self._current(actor, selection, document_id)
            stored = cache.get_or_build(source)
            record = {"document_id": document_id, "selection": selection,
                "authority_version_sha256": document.version, "structure_sha256": stored.structure.sha256,
                "source_version_sha256": source.source_version_sha256,
                "coverage": document.coverage, "title": document.title}
            records.append(record)
            loaded[document_id] = (document, source, stored.structure)
        manifest = {"version": "citequest-source-navigation-scope/1", "owner_id": str(actor.owner_id),
            "task_id": str(task_id), "sources": records}
        check_scope()
        return manifest, loaded

    def verify(self, actor, task_id, manifest, selections):
        if (manifest.get("version") != "citequest-source-navigation-scope/1"
                or manifest.get("owner_id") != str(actor.owner_id) or manifest.get("task_id") != str(task_id)
                or not 1 <= len(manifest.get("sources", [])) <= 20):
            raise DomainError("navigation_scope_identity_changed", 409)
        allowed = {(row["kind"], row["id"]) for row in selections}
        actual = {(row["selection"]["kind"], row["selection"]["id"]) for row in manifest["sources"]}
        if actual != allowed or len(actual) != len(manifest["sources"]):
            raise DomainError("navigation_scope_selection_changed", 409)
        for record in manifest["sources"]:
            document = self.windows._load(actor, self.windows._selection(record["selection"]), record["document_id"])
            if document.version != record["authority_version_sha256"]:
                raise DomainError("navigation_source_authority_changed", 409)

    def expand(self, actor, record, node_id):
        document, source, cache = self._current(actor, record["selection"], record["document_id"])
        if document.version != record["authority_version_sha256"]:
            raise DomainError("navigation_source_authority_changed", 409)
        stored = cache.get(source)
        if stored is None or stored.structure.sha256 != record["structure_sha256"]:
            raise DomainError("navigation_structure_pointer_changed", 409)
        leaf = read_leaf(stored.structure, node_id, source)
        unit = document.units[leaf["unit_index"]]
        # Existing source-window issuance and verification remain the authority.
        return self.windows._evidence(actor, document, unit, leaf["span_start"], leaf["span_end"])


def scope_evidence(actor, task_id, manifest):
    source_hash = digest(manifest)
    return {"ref_id": "structure-scope:" + digest({"owner": str(actor.owner_id), "task": str(task_id), "scope": source_hash}),
        "evidence_type": "source_structure_scope", "source_sha256": source_hash,
        "source_ref": {"kind": "source_structure_scope", "id": source_hash}, "manifest": manifest}
