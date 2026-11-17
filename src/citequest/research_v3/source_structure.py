"""Deterministic source navigation metadata, independent of retrieval and models.

Descriptions are bounded routing hints with exact literal quote locations. Only
``read_leaf`` returns source text; its caller still owns current authorization
and evidence issuance. A stored structure is never an access grant.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from citequest.v2.storage import CAS, StoredBlob

VERSION = "citequest-source-structure/1"
POLICY_VERSION = "literal-structural-navigation/1"
SHA = re.compile(r"[a-f0-9]{64}\Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _sha(_canonical(value))


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


@dataclass(frozen=True)
class StructurePolicy:
    version: str = POLICY_VERSION
    maximum_leaf_characters: int = 1600
    maximum_description_characters: int = 240
    maximum_title_characters: int = 160
    maximum_units: int = 32000
    maximum_nodes: int = 64000
    maximum_source_characters: int = 16000000
    maximum_sidecar_bytes: int = 16 * 1024**2

    def validate(self) -> None:
        _require(self.version == POLICY_VERSION, "unsupported_structure_policy")
        for name, low, high in (
            ("maximum_leaf_characters", 64, 16000), ("maximum_description_characters", 32, 240),
            ("maximum_title_characters", 16, 160), ("maximum_units", 1, 32000),
            ("maximum_nodes", 3, 64000), ("maximum_source_characters", 1, 16000000),
            ("maximum_sidecar_bytes", 1024, 16 * 1024**2),
        ):
            value = getattr(self, name)
            _require(type(value) is int and low <= value <= high, "invalid_structure_policy_limit")
        _require(self.maximum_title_characters <= self.maximum_description_characters, "structure_title_exceeds_description")


@dataclass(frozen=True)
class StructureUnit:
    reference: dict
    text: str
    source_text_sha256: str
    coordinate_system: str
    locator: dict = field(default_factory=dict)
    native_provenance: dict = field(default_factory=dict)
    annotations: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    start: int = 0
    end: int | None = None


@dataclass(frozen=True)
class StructureSource:
    selection: dict
    source_version_sha256: str
    representation_id: str
    security_domain: str
    title: str | None
    units: tuple[StructureUnit, ...]
    coverage: str = "unknown"
    missing_ranges: tuple[dict, ...] = ()


@dataclass(frozen=True)
class SourceStructure:
    _payload: dict
    sha256: str

    def to_dict(self) -> dict:
        return copy.deepcopy(self._payload)

    @property
    def root_id(self) -> str:
        return self._payload["root_id"]

    @property
    def nodes(self) -> tuple[dict, ...]:
        return tuple(copy.deepcopy(self._payload["nodes"]))

    def node(self, node_id: str) -> dict:
        matches = [node for node in self._payload["nodes"] if node["id"] == node_id]
        _require(len(matches) == 1, "structure_node_not_found")
        return copy.deepcopy(matches[0])


@dataclass(frozen=True)
class StoredStructure:
    pointer: StoredBlob
    structure: SourceStructure


def _bounds(unit: StructureUnit) -> tuple[int, int]:
    end = len(unit.text) if unit.end is None else unit.end
    _require(type(unit.start) is int and type(end) is int and 0 <= unit.start < end <= len(unit.text),
             "structure_unit_range_invalid")
    return unit.start, end


def _source_binding(source: StructureSource, policy: StructurePolicy) -> tuple[dict, list[dict]]:
    policy.validate()
    _require(isinstance(source, StructureSource), "typed_structure_source_required")
    _require(type(source.selection) is dict and set(source.selection) == {"kind", "id"} and source.selection.get("kind") in
             {"native_representation", "metadata_projection", "material_file"}
             and type(source.selection.get("id")) is str and 1 <= len(source.selection["id"]) <= 64,
             "typed_structure_selection_required")
    _require(type(source.source_version_sha256) is str and SHA.fullmatch(source.source_version_sha256) is not None
             and type(source.representation_id) is str and SHA.fullmatch(source.representation_id) is not None,
             "structure_source_version_required")
    _require(type(source.security_domain) is str and 1 <= len(source.security_domain) <= 100,
             "structure_security_domain_required")
    _require(source.title is None or (type(source.title) is str and len(source.title) <= 8000), "invalid_structure_title")
    _require(type(source.coverage) is str and 1 <= len(source.coverage) <= 100
             and isinstance(source.missing_ranges, (tuple, list)) and len(source.missing_ranges) <= 500,
             "invalid_structure_coverage")
    _require(isinstance(source.units, (tuple, list)) and 1 <= len(source.units) <= policy.maximum_units,
             "structure_source_unit_limit")
    units, characters = [], 0
    for index, unit in enumerate(source.units):
        _require(isinstance(unit, StructureUnit) and type(unit.text) is str and unit.text and "\x00" not in unit.text,
                 "literal_structure_unit_required")
        _require(type(unit.source_text_sha256) is str and _sha(unit.text.encode()) == unit.source_text_sha256,
                 "structure_unit_hash_mismatch")
        _require(type(unit.reference) is dict and unit.reference and type(unit.coordinate_system) is str
                 and 1 <= len(unit.coordinate_system) <= 160 and type(unit.locator) is dict
                 and type(unit.native_provenance) is dict and type(unit.annotations) is dict,
                 "structure_unit_metadata_invalid")
        start, end = _bounds(unit)
        characters += len(unit.text)
        _require(characters <= policy.maximum_source_characters, "structure_source_character_limit")
        for spans in unit.annotations.values():
            _require(isinstance(spans, (list, tuple)), "structure_annotations_invalid")
            for span in spans:
                _require(isinstance(span, (list, tuple)) and len(span) == 2
                         and all(type(value) is int for value in span) and 0 <= span[0] < span[1] <= len(unit.text),
                         "structure_annotation_range_invalid")
        units.append({"unit_index": index, "reference": copy.deepcopy(unit.reference),
            "source_text_sha256": unit.source_text_sha256, "source_characters": len(unit.text),
            "available_start": start, "available_end": end, "coordinate_system": unit.coordinate_system,
            "locator": copy.deepcopy(unit.locator), "native_provenance_sha256": _digest(unit.native_provenance),
            "annotations_sha256": _digest(unit.annotations)})
    binding = {"selection": copy.deepcopy(source.selection), "source_version_sha256": source.source_version_sha256,
        "representation_id": source.representation_id, "security_domain": source.security_domain,
        "title": source.title, "coverage": source.coverage, "missing_ranges": copy.deepcopy(list(source.missing_ranges)),
        "units_sha256": _digest(units)}
    return binding, units


def source_binding_sha256(source: StructureSource, policy: StructurePolicy | None = None) -> str:
    """An immutable-input fingerprint, never the caller's current access scope."""
    return _digest(_source_binding(source, policy or StructurePolicy())[0])


def policy_sha256(policy: StructurePolicy | None = None) -> str:
    policy = policy or StructurePolicy()
    policy.validate()
    return _digest(asdict(policy))


def _span(source: StructureSource, index: int, start: int, end: int) -> dict:
    return {"unit_index": index, "span_start": start, "span_end": end,
            "source_sha256": _sha(source.units[index].text[start:end].encode())}


def _quote(source: StructureSource, span: dict, limit: int) -> dict | None:
    index, start, end = span["unit_index"], span["span_start"], span["span_end"]
    text = source.units[index].text
    while start < end and text[start].isspace():
        start += 1
    end = min(end, start + limit)
    while end > start and text[end - 1].isspace():
        end -= 1
    return _span(source, index, start, end) if start < end else None


def _literal(source: StructureSource, span: dict) -> str:
    _require(type(span) is dict and set(span) == {"unit_index", "span_start", "span_end", "source_sha256"},
             "invalid_structure_span")
    index, start, end = (span[key] for key in ("unit_index", "span_start", "span_end"))
    _require(type(index) is int and 0 <= index < len(source.units), "structure_span_unit_invalid")
    lower, upper = _bounds(source.units[index])
    _require(type(start) is int and type(end) is int and lower <= start < end <= upper, "structure_span_range_invalid")
    text = source.units[index].text[start:end]
    _require(_sha(text.encode()) == span["source_sha256"], "structure_span_hash_mismatch")
    return text


def _node(root: str, kind: str, anchor: dict | None, parent: str | None, *, title=None,
          title_origin="unknown", title_span=None, metadata=None, unknown=()) -> dict:
    return {"id": root if kind == "document" else _digest({"root": root, "kind": kind, "anchor": anchor}),
        "kind": kind, "parent_id": parent, "child_ids": [], "anchor": anchor, "title": title,
        "title_origin": title_origin, "title_span": title_span, "description": "", "description_origin": "unknown",
        "description_spans": [], "source_span": anchor if kind == "leaf" else None,
        "metadata_unknown": list(unknown), "structure_metadata": metadata or {}}


def _describe(node: dict, source: StructureSource, candidate: dict | None, policy: StructurePolicy) -> None:
    parts, quotes = [], []
    if node["title"]:
        parts.append(node["title"])
        if node["title_span"] is not None:
            quotes.append(node["title_span"])
    remaining = policy.maximum_description_characters - len("\n\n".join(parts)) - (2 if parts else 0)
    quote = _quote(source, candidate, remaining) if candidate and remaining > 0 else None
    if quote and quote != node["title_span"]:
        parts.append(_literal(source, quote))
        quotes.append(quote)
    node["description"], node["description_spans"] = "\n\n".join(parts), quotes
    node["description_origin"] = ("source_metadata_and_literal_quotes" if node["title_origin"] == "source_metadata" and quotes
        else "source_metadata" if node["title_origin"] == "source_metadata"
        else "literal_quotes" if quotes else "unknown")


def build_structure(source: StructureSource, policy: StructurePolicy | None = None) -> SourceStructure:
    policy = policy or StructurePolicy()
    binding, units = _source_binding(source, policy)
    root_id = _digest({"schema_version": VERSION, "source_binding": binding, "policy": asdict(policy)})
    root = _node(root_id, "document", None, None,
        title=source.title[:policy.maximum_title_characters] if source.title else None,
        title_origin="source_metadata" if source.title else "unknown",
        unknown=("semantic_scope",) if source.title else ("document_title", "semantic_scope"))
    nodes, sections, leaves, heading_refs = [root], [], [], {}
    current = None
    for index, unit in enumerate(source.units):
        lower, upper = _bounds(unit)
        headers = sorted(set(tuple(span) for span in unit.annotations.get("sectionheader", [])
                             if lower <= span[0] < span[1] <= upper))
        if not headers and unit.native_provenance.get("item_label") == "section_header":
            headers = [(lower, upper)]
        _require(all(left[1] <= right[0] for left, right in zip(headers, headers[1:])), "overlapping_structure_headings")
        events = dict(headers)
        boundaries = {lower, upper, *(value for span in headers for value in span)}
        for start, end in unit.annotations.get("paragraph", []):
            boundaries.update(value for value in (start, end) if lower < value < upper)
        ordered = sorted(boundaries)
        for left, right in zip(ordered, ordered[1:]):
            if left in events:
                heading = _span(source, index, left, events[left])
                title_span = _quote(source, heading, min(policy.maximum_title_characters, policy.maximum_description_characters))
                native_ref = unit.native_provenance.get("item_ref")
                key = str(native_ref) if native_ref else None
                # Consecutive fragments of one declared native heading remain one section.
                existing = heading_refs.get(key) if key and unit.native_provenance.get("item_label") == "section_header" else None
                if existing is not None and existing is current:
                    if "complete_heading_title" not in current["metadata_unknown"]:
                        current["metadata_unknown"].append("complete_heading_title")
                else:
                    level = unit.native_provenance.get("section_level")
                    _require(level is None or (type(level) is int and level >= 1), "invalid_declared_heading_level")
                    current = _node(root_id, "section", heading, root_id,
                        title=_literal(source, title_span) if title_span else None,
                        title_origin="source_literal" if title_span else "unknown", title_span=title_span,
                        metadata={"membership_basis": "source_order_hint", "declared_level": level,
                                  "declared_parent_ref": unit.native_provenance.get("parent_ref"), "native_item_ref": native_ref},
                        unknown=("heading_hierarchy", "semantic_scope"))
                    if title_span and title_span["span_end"] < heading["span_end"]:
                        current["metadata_unknown"].append("complete_heading_title")
                    nodes.append(current)
                    sections.append(current)
                    root["child_ids"].append(current["id"])
                    if key:
                        heading_refs[key] = current
            for start in range(left, right, policy.maximum_leaf_characters):
                end = min(right, start + policy.maximum_leaf_characters)
                if current is None:
                    current = _node(root_id, "section", {"unit_index": index, "span_start": start}, root_id,
                        metadata={"membership_basis": "unknown"}, unknown=("section_title", "heading_hierarchy", "semantic_scope"))
                    nodes.append(current)
                    sections.append(current)
                    root["child_ids"].append(current["id"])
                span = _span(source, index, start, end)
                leaf = _node(root_id, "leaf", span, current["id"],
                    metadata={"membership_basis": "source_order_hint", "item_label": unit.native_provenance.get("item_label"),
                              "declared_parent_ref": unit.native_provenance.get("parent_ref")},
                    unknown=("section_membership", "semantic_scope"))
                _describe(leaf, source, span, policy)
                nodes.append(leaf)
                leaves.append(leaf)
                _require(len(nodes) <= policy.maximum_nodes, "structure_node_limit")
    # A declared parent is stronger than positional grouping, even when stored out of order.
    by_id = {node["id"]: node for node in nodes}
    for leaf in leaves:
        parent_ref = leaf["structure_metadata"]["declared_parent_ref"]
        declared = heading_refs.get(str(parent_ref)) if parent_ref else None
        if declared is not None:
            leaf["parent_id"] = declared["id"]
            leaf["structure_metadata"]["membership_basis"] = "declared_native_parent"
            leaf["metadata_unknown"].remove("section_membership")
        by_id[leaf["parent_id"]]["child_ids"].append(leaf["id"])
    for section in sections:
        candidates = [by_id[key]["source_span"] for key in section["child_ids"]
                      if by_id[key]["source_span"] != section["anchor"]]
        _describe(section, source, next((span for span in candidates if _literal(source, span).strip()), None), policy)
    _describe(root, source, next((leaf["source_span"] for leaf in leaves
                                if _literal(source, leaf["source_span"]).strip()), None), policy)
    value = {"schema_version": VERSION, "source_binding": binding, "policy": asdict(policy),
        "policy_sha256": _digest(asdict(policy)), "root_id": root_id, "units": units, "nodes": nodes,
        "coverage": {"available_characters": sum(unit["available_end"] - unit["available_start"] for unit in units),
                     "leaf_characters": sum(leaf["source_span"]["span_end"] - leaf["source_span"]["span_start"] for leaf in leaves),
                     "available_text_lossless": True, "complete_original_document_claimed": False,
                     "source_semantics_verified": False},
        "metadata_unknown": ["semantic_topology", "intrinsic_document_reading_order"]}
    raw = _canonical(value)
    _require(len(raw) <= policy.maximum_sidecar_bytes, "structure_sidecar_byte_limit")
    return SourceStructure(value, _sha(raw))


def _validate_structure(structure: SourceStructure, source: StructureSource, policy: StructurePolicy) -> None:
    binding, units = _source_binding(source, policy)
    value = structure._payload
    _require(_digest(value) == structure.sha256 and value.get("schema_version") == VERSION,
             "structure_manifest_integrity_mismatch")
    _require(value.get("source_binding") == binding and value.get("units") == units,
             "structure_source_binding_changed")
    _require(value.get("policy") == asdict(policy) and value.get("policy_sha256") == _digest(asdict(policy)),
             "structure_policy_changed")
    nodes = value.get("nodes")
    _require(type(nodes) is list and 3 <= len(nodes) <= policy.maximum_nodes, "structure_node_limit")
    by_id = {node["id"]: node for node in nodes}
    _require(len(by_id) == len(nodes) and value["root_id"] in by_id, "structure_node_identity_invalid")
    root_id = _digest({"schema_version": VERSION, "source_binding": binding, "policy": asdict(policy)})
    _require(value["root_id"] == root_id, "structure_root_binding_changed")
    intervals: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(units))}
    for node in nodes:
        kind, parent = node["kind"], node["parent_id"]
        _require(kind in {"document", "section", "leaf"}, "structure_node_kind_invalid")
        _require(type(node["metadata_unknown"]) is list and "semantic_scope" in node["metadata_unknown"],
                 "structure_semantic_scope_not_verified")
        _require(node["id"] == (root_id if kind == "document" else _digest({"root": root_id, "kind": kind, "anchor": node["anchor"]})),
                 "structure_node_identity_invalid")
        _require((kind == "document" and node["id"] == root_id and parent is None)
                 or (kind == "section" and parent == root_id)
                 or (kind == "leaf" and parent in by_id and by_id[parent]["kind"] == "section"), "structure_parent_invalid")
        _require(len(set(node["child_ids"])) == len(node["child_ids"])
                 and all(child in by_id and by_id[child]["parent_id"] == node["id"] for child in node["child_ids"]),
                 "structure_children_invalid")
        if parent:
            _require(node["id"] in by_id[parent]["child_ids"], "structure_child_unreachable")
        quotes = [_literal(source, span) for span in node["description_spans"]]
        parts = ([node["title"]] if node["title_origin"] == "source_metadata" and node["title"] else []) + quotes
        _require(node["description"] == "\n\n".join(parts)
                 and len(node["description"]) <= policy.maximum_description_characters, "structure_description_not_literal")
        if node["title_origin"] == "source_metadata":
            _require(kind == "document" and node["title"] == (source.title or "")[:policy.maximum_title_characters],
                     "structure_metadata_title_changed")
        elif node["title_origin"] == "source_literal":
            _require(node["title_span"] is not None and node["title"] == _literal(source, node["title_span"]),
                     "structure_title_not_literal")
        if kind == "section":
            anchor, metadata = node["anchor"], node["structure_metadata"]
            _require("heading_hierarchy" in node["metadata_unknown"], "structure_heading_hierarchy_not_verified")
            if "source_sha256" in anchor:
                _literal(source, anchor)
                unit = source.units[anchor["unit_index"]]
                lower, upper = _bounds(unit)
                declared_header = unit.native_provenance.get("item_label") == "section_header"
                header_range = (anchor["span_start"], anchor["span_end"])
                _require(header_range in [tuple(span) for span in unit.annotations.get("sectionheader", [])]
                         or (declared_header and header_range == (lower, upper)), "structure_heading_not_in_source")
                expected = {"membership_basis": "source_order_hint",
                    "declared_level": unit.native_provenance.get("section_level"),
                    "declared_parent_ref": unit.native_provenance.get("parent_ref"),
                    "native_item_ref": unit.native_provenance.get("item_ref")}
                _require(metadata == expected, "structure_declared_metadata_changed")
            else:
                _require(metadata == {"membership_basis": "unknown"} and node["title"] is None
                         and "section_title" in node["metadata_unknown"], "structure_unknown_heading_changed")
        if kind == "leaf":
            span = node["source_span"]
            _literal(source, span)
            _require(span == node["anchor"] and not node["child_ids"]
                     and span["span_end"] - span["span_start"] <= policy.maximum_leaf_characters, "structure_leaf_invalid")
            unit = source.units[span["unit_index"]]
            parent_metadata = by_id[parent]["structure_metadata"]
            parent_ref = unit.native_provenance.get("parent_ref")
            declared = bool(parent_ref and parent_metadata.get("native_item_ref") is not None
                            and str(parent_ref) == str(parent_metadata["native_item_ref"]))
            _require(node["structure_metadata"] == {"membership_basis": "declared_native_parent" if declared else "source_order_hint",
                "item_label": unit.native_provenance.get("item_label"), "declared_parent_ref": parent_ref}
                and (declared or "section_membership" in node["metadata_unknown"]), "structure_declared_metadata_changed")
            intervals[span["unit_index"]].append((span["span_start"], span["span_end"]))
    for index, spans in intervals.items():
        cursor = units[index]["available_start"]
        for start, end in sorted(spans):
            _require(start == cursor, "structure_leaf_coverage_gap_or_overlap")
            cursor = end
        _require(cursor == units[index]["available_end"], "structure_leaf_coverage_incomplete")
    available = sum(unit["available_end"] - unit["available_start"] for unit in units)
    _require(value.get("coverage") == {"available_characters": available, "leaf_characters": available,
        "available_text_lossless": True, "complete_original_document_claimed": False, "source_semantics_verified": False},
        "structure_coverage_claim_changed")
    _require(value.get("metadata_unknown") == ["semantic_topology", "intrinsic_document_reading_order"],
             "structure_unknown_metadata_changed")


def store_structure(store: CAS, source: StructureSource, policy: StructurePolicy | None = None) -> StoredStructure:
    policy = policy or StructurePolicy()
    structure = build_structure(source, policy)
    pointer = store.put(source.security_domain, io.BytesIO(_canonical(structure.to_dict())), policy.maximum_sidecar_bytes)
    _require(pointer.sha256 == structure.sha256, "structure_store_identity_mismatch")
    return StoredStructure(pointer, structure)


def load_structure(store: CAS, pointer: StoredBlob, source: StructureSource,
                   policy: StructurePolicy | None = None) -> SourceStructure:
    policy = policy or StructurePolicy()
    policy.validate()
    _require(pointer.security_domain == source.security_domain and 0 < pointer.size_bytes <= policy.maximum_sidecar_bytes,
             "structure_pointer_scope_invalid")
    raw = store.read(pointer)
    value = json.loads(raw)
    _require(_canonical(value) == raw, "structure_sidecar_not_canonical")
    structure = SourceStructure(value, pointer.sha256)
    _validate_structure(structure, source, policy)
    return structure


def read_leaf(structure: SourceStructure, node_id: str, source: StructureSource) -> dict:
    policy = StructurePolicy(**structure._payload["policy"])
    _validate_structure(structure, source, policy)
    node = structure.node(node_id)
    _require(node["kind"] == "leaf", "structure_leaf_required")
    span = node["source_span"]
    unit = source.units[span["unit_index"]]
    literal = _literal(source, span)
    locator = copy.deepcopy(unit.locator)
    locator_start, locator_end = locator.get("char_start"), locator.get("char_end")
    if (type(locator_start) is int and type(locator_end) is int and locator_end - locator_start == len(unit.text)):
        locator["char_start"], locator["char_end"] = locator_start + span["span_start"], locator_start + span["span_end"]
        if type(locator.get("line_1based")) is int:
            locator["line_1based"] += len(re.findall(r"\r\n|\r|\n", unit.text[:span["span_start"]]))
    return {"node_id": node_id, "structure_sha256": structure.sha256, "unit_index": span["unit_index"],
        "reference": copy.deepcopy(unit.reference), "span_start": span["span_start"], "span_end": span["span_end"],
        "text": literal, "source_sha256": span["source_sha256"], "unit_text_sha256": unit.source_text_sha256,
        "coordinate_system": unit.coordinate_system, "locator": locator,
        "native_provenance": copy.deepcopy(unit.native_provenance), "representation_id": source.representation_id,
        "source_version_sha256": source.source_version_sha256, "evidence_issued": False}
