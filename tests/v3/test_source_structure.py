"""Exact source coverage, immutable binding and derived CAS safety checks."""
import copy
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from citequest.research_v3.source_structure import (
    POLICY_VERSION, SourceStructure, StructurePolicy, StructureSource, StructureUnit,
    build_structure, load_structure, read_leaf, store_structure,
)
from citequest.v2.storage import CAS


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def unit(text, **changes):
    return StructureUnit(reference={"kind": "native_representation", "id": "a" * 64}, text=text,
        source_text_sha256=sha(text), coordinate_system="unicode_codepoints_in_native_literal", **changes)


def source(*units, **changes):
    return StructureSource(selection={"kind": "native_representation", "id": "a" * 64},
        source_version_sha256="b" * 64, representation_id="a" * 64, security_domain="public",
        title="Recorded metadata title", units=units, **changes)


def literal_fixture():
    text = "Opening context.\nFirst heading\nThe complete original method is here.\nSecond heading\nAn opposing condition is in this section.\n"
    heading1, heading2 = text.index("First heading"), text.index("Second heading")
    paragraph1, paragraph2 = text.index("The complete"), text.index("An opposing")
    annotations = {"sectionheader": [(heading1, heading1 + len("First heading")),
                                      (heading2, heading2 + len("Second heading"))],
                   "paragraph": [(0, len("Opening context.")), (paragraph1, heading2 - 1),
                                 (paragraph2, len(text) - 1)]}
    return source(unit(text, annotations=annotations))


class SourceStructureTests(unittest.TestCase):
    def test_native_structure_covers_all_available_text_outside_any_frontier(self):
        original = literal_fixture()
        structure = build_structure(original)
        leaves = [node for node in structure.nodes if node["kind"] == "leaf"]
        values = [read_leaf(structure, node["id"], original) for node in leaves]
        values.sort(key=lambda value: value["span_start"])
        self.assertEqual("".join(value["text"] for value in values), original.units[0].text)
        self.assertTrue(any("opposing condition" in value["text"] for value in values))
        self.assertTrue(all(not value["evidence_issued"] for value in values))
        self.assertEqual(structure.to_dict()["coverage"]["available_characters"], len(original.units[0].text))
        self.assertFalse(structure.to_dict()["coverage"]["complete_original_document_claimed"])

    def test_descriptions_are_bounded_exact_quotes_and_metadata_title_is_distinct(self):
        original = literal_fixture()
        structure = build_structure(original)
        root = structure.node(structure.root_id)
        self.assertEqual(root["title_origin"], "source_metadata")
        self.assertEqual(root["description_origin"], "source_metadata_and_literal_quotes")
        for node in structure.nodes:
            self.assertLessEqual(len(node["description"]), 240)
            for span in node["description_spans"]:
                quote = original.units[span["unit_index"]].text[span["span_start"]:span["span_end"]]
                self.assertEqual(sha(quote), span["source_sha256"])
                self.assertIn(quote, node["description"])

    def test_native_heading_position_does_not_invent_hierarchy_or_semantics(self):
        structure = build_structure(literal_fixture())
        sections = [node for node in structure.nodes if node["kind"] == "section" and node["title"]]
        self.assertEqual([node["title"] for node in sections], ["First heading", "Second heading"])
        for node in sections:
            self.assertIsNone(node["structure_metadata"]["declared_level"])
            self.assertEqual(node["structure_metadata"]["membership_basis"], "source_order_hint")
            self.assertIn("heading_hierarchy", node["metadata_unknown"])
            self.assertIn("semantic_scope", node["metadata_unknown"])

    def test_missing_structure_keeps_explicit_unknowns_and_literal_fallback(self):
        original = source(unit("Unstructured source " * 30))
        structure = build_structure(original)
        section = next(node for node in structure.nodes if node["kind"] == "section")
        self.assertIsNone(section["title"])
        self.assertIn("section_title", section["metadata_unknown"])
        self.assertEqual(section["structure_metadata"]["membership_basis"], "unknown")

    def test_declared_material_parent_wins_over_positional_hint(self):
        units = (
            unit("Actual heading", native_provenance={"item_label": "section_header", "item_ref": "#/texts/0", "section_level": 2}),
            unit("Different heading", native_provenance={"item_label": "section_header", "item_ref": "#/texts/1", "section_level": 1}),
            unit("Actual child of the earlier heading", native_provenance={"item_label": "text", "parent_ref": "#/texts/0"}),
        )
        original = source(*units)
        structure = build_structure(original)
        earlier = next(node for node in structure.nodes if node["kind"] == "section" and node["title"] == "Actual heading")
        child = next(node for node in structure.nodes if node["kind"] == "leaf" and node["source_span"]["unit_index"] == 2)
        self.assertEqual(child["parent_id"], earlier["id"])
        self.assertEqual(child["structure_metadata"]["membership_basis"], "declared_native_parent")
        self.assertNotIn("section_membership", child["metadata_unknown"])
        self.assertEqual(earlier["structure_metadata"]["declared_level"], 2)

    def test_consecutive_native_heading_fragments_do_not_make_fake_sections(self):
        provenance = {"item_label": "section_header", "item_ref": "#/texts/0", "section_level": 1}
        structure = build_structure(source(unit("Long heading first part", native_provenance=provenance),
            unit("heading continuation", native_provenance=provenance), unit("Body text")))
        sections = [node for node in structure.nodes if node["kind"] == "section"]
        self.assertEqual(len(sections), 1)
        self.assertIn("complete_heading_title", sections[0]["metadata_unknown"])

    def test_overlap_exclusion_preserves_unicode_and_adjusts_original_locator(self):
        text = "重复\r\nαβ conditions"
        original = source(unit(text, start=4, locator={"kind": "text", "char_start": 100,
            "char_end": 100 + len(text), "line_1based": 3}))
        structure = build_structure(original)
        leaf = next(node for node in structure.nodes if node["kind"] == "leaf")
        value = read_leaf(structure, leaf["id"], original)
        self.assertEqual(value["text"], text[4:])
        self.assertEqual((value["span_start"], value["span_end"]), (4, len(text)))
        self.assertEqual(value["locator"]["char_start"], 104)
        self.assertEqual(value["locator"]["line_1based"], 4)
        self.assertEqual(value["coordinate_system"], original.units[0].coordinate_system)
        self.assertEqual(value["reference"], original.units[0].reference)
        self.assertEqual(original.units[0].locator["char_start"], 100)

    def test_character_policy_splits_losslessly_without_normalizing(self):
        original = source(unit(("甲😀\r\nβ  " * 70)))
        policy = StructurePolicy(maximum_leaf_characters=64)
        structure = build_structure(original, policy)
        leaves = [node for node in structure.nodes if node["kind"] == "leaf"]
        self.assertTrue(all(node["source_span"]["span_end"] - node["source_span"]["span_start"] <= 64 for node in leaves))
        self.assertEqual("".join(read_leaf(structure, node["id"], original)["text"] for node in leaves), original.units[0].text)
        self.assertNotEqual(structure.sha256, build_structure(original).sha256)

    def test_source_hash_and_invalid_annotations_are_refused(self):
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            build_structure(source(replace(unit("Actual source"), source_text_sha256="0" * 64)))
        for span in [(-1, 4), (0, 999), (True, 4), (4, 4)]:
            with self.subTest(span=span), self.assertRaisesRegex(ValueError, "annotation_range_invalid"):
                build_structure(source(unit("Actual source", annotations={"paragraph": [span]})))
        with self.assertRaisesRegex(ValueError, "overlapping_structure_headings"):
            build_structure(source(unit("Actual source", annotations={"sectionheader": [(0, 6), (3, 8)]})))

    def test_cas_is_deterministic_domain_bound_and_has_no_full_literal_text_field(self):
        original = literal_fixture()
        with tempfile.TemporaryDirectory() as directory:
            store = CAS(Path(directory))
            first = store_structure(store, original)
            second = store_structure(store, original)
            self.assertEqual(first.pointer, second.pointer)
            self.assertEqual(first.pointer.sha256, first.structure.sha256)
            restored = load_structure(store, first.pointer, original)
            self.assertEqual(restored.to_dict(), first.structure.to_dict())
            serialized = store.read(first.pointer).decode()
            self.assertNotIn('"literal_text"', serialized)
            self.assertNotIn('"text"', serialized)
            with self.assertRaisesRegex(ValueError, "pointer_scope_invalid"):
                load_structure(store, replace(first.pointer, security_domain="owner:other"), original)

    def test_cached_structure_does_not_allow_changed_source_version_or_metadata(self):
        original = literal_fixture()
        with tempfile.TemporaryDirectory() as directory:
            store = CAS(Path(directory))
            stored = store_structure(store, original)
            for changed in (replace(original, source_version_sha256="c" * 64), replace(original, title="Changed title"),
                            replace(original, units=(replace(original.units[0], annotations={}),))):
                with self.subTest(changed=changed.source_version_sha256), self.assertRaisesRegex(ValueError, "source_binding_changed"):
                    load_structure(store, stored.pointer, changed)
            with self.assertRaisesRegex(ValueError, "policy_changed|source_binding_changed|root_binding_changed"):
                load_structure(store, stored.pointer, original, StructurePolicy(maximum_leaf_characters=64))

    def test_stored_fabricated_description_is_not_accepted_as_source_quote(self):
        original = literal_fixture()
        structure = build_structure(original)
        value = structure.to_dict()
        value["nodes"][-1]["description"] = "Invented routing summary"
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as directory:
            store = CAS(Path(directory))
            pointer = store.put("public", io.BytesIO(raw), 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "description_not_literal"):
                load_structure(store, pointer, original)

    def test_rehashed_sidecar_cannot_invent_heading_metadata_or_complete_source_claim(self):
        original = literal_fixture()
        for change in ("level", "coverage", "unknown"):
            value = build_structure(original).to_dict()
            section = next(node for node in value["nodes"] if node["kind"] == "section" and node["title"])
            if change == "level":
                section["structure_metadata"]["declared_level"] = 1
            elif change == "coverage":
                value["coverage"]["complete_original_document_claimed"] = True
            else:
                section["metadata_unknown"].remove("heading_hierarchy")
            raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            with tempfile.TemporaryDirectory() as directory:
                store = CAS(Path(directory))
                pointer = store.put("public", io.BytesIO(raw), 1024 * 1024)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_structure(store, pointer, original)

    def test_gap_cannot_be_hidden_by_rehashing_sidecar(self):
        original = literal_fixture()
        value = build_structure(original).to_dict()
        removed = next(node for node in value["nodes"] if node["kind"] == "leaf")
        value["nodes"] = [node for node in value["nodes"] if node["id"] != removed["id"]]
        parent = next(node for node in value["nodes"] if node["id"] == removed["parent_id"])
        parent["child_ids"].remove(removed["id"])
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as directory:
            store = CAS(Path(directory))
            pointer = store.put("public", io.BytesIO(raw), 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "coverage_gap_or_overlap|coverage_incomplete"):
                load_structure(store, pointer, original)

    def test_copy_out_does_not_mutate_structure_and_nonleaf_cannot_return_text(self):
        original = literal_fixture()
        structure = build_structure(original)
        copied = structure.to_dict()
        copied["nodes"][0]["description"] = "Changed copy"
        self.assertNotEqual(copied["nodes"][0]["description"], structure.node(structure.root_id)["description"])
        with self.assertRaisesRegex(ValueError, "structure_leaf_required"):
            read_leaf(structure, structure.root_id, original)
        damaged = copy.deepcopy(structure._payload)
        damaged["nodes"][0]["description"] = "Changed internal data"
        with self.assertRaisesRegex(ValueError, "manifest_integrity_mismatch"):
            read_leaf(SourceStructure(damaged, structure.sha256), structure.root_id, original)

    def test_limits_are_explicit_and_missing_source_ranges_remain_visible(self):
        for policy in (StructurePolicy(version="other"), StructurePolicy(maximum_leaf_characters=0),
                       StructurePolicy(maximum_description_characters=241)):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                build_structure(literal_fixture(), policy)
        with self.assertRaisesRegex(ValueError, "structure_node_limit"):
            build_structure(source(unit("X" * 200)), StructurePolicy(maximum_leaf_characters=64, maximum_nodes=3))
        original = source(unit("Available source"), coverage="material_partial", missing_ranges=({"page": 2, "reason": "no_text"},))
        result = build_structure(original).to_dict()
        self.assertEqual(result["source_binding"]["coverage"], "material_partial")
        self.assertEqual(result["source_binding"]["missing_ranges"], [{"page": 2, "reason": "no_text"}])
        self.assertEqual(result["policy"]["version"], POLICY_VERSION)
        self.assertFalse(result["coverage"]["complete_original_document_claimed"])


if __name__ == "__main__":
    unittest.main()
