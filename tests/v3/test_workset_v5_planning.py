"""No network/PG/model: lossless grouping, exact raw packs and resource bounds."""
import copy
import unittest

from citequest.agents_v3.contracts import AgentError, digest
from citequest.research_v3.quote_spans import QuoteProtocolError, QuoteRegistry
from citequest.research_v3.workset_v5_planning import group_windows, raw_quote_pack, resource_plan


def source(ref, document, text):
    return {"ref_id": ref, "document_id": document, "text": text, "span_start": 100, "span_end": 100 + len(text),
        "source_ref": {"kind": "fixture", "id": ref}, "source_version_sha256": "a" * 64, "source_sha256": digest(text.encode()),
        "unit_text_sha256": digest(text.encode()), "coverage_kind": "native_literal", "coordinate_system": "unicode_codepoint"}


def edge_for(registry, ref, start, end):
    scope = registry.scope(list(registry.sources.values()))
    # Candidates use exact coordinates and all canonical version pins. Build the
    # locator through the same public codec, then test a deliberately later
    # occurrence rather than searching duplicate text.
    from citequest.research_v3.quote_spans import Unit
    item = registry.sources[ref]
    unit = Unit(ref, start, end, item["text"][start - item["span_start"]:end - item["span_start"]])
    return {"ref_id": ref, "quote": unit.text, "quote_locator": scope.locator(unit), "relation": "supports", "rationale": "fixture", "conditions": []}


class PlanningTests(unittest.TestCase):
    def test_complete_grouping_splits_by_actual_encoder_without_text_loss(self):
        originals = {str(i): {"evidence": [{"text": "x" * 10, "ref_id": "r" + str(i)}]} for i in range(9)}
        seen = []
        def encoder(values):
            seen.append(len(values))
            return 7 + sum(len(value["evidence"][0]["text"]) for value in values)
        groups = group_windows(list(originals), originals, encoder, maximum=38)
        self.assertEqual([len(group.window_ids) for group in groups], [3, 3, 3])
        self.assertEqual([key for group in groups for key in group.window_ids], list(originals))
        self.assertIn(4, seen)
        self.assertTrue(all(group.wire_bytes == 37 for group in groups))

    def test_encoder_cannot_mutate_original_source_buffer(self):
        originals = {"a": {"evidence": [{"text": "source"}]}}
        before = copy.deepcopy(originals)
        def encoder(values):
            values[0]["evidence"][0]["text"] = "attacker"
            return 10
        group_windows(["a"], originals, encoder)
        self.assertEqual(originals, before)

    def test_oversized_single_window_is_explicit_and_not_clipped(self):
        value = {"a": {"evidence": [{"text": "source"}]}}
        with self.assertRaisesRegex(AgentError, "single_window_exceeds_request_capacity"):
            group_windows(["a"], value, lambda _: 131073)
        self.assertEqual(value["a"]["evidence"][0]["text"], "source")

    def test_missing_or_duplicate_window_manifest_refused(self):
        for ids, windows in ((["a"], {}), (["a", "a"], {"a": {"evidence": [1]}})):
            with self.subTest(ids=ids), self.assertRaises(AgentError):
                group_windows(ids, windows, lambda _: 1)

    def test_raw_pack_uses_exact_later_occurrence_and_original_identity(self):
        first = source("r1", "D01", "repeat "+"x" * 800+" repeat "+"y" * 800)
        second = source("r2", "D02", "z" * 800+"\ue123 counterexample "+"q" * 800)
        registry = QuoteRegistry([first, second])
        edges = [edge_for(registry, "r1", 907, 913), edge_for(registry, "r2", 900, 916)]
        pack, parents = raw_quote_pack(registry, edges)
        self.assertEqual(parents, ["r1", "r2"])
        self.assertEqual(pack.raw_groups[0]["start"], 747)
        self.assertEqual(pack.raw_groups[0]["end"], 1073)
        self.assertEqual(registry.sources["r1"], first)
        self.assertTrue(any("\ue123" in group["context"]["text"] for group in pack.raw_groups))
        packed = pack.compact_candidate({"claims": [{"id": "C1", "edges": edges}]})
        self.assertEqual(len(packed["claims"][0]["edges"]), 2)

    def test_forged_locator_or_changed_version_never_reselected_by_text(self):
        registry = QuoteRegistry([source("r1", "D01", "a" * 1000), source("r2", "D02", "b" * 1000)])
        edges = [edge_for(registry, "r1", 200, 220), edge_for(registry, "r2", 200, 220)]
        edges[0]["quote_locator"]["source_version_sha256"] = "b" * 64
        with self.assertRaises(QuoteProtocolError):
            raw_quote_pack(registry, edges)

    def test_summary_prose_cannot_be_passed_as_pack_evidence(self):
        registry = QuoteRegistry([source("r1", "D01", "a" * 1000), source("r2", "D02", "b" * 1000)])
        edges = [edge_for(registry, "r1", 200, 220), edge_for(registry, "r2", 200, 220)]
        edges[0]["quote"] = "A summary card says this is true"
        with self.assertRaises(QuoteProtocolError):
            raw_quote_pack(registry, edges)

    def test_comparison_must_cover_two_documents_and_no_duplicate_edge(self):
        registry = QuoteRegistry([source("r1", "D01", "a" * 1000)])
        edge = edge_for(registry, "r1", 200, 220)
        with self.assertRaisesRegex(AgentError, "distinct_documents"):
            raw_quote_pack(registry, [edge])
        with self.assertRaisesRegex(AgentError, "duplicate_evidence_locator"):
            raw_quote_pack(registry, [edge, edge])

    def test_three_and_ten_document_resources_are_explicit(self):
        small = resource_plan(documents=3, windows=9, groups=3, comparison_units=5, source_parts=9)
        self.assertEqual((small["required_models"], small["required_tools"], small["reserved_output_tokens"]), (23, 18, 59904))
        large = resource_plan(documents=10, windows=29, groups=10, comparison_units=12, source_parts=29)
        self.assertEqual((large["required_models"], large["required_tools"], large["reserved_output_tokens"]), (51, 45, 128000))
        self.assertTrue(large["capacity_admitted"])
        self.assertFalse(large["latency_or_semantic_success_guaranteed"])

    def test_actual_extra_groups_can_refuse_without_resetting_budget(self):
        value = resource_plan(documents=10, windows=50, groups=50, comparison_units=12, source_parts=50)
        self.assertFalse(value["capacity_admitted"])
        self.assertGreater(value["required_models"], 64)


if __name__ == "__main__":
    unittest.main()
