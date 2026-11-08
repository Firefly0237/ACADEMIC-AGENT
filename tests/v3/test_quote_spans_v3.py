"""Self-contained quote codec fixtures; zero model calls or archived run inputs."""

from __future__ import annotations

import copy
import unittest

from pydantic import ValidationError

from citequest.agents_v3.contracts import digest
from citequest.research_v3.quote_spans import (
    DocumentSummaryV3,
    QuoteProtocolError,
    QuoteRegistry,
    SelectorPlainReview,
    SelectorSummary,
    source_warnings,
)

def source(
    text="Alpha  β\n\nRange 0 \uf8ff score \uf8ff 1.\nUnicode😀 e\u0301 remains exact. ",
    *,
    ref="source-window:" + "a" * 64,
    start=100,
):
    return {
        "ref_id": ref,
        "evidence_type": "source_window",
        "document_id": "D01",
        "source_ref": {"kind": "native_representation", "id": "a" * 64},
        "source_version_sha256": "b" * 64,
        "text": text,
        "span_start": start,
        "span_end": start + len(text),
        "source_sha256": digest(text.encode()),
        "unit_text_sha256": digest(text.encode()),
        "coordinate_system": "unicode_codepoints_in_native_literal",
    }


def wire(identifier, *, claims=1):
    return {
        "claims": [
            {
                "id": "C" + str(i + 1),
                "statement": "Explicit fixture only.",
                "status": "supported",
                "edges": [
                    {
                        "quote_id": identifier,
                        "relation": "supports",
                        "rationale": "Fixture, not scientific evidence.",
                        "conditions": [],
                    }
                ],
                "limitations": ["Explicit fixture."],
            }
            for i in range(claims)
        ],
        "gaps": [],
        "evidence_status": "supported",
    }


class QuoteCodecControls(unittest.TestCase):
    def setUp(self):
        self.evidence = [source()]
        self.scope = QuoteRegistry(self.evidence).scope(self.evidence)

    def test_raw_source_whitespace_unicode_and_pua_reconstruct_exactly(self):
        actual = "".join(
            unit["text"] for group in self.scope.contexts for unit in group["quote_units"] if "text" in unit
        )
        self.assertEqual(actual, self.evidence[0]["text"])
        self.assertEqual(digest(actual.encode()), self.evidence[0]["source_sha256"])
        warning = source_warnings(self.evidence)[0]
        self.assertEqual(warning["count"], 2)
        self.assertEqual(warning["code_points"], ["U+F8FF"])

    def test_server_extracts_original_quote_and_versioned_locator(self):
        identifier = next(iter(sorted(self.scope.allowed)))
        answer = self.scope.expand_summary(SelectorSummary.model_validate(wire(identifier), strict=True))
        edge = answer.claims[0].edges[0]
        self.assertIn("\uf8ff", edge.quote)
        self.assertEqual(edge.quote, self.evidence[0]["text"])
        self.assertEqual(edge.quote_locator.span_start, 100)
        self.assertEqual(edge.quote_locator.quote_sha256, digest(edge.quote.encode()))
        self.assertEqual(edge.quote_locator.source_version_sha256, "b" * 64)

    def test_model_schema_has_no_quote_ref_or_locator_writable_field(self):
        schema = self.scope.schema(SelectorSummary)

        def inspect(value):
            if isinstance(value, dict):
                properties = value.get("properties", {})
                self.assertFalse({"quote", "ref_id", "quote_locator"} & set(properties))
                if "quote_id" in properties:
                    self.assertEqual(properties["quote_id"]["enum"], sorted(self.scope.allowed))
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)

        inspect(schema)

    def test_unknown_selector_and_supplied_quote_rejected(self):
        with self.assertRaises(QuoteProtocolError):
            self.scope.expand_summary(SelectorSummary.model_validate(wire("Q99999"), strict=True))
        value = wire("Q1")
        value["claims"][0]["edges"][0]["quote"] = "model invented quotation"
        with self.assertRaises(ValidationError):
            SelectorSummary.model_validate(value, strict=True)

    def test_reference_scope_and_source_revision_change_binding(self):
        changed = copy.deepcopy(self.evidence)
        changed[0]["source_version_sha256"] = "c" * 64
        other = QuoteRegistry(changed).scope(changed)
        self.assertNotEqual(other.binding["binding_sha256"], self.scope.binding["binding_sha256"])
        candidate = self.scope.expand_summary(SelectorSummary.model_validate(wire("Q1"), strict=True)).model_dump()
        with self.assertRaises(QuoteProtocolError):
            QuoteRegistry(changed).scope(changed, candidates=[candidate])

    def test_material_real_provenance_fields_change_binding_without_fake_source_version(self):
        evidence = [source()]
        evidence[0].pop("source_version_sha256")
        evidence[0].pop("evidence_type")
        evidence[0].update(
            file_id="fixture-file",
            file_revision=1,
            material_version_id="fixture-version",
            representation_id="c" * 64,
            source_file_sha256="d" * 64,
        )
        scope = QuoteRegistry(evidence).scope(evidence)
        canonical = scope.expand_summary(SelectorSummary.model_validate(wire("Q1"), strict=True)).model_dump()
        self.assertIsNone(canonical["claims"][0]["edges"][0]["quote_locator"]["source_version_sha256"])
        for field in ("file_id", "file_revision", "material_version_id", "representation_id", "source_file_sha256"):
            changed = copy.deepcopy(evidence)
            changed[0][field] = 2 if field == "file_revision" else "changed"
            with self.subTest(field=field), self.assertRaises(QuoteProtocolError):
                QuoteRegistry(changed).scope(changed, candidates=[canonical])

    def test_explicit_empty_evidence_scope_allows_only_no_edge_insufficiency(self):
        with self.assertRaises(QuoteProtocolError):
            QuoteRegistry([])
        scope = QuoteRegistry([], allow_empty=True).scope([])
        value = wire("Q1")
        with self.assertRaises(QuoteProtocolError):
            scope.expand_summary(SelectorSummary.model_validate(value, strict=True))
        value["claims"][0].update(status="insufficient", edges=[])
        value["evidence_status"] = "insufficient"
        summary = scope.expand_summary(SelectorSummary.model_validate(value, strict=True))
        self.assertEqual(summary.claims[0].edges, [])
        schema = scope.schema(SelectorSummary)
        self.assertEqual(schema["$defs"]["SelectorClaim"]["properties"]["edges"]["maxItems"], 0)

    def test_actual_nested_source_citation_and_material_coordinates_are_version_pins(self):
        evidence = [source()]
        evidence[0]["source"] = {
            "source_namespace": "explicit-fixture",
            "native_id": "fixture-native-id",
            "source_revision": "fixture-revision-1",
            "projection_sha256": "1" * 64,
            "source_pointer_sha256": "2" * 64,
        }
        evidence[0]["locator"] = {"page": 1, "paragraph": 4}
        evidence[0]["missing_ranges"] = ["page 3: unavailable"]
        scope = QuoteRegistry(evidence).scope(evidence)
        candidate = scope.expand_summary(SelectorSummary.model_validate(wire("Q1"), strict=True)).model_dump()
        for field in evidence[0]["source"]:
            changed = copy.deepcopy(evidence)
            changed[0]["source"][field] = "changed"
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(QuoteProtocolError, "quote_candidate_locator_pin_changed"),
            ):
                QuoteRegistry(changed).scope(changed, candidates=[candidate])
        for field, replacement in (("locator", {"page": 2, "paragraph": 4}), ("missing_ranges", [])):
            changed = copy.deepcopy(evidence)
            changed[0][field] = replacement
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(QuoteProtocolError, "quote_candidate_locator_pin_changed"),
            ):
                QuoteRegistry(changed).scope(changed, candidates=[candidate])

    def test_forged_source_span_or_hash_rejected(self):
        for field, value in (
            ("source_sha256", "0" * 64),
            ("span_start", 99),
            ("span_end", 9999),
            ("unit_text_sha256", "bad"),
        ):
            changed = copy.deepcopy(self.evidence)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(QuoteProtocolError):
                QuoteRegistry(changed).scope(changed)

    def test_legacy_quote_must_be_literal_no_fuzzy_or_pua_deletion(self):
        candidate = {
            "claims": [{"id": "C1", "edges": [{"ref_id": self.evidence[0]["ref_id"], "quote": "Range 0  score  1."}]}]
        }
        with self.assertRaises(QuoteProtocolError):
            QuoteRegistry(self.evidence).scope(self.evidence, candidates=[candidate])

    def test_legacy_duplicate_without_locator_is_ambiguous(self):
        evidence = [source("same sentence. other. same sentence.")]
        candidate = {"claims": [{"id": "C1", "edges": [{"ref_id": evidence[0]["ref_id"], "quote": "same sentence."}]}]}
        with self.assertRaisesRegex(QuoteProtocolError, "quote_candidate_not_exact_unique_literal"):
            QuoteRegistry(evidence).scope(evidence, candidates=[candidate])

    def test_overlap_merges_without_gaps_and_without_quote_text_duplication(self):
        evidence = [source("".join(str(i % 10) for i in range(1500)))]
        original = evidence[0]

        def context(start, end):
            return {
                **original,
                "text": original["text"][start:end],
                "context_span_start": 100 + start,
                "context_span_end": 100 + end,
            }

        contexts = [context(0, 700), context(500, 1000), context(1200, 1500)]
        # Repeated digits are deliberately ambiguous without a locator; source spans
        # are tested independently by the interval union before candidate conversion.
        scope = QuoteRegistry(evidence).scope(contexts)
        groups = ["".join(unit["text"] for unit in group["quote_units"] if "text" in unit) for group in scope.contexts]
        self.assertEqual(groups, [original["text"][:1000], original["text"][1200:1500]])
        self.assertEqual(sum(map(len, groups)), 1300)

    def test_unique_legacy_quote_selector_preserves_context_once(self):
        evidence = [source("left-prefix UNIQUE-ONE middle UNIQUE-TWO suffix-right")]
        candidates = [
            {"claims": [{"id": "C1", "edges": [{"ref_id": evidence[0]["ref_id"], "quote": "UNIQUE-ONE middle"}]}]}
        ]
        scope = QuoteRegistry(evidence).scope(evidence, candidates=candidates)
        compact = scope.compact_candidate(candidates[0])
        self.assertIn("quote_id", compact["claims"][0]["edges"][0])
        text = "".join(unit["text"] for group in scope.contexts for unit in group["quote_units"] if "text" in unit)
        self.assertEqual(text, evidence[0]["text"])

    def test_more_than_three_atomic_claims_is_versioned_not_legacy_mutation(self):
        answer = self.scope.expand_summary(SelectorSummary.model_validate(wire("Q1", claims=8), strict=True))
        self.assertEqual(len(answer.claims), 8)
        self.assertIsInstance(answer, DocumentSummaryV3)
        from citequest.research_v3.workset_contracts import DocumentSummary

        with self.assertRaises(ValidationError):
            DocumentSummary.model_validate(answer.model_dump(), strict=True)

    def test_independent_review_must_cover_every_span_not_only_same_parent_ref(self):
        from citequest.research_v3.contracts import Review

        evidence = [source("A" * 900 + "B" * 700)]
        initial = QuoteRegistry(evidence).scope(evidence)
        value = wire("Q1")
        value["claims"][0]["edges"].append({**value["claims"][0]["edges"][0], "quote_id": "Q2"})
        canonical = initial.expand_summary(SelectorSummary.model_validate(value, strict=True)).model_dump()
        scope = QuoteRegistry(evidence).scope(evidence, candidates=[canonical])
        candidate = scope.compact_candidate(canonical, track_review=True)
        ids = [item["quote_id"] for item in candidate["claims"][0]["edges"]]
        review = {
            "claims": [
                {
                    "claim_id": "C1",
                    "verdict": "entailed",
                    "edge_ref_ids": ids,
                    "numeric_and_conditions_preserved": True,
                    "explanation": "Fixture only.",
                }
            ],
            "source_instruction_injection_detected": False,
            "requested_stance_bias_detected": False,
        }
        restored = scope.expand_review(SelectorPlainReview.model_validate(review, strict=True), Review)
        self.assertEqual(restored.claims[0].edge_ref_ids, [evidence[0]["ref_id"]] * 2)
        missing = copy.deepcopy(review)
        missing["claims"][0]["edge_ref_ids"] = ids[:1]
        with self.assertRaisesRegex(QuoteProtocolError, "quote_review_span_coverage_mismatch"):
            scope.expand_review(SelectorPlainReview.model_validate(missing, strict=True), Review)
        duplicate = {**review, "claims": review["claims"] * 2}
        with self.assertRaisesRegex(QuoteProtocolError, "quote_review_claim_coverage_mismatch"):
            scope.expand_review(SelectorPlainReview.model_validate(duplicate, strict=True), Review)

    def test_metadata_scope_flags_survive_transport_and_change_version_binding(self):
        evidence = [source()]
        evidence[0]["coverage_kind"] = "metadata_only"
        scope = QuoteRegistry(evidence).scope(evidence)
        self.assertEqual(scope.contexts[0]["coverage_kind"], "metadata_only")
        self.assertTrue(
            all(unit["coverage_kind"] == "metadata_only" for group in scope.contexts for unit in group["quote_units"])
        )
        canonical = scope.expand_summary(SelectorSummary.model_validate(wire("Q1"), strict=True)).model_dump()
        evidence[0]["coverage_kind"] = "native_literal"
        with self.assertRaisesRegex(QuoteProtocolError, "quote_candidate_locator_pin_changed"):
            QuoteRegistry(evidence).scope(evidence, candidates=[canonical])

if __name__ == "__main__":
    unittest.main()
