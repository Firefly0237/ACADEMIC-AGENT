"""Mechanism and authority checks using explicitly labelled fixture sources.

The judge is a deterministic test double. These checks make no model-quality,
deployment-resource or database-integration claim.
"""
from __future__ import annotations

import copy
import hashlib
import unittest
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from citequest.agents_v3.contracts import AgentError
from citequest.research_v3.decisionrag import DecisionReceipt
from citequest.research_v3.source_navigation import SourceNavigator, validate_structure_receipt
from citequest.research_v3.source_navigation_sources import NavigationSources, scope_evidence
from citequest.research_v3.source_structure import StructureSource, StructureUnit, build_structure
from citequest.research_v3.source_windows import Document, FullSourceWindows, Unit
from citequest.research_v3.workset_contracts import SourceSelection
from citequest.v2.domain import Actor, DomainError


OWNER = UUID("11111111-1111-4111-8111-111111111111")
TASK = UUID("22222222-2222-4222-8222-222222222222")


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def configuration(**changes):
    value = dict(hybrid_keep=1, active_seconds=60, maximum_forward_calls=12,
        max_shared_bytes=32768, local_input_token_limit=100000, max_batches_per_level=4,
        shortlist_count=8, minimum_route_probability=0.5, document_keep=1, section_keep=1,
        passage_keep=1, maximum_leaf_reads=3)
    value.update(changes)
    return SimpleNamespace(**value)


def fixture_source(identifier="a", section_count=2):
    text, headers, paragraphs = "", [], []
    for index in range(section_count):
        heading = "Opening" if index == 0 else "Discovery" if index == section_count - 1 else f"Section {index}"
        start = len(text)
        text += heading
        headers.append((start, len(text)))
        text += "\n"
        contents = (["INITIAL: an existing retrieval passage."] if index == 0 else
            ["TARGET: independently labelled fixture evidence.", "NEIGHBOR: an adjacent applicability condition."]
            if index == section_count - 1 else ["UNREAD: unrelated fixture body."])
        for paragraph in contents:
            start = len(text)
            text += paragraph
            paragraphs.append((start, len(text)))
            text += "\n"
    representation = identifier * 64
    reference = {"kind": "native_representation", "id": representation}
    return StructureSource(reference, representation, representation, "public", "Fixture source title",
        (StructureUnit(reference, text, sha(text), "unicode_codepoints_in_native_literal",
            annotations={"sectionheader": headers, "paragraph": paragraphs}),), "native_literal")


class FixtureCache:
    def __init__(self, structure):
        self.structure, self.missing = structure, False

    def get_or_build(self, source):
        return SimpleNamespace(structure=self.structure)

    def get(self, source):
        return None if self.missing else SimpleNamespace(structure=self.structure)


class FixtureWindows:
    _selection = staticmethod(FullSourceWindows._selection)
    _evidence = staticmethod(FullSourceWindows._evidence)

    def __init__(self, sources):
        self.sources, self.revoked, self.changed, self.reads = sources, set(), set(), []

    def _load(self, actor, selection, document_id):
        if actor.owner_id != OWNER or selection.id in self.revoked:
            raise DomainError("not_found", 404)
        source = self.sources[selection.id]
        self.reads.append(selection.id)
        version = sha("authority:" + selection.id + (":changed" if selection.id in self.changed else ""))
        units = [Unit(unit.reference, unit.text, unit.source_text_sha256, unit.coordinate_system,
            {"representation_id": source.representation_id}, unit.start, unit.end) for unit in source.units]
        return Document(document_id, selection, source.title, version, source.coverage, [], units,
            len(units), sum(len(unit.text.encode()) for unit in units))


class FixtureSources(NavigationSources):
    def __init__(self, source_list):
        self.sources = {source.selection["id"]: source for source in source_list}
        self.structures = {identifier: build_structure(source) for identifier, source in self.sources.items()}
        self.caches = {identifier: FixtureCache(structure) for identifier, structure in self.structures.items()}
        self.windows = FixtureWindows(self.sources)
        self.expanded = []

    def _current(self, actor, selection, document_id):
        selected = self.windows._selection(selection)
        return (self.windows._load(actor, selected, document_id), self.sources[selected.id], self.caches[selected.id])

    def expand(self, actor, record, node_id):
        evidence = super().expand(actor, record, node_id)
        self.expanded.append(evidence)
        return evidence

    def labels(self):
        source = next(iter(self.sources.values()))
        structure = self.structures[source.selection["id"]]
        targets = {structure.root_id}
        for node in structure.nodes:
            if node["kind"] == "section" and node["title"] == "Discovery":
                targets.add(node["id"])
            if node["kind"] == "leaf" and "TARGET:" in node["description"]:
                targets.add(node["id"])
        return targets

    def initial(self):
        source = next(iter(self.sources.values()))
        structure = self.structures[source.selection["id"]]
        leaf = next(node for node in structure.nodes if node["kind"] == "leaf" and "INITIAL:" in node["description"])
        document = self.windows._load(Actor(OWNER), SourceSelection(**source.selection), "N01")
        span = leaf["source_span"]
        return self.windows._evidence(Actor(OWNER), document, document.units[span["unit_index"]],
            span["span_start"], span["span_end"])


class FixtureJudge:
    def __init__(self, labels, *, fault=None, after_score=None):
        self.labels, self.fault, self.after_score = labels, fault, after_score
        self.calls = []

    def preflight(self, request):
        if self.fault == "preflight_capacity":
            raise AgentError("laya_complete_context_exceeds_token_budget")
        return {identifier: {"sequence_tokens": 32} for identifier, _ in request.candidates}

    def score(self, request):
        self.calls.append(request)
        if self.after_score:
            self.after_score()
        if self.fault == "runtime":
            raise RuntimeError("fixture compute failure")
        if self.fault == "unknown":
            raise AgentError("fixture_forward_unknown")
        if self.fault == "not_started":
            error = AgentError("structure_navigation_request_capacity")
            error.forward_not_started = True
            raise error
        probabilities = {identifier: (0.9 if identifier in self.labels and self.fault != "below_threshold" else 0.1)
                         for identifier, _ in request.candidates}
        usage = {"observed_forward_calls": 1, "truncated": False, "state_tokens_dropped": 0,
                 "input_tokens": 32 * len(request.candidates), "external_API_cash_ncny": 0}
        if self.fault == "bad_usage":
            usage["input_tokens"] += 1
        return DecisionReceipt("wrong" if self.fault == "bad_identity" else request.fingerprint(), probabilities,
            {"provider": "explicit_fixture_judge", "version": "1"}, usage)


class SourceNavigationTests(unittest.TestCase):
    def fixture(self, *, section_count=2, extra_source=False):
        sources = FixtureSources([fixture_source(section_count=section_count)] + ([fixture_source("b")] if extra_source else []))
        return sources, [source.selection for source in sources.sources.values()]

    def run_fixture(self, sources, selections, *, config=None, judge=None, check_task=lambda: None):
        initial, registered = sources.initial(), []
        config = config or configuration()
        result = SourceNavigator(sources, config).run("Fixture query", Actor(OWNER), TASK, selections,
            [initial], judge or FixtureJudge(sources.labels()), lambda rows: registered.extend(rows), check_task)
        validate_structure_receipt(config, result, [initial["ref_id"]])
        return result, initial, registered

    def test_expands_full_source_leaf_outside_initial_frontier_and_preserves_hybrid(self):
        sources, selections = self.fixture()
        result, initial, registered = self.run_fixture(sources, selections)
        self.assertEqual(result["status"], "ok")
        self.assertIn(initial["ref_id"], result["evidence_ref_ids"])
        targets = [row for row in result["evidence"] if "TARGET:" in row["text"]]
        self.assertEqual(len(targets), 1)
        self.assertNotEqual(targets[0]["ref_id"], initial["ref_id"])
        self.assertIn(targets[0]["ref_id"], result["structure"]["supplemental_evidence_ref_ids"])
        self.assertTrue(any(row["evidence_type"] == "source_structure_scope" for row in registered))
        self.assertTrue(all(row["ref_id"] in {value["ref_id"] for value in registered} for row in targets))

    def test_adjacent_expansion_uses_informative_ranges_without_spending_reads_on_blank_text(self):
        sources, selections = self.fixture()
        result, _, _ = self.run_fixture(sources, selections)
        self.assertTrue(all(row["text"].strip() for row in sources.expanded))
        self.assertTrue(any("NEIGHBOR:" in row["text"] for row in result["evidence"]))

    def test_read_states_remain_partial_for_nonexhaustive_bounded_expansion(self):
        sources, selections = self.fixture(section_count=4)
        result, _, _ = self.run_fixture(sources, selections, config=configuration(maximum_leaf_reads=1))
        document = next(node for node in result["structure"]["nodes"] if node["kind"] == "document")
        self.assertEqual(document["read_state"], "partially_read")
        self.assertNotIn("fully_read", {node["read_state"] for node in result["structure"]["nodes"]})
        self.assertNotEqual(result["structure"]["scope"]["coverage"], "complete")
        self.assertGreater(result["structure"]["scope"]["passages"], len(sources.expanded))

    def test_below_threshold_reads_no_new_raw_text_and_does_not_claim_full_coverage(self):
        sources, selections = self.fixture()
        result, initial, _ = self.run_fixture(sources, selections, judge=FixtureJudge(sources.labels(), fault="below_threshold"))
        self.assertEqual(result["evidence_ref_ids"], [initial["ref_id"]])
        self.assertEqual(sources.expanded, [])
        self.assertEqual(result["structure"]["supplemental_evidence_ref_ids"], [])
        self.assertTrue(all(node["read_state"] != "fully_read" for node in result["structure"]["nodes"]))

    def test_ranked_protocol_reads_promising_ranges_even_when_all_scores_are_below_half(self):
        class LowScores(FixtureJudge):
            def score(self, request):
                original = super().score(request)
                return replace(original, probabilities={key:value/4 for key,value in original.probabilities.items()})
        for protocol in ("structure1", "structure2"):
            sources, selections = self.fixture()
            config = configuration(navigation_protocol=protocol, minimum_route_probability=.5 if protocol=="structure1" else 0.)
            result, initial, _ = self.run_fixture(sources,selections,config=config,judge=LowScores(sources.labels()))
            if protocol=="structure1":
                self.assertEqual(result["evidence_ref_ids"],[initial["ref_id"]])
            else:
                self.assertTrue(any("TARGET:" in row["text"] for row in result["evidence"]))
                self.assertTrue(result["raw_text_requires_independent_review"])
                self.assertIs(result["absolute_probability_gate"],False)
                self.assertLessEqual(len(sources.expanded),config.maximum_leaf_reads)
                altered = copy.deepcopy(result)
                altered["absolute_probability_gate"] = True
                with self.assertRaisesRegex(AgentError,"structure_navigation_routing_contract_invalid"):
                    validate_structure_receipt(config,altered,[initial["ref_id"]])

    def test_ranked_protocol_keeps_resource_limit_and_authority_guards(self):
        sources,selections = self.fixture()
        result,initial,_ = self.run_fixture(sources,selections,
            config=configuration(navigation_protocol="structure2",minimum_route_probability=0.,local_input_token_limit=1))
        self.assertEqual(result["status"],"partial")
        self.assertEqual(result["evidence_ref_ids"],[initial["ref_id"]])
        self.assertEqual(sources.expanded,[])

    def test_forward_token_and_request_budgets_keep_fallback_explicitly_partial(self):
        for changes in ({"maximum_forward_calls": 1}, {"local_input_token_limit": 1}, {"max_shared_bytes": 8}):
            sources, selections = self.fixture()
            result, initial, _ = self.run_fixture(sources, selections, config=configuration(**changes))
            with self.subTest(changes=changes):
                self.assertEqual(result["status"], "partial")
                self.assertEqual(result["evidence_ref_ids"], [initial["ref_id"]])
                self.assertEqual(result["resource_admission"]["status"], "rejected")
                self.assertFalse(result["unknown_outcome"])
                self.assertFalse(sources.expanded)

    def test_level_and_preflight_capacity_are_not_silent_success(self):
        sources, selections = self.fixture(section_count=10)
        result, _, _ = self.run_fixture(sources, selections, config=configuration(max_batches_per_level=1))
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["error_code"], "structure_navigation_level_capacity")
        sources, selections = self.fixture()
        result, _, _ = self.run_fixture(sources, selections,
            judge=FixtureJudge(sources.labels(), fault="preflight_capacity"))
        self.assertEqual(result["status"], "partial")
        self.assertFalse(sources.expanded)

    def test_unverified_or_unknown_forward_outcome_never_returns_evidence_or_known_usage(self):
        for fault in ("unknown", "bad_identity", "bad_usage", "runtime"):
            sources, selections = self.fixture()
            result, _, _ = self.run_fixture(sources, selections, judge=FixtureJudge(sources.labels(), fault=fault))
            with self.subTest(fault=fault):
                self.assertEqual(result["status"], "error")
                self.assertTrue(result["unknown_outcome"])
                self.assertIsNone(result["decision_calls"])
                self.assertIsNone(result["local_usage"]["input_tokens"])
                self.assertEqual(result["evidence"], [])
                self.assertEqual(sources.expanded, [])

    def test_explicit_forward_not_started_is_known_partial_without_usage(self):
        sources, selections = self.fixture()
        result, initial, _ = self.run_fixture(sources, selections, judge=FixtureJudge(sources.labels(), fault="not_started"))
        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["unknown_outcome"])
        self.assertEqual(result["local_usage"]["input_tokens"], 0)
        self.assertEqual(result["admissions"][0]["execution_status"], "not_executed")
        self.assertEqual(result["evidence_ref_ids"], [initial["ref_id"]])

    def test_revocation_or_version_change_in_any_selected_source_rejects_all_results(self):
        for fault in ("revoked", "changed"):
            sources, selections = self.fixture(extra_source=True)
            unrelated = selections[1]["id"]
            def change_scope():
                getattr(sources.windows, fault).add(unrelated)
            judge = FixtureJudge(sources.labels(), after_score=change_scope)
            try:
                result, _, _ = self.run_fixture(sources, selections, judge=judge)
            except (AgentError, DomainError):
                continue  # A propagated authority failure is also fail-closed.
            with self.subTest(fault=fault):
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["evidence"], [])
                self.assertFalse(sources.expanded)

    def test_pointer_integrity_failure_is_not_classified_as_safe_capacity(self):
        sources, selections = self.fixture()
        def remove_pointer():
            sources.caches[selections[0]["id"]].missing = True
        result, _, _ = self.run_fixture(sources, selections,
            judge=FixtureJudge(sources.labels(), after_score=remove_pointer))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["evidence"], [])
        self.assertFalse(result["unknown_outcome"])

    def test_scope_manifest_is_bound_to_owner_task_and_exact_selections(self):
        sources, selections = self.fixture(extra_source=True)
        manifest, _ = sources.capture(Actor(OWNER), TASK, selections, lambda: None)
        sources.verify(Actor(OWNER), TASK, manifest, selections)
        for actor, task, selected in ((Actor(UUID(int=9)), TASK, selections),
                                      (Actor(OWNER), UUID(int=8), selections), (Actor(OWNER), TASK, selections[:1])):
            with self.subTest(owner=actor.owner_id, task=task, count=len(selected)), self.assertRaises(DomainError):
                sources.verify(actor, task, manifest, selected)
        pin = scope_evidence(Actor(OWNER), TASK, manifest)
        self.assertNotEqual(pin["ref_id"], scope_evidence(Actor(OWNER), UUID(int=8), manifest)["ref_id"])
        with self.assertRaises(DomainError):
            sources.capture(Actor(OWNER), TASK, [selections[0], selections[0]], lambda: None)

    def test_task_cancellation_after_a_decision_does_not_return_source_text(self):
        sources, selections = self.fixture()
        state = {"cancelled": False}
        def check_task():
            if state["cancelled"]:
                raise AgentError("fixture_task_cancelled")
        judge = FixtureJudge(sources.labels(), after_score=lambda: state.update(cancelled=True))
        result, _, _ = self.run_fixture(sources, selections, judge=judge, check_task=check_task)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(sources.expanded, [])

    def test_receipt_rejects_negative_tokens_or_a_total_above_the_admitted_budget(self):
        sources, selections = self.fixture()
        body, initial, _ = self.run_fixture(sources, selections)
        bad = copy.deepcopy(body)
        bad["trace"][0]["usage"]["input_tokens"] = -1
        total = sum(row["usage"]["input_tokens"] for row in bad["trace"])
        bad["local_usage"].update(known_input_tokens=total, input_tokens=total)
        with self.assertRaises(AgentError):
            validate_structure_receipt(configuration(), bad, [initial["ref_id"]])
        with self.assertRaises(AgentError):
            validate_structure_receipt(configuration(local_input_token_limit=1), body, [initial["ref_id"]])

    def test_unknown_receipt_cannot_fabricate_known_forward_or_drop_counts(self):
        sources, selections = self.fixture()
        body, initial, _ = self.run_fixture(sources, selections, judge=FixtureJudge(sources.labels(), fault="unknown"))
        for field in ("observed_forward_calls", "source_tokens_dropped"):
            bad = copy.deepcopy(body)
            bad["local_usage"][field] = 0
            with self.subTest(field=field), self.assertRaises(AgentError):
                validate_structure_receipt(configuration(), bad, [initial["ref_id"]])

    def test_error_receipt_never_carries_usable_evidence_even_with_known_usage(self):
        sources, selections = self.fixture()
        body, initial, _ = self.run_fixture(sources, selections)
        bad = copy.deepcopy(body)
        bad.update(status="error", error_code="fixture_integrity_failure")
        with self.assertRaises(AgentError):
            validate_structure_receipt(configuration(), bad, [initial["ref_id"]])

    def test_native_adapter_rejects_a_different_draft_identity_even_with_equal_body(self):
        source = fixture_source()
        windows = FixtureWindows({source.selection["id"]: source})
        native = SimpleNamespace(id=source.selection["id"], storage_security_domain="public", draft_sha256="c" * 64,
            security_domain="public", source_text_sha256=source.units[0].source_text_sha256)
        blob = SimpleNamespace(security_domain="public", sha256="c" * 64, size_bytes=1)
        class Session:
            def get(self, model, identifier):
                return native if getattr(model, "__tablename__", "") == "ingest_native_representations" else blob
        class Factory:
            @contextmanager
            def begin(self):
                yield Session()
        service = SimpleNamespace(factory=Factory(), store=SimpleNamespace(read=lambda pointer: b"x"))
        adapter = NavigationSources(service, None)
        adapter.windows = windows
        wrong_draft = SimpleNamespace(representation_id="d" * 64, literal_text=source.units[0].text,
            source_text_sha256=source.units[0].source_text_sha256)
        with patch("citequest.research_v3.source_navigation_sources.NativeRepresentationDraft.model_validate_json", return_value=wrong_draft), \
             patch("citequest.research_v3.source_navigation_sources.native_structure_source", return_value=replace(source, representation_id="d" * 64)):
            with self.assertRaises(DomainError):
                adapter._current(Actor(OWNER), source.selection, "N01")

    def test_native_adapter_denies_a_foreign_domain_before_reading_its_cas_bytes(self):
        source = fixture_source()
        native = SimpleNamespace(id=source.selection["id"], storage_security_domain="owner:" + str(UUID(int=9)),
            security_domain="owner:" + str(UUID(int=9)), draft_sha256="c" * 64,
            source_text_sha256=source.units[0].source_text_sha256)
        blob = SimpleNamespace(security_domain=native.storage_security_domain, sha256="c" * 64, size_bytes=1)
        class Session:
            def get(self, model, identifier):
                return native if getattr(model, "__tablename__", "") == "ingest_native_representations" else blob
        class Factory:
            @contextmanager
            def begin(self):
                yield Session()
        reads = []
        service = SimpleNamespace(factory=Factory(), store=SimpleNamespace(read=lambda pointer: reads.append(pointer) or b"x"))
        adapter = NavigationSources(service, None)
        adapter.windows = FixtureWindows({source.selection["id"]: source})
        draft = SimpleNamespace(representation_id=source.selection["id"], literal_text=source.units[0].text,
            source_text_sha256=source.units[0].source_text_sha256)
        with patch("citequest.research_v3.source_navigation_sources.NativeRepresentationDraft.model_validate_json", return_value=draft), \
             patch("citequest.research_v3.source_navigation_sources.native_structure_source", return_value=source):
            with self.assertRaises(DomainError):
                adapter._current(Actor(OWNER), source.selection, "N01")
        self.assertEqual(reads, [])


if __name__ == "__main__":
    unittest.main()
