"""Request capacity controls with explicit source fixtures, no provider or DB."""
from __future__ import annotations

import copy
import json
import unittest
from types import SimpleNamespace

from citequest.agents_v3.context import wire_bytes
from citequest.agents_v3.contracts import PROFILES, AgentError, digest, graph_version_for_kind
from citequest.agents_v3.workset import WorksetNodes
from citequest.agents_v3.workset_v3 import GENERATION_POLICY
from citequest.research_v3.quote_spans import QuoteRegistry, SelectorSummary


def nodes(profile, evidence):
    selected = {"kind": "native_representation", "id": digest("capacity-fixture-selection")}
    task = {"profile": PROFILES[profile].model_dump(), "brief": {"kind": "summary", "question": "Summarize all supplied literal contexts.",
        "selected_papers": [selected], "requested_direction": "neutral"}, "output_tokens": 0}
    context = {"hard_memory_refs": [], "memory_refs": [], "excluded": []}
    graph = SimpleNamespace(execution=SimpleNamespace(snapshot=lambda: copy.deepcopy(task)),
        projection=lambda **kwargs: ({"confirmed_user_memory_data": [], "selected_user_skills": []}, copy.deepcopy(context)),
        user_turns=lambda: [{"turn_id": "U1", "text": task["brief"]["question"]}])
    scope = QuoteRegistry(evidence).scope(evidence)
    return WorksetNodes(graph), scope


def evidence(size=12000):
    selected = {"kind": "native_representation", "id": digest("capacity-fixture-selection")}
    values = []
    for index in range(5):
        text = ("x" * (size - 2)) + "\ue032\ue001"
        values.append({"ref_id": "source-window:"+digest(str(index)), "document_id": "D01", "selection": selected,
            "source_ref": selected, "source_sha256": digest(text.encode()), "unit_text_sha256": digest(text.encode()),
            "source_version_sha256": digest("fixture-version"), "source_pin_sha256": digest("fixture-pin"),
            "span_start": 0, "span_end": len(text), "text": text, "coordinate_system": "unicode_codepoints",
            "coverage_kind": "native_literal", "evidence_type": "source_window"})
    return values


class CapacityV4(unittest.TestCase):
    def payload(self, version, values):
        base, scope = nodes(version, values)
        return base.payload({"workset_context_manifest": {"hard_memory_refs": []}},
            {"untrusted_evidence_data": scope.contexts}, SelectorSummary, "citequest_document_reduce",
            GENERATION_POLICY, 3072, alias_scope=scope), scope

    def test_version_changes_only_id_and_request_limit(self):
        previous, current = PROFILES["workset-v3"].model_dump(), PROFILES["workset-v4"].model_dump()
        differences = {key for key in previous if previous[key] != current[key]}
        self.assertEqual(differences, {"profile_id", "request_bytes"})
        self.assertEqual(previous["request_bytes"], 49152)
        self.assertEqual(current["request_bytes"], 131072)
        self.assertEqual(current["active_seconds"], 600)
        self.assertNotEqual(graph_version_for_kind("summary", "workset-v3"), graph_version_for_kind("summary", "workset-v4"))

    def test_reducer_full_source_over_old_cap_preserved_with_PUA(self):
        values = evidence()
        with self.assertRaisesRegex(AgentError, "required_context_exceeds_wire_limit"):
            self.payload("workset-v3", values)
        payload, scope = self.payload("workset-v4", values)
        self.assertGreater(wire_bytes(payload), 49152)
        self.assertLessEqual(wire_bytes(payload), 131072)
        prompt = json.loads(payload["input"][0]["content"])
        self.assertEqual(prompt["untrusted_evidence_data"], scope.contexts)
        actual = "".join(unit["text"] for group in prompt["untrusted_evidence_data"] for unit in group["quote_units"] if "text" in unit)
        self.assertEqual(actual, "".join(item["text"] for item in values))
        self.assertEqual(actual.count("\ue032\ue001"), 5)

    def test_still_rejects_oversize_instead_of_clipping_hard_sources(self):
        with self.assertRaisesRegex(AgentError, "required_context_exceeds_wire_limit"):
            self.payload("workset-v4", evidence(31000))

    def test_exact_dynamic_enum_and_quote_binding_unchanged(self):
        payload, scope = self.payload("workset-v4", evidence(100))
        self.assertEqual(payload["text"]["format"]["schema"], scope.schema(SelectorSummary))
        self.assertEqual(json.loads(payload["input"][0]["content"])["source_transport"], scope.binding)
        self.assertEqual(payload["text"]["format"]["name"], "citequest_document_reduce_v3")


if __name__ == "__main__":
    unittest.main()
