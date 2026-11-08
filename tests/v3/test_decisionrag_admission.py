"""Targeted pure checks for navigation admission and pinned component loading.

Wheel bytes and EvidenceAccess here are explicit test doubles; no dependency,
model, database, Docker or network is loaded. These do not prove product ACLs.
"""
from __future__ import annotations

import ast
import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from citequest.agents_v3.contracts import AgentError
from citequest.research_v3.decisionrag import NavigationLimits
from citequest.research_v3.decisionrag_evidence import navigate_issued_evidence

ROOT = Path(__file__).resolve().parents[2]


def load_function(path: str, names: set[str], namespace: dict) -> dict:
    source = ast.parse((ROOT/path).read_text(encoding="utf-8-sig"))
    chosen = [node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(chosen) == len(names)
    exec(compile(ast.Module(body=chosen,type_ignores=[]),path,"exec"),namespace)
    return namespace


class AdmissionControls(unittest.TestCase):
    def test_wheel_and_receipt_verified_without_dependency_search_path_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root/"vendor/laya-0.4.22-py3-none-any.whl"
            wheel.parent.mkdir()
            wheel.write_bytes(b"EXPLICIT_SYNTHETIC_WHEEL_NOT_IMPORTED")
            receipt = root/"receipt.json"
            receipt.write_bytes(b"{}")
            namespace = {"Path":Path,"hashlib":hashlib,"FIXED_WHEEL_SHA256":hashlib.sha256(wheel.read_bytes()).hexdigest()}
            verify = load_function("scripts/s14/run_laya_component.py",{"verify_before_import"},namespace)["verify_before_import"]
            original_path, original_modules = list(sys.path), set(sys.modules)
            self.assertEqual(verify(root,receipt,hashlib.sha256(b"{}").hexdigest()),wheel)
            self.assertEqual(sys.path,original_path)
            self.assertEqual(set(sys.modules),original_modules)
            with self.assertRaisesRegex(SystemExit,"integrity_failed"):
                verify(root,receipt,"0"*64)
            wheel.write_bytes(b"MODIFIED_SYNTHETIC_WHEEL")
            with self.assertRaisesRegex(SystemExit,"integrity_failed"):
                verify(root,receipt,hashlib.sha256(b"{}").hexdigest())

    def test_initial_baseline_source_io_consumes_same_navigation_window(self):
        clock, reads, guards = [0.0], [], []
        class FixtureAccess:
            def read(self, actor, task_id, ref):
                reads.append(ref)
                clock[0] = 31.0
                return {}  # The expired read must stop before interpreting any DTO.
        fixture_module = types.ModuleType("citequest.research_v3.evidence")
        fixture_module.EvidenceAccess = FixtureAccess
        with patch.dict(sys.modules,{fixture_module.__name__:fixture_module}), \
                patch("citequest.research_v3.decisionrag_evidence.time.monotonic",side_effect=lambda:clock[0]), \
                self.assertRaisesRegex(AgentError,"active_budget_exhausted"):
            navigate_issued_evidence("Question",["synthetic:ref"],None,access=FixtureAccess(),actor=None,task_id=None,
                current_task_scope=lambda:guards.append("called"),before_decision=lambda request:self.fail("no inference admission"),
                limits=NavigationLimits())
        self.assertEqual(reads,["synthetic:ref"])
        self.assertEqual(guards,[])

if __name__ == "__main__":
    unittest.main()
