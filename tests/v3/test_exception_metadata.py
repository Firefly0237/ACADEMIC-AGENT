"""One focused data-minimization check; no runtime or provider operation."""
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from uuid import UUID

from citequest.agents_v3.contracts import AgentError
from citequest.agents_v3.exception_metadata import exception_metadata, private_runtime_phase


class ExceptionMetadataControls(unittest.TestCase):
    def test_group_leaves_preserved_without_messages_arguments_or_locals(self):
        secret = "DO_NOT_EXPORT_DSN_OR_SQL_PARAMETER"
        try:
            try:
                raise RuntimeError(secret)
            except RuntimeError as cause:
                raise ValueError(secret) from cause
        except ValueError as error:
            group = ExceptionGroup(secret,[error,AgentError("scope_dependency_unavailable")])
        result = exception_metadata(group)
        encoded = json.dumps(result)
        self.assertNotIn(secret,encoded)
        self.assertFalse(result["exception_values_included"])
        self.assertEqual([item["type"] for item in result["leaves"]],["ValueError","AgentError"])
        self.assertEqual(result["leaves"][1]["code"],"scope_dependency_unavailable")
        self.assertTrue(result["leaves"][0]["frames"])
        self.assertIn("RuntimeError",[item["type"] for item in result["causal_nodes"]])
        self.assertIn("cause",[item["relation"] for item in result["causal_nodes"]])
        stderr, stdout = io.StringIO(), io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(stdout), self.assertRaises(ExceptionGroup), private_runtime_phase(
                "mcp_client_startup","initialize",task_id=UUID("00000000-0000-0000-0000-000000000001")):
            raise group
        records = [json.loads(line) for line in stderr.getvalue().splitlines()]
        self.assertNotIn(secret,stderr.getvalue())
        self.assertEqual(stdout.getvalue(),"")
        self.assertEqual([item["state"] for item in records],["started","failed"])
        self.assertEqual(records[-1]["child_exit_code"],"unknown")
        self.assertGreaterEqual(records[-1]["monotonic_elapsed_seconds"],0)


if __name__ == "__main__":
    unittest.main()
