"""Real AnyIO task lifetime controls with explicit session/Execution fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import anyio

import citequest.agents_v3.tools as module
from citequest.agents_v3.contracts import AgentError, digest


class Execution:
    def __init__(self, *, actual_rpc=False):
        self.events = []
        self.actual_rpc = actual_rpc
        self.settlements = []

    def snapshot(self):
        return {"kind": "research", "brief": {}, "profile": {"active_seconds": 120}, "elapsed": 0}

    def scope_event(self, name, value):
        self.events.append({"name": name, "value": value})

    def reserve(self, *args, **kwargs):
        return {"id": uuid4(), "status": "reserved" if self.actual_rpc else "settled", "response": {"fixture": True}}

    def mark_sent(self, attempt_id):
        return 30

    def settle(self, attempt_id, result, usage):
        self.settlements.append(result)

    def unknown(self, *args):
        raise AssertionError("fixture_transport_must_not_leave_unknown_call")


class Transport:
    def __init__(self, *, block_init=False, fail_init=False, block_exit=False):
        self.assigned, self.initialize_release, self.exit_entered, self.exit_release = (
            asyncio.Event() for _ in range(4)
        )
        if not block_init:
            self.initialize_release.set()
        if not block_exit:
            self.exit_release.set()
        self.fail_init = fail_init
        self.enters, self.exits, self.scope_calls, self.early_calls = [], [], 0, 0
        self.initialized = False

    @asynccontextmanager
    async def stdio(self, parameters):
        self.enters.append(asyncio.current_task())
        async with anyio.create_task_group():
            try:
                yield "explicit-reader-fixture", "explicit-writer-fixture"
            finally:
                self.exit_entered.set()
                await self.exit_release.wait()
                self.exits.append(asyncio.current_task())

    def factory(self, reader, writer):
        outer = self

        class Session:
            async def __aenter__(self):
                self.group = anyio.create_task_group()
                await self.group.__aenter__()
                self.owner = asyncio.current_task()
                return self

            async def __aexit__(self, *exc):
                assert asyncio.current_task() is self.owner
                self.group.cancel_scope.cancel()
                return await self.group.__aexit__(*exc)

            async def initialize(self):
                outer.assigned.set()
                await outer.initialize_release.wait()
                if outer.fail_init:
                    raise AgentError("explicit_fixture_initialization_failed")
                outer.initialized = True

            async def list_tools(self):
                names = module.TOOL_NAMES | module.WORKSET_TOOL_NAMES
                return SimpleNamespace(
                    tools=[SimpleNamespace(name=name, inputSchema={"type": "object"}) for name in names]
                )

            async def read_resource(self, uri):
                if not outer.initialized:
                    outer.early_calls += 1
                    raise AgentError("explicit_fixture_RPC_before_initialized")
                outer.scope_calls += 1
                return SimpleNamespace(
                    contents=[
                        SimpleNamespace(
                            text=json.dumps({"status": "ready", "brief_sha256": digest({}), "source_reads": False})
                        )
                    ]
                )

        return Session()


class LifetimeControls(unittest.IsolatedAsyncioTestCase):
    def patches(self, transport):
        return patch.object(module, "stdio_client", transport.stdio), patch.object(
            module, "ClientSession", transport.factory
        )

    async def test_lazy_missing_input_opens_no_context_or_scope(self):
        transport = Transport()
        with self.patches(transport)[0], self.patches(transport)[1]:
            async with module.MCPTools(Execution(), Path("EXPLICIT_NO_CONFIG"), lazy=True) as tools:
                self.assertIsNone(tools._owner)
                self.assertIsNone(tools.session)
            self.assertEqual((transport.enters, transport.scope_calls), ([], 0))
            with self.assertRaises(AgentError) as caught:
                await tools.open()
            self.assertEqual(caught.exception.code, "mcp_transport_closed")

    async def test_single_owner_enter_exit_from_distinct_lazy_consumers(self):
        transport = Transport()
        with self.patches(transport)[0], self.patches(transport)[1]:
            async with module.MCPTools(Execution(), Path("EXPLICIT_NO_CONFIG"), lazy=True) as tools:
                await asyncio.create_task(tools.check_scope())
                await asyncio.create_task(tools.call("search_evidence", {"query": "fixture"}, logical_key="fixture"))
                self.assertIsNot(tools._owner, asyncio.current_task())
            self.assertEqual(len(transport.enters), 1)
            self.assertEqual(transport.enters, transport.exits)
            self.assertTrue(tools._owner.done())
            self.assertIsNone(tools.session)

    async def test_session_assigned_but_not_ready_public_methods_wait(self):
        transport = Transport(block_init=True)
        with self.patches(transport)[0], self.patches(transport)[1]:
            async with module.MCPTools(Execution(), Path("EXPLICIT_NO_CONFIG"), lazy=True) as tools:
                opener = asyncio.create_task(tools.open())
                await transport.assigned.wait()
                scope = asyncio.create_task(tools.check_scope())
                call = asyncio.create_task(tools.call("search_evidence", {"query": "fixture"}, logical_key="fixture"))
                await asyncio.sleep(0)
                self.assertFalse(scope.done())
                self.assertFalse(call.done())
                self.assertEqual(transport.early_calls, 0)
                transport.initialize_release.set()
                await asyncio.gather(opener, scope, call)
            self.assertEqual(len(transport.enters), 1)

    async def test_concurrent_twenty_opens_have_one_lifetime_owner(self):
        transport = Transport(block_init=True)
        with self.patches(transport)[0], self.patches(transport)[1]:
            async with module.MCPTools(Execution(), Path("EXPLICIT_NO_CONFIG"), lazy=True) as tools:
                consumers = [asyncio.create_task(tools.open()) for _ in range(20)]
                await transport.assigned.wait()
                self.assertEqual(len(transport.enters), 1)
                transport.initialize_release.set()
                await asyncio.gather(*consumers)
            self.assertEqual(transport.enters, transport.exits)

    async def test_cancelled_consumer_does_not_cancel_shared_initialization(self):
        transport = Transport(block_init=True)
        with self.patches(transport)[0], self.patches(transport)[1]:
            async with module.MCPTools(Execution(), Path("EXPLICIT_NO_CONFIG"), lazy=True) as tools:
                consumer = asyncio.create_task(tools.open())
                await transport.assigned.wait()
                consumer.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await consumer
                self.assertFalse(tools._owner.cancelled())
                transport.initialize_release.set()
                await asyncio.create_task(tools.check_scope())
            self.assertEqual(transport.enters, transport.exits)

    async def test_close_with_session_still_present_rejects_new_consumers(self):
        transport = Transport(block_exit=True)
        with self.patches(transport)[0], self.patches(transport)[1]:
            tools = module.MCPTools(Execution(), Path("EXPLICIT_NO_CONFIG"), lazy=True)
            await tools.__aenter__()
            await tools.open()
            closer = asyncio.create_task(tools.__aexit__(None, None, None))
            await transport.exit_entered.wait()
            self.assertIsNotNone(tools.session)
            for request in (
                tools.check_scope(),
                tools.call("search_evidence", {"query": "fixture"}, logical_key="fixture"),
            ):
                with self.assertRaises(AgentError) as caught:
                    await request
                self.assertEqual(caught.exception.code, "mcp_transport_closed")
            transport.exit_release.set()
            await closer

    async def test_cancellation_during_cleanup_waits_then_propagates(self):
        transport = Transport(block_exit=True)
        with self.patches(transport)[0], self.patches(transport)[1]:
            tools = module.MCPTools(Execution(), Path("EXPLICIT_NO_CONFIG"), lazy=True)
            await tools.open()
            closer = asyncio.create_task(tools.__aexit__(None, None, None))
            await transport.exit_entered.wait()
            closer.cancel()
            await asyncio.sleep(0)
            closer.cancel()
            await asyncio.sleep(0)
            self.assertFalse(tools._owner.done())
            transport.exit_release.set()
            with self.assertRaises(asyncio.CancelledError):
                await closer
            self.assertEqual(transport.enters, transport.exits)
            self.assertTrue(tools._owner.done())

    async def test_body_failure_preserved_after_owner_release(self):
        transport = Transport()
        with self.patches(transport)[0], self.patches(transport)[1]:
            with self.assertRaisesRegex(RuntimeError, "explicit_body_failure"):
                async with module.MCPTools(Execution(), Path("EXPLICIT_NO_CONFIG"), lazy=True) as tools:
                    await tools.open()
                    raise RuntimeError("explicit_body_failure")
            self.assertEqual(transport.enters, transport.exits)

    async def test_initialization_failure_closes_and_does_not_retry(self):
        transport = Transport(fail_init=True)
        with self.patches(transport)[0], self.patches(transport)[1]:
            tools = module.MCPTools(Execution(), Path("EXPLICIT_NO_CONFIG"), lazy=True)
            for request in (tools.open(), tools.open(), tools.__aexit__(None, None, None)):
                with self.assertRaises(BaseExceptionGroup) as caught:
                    await request
                self.assertTrue(
                    any(
                        getattr(error, "code", None) == "explicit_fixture_initialization_failed"
                        for error in leaves(caught.exception)
                    )
                )
            self.assertEqual(transport.enters, transport.exits)
            self.assertEqual(len(transport.enters), 1)
            self.assertTrue(tools._owner.done())

    async def test_cancelled_cleanup_keeps_cancel_primary_when_owner_then_fails(self):
        transport = Transport(block_init=True, fail_init=True)
        with self.patches(transport)[0], self.patches(transport)[1]:
            tools = module.MCPTools(Execution(), Path("EXPLICIT_NO_CONFIG"), lazy=True)
            waiter = asyncio.create_task(tools.open())
            await transport.assigned.wait()
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            closer = asyncio.create_task(tools.__aexit__(None, None, None))
            await asyncio.sleep(0)
            closer.cancel()
            await asyncio.sleep(0)
            transport.initialize_release.set()
            with self.assertRaises(asyncio.CancelledError) as caught:
                await closer
            self.assertTrue(
                any(
                    getattr(error, "code", None) == "explicit_fixture_initialization_failed"
                    for error in leaves(caught.exception.__cause__)
                )
            )
            await asyncio.sleep(0)
            # Test-only observation of the pinned Python Future diagnostic flag:
            # no caller manually retrieves ready.exception() in this control.
            self.assertFalse(tools._ready._log_traceback)
            self.assertEqual(transport.enters, transport.exits)


def leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in leaves(child)]
    return [error]


FIXTURE_SERVER = None
REAL_CHILD_PIDS = set()


class RealStdioControls(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        if FIXTURE_SERVER is None:
            self.skipTest("explicit fixture server reference is required")

    def parameters(self, pidfile, *, bad_catalog=False):
        original = module.StdioServerParameters

        def construct(*, command, args, env):
            self.assertFalse(any("KEY" in key or "TOKEN" in key or "DSN" in key for key in env))
            return original(
                command=command,
                args=[str(FIXTURE_SERVER), "--pid-file", str(pidfile)] + (["--bad-catalog"] if bad_catalog else []),
                env=env,
            )

        return patch.object(module, "StdioServerParameters", construct)

    def assert_gone(self, pidfile):
        pid = int(pidfile.read_text())
        REAL_CHILD_PIDS.add(pid)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_official_stdio_real_child_lazy_consumers_and_process_release(self):
        with tempfile.TemporaryDirectory(prefix="cq-mcp-lifetime-") as directory:
            pidfile = Path(directory) / "fixture-child.pid"
            with self.parameters(pidfile):
                execution = Execution(actual_rpc=True)
                async with module.MCPTools(execution, Path("EXPLICIT_UNUSED_CONFIG"), lazy=True) as tools:
                    self.assertFalse(pidfile.exists())
                    await asyncio.create_task(tools.check_scope())
                    result = await asyncio.create_task(
                        tools.call(
                            "search_evidence",
                            {"query": "explicit fixture", "limit": 1},
                            logical_key="fixture:real-stdio",
                        )
                    )
                    self.assertTrue(result["fixture"])
                    self.assertEqual(len(execution.settlements), 1)
                self.assert_gone(pidfile)
                self.assertTrue(tools._owner.done())

    async def test_official_stdio_bad_catalog_releases_real_child_without_retry(self):
        with tempfile.TemporaryDirectory(prefix="cq-mcp-lifetime-") as directory:
            pidfile = Path(directory) / "fixture-child.pid"
            with self.parameters(pidfile, bad_catalog=True):
                tools = module.MCPTools(Execution(), Path("EXPLICIT_UNUSED_CONFIG"), lazy=True)
                for request in (tools.open(), tools.open(), tools.__aexit__(None, None, None)):
                    with self.assertRaises(BaseExceptionGroup) as caught:
                        await request
                    self.assertTrue(
                        any(
                            getattr(error, "code", None) == "mcp_tool_catalog_mismatch"
                            for error in leaves(caught.exception)
                        )
                    )
                self.assert_gone(pidfile)
                self.assertTrue(tools._owner.done())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture-server", type=Path, required=True)
    args = parser.parse_args()
    FIXTURE_SERVER = args.fixture_server
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(
        unittest.TestSuite(
            unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (LifetimeControls, RealStdioControls)
        )
    )
    args.out.write_text(
        json.dumps(
            {
                "status": "PASS" if result.wasSuccessful() else "FAIL",
                "controls": result.testsRun,
                "scope": "real asyncio/AnyIO plus2real official MCP stdio child controls; explicit source/Execution fixtures",
                "real_stdio_subprocesses_observed": len(REAL_CHILD_PIDS),
                "real_stdio_subprocess_controls_planned": 2,
                "skipped": len(result.skipped),
                "real_PG": "not_run",
                "model_requests_sent": 0,
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
