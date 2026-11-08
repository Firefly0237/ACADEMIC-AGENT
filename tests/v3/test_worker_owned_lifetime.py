"""Owned thread/cleanup regressions. Queue, model and source services are doubles."""
import asyncio
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from citequest.agents_v3 import worker


def test_unavailable_decisionrag_configuration_is_rejected_before_claim():
    values = {"limits": {"mode": "bounded"}, "business_dsn_file": Path("business"),
        "checkpoint_dsn_file": Path("checkpoint"), "memory_dsn_file": Path("memory"),
        "skill_root": Path("skills"), "temporary_directory": Path("tmp"), "mode": "live", "tool_mode": "search"}
    config = worker.WorkerConfig.model_validate(values)
    assert "decisionrag_config_file" not in config.model_dump(mode="json")
    with pytest.raises(ValueError, match="DecisionRAG runtime configuration is not enabled"):
        worker.WorkerConfig.model_validate({**values, "decisionrag_config_file": Path("candidate.json")})


@pytest.fixture
def consumer(monkeypatch, tmp_path):
    def build(run, *, research=False, cleanup_failure=None):
        events, providers = [], []
        owner, task_id, run_id = uuid4(), uuid4(), uuid4()
        database = SimpleNamespace(development_dispatch_policy=None, development_campaign_policy=None)
        execution = SimpleNamespace(task_id=task_id, run_id=run_id, epoch=1,
            snapshot=lambda: {"kind": "research" if research else "qa", "owner_id": owner,
                "profile": {"profile_id": "research-v4"}},
            finish=lambda *_args, **_kwargs: events.append("finish"))

        @contextmanager
        def claim(*_args, **_kwargs):
            events.append("claim_enter")
            try:
                yield execution
            finally:
                events.append("claim_exit")

        class Provider:
            def __init__(self, *_args, **_kwargs):
                providers.append(self)

            async def close(self):
                events.append("provider_close")
                if cleanup_failure == "provider":
                    raise RuntimeError("synthetic_provider_close_failed")

        class Tools:
            def __init__(self, *_args, **_kwargs):
                pass

            async def __aenter__(self):
                events.append("mcp_enter")
                return self

            async def __aexit__(self, *_args):
                events.append("mcp_exit")
                if cleanup_failure == "mcp":
                    raise RuntimeError("synthetic_mcp_close_failed")

        class Graph:
            def __init__(self, *_args):
                pass

            def run(self):
                run(self, events)

        config = SimpleNamespace(mode="live", database=lambda: database, limits=object(),
            api_key_file=Path("synthetic-key"), metadata_receipt_file=Path("synthetic-receipt"),
            temporary_directory=tmp_path, tool_mode="fixture", business_dsn_file=Path("synthetic-dsn"),
            search_config_file=Path("synthetic-search") if research else None,
            material_config_file=Path("synthetic-material") if research else None,
            fixture_evidence_file=None, decisionrag_config_file=None, catalog_scope_policy=None,
            skill_root=tmp_path)
        monkeypatch.setattr(worker, "claim", claim)
        monkeypatch.setattr(worker, "MemoryService", lambda _db: SimpleNamespace(flush=lambda: None))
        monkeypatch.setattr(worker, "ResponsesProvider", Provider)
        monkeypatch.setattr(worker, "MCPTools", Tools)
        monkeypatch.setattr(worker, "AgentGraph", Graph)
        monkeypatch.setattr(worker, "private_runtime_observation", lambda *_args, **_kwargs: None)
        return config, events, providers

    return build


@pytest.mark.parametrize("thread_fails,cleanup_failure", [(False, None), (True, None), (False, "mcp"), (False, "provider")])
def test_cancellation_waits_for_owned_graph_and_rejects_later_dispatch(consumer, thread_fails, cleanup_failure):
    async def scenario():
        started, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def run(_graph, events):
            events.append("graph_started")
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            events.append("graph_finished")
            if thread_fails:
                raise RuntimeError("synthetic_graph_failure")

        config, events, providers = consumer(run, cleanup_failure=cleanup_failure)
        task = asyncio.create_task(worker.run_once(config))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert events == ["claim_enter", "mcp_enter", "graph_started"]
            with pytest.raises(worker.AgentError, match="worker_cancelled_before_dispatch"):
                await providers[0].before_dispatch()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        if thread_fails or cleanup_failure is not None:
            assert isinstance(caught.value.__cause__, RuntimeError)
        assert events == ["claim_enter", "mcp_enter", "graph_started", "graph_finished",
            "mcp_exit", "finish", "provider_close", "claim_exit"]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["material_constructor", "material_close"])
def test_reference_preflight_cleans_native_clients_and_sql_engines(consumer, monkeypatch, failure):
    cleanup = []

    def close(name):
        cleanup.append(name)
        if failure == "material_close" and name == "material_native":
            raise RuntimeError("synthetic_native_close_failed")

    search = SimpleNamespace(native=SimpleNamespace(close=lambda: close("search_native")),
        factory=SimpleNamespace(kw={"bind": SimpleNamespace(dispose=lambda: close("search_sql"))}))
    material = SimpleNamespace(native_index=SimpleNamespace(native=SimpleNamespace(close=lambda: close("material_native"))),
        factory=SimpleNamespace(kw={"bind": SimpleNamespace(dispose=lambda: close("material_sql"))}))

    def material_service(_config):
        if failure == "material_constructor":
            raise RuntimeError("synthetic_material_constructor_failed")
        return material

    for name, module in {
        "citequest.search_v3.runtime": SimpleNamespace(service_from_config=lambda _config: search),
        "citequest.material_evidence_v3": SimpleNamespace(service_from_config=material_service),
        "citequest.research_v3.evidence": SimpleNamespace(EvidenceAccess=lambda *_args: SimpleNamespace()),
        "citequest.research_v3.school_export": SimpleNamespace(reference_preflight=lambda *_args, **_kwargs: {}),
        "citequest.agents_v3.research_task": SimpleNamespace(**{name: lambda graph: graph
            for name in ("ResearchTaskGraph", "ResearchTaskGraphV4", "ResearchTaskGraphV5")}),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    def run(graph, _events):
        graph.reference_preflight({})

    config, events, _providers = consumer(run, research=True)
    result = asyncio.run(worker.run_once(config))
    assert result["status"] == "failed" and result["error_code"] == "RuntimeError"
    assert cleanup == (["search_native", "search_sql"] if failure == "material_constructor"
        else ["material_native", "material_sql", "search_native", "search_sql"])
    assert events[-4:] == ["mcp_exit", "finish", "provider_close", "claim_exit"]
