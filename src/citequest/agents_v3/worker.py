"""A bounded PG command consumer. LangGraph owns all Agent control flow."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID, uuid4

os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import httpx2
from pydantic import Field, model_validator

from .catalog_scope import CatalogScopePolicy
from .contracts import AgentError, RuntimeLimits, Strict
from .database import Database, claim
from .development_campaign import load_campaign
from .development_dispatch import load_policy
from .exception_metadata import private_runtime_observation
from .graph import AgentGraph
from .memory import MemoryService
from .provider import ResponsesProvider
from .tools import MCPTools

if TYPE_CHECKING:
    from citequest.v2.domain import Actor


class WorkerConfig(Strict):
    limits: RuntimeLimits
    business_dsn_file: Path
    checkpoint_dsn_file: Path
    memory_dsn_file: Path
    api_key_file: Path | None = None
    metadata_receipt_file: Path | None = None
    model_metadata_receipt_files: dict[str, Path] | None = Field(default=None, max_length=2, exclude_if=lambda value: value is None)
    search_config_file: Path | None = None
    material_config_file: Path | None = None
    decisionrag_config_file: Path | None = Field(default=None, exclude_if=lambda value: value is None)
    decisionrag_config_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")] | None = Field(default=None, exclude_if=lambda value: value is None)
    fixture_evidence_file: Path | None = None
    skill_root: Path
    temporary_directory: Path
    mode: str = Field(pattern="^(fixture|live)$")
    tool_mode: str = Field(pattern="^(fixture|search)$")
    dispatch_policy_file: Path | None = None
    dispatch_lane: Literal["qa", "long"] | None = None
    catalog_scope_policy: CatalogScopePolicy | None = None
    development_dispatch_policy_file: Path | None = None
    development_dispatch_policy_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")] | None = None
    development_campaign_file: Path | None = None
    development_campaign_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")] | None = None

    @model_validator(mode="after")
    def paired_development_policy_reference(self) -> WorkerConfig:
        if self.model_metadata_receipt_files is not None:
            from .model_contracts import model_contract
            for alias in self.model_metadata_receipt_files:
                model_contract(alias)
        if (self.decisionrag_config_file is None) != (self.decisionrag_config_sha256 is None):
            raise ValueError("DecisionRAG requires both trusted configuration file and exact SHA")
        if self.decisionrag_config_file is not None:
            if self.mode != "live" or self.tool_mode != "search" or self.catalog_scope_policy != "task-catalog-snapshot/1":
                raise ValueError("DecisionRAG requires live search and pinned product catalog scope")
            from citequest.research_v3.decisionrag_task import load_task_config
            load_task_config(self.decisionrag_config_file, self.decisionrag_config_sha256)
        if (self.development_dispatch_policy_file is None) != (self.development_dispatch_policy_sha256 is None):
            raise ValueError("development policy requires both trusted file and exact SHA")
        if (self.development_campaign_file is None) != (self.development_campaign_sha256 is None):
            raise ValueError("development campaign requires both trusted file and exact SHA")
        if self.development_campaign_file is not None and self.development_dispatch_policy_file is not None:
            raise ValueError("named-task and campaign development policies are mutually exclusive")
        return self

    def database(self) -> Database:
        development = (load_policy(self.development_dispatch_policy_file, self.development_dispatch_policy_sha256)
            if self.development_dispatch_policy_file is not None else None)
        campaign = (load_campaign(self.development_campaign_file, self.development_campaign_sha256)
            if self.development_campaign_file is not None else None)
        if campaign is not None:
            campaign.require_configuration(self.limits, self.catalog_scope_policy)
            campaign.require_worker(None, self.mode)
        return Database.from_files(self.business_dsn_file, self.checkpoint_dsn_file, self.memory_dsn_file,
            dispatch_policy_file=self.dispatch_policy_file, worker_lane=self.dispatch_lane,
            catalog_scope_policy=self.catalog_scope_policy, development_dispatch_policy=development,
            development_campaign_policy=campaign)


async def _run_graph_to_completion(run: Callable[[], None], on_cancel: Callable[[], None]) -> None:
    # Cancelling to_thread's awaiter cannot stop its already running thread.
    # Keep the MCP session, provider and Saver lock alive until that owner ends.
    owner = asyncio.create_task(asyncio.to_thread(run), name="citequest-agent-graph")
    cancelled = False
    failure: BaseException | None = None
    while not owner.done():
        try:
            await asyncio.shield(owner)
        except asyncio.CancelledError:
            cancelled = True
            on_cancel()
        except BaseException as error:
            failure = error
            break
    if failure is None:
        try:
            owner.result()
        except BaseException as error:
            failure = error
    if cancelled:
        raise asyncio.CancelledError from failure
    if failure is not None:
        raise failure


async def run_once(config: WorkerConfig, *, transport: httpx2.MockTransport | None = None, worker_id: str | None = None,
                   task_id: UUID | None = None) -> dict:
    started = time.monotonic()
    if (config.mode == "fixture" and transport is None) or (config.mode == "live" and transport is not None):
        raise AgentError("worker_provider_mode_invalid")
    database = config.database()
    if database.development_dispatch_policy is not None:
        # This check precedes memory.flush and any claim/Task mutation. A
        # development worker can never consume the ordinary product queue.
        database.development_dispatch_policy.require_worker(task_id, config.mode)
    campaign = getattr(database, "development_campaign_policy", None)
    if campaign is not None:
        campaign.require_worker(task_id, config.mode)
    memory = MemoryService(database)
    if campaign is None:
        memory.flush()
    with claim(database, worker_id or str(uuid4()), limits=config.limits, task_id=task_id) as execution:
        if execution is None:
            return {"status": "idle"}
        provider = None
        graph_stopping = asyncio.Event()
        try:
            from citequest.research_v3.decisionrag_task import task_requests_navigation
            if task_requests_navigation(execution.snapshot()) and config.decisionrag_config_file is None:
                raise AgentError("decisionrag_runtime_not_configured")
            if campaign is not None:
                # Claim first validates the exact history and scoped durable queue;
                # an unknown hold must not flush unrelated state before admission.
                memory.flush()
            alias = execution.snapshot()["profile"]["model_alias"]
            receipt = (config.model_metadata_receipt_files.get(alias, config.metadata_receipt_file)
                if config.model_metadata_receipt_files is not None else config.metadata_receipt_file)
            provider = ResponsesProvider(execution, api_key_file=config.api_key_file,
                metadata_receipt_file=receipt,
                transport=transport, fixture=config.mode == "fixture")
            config.temporary_directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="s08-mcp-", dir=config.temporary_directory) as directory:
                tool_config = Path(directory) / "tool.json"
                tool_config.write_text(json.dumps({"mode": config.tool_mode, "task_id": str(execution.task_id),
                    "epoch": execution.epoch, "business_dsn_file": str(config.business_dsn_file),
                    "search_config_file": str(config.search_config_file) if config.search_config_file else None,
                    "material_config_file": str(config.material_config_file) if config.material_config_file else None,
                    "fixture_evidence_file": str(config.fixture_evidence_file) if config.fixture_evidence_file else None,
                    **({"decisionrag_config_file": str(config.decisionrag_config_file),
                        "decisionrag_config_sha256": config.decisionrag_config_sha256}
                        if config.decisionrag_config_file is not None else {}),
                    **({'catalog_scope_policy': config.catalog_scope_policy} if config.catalog_scope_policy is not None else {})}), encoding="utf-8")
                research = execution.snapshot()["kind"] == "research"
                async with MCPTools(execution, tool_config, lazy=research) as tools:
                    async def source_scope_before_wire() -> None:
                        if graph_stopping.is_set():
                            raise AgentError("worker_cancelled_before_dispatch")
                        task = execution.snapshot()
                        if research and task["brief"].get("research_action") == "suggest":
                            from .research_runtime import current_planning_input
                            value = await asyncio.to_thread(current_planning_input, execution)
                            execution.scope_event("scope_checked", {"operation": "research_intake_only", "source_reads": False,
                                "brief_id": value["brief_id"], "brief_sha256": value["brief_sha256"],
                                "source_contents_included": False})
                        else:
                            await tools.check_scope()
                    provider.before_dispatch = source_scope_before_wire

                    async def validate_context_before_wire(payload: dict) -> None:
                        task = execution.snapshot()
                        await asyncio.to_thread(memory.validate_wire_context, task, payload)
                        from .user_skills import validate_task_skills
                        await asyncio.to_thread(validate_task_skills, database, task, payload)
                        if research:
                            from .research_runtime import validate_wire_intake
                            await asyncio.to_thread(validate_wire_intake, execution, payload)

                    provider.before_payload_dispatch = validate_context_before_wire
                    workflow = AgentGraph(execution, provider, tools, memory, config.skill_root)

                    def authorize_dialogue_refs(actor: Actor, task_id: UUID, refs: list[str]) -> None:
                        # A read-only projection hook for explicit new graph
                        # versions; it cannot widen the running Task identity.
                        from citequest.research_v3.evidence import EvidenceAccess
                        from citequest.search_v3.runtime import service_from_config

                        current = execution.snapshot()
                        if (actor.owner_id,task_id) != (current["owner_id"],execution.task_id):
                            raise AgentError("dialogue_anchor_task_identity_changed",403)
                        if config.search_config_file is None:
                            raise AgentError("dialogue_anchor_source_authorization_unavailable",503)
                        search, material = None, None
                        try:
                            search = service_from_config(config.search_config_file)
                            if config.material_config_file is not None:
                                from citequest.material_evidence_v3 import service_from_config as material_service
                                material = material_service(config.material_config_file)
                            EvidenceAccess(database,search,material).authorize(actor,task_id,refs)
                        finally:
                            # ExitStack attempts every close even when one
                            # native client fails; SQL engines are still disposed.
                            with ExitStack() as cleanup:
                                if search is not None:
                                    cleanup.callback(search.factory.kw["bind"].dispose)
                                    cleanup.callback(search.native.close)
                                if material is not None:
                                    cleanup.callback(material.factory.kw["bind"].dispose)
                                    if material.native_index is not None:
                                        cleanup.callback(material.native_index.native.close)

                    workflow.authorize_dialogue_refs = authorize_dialogue_refs
                    if research:
                        if config.mode == "live" and config.search_config_file is not None:
                            def reference_preflight(state: dict) -> dict:
                                from citequest.research_v3.evidence import EvidenceAccess
                                from citequest.research_v3.school_export import reference_preflight as check_references
                                from citequest.search_v3.runtime import service_from_config
                                from citequest.v2.domain import Actor
                                search, material = None, None
                                try:
                                    search = service_from_config(config.search_config_file)
                                    if config.material_config_file is not None:
                                        from citequest.material_evidence_v3 import service_from_config as material_service
                                        material = material_service(config.material_config_file)
                                    access = EvidenceAccess(database, search, material)
                                    actor = Actor(execution.snapshot()["owner_id"])
                                    return check_references(state, read_evidence=lambda ref: access.read(actor, execution.task_id, ref),
                                        read_paper=lambda locator: search.detail(actor, locator))
                                finally:
                                    with ExitStack() as cleanup:
                                        if search is not None:
                                            cleanup.callback(search.factory.kw["bind"].dispose)
                                            cleanup.callback(search.native.close)
                                        if material is not None:
                                            cleanup.callback(material.factory.kw["bind"].dispose)
                                            if material.native_index is not None:
                                                cleanup.callback(material.native_index.native.close)
                            workflow.reference_preflight = reference_preflight
                        from .research_task import ResearchTaskGraph, ResearchTaskGraphV4, ResearchTaskGraphV5
                        workflow = (ResearchTaskGraphV5 if execution.snapshot()["profile"]["profile_id"] == "research-v5"
                            else ResearchTaskGraphV4 if execution.snapshot()["profile"]["profile_id"] == "research-v4"
                            else ResearchTaskGraph)(workflow)
                    await _run_graph_to_completion(workflow.run, graph_stopping.set)
                return {"status": "processed", "task_id": str(execution.task_id), "run_id": str(execution.run_id)}
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                graph_stopping.set()
            private_runtime_observation("agent_worker_failure","run_once","failed",
                task_id=execution.task_id,elapsed_seconds=time.monotonic()-started,error=exc)
            leaf = exc
            while isinstance(leaf, BaseExceptionGroup) and leaf.exceptions:
                leaf = leaf.exceptions[0]
            code = leaf.code if isinstance(leaf, AgentError) else type(leaf).__name__
            # A lost owner cannot finish; preserve the dispatch records
            # for recovery without replacing the original exception.
            with suppress(Exception):
                execution.finish("failed", error=code)
            if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
                raise
            if graph_stopping.is_set():
                raise asyncio.CancelledError from exc
            return {"status": "failed", "task_id": str(execution.task_id), "error_code": code}
        finally:
            if provider is not None:
                try:
                    await provider.close()
                except BaseException as error:
                    if graph_stopping.is_set():
                        raise asyncio.CancelledError from error
                    raise


async def main_async(config: WorkerConfig, once: bool, *, drain_on_signal: bool = False) -> None:
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    previous_handlers = {}

    def request_drain() -> None:
        if not stopping.is_set():
            stopping.set()
            print(json.dumps({"status": "draining"}), flush=True)  # noqa: T201

    try:
        if drain_on_signal:
            # Do not cancel run_once: its graph runs in a thread and must finish
            # before its Saver lock, provider and MCP clients can be released.
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.getsignal(signum)
                loop.add_signal_handler(signum, request_drain)
        while not stopping.is_set():
            result = await run_once(config)
            print(json.dumps(result), flush=True)  # noqa: T201 -- bounded worker operational receipt
            if once:
                return
            if result["status"] == "idle":
                with suppress(TimeoutError):
                    await asyncio.wait_for(stopping.wait(), timeout=1)
    finally:
        for signum, previous in previous_handlers.items():
            loop.remove_signal_handler(signum)
            signal.signal(signum, previous)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--drain-on-signal", action="store_true",
                        help="Linux service: stop claiming after SIGTERM/SIGINT and finish the active command")
    args = parser.parse_args()
    config = WorkerConfig.model_validate_json(args.config.read_text(encoding="utf-8"), strict=True)
    if config.mode != "live":
        raise AgentError("fixture_worker_requires_explicit_test_transport")
    asyncio.run(main_async(config, args.once, drain_on_signal=args.drain_on_signal))


if __name__ == "__main__":
    main()
