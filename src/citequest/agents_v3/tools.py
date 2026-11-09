"""Real official MCP stdio client. A bounded tool call is a durable business attempt."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from uuid import UUID, uuid4

from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

from .capabilities import FREE_READ_EFFECT, FREE_READ_TOOLS, LOCAL_COMPUTE_EFFECT, LOCAL_COMPUTE_TOOLS
from .chunk_tool_policy import uses_typed_chunk_tools
from .contracts import AgentError, canonical, digest
from .database import Execution
from .exception_metadata import private_runtime_phase

TOOL_NAMES = {"search_evidence", "read_evidence"}
WORKSET_TOOL_NAMES = {"plan_workset", "read_source_window", "verify_evidence_batch"}
TOOL_ARGUMENTS = {"search_evidence": {"query", "limit"}, "read_evidence": {"ref_id"},
    "plan_workset": set(), "read_source_window": {"window_id"}, "verify_evidence_batch": {"ref_ids"},
    "search_chunk_evidence": {"query", "limit"}, "navigate_issued_evidence": {"query", "ref_ids"}}


def require_atomic_stance_review(execution: Execution, binding: dict) -> None:
    """Before any tool reservation, bind both settled model roles and verdict.

    Graph ordering is complemented by this gateway check. Merely possessing a
    planner attempt ID cannot bypass stance-v2's independent pre-retrieval review.
    Stored model bodies remain private and are never added to public tool events.
    """
    from citequest.research_v3.typed_feedback import strict_object

    from .claim_plan import PlanReview, validate_plan_review
    from .provider import output_text
    from .stance_v2 import StancePlanV2

    try:
        planner_id = UUID(binding["planner_attempt_id"])
        reviewer_id = UUID(binding["plan_review_attempt_id"])
        if planner_id == reviewer_id:
            raise ValueError("roles must have distinct attempts")
        with execution.database.connect() as db:
            rows = db.execute("""SELECT id,kind,status,request_hash,usage,error_code,request,response,logical_key
                FROM cq_app.agent_attempts WHERE task_id=%s AND id=ANY(%s)""",
                (execution.task_id, [planner_id, reviewer_id])).fetchall()
        by_id = {str(row["id"]): row for row in rows}
        revisions = {}
        prefix = str(execution.command["id"]) + ":qa5:"
        for identifier, role, expected_hash in ((planner_id, "plan", binding["planner_request_sha256"]),
                (reviewer_id, "plan_review", binding["plan_review_request_sha256"])):
            row = by_id[str(identifier)]
            match = re.fullmatch(re.escape(prefix + role + ":") + r"(0|[1-9][0-9]{0,5})", row["logical_key"])
            if (row["kind"] != "model" or row["status"] != "settled" or row["usage"] is None
                    or row["error_code"] is not None or row["request_hash"] != expected_hash
                    or digest(row["request"]) != expected_hash or match is None
                    or row["request"]["text"]["format"]["name"] != "citequest_qa5_" + role):
                raise ValueError("unsettled or mismatched planning role")
            revisions[role] = int(match[1])
        if revisions["plan_review"] < revisions["plan"]:
            raise ValueError("independent review predates the current plan revision")
        plan = StancePlanV2.model_validate(strict_object(output_text(by_id[str(planner_id)]["response"])), strict=True)
        if plan.claim_plan is None or digest(plan.model_dump()) != binding["plan_sha256"]:
            raise ValueError("tool plan differs from settled planner")
        claim_sha = digest(plan.claim_plan.model_dump())
        review_row = by_id[str(reviewer_id)]
        prompt = json.loads(review_row["request"]["input"][0]["content"])
        if (claim_sha != binding["claim_plan_sha256"] or prompt.get("claim_plan_sha256") != claim_sha
                or prompt.get("claim_plan") != plan.claim_plan.model_dump()
                or prompt.get("requirements_candidate_sha256") != binding["candidate_sha256"]):
            raise ValueError("independent review is not bound to this atomic plan")
        validate_plan_review(plan.claim_plan,
            PlanReview.model_validate(strict_object(output_text(review_row["response"])), strict=True))
    except (AgentError, KeyError, TypeError, ValueError, AttributeError):
        raise AgentError("stance2_independent_plan_review_provenance_invalid") from None


def require_supplement_planning(execution: Execution, binding: dict, supplement: dict,
                                name: str, arguments: dict) -> None:
    """Bind a corrective query to its own known planner, never the old plan."""
    from citequest.research_v3.typed_feedback import strict_object

    from .product_profiles import qa_retrieval_protocol
    from .provider import output_text
    from .qa_adaptive_contracts import PROTOCOL, SUPPLEMENT_SCHEMA_NAME, validate_supplement_plan

    try:
        if qa_retrieval_protocol(execution.snapshot()) != PROTOCOL or name != "search_chunk_evidence":
            raise ValueError("supplemental planning is not enabled for this operation")
        required = {"attempt_id", "request_sha256", "plan_sha256", "candidate_sha256", "base_key"}
        if not isinstance(supplement, dict) or set(supplement) != required:
            raise ValueError("complete supplemental receipt required")
        if (any(not isinstance(supplement[key], str) or re.fullmatch(r"[a-f0-9]{64}", supplement[key]) is None
                for key in ("request_sha256", "plan_sha256", "candidate_sha256"))
                or supplement["candidate_sha256"] != binding["candidate_sha256"]):
            raise ValueError("supplemental goal identity changed")
        prefix = re.escape(str(execution.command["id"]) + ":qa5:supplement_plan:")
        if not isinstance(supplement["base_key"], str) or re.fullmatch(prefix + r"(0|[1-9][0-9]*)", supplement["base_key"]) is None:
            raise ValueError("supplemental planner belongs to another command")
        identifier = UUID(supplement["attempt_id"])
        with execution.database.connect() as db:
            row = db.execute("""SELECT id,kind,status,request_hash,usage,error_code,request,response,logical_key
                FROM cq_app.agent_attempts WHERE task_id=%s AND id=%s""", (execution.task_id, identifier)).fetchone()
        if (not row or row["kind"] != "model" or row["status"] != "settled" or row["usage"] is None
                or row["error_code"] is not None or row["request_hash"] != supplement["request_sha256"]
                or digest(row["request"]) != supplement["request_sha256"]
                or row["logical_key"] != execution.effective_model_source_key(supplement["base_key"])
                or row["request"]["text"]["format"]["name"] != SUPPLEMENT_SCHEMA_NAME):
            raise ValueError("supplemental planner receipt is not settled and bound")
        prompt = strict_object(row["request"]["input"][0]["content"])
        contract = prompt["adaptive_retrieval_contract"]
        if contract["candidate_sha256"] != binding["candidate_sha256"]:
            raise ValueError("supplemental planner changed reviewed requirements")
        plan = validate_supplement_plan(strict_object(output_text(row["response"])),
            candidate_sha256=contract["candidate_sha256"], evidence_sha256=contract["evidence_sha256"],
            target_requirement_ids=contract["target_requirement_ids"], existing_ref_ids=contract["existing_ref_ids"],
            allowed_limits=contract["allowed_limits"])
        if (plan.action != "retrieve" or digest(plan.model_dump()) != supplement["plan_sha256"]
                or arguments != plan.queries[0].model_dump()):
            raise ValueError("tool arguments differ from the settled supplemental plan")
    except (AgentError, KeyError, TypeError, ValueError, AttributeError):
        raise AgentError("qa_adaptive_tool_planning_provenance_invalid") from None


class MCPTools:
    def __init__(self, execution: Execution, config_file: Path, *, lazy: bool = False) -> None:
        self.execution, self.config_file = execution, config_file
        self.session = None
        self.catalog: dict[str, dict] = {}
        self.lazy = lazy
        self._owner: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] | None = None
        self._stop = asyncio.Event()
        self._closing = False
        from citequest.research_v3.decisionrag_task import load_task_config
        trusted = json.loads(config_file.read_bytes())
        path, sha = trusted.get("decisionrag_config_file"), trusted.get("decisionrag_config_sha256")
        if (path is None) != (sha is None):
            raise AgentError("decisionrag_trusted_configuration_required")
        self.navigation_config = load_task_config(Path(path), sha) if path is not None else None
        self.navigation_config_sha256 = sha
        from citequest.research_v3.rerank import load_config as load_rerank_config
        rerank_path, rerank_sha = trusted.get("rerank_config_file"), trusted.get("rerank_config_sha256")
        if (rerank_path is None) != (rerank_sha is None):
            raise AgentError("rerank_trusted_configuration_required", 503)
        self.rerank_config = load_rerank_config(Path(rerank_path), rerank_sha) if rerank_path is not None else None
        self.rerank_config_sha256 = rerank_sha

    @property
    def rerank_enabled(self) -> bool:
        from .product_profiles import retrieval_rerank_protocol
        task = self.execution.snapshot()
        requested = retrieval_rerank_protocol(task) is not None
        if requested and (self.rerank_config is None
                or self.rerank_config_sha256 != task["profile"]["rerank_config_sha256"]):
            raise AgentError("rerank_runtime_configuration_not_bound", 503)
        return requested

    @property
    def decisionrag_enabled(self) -> bool:
        from citequest.research_v3.decisionrag_task import task_requests_navigation
        from .product_profiles import evidence_navigation_protocol
        task = self.execution.snapshot()
        requested = task_requests_navigation(task)
        if requested and self.navigation_config is None:
            raise AgentError("decisionrag_runtime_not_configured")
        if requested and self.navigation_config.navigation_protocol != evidence_navigation_protocol(task):
            raise AgentError("decisionrag_task_runtime_protocol_mismatch")
        return requested

    @property
    def decisionrag_frontier_limit(self) -> int:
        return self.navigation_config.max_passages if self.decisionrag_enabled else 0

    async def navigate_issued(self, query: str, ref_ids: list[str], *, logical_key: str,
                              planning_context: dict | None = None) -> dict:
        if not self.decisionrag_enabled:
            raise AgentError("decisionrag_Task_profile_not_enabled")
        if (type(ref_ids) is not list or not 1 <= len(ref_ids) <= self.decisionrag_frontier_limit
                or len(set(ref_ids)) != len(ref_ids)):
            raise AgentError("decisionrag_complete_frontier_capacity")
        await self.check_scope()  # Also reauthorizes a durable settled replay.
        result = await self.call("navigate_issued_evidence", {"query": query, "ref_ids": ref_ids},
            logical_key=logical_key, planning_context=planning_context)
        await self.check_scope()
        return result

    async def __aenter__(self) -> MCPTools:
        if self.lazy:
            return self
        try:
            return await self.open()
        except BaseException:
            await self._close()
            raise

    async def open(self) -> MCPTools:
        # AnyIO task groups inside the official stdio/session contexts must be
        # entered and exited by the SAME Task. Lazy graph calls can arrive via
        # run_coroutine_threadsafe from different consumer Tasks; only this
        # lifetime owner may manage those contexts. No transport starts until
        # open() is actually needed (missing-input research stays source-free).
        if self._closing:
            raise AgentError("mcp_transport_closed")
        if self._owner is None:
            self._ready = asyncio.get_running_loop().create_future()
            # A cancelled sole consumer may leave no waiter for a later startup
            # failure. Observe the Future to avoid an unhandled-Future warning;
            # await(open) and owner.result() still propagate the same failure.
            self._ready.add_done_callback(lambda value: None if value.cancelled() else value.exception())
            self._owner = asyncio.create_task(self._own_transport(), name="citequest-mcp-lifetime")
        if self._owner.done():
            await self._owner
            raise AgentError("mcp_transport_closed")
        assert self._ready is not None
        await asyncio.shield(self._ready)
        return self

    async def _own_transport(self) -> None:
        assert self._ready is not None
        try:
            async with AsyncExitStack() as stack:
                await self._connect(stack)
                await self._check_scope_open_session()
                self._ready.set_result(None)
                await self._stop.wait()
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
            raise
        finally:
            self.session = None

    async def _connect(self, stack: AsyncExitStack) -> None:
        # Deliberately do not inherit model keys, cloud credentials, proxies, or arbitrary user env.
        environment = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP") if key in os.environ}
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        parameters = StdioServerParameters(command=sys.executable,
            args=["-m", "citequest.agents_v3.mcp_server", "--config", str(self.config_file)], env=environment)
        with private_runtime_phase("mcp_client_startup","stdio_open",task_id=self.execution.task_id):
            reader, writer = await stack.enter_async_context(stdio_client(parameters))
        with private_runtime_phase("mcp_client_startup","session_enter",task_id=self.execution.task_id):
            self.session = await stack.enter_async_context(ClientSession(reader, writer))
        with private_runtime_phase("mcp_client_startup","initialize",task_id=self.execution.task_id):
            await asyncio.wait_for(self.session.initialize(), 10)
        with private_runtime_phase("mcp_client_startup","list_tools",task_id=self.execution.task_id):
            listing = await asyncio.wait_for(self.session.list_tools(), 10)
        with private_runtime_phase("mcp_client_startup","catalog_validation",task_id=self.execution.task_id):
            task = self.execution.snapshot()
            kind = task["kind"]
            expected = ({"search_chunk_evidence", "read_evidence", "verify_evidence_batch"}
                if uses_typed_chunk_tools(task) else
                TOOL_NAMES | WORKSET_TOOL_NAMES if kind == "research" else WORKSET_TOOL_NAMES if kind in {"summary", "compare"} else TOOL_NAMES)
            if self.decisionrag_enabled:
                expected.add("navigate_issued_evidence")
            if {tool.name for tool in listing.tools} != expected:
                raise AgentError("mcp_tool_catalog_mismatch")
            self.catalog = {tool.name: tool.inputSchema for tool in listing.tools}
            for schema in self.catalog.values():
                Draft202012Validator.check_schema(schema)

    async def check_scope(self) -> dict:
        await self.open()
        return await self._check_scope_open_session()

    async def _check_scope_open_session(self) -> dict:
        assert self.session is not None
        operation = str(uuid4())
        self.execution.scope_event("scope_check_started", {"resource_operation_id": operation,
            "tool_call": False, "model_call": False, "resource": "citequest://task/scope"})
        task = self.execution.snapshot()
        try:
            preflight = await asyncio.wait_for(self.session.read_resource(AnyUrl("citequest://task/scope")),
                min(30, max(0.001, task["profile"]["active_seconds"] - task["elapsed"])))
        except BaseException:
            self.execution.scope_event("scope_check_failed", {"resource_operation_id": operation,
                "error_code": "scope_check_transport_failed", "tool_call": False, "automatic_retry": False})
            raise
        if len(preflight.contents) != 1 or not hasattr(preflight.contents[0], "text"):
            raise AgentError("scope_preflight_structure_invalid", 503)
        status = json.loads(preflight.contents[0].text)
        self.execution.scope_event("scope_checked", {"resource_operation_id": operation, "tool_call": False,
            "model_call": False, "status": status.get("status"), "error_code": status.get("error_code"),
            "source_reads": status.get("source_reads"), "pinned_refs_checked": status.get("pinned_refs_checked", 0),
            "manifest_documents_replanned": status.get("manifest_documents_replanned", 0),
            "historical_refs_excluded_from_current_literature_role": status.get("historical_refs_excluded_from_current_literature_role", 0)})
        if status.get("status") != "ready":
            code = status.get("error_code", "scope_not_ready")
            if not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,120}", code):
                code = "scope_not_ready"
            raise AgentError(code, 503)
        if status.get("brief_sha256") != digest(task["brief"]):
            raise AgentError("scope_changed_after_preflight", 409)
        if status.get('catalog_scope_policy') != task['profile'].get('catalog_scope_policy'):
            raise AgentError('task_catalog_scope_policy_not_acknowledged', 503)
        return status

    async def __aexit__(self, *exc: object) -> None:
        await self._close()

    async def _close(self) -> None:
        self._closing = True
        self._stop.set()
        if self._owner is None:
            return
        # Consumer cancellation must not cancel the context-owning Task halfway
        # through official subprocess/session cleanup. Retain cancellation and
        # propagate it after the lifetime owner has finished.
        cancelled = False
        failure: BaseException | None = None
        while not self._owner.done():
            try:
                await asyncio.shield(self._owner)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException as error:
                failure = error
                break
        if failure is None:
            try:
                self._owner.result()
            except BaseException as error:
                failure = error
        if cancelled:
            raise asyncio.CancelledError from failure
        if failure is not None:
            raise failure

    def provider_tools(self) -> list[dict]:
        descriptions = {
            "search_evidence": "Search permitted literal evidence for this research Task.",
            "read_evidence": "Read a permitted exact evidence reference returned by this Task search.",
            "plan_workset": "Plan all available literal source windows in the server-owned Task workset or explicit draft subset.",
            "read_source_window": "Read one exact server-issued complete available source window with current authorization and version checks.",
            "verify_evidence_batch": "Reauthorize up to64 Task-issued source-window pins and return current literal hashes without source text.",
            "search_chunk_evidence": "Retrieve permitted pinned source chunks with explicit per-document candidate allocation.",
            "navigate_issued_evidence": "Rank a bounded frontier of existing Task-issued refs with two real local Laya probability decisions; no text clipping or authority expansion.",
        }
        result = []
        for name, schema in sorted(self.catalog.items()):
            if name == "navigate_issued_evidence" or name == "search_chunk_evidence" and self.rerank_enabled:
                schema = json.loads(json.dumps(schema))
                schema.get("properties", {}).pop("remaining_ms", None)
                schema["required"] = [field for field in schema.get("required", []) if field != "remaining_ms"]
            result.append({"type": "function", "name": name, "description": descriptions[name], "parameters": schema})
        return result

    async def call(self, name: str, arguments: dict, *, logical_key: str, planning_context: dict | None = None) -> dict:
        await self.open()
        if name not in self.catalog or self.session is None:
            raise AgentError("mcp_tool_not_allowed")
        if not isinstance(arguments, dict) or set(arguments) - TOOL_ARGUMENTS[name]:
            raise AgentError("mcp_arguments_not_allowed")
        navigation = name == "navigate_issued_evidence"
        reranking = name == "search_chunk_evidence" and self.rerank_enabled
        if navigation and not self.decisionrag_enabled:
            raise AgentError("decisionrag_Task_profile_not_enabled")
        Draft202012Validator(self.catalog[name]).validate({**arguments, "remaining_ms": 1} if navigation or reranking else arguments)
        if name not in FREE_READ_TOOLS | LOCAL_COMPUTE_TOOLS:
            raise AgentError("unclassified_tool_effect")
        request = {"name": name, "arguments": arguments,
            "effect": LOCAL_COMPUTE_EFFECT if navigation or reranking else FREE_READ_EFFECT}
        if navigation:
            request["local_compute"] = {"navigation_protocol": self.navigation_config.navigation_protocol,
                "max_forward_calls": self.navigation_config.max_forward_calls,
                "configuration_sha256": self.navigation_config_sha256,
                "external_API_cash_ncny": 0, "deadline_source": "Execution.mark_sent_remaining_active_time"}
            if self.navigation_config.navigation_protocol == "laya2":
                request["local_compute"].update(maximum_frontier_refs=self.navigation_config.max_passages,
                    max_batches_per_level=self.navigation_config.max_batches_per_level,
                    probability_comparison="batch_screening_then_shared_final_rescore",
                    physical_compute_cost_not_measured=True)
        if reranking:
            request["local_compute"] = {"rerank_protocol": "tei-cross-encoder1", "max_rerank_HTTP_requests": 1,
                "maximum_pair_scores": self.rerank_config.maximum_candidates,
                "configuration_sha256": self.rerank_config_sha256, "external_API_cash_ncny": 0,
                "deadline_source": "Execution.mark_sent_remaining_active_time"}
        task = self.execution.snapshot()
        qa5 = uses_typed_chunk_tools(task)
        if qa5:
            base_planning = dict(planning_context) if isinstance(planning_context, dict) else planning_context
            supplemental = isinstance(base_planning, dict) and "supplement_planning" in base_planning
            supplement = base_planning.pop("supplement_planning", None) if isinstance(base_planning, dict) else None
            required = {"planner_attempt_id", "planner_request_sha256", "candidate_sha256", "plan_sha256"}
            if task["profile"]["profile_id"] in {"stance-v2", "stance-v3"}:
                required |= {"plan_review_attempt_id", "plan_review_request_sha256", "claim_plan_sha256"}
            if not isinstance(base_planning, dict) or set(base_planning) != required:
                raise AgentError("qa5_tool_planning_provenance_required")
            if any(not isinstance(base_planning[key], str) or re.fullmatch(r"[a-f0-9]{64}", base_planning[key]) is None
                   for key in required - {"planner_attempt_id", "plan_review_attempt_id"}):
                raise AgentError("qa5_tool_planning_provenance_invalid")
            try:
                planner_id = UUID(base_planning["planner_attempt_id"])
            except (ValueError, TypeError):
                raise AgentError("qa5_tool_planning_provenance_invalid") from None
            with self.execution.database.connect() as db:
                parent = db.execute("""SELECT kind,status,request_hash,usage,error_code,
                    request->'text'->'format'->>'name' AS role FROM cq_app.agent_attempts
                    WHERE id=%s AND task_id=%s""", (planner_id,self.execution.task_id)).fetchone()
            if (not parent or parent["kind"] != "model" or parent["status"] != "settled" or parent["usage"] is None
                    or parent["error_code"] is not None or parent["request_hash"] != base_planning["planner_request_sha256"]
                    or parent["role"] != "citequest_qa5_plan"):
                raise AgentError("qa5_tool_planning_attempt_not_settled_or_bound")
            if task["profile"]["profile_id"] in {"stance-v2", "stance-v3"}:
                require_atomic_stance_review(self.execution, base_planning)
            if supplemental:
                require_supplement_planning(self.execution, base_planning, supplement, name, arguments)
            request["structured_planning"] = {**planning_context, "vendor_call_id":"not_applicable_structured_planning"}
        elif planning_context is not None:
            raise AgentError("planning_provenance_not_enabled_for_legacy_tool_protocol")
        attempt = self.execution.reserve(logical_key, request, kind="tool")
        if attempt["status"] == "settled":
            if attempt["response"].get("status") == "error":
                raise AgentError(attempt["response"]["error_code"])
            return attempt["response"]
        settled = False
        received = False
        body = None
        try:
            if qa5:
                self.execution.scope_event("tool_plan_bound", {"tool_execution_id":str(attempt["id"]),
                    **planning_context,"vendor_call_id":"not_applicable_structured_planning",
                    "name":name,"arguments_sha256":digest(arguments)})
            seconds = self.execution.mark_sent(attempt["id"])
            wire_arguments = arguments
            if navigation:
                wire_arguments = {**arguments, "remaining_ms": max(1, min(25000,
                    self.navigation_config.active_seconds * 1000, int(seconds * 1000)))}
                self.execution.progress("decisionrag_navigation", "started")
            if reranking:
                wire_arguments = {**arguments, "remaining_ms": max(1, min(30000, int(seconds * 1000)))}
                self.execution.progress("cross_encoder_rerank", "started")
            result = await asyncio.wait_for(self.session.call_tool(name, wire_arguments), min(seconds, 30))
            received = True
            if result.isError:
                texts = [item.text for item in result.content if item.type == "text"]
                candidate = texts[0].rsplit(": ", 1)[-1] if texts else ""
                raise AgentError(candidate if re.fullmatch(r"[a-z][a-z0-9_]{1,120}", candidate) else "mcp_tool_execution_error")
            body = result.structuredContent
            if body is None:
                texts = [item.text for item in result.content if item.type == "text"]
                if len(texts) != 1:
                    raise AgentError("mcp_result_structure_invalid")
                body = json.loads(texts[0])
            research_v5 = task["kind"] == "research" and task["profile"].get("profile_id") in {"research-v5", "research-v6"}
            if not isinstance(body, dict) or len(canonical(body)) > (131072 if qa5 or research_v5 else 32000):
                raise AgentError("mcp_result_byte_limit")
            if reranking:
                from citequest.research_v3.rerank import validate_receipt
                validate_receipt(body, arguments["query"], self.rerank_config, self.rerank_config_sha256)
            if navigation:
                usage = body.get("local_usage") or {}
                refs, trace = body.get("evidence_ref_ids"), body.get("trace")
                if self.navigation_config.navigation_protocol == "laya2":
                    from citequest.research_v3.decisionrag_task import validate_task_receipt
                    validate_task_receipt(self.navigation_config, body, arguments["ref_ids"])
                    if body["status"] == "error":
                        code = body["error_code"]
                        # A response observation does not establish complete
                        # native usage. Preserve partial receipts privately.
                        received = False
                        self.execution.observe_tool_receipt(attempt["id"], body)
                        if body.get("unknown_outcome") is True:
                            self.execution.unknown(attempt["id"], code)
                        else:
                            self.execution.settle(attempt["id"], body, None)
                        settled = True
                        raise AgentError(code)
                elif (body.get("status") != "ok" or body.get("navigation_protocol") != "laya1"
                        or body.get("input_frontier_ref_ids") != arguments["ref_ids"]
                        or not isinstance(refs, list) or not refs or len(set(refs)) != len(refs)
                        or not set(refs) <= set(arguments["ref_ids"])
                        or body.get("decision_calls") != 2 or not isinstance(trace, list) or len(trace) != 2
                        or [item.get("stage") for item in trace] != ["documents", "passages"]
                        or usage.get("observed_forward_calls") != 2 or usage.get("source_tokens_dropped") != 0
                        or usage.get("external_API_cash_ncny") != 0
                        or type(usage.get("input_tokens")) is not int
                        or not 1 <= usage["input_tokens"] <= self.navigation_config.local_input_token_limit):
                    raise AgentError("decisionrag_MCP_receipt_contract_invalid")
            self.execution.settle(attempt["id"], body, None)
            settled = True
            self.execution.snapshot()
            if navigation:
                self.execution.progress("decisionrag_navigation", "completed")
            if reranking:
                self.execution.progress("cross_encoder_rerank", "completed")
            return body
        except BaseException as exc:
            if not settled:
                if navigation and self.navigation_config.navigation_protocol == "laya2":
                    code = exc.code if isinstance(exc, AgentError) else type(exc).__name__
                    try:
                        if received and isinstance(body, dict):
                            self.execution.observe_tool_receipt(attempt["id"], body)
                    finally:
                        self.execution.unknown(attempt["id"], code)
                elif received:
                    code = exc.code if isinstance(exc, AgentError) else "mcp_terminal_result_invalid"
                    self.execution.settle(attempt["id"], {"status": "error", "error_code": code}, None)
                else:
                    self.execution.unknown(attempt["id"], type(exc).__name__)
            raise

