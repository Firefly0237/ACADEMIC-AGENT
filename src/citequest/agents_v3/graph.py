"""One explicit LangGraph Graph API, with official persistent Saver and Store."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from functools import wraps
from pathlib import Path
from typing import Any

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from citequest.research_v3.contracts import (
    REVIEW_POLICY,
    SYSTEM_POLICY,
    ResearchAnswer,
    Review,
    render_text,
    validate_literal,
    validate_review,
)

from .context import fit_context
from .contracts import POLICY_VERSION, AgentError, ResearchBrief, canonical, digest
from .database import Execution
from .memory import MemoryService
from .provider import MODEL, POLICY_MARKER, ResponsesProvider, output_items, output_text
from .stance import StanceNodes
from .tools import MCPTools
from .workset import WorksetNodes, WorksetState


class State(WorksetState, total=False):
    data: dict  # Workset5 compiled subgraph's private checkpoint state.
    qa5: dict
    requirements_state: dict  # Accepted current goals persist across commands.
    workset_requirements: dict  # Reviewed interpretation, committed with public answer.
    task_id: str
    command_id: str
    question: str
    tool_request: dict
    tool_outputs: list[dict]
    validated_tool_calls: list[dict]
    evidence: list[dict]
    answer: dict | None
    result: dict
    claim_plan: dict
    plan_review: dict
    claim_index: int
    claim_results: list[dict]
    claim_reviews: list[dict]
    claim_coverage: list[dict]
    current_claim_evidence: list[dict]
    current_claim_candidate: dict
    claim_skip_reason: str | None
    stance_rejected: bool
    plan_context_manifest: dict
    qa4_revision: int
    qa4_feedback: dict | None
    qa4_original_claims: dict
    qa4_preserved_claims: dict
    qa4_affected_claims: list[str]
    qa4_audit: list[dict]
    qa4_repair_target: str
    qa4_terminal: bool
    qa4_author_raw: str
    qa4_baseline_raw: str
    qa4_review_raw: str
    qa4_manifest: dict
    qa4_author_valid: bool
    qa4_review: dict
    qa4_grid: dict
    qa4_grid_binding: dict
    qa4_accepted: bool


class AgentGraph:
    def __init__(
        self,
        execution: Execution,
        provider: ResponsesProvider,
        tools: MCPTools,
        memory: MemoryService,
        skill_root: Path,
    ) -> None:
        self.execution, self.provider, self.tools, self.memory = execution, provider, tools, memory
        self.skill_root = skill_root.resolve()
        self.loop = asyncio.get_running_loop()
        self.saver = PostgresSaver(execution.saver_connection)
        self.stance = StanceNodes(self)
        self.workset = WorksetNodes(self)
        from .qa_revision import RevisionQANodes
        self.qa4 = RevisionQANodes(self)
        from .workset_v5 import Workset5
        self.workset5 = Workset5(self)
        from .qa_v5 import QA5
        self.qa5 = QA5(self)
        from .stance_v2 import StanceV2
        self.stance2 = StanceV2(self)
        from .workset_v6 import Workset6
        self.workset6 = Workset6(self)
        from .qa_v6 import QA6
        from .product_profiles import qa_execution_protocol
        qa_protocol = qa_execution_protocol(self.execution.snapshot())
        if self.execution.snapshot()["profile"]["profile_id"] == "qa-v9":
            from .product_profiles import qa_content_protocol, qa_retrieval_protocol
            if qa_content_protocol(self.execution.snapshot()) == "isolated1":
                if qa_retrieval_protocol(self.execution.snapshot()) == "adaptive1":
                    from .qa_adaptive_retrieval import QAAdaptive
                    self.qa6 = QAAdaptive(self)
                else:
                    from .qa_separated import QASeparated
                    self.qa6 = QASeparated(self)
            else:
                from .qa_v9 import QA9
                self.qa6 = QA9(self)
        elif self.execution.snapshot()["profile"]["profile_id"] == "qa-v8":
            from .qa_v8 import QA8
            self.qa6 = QA8(self)
        elif self.execution.snapshot()["profile"]["profile_id"] == "qa-v7":
            from .qa_v7 import QA7
            self.qa6 = QA7(self)
        elif qa_protocol == "grounded3":
            from .qa_verified_dialogue import QADialogueVerified
            self.qa6 = QADialogueVerified(self)
        elif qa_protocol == "grounded2":
            from .qa_audited_dialogue import QADialogueAudited
            self.qa6 = QADialogueAudited(self)
        elif qa_protocol == "grounded1":
            from .qa_dialogue import QADialogueGrounded
            self.qa6 = QADialogueGrounded(self)
        else:
            self.qa6 = QA6(self)
        from .stance_v3 import StanceV3
        self.stance3 = StanceV3(self)
        from .workset_v7 import Workset7
        self.workset7 = Workset7(self)
        builder = StateGraph(State)
        self.workset.register(builder, self.observed_node)
        self.qa4.register(builder, self.observed_node)
        # Shared state keys permit direct compiled-subgraph registration. This
        # lets official get_state(subgraphs=True) discover its Saver namespace;
        # individual trusted child phases already emit public progress events.
        builder.add_node("workset5", self.workset5.workflow)
        builder.add_node("qa5", self.qa5.workflow)
        builder.add_node("stance2", self.stance2.workflow)
        builder.add_node("workset6", self.workset6.workflow)
        builder.add_node("qa6", self.qa6.workflow)
        builder.add_node("stance3", self.stance3.workflow)
        builder.add_node("workset7", self.workset7.workflow)
        for name in ("clarify", "request_evidence", "execute_tools", "answer", "independent_review", "validate"):
            builder.add_node(name, self.observed_node(name, getattr(self, name)))
        for name, node in (
            ("plan_claims", self.stance.plan),
            ("review_claim_plan", self.stance.review_plan),
            ("retrieve_claim", self.stance.retrieve),
            ("skip_claim", self.stance.skip),
            ("write_claim", self.stance.write),
            ("review_claim", self.stance.review),
            ("assemble_stance", self.stance.assemble),
        ):
            builder.add_node(name, self.observed_node(name, node))
        builder.add_edge(START, "clarify")
        builder.add_conditional_edges(
            "clarify",
            lambda state: (
                "stance3"
                if self.execution.snapshot()["kind"] == "stance" and self.execution.snapshot()["profile"]["profile_id"] == "stance-v3"
                else "qa6"
                if self.execution.snapshot()["profile"]["profile_id"] in {"qa-v6", "qa-v7", "qa-v8", "qa-v9"}
                else "workset7"
                if self.execution.snapshot()["kind"] in {"summary", "compare"} and self.execution.snapshot()["profile"]["profile_id"] == "workset-v7"
                else
                "stance2"
                if self.execution.snapshot()["kind"] == "stance" and self.execution.snapshot()["profile"]["profile_id"] == "stance-v2"
                else "plan_claims"
                if self.execution.snapshot()["kind"] == "stance"
                else "qa5"
                if self.execution.snapshot()["profile"]["profile_id"] == "qa-v5"
                else "workset6"
                if self.execution.snapshot()["kind"] in {"summary", "compare"} and self.execution.snapshot()["profile"]["profile_id"] == "workset-v6"
                else "workset5"
                if self.execution.snapshot()["kind"] in {"summary", "compare"} and self.execution.snapshot()["profile"]["profile_id"] == "workset-v5"
                else "workset_plan"
                if self.execution.snapshot()["kind"] in {"summary", "compare"}
                else "request_evidence"
            ),
            ["stance3", "qa6", "workset7", "stance2", "plan_claims", "qa5", "workset6", "workset5", "workset_plan", "request_evidence"],
        )
        builder.add_edge("request_evidence", "execute_tools")
        builder.add_conditional_edges("execute_tools", lambda state: "qa4_initialize"
            if self.execution.snapshot()["profile"]["profile_id"] == "qa-v4" else "answer", ["qa4_initialize", "answer"])
        builder.add_edge("answer", "independent_review")
        builder.add_edge("independent_review", "validate")
        builder.add_edge("plan_claims", "review_claim_plan")
        builder.add_conditional_edges(
            "review_claim_plan", self.stance.route_next, ["retrieve_claim", "assemble_stance"]
        )
        builder.add_conditional_edges("retrieve_claim", self.stance.route_retrieved, ["skip_claim", "write_claim"])
        builder.add_conditional_edges("skip_claim", self.stance.route_next, ["retrieve_claim", "assemble_stance"])
        builder.add_edge("write_claim", "review_claim")
        builder.add_conditional_edges("review_claim", self.stance.route_next, ["retrieve_claim", "assemble_stance"])
        builder.add_edge("assemble_stance", "validate")
        builder.add_edge("validate", END)
        builder.add_edge("workset_validate", END)
        builder.add_edge("workset5", END)
        builder.add_edge("qa5", END)
        builder.add_edge("stance2", END)
        builder.add_edge("workset6", END)
        builder.add_edge("qa6", END)
        builder.add_edge("stance3", END)
        builder.add_edge("workset7", END)
        self.workflow = builder.compile(checkpointer=self.saver)
        task = self.execution.snapshot()
        recursion_limit = (
            4 * task["profile"]["model_calls"] + 3 * len(task["brief"]["selected_papers"]) + 16
            if task["kind"] in {"summary", "compare"}
            else 4 * task["profile"]["model_calls"] + 24 if task["profile"]["profile_id"] in {"qa-v5", "qa-v6", "qa-v7", "qa-v8", "qa-v9", "stance-v2", "stance-v3"} else 48
        )
        self.config = {"configurable": {"thread_id": str(execution.task_id)}, "recursion_limit": recursion_limit}

    def observed_node(self, phase: str, node: Callable[[dict], dict]) -> Callable[[dict], dict]:
        """Public attempt milestones, not private prompts, sources or chain of thought.

        A recovered node may emit another attempt. Provider/tool logical keys
        remain unchanged, so progress is never used as a billing deduplication key.
        """

        @wraps(node)
        def observed(state: dict) -> dict:
            self.execution.progress(phase, "started")
            try:
                result = node(state)
            except GraphInterrupt:
                self.execution.progress(phase, "waiting")
                raise
            except Exception:
                self.execution.progress(phase, "failed")
                raise
            self.execution.progress(phase, "completed")
            return result

        return observed

    def user_turns(self) -> list[dict]:
        """User questions survive answer invalidation and are not generated summaries."""
        task = self.execution.snapshot()
        with self.execution.database.connect() as db:
            commands = db.execute(
                "SELECT seq,kind,payload FROM cq_app.agent_commands WHERE task_id=%s AND owner_id=%s AND kind IN ('start','resume','followup') ORDER BY seq",
                (self.execution.task_id, task["owner_id"]),
            ).fetchall()
        return [
            {
                "turn_id": "U" + str(item["seq"]),
                "text": item["payload"].get("brief", {}).get("question", "")
                if item["kind"] == "start"
                else item["payload"].get("message", ""),
            }
            for item in commands
        ]

    def wait(self, coroutine: Coroutine[Any, Any, dict]) -> dict:
        # The graph runs in a worker thread; SDK and MCP stay on their single owning asyncio loop.
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    def skill(self) -> tuple[str, str]:
        # Only this registered read-only Skill is enabled. User text cannot choose a path or execute scripts.
        path = self.skill_root / "evidence-answer" / "SKILL.md"
        if path.is_symlink() or not path.resolve().is_relative_to(self.skill_root):
            raise AgentError("skill_path_invalid")
        raw = path.read_bytes()
        if len(raw) > 12000:
            raise AgentError("skill_byte_limit")
        text = raw.decode("utf-8")
        if not text.startswith("---\n") or "name: evidence-answer" not in text.split("---", 2)[1]:
            raise AgentError("skill_metadata_invalid")
        return text.split("---", 2)[2].strip(), digest(raw)

    def projection(self, *, evidence: list[dict] | None = None, include_user_skill_body: bool = True) -> tuple[dict, dict]:
        task = self.execution.snapshot()
        brief = ResearchBrief.model_validate_json(json.dumps(task["brief"]), strict=True)
        memories = self.memory.current(task["owner_id"], task["project_id"])
        # Source-dependent memories are not evidence and need an explicit source authorization bridge.
        eligible = [memory for memory in memories if not memory.get("source_refs")]
        active = {(memory["memory_id"], memory["revision"]) for memory in eligible}
        history = []
        excluded = []
        from .user_skills import activate_for_task
        user_skills = activate_for_task(self.execution.database, task)
        for turn in task["history"]:
            if turn["answer"].get("context_manifest", {}).get("skill_refs", []) != user_skills["pins"]:
                excluded.append("history_selected_skill_changed")
                continue
            dependencies = turn["answer"].get("context_manifest", {}).get("memory_refs", [])
            if any((item["id"], item["revision"]) not in active for item in dependencies):
                excluded.append("history_memory_invalidated")
                history.append({"question": turn["question"], "answer_status": "omitted_memory_invalidated"})
                continue
            # Historical citations must be authorized by the evidence bridge before reusing their text.
            if turn["answer"].get("citations"):
                excluded.append("history_evidence_requires_current_retrieval")
                history.append(
                    {
                        "question": turn["question"],
                        "answer_status": "omitted_until_current_evidence_retrieval",
                        "source_selections": turn["answer"].get("context_manifest", {}).get("evidence_refs", []),
                    }
                )
                continue
            history.append({"question": turn["question"], "answer": turn["answer"]["text"]})
        skill, skill_hash = self.skill()
        manifest = {
            "policy_version": POLICY_VERSION,
            "brief_revision": task["revision"],
            "brief_sha256": digest(brief.model_dump(mode="json")),
            "profile_hash": task["profile_hash"],
            "graph_version": task["graph_version"],
            "skill_id": "builtin:evidence-answer",
            "skill_hash": skill_hash,
            "skill_refs": user_skills["pins"],
            "memory_refs": [{"id": item["memory_id"], "revision": item["revision"]} for item in eligible],
            "hard_memory_refs": [
                {"id": item["memory_id"], "revision": item["revision"]}
                for item in eligible
                if item.get("kind") != "preference"
            ],
            "evidence_refs": [
                {
                    "ref_id": item["ref_id"],
                    "source_sha256": item["source_sha256"],
                    "unit_text_sha256": item["unit_text_sha256"],
                    "coverage": item.get("coverage", item.get("coverage_kind")),
                    "source_ref": item["source_ref"],
                    **{
                        key: item[key]
                        for key in (
                            "span_start",
                            "span_end",
                            "coordinate_system",
                            "work_version_id",
                            "representation_id",
                        )
                        if key in item
                    },
                }
                for item in evidence or []
            ],
            "source_scope": self.source_scope(brief, evidence or []),
            "excluded": excluded
            + (["source_dependent_memory_requires_authorization"] if len(eligible) < len(memories) else []),
            "counting": "UTF8 byte admission; actual provider usage settled before validation",
        }
        prompt = {
            "canonical_brief": brief.model_dump(mode="json"),
            "prior_turns": history,
            "user_turns": self.user_turns(),
            "confirmed_user_memory_data": eligible,
            "untrusted_evidence_data": evidence or [],
            "skill_instructions": skill,
            "selected_user_skills": user_skills["instructions"] if include_user_skill_body else [],
            "context_manifest": manifest,
        }
        if not include_user_skill_body:
            manifest["excluded"].append("selected_user_skill_bodies_omitted_from_independent_review")
        return prompt, manifest

    @staticmethod
    def source_scope(brief: ResearchBrief, evidence: list[dict]) -> list[dict]:
        """Actual issued windows per selected source; no model-generated coverage counts."""
        result = []
        for selection in brief.selected_papers:
            seen = [
                item
                for item in evidence
                if item.get("selection") == selection.model_dump()
                or (
                    item.get("file_id") == selection.id
                    if selection.kind == "material_file"
                    else item.get("source_ref") == selection.model_dump()
                )
            ]
            result.append(
                {
                    "selection": selection.model_dump(),
                    "state": "uncovered"
                    if not seen
                    else ("metadata_only" if selection.kind == "metadata_projection" else "bounded_literal_windows"),
                    "full_document_coverage_claimed": False,
                    "original_PDF_coverage_claimed": False,
                    "windows": [
                        {
                            key: item[key]
                            for key in (
                                "ref_id",
                                "source_sha256",
                                "unit_text_sha256",
                                "span_start",
                                "span_end",
                                "coordinate_system",
                                "work_version_id",
                                "representation_id",
                                "coverage",
                                "coverage_kind",
                                "source_version_sha256",
                                "missing_ranges",
                            )
                            if key in item
                        }
                        for item in seen
                    ],
                }
            )
        return result

    def payload(
        self, prompt: dict, *, history: list[dict] | None = None, tools: bool = False, review: bool = False,
        response_model: Any = None, instructions: str | None = None, schema_name: str | None = None,
        logical_key: str | None = None
    ) -> dict:
        task = self.execution.snapshot()
        batch_protocol = task["profile"]["profile_id"] in {"qa-v2", "qa-v3", "qa-v4"}
        fixed_grid = task["profile"]["profile_id"] in {"qa-v3", "qa-v4"}
        settled = None
        if fixed_grid:
            from .qa_quotes import require_original_turns
            require_original_turns(prompt)
            logical_key = logical_key or str(self.execution.command["id"]) + (
                ":request-evidence" if tools else ":independent-review" if review else ":answer")
            settled = self.execution.settled_model_request(logical_key)
        remaining = task["profile"]["output_tokens"] - task["output_tokens"]
        if remaining <= 0 and settled is None:
            raise AgentError("output_budget_exhausted")
        result = {
            "model": MODEL,
            "reasoning": {"effort": "none"},
            "truncation": "disabled",
            "max_output_tokens": min(task["profile"]["per_call_output"], remaining),
            "instructions": POLICY_MARKER
            + "\n"
            + (instructions if instructions is not None else REVIEW_POLICY if review else SYSTEM_POLICY)
            + "\nWrite conclusions in the current user's language; preserve quotations in their source language.",
            "input": [{"role": "user", "content": canonical(prompt).decode()}] + (history or []),
        }
        if batch_protocol:
            role_cap = 1024 if tools else (4096 if fixed_grid else 2048) if review else 3072
            if task["profile"]["profile_id"] == "qa-v3" and settled is None:
                # The whole remaining protocol is admitted before each stage;
                # a smaller leftover must not silently truncate a required role.
                required_models = 3 if tools else 1 if review else 2
                required_output = 8192 if tools else 4096 if review else 7168
                if task["profile"]["model_calls"] - task["model_calls"] < required_models:
                    raise AgentError("qa_grid_remaining_model_budget_insufficient")
                if remaining < required_output:
                    raise AgentError("qa_grid_remaining_output_budget_insufficient")
            if (task["profile"]["profile_id"] == "qa-v4" and settled is None
                    and (task["profile"]["model_calls"] - task["model_calls"] < 1 or remaining < role_cap)):
                raise AgentError("qa_revision_role_reservation_unavailable")
            if settled is not None:
                if settled["request"].get("max_output_tokens") != role_cap:
                    raise AgentError("settled_qa_grid_role_cap_changed")
                result["max_output_tokens"] = role_cap
            else:
                result["max_output_tokens"] = min(role_cap, task["profile"]["per_call_output"], remaining)
        if tools:
            result["tools"] = self.tools.provider_tools()
            if batch_protocol:
                result["tools"] = [tool for tool in result["tools"] if tool["name"] == "search_evidence"]
                result["instructions"] += ("\nRetrieval phase: issue one or at most two search_evidence calls, "
                    "each with a distinct call_id and at most three evidence items. The server executes "
                    "them sequentially. Use queries for the requested subquestions; source scope is "
                    "server-owned. Return no answer before inspecting their results.")
            result["tool_choice"] = {"type": "function", "name": "search_evidence"}
            if not batch_protocol:
                result["parallel_tool_calls"] = False
        else:
            schema = (response_model or (Review if review else ResearchAnswer)).model_json_schema()
            result["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name or ("citequest_claim_review" if review else "citequest_research_answer"),
                    "schema": schema,
                    "strict": True,
                }
            }
        result = fit_context(result, task["profile"]["request_bytes"])
        if settled is not None and digest(result) != settled["request_hash"]:
            raise AgentError("settled_qa_grid_request_changed")
        if "context_manifest" in prompt:
            chosen = json.loads(result["input"][0]["content"])["context_manifest"]
            prompt["context_manifest"].clear()
            prompt["context_manifest"].update(chosen)
        return result

    def clarify(self, state: State) -> dict:
        task = self.execution.snapshot()
        question = task["brief"]["question"]
        if not question.strip():
            response = interrupt(
                {
                    "kind": "question_required",
                    "fields": ["message"],
                    "question": "What research question should this Task answer?",
                }
            )
            if (
                not isinstance(response, dict)
                or set(response) != {"message"}
                or not isinstance(response["message"], str)
                or not response["message"].strip()
            ):
                raise AgentError("clarification_response_invalid")
            question = response["message"]
        return {"question": question}

    def request_evidence(self, state: State) -> dict:
        prompt, _ = self.projection()
        prompt["canonical_brief"]["question"] = state["question"]
        if self.execution.snapshot()["profile"]["profile_id"] == "qa-v4":
            self.qa4.budget("request", str(self.execution.command["id"]) + ":request-evidence")
        response = self.wait(
            self.provider.complete(
                self.payload(prompt, tools=True), str(self.execution.command["id"]) + ":request-evidence"
            )
        )
        calls = [item for item in output_items(response) if item["type"] == "function_call"]
        task = self.execution.snapshot()
        if task["profile"]["profile_id"] in {"qa-v2", "qa-v3", "qa-v4"}:
            from .tool_batch import admit_search_batch
            validated = admit_search_batch(response, self.tools.catalog["search_evidence"],
                tools_remaining=task["profile"]["tool_calls"] - task["tool_calls"],
                models_remaining=task["profile"]["model_calls"] - task["model_calls"])
            return {"tool_request": response, "validated_tool_calls": validated}
        if len(calls) != 1 or len({item["call_id"] for item in calls}) != len(calls):
            raise AgentError("model_tool_calls_invalid")
        task = self.execution.snapshot()
        if task["tool_calls"] + len(calls) > task["profile"]["tool_calls"]:
            raise AgentError("tool_budget_exhausted")
        if any(item["name"] != "search_evidence" for item in calls):
            raise AgentError("unexpected_forced_tool")
        return {"tool_request": response}

    def execute_tools(self, state: State) -> dict:
        outputs, evidence = [], []
        task = self.execution.snapshot()
        batch_protocol = task["profile"]["profile_id"] in {"qa-v2", "qa-v3", "qa-v4"}
        calls = state["validated_tool_calls"] if batch_protocol else output_items(state["tool_request"])
        for item in calls:
            if item["type"] != "function_call":
                continue
            arguments = item["validated_arguments"] if batch_protocol else json.loads(item["arguments"])
            result = self.wait(
                self.tools.call(
                    item["name"], arguments, logical_key=str(self.execution.command["id"]) + ":tool:" + item["call_id"]
                )
            )
            evidence.extend(result.get("evidence", []))
            outputs.append(
                {"type": "function_call_output", "call_id": item["call_id"], "output": canonical(result).decode()}
            )
        if len({item["ref_id"] for item in evidence}) != len(evidence):
            evidence = list({item["ref_id"]: item for item in evidence}.values())
        return {"tool_outputs": outputs, "evidence": evidence}

    def answer(self, state: State) -> dict:
        task = self.execution.snapshot()
        if task["profile"]["profile_id"] in {"qa-v2", "qa-v3"}:
            from .qa_quotes import QuoteQANodes
            return QuoteQANodes(self).answer(state)
        check_owner = self.execution.command["id"] if task["profile"]["profile_id"] in {"qa-v2", "qa-v3", "qa-v4"} else self.execution.run_id
        # Re-read every included source before dispatch so persisted tool results cannot bypass current ACL.
        for item in state["evidence"]:
            current = self.wait(
                self.tools.call(
                    "read_evidence",
                    {"ref_id": item["ref_id"]},
                    logical_key=str(check_owner) + ":recheck:" + item["ref_id"],
                )
            )
            if (
                current["source_sha256"] != item["source_sha256"]
                or current["unit_text_sha256"] != item["unit_text_sha256"]
            ):
                raise AgentError("evidence_version_changed")
        prompt, manifest = self.projection(evidence=state["evidence"])
        prompt["canonical_brief"]["question"] = state["question"]
        # Preserve every actual call_id/output pair. In v2 a source can occur in
        # both search results; the whole encoded history still passes wire admission.
        prompt["untrusted_evidence_data"] = [
            {key: value for key, value in item.items() if key != "text"} for item in state["evidence"]
        ]
        history = output_items(state["tool_request"]) + state["tool_outputs"]
        response = self.wait(
            self.provider.complete(self.payload(prompt, history=history), str(self.execution.command["id"]) + ":answer")
        )
        answer = ResearchAnswer.model_validate_json(output_text(response), strict=True)
        task = self.execution.snapshot()
        if answer.kind != task["brief"]["kind"] or answer.requested_direction != task["brief"]["requested_direction"]:
            raise AgentError("candidate_task_kind_or_direction_changed")
        validate_literal(answer, state["evidence"])
        citations = [{"ref_id": edge.ref_id, "quote": edge.quote} for claim in answer.claims for edge in claim.edges]
        return {
            "answer": answer.model_dump(),
            "result": {
                **answer.model_dump(),
                "text": render_text(answer),
                "citations": citations,
                "context_manifest": manifest,
                "publication_status": "candidate",
                "source_scope": manifest["source_scope"],
                "completion_status": "partial_named_source_coverage"
                if any(item["state"] == "uncovered" for item in manifest["source_scope"])
                else "bounded_windows_evaluated",
                "quality": {
                    "format": "validated",
                    "citation_literal": "validated",
                    "semantic_support": "pending_independent_ai_review",
                },
                "fixture": any(item.get("fixture") for item in state["evidence"]),
            },
        }

    def independent_review(self, state: State) -> dict:
        self.execution.snapshot()
        if state["result"].get("citation_protocol") == "citequest-quote-spans/3":
            from .qa_quotes import QuoteQANodes
            return QuoteQANodes(self).review(state)
        prompt, _ = self.projection(evidence=state["evidence"], include_user_skill_body=False)
        prompt.update(candidate=state["answer"], rubric_version="claim-review/1")
        response = self.wait(
            self.provider.complete(
                self.payload(prompt, review=True), str(self.execution.command["id"]) + ":independent-review"
            )
        )
        review = Review.model_validate_json(output_text(response), strict=True)
        candidate = ResearchAnswer.model_validate(state["answer"], strict=True)
        accepted = True
        try:
            validate_review(candidate, review)
        except ValueError:
            accepted = False
        return {
            "result": {
                **state["result"],
                "review": review.model_dump(),
                "publication_status": "validated_ai_review" if accepted else "rejected",
                "quality": {
                    **state["result"]["quality"],
                    "semantic_support": "ai_review_not_human_calibrated" if accepted else "ai_review_rejected",
                },
            }
        }

    def validate(self, state: State) -> dict:
        from .user_skills import validate_task_skills
        validate_task_skills(self.execution.database, self.execution.snapshot())
        if state.get("answer") is None and state["result"].get("publication_status") == "not_evaluated":
            return {"result": state["result"]}
        evidence = {item["ref_id"]: item for item in state["evidence"]}
        task = self.execution.snapshot()
        check_owner = self.execution.command["id"] if task["profile"]["profile_id"] in {"qa-v2", "qa-v3", "qa-v4"} else self.execution.run_id
        if state["result"].get("citation_protocol") == "citequest-quote-spans/3":
            from citequest.research_v3.quote_spans import ResearchAnswerV3
            answer = ResearchAnswerV3.model_validate(state["answer"], strict=True)
        else:
            answer = ResearchAnswer.model_validate(state["answer"], strict=True)
        validate_literal(answer, state["evidence"])
        for citation in {(edge.ref_id, edge.quote): edge for claim in answer.claims for edge in claim.edges}.values():
            if citation.ref_id not in evidence or citation.quote not in evidence[citation.ref_id]["text"]:
                raise AgentError("citation_not_literal_in_allowed_evidence")
            current_source = self.wait(
                self.tools.call(
                    "read_evidence",
                    {"ref_id": citation.ref_id},
                    logical_key=str(check_owner) + ":publish-check:" + citation.ref_id,
                )
            )
            if (
                current_source["source_sha256"] != evidence[citation.ref_id]["source_sha256"]
                or citation.quote not in current_source["text"]
            ):
                raise AgentError("evidence_changed_before_publication")
            if state["result"].get("citation_protocol") == "citequest-quote-spans/3":
                from .qa_quotes import PIN_FIELDS
                if any(current_source.get(field) != evidence[citation.ref_id].get(field) for field in PIN_FIELDS):
                    raise AgentError("evidence_version_changed_before_publication")
        # Memory corrections/deletions after generation invalidate this candidate before publication.
        _, current = self.projection(evidence=state["evidence"])
        included = state["result"]["context_manifest"]["memory_refs"]
        current_refs = {(item["id"], item["revision"]) for item in current["memory_refs"]}
        if any((item["id"], item["revision"]) not in current_refs for item in included):
            raise AgentError("memory_changed_during_generation")
        if current["hard_memory_refs"] != state["result"]["context_manifest"].get("hard_memory_refs", []):
            raise AgentError("hard_memory_changed_during_generation")
        if state["result"].get("citation_protocol") == "citequest-quote-spans/3":
            # Stable command keys replay settled free-read attempts. Their cache
            # is not current authority after a cold resume at publication.
            self.wait(self.tools.check_scope())
        return {
            "result": {**state["result"], "quality": {**state["result"]["quality"], "citation_literal": "validated"}}
        }

    def run(self) -> dict:
        previous = self.workflow.get_state(self.config)
        if previous.next:
            if self.execution.command["kind"] == "resume":
                value = Command(resume={"message": self.execution.command["payload"]["message"]})
            else:
                value = None
        else:
            value = {"task_id": str(self.execution.task_id), "command_id": str(self.execution.command["id"])}
        result = self.workflow.invoke(value, self.config, durability="sync")
        snapshot = self.workflow.get_state(self.config)
        checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
        if result.get("__interrupt__"):
            self.execution.finish(
                "waiting_input",
                result={"interrupts": [item.value for item in result["__interrupt__"]]},
                checkpoint_id=checkpoint_id,
            )
        else:
            answer = result["result"]
            if answer.get("publication_status") == "validated_ai_review":
                self.memory.validate_result_context(self.execution.snapshot(), answer)
            rejected = answer.get("publication_status") == "rejected"
            partial = (
                answer.get("completion_status", "").startswith("partial_") or answer["evidence_status"] != "supported"
            )
            self.execution.finish(
                "failed" if rejected else ("partial" if partial else "succeeded"),
                result=answer,
                checkpoint_id=checkpoint_id,
                error="semantic_review_rejected" if rejected else None,
            )
        return result
