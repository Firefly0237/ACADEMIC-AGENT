"""Fixed local evidence MCP server. Identity/scope come from the admitted Task."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from uuid import UUID

from .exception_metadata import private_runtime_observation, private_runtime_phase

with private_runtime_phase("mcp_server_boot","runtime_imports"):
    import psycopg
    from mcp.server.fastmcp import FastMCP
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    from pydantic import Field

    from .catalog_scope import POLICY, BusinessCatalogDB, CatalogPins, CatalogScopePolicy, require_runtime, task_policy
    from .contracts import AgentError, ResearchBrief, Strict, canonical, digest

if TYPE_CHECKING:
    from citequest.research_v3.workset_contracts import WorksetManifest


class ToolConfig(Strict):
    mode: str = Field(pattern=r"^(fixture|search)$")
    task_id: UUID
    epoch: int = Field(ge=1)
    business_dsn_file: Path
    search_config_file: Path | None = None
    material_config_file: Path | None = None
    fixture_evidence_file: Path | None = None
    catalog_scope_policy: CatalogScopePolicy | None = None
    decisionrag_config_file: Path | None = None
    decisionrag_config_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")] | None = None
    rerank_config_file: Path | None = None
    rerank_config_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")] | None = None


def build_server(config: ToolConfig) -> FastMCP:
    with private_runtime_phase("mcp_server_boot","task_lookup",task_id=config.task_id):
        dsn = config.business_dsn_file.read_text(encoding="utf-8").strip().replace("postgresql+psycopg://", "postgresql://", 1)
        with psycopg.connect(dsn, row_factory=dict_row, connect_timeout=3) as db:
            initial = db.execute("SELECT t.* FROM cq_app.agent_tasks t JOIN cq_app.projects p ON (p.id,p.owner_id)=(t.project_id,t.owner_id) WHERE t.id=%s AND t.epoch=%s AND t.state='running' AND NOT t.cancellation_requested",
                (config.task_id, config.epoch)).fetchone()
            if initial is None:
                raise AgentError("tool_task_owner_or_scope_invalid")
            require_runtime(initial, config.catalog_scope_policy)
            if config.mode == 'fixture' and task_policy(initial) is not None:
                raise AgentError('task_catalog_snapshot_requires_product_search_scope', 503)
    with private_runtime_phase("mcp_server_boot","source_selection",task_id=config.task_id):
        initial_extract = initial["brief"]["kind"] == "research" and initial["brief"].get("research_action") == "extract"
        from citequest.research_v3.evidence import draft_selections, reference_allows, reference_selections
        selected = draft_selections(initial["brief"]) if initial_extract else reference_selections(initial["brief"])
        workset_kind = initial["brief"]["kind"] in {"summary", "compare"}
    search, material = None, None
    if config.mode == "search":
        if (config.search_config_file is None and config.material_config_file is None) or config.fixture_evidence_file is not None:
            raise AgentError("search_tool_config_invalid")
        needs_material = any(item["kind"] == "material_file" for item in selected)
        needs_papers = not selected or any(item["kind"] != "material_file" for item in selected)
        if needs_papers and config.search_config_file:
            with private_runtime_phase("mcp_server_boot","search_service",task_id=config.task_id):
                from citequest.search_v3.runtime import service_from_config
                search = service_from_config(config.search_config_file)
        if needs_material and config.material_config_file:
            with private_runtime_phase("mcp_server_boot","material_service",task_id=config.task_id):
                from citequest.material_evidence_v3 import service_from_config
                material = service_from_config(config.material_config_file)
                if material.retrieval_mode != "native_hybrid" and not initial_extract:
                    raise AgentError("diagnostic_material_retrieval_not_enabled_for_product_tools", 503)
    elif config.fixture_evidence_file is None or config.search_config_file is not None or config.material_config_file is not None:
        raise AgentError("fixture_tool_config_invalid")
    registration_started = time.monotonic()
    private_runtime_observation("mcp_server_boot","tool_registration","started",task_id=config.task_id)
    server = FastMCP("citequest-evidence-v1", log_level="ERROR")
    catalog_pins = CatalogPins(BusinessCatalogDB(config.business_dsn_file, config.catalog_scope_policy), search)
    from .chunk_tool_policy import uses_typed_chunk_tools
    qa5 = uses_typed_chunk_tools(initial)
    if qa5 and (config.mode != "search" or task_policy(initial) != POLICY):
        raise AgentError("qa5_chunk_profile_and_task_catalog_pins_required", 503)
    from citequest.research_v3.decisionrag_task import load_task_config, task_requests_navigation
    navigation_config = None
    if (config.decisionrag_config_file is None) != (config.decisionrag_config_sha256 is None):
        raise AgentError("decisionrag_trusted_configuration_required")
    if task_requests_navigation(initial):
        if not qa5 or config.mode != "search" or config.decisionrag_config_file is None:
            raise AgentError("decisionrag_runtime_not_configured")
        navigation_config = load_task_config(config.decisionrag_config_file, config.decisionrag_config_sha256)
        from .product_profiles import evidence_navigation_protocol
        if navigation_config.navigation_protocol != evidence_navigation_protocol(initial):
            raise AgentError("decisionrag_task_runtime_protocol_mismatch")
    from .product_profiles import retrieval_rerank_protocol
    from citequest.research_v3.rerank import TEICrossEncoder, load_config as load_rerank_config
    reranker = None
    if (config.rerank_config_file is None) != (config.rerank_config_sha256 is None):
        raise AgentError("rerank_trusted_configuration_required", 503)
    if retrieval_rerank_protocol(initial) is not None:
        if (not qa5 or config.mode != "search" or config.rerank_config_file is None
                or config.rerank_config_sha256 != initial["profile"]["rerank_config_sha256"]):
            raise AgentError("rerank_runtime_configuration_not_bound", 503)
        reranker = TEICrossEncoder(load_rerank_config(config.rerank_config_file, config.rerank_config_sha256),
            config.rerank_config_sha256)

    def context() -> dict:
        with psycopg.connect(dsn, row_factory=dict_row, connect_timeout=3) as db:
            task = db.execute("""SELECT t.* FROM cq_app.agent_tasks t JOIN cq_app.projects p
                ON (p.id,p.owner_id)=(t.project_id,t.owner_id) WHERE t.id=%s AND t.epoch=%s AND t.state='running'
                AND NOT t.cancellation_requested""", (config.task_id, config.epoch)).fetchone()
            if task is None:
                raise AgentError("tool_task_owner_or_scope_invalid")
            require_runtime(task, config.catalog_scope_policy)
            if config.mode == "fixture" and not db.execute("SELECT current_database() LIKE 'citequest_s08_fixture%%' AS ok").fetchone()["ok"]:
                raise AgentError("fixture_requires_owned_fixture_database")
            return task

    def fixture_items() -> list[dict]:
        items = json.loads(config.fixture_evidence_file.read_text(encoding="utf-8"))
        if not isinstance(items, list) or len(items) > 10:
            raise AgentError("fixture_evidence_invalid")
        result = []
        for item in items:
            text = item["text"]
            result.append({"ref_id": "fixture:" + digest(item), "text": text,
                "source_ref": {"kind": "fixture", "id": item["id"]}, "source_sha256": digest(text.encode()),
                "unit_text_sha256": digest(text.encode()), "span_start": 0, "span_end": len(text),
                "coordinate_system": "unicode_codepoints", "coverage": "fixture_literal",
                "fixture": True, "title": item.get("title", item["id"])})
        return result

    def register(items: list[dict]) -> None:
        with psycopg.connect(dsn, connect_timeout=3) as db:
            for item in items:
                db.execute("INSERT INTO cq_app.agent_evidence_refs(task_id,ref_id,source_hash,source_ref,payload) VALUES(%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (config.task_id, item["ref_id"], item["source_sha256"], Jsonb(item["source_ref"]), Jsonb(item)))

    def registered(ref_id: str) -> dict:
        with psycopg.connect(dsn, row_factory=dict_row, connect_timeout=3) as db:
            row = db.execute("SELECT * FROM cq_app.agent_evidence_refs WHERE task_id=%s AND ref_id=%s", (config.task_id, ref_id)).fetchone()
            if row is None:
                raise AgentError("evidence_ref_not_issued_to_task", 403)
            return row["payload"]

    def window_selections(task: dict) -> list[dict]:
        if task["brief"]["kind"] in {"summary", "compare"}:
            return task["brief"]["selected_papers"]
        return draft_selections(task["brief"])

    def uses_windows(task: dict) -> bool:
        return task["brief"]["kind"] in {"summary", "compare"} or (
            task["brief"]["kind"] == "research" and task["brief"].get("research_action") == "extract")

    @server.resource("citequest://task/scope")
    def scope_preflight() -> str:
        """Host-only admission resource, never a model capability or a paid tool."""
        from citequest.v2.domain import Actor, DomainError
        try:
            task = context()
            checked, replanned, reads, role_excluded = 0, 0, {}, 0
            pins = {}
            if config.mode != "fixture":
                from citequest.v2.domain import Actor
                with psycopg.connect(dsn, row_factory=dict_row, connect_timeout=3) as db:
                    issued = [row["payload"] for row in db.execute("SELECT payload FROM cq_app.agent_evidence_refs WHERE task_id=%s ORDER BY ref_id", (config.task_id,)).fetchall()]
                if uses_windows(task):
                    from citequest.research_v3.source_windows import FullSourceWindows
                    windows = FullSourceWindows(search, material)
                    current = windows.plan(Actor(task["owner_id"]), window_selections(task))
                    replanned = len(current.documents)
                    manifest_pins = [item for item in issued if item.get("ref_id") == "workset-manifest:" + str(task["revision"])]
                    if manifest_pins and manifest_pins[0]["manifest"] != current.model_dump(mode="json"):
                        raise AgentError("current_scope_workset_version_changed", 409)
                    allowed = {(item.document_id, item.source_version_sha256) for item in current.documents}
                    evidence = [item for item in issued if item.get("evidence_type") == "source_window" and
                        (item.get("document_id"), item.get("source_version_sha256")) in allowed]
                    for start in range(0, len(evidence), 64):
                        verified = windows.verify_pinned_batch(Actor(task["owner_id"]), evidence[start:start+64])
                        checked += len(verified["pins"])
                        for key, value in verified["source_reads"].items():
                            reads[key] = reads.get(key, 0) + value
                else:
                    from citequest.research_v3.evidence import admit_scope
                    if task_policy(task) == POLICY:
                        pins = catalog_pins.ensure(Actor(task['owner_id']), config.task_id)
                        files = [UUID(value['id']) for value in reference_selections(task['brief']) if value['kind'] == 'material_file']
                        if files:
                            if material is None:
                                raise AgentError('material_semantic_index_not_configured', 503)
                            material.validate_selection(Actor(task['owner_id']), files)
                        eligible_issued = [item for item in issued if task['brief']['kind'] != 'research' or reference_allows(task['brief'], item)]
                        catalog_pins.authorize_evidence(Actor(task['owner_id']), config.task_id, eligible_issued)
                    else:
                        admit_scope(Actor(task["owner_id"]), ResearchBrief.model_validate_json(json.dumps(task["brief"]), strict=True), search, material)
                    for item in issued:
                        if item.get("source_ref", {}).get("kind") == "workset_manifest":
                            continue  # Durable control metadata, never a Paper EvidenceDTO.
                        if task["brief"]["kind"] == "research" and not reference_allows(task["brief"], item):
                            role_excluded += 1
                            continue
                        if item.get("evidence_type") == "source_window":
                            selected_keys = {(value["kind"], value["id"]) for value in reference_selections(task["brief"])}
                            selection = item.get("selection", {})
                            if (selection.get("kind"), selection.get("id")) not in selected_keys:
                                continue
                            from citequest.research_v3.source_windows import FullSourceWindows
                            FullSourceWindows(search, material).verify_pinned_batch(Actor(task["owner_id"]), [item])
                        elif item.get("source_ref", {}).get("kind") == "material_fragment":
                            material.read_pinned(Actor(task["owner_id"]), item)
                        else:
                            from citequest.search_v3.schemas import EvidenceDTO
                            search.read_pinned_evidence(Actor(task["owner_id"]), EvidenceDTO.model_validate_json(json.dumps(item), strict=True))
                        checked += 1
            return json.dumps({"status": "ready", "brief_sha256": digest(task["brief"]), "fixture": config.mode == "fixture",
                "pinned_refs_checked": checked, "manifest_documents_replanned": replanned, "source_reads": reads,
                "historical_refs_excluded_from_current_literature_role": role_excluded,
                **({'catalog_scope_policy': POLICY, 'catalog_pins': [{'catalog': pin.catalog,
                    'generation_id': str(pin.generation_id), 'epoch': pin.epoch,
                    'binding_sha256': digest(pin.model_dump(mode='json'))} for pin in pins.values()]}
                    if task_policy(task) == POLICY else {})})
        except (AgentError, DomainError) as exc:
            return json.dumps({"status": "not_ready", "error_code": exc.code})
        except Exception as exc:
            # stdio belongs to the MCP wire. Diagnostics go ONLY to stderr;
            # never emit exception messages, SQL params or credentials.
            import sys

            from .exception_metadata import exception_metadata

            print(json.dumps({"event":"scope_dependency_exception","diagnostic":exception_metadata(exc)}),file=sys.stderr,flush=True)  # noqa: T201 - Data-minimized private process diagnostics.
            return json.dumps({"status": "not_ready", "error_code": "scope_dependency_unavailable"})

    @server.tool()
    def search_evidence(query: Annotated[str, Field(min_length=1, max_length=2000)], limit: Annotated[int, Field(ge=1, le=8)] = 2) -> dict:
        """Search only the current Task's permitted evidence catalog. Owner is server-derived."""
        if not isinstance(query, str) or not 1 <= len(query) <= 2000 or type(limit) is not int or not 1 <= limit <= 8:
            raise AgentError("search_tool_arguments_invalid")
        task = context()
        research_v5 = task["kind"] == "research" and task["profile"].get("profile_id") in {"research-v5", "research-v6"}
        if limit > 3 and not research_v5:
            raise AgentError("search_tool_arguments_invalid")
        if task["brief"]["kind"] == "research" and task["brief"].get("research_action") == "extract":
            raise AgentError("research_draft_extract_requires_full_windows")
        if config.mode == "fixture":
            items = fixture_items()[:limit]
            register(items)
            return {"status": "ok", "empty": not items, "evidence": items, "fixture": True}
        from citequest.search_v3.schemas import SearchRequest
        from citequest.v2.domain import Actor
        pins = catalog_pins.ensure(Actor(task['owner_id']), config.task_id) if task_policy(task) == POLICY else None
        selected = reference_selections(task["brief"])
        if selected:
            if research_v5 and len(selected) > limit:
                raise AgentError("research5_search_capacity_must_include_selected_sources")
            paper_refs = [item for item in selected if item["kind"] != "material_file"]
            material_ids = [UUID(item["id"]) for item in selected if item["kind"] == "material_file"]
            evidence = []
            if paper_refs:
                if search is None:
                    raise AgentError("paper_evidence_bridge_unavailable", 503)
                from citequest.search_v3.schemas import PaperLocator
                locators = [PaperLocator.model_validate(item, strict=True) for item in paper_refs]
                paper_limit = limit - len(material_ids) if research_v5 else limit
                values = (search.search_selected_pinned_evidence(Actor(task['owner_id']), locators, query, paper_limit, pins,
                    maximum=8 if research_v5 else 5)
                    if pins is not None else search.search_selected_evidence(Actor(task["owner_id"]), locators, query, limit))
                evidence.extend(item.model_dump(mode='json') for item in values)
            if material_ids:
                if material is None:
                    raise AgentError("material_evidence_bridge_unavailable", 503)
                if research_v5:
                    # One reserved slot per selected material prevents the paper
                    # branch from hiding a material behind its global prefix.
                    per_file = 1 if paper_refs else max(1, limit // len(material_ids))
                    for identifier in material_ids:
                        evidence.extend(material.search_selected(Actor(task["owner_id"]), [identifier], query, per_file))
                else:
                    evidence.extend(material.search_selected(Actor(task["owner_id"]), material_ids, query, limit))
            # Explicit selected sets are each searched before bounded interleaving, never global-topK filtering.
            evidence = evidence[:limit]
            register(evidence)
            return {"status": "ok", "empty": not evidence, "evidence": evidence, "fixture": False,
                "scope": "selected_papers_and_materials", "ranking": "selected_sources_in_declared_order"}
        if search is None:
            raise AgentError("paper_evidence_bridge_unavailable", 503)
        request = SearchRequest.model_validate({"q": query, "mode": "hybrid", "catalog": task["brief"]["catalog"],
            "page_size": limit, "candidate_limit": 50, "ttl_seconds": 1800, "sort": "relevance", "filters": {}}, strict=True)
        if pins is not None and task['brief']['catalog'] not in pins:
            raise AgentError('task_catalog_search_not_in_active_role', 403)
        page = (search.search_pinned(Actor(task['owner_id']), request, pins[task['brief']['catalog']])
            if pins is not None else search.search(Actor(task["owner_id"]), request))
        evidence = []
        for hit in page.results:
            for ref_id in hit.evidence_refs[:1]:
                value = search.read_evidence(Actor(task["owner_id"]), ref_id).model_dump(mode="json")
                evidence.append(value)
        register(evidence[:limit])
        return {"status": "ok", "empty": not evidence, "evidence": evidence[:limit], "snapshot_id": str(page.snapshot_id), "fixture": False}

    @server.tool()
    def read_evidence(ref_id: str) -> dict:
        """Read an exact current-ACL evidence ref; never accept raw document paths/collection names."""
        if not isinstance(ref_id, str) or not 1 <= len(ref_id) <= 300:
            raise AgentError("read_tool_arguments_invalid")
        task = context()
        if task["brief"]["kind"] == "research" and task["brief"].get("research_action") == "extract":
            raise AgentError("research_draft_extract_requires_full_windows")
        pinned = registered(ref_id)
        if config.mode == "fixture":
            for value in fixture_items():
                if value["ref_id"] == ref_id:
                    return value
            raise AgentError("fixture_evidence_not_found", 404)
        from citequest.v2.domain import Actor
        if task_policy(task) == POLICY:
            catalog_pins.authorize_evidence(Actor(task['owner_id']), config.task_id, [pinned])
        if task["brief"]["kind"] == "research" and task["brief"].get("research_action") != "extract" and not reference_allows(task["brief"], pinned):
            raise AgentError("research_evidence_outside_current_literature_role", 403)
        if pinned.get("evidence_type") == "source_window":
            selection = pinned.get("selection", {})
            if selection not in task["brief"].get("selected_papers", []):
                raise AgentError("evidence_not_in_task_selection", 403)
            if task["brief"]["kind"] == "research" and task["brief"].get("research_action") != "extract" and selection not in reference_selections(task["brief"]):
                raise AgentError("research_draft_is_intake_not_literature", 403)
            from citequest.research_v3.source_windows import FullSourceWindows
            FullSourceWindows(search, material).verify_pinned_batch(Actor(task["owner_id"]), [pinned])
            return pinned
        if pinned["source_ref"]["kind"] == "material_fragment":
            if material is None:
                raise AgentError("material_evidence_bridge_unavailable", 503)
            return material.read_pinned(Actor(task["owner_id"]), pinned)
        from citequest.search_v3.schemas import EvidenceDTO
        value = search.read_pinned_evidence(Actor(task["owner_id"]), EvidenceDTO.model_validate_json(json.dumps(pinned), strict=True)).model_dump(mode="json")
        selected = {item["id"] for item in reference_selections(task["brief"]) if item["kind"] != "material_file"}
        paper_id = str((value.get("source_ref") or {}).get("id", ""))
        if selected and paper_id not in selected:
            raise AgentError("evidence_not_in_task_selection", 403)
        if len(value["text"]) > 16000:
            raise AgentError("evidence_window_too_large")
        return value

    def workset_manifest(task: dict) -> WorksetManifest:
        from citequest.research_v3.workset_contracts import WorksetManifest
        pin = registered("workset-manifest:" + str(task["revision"]))
        return WorksetManifest.model_validate(pin["manifest"], strict=True)

    from .qa5_evidence_tools import QA5EvidenceTools
    qa5_tools = QA5EvidenceTools(BusinessCatalogDB(config.business_dsn_file, config.catalog_scope_policy),
        search, material, context, registered, register, catalog_pins, reranker=reranker) if qa5 else None

    def navigate_current_evidence(query: str, ref_ids: list[str], remaining_ms: int) -> dict:
        if navigation_config is None or qa5_tools is None:
            raise AgentError("decisionrag_runtime_not_configured")
        from citequest.research_v3.decisionrag_task import navigate_task_evidence
        return navigate_task_evidence(navigation_config, query, ref_ids, access=qa5_tools.access,
            context=context, remaining_ms=remaining_ms)

    if navigation_config is not None and navigation_config.navigation_protocol == "laya2":
        @server.tool()
        def navigate_issued_evidence(query: Annotated[str, Field(min_length=1, max_length=8000)],
                ref_ids: Annotated[list[str], Field(min_length=1, max_length=20)],
                remaining_ms: Annotated[int, Field(ge=1, le=25000)]) -> dict:
            """Navigate complete issued refs by batch screening and a shared final rescore.

            The trusted client injects remaining_ms. All real forwards and
            cumulative local usage belong to this single bounded tool attempt.
            """
            return navigate_current_evidence(query, ref_ids, remaining_ms)
    else:
        @server.tool()
        def navigate_issued_evidence(query: Annotated[str, Field(min_length=1, max_length=8000)],
                ref_ids: Annotated[list[str], Field(min_length=1, max_length=8)],
                remaining_ms: Annotated[int, Field(ge=1, le=25000)]) -> dict:
            """Navigate only current Task-issued complete refs using two real local probability forwards.

            remaining_ms is injected by the trusted Execution client, never a user or
            model argument. No retrieval, authority expansion, clipping or fallback.
            """
            return navigate_current_evidence(query, ref_ids, remaining_ms)

    if reranker is None:
        @server.tool()
        def search_chunk_evidence(query: Annotated[str, Field(min_length=1, max_length=2000)],
                                  limit: Annotated[int, Field(ge=1, le=20)] = 8) -> dict:
            """QA5-only pinned chunk retrieval; each selected document gets bounded candidate slots."""
            if qa5_tools is None:
                raise AgentError("qa5_chunk_profile_and_task_catalog_pins_required", 503)
            return qa5_tools.search_chunks(query, limit)
    else:
        @server.tool()
        def search_chunk_evidence(query: Annotated[str, Field(min_length=1, max_length=2000)],
                                  remaining_ms: Annotated[int, Field(ge=1, le=30000)],
                                  limit: Annotated[int, Field(ge=1, le=20)] = 8) -> dict:
            """Pinned chunk retrieval and full-candidate TEI ranking; trusted client supplies the deadline."""
            if qa5_tools is None:
                raise AgentError("qa5_chunk_profile_and_task_catalog_pins_required", 503)
            return qa5_tools.search_chunks(query, limit, remaining_ms=remaining_ms)

    @server.tool()
    def plan_workset() -> dict:
        """Plan the current Task's complete available-text window workset; no top-k substitution."""
        task = context()
        if not uses_windows(task):
            raise AgentError("workset_requires_summary_compare_or_draft_extract")
        if config.mode == "fixture":
            raise AgentError("workset_fixture_not_configured")
        from citequest.research_v3.source_windows import FullSourceWindows
        from citequest.v2.domain import Actor
        manifest = FullSourceWindows(search, material).plan(Actor(task["owner_id"]), window_selections(task))
        pin = {"ref_id": "workset-manifest:" + str(task["revision"]), "source_sha256": manifest.manifest_id,
            "source_ref": {"kind": "workset_manifest", "id": manifest.manifest_id}, "manifest": manifest.model_dump(mode="json")}
        register([pin])
        if registered(pin["ref_id"]) != pin:
            raise AgentError("task_workset_manifest_changed")
        return {"manifest": manifest.model_dump(mode="json"), "source_processing_mode": "bounded_full_source_windows"}

    @server.tool()
    def read_source_window(window_id: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]) -> dict:
        """Read one server-issued current-source window, with actual source read counts."""
        task = context()
        if not uses_windows(task):
            raise AgentError("workset_requires_summary_compare_or_draft_extract")
        from citequest.research_v3.source_windows import FullSourceWindows
        from citequest.v2.domain import Actor
        result = FullSourceWindows(search, material).read_window(Actor(task["owner_id"]), workset_manifest(task), window_id)
        if len(canonical(result)) > 16384:
            raise AgentError("source_window_encoded_result_exceeds_limit")
        register(result["evidence"])
        return result

    @server.tool()
    def verify_evidence_batch(ref_ids: Annotated[list[str], Field(min_length=1, max_length=64)]) -> dict:
        """Reauthorize a bounded set of Task-issued pins, returning hashes and counts without source text."""
        task = context()
        if qa5_tools is not None:
            return qa5_tools.verify(ref_ids)
        if not uses_windows(task) or len(set(ref_ids)) != len(ref_ids):
            raise AgentError("invalid_evidence_verification_batch")
        if len(canonical({"ref_ids": ref_ids})) > 20000:
            raise AgentError("evidence_verification_request_byte_limit")
        manifest = workset_manifest(task)
        evidence = [registered(ref_id) for ref_id in ref_ids]
        allowed = {(item.document_id, item.source_version_sha256) for item in manifest.documents}
        if any((item.get("document_id"), item.get("source_version_sha256")) not in allowed for item in evidence):
            raise AgentError("evidence_not_in_task_workset")
        from citequest.research_v3.source_windows import FullSourceWindows
        from citequest.v2.domain import Actor
        return FullSourceWindows(search, material).verify_pinned_batch(Actor(task["owner_id"]), evidence)

    allowed = ({"search_chunk_evidence", "read_evidence", "verify_evidence_batch"} if qa5 else
        {"search_evidence", "read_evidence", "plan_workset", "read_source_window", "verify_evidence_batch"}
        if initial["brief"]["kind"] == "research" else {"plan_workset", "read_source_window", "verify_evidence_batch"}
        if workset_kind else {"search_evidence", "read_evidence"})
    if navigation_config is not None:
        allowed.add("navigate_issued_evidence")
    for name in {"navigate_issued_evidence", "search_chunk_evidence", "search_evidence", "read_evidence", "plan_workset", "read_source_window", "verify_evidence_batch"} - allowed:
        server.remove_tool(name)

    private_runtime_observation("mcp_server_boot","tool_registration","completed",task_id=config.task_id,
        elapsed_seconds=time.monotonic()-registration_started)
    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    with private_runtime_phase("mcp_server_boot","config_load"):
        config = ToolConfig.model_validate_json(args.config.read_text(encoding="utf-8"), strict=True)
    with private_runtime_phase("mcp_server_boot","build_server",task_id=config.task_id):
        server = build_server(config)
    with private_runtime_phase("mcp_server_boot","stdio_run",task_id=config.task_id):
        server.run(transport="stdio")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Each boot phase emitted data-minimized metadata. Do not let Python
        # append a raw SQL/credential-bearing exception message to stderr.
        raise SystemExit(1) from None
