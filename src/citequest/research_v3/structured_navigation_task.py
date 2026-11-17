"""One ledger-owned local-compute attempt for selected-source navigation."""
from __future__ import annotations

import time

from citequest.agents_v3.contracts import AgentError, digest
from citequest.v2.domain import Actor

from .decisionrag_remote import RemoteBinaryJudge
from .source_navigation import SourceNavigator
from .source_navigation_sources import NavigationSources


def task_binding(task):
    return digest({key: task[key] for key in ("id", "owner_id", "brief", "profile", "graph_version")}
        | {"id": str(task["id"]), "owner_id": str(task["owner_id"])})


def navigate_task_sources(config, query, ref_ids, *, access, context, register, remaining_ms):
    if (config.navigation_protocol not in {"structure1", "structure2"} or type(remaining_ms) is not int
            or not 1 <= remaining_ms <= config.active_seconds*1000
            or type(ref_ids) is not list or len(ref_ids) > config.max_passages or len(set(ref_ids)) != len(ref_ids)):
        raise AgentError("structure_navigation_task_contract_invalid")
    started = time.monotonic()
    task = context()
    if task["kind"] != "qa" or task["profile"].get("evidence_navigation_protocol") != config.navigation_protocol:
        raise AgentError("structure_navigation_task_protocol_required")
    actor, binding = Actor(task["owner_id"]), task_binding(task)
    selections = task["brief"]["selected_papers"]
    if not 1 <= len(selections) <= 20:
        raise AgentError("structure_navigation_requires_selected_sources", 422)
    def check_task():
        if task_binding(context()) != binding:
            raise AgentError("structure_navigation_task_scope_changed")
        if time.monotonic()-started >= remaining_ms/1000:
            raise AgentError("structure_navigation_active_budget_exhausted")
    initial = []
    for ref in ref_ids:
        check_task()
        initial.append(access.read(actor, task["id"], ref))
    sources = NavigationSources(access.search, access.materials)
    effective = config.model_copy(update={"active_seconds": min(config.active_seconds, remaining_ms/1000)})
    judge = RemoteBinaryJudge(config.judge, deadline=started+remaining_ms/1000,
        expected_resource_profile_sha256=config.resource_profile_sha256)
    result = SourceNavigator(sources, effective).run(query, actor, task["id"], selections, initial,
        judge, register, check_task, started=started)
    check_task()
    return result
