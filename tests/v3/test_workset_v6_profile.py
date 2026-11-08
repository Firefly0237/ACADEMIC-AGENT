"""Pure registration contract; not a graph, model or product acceptance test."""
from uuid import uuid4

from citequest.agents_v3.contracts import (
    CURRENT_GRAPH_VERSIONS, PROFILES, WORKSET_V5_GRAPH_VERSION,
    WORKSET_V6_GRAPH_VERSION, ResearchBrief, TaskCreate, graph_version_for_kind,
)
from citequest.agents_v3.development_campaign import ALLOWED_PAIRS


def test_workset_v6_is_explicit_with_unchanged_resource_limits_and_v5_route():
    old = PROFILES["workset-v5"].model_dump()
    new = PROFILES["workset-v6"].model_dump()
    assert {**old, "profile_id": "workset-v6"} == new
    assert WORKSET_V6_GRAPH_VERSION in CURRENT_GRAPH_VERSIONS
    assert {("summary", "workset-v6"), ("compare", "workset-v6")} <= ALLOWED_PAIRS
    assert {("summary", "workset-v5"), ("compare", "workset-v5")}.isdisjoint(ALLOWED_PAIRS)
    for kind in ("summary", "compare"):
        for profile, graph in (("workset-v5", WORKSET_V5_GRAPH_VERSION),
                               ("workset-v6", WORKSET_V6_GRAPH_VERSION)):
            request = TaskCreate(project_id=uuid4(), operation_id=uuid4(), kind=kind,
                                 profile_id=profile, brief=ResearchBrief(kind=kind))
            assert request.profile_id == profile
            assert graph_version_for_kind(kind, profile) == graph
