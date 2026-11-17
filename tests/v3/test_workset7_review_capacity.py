"""One pure request/reservation control with explicit provider/execution doubles."""
from types import SimpleNamespace

from citequest.agents_v3.workset_v5_claims import DirectedClaimFlow
from citequest.agents_v3.workset_v7 import Workset7RequirementsGate, Workset7Sources
from citequest.research_v3.workset_v5_planning import resource_plan


def test_workset7_review_request_cap_matches_full_reservation_and_keeps_legacy_defaults():
    observed = []
    task = {"model_calls": 0, "output_tokens": 0,
        "profile": {"request_bytes": 131072, "model_calls": 64, "output_tokens": 160000}}

    def complete(request, _key):
        observed.append(request)
        return {"output": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "{}"}]}]}

    execution = SimpleNamespace(command={"id": "explicit-pure-command"},
        snapshot=lambda: task, settled_model_request=lambda _key: None)
    graph = SimpleNamespace(execution=execution, provider=SimpleNamespace(complete=complete), wait=lambda value: value)
    flow = DirectedClaimFlow.__new__(DirectedClaimFlow)
    flow.graph = graph
    owner = SimpleNamespace(graph=graph, project=lambda *_args, **_kwargs: ({"context_manifest": {}}, {}))
    flow.owner = owner
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"], "additionalProperties": False}
    for configured in (None, Workset7Sources.review_output_cap):
        owner.review_output_cap = configured
        for revision in (0, 1):
            state = {"job": {"anchor": {}, "evidence": [], "job_id": "D1:final_review"},
                "local_revision": revision}
            for role in ("document_review", "atomic_review"):
                raw, _ = flow.send(state, role, schema, {}, reviewer=True)
                expected = 4096 if configured else (1536 if role == "document_review" and not revision else 2048)
                assert raw == "{}" and observed[-1]["max_output_tokens"] == expected
                assert observed[-1]["text"]["format"]["name"] == "citequest_workset5_" + role
                assert observed[-1]["truncation"] == "disabled"
    assert Workset7RequirementsGate.review_output_cap == Workset7Sources.review_output_cap == 4096
    dimensions = dict(documents=2, windows=2, groups=2, comparison_units=12,
        source_parts=2, revision_credits=2, kind="compare")
    legacy = resource_plan(**dimensions)
    assert resource_plan(**dimensions, review_output_cap=None) == legacy
    # Two source maps, two document reviews, comparison plan/review/coverage,
    # twelve atomic author/review pairs and two shared author/review repairs.
    assert legacy["reserved_output_tokens"] == 91136
    current = resource_plan(**dimensions, review_output_cap=Workset7Sources.review_output_cap)
    assert current["reserved_output_tokens"] == 2 * 3072 + 2 * 4096 + 3072 + 2048 + 3072 + 12 * (3072 + 4096) + 2 * (4096 + 4096)
    assert current["required_models"] == legacy["required_models"] == 35
    assert current["required_tools"] == legacy["required_tools"]
    assert current["candidate_output_limit"] == legacy["candidate_output_limit"] == 160000
    assert current["reserved_output_tokens"] == 124928 and current["capacity_admitted"]
    minimum = resource_plan(documents=1, windows=1, groups=1, comparison_units=0,
        source_parts=1, revision_credits=0, kind="summary", review_output_cap=4096)
    assert minimum["reserved_output_tokens"] == 3072 + 4096 + 3072
