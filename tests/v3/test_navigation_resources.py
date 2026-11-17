"""Capacity mathematics and receipt boundaries; no model or performance result."""
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import AgentError
from citequest.research_v3.navigation_resources import (
    CalibrationRecord, CalibrationSet, NavigationResourceProfile, RequestShape,
    RuntimeIdentity, evaluate_admission, load_calibration,
)


def profile(**changes):
    return NavigationResourceProfile.model_validate({"version": "citequest-navigation-resources/1", "profile_id": "fixture-cpu",
        "device": "cpu", "execution_mode": "supported", "cpu_threads": 2, "memory_budget_bytes": 4 * 1024**3,
        "address_space_limit_bytes": 8 * 1024**3, "uncalibrated_latency_policy": "allow_with_physical_deadline", **changes})


def identity(**changes):
    return RuntimeIdentity.model_validate({"device": "cpu", "model_weights_sha256": "a" * 64,
        "model_revision": "fixed", "sdk_version": "fixed", "torch_version": "fixed", "backend": "eager",
        "dtype": "float32", "cpu_threads": 2, "hardware_sha256": "b" * 64, **changes})


def calibration(**changes):
    record = CalibrationRecord.model_validate({"record_id": "explicit-measurement-contract-fixture",
        "observed_at": datetime(2026, 11, 17, tzinfo=UTC), "source_receipt_sha256": "c" * 64,
        "runtime": identity(), "shape": RequestShape(sequence_tokens=[2048, 2000]),
        "memory_budget_bytes": 4 * 1024**3, "outcome": "completed", "elapsed_ms": 1000.0,
        "peak_memory_bytes": 2 * 1024**3, "peak_scope": "process_lifetime_RSS", **changes})
    return CalibrationSet(version="citequest-navigation-calibration/1", records=[record])


def test_same_complete_tokens_can_have_different_padded_and_quadratic_shape():
    left, right = RequestShape(sequence_tokens=[1000, 1000]), RequestShape(sequence_tokens=[1, 1999])
    assert left.dimensions()["complete_input_tokens"] == right.dimensions()["complete_input_tokens"]
    assert left.dimensions()["padded_tokens"] < right.dimensions()["padded_tokens"]
    assert evaluate_admission(left, profile())["estimated_peak_memory_bytes"] < evaluate_admission(right, profile())["estimated_peak_memory_bytes"]


def test_shape_and_memory_rejection_never_reduce_candidates():
    shape = RequestShape(sequence_tokens=[8192] * 8)
    result = evaluate_admission(shape, profile())
    assert not result["admitted"] and "navigation_estimated_peak_memory_capacity" in result["reasons"]
    assert result["shape"]["question_rows"] == 8 and result["candidates_reduced"] is False
    assert result["memory_estimate_basis"] == "analytical_unmeasured"
    assert result["estimated_latency_ms"] is None


def test_larger_CPU_budget_changes_capacity_without_inventing_speed():
    shape = RequestShape(sequence_tokens=[4096] * 4)
    small = evaluate_admission(shape, profile())
    large = evaluate_admission(shape, profile(cpu_threads=8, memory_budget_bytes=32 * 1024**3, address_space_limit_bytes=64 * 1024**3))
    assert not small["admitted"] and large["admitted"]
    assert large["latency_prediction_status"] == "unknown" and large["cross_hardware_speedup_claimed"] is False


def test_calibration_requires_same_runtime_and_dominating_shape():
    shape = RequestShape(sequence_tokens=[1024, 900])
    measured = evaluate_admission(shape, profile(), calibration=calibration(), runtime_identity=identity(), remaining_ms=2000)
    assert measured["admitted"] and measured["estimated_latency_ms"] == 1500
    assert measured["calibration_source_receipt_sha256"] == ["c" * 64]
    for runtime in (identity(hardware_sha256="d" * 64), identity(cpu_threads=8), identity(backend="another")):
        missing = evaluate_admission(shape, profile(), calibration=calibration(), runtime_identity=runtime)
        assert missing["estimated_latency_ms"] is None and not missing["calibration_record_ids"]
    bigger = evaluate_admission(RequestShape(sequence_tokens=[4096] * 3), profile(), calibration=calibration(), runtime_identity=identity())
    assert bigger["estimated_latency_ms"] is None


def test_deadline_unknown_policy_calibrated_deadline_and_concurrency_are_distinct():
    shape = RequestShape(sequence_tokens=[512])
    assert "navigation_latency_calibration_required" in evaluate_admission(shape, profile(uncalibrated_latency_policy="reject"), remaining_ms=25000)["reasons"]
    assert evaluate_admission(shape, profile(), remaining_ms=25000)["admitted"]
    assert "navigation_calibrated_latency_exceeds_deadline" in evaluate_admission(shape, profile(), calibration=calibration(), runtime_identity=identity(), remaining_ms=1000)["reasons"]
    assert "navigation_concurrency_capacity" in evaluate_admission(shape, profile(), active_requests=1)["reasons"]


def test_failed_small_memory_observation_is_not_reused_as_larger_memory_proof():
    failed = calibration(outcome="oom", peak_memory_bytes=None)
    shape = RequestShape(sequence_tokens=[2048, 2048])
    assert "navigation_recorded_OOM_shape" in evaluate_admission(shape, profile(), calibration=failed, runtime_identity=identity())["reasons"]
    larger = evaluate_admission(shape, profile(memory_budget_bytes=8 * 1024**3), calibration=failed, runtime_identity=identity())
    assert "navigation_recorded_OOM_shape" not in larger["reasons"] and larger["estimated_latency_ms"] is None


def test_GPU_target_is_planning_only_and_cannot_start_CPU_adapter():
    gpu = profile(profile_id="fixture-GPU-target", device="cuda", execution_mode="planning_only",
        memory_budget_bytes=24 * 1024**3, address_space_limit_bytes=32 * 1024**3,
        attention_workspace_bytes_per_cell=0, attention_assumption="unverified_fused_linear_target")
    shape = RequestShape(sequence_tokens=[8192] * 8)
    assert not evaluate_admission(shape, gpu)["admitted"]
    plan = evaluate_admission(shape, gpu, planning=True)
    assert plan["admitted"] and not plan["execution_supported"] and plan["estimated_latency_ms"] is None
    with pytest.raises(ValidationError):
        gpu.model_copy(update={"execution_mode": "supported"}).__class__.model_validate({**gpu.model_dump(), "execution_mode": "supported"})
    with pytest.raises(ValidationError):
        profile(max_concurrency=2)


def test_token_preflight_and_calibration_integrity_fail_closed(tmp_path: Path):
    with pytest.raises(AgentError, match="complete_token_preflight"):
        RequestShape.from_preflight({"N1": {"sequence_tokens": 512, "truncated": True, "state_tokens_dropped": 1}})
    path = tmp_path / "calibration.json"
    path.write_text(calibration().model_dump_json(), encoding="utf-8")
    with pytest.raises(AgentError, match="integrity"):
        load_calibration(path, "0" * 64)


def test_legacy_gateway_and_explicit_CPU_profile_parse_without_starting_processes():
    from citequest.research_v3.decisionrag_gateway import GatewayConfig
    root = Path(__file__).resolve().parents[2] / "runtime/decisionrag"
    for filename, length in (("gateway-linux.example.json", 1024), ("gateway-linux-laya2.example.json", 8192)):
        old = GatewayConfig.model_validate_json((root / filename).read_bytes(), strict=True)
        assert old.resources is None and old.sequence_tokens == length and old.calibration_file is None
    new = GatewayConfig.model_validate_json((root / "gateway-linux-cpu-8.example.json").read_bytes(), strict=True)
    assert new.resources.cpu_threads == 8 and new.resources.address_space_limit_bytes == 32 * 1024**3
    gpu = NavigationResourceProfile.model_validate_json((root / "resources-cuda-24g-target.example.json").read_bytes(), strict=True)
    with pytest.raises(ValidationError, match="GPU targets cannot execute"):
        GatewayConfig.model_validate({**new.model_dump(), "resources": gpu})
    with pytest.raises(ValidationError, match="exact SHA"):
        GatewayConfig.model_validate({**new.model_dump(), "calibration_file": Path("unbound.json")})


def test_resource_rejection_occurs_before_any_score_forward_and_legacy_threads_stay_fixed():
    from citequest.research_v3.decisionrag_laya import LayaLocalConfig, LayaLocalJudge
    with pytest.raises(AgentError, match="unvalidated_resource_profile"):
        LayaLocalJudge(LayaLocalConfig(Path("unused"), Path("unused"), Path("unused"), "0" * 64,
            Path("unused"), cpu_threads=8))
    judge = LayaLocalJudge.__new__(LayaLocalJudge)
    judge.config = SimpleNamespace(resources=profile(), calibration=None)
    judge.resource_runtime_identity = identity()
    judge._lock = Lock()
    judge._preflight = lambda _: {"N" + str(i): {"sequence_tokens": 8192, "truncated": False,
        "state_tokens_dropped": 0} for i in range(8)}
    with pytest.raises(AgentError, match="deadline_context_required"):
        judge.score(None)
    with pytest.raises(AgentError, match="estimated_peak_memory_capacity"):
        judge.score(None, remaining_ms=25000)  # No agent/model exists: reject before access.
    assert judge.last_resource_admission["shape"]["question_rows"] == 8
    assert judge.last_resource_admission["candidates_reduced"] is False
    with judge._lock:
        with pytest.raises(AgentError, match="concurrency_capacity"):
            judge.preflight(None)


@pytest.mark.parametrize("forward_entered", [False, True])
def test_gateway_zero_forward_proof_uses_counter_not_resource_error_name(forward_entered):
    from citequest.research_v3.decisionrag import DecisionRequest
    from citequest.research_v3.decisionrag_gateway import GatewayCall, GatewayConfig, _decision_response
    config = GatewayConfig.model_validate_json((Path(__file__).resolve().parents[2]
        / "runtime/decisionrag/gateway-linux-cpu-8.example.json").read_bytes(), strict=True)
    request = DecisionRequest("fixture query", "passages", (("A", "literal"),), "f" * 64)
    call = GatewayCall.model_validate({"protocol": config.protocol, "operation": "score", "remaining_ms": 25000,
        "request": {"query": request.query, "stage": request.stage, "candidates": list(request.candidates),
            "scope_binding": request.scope_binding}, "request_sha256": request.fingerprint()})

    class CounterBoundaryFixture:
        _forward_count = 12
        model_binding = {"provider": "explicit_counter_boundary_fixture_no_model"}
        last_resource_admission = {"admitted": False, "reasons": ["navigation_estimated_peak_memory_capacity"]}

        def score(self, *_args, **_kwargs):
            if forward_entered:
                self._forward_count += 1
            raise AgentError("navigation_estimated_peak_memory_capacity")

    response = _decision_response(CounterBoundaryFixture(), call, config)
    assert response["request_sha256"] == request.fingerprint()
    if forward_entered:
        assert response["forward_started"] is True and "forwards" not in response
    else:
        assert response["forward_started"] is False and response["forwards"] == 0


def test_internal_startup_diagnostic_never_records_messages_paths_or_locals():
    import json
    from citequest.research_v3.decisionrag_gateway import _safe_process_diagnostic
    try:
        raise MemoryError("private-token /private/model/path https://secret.example/password cannot allocate")
    except MemoryError as error:
        result = _safe_process_diagnostic(error)
    encoded = json.dumps(result)
    assert result["memory_error_hint"] is True and result["exception_type"] == "MemoryError" and result["frames"]
    for secret in ("private-token", "/private", "secret.example", "password"):
        assert secret not in encoded
