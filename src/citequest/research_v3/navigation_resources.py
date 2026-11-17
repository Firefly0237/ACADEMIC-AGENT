"""Versioned navigation capacity admission and explicitly unmeasured forecasts.

Sequence rows are padded together by the current adapter. Complete token count
alone therefore does not describe its memory or compute shape. Calibration is
restricted to the same model, runtime, hardware and thread allocation; forecasts
never invent measured latency or extrapolate speedups to another device.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from citequest.agents_v3.contracts import AgentError, Strict, canonical

PARAMETER_ELEMENTS = 321908998


class RequestShape(Strict):
    sequence_tokens: list[int] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def complete_rows(self):
        if any(type(value) is not int or not 1 <= value <= 8192 for value in self.sequence_tokens):
            raise ValueError("complete positive sequence lengths required")
        return self

    @classmethod
    def from_preflight(cls, checks: dict):
        if (type(checks) is not dict or not checks or any(type(row) is not dict
                or row.get("truncated") is not False or row.get("state_tokens_dropped") != 0 for row in checks.values())):
            raise AgentError("navigation_complete_token_preflight_required")
        return cls(sequence_tokens=[row["sequence_tokens"] for row in checks.values()])

    def dimensions(self):
        rows, longest = len(self.sequence_tokens), max(self.sequence_tokens)
        return {"question_rows": rows, "longest_sequence_tokens": longest,
            "complete_input_tokens": sum(self.sequence_tokens), "padded_tokens": rows * longest,
            "attention_cells": rows * longest * longest}


class NavigationResourceProfile(Strict):
    version: Literal["citequest-navigation-resources/1"]
    profile_id: str = Field(min_length=1, max_length=120)
    device: Literal["cpu", "cuda"]
    execution_mode: Literal["supported", "planning_only"]
    cpu_threads: int = Field(ge=1, le=64)
    hardware_instance_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    memory_budget_bytes: int = Field(ge=2 * 1024**3, le=512 * 1024**3)
    address_space_limit_bytes: int = Field(ge=4 * 1024**3, le=1024 * 1024**3)
    max_concurrency: int = Field(default=1, ge=1, le=16)
    max_question_rows: int = Field(default=8, ge=1, le=8)
    max_sequence_tokens: Literal[1024, 8192] = 8192
    max_padded_tokens: int = Field(default=65536, ge=1, le=65536)
    startup_timeout_seconds: int = Field(default=90, ge=1, le=600)
    minimum_remaining_ms: int = Field(default=250, ge=1, le=25000)
    uncalibrated_latency_policy: Literal["allow_with_physical_deadline", "reject"] = "reject"
    runtime_overhead_bytes: int = Field(default=536870912, ge=268435456, le=16 * 1024**3)
    linear_workspace_bytes_per_token: int = Field(default=32768, ge=1, le=1048576)
    attention_workspace_bytes_per_cell: int = Field(default=96, ge=0, le=1024)
    memory_headroom_ratio: float = Field(default=1.25, ge=1.0, le=4.0, allow_inf_nan=False)
    latency_headroom_ratio: float = Field(default=1.5, ge=1.0, le=4.0, allow_inf_nan=False)
    attention_assumption: Literal["unverified_math_workspace_upper", "unverified_fused_linear_target"] = "unverified_math_workspace_upper"

    @model_validator(mode="after")
    def truthful_adapter(self):
        if self.device == "cuda" and self.execution_mode != "planning_only":
            raise ValueError("CUDA execution adapter is not implemented")
        if self.device == "cpu" and self.execution_mode == "supported" and self.max_concurrency != 1:
            raise ValueError("the current hot CPU worker supports one in-flight request")
        if self.device == "cpu" and (self.attention_workspace_bytes_per_cell < 48
                or self.attention_assumption != "unverified_math_workspace_upper"):
            raise ValueError("CPU forecast retains an unverified math-workspace allowance")
        if self.address_space_limit_bytes < self.memory_budget_bytes:
            raise ValueError("address-space budget must cover the declared resident-memory budget")
        return self

    def fingerprint(self):
        return hashlib.sha256(canonical(self.model_dump(mode="json"))).hexdigest()


class RuntimeIdentity(Strict):
    device: Literal["cpu", "cuda"]
    model_weights_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_revision: str = Field(min_length=1, max_length=120)
    sdk_version: str = Field(min_length=1, max_length=80)
    torch_version: str = Field(min_length=1, max_length=80)
    backend: str = Field(min_length=1, max_length=80)
    dtype: str = Field(min_length=1, max_length=80)
    cpu_threads: int = Field(ge=1, le=64)
    hardware_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class CalibrationRecord(Strict):
    record_id: str = Field(min_length=1, max_length=120)
    observed_at: AwareDatetime
    source_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    runtime: RuntimeIdentity
    shape: RequestShape
    memory_budget_bytes: int = Field(gt=0)
    outcome: Literal["completed", "oom", "timeout"]
    elapsed_ms: float = Field(gt=0, allow_inf_nan=False)
    peak_memory_bytes: int | None = Field(default=None, gt=0)
    peak_scope: Literal["process_lifetime_RSS", "cgroup_lifetime_peak", "cuda_reserved_peak"]

    @model_validator(mode="after")
    def completed_memory(self):
        if self.outcome == "completed" and self.peak_memory_bytes is None:
            raise ValueError("completed calibration requires a measured memory peak")
        if (self.runtime.device == "cpu" and self.peak_scope == "cuda_reserved_peak"
                or self.runtime.device == "cuda" and self.peak_scope != "cuda_reserved_peak"):
            raise ValueError("peak metric must belong to the actual execution device")
        return self


class CalibrationSet(Strict):
    version: Literal["citequest-navigation-calibration/1"]
    records: list[CalibrationRecord] = Field(max_length=256)


def load_calibration(path: Path, expected_sha256: str) -> CalibrationSet:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024**2:
        raise AgentError("navigation_calibration_regular_bounded_file_required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise AgentError("navigation_calibration_integrity_failed")
    return CalibrationSet.model_validate_json(raw, strict=True)


def dominates(larger: RequestShape, smaller: RequestShape):
    left, right = larger.dimensions(), smaller.dimensions()
    return all(left[key] >= right[key] for key in ("question_rows", "longest_sequence_tokens", "padded_tokens", "attention_cells"))


def startup_memory_estimate(profile: NavigationResourceProfile) -> dict:
    weights = PARAMETER_ELEMENTS * (4 if profile.device == "cpu" else 2)
    checkpoint = PARAMETER_ELEMENTS * 2
    return {"estimated_peak_memory_bytes": math.ceil((weights + checkpoint + profile.runtime_overhead_bytes) * profile.memory_headroom_ratio),
        "basis": "unmeasured_parameter_checkpoint_and_runtime_allowance_not_measured_RSS",
        "model_parameter_bytes": weights, "checkpoint_staging_bytes": checkpoint,
        "execution_supported": profile.execution_mode == "supported"}


def evaluate_admission(shape: RequestShape, profile: NavigationResourceProfile, *, calibration: CalibrationSet | None = None,
        runtime_identity: RuntimeIdentity | None = None, remaining_ms: int | None = None,
        active_requests: int = 0, planning: bool = False) -> dict:
    """Return a reviewable decision; no candidate, deadline or profile is reduced."""
    if type(active_requests) is not int or active_requests < 0 or (remaining_ms is not None
            and (type(remaining_ms) is not int or remaining_ms < 0)):
        raise AgentError("navigation_resource_context_invalid")
    dimensions, reasons = shape.dimensions(), []
    if not planning and profile.execution_mode != "supported":
        reasons.append("navigation_execution_adapter_unavailable")
    if active_requests >= profile.max_concurrency:
        reasons.append("navigation_concurrency_capacity")
    if (dimensions["question_rows"] > profile.max_question_rows or dimensions["longest_sequence_tokens"] > profile.max_sequence_tokens
            or dimensions["padded_tokens"] > profile.max_padded_tokens):
        reasons.append("navigation_padded_shape_capacity")
    weight_bytes = PARAMETER_ELEMENTS * (4 if profile.device == "cpu" else 2)
    components = {"model_parameter_bytes": weight_bytes, "runtime_overhead_bytes": profile.runtime_overhead_bytes,
        "linear_workspace_bytes": dimensions["padded_tokens"] * profile.linear_workspace_bytes_per_token,
        "attention_workspace_bytes": dimensions["attention_cells"] * profile.attention_workspace_bytes_per_cell}
    predicted_memory = math.ceil(sum(components.values()) * profile.memory_headroom_ratio)
    memory_basis, matching = "analytical_unmeasured", []
    if (calibration is not None and runtime_identity is not None and runtime_identity.device == profile.device
            and runtime_identity.cpu_threads == profile.cpu_threads):
        matching = [record for record in calibration.records if record.runtime == runtime_identity]
    covering = [record for record in matching if record.outcome == "completed" and dominates(record.shape, shape)]
    if covering:
        predicted_memory = math.ceil(max(record.peak_memory_bytes for record in covering) * profile.memory_headroom_ratio)
        memory_basis = "same_runtime_measured_shape_envelope_not_a_guarantee"
    if predicted_memory * (active_requests + 1) > profile.memory_budget_bytes:
        reasons.append("navigation_estimated_peak_memory_capacity")
    latency_ms = max((record.elapsed_ms for record in covering), default=None)
    if latency_ms is not None:
        latency_ms = math.ceil(latency_ms * profile.latency_headroom_ratio)
    if remaining_ms is not None:
        if remaining_ms < profile.minimum_remaining_ms:
            reasons.append("navigation_remaining_deadline_capacity")
        elif latency_ms is not None and latency_ms > remaining_ms:
            reasons.append("navigation_calibrated_latency_exceeds_deadline")
        elif latency_ms is None and profile.uncalibrated_latency_policy == "reject":
            reasons.append("navigation_latency_calibration_required")
    for record in matching:
        if record.outcome == "oom" and profile.memory_budget_bytes <= record.memory_budget_bytes and dominates(shape, record.shape):
            reasons.append("navigation_recorded_OOM_shape")
        if (record.outcome == "timeout" and remaining_ms is not None and remaining_ms <= record.elapsed_ms
                and dominates(shape, record.shape)):
            reasons.append("navigation_recorded_timeout_shape")
    return {"version": "citequest-navigation-admission/1", "admitted": not reasons, "reasons": list(dict.fromkeys(reasons)),
        "planning_only": planning, "execution_supported": profile.execution_mode == "supported", "device": profile.device,
        "profile_id": profile.profile_id, "profile_sha256": profile.fingerprint(), "shape": dimensions,
        "estimated_peak_memory_bytes": predicted_memory, "memory_estimate_basis": memory_basis,
        "analytical_memory_components": components, "memory_headroom_ratio": profile.memory_headroom_ratio,
        "attention_assumption": profile.attention_assumption, "memory_budget_bytes": profile.memory_budget_bytes,
        "estimated_latency_ms": latency_ms, "latency_prediction_status": "same_runtime_measured_envelope_not_SLA" if covering else "unknown",
        "latency_headroom_ratio": profile.latency_headroom_ratio, "remaining_ms": remaining_ms,
        "active_requests": active_requests, "max_concurrency": profile.max_concurrency,
        "calibration_record_ids": [record.record_id for record in covering],
        "calibration_source_receipt_sha256": sorted({record.source_receipt_sha256 for record in covering}),
        "cross_hardware_speedup_claimed": False, "candidates_reduced": False}
