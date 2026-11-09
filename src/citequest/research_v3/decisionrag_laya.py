"""Opt-in local Laya decision judge; no implicit install, download or fallback.

Uses the author's fixed 0.4.22 wheel and actual tokenizer/forward. This is a
System One engineering adapter, not the DecisionRAG paper's Jev1.13 benchmark.
All model/cache paths belong to an explicitly prepared F-backed runtime.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from citequest.agents_v3.contracts import AgentError

from .decisionrag import DecisionReceipt, DecisionRequest, validate_receipt

SDK_VERSION = "0.4.22"
MODEL_ID = "convaiinnovations/laya"
REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
WEIGHTS_SHA256 = "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204"
WHEEL_SHA256 = "7e8cad8ca4cb76588ef608e70ad6753bae612885e015aeeeae7b43268ec5fd5a"
FILES = {"model.safetensors", "rl_agent_config.json", "encoder/config.json",
    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}


def verify_before_import(assets: Path, receipt: Path, receipt_sha256: str) -> Path:
    """Stdlib-only integrity gate shared by both deployable gateway platforms."""
    wheel = assets / "vendor/laya-0.4.22-py3-none-any.whl"
    if (wheel.is_symlink() or receipt.is_symlink() or not wheel.is_file() or not receipt.is_file()
            or wheel.stat().st_size > 1024**2 or receipt.stat().st_size > 131072
            or file_sha(wheel) != WHEEL_SHA256 or file_sha(receipt) != receipt_sha256):
        raise AgentError("laya_artifact_integrity_failed_before_dependency_import")
    return wheel


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


@dataclass(frozen=True)
class LayaLocalConfig:
    checkpoint_directory: Path
    wheel_file: Path
    asset_receipt_file: Path
    asset_receipt_sha256: str
    cache_root: Path
    max_len: int = 1024
    max_questions: int = 8
    cpu_threads: int = 2
    navigation_protocol: str = "laya1"


class LayaLocalJudge:
    def __init__(self, config: LayaLocalConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._forward_count = 0
        self._closed = False
        if ((config.navigation_protocol, config.max_len) not in {("laya1", 1024), ("laya2", 8192)}
                or type(config.max_questions) is not int or not 1 <= config.max_questions <= 8
                or type(config.cpu_threads) is not int or not 1 <= config.cpu_threads <= 2):
            raise AgentError("laya_unvalidated_resource_profile")
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise AgentError("laya_explicit_offline_runtime_required")
        cache = config.cache_root.resolve(strict=True)
        for key in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR"):
            configured = os.environ.get(key)
            if not configured or not Path(configured).resolve().is_relative_to(cache):
                raise AgentError("laya_cache_outside_explicit_runtime_root")
        checkpoint = config.checkpoint_directory.resolve(strict=True)
        wheel = config.wheel_file.resolve(strict=True)
        if os.name == "nt" and (checkpoint.drive.upper() != "F:" or cache.drive.upper() != "F:" or wheel.drive.upper() != "F:"):
            raise AgentError("laya_F_drive_assets_and_cache_required")
        if file_sha(config.asset_receipt_file) != config.asset_receipt_sha256:
            raise AgentError("laya_asset_receipt_changed")
        receipt = json.loads(config.asset_receipt_file.read_bytes())
        if (receipt.get("status") != "PASS_download_and_integrity_only_NOT_runtime_or_quality"
                or receipt.get("model_id") != MODEL_ID or receipt.get("revision") != REVISION
                or receipt.get("subfolder") != "multilingual" or receipt.get("license") != "Apache-2.0"
                or receipt.get("laya_version") != SDK_VERSION or set(receipt.get("files",{})) != FILES
                or receipt["files"]["model.safetensors"]["sha256"] != WEIGHTS_SHA256
                or receipt.get("wheel",{}).get("sha256") != WHEEL_SHA256):
            raise AgentError("laya_fixed_release_binding_required")
        actual_files = {str(path.relative_to(checkpoint)).replace("\\","/") for path in checkpoint.rglob("*") if path.is_file()}
        if actual_files != FILES or any(path.is_symlink() for path in checkpoint.rglob("*")):
            raise AgentError("laya_exact_materialized_checkpoint_required")
        expected = {name:item["sha256"] for name,item in receipt["files"].items()}
        if (any((checkpoint/name).stat().st_size != receipt["files"][name]["bytes"] for name in FILES)
                or sum(item["bytes"] for item in receipt["files"].values()) > 768*1024**2
                or file_sha(wheel) != WHEEL_SHA256):
            raise AgentError("laya_asset_size_or_wheel_integrity_changed")
        # Refuse the upstream tokenizer compatibility rewrite before it can
        # alter a pinned checkpoint. This release's multilingual config is
        # already native and needs no remote Python or metadata transformation.
        tokenizer = json.loads((checkpoint/"tokenizer/tokenizer_config.json").read_bytes())
        encoder = json.loads((checkpoint/"encoder/config.json").read_bytes())
        native = json.loads((checkpoint/"rl_agent_config.json").read_bytes())
        if (tokenizer.get("tokenizer_class") != "PreTrainedTokenizerFast"
                or isinstance(tokenizer.get("extra_special_tokens"),list)
                or encoder.get("model_type") != "modernbert" or encoder.get("auto_map")
                or tokenizer.get("auto_map") or native.get("encoder") != "jhu-clsp/mmBERT-base"
                or native.get("max_len") != 1024 or native.get("head_max_len") != 256):
            raise AgentError("laya_checkpoint_requires_explicit_compatibility_review")
        if str(wheel) not in sys.path:
            raise AgentError("laya_verified_wheel_import_path_required")
        # Dependencies must already belong to the matching prepared parser
        # runtime. No package manager or hub client is invoked by this adapter.
        import laya
        import torch
        from laya.common import build_sequence
        if (importlib.metadata.version("laya") != SDK_VERSION
                or not str(laya.__file__).replace("\\","/").startswith(str(wheel).replace("\\","/")+"/laya/")):
            raise AgentError("laya_verified_wheel_origin_required")
        if torch.get_num_threads() > config.cpu_threads or torch.get_num_interop_threads() > config.cpu_threads:
            raise AgentError("laya_cpu_thread_budget_required")
        started = time.monotonic()
        self.agent = laya.load(str(checkpoint), device="cpu", backend="eager", fast=False, compile=False,
            expected_sha256=expected)
        self.load_seconds = time.monotonic()-started
        self._build_sequence = build_sequence
        if (self.agent.device.type != "cpu" or self.agent.backend != "eager" or self.agent.hooks
                or self.agent.cfg.get("max_len") != 1024 or self.agent.cfg.get("head_max_len") != 256):
            self.close()
            raise AgentError("laya_runtime_mode_or_hooks_changed")
        self.model_binding = {"provider":"local_author_Laya_SystemOne", "model_id":MODEL_ID,
            "revision":REVISION, "subfolder":"multilingual", "weights_sha256":WEIGHTS_SHA256,
            "sdk_version":SDK_VERSION, "sdk_wheel_sha256":WHEEL_SHA256,
            "device":"cpu", "backend":"eager", "max_len":1024, "head_max_len":256,
            "paper_Jev_1_13_reproduction":False, "shared_state":"same complete alternatives in every question row",
            "KV_state_read_once_claim":False, "remote_code":False, "automatic_device_fallback":False}
        if config.navigation_protocol == "laya2":
            if encoder.get("max_position_embeddings") != 8192:
                self.close()
                raise AgentError("laya_encoder_long_context_binding_required")
            self.model_binding.update(max_len=8192, native_default_max_len=1024,
                encoder_max_position_embeddings=8192, request_context_protocol="laya2")
        self._forward_handle = self.agent.model.register_forward_pre_hook(self._observed_forward)

    def _observed_forward(self, _module: object, _args: tuple) -> None:
        self._forward_count += 1

    def _preflight(self, request: DecisionRequest) -> dict:
        if self._closed:
            raise AgentError("laya_judge_closed")
        state, questions, _ = request.wire_data()
        if not 1 <= len(questions) <= self.config.max_questions:
            raise AgentError("laya_shared_question_capacity")
        if self.agent.tok.mask_token in json.dumps(state, ensure_ascii=False):
            raise AgentError("laya_reserved_mask_token_in_source")
        stats = {}
        for identifier, question in questions.items():
            internal = self.agent._to_internal(question)  # Pinned official 0.4.22 conversion.
            ids, markers, options, dropped = self._build_sequence(self.agent.tok, state, internal,
                max_len=self.config.max_len, head_max_len=256, return_stats=True, return_truncation_stats=True)
            unbounded_head, _ = self._build_sequence(self.agent.tok, "", internal, max_len=8192, head_max_len=8192)
            real_head = ids[:len(ids)-dropped["state_tokens_used"]-1]
            if (dropped["truncated"] or dropped["state_tokens_dropped"] != 0
                    or len(markers) != 2 or options["options"] != 2 or options["options_distinct"] != 2
                    or real_head != unbounded_head[:-1]):
                raise AgentError("laya_complete_context_exceeds_token_budget")
            stats[identifier] = {"sequence_tokens":len(ids), **dropped}
        return stats

    def preflight(self, request: DecisionRequest) -> dict:
        with self._lock:
            return self._preflight(request)

    def score(self, request: DecisionRequest) -> DecisionReceipt:
        with self._lock:
            token_checks = self._preflight(request)
            state, questions, labels = request.wire_data()
            if self.agent.hooks:
                raise AgentError("laya_unapproved_prediction_hook")
            forward_before = self._forward_count
            fallback_before = self.agent.cpu_fallback_count
            started = time.monotonic()
            raw = self.agent.predict(state, questions, max_len=self.config.max_len, head_max_len=256)
            seconds = time.monotonic()-started
            usage = raw.get("usage")
            answers = raw.get("answers")
            if (not isinstance(usage,dict) or usage.get("truncated") is not False
                    or usage.get("state_tokens_dropped") != 0 or usage.get("output_tokens") != 0
                    or type(usage.get("input_tokens")) is not int
                    or usage["input_tokens"] != sum(item["sequence_tokens"] for item in token_checks.values())
                    or type(usage.get("state_tokens")) is not int or usage["state_tokens"] < 1
                    or usage.get("options") or not isinstance(answers,dict) or set(answers) != set(labels)
                    or self.agent.cpu_fallback_count != fallback_before or self.agent.device.type != "cpu"
                    or self._forward_count-forward_before != 1):
                raise AgentError("laya_unapproved_truncation_fallback_or_forward_count")
            if any(not isinstance(answer,dict) or answer.get("type") != "noul" for answer in answers.values()):
                raise AgentError("laya_typed_probability_contract_changed")
            result = DecisionReceipt(request.fingerprint(), {identifier:answers[label]["noul"] for label,identifier in labels.items()},
                self.model_binding, {**usage, "exact_token_preflight":token_checks, "observed_forward_calls":1,
                    "seconds":seconds, "external_API_cash_ncny":0, "physical_compute_cost_not_measured":True})
            validate_receipt(request, result)
            return result

    def close(self) -> None:
        self._closed = True
        handle = getattr(self,"_forward_handle",None)
        if handle is not None:
            handle.remove()
        agent = getattr(self,"agent",None)
        if agent is not None:
            agent.__exit__(None,None,None)
