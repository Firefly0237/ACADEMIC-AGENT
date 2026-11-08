"""Explicit new GPU-Candle-F32 candidate, never a CPU equivalence claim."""

from __future__ import annotations

import json
from pathlib import Path

from citequest.index_v2.rows import canonical_sha
from citequest.index_v2.schema import IndexProfile

IMAGE = "ghcr.io/huggingface/text-embeddings-inference@sha256:bd8e5b1954146f7fe8590b64b959bc194433c6c38c036592a84d736841ca9400"
ENDPOINT = "http://tei-strict-gpu:8080"
INFO_URL = "http://127.0.0.1:18098/info"


def candidate_profile() -> dict:
    return {
        "schema_version": "cq.encoder-backend/1",
        "profile_id": "GPU-Candle-F32-Blackwell-candidate/1",
        "semantic_index_profile_sha256": IndexProfile().fingerprint(),
        "candidate": True,
        "CPU_equivalence_claimed": False,
        "image": IMAGE,
        "tei_version": "1.9.4",
        "backend": "Candle-CUDA",
        "dtype": "float32",
        "dimension": 1024,
        "model_id": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "pytorch_weight_sha256": "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38",
        "pytorch_weight_bytes": 2271145830,
        "device_compute_capability": "12.0",
        "flash_attention": False,
        "gelu": "official_Candle_tanh_approximation_as_logged",
        "GELU_as_cause_of_CPU_difference": "candidate_explanation_not_proven",
        "pooling": "cls",
        "normalization": True,
        "maximum_input_tokens": 2048,
        "maximum_batch_tokens": 2048,
        "maximum_client_batch_size": 2,
        "maximum_batch_requests_effective": 2,
        "maximum_concurrent_requests": 4,
        "strict_per_request_truncate": False,
        "router_auto_truncate_default": True,
        "function_endpoint": ENDPOINT,
        "failed_equivalence_evidence": "reports/stages/v3/S05-full/gpu-comparison-r1/verification.json",
        "publication_requires_migrator_adoption": True,
        "quality_gate": {
            "standard_units": 64183,
            "queries": 597,
            "qrels_edges": 639,
            "modes": ["bm25", "dense", "hybrid"],
            "recall_k": 20,
            "ndcg_k": 10,
            "minimum_hybrid_minus_BM25_recall20": -0.02,
            "paired_difference_intervals_required": True,
            "excluded_queries": 0,
        },
    }


def read_profile(path: Path) -> tuple[dict, str, str]:
    actual = json.loads(path.read_text(encoding="utf-8"))
    expected = candidate_profile()
    if actual != expected:
        raise ValueError("exact_explicit_GPU_Candle_F32_candidate_profile_required")
    canonical = json.dumps(actual, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return actual, canonical_sha(actual), canonical
