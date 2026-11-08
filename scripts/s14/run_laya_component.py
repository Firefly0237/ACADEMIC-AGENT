"""One explicit offline CPU component run; never a product or benchmark pass.

Run only in the existing locked parser runtime with externally enforced memory,
CPU and physical timeout limits. This entry does not launch Docker or fetch data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

FIXED_WHEEL_SHA256 = "7e8cad8ca4cb76588ef608e70ad6753bae612885e015aeeeae7b43268ec5fd5a"


def verify_before_import(assets: Path, receipt: Path, expected_receipt_sha256: str) -> Path:
    """Use only stdlib BEFORE putting any wheel into dependency search paths."""
    wheel = assets / "vendor/laya-0.4.22-py3-none-any.whl"
    if (wheel.is_symlink() or receipt.is_symlink() or not wheel.is_file() or not receipt.is_file()
            or wheel.stat().st_size > 1024**2 or receipt.stat().st_size > 131072):
        raise SystemExit("fixed_local_wheel_and_receipt_required_before_import")
    with wheel.open("rb") as stream:
        actual_wheel = hashlib.file_digest(stream,"sha256").hexdigest()
    with receipt.open("rb") as stream:
        actual_receipt = hashlib.file_digest(stream,"sha256").hexdigest()
    if actual_wheel != FIXED_WHEEL_SHA256 or actual_receipt != expected_receipt_sha256:
        raise SystemExit("artifact_integrity_failed_before_dependency_import")
    return wheel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-local-cpu", action="store_true", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--asset-receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.input.stat().st_size > 32768:
        raise SystemExit("new_output_and_bounded_component_input_required")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    root = args.cache_root.resolve(strict=True)
    for key in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR", "TEMP", "TMP"):
        if not os.environ.get(key) or not Path(os.environ[key]).resolve().is_relative_to(root):
            raise SystemExit("explicit_external_F_backed_cache_and_temp_environment_required")
    if os.name != "posix":
        raise SystemExit("locked_Linux_parser_runtime_required")
    expected = {"torch":"2.15.0+cpu", "transformers":"5.22.1", "huggingface-hub":"1.33.0",
                "tokenizers":"0.23.3", "safetensors":"0.8.0", "numpy":"2.5.4"}
    if any(importlib.metadata.version(name) != version for name,version in expected.items()):
        raise SystemExit("existing_parser_lock_mismatch_no_install_or_fallback")
    assets = args.assets.resolve(strict=True)
    wheel = verify_before_import(assets,args.asset_receipt,args.expected_receipt_sha256)
    sys.path.insert(0, str(wheel))
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    from citequest.agents_v3.contracts import AgentError, canonical, digest
    from citequest.research_v3.decisionrag import DecisionRequest, Document, NavigationLimits, Passage, navigate
    from citequest.research_v3.decisionrag_laya import LayaLocalConfig, LayaLocalJudge
    raw = args.input.read_bytes()
    value = json.loads(raw)
    if (not isinstance(value,dict) or set(value) != {"query", "documents"}
            or not isinstance(value["documents"],list) or not 1 <= len(value["documents"]) <= 2):
        raise SystemExit("one_or_two_explicit_component_documents_required")
    documents = []
    for item in value["documents"]:
        if not isinstance(item,dict) or set(item) != {"id","description","source_version","passages"}:
            raise SystemExit("component_document_contract_invalid")
        passages = tuple(Passage(**passage) for passage in item["passages"])
        documents.append(Document(item["id"],item["description"],item["source_version"],passages))
    component_binding = digest({"input_sha256":hashlib.sha256(raw).hexdigest(),"component_only":True})
    admissions, judge = [], None
    started = time.monotonic()
    result = {"status":"not_run", "component_only":True, "product_authorization_verified":False,
              "semantic_quality_assessed":False,"paper_benchmark_reproduction":False,
              "input_sha256":hashlib.sha256(raw).hexdigest(),"dependency_versions":expected,
              "external_network_required":False,"physical_limits_owned_by_supervisor":True}
    try:
        judge = LayaLocalJudge(LayaLocalConfig(assets/"model",wheel,args.asset_receipt,
            args.expected_receipt_sha256,root))
        outcome = navigate(value["query"],tuple(documents),judge,scope_binding=component_binding,
            check_scope=lambda:component_binding,
            before_decision=lambda request:admissions.append({"stage":request.stage,"sha256":request.fingerprint()}),
            limits=NavigationLimits(document_keep=1,passage_keep=2,max_documents=2,max_passages=8,active_seconds=30))
        forward_before = judge._forward_count
        try:
            judge.preflight(DecisionRequest(value["query"],"passages",(("over-limit","literal "*6000),),component_binding))
        except AgentError as error:
            if error.code != "laya_complete_context_exceeds_token_budget" or judge._forward_count != forward_before:
                raise
            overflow = {"status":"rejected_before_inference","error":error.code,"additional_forwards":0}
        else:
            raise AgentError("laya_overflow_preflight_did_not_reject")
        result.update(status="PASS_local_CPU_mechanism_NOT_product_or_semantic_quality",outcome=outcome,
                      model_load_seconds=judge.load_seconds,overflow_control=overflow)
    except BaseException as exc:
        result.update(status="FAIL_local_CPU_component", error=exc.code if isinstance(exc,AgentError) else type(exc).__name__)
        raise
    finally:
        if judge is not None:
            judge.close()
        import resource
        result.update(total_seconds=time.monotonic()-started, peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      local_decision_admissions=admissions)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("xb") as stream:
            stream.write(canonical(result)+b"\n")


if __name__ == "__main__":
    main()
