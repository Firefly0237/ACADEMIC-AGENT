"""Explicit fixed-revision F-workspace fetch, never an import-time download.

Root authorized the five Apache-2.0 multilingual checkpoint files (<=0.75GiB)
and the fixed official wheel. Does not install packages, use Docker or load a
model. Network failures are recorded without signed redirect URLs or retries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
MODEL = "convaiinnovations/laya"
WEIGHTS_SHA256 = "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204"
WHEEL_SHA256 = "7e8cad8ca4cb76588ef608e70ad6753bae612885e015aeeeae7b43268ec5fd5a"
WHEEL_URL = "https://files.pythonhosted.org/packages/65/74/ed14b1033ac5303ccc29ecbf3ba6ff24d2d19e58bab0259b3d4e0b4fcb33/laya-0.4.22-py3-none-any.whl"
FILES = (("rl_agent_config.json",65536,None), ("encoder/config.json",65536,None),
    ("tokenizer/tokenizer_config.json",65536,None), ("tokenizer/tokenizer.json",64*1024**2,None),
    ("model.safetensors",700*1024**2,WEIGHTS_SHA256))
MAX_BYTES = 768*1024**2


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def fetch(url: str, path: Path, maximum: int, expected: str | None) -> dict:
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise ValueError("target_already_exists_no_implicit_retry_or_overwrite")
    partial = path.with_name(path.name+".partial")
    if partial.exists():
        raise ValueError("partial_download_requires_explicit_recovery")
    count, started, checksum = 0, time.monotonic(), hashlib.sha256()
    request = urllib.request.Request(url,headers={"User-Agent":"CiteQuest-bounded-asset-fetch/1"})
    with urllib.request.urlopen(request,timeout=45) as response, partial.open("xb") as output:
        if response.status != 200:
            raise ValueError("asset_http_status_not_200")
        stated = response.headers.get("Content-Length")
        if stated and int(stated)>maximum:
            raise ValueError("asset_advertised_size_exceeds_bound")
        while chunk := response.read(4*1024**2):
            count+=len(chunk)
            if count>maximum:
                raise ValueError("asset_stream_size_exceeds_bound")
            checksum.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    actual = checksum.hexdigest()
    if not count or (expected is not None and actual!=expected):
        raise ValueError("asset_empty_or_sha256_mismatch")
    # Small JSON/tokenizer files are pinned by the immutable Hub commit and
    # recorded hashes; the public LFS weight SHA and PyPI wheel SHA are checked.
    if path.suffix==".json":
        json.loads(partial.read_text(encoding="utf-8"))
    partial.replace(path)
    return {"bytes":count,"sha256":actual,"expected_sha256":expected,"seconds":time.monotonic()-started}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    target=args.destination.resolve()
    if not target.is_relative_to((root/"data/decisionrag").resolve()) or target==root/"data/decisionrag":
        raise ValueError("only_explicit_project_data_decisionrag_subdirectory_allowed")
    if os.name=="nt" and target.drive.upper()!="F:":
        raise ValueError("F_drive_required_no_C_model_cache")
    if target.exists() or args.receipt.exists():
        raise ValueError("new_destination_and_receipt_required")
    target.mkdir(parents=True)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    result={"status":"started","model_id":MODEL,"revision":REVISION,"subfolder":"multilingual",
        "license":"Apache-2.0","laya_version":"0.4.22","installed":False,"model_loaded":False,
        "automatic_retry":False,"files":{},"wheel":None,"maximum_model_bytes":MAX_BYTES}
    def save() -> None:
        args.receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    save()
    try:
        for name,cap,expected in FILES:
            url=f"https://huggingface.co/{MODEL}/resolve/{REVISION}/multilingual/{name}"
            result["files"][name]=fetch(url,target/"model"/name,cap,expected)
            if sum(value["bytes"] for value in result["files"].values())>MAX_BYTES:
                raise ValueError("total_model_asset_budget_exceeded")
            save()
            print(json.dumps({"asset":name,"bytes":result["files"][name]["bytes"],"sha_verified":bool(expected)}),flush=True)  # noqa: T201 - Explicit CLI metadata, no URLs or source content.
        wheel=target/"vendor/laya-0.4.22-py3-none-any.whl"
        result["wheel"]=fetch(WHEEL_URL,wheel,1024**2,WHEEL_SHA256)
        result["status"]="PASS_download_and_integrity_only_NOT_runtime_or_quality"
    except Exception as error:
        result["status"]="failed_stopped_no_retry"
        result["error_type"]=type(error).__name__
        if isinstance(error,urllib.error.HTTPError):
            result["http_status"]=error.code
        # Never persist an exception URL: CDN redirects can contain signed tokens.
        save()
        print(json.dumps({"status":result["status"],"error_type":result["error_type"],"http_status":result.get("http_status")}),flush=True)  # noqa: T201 - Sanitized CLI failure classification.
        return 1
    save()
    return 0


if __name__=="__main__":
    sys.exit(main())
