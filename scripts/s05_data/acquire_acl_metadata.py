"""Bounded anonymous official ACL metadata acquisition and SDK-only read audit."""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import importlib.metadata
import json
import shutil
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import httpx
from acl_anthology import Anthology
from acl_anthology.collections import Paper, Volume
from acl_anthology.utils.ids import parse_id

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / "reports/stages/v2/S05/rights-preparation"
ORDER = ROOT / "reports/stages/v2/S05/preparation/reconciliation-20261105-r1/candidate-order-provenance.jsonl"
REPO = "https://api.github.com/repos/acl-org/acl-anthology"
MAX_SINGLE = 8 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024
MAX_FILES = 650
PINNED_ACL_COMMIT = "98d89b457f355afa93ba6659b8132ea5ac46ed5d"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


class Fetch:
    def __init__(self, receipts: Path) -> None:
        self.client = httpx.Client(
            timeout=30,
            follow_redirects=False,
            headers={"User-Agent": "CiteQuest-source-audit/0.5 (bounded metadata verification)"},
        )
        self.receipts, self.count, self.total, self.last = receipts, 0, 0, 0.0

    def get(self, url: str, limit: int) -> bytes:
        host = httpx.URL(url).host
        if host not in {"api.github.com", "raw.githubusercontent.com", "aclanthology.org"}:
            raise ValueError("Source host outside explicit official allowlist")
        time.sleep(max(0, 1 - (time.monotonic() - self.last)))
        self.last = time.monotonic()
        self.count += 1
        data = bytearray()
        with self.client.stream("GET", url) as response:
            response.raise_for_status()
            for block in response.iter_bytes():
                if len(data) + len(block) > limit or self.total + len(block) > MAX_TOTAL:
                    raise ValueError("Bounded source download limit exceeded")
                data.extend(block)
                self.total += len(block)
            value = bytes(data)
            write(
                self.receipts / f"http-{self.count:04d}.json",
                {
                    "url": url,
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "status": response.status_code,
                    "bytes": len(value),
                    "sha256": sha(value),
                    "etag": response.headers.get("etag"),
                    "last_modified": response.headers.get("last-modified"),
                },
            )
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit bounded official source acquisition required")
    if importlib.metadata.version("acl-anthology") != "1.3.3":
        raise RuntimeError("Official SDK1.3.3 required")
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise RuntimeError("Require at least2GiB free before bounded metadata fetch")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run = PREP / ("acl-sdk-" + stamp)
    fetch = Fetch(run / "http")
    inventory = []
    with ORDER.open(encoding="utf-8") as stream:
        for line in stream:
            if len(inventory) == 3000:
                break
            inventory.append(json.loads(line))
    if len(inventory) != 3000:
        raise ValueError("Exact frozen primary prefix required")
    needed, malformed = {}, {}
    for rank, row in enumerate(inventory):
        acl = (row.get("externalids") or {}).get("acl")
        if not acl:
            continue
        try:
            collection, volume, paper = parse_id(acl)
            if not volume or not paper or "/" in collection or "\\" in collection:
                raise ValueError("Paper ID required")
        except (ValueError, TypeError) as exc:
            malformed[rank] = type(exc).__name__
            continue
        needed.setdefault(collection, []).append((rank, acl))
    try:
        commit_raw = fetch.get(REPO + "/commits/" + PINNED_ACL_COMMIT, 2 * 1024 * 1024)
        commit = json.loads(commit_raw)["sha"]
        if commit != PINNED_ACL_COMMIT:
            raise ValueError("Pinned official ACL commit mismatch")
        root_tree_raw = fetch.get(REPO + "/git/trees/" + commit, 2 * 1024 * 1024)
        root_tree = json.loads(root_tree_raw)
        data_sha = next(x["sha"] for x in root_tree["tree"] if x["path"] == "data" and x["type"] == "tree")
        data_tree_raw = fetch.get(REPO + "/git/trees/" + data_sha, 2 * 1024 * 1024)
        data_tree = json.loads(data_tree_raw)
        xml_sha = next(x["sha"] for x in data_tree["tree"] if x["path"] == "xml" and x["type"] == "tree")
        xml_tree_raw = fetch.get(REPO + "/git/trees/" + xml_sha, 4 * 1024 * 1024)
        xml_tree = json.loads(xml_tree_raw)
        if any(t.get("truncated") for t in (root_tree, data_tree, xml_tree)):
            raise ValueError("Complete official tree preflight required")
        entries = {x["path"]: x for x in xml_tree["tree"] if x["type"] == "blob"}
        names = ["schema.rnc", *[key + ".xml" for key in needed]]
        planned = [entries[name] for name in names if name in entries and 0 < entries[name]["size"] <= MAX_SINGLE]
        projected = sum(x["size"] for x in planned)
        run.mkdir(parents=True, exist_ok=True)
        (run / "source-commit-response-raw.json").write_bytes(commit_raw)
        (run / "source-xml-tree-raw.json").write_bytes(xml_tree_raw)
        write(
            run / "preflight.json",
            {
                "state": "preparation_only",
                "source_commit": commit,
                "SDK": "acl-anthology1.3.3",
                "candidate_count": 3000,
                "collections_needed": len(needed),
                "planned_files": len(planned),
                "planned_source_bytes": projected,
                "limits": {
                    "single_bytes": MAX_SINGLE,
                    "total_transfer_bytes": MAX_TOTAL,
                    "files": MAX_FILES,
                    "requests_per_second": 1,
                },
                "identity_only_not_copyright_verification": True,
                "no_qrels_or_independent_labels_read": True,
                "candidate_order_sha256": sha(ORDER.read_bytes()),
            },
        )
        if (
            projected + fetch.total > MAX_TOTAL
            or len(planned) > MAX_FILES
            or "schema.rnc" not in {x["path"] for x in planned}
        ):
            raise ValueError("Whole required bounded XML preflight failed; no unannounced subset selection")
        data = ROOT / "data/s05/acl-source" / commit
        xml = data / "xml"
        xml.mkdir(parents=True, exist_ok=True)
        write(run / "source-commit-response.json", json.loads(commit_raw))
        write(run / "source-xml-tree.json", xml_tree)
        source_files = []
        for item in planned:
            target = xml / item["path"]
            if target.exists():
                content = target.read_bytes()
            else:
                content = fetch.get(
                    f"https://raw.githubusercontent.com/acl-org/acl-anthology/{commit}/data/xml/{item['path']}",
                    MAX_SINGLE,
                )
            git_blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if len(content) != item["size"] or git_blob != item["sha"]:
                raise ValueError("Pinned Git blob/source bytes mismatch")
            if not target.exists():
                target.write_bytes(content)
            if len(source_files) % 50 == 0:
                print(  # noqa: T201
                    json.dumps(
                        {
                            "event": "official_XML_progress",
                            "verified_files": len(source_files),
                            "planned_files": len(planned),
                        }
                    ),
                    flush=True,
                )
            source_files.append(
                {
                    "path": target.relative_to(ROOT).as_posix(),
                    "bytes": len(content),
                    "sha256": sha(content),
                    "git_blob_sha1": git_blob,
                }
            )
            if item["path"] == "schema.rnc":
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always")
                    Anthology(data, verbose=False)
                if observed:
                    write(
                        run / "schema-warnings.json",
                        [{"type": type(w.message).__name__, "message": str(w.message)} for w in observed],
                    )
                    raise RuntimeError("Source schema differs from official SDK; no parser substitution")
        policy = fetch.get("https://aclanthology.org/faq/copyright/", 512 * 1024)
        policy_path = data / ("copyright-policy-" + sha(policy) + ".html")
        if not policy_path.exists():
            policy_path.write_bytes(policy)
        outcomes = [
            {
                "rank": rank,
                "native_id": row["native_id"],
                "raw_license_claim": row["license_raw"],
                "acl_id": (row.get("externalids") or {}).get("acl"),
                "status": "not_an_acl_source" if not (row.get("externalids") or {}).get("acl") else "pending",
                "rights_status": "unknown",
                "explicit_paper_license": None,
            }
            for rank, row in enumerate(inventory)
        ]
        present = {x["path"][:-4] for x in planned if x["path"].endswith(".xml")}
        for collection, targets in needed.items():
            if collection not in present:
                for rank, _acl in targets:
                    outcomes[rank]["status"] = "source_XML_missing_or_over_single_file_cap"
                continue
            anthology = Anthology(data, verbose=False)
            for rank, acl in targets:
                try:
                    paper = anthology.get(acl)
                    if not isinstance(paper, Paper) or not isinstance(paper.parent, Volume):
                        raise ValueError("Official paper/volume lookup failed")
                    volume = paper.parent
                    pdf = paper.pdf
                    outcomes[rank].update(
                        {
                            "status": "official_metadata_matched",
                            "official_id": paper.full_id,
                            "title": str(paper.title),
                            "doi": paper.doi,
                            "volume_id": volume.full_id,
                            "publisher": volume.publisher,
                            "year": volume.year,
                            "official_pdf_crc32": pdf.checksum if pdf else None,
                            "source_pdf_sha1_claim": inventory[rank].get("pdfsha_raw"),
                            "checksum_algorithms_directly_comparable": False,
                            "PDF_identity_verification": "requires_actual_PDF_bytes_CRC32_legacySHA1_and_new_SHA256",
                            "publisher_acl_exact_signal": volume.publisher
                            == "Association for Computational Linguistics",
                            "rights_status": "unknown_explicit_article_or_volume_license_not_exposed_by_SDK",
                            "copyright_policy_sha256": sha(policy),
                            "metadata_match_does_not_verify_copyright": True,
                        }
                    )
                except Exception as exc:
                    outcomes[rank].update(
                        {
                            "status": "official_SDK_metadata_error",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        }
                    )
            del anthology
            gc.collect()
        for rank, error in malformed.items():
            outcomes[rank].update({"status": "malformed_source_acl_ID", "error_type": error})
        counts = collections.Counter(row["status"] for row in outcomes)
        with (run / "metadata-outcomes.jsonl").open("x", encoding="utf-8") as stream:
            for row in outcomes:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        write(
            run / "verification.json",
            {
                "status": "official_metadata_preparation_complete_not_rights_pass",
                "SDK": "acl-anthology1.3.3",
                "source_commit": commit,
                "positions": len(outcomes),
                "status_counts": dict(counts),
                "publisher_acl_exact_signals": sum(bool(row.get("publisher_acl_exact_signal")) for row in outcomes),
                "identity_verified_from_metadata_alone": 0,
                "copyright_verified_articles": 0,
                "article_license_field_exposed_by_SDK": False,
                "remaining": "original article or official volume copyright/license notice verification; metadata/publisher alone not sufficient",
                "source_files": source_files,
                "copyright_policy": {"path": policy_path.relative_to(ROOT).as_posix(), "sha256": sha(policy)},
                "runtime_lock_sha256": sha((ROOT / "runtime/s05-data/uv.lock").read_bytes()),
                "http_requests": fetch.count,
                "download_bytes": fetch.total,
                "database_writes": 0,
                "model_calls": 0,
                "index_published": False,
            },
        )
        print(  # noqa: T201
            json.dumps(
                {
                    "status": "official_metadata_preparation_complete_not_rights_pass",
                    "positions": len(outcomes),
                    "status_counts": dict(counts),
                    "evidence": run.relative_to(ROOT).as_posix(),
                }
            )
        )
    except BaseException as exc:
        write(
            run / "stop.json",
            {
                "state": "failed_preparation",
                "type": type(exc).__name__,
                "reason": str(exc),
                "http_requests": fetch.count,
                "download_bytes": fetch.total,
            },
        )
        raise
    finally:
        fetch.client.close()


if __name__ == "__main__":
    main()
