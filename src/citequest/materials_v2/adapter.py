"""Typed, DB-free adapter for the resource-limited parser implementation."""
from __future__ import annotations

import hashlib
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from .interfaces import ParseBlock, ParsedMaterial, ParserInput
from .isolation import run_isolated
from .parser_profile import features
from .text import MaterialError, load_limits


@dataclass(frozen=True)
class JobParser:
    root: Path
    profile_path: Path
    artifact_root: Path

    def parse(self, request: ParserInput) -> ParsedMaterial:
        profile_bytes = self.profile_path.read_bytes()
        profile_sha = hashlib.sha256(profile_bytes).hexdigest()
        if profile_sha != request.parser_profile_sha256:
            raise MaterialError("PARSER_PROFILE_CHANGED")
        selected = features(json.loads(profile_bytes))
        if request.limits != load_limits(self.root / "contracts/v2/policy.json"):
            raise MaterialError("PARSER_LIMIT_CONTRACT_CHANGED")
        receipt = run_isolated({"source_path": str(request.file_path), "source_sha256": request.source_sha256,
                                "media_type": request.detected_media_type, "expected_profile_sha256": profile_sha}, root=self.root, profile_path=self.profile_path,
                               output=self.artifact_root / uuid.uuid4().hex, python=Path(sys.executable))
        result = receipt["result"]
        if result["state"] not in {"ready", "partial"}:
            raise MaterialError(result["code"])
        if receipt.get("profile_sha256") != profile_sha or result.get("profile_sha256") != profile_sha:
            raise MaterialError("PARSER_RESULT_PROFILE_MISMATCH")
        if result["source_sha256"] != request.source_sha256:
            raise MaterialError("PARSER_RESULT_SOURCE_MISMATCH")
        if request.detected_media_type == "application/pdf" and result["code"] in {"PDF_PARSED", "PDF_PARTIAL_ANCHORS"}:
            if (result.get("ocr_enabled") is not selected.ocr_enabled
                    or result.get("formula_enrichment_enabled", False) is not selected.formula_enabled
                    or result.get("formula_semantics_verified") is not False):
                raise MaterialError("PARSER_RESULT_CAPABILITY_MISMATCH")
        blocks = tuple(ParseBlock(row["block_id"], row["text"], row["locator"], row["source_quote"], hashlib.sha256(row["source_quote"].encode()).hexdigest()) for row in result["blocks"])
        return ParsedMaterial(source_sha256=result["source_sha256"], parser_id=json.loads(profile_bytes)["profile_id"],
                              parser_profile_sha256=profile_sha, state=result["state"], blocks=blocks,
                              pages_requiring_ocr=tuple(result.get("pages_requiring_ocr_or_review", [])),
                              representation_id=result.get("representation_id"), can_qa=result.get("can_qa", False),
                              missing_ranges=tuple(row["item_ref"] for row in result.get("missing_ranges", [])),
                              usable_block_ids=tuple(row["block_id"] for row in result["blocks"]),
                              warnings=tuple(dict.fromkeys([*result.get("warnings", []), *result.get("formula_warnings", []),
                                  *result.get("table_warnings", []), *([result["code"]] if result["state"] == "partial" else [])])))
