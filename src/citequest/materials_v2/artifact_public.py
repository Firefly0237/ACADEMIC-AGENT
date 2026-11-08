"""Public source assets are distinct from complete server execution audits."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from citequest.v2.domain import DomainError

PUBLIC_ROLES = frozenset(
    {
        "native_document",
        "source_representation",
        "source_pages",
        "unmapped_blocks",
        "normalized_text",
        "source_paragraphs",
    }
)
INTERNAL_ROLES = frozenset(
    {"parser_result", "parser_options", "parser_resource_receipt", "chunk_resource_receipt", "preflight_pages"}
)
SHA = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


def _unique_pairs(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_asset_key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite_asset_value")


class PublicProvenance(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    schema_version: Literal["cq.material.asset/1"]
    source_sha256: SHA
    source_view_id: SHA | None
    acquisition_mode: Literal["captured_before_publish", "verified_backfill"]


class PublicAsset(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    role: Annotated[str, Field(pattern=r"^[a-z_]{1,60}$")]
    sha256: SHA
    size_bytes: Annotated[int, Field(ge=1, le=64 * 1024 * 1024)]
    availability: Literal["registered", "internal_audit_only"]
    content_url: str | None
    provenance: PublicProvenance
    audit_fields_withheld: bool


class PublicAssets(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    schema_version: Literal["cq.material.public-assets/1"]
    file_id: UUID
    representation_id: SHA
    items: list[PublicAsset]


def public_assets(value: dict) -> dict:
    if value.get("schema_version") != "cq.material.artifacts/1":
        raise DomainError("artifact_response_contract_failed", 500)
    items = []
    keys = {"schema_version", "source_sha256", "source_view_id", "acquisition_mode"}
    for item in value["items"]:
        role = item["role"]
        if role not in PUBLIC_ROLES | INTERNAL_ROLES:
            raise DomainError("artifact_role_not_profiled", 500)
        provenance = {key: item["provenance"].get(key) for key in keys}
        public = role in PUBLIC_ROLES
        items.append(
            {
                "role": role,
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
                "availability": "registered" if public else "internal_audit_only",
                "content_url": f"/api/v2/files/{value['file_id']}/representations/{value['representation_id']}/artifacts/{role}/content"
                if public
                else None,
                "provenance": provenance,
                "audit_fields_withheld": bool(set(item["provenance"]) - keys),
            }
        )
    result = {
        "schema_version": "cq.material.public-assets/1",
        "file_id": value["file_id"],
        "representation_id": value["representation_id"],
        "items": items,
    }
    try:
        PublicAssets.model_validate_json(json.dumps(result, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise DomainError("artifact_response_contract_failed", 500) from exc
    return result


def ensure_public_content(role: str, data: bytes, *, root: Path) -> None:
    """Inspect without rewriting: the returned body must retain the original SHA.

    Source text may legitimately describe filesystem paths. Detect actual local
    installation prefixes everywhere; separately reject absolute path values in
    metadata fields, preserving ordinary scientific text as source data.
    """
    if role not in PUBLIC_ROLES:
        raise DomainError("artifact_internal_audit_only", 403)
    if len(data) > 64 * 1024 * 1024:
        raise DomainError("artifact_bytes_exceeded", 413)
    try:
        value = json.loads(data, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise DomainError("artifact_invalid_json", 422) from exc
    prefixes = {
        str(path.resolve()).replace("\\", "/").casefold()
        for path in [root, Path.home(), Path(sys.executable).parent, Path(sys.base_prefix)]
    }
    text_keys = {"text", "orig", "source_quote", "source_text", "normalized_text"}
    stack = [("", value)]
    while stack:
        key, item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.items())
        elif isinstance(item, list):
            stack.extend((key, part) for part in item)
        elif isinstance(item, str):
            normalized = item.replace("\\", "/").casefold()
            if any(prefix in normalized for prefix in prefixes):
                raise DomainError("artifact_contains_internal_path", 403)
            if key not in text_keys and (re.match(r"^[a-zA-Z]:[/\\]", item) or item.startswith(("\\\\", "/"))):
                raise DomainError("artifact_contains_absolute_metadata_path", 403)
            if key == "filename" and ("/" in item or "\\" in item):
                raise DomainError("artifact_filename_is_not_basename", 403)
