"""API-only trusted configuration; no Docker, worker, provider or model key."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from sqlalchemy.orm import Session, sessionmaker

from citequest.agents_v3.contracts import Strict
from citequest.agents_v3.database import TaskService
from citequest.agents_v3.memory import MemoryService
from citequest.platform_v3.blob_config import BlobStoreConfig, create_blob_store
from citequest.research_v3.evidence import EvidenceAccess
from citequest.research_v3.school_contracts import SHA
from citequest.search_v3.service import SearchService
from citequest.v2.pagination import Pager

from .product import ExportDomain, build_export_domain
from .release import load_release, verify_release_assets


class ExportAPIConfig(Strict):
    schema_version: Literal["citequest-export-api/1"]
    workspace: Path
    release_profile_file: Path
    release_profile_sha256: SHA
    artifact_store: BlobStoreConfig


def configured_api(path: Path) -> ExportAPIConfig:
    path = path.resolve(strict=True)
    value = json.loads(path.read_bytes())
    for key in ("workspace", "release_profile_file"):
        reference = Path(value[key])
        value[key] = str(reference if reference.is_absolute() else path.parent / reference)
    for key in ("filesystem_root", "spool_dir"):
        if value.get("artifact_store", {}).get(key) is not None:
            reference = Path(value["artifact_store"][key])
            value["artifact_store"][key] = str(reference if reference.is_absolute() else path.parent / reference)
    return ExportAPIConfig.model_validate_json(json.dumps(value), strict=True)


def domain_from_config(
    config: ExportAPIConfig,
    *,
    factory: sessionmaker[Session],
    tasks: TaskService,
    memory: MemoryService,
    evidence: EvidenceAccess,
    search: SearchService,
    cursor_key: bytes,
) -> ExportDomain:
    config = ExportAPIConfig.model_validate(config.model_dump(mode="python"), strict=True)
    release = load_release(config.release_profile_file, config.release_profile_sha256)
    verify_release_assets(config.workspace, release)
    return build_export_domain(
        factory,
        tasks,
        memory,
        evidence,
        search,
        create_blob_store(config.artifact_store),
        release.profile(),
        Pager(cursor_key),
    )
