"""Trusted export-only process configuration; never loads an LLM provider key."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy.engine import Engine

from citequest.agents_v3.contracts import RuntimeLimits, Strict
from citequest.agents_v3.database import Database, TaskService
from citequest.agents_v3.memory import MemoryService
from citequest.material_evidence_v3 import service_from_config as materials_from_config
from citequest.platform_v3.blob_config import BlobStoreConfig, create_blob_store
from citequest.research_v3.evidence import EvidenceAccess
from citequest.research_v3.school_contracts import SHA
from citequest.search_v3.runtime import service_from_config as search_from_config
from citequest.v2.db import engine, sessions
from citequest.v2.pagination import Pager

from .docker_renderer import DockerRenderer
from .product import ExportDomain, build_export_domain
from .release import CertifiedRelease as CertifiedRelease
from .release import asset, load_release, verify_release_assets
from .worker import ExportWorker


class ExportProcessConfig(Strict):
    schema_version: Literal["citequest-export-process/1"]
    workspace: Path
    release_profile_file: Path
    release_profile_sha256: SHA
    business_dsn_file: Path
    checkpoint_dsn_file: Path
    memory_dsn_file: Path
    cursor_key_file: Path
    search_config_file: Path
    material_config_file: Path | None = None
    limits: RuntimeLimits
    artifact_store: BlobStoreConfig
    worker_spool: Path
    docker_cli: Path
    docker_config: Path


def configured(config_file: Path) -> ExportProcessConfig:
    """Resolve trusted file references without reading any model configuration."""
    config_file = config_file.resolve(strict=True)
    value = json.loads(config_file.read_text(encoding="utf8"))
    for key in (
        "workspace",
        "release_profile_file",
        "business_dsn_file",
        "checkpoint_dsn_file",
        "memory_dsn_file",
        "cursor_key_file",
        "search_config_file",
        "material_config_file",
        "worker_spool",
        "docker_cli",
        "docker_config",
    ):
        if value.get(key) is not None:
            path = Path(value[key])
            value[key] = str(path if path.is_absolute() else config_file.parent / path)
    store = value.get("artifact_store", {})
    for key in ("spool_dir", "filesystem_root"):
        if store.get(key) is not None:
            path = Path(store[key])
            store[key] = str(path if path.is_absolute() else config_file.parent / path)
    return ExportProcessConfig.model_validate_json(json.dumps(value), strict=True)


def released(config: ExportProcessConfig) -> CertifiedRelease:
    return load_release(config.release_profile_file, config.release_profile_sha256)


@dataclass
class ExportProcess:
    domain: ExportDomain
    worker: ExportWorker
    engines: list[Engine]

    def close(self) -> None:
        for resource in self.engines:
            resource.dispose()


def create_process(config: ExportProcessConfig, *, worker_id: str) -> ExportProcess:
    """No migration, Saver setup, Task creation, LLM SDK or provider secret read."""
    config = ExportProcessConfig.model_validate(config.model_dump(mode="python"), strict=True)
    release = released(config)
    identity = verify_release_assets(config.workspace, release)
    database = Database.from_files(config.business_dsn_file, config.checkpoint_dsn_file, config.memory_dsn_file)
    dsn = (
        config.business_dsn_file.read_text(encoding="utf8").strip().replace("postgresql://", "postgresql+psycopg://", 1)
    )
    business = engine(dsn)
    search = search_from_config(config.search_config_file)
    materials = materials_from_config(config.material_config_file) if config.material_config_file is not None else None
    evidence = EvidenceAccess(database, search, materials)
    store = create_blob_store(config.artifact_store)
    domain = build_export_domain(
        sessions(business),
        TaskService(database, config.limits),
        MemoryService(database),
        evidence,
        search,
        store,
        release.profile(),
        Pager(config.cursor_key_file.read_bytes()),
    )
    renderer = DockerRenderer(
        docker=config.docker_cli,
        docker_config=config.docker_config,
        image_id=release.image_id,
        template=asset(config.workspace, "assets/export/sjtu-proposal.docx"),
        csl=asset(config.workspace, "assets/export/harvard.csl"),
        profile=release.profile(),
        toolchain_identity=identity,
        certified_capacity_manifest=asset(config.workspace, release.capacity_certification_manifest),
        certified_capacity_manifest_sha256=release.capacity_certification_manifest_sha256,
    )
    worker = ExportWorker(
        domain.service, renderer, worker_id=worker_id, spool_dir=config.worker_spool, admit_references=domain.references
    )
    engines = [business, search.factory.kw["bind"]]
    if materials is not None:
        engines.append(materials.factory.kw["bind"])
    return ExportProcess(domain, worker, engines)
