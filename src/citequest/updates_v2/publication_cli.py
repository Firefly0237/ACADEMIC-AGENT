"""Explicit observed/prepared/build/publish/recover operations, never implicit sync."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field
from sqlalchemy import select

from citequest.v2.domain import Actor, DomainError, Input
from . import models as update


class PublicationCommand(Input):
    schema_version: Literal["citequest-publication-command/1"]
    search_config_file: Path
    owner_id: UUID
    phase: Literal["observed", "prepare", "build", "publish", "recover"]
    window_id: UUID | None = None
    base_generation_id: UUID
    observation_seal_ids: list[UUID] = Field(min_length=1, max_length=100)
    publication_snapshot_id: UUID
    index_generation_id: UUID
    execution_seconds: int = Field(default=900, ge=30, le=3600)
    expected_epoch: int = Field(ge=0)
    intent_id: UUID
    backend_profile_file: Path | None = None


def execute(config: PublicationCommand):
    from psycopg import connect
    from psycopg.rows import dict_row
    from citequest.index_v2.models import IndexCatalog, IndexGeneration
    from citequest.index_v2.publication import AliasTransport, PublicationCoordinator
    from citequest.index_v3.publication import FencedAliasCalls, PublicationSessionFence
    from citequest.index_v3.publication_controller import PublicationIndexController
    from citequest.index_v3.publication_native import PublicationNative
    from citequest.index_v3.repository_bounded import BoundedFullBuildRepository
    from citequest.materials_v2.chunking import SourceChunker
    from citequest.search_v3.runtime import service_from_config
    from citequest.v2.services import Catalog
    from .publication import SnapshotService

    search = service_from_config(config.search_config_file)
    actor = Actor(config.owner_id, publisher=True)
    def transaction(operation):
        with search.factory.begin() as db:
            return operation(BoundedFullBuildRepository(Catalog(db, actor), store=search.store,
                policy_sha256=search.policy_sha256, evaluation_sha256=search.evaluation_sha256))
    try:
        if config.phase == "observed":
            with search.factory() as db:
                window = db.get(update.UpdateWindow, config.window_id) if config.window_id else None
                if window is None or window.owner_id != actor.owner_id:
                    raise DomainError("source_window_not_found", 404)
                seal = db.scalar(select(update.UpdateSeal).where(update.UpdateSeal.window_id == window.id))
                if window.state != "observed" or seal is None:
                    return {"state": window.state, "search_published": False}
                return {"state": "observed", "seal_id": str(seal.id), "observations_publication_eligible": False,
                    "search_published": False, "current_status_confirmed": False}
        if config.phase == "prepare":
            snapshot = transaction(lambda repository: SnapshotService(repository).create(config.base_generation_id,
                config.observation_seal_ids, config.publication_snapshot_id))
            generation = transaction(lambda repository: repository.begin_source("publication_snapshot", config.publication_snapshot_id,
                config.index_generation_id, tokenizer_profile_sha256=search.query_tokenizer_profile_sha256, unit_limit=None))
            return {"state": "prepared", "snapshot": snapshot, "generation": generation, "search_published": False}
        native = PublicationNative(search.native, backend_profile_file=config.backend_profile_file)
        native.absolute_deadline = time.monotonic() + config.execution_seconds
        if config.phase == "build":
            # The controller uses the pinned tokenizer already required by the
            # search deployment; it never downloads a replacement asset.
            from citequest.search_v3.runtime import SearchRuntimeConfig
            runtime_path = config.search_config_file.resolve(strict=True)
            runtime = SearchRuntimeConfig.model_validate_json(runtime_path.read_bytes())
            root = Path(runtime.workspace)
            root = root if root.is_absolute() else runtime_path.parent / root
            controller = PublicationIndexController(search.factory, actor, store=search.store,
                chunker=SourceChunker(root), native=native, policy_sha256=search.policy_sha256,
                evaluation_sha256=search.evaluation_sha256)
            try:
                result = controller.run("publication_snapshot", config.publication_snapshot_id, config.index_generation_id,
                    unit_limit=None, seconds=config.execution_seconds, checkpoint=lambda event: print(json.dumps({
                        "event": event.get("event"), "generation_id": str(config.index_generation_id), "search_published": False}), flush=True))
            except DomainError as error:
                return {"state": "prepared", "status": "blocked", "error": error.code, "search_published": False,
                    "maintenance": "python -m citequest.updates_v2.backend_maintenance --config /run/citequest/config/backend-binding.json",
                    "requires": "migrator identity, exact prepared generation manifest and pinned backend profile; no automatic adoption"}
            return {"state": "prepared", "indexing": result, "search_published": False}
        bind = search.factory.kw["bind"]
        dsn = bind.url.render_as_string(hide_password=False).replace("postgresql+psycopg://", "postgresql://", 1)
        with connect(dsn, autocommit=True, row_factory=dict_row) as connection:
            fence = PublicationSessionFence(connection, "standard_metadata", "public")
            def authority():
                with search.factory() as db:
                    catalog = db.get(IndexCatalog, ("standard_metadata", "public"))
                    if catalog is None:
                        return None
                    if catalog.pending_intent_id is not None and config.phase != "recover":
                        from citequest.index_v2.models import IndexPublishIntent
                        intent = db.get(IndexPublishIntent, catalog.pending_intent_id)
                        identifier = intent.target_generation_id
                    else:
                        identifier = catalog.active_generation_id
                    generation = db.get(IndexGeneration, identifier) if identifier else None
                    return generation.collection_name if generation else None
            with fence.held():
                aliases = AliasTransport(FencedAliasCalls(native, fence, authority))
                coordinator = PublicationCoordinator(transaction, aliases)
                result = (coordinator.recover(config.intent_id) if config.phase == "recover" else coordinator.publish(
                    config.index_generation_id, config.intent_id, expected_epoch=config.expected_epoch))
            with search.factory() as db:
                catalog = db.get(IndexCatalog, ("standard_metadata", "public"))
                published = catalog is not None and catalog.active_generation_id == config.index_generation_id
            return {"state": "published" if published else "prepared", "search_published": published,
                "publication": result, "cross_database_atomic_transaction_claimed": False}
    except DomainError as error:
        if config.phase in {"build", "publish", "recover"}:
            return {"state": "prepared", "status": "blocked", "error": error.code, "search_published": False,
                "maintenance": "python -m citequest.updates_v2.backend_maintenance --config /run/citequest/config/backend-adoption.json"
                    if "backend" in error.code else None,
                "automatic_adoption_performed": False}
        raise
    finally:
        search.native.close()
        search.factory.kw["bind"].dispose()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        path = args.config.resolve(strict=True)
        value = json.loads(path.read_text("utf8"))
        for key in ("search_config_file", "backend_profile_file"):
            if value.get(key) is not None:
                selected = Path(value[key])
                value[key] = str(selected if selected.is_absolute() else path.parent / selected)
        result = execute(PublicationCommand.model_validate_json(json.dumps(value), strict=True))
        print(json.dumps(result, ensure_ascii=True, default=str))
        return 0
    except DomainError as error:
        print(json.dumps({"state": "not_published", "error": error.code}))
    except Exception as error:
        print(json.dumps({"state": "not_published", "error": type(error).__name__}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
