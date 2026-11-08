"""Container entrypoint, explicit configuration, no runtime DDL or fake login."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from citequest.search_v3.runtime import service_from_config
from citequest.v2.db import engine, sessions
from citequest.v2.origin import OriginPolicy
from citequest.v2.storage_config import app_storage

from .app import create_app
from .auth import AuthConfig


class Config(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: str = Field(pattern="^citequest-web-container/1$")
    mode: Literal["development", "production"]
    root: str
    origin: str
    issuer: str
    client_id: str = Field(default="citequest-web", min_length=1, max_length=200)
    metadata_url: str | None = None
    credentials_file: str
    business_dsn_file: str
    cursor_key_file: str
    search_config_file: str
    agent_config_file: str | None = None
    export_config_file: str | None = None
    port: int = Field(default=8080, ge=1024, le=65535)


def application(config: Config) -> FastAPI:
    if config.export_config_file and not config.agent_config_file:
        raise ValueError("export_requires_product_agent_configuration")
    root = Path(config.root).resolve(strict=True)
    credentials = json.loads(Path(config.credentials_file).read_text(encoding="utf-8"))
    auth_config = AuthConfig(
        OriginPolicy(config.origin, development=config.mode == "development"),
        config.issuer,
        config.client_id,
        credentials["client_secret"],
        credentials["session_secret"],
        config.metadata_url,
    )
    factory = sessions(engine(Path(config.business_dsn_file).read_text(encoding="utf-8").strip()))
    search = service_from_config(Path(config.search_config_file))
    app = create_app(
        factory,
        store=app_storage(root).cas(),
        root=root,
        cursor_key=Path(config.cursor_key_file).read_bytes(),
        auth_config=auth_config,
        search=search,
    )
    if config.agent_config_file:
        from citequest.agents_v3.api_config import AgentAPIConfig
        from citequest.agents_v3.database import TaskService
        from citequest.agents_v3.memory import MemoryService
        from citequest.material_evidence_v3 import service_from_config as materials_from_config
        from citequest.research_v3.evidence import EvidenceAccess

        from .agent_routes import product_agent_router

        agent_config = AgentAPIConfig.model_validate_json(Path(config.agent_config_file).read_bytes(), strict=True)
        database = agent_config.database()
        materials = materials_from_config(agent_config.material_config_file) if agent_config.material_config_file else None
        if materials is not None:
            app.state.material_evidence_service = materials
        tasks = TaskService(database, limits=agent_config.limits,
                            new_task_profile_policy=agent_config.task_profile_policy)
        memory = MemoryService(database)
        evidence = EvidenceAccess(database, search, materials)
        router = product_agent_router(
            tasks,
            memory,
            app.state.auth_service,
            evidence,
        )
        pages = [route for route in app.router.routes if getattr(route, "name", "") == "page"]
        app.router.routes[:] = [
            route for route in app.router.routes if getattr(route, "name", "") not in {"agent_not_configured", "page"}
        ]
        app.include_router(router)
        app.state.required_business_tables.update({"agent_tasks", "agent_attempts", "agent_evidence_refs",
            "agent_memories", "user_skills", "user_skill_versions", "user_skill_operations"})
        if agent_config.catalog_scope_policy is not None:
            app.state.required_business_tables.add("agent_catalog_pins")
        if agent_config.dispatch_policy_file is not None:
            app.state.required_business_tables.update({"agent_dispatch_control", "agent_command_queues", "agent_model_waiters"})
        if config.export_config_file:
            from citequest.exports_v3.api_config import configured_api, domain_from_config
            from .export_routes import product_export_router
            export_config = configured_api(Path(config.export_config_file))
            if export_config.workspace.resolve(strict=True) != root:
                raise ValueError("export_and_application_workspace_mismatch")
            domain = domain_from_config(export_config, factory=factory, tasks=tasks, memory=memory,
                evidence=evidence, search=search, cursor_key=Path(config.cursor_key_file).read_bytes())
            app.include_router(product_export_router(domain, app.state.auth_service))
            app.state.export_domain = domain
            app.state.required_business_tables.update({"agent_document_exports", "agent_export_operations", "agent_export_events"})
        app.router.routes.extend(pages)
        app.openapi_schema = None
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = Config.model_validate_json(parser.parse_args().config.read_bytes())
    uvicorn.run(
        application(config),
        host="0.0.0.0",
        port=config.port,
        access_log=False,
        timeout_keep_alive=5,
        timeout_graceful_shutdown=20,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
