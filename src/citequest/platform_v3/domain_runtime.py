"""Linux CF-01 entry point over the existing domain factory; never runtime DDL."""
from __future__ import annotations

import argparse
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from citequest.v2.api import create_app
from citequest.v2.db import engine, sessions

from .runtime_config import RuntimeConfig, load_runtime, owned_cf01_dsn

REQUIRED_COLUMNS = {
    "owners": {"id", "can_publish"}, "works": {"id", "security_domain", "metadata_revision"},
    "local_sessions": {"token_sha256", "csrf_sha256", "expires_at", "revoked"},
    "idempotency_keys": {"owner_id", "key", "payload_sha256", "result"},
    "outbox": {"payload", "schema_version", "entity_type", "security_domain"},
    "library_items": {"owner_id", "work_id", "revision", "deleted_at"},
    "collections": {"owner_id", "revision", "deleted_at"},
}


def schema_ready(db: Engine, expected: str) -> bool:
    with db.begin() as connection:
        connection.exec_driver_sql("SET LOCAL statement_timeout='1500ms'")
        if connection.exec_driver_sql("SELECT current_database(),current_user").one() != ("citequest_cf01", "cq_cf01_app"):
            return False
        versions = set(connection.exec_driver_sql("SELECT version_num FROM cq_app.alembic_version").scalars())
        if versions != {expected}:
            return False
        roles = connection.exec_driver_sql("SELECT rolsuper,rolcreatedb,rolcreaterole FROM pg_roles WHERE rolname=current_user").one()
        if any(roles) or connection.exec_driver_sql("SELECT has_schema_privilege(current_user,'cq_app','CREATE')").scalar():
            return False
        if connection.exec_driver_sql("SELECT has_table_privilege(current_user,'cq_app.alembic_version','INSERT,UPDATE,DELETE,TRUNCATE')").scalar():
            return False
        observed: dict[str, set[str]] = {}
        for table, column in connection.execute(text("SELECT table_name,column_name FROM information_schema.columns WHERE table_schema='cq_app'")):
            observed.setdefault(table, set()).add(column)
        return all(columns <= observed.get(table, set()) for table, columns in REQUIRED_COLUMNS.items())


def create_runtime_app(config: RuntimeConfig) -> FastAPI:
    # Reject unvalidated Pydantic model_copy updates at this IO boundary too.
    config = RuntimeConfig.model_validate(config.model_dump(mode="python"), strict=True)
    dsn = owned_cf01_dsn(config.database_dsn_file.read_text(encoding="utf-8"), "app")
    key = config.cursor_key_file.read_bytes().strip()
    if not 32 <= len(key) <= 256:
        raise ValueError("explicit_shared_cursor_key_required")
    config.temporary_directory.mkdir(parents=True, exist_ok=True)
    db = engine(dsn, pool_size=config.pool.pool_size, max_overflow=config.pool.max_overflow,
        pool_timeout=config.pool.pool_timeout, connect_timeout=config.pool.connect_timeout)
    try:
        app = create_app(sessions(db), cursor_key=key, contract_path=config.contract_path,
            origin=config.origin, include_file_routes=False)
    except Exception:
        db.dispose()
        raise
    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            async with previous_lifespan(application):
                yield
        finally:
            db.dispose()

    app.router.lifespan_context = lifespan
    app.state.runtime_config = config
    app.state.runtime_engine = db

    @app.get("/readyz", include_in_schema=False)
    def readiness() -> JSONResponse:
        try:
            ready = schema_ready(db, config.expected_schema_revision)
        except SQLAlchemyError:
            ready = False
        return JSONResponse({"status": "ready" if ready else "not_ready", "instance_id": config.instance_id,
            "scope": "local_domain_crud_only", "schema_revision": config.expected_schema_revision if ready else None,
            "capabilities": config.effective()["capabilities"]}, status_code=200 if ready else 503)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="Validate configuration without DB/secret IO")
    args = parser.parse_args()
    config = load_runtime(args.config, port=os.environ.get("PORT"))
    if args.check:
        print(json.dumps({"fingerprint": config.fingerprint, "effective": config.effective()}, sort_keys=True))
        return
    import uvicorn

    app = create_runtime_app(config)
    uvicorn.run(app, host="0.0.0.0", port=config.port, workers=1, proxy_headers=False,
        access_log=False, log_level="warning", timeout_graceful_shutdown=10)


if __name__ == "__main__":
    main()
