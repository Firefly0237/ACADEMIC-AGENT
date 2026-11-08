"""Explicit product persistence bootstrap. Root coordinates main migration/setup execution."""
from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/s03"))
from bootstrap import dsn

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from sqlalchemy.engine import URL, make_url

DATA = ROOT / "data/runtime-s08-product"
ROLES = {"migrator": "cq_s08_persistence_migrator", "checkpoint": "cq_s08_checkpoint", "memory": "cq_s08_memory"}


def product_dsn(role: str, host: str) -> str:
    if host not in {"127.0.0.1", "host.docker.internal"}:
        raise RuntimeError("only_explicit_local_docker_target_allowed")
    original = make_url(dsn(role))
    if (original.host, original.port, original.database, original.username) != ("127.0.0.1", 25434, "citequest_s03", role):
        raise RuntimeError("existing_local_development_identity_mismatch")
    return original.set(drivername="postgresql", host=host).render_as_string(hide_password=False)


def prepare_secrets(host: str = "127.0.0.1") -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "credentials.json"
    if not path.exists():
        path.write_text(json.dumps({name: secrets.token_hex(32) for name in ROLES.values()}), encoding="utf-8")
    values = json.loads(path.read_text(encoding="utf-8"))
    for label, role in ROLES.items():
        value = URL.create("postgresql", username=role, password=values[role], host=host, port=25434,
            database="citequest_s03").render_as_string(hide_password=False)
        (DATA / (label + "_dsn")).write_text(value, encoding="utf-8")
    (DATA / "business_dsn").write_text(product_dsn("cq_s03_app", host), encoding="utf-8")
    print("Product S08 secret references prepared; no database operation or credential logging.")


def setup(host: str = "127.0.0.1") -> None:
    if not (DATA / "credentials.json").exists():
        raise RuntimeError("prepare_secrets_separately_first")
    values = json.loads((DATA / "credentials.json").read_text())
    with psycopg.connect(product_dsn("cq_s03_admin", host), row_factory=dict_row) as db:
        if db.execute("SELECT current_database() AS db,current_user AS role").fetchone() != {"db": "citequest_s03", "role": "cq_s03_admin"}:
            raise RuntimeError("explicit_product_database_identity_required")
        if db.execute("SELECT version_num FROM cq_app.alembic_version").fetchall() != [{"version_num": "s08_agent_0001"}]:
            raise RuntimeError("coordinated_S08_business_migration_required_before_setup")
        identity = db.execute("SELECT system_identifier::text AS system_identifier FROM pg_control_system()").fetchone()
        for role in ROLES.values():
            if not db.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,)).fetchone():
                db.execute(sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {}").format(sql.Identifier(role), sql.Literal(values[role])))
        for schema in ("cq_agent_checkpoint", "cq_agent_memory"):
            db.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {} AUTHORIZATION cq_s08_persistence_migrator").format(sql.Identifier(schema)))
            owner = db.execute("SELECT pg_get_userbyid(nspowner) AS owner FROM pg_namespace WHERE nspname=%s", (schema,)).fetchone()
            if owner["owner"] != ROLES["migrator"]:
                raise RuntimeError("persistence_schema_owner_mismatch")
            db.execute(sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC").format(sql.Identifier(schema)))
    migrator = (DATA / "migrator_dsn").read_text()
    for schema, factory in (("cq_agent_checkpoint", PostgresSaver), ("cq_agent_memory", PostgresStore)):
        with psycopg.connect(migrator, autocommit=True, row_factory=dict_row, prepare_threshold=0,
                options="-c search_path=" + schema) as connection:
            factory(connection).setup()
    with psycopg.connect(migrator) as db:
        for schema, role in (("cq_agent_checkpoint", ROLES["checkpoint"]), ("cq_agent_memory", ROLES["memory"])):
            db.execute(sql.SQL("REVOKE ALL ON SCHEMA {} FROM {}").format(sql.Identifier(schema), sql.Identifier(role)))
            db.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(schema), sql.Identifier(role)))
            db.execute(sql.SQL("GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA {} TO {}").format(sql.Identifier(schema), sql.Identifier(role)))
            for migration_table in ("checkpoint_migrations", "store_migrations"):
                exists = db.execute("SELECT 1 FROM information_schema.tables WHERE table_schema=%s AND table_name=%s", (schema, migration_table)).fetchone()
                if exists:
                    db.execute(sql.SQL("REVOKE INSERT,UPDATE,DELETE,TRUNCATE ON {}.{} FROM {}").format(sql.Identifier(schema), sql.Identifier(migration_table), sql.Identifier(role)))
    checks = {}
    for label, schema in (("checkpoint", "cq_agent_checkpoint"), ("memory", "cq_agent_memory")):
        with psycopg.connect((DATA / (label + "_dsn")).read_text(), row_factory=dict_row) as db:
            row = db.execute("SELECT rolsuper,rolcreatedb,rolcreaterole FROM pg_roles WHERE rolname=current_user").fetchone()
            if any(row.values()) or db.execute("SELECT has_schema_privilege(current_user,%s,'CREATE') AS ddl", (schema,)).fetchone()["ddl"]:
                raise RuntimeError("runtime_role_has_DDL_or_admin_privilege")
            try:
                db.execute("SELECT id FROM cq_app.agent_tasks LIMIT 1")
            except psycopg.errors.InsufficientPrivilege:
                checks[label + "_business_access_denied"] = True
            else:
                raise RuntimeError("persistence_role_has_business_access")
    (DATA / "setup-receipt.json").write_text(json.dumps({"status": "PASS", "database": "citequest_s03",
        "schema_revision": "s08_agent_0001", "local_target_host": host, "local_target_port": 25434,
        "cluster_identity": identity, "checks": checks, "credentials_logged": False}, indent=2), encoding="utf-8")
    print("Product official Saver/Store setup completed; runtime no-DDL/isolation checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare-secrets", "setup"])
    parser.add_argument("--host", choices=["127.0.0.1", "host.docker.internal"], default="127.0.0.1")
    args = parser.parse_args()
    prepare_secrets(args.host) if args.phase == "prepare-secrets" else setup(args.host)
