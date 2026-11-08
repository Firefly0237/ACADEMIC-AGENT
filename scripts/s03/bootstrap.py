"""Initialize only the new S03 database credentials/roles; never print secrets."""

import argparse
import json
import secrets
from pathlib import Path

import psycopg
from psycopg import sql
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/runtime-v2-app"
SECRETS = DATA / "credentials.json"


def credentials() -> dict:
    return json.loads(SECRETS.read_text(encoding="utf-8"))


def dsn(role: str, port: int = 25434, database: str = "citequest_s03") -> str:
    if port not in {25434, 25435} or not (
        database == "citequest_s03" or (port == 25435 and database.startswith("cq_s03_"))
    ):
        raise ValueError("Only owned S03 databases are allowed")
    return URL.create(
        "postgresql+psycopg",
        username=role,
        password=credentials()[role],
        host="127.0.0.1",
        port=port,
        database=database,
    ).render_as_string(hide_password=False)


def prepare() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if not SECRETS.exists():
        values = {r: secrets.token_hex(32) for r in ("cq_s03_admin", "cq_s03_migrator", "cq_s03_app")}
        SECRETS.write_text(json.dumps(values), encoding="utf-8")
    password = DATA / "postgres-password"
    expected = credentials()["cq_s03_admin"]
    if password.exists() and password.read_text().strip() != expected:
        raise RuntimeError("Existing S03 secret files disagree; no overwrite")
    if not password.exists():
        password.write_text(expected, encoding="utf-8")


def roles() -> None:
    values = credentials()
    with psycopg.connect(
        host="127.0.0.1", port=25434, user="cq_s03_admin", password=values["cq_s03_admin"], dbname="citequest_s03"
    ) as db:
        if db.execute("SELECT current_database(),current_user").fetchone() != ("citequest_s03", "cq_s03_admin"):
            raise RuntimeError("Wrong database identity")
        for name in ("cq_s03_migrator", "cq_s03_app"):
            if not db.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (name,)).fetchone():
                db.execute(
                    sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(name), sql.Literal(values[name]))
                )
        db.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        db.execute("CREATE SCHEMA IF NOT EXISTS cq_app AUTHORIZATION cq_s03_migrator")
        db.execute("GRANT USAGE ON SCHEMA cq_app TO cq_s03_app")
        db.execute(
            "ALTER DEFAULT PRIVILEGES FOR ROLE cq_s03_migrator IN SCHEMA cq_app GRANT SELECT,INSERT,UPDATE,DELETE ON TABLES TO cq_s03_app"
        )


def permissions(port: int = 25434, database: str = "citequest_s03", *, restored: bool = False) -> None:
    dsn("cq_s03_admin", port, database)  # Validate the narrow owned target.
    values = credentials()
    with psycopg.connect(
        host="127.0.0.1", port=port, user="cq_s03_admin", password=values["cq_s03_admin"], dbname=database
    ) as db:
        for name in ("cq_s03_migrator", "cq_s03_app"):
            if not db.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (name,)).fetchone():
                db.execute(
                    sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(name), sql.Literal(values[name]))
                )
        if restored:
            if port != 25435 or not database.startswith("cq_s03_"):
                raise ValueError("Restore ownership may only target a new owned restore database")
            db.execute("ALTER SCHEMA cq_app OWNER TO cq_s03_migrator")
            for (name,) in db.execute("SELECT tablename FROM pg_tables WHERE schemaname='cq_app'"):
                db.execute(sql.SQL("ALTER TABLE cq_app.{} OWNER TO cq_s03_migrator").format(sql.Identifier(name)))
            db.execute("ALTER FUNCTION cq_app.keep_outbox_payload() OWNER TO cq_s03_migrator")
        db.execute("GRANT USAGE ON SCHEMA cq_app TO cq_s03_app")
        db.execute("GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA cq_app TO cq_s03_app")
        db.execute("REVOKE INSERT,UPDATE,DELETE,TRUNCATE ON cq_app.alembic_version FROM cq_s03_app")
        db.execute("GRANT SELECT ON cq_app.alembic_version TO cq_s03_app")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "roles"])
    args = parser.parse_args()
    {"prepare": prepare, "roles": roles}[args.action]()
    print(json.dumps({"status": "pass", "action": args.action, "secrets_printed": False}))  # noqa: T201
