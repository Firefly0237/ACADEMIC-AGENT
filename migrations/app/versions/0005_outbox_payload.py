"""Versioned immutable transactional outbox; do not invent old full snapshots."""

import hashlib
import json

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "s03_0005"
down_revision = "s03_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for name, type_ in (
        ("payload", JSONB()),
        ("schema_version", sa.String(100)),
        ("entity_type", sa.String(100)),
        ("security_domain", sa.String(100)),
    ):
        op.add_column("outbox", sa.Column(name, type_, nullable=True), schema="cq_app")
    db = op.get_bind()
    rows = db.exec_driver_sql("SELECT id,entity_id,payload_sha256 FROM cq_app.outbox").mappings().all()
    for row in rows:
        recovered = None
        for kind, key in (("Work", "work_id"), ("File", "file_id")):
            data = {key: row["entity_id"]}
            sha = hashlib.sha256(
                json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest()
            if sha == row["payload_sha256"]:
                recovered = kind, data
                break
        if recovered is None:
            raise RuntimeError(
                "Old hash-only outbox payload cannot be truthfully recovered; explicit reconciliation required"
            )
        kind, data = recovered
        table = "works" if kind == "Work" else "files"
        domain = db.exec_driver_sql(
            f"SELECT security_domain FROM cq_app.{table} WHERE id=%s", (row["entity_id"],)
        ).scalar_one()
        db.execute(
            sa.text(
                "UPDATE cq_app.outbox SET payload=CAST(:payload AS jsonb),schema_version='cq.outbox.legacy-id-only/0',entity_type=:kind,security_domain=:domain,state='error' WHERE id=:id"
            ),
            {"payload": json.dumps(data), "kind": kind, "domain": domain, "id": row["id"]},
        )
    for name, type_ in (
        ("payload", JSONB()),
        ("schema_version", sa.String(100)),
        ("entity_type", sa.String(100)),
        ("security_domain", sa.String(100)),
    ):
        op.alter_column("outbox", name, existing_type=type_, nullable=False, schema="cq_app")
    op.execute("""CREATE FUNCTION cq_app.keep_outbox_payload() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF ROW(NEW.idempotency_key,NEW.entity_id,NEW.entity_revision,NEW.operation,NEW.payload_sha256,NEW.payload,NEW.schema_version,NEW.entity_type,NEW.security_domain)
       IS DISTINCT FROM ROW(OLD.idempotency_key,OLD.entity_id,OLD.entity_revision,OLD.operation,OLD.payload_sha256,OLD.payload,OLD.schema_version,OLD.entity_type,OLD.security_domain)
      THEN RAISE EXCEPTION 'Outbox event payload is immutable' USING ERRCODE='23514'; END IF;
      RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER immutable_outbox_payload BEFORE UPDATE ON cq_app.outbox FOR EACH ROW EXECUTE FUNCTION cq_app.keep_outbox_payload()"
    )


def downgrade() -> None:
    raise RuntimeError("Dropping replayable payloads loses source history; use verified backup restore instead")
