"""Scope private identities and versions; preserve one public canonical domain."""

import sqlalchemy as sa
from alembic import op

revision = "s03_0004"
down_revision = "s03_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    db = op.get_bind()
    if db.exec_driver_sql(
        "SELECT count(*) FROM cq_app.works w WHERE NOT w.public AND (SELECT count(*) FROM cq_app.work_access a WHERE a.work_id=w.id)<>1"
    ).scalar():
        raise RuntimeError("Explicit private Work scope backfill required; cannot infer one originating owner")
    if db.exec_driver_sql(
        "SELECT count(*) FROM cq_app.work_versions v WHERE (SELECT count(DISTINCT s.raw_security_domain) FROM cq_app.source_records s WHERE s.work_id=v.work_id AND s.raw_sha256=v.source_sha256)<>1"
    ).scalar():
        raise RuntimeError("Explicit version scope backfill required; cannot infer one source security domain")
    op.add_column(
        "owners", sa.Column("can_publish", sa.Boolean(), nullable=False, server_default=sa.false()), schema="cq_app"
    )
    op.add_column("works", sa.Column("security_domain", sa.String(100), nullable=True), schema="cq_app")
    op.execute(
        "UPDATE cq_app.works w SET security_domain=CASE WHEN w.public THEN 'public' ELSE 'owner:'||(SELECT a.owner_id::text FROM cq_app.work_access a WHERE a.work_id=w.id) END"
    )
    op.alter_column("works", "security_domain", nullable=False, schema="cq_app")
    op.create_index("ix_works_security_domain", "works", ["security_domain"], schema="cq_app")
    op.add_column("work_identifiers", sa.Column("security_domain", sa.String(100), nullable=True), schema="cq_app")
    op.execute(
        "UPDATE cq_app.work_identifiers i SET security_domain=w.security_domain FROM cq_app.works w WHERE i.work_id=w.id"
    )
    op.alter_column("work_identifiers", "security_domain", nullable=False, schema="cq_app")
    op.drop_constraint("pk_work_identifiers", "work_identifiers", type_="primary", schema="cq_app")
    op.create_primary_key(
        "pk_work_identifiers", "work_identifiers", ["security_domain", "namespace", "normalized"], schema="cq_app"
    )
    op.add_column("work_versions", sa.Column("security_domain", sa.String(100), nullable=True), schema="cq_app")
    op.execute(
        "UPDATE cq_app.work_versions v SET security_domain=(SELECT min(s.raw_security_domain) FROM cq_app.source_records s WHERE s.work_id=v.work_id AND s.raw_sha256=v.source_sha256)"
    )
    op.alter_column("work_versions", "security_domain", nullable=False, schema="cq_app")
    op.create_index("ix_work_versions_security_domain", "work_versions", ["security_domain"], schema="cq_app")
    op.add_column("source_records", sa.Column("security_domain", sa.String(100), nullable=True), schema="cq_app")
    op.execute("UPDATE cq_app.source_records SET security_domain=raw_security_domain")
    op.alter_column("source_records", "security_domain", nullable=False, schema="cq_app")
    op.create_table(
        "source_identities",
        sa.Column("security_domain", sa.String(100), primary_key=True),
        sa.Column("source", sa.String(100), primary_key=True),
        sa.Column("native_id", sa.String(512), primary_key=True),
        sa.Column("work_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["work_id"], ["cq_app.works.id"], name="fk_source_identities_work_id_works"),
        sa.PrimaryKeyConstraint("security_domain", "source", "native_id", name="pk_source_identities"),
        sa.UniqueConstraint("security_domain", "source", "native_id", "work_id", name="uq_source_identity_work"),
        schema="cq_app",
    )
    if db.exec_driver_sql(
        "SELECT count(*) FROM (SELECT security_domain,source,native_id FROM cq_app.source_records GROUP BY security_domain,source,native_id HAVING count(DISTINCT work_id)>1) s"
    ).scalar():
        raise RuntimeError("Conflicting source identities need explicit reconciliation")
    op.execute(
        "INSERT INTO cq_app.source_identities(security_domain,source,native_id,work_id) SELECT DISTINCT security_domain,source,native_id,work_id FROM cq_app.source_records"
    )
    op.drop_constraint(
        "uq_source_records_source_native_id_source_revision_raw_sha256",
        "source_records",
        type_="unique",
        schema="cq_app",
    )
    op.create_unique_constraint(
        "uq_source_record_scoped_revision",
        "source_records",
        ["security_domain", "source", "native_id", "source_revision", "raw_sha256"],
        schema="cq_app",
        postgresql_nulls_not_distinct=True,
    )
    op.create_foreign_key(
        "fk_source_record_scoped_identity",
        "source_records",
        "source_identities",
        ["security_domain", "source", "native_id", "work_id"],
        ["security_domain", "source", "native_id", "work_id"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )


def downgrade() -> None:
    raise RuntimeError(
        "Domain collapse could expose private identities or collide; use verified backup restore instead"
    )
