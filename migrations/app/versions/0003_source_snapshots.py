"""Retain exact raw source blobs and nullable unknown source revisions."""

import sqlalchemy as sa
from alembic import op

revision = "s03_0003"
down_revision = "s03_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing rows without a truthful retrieved_at/raw blob require a real
    # backfill; PostgreSQL refuses rather than inventing historical values.
    op.add_column(
        "source_records", sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False), schema="cq_app"
    )
    op.add_column("source_records", sa.Column("raw_security_domain", sa.String(100), nullable=False), schema="cq_app")
    op.alter_column("source_records", "source_revision", nullable=True, schema="cq_app")
    name = "uq_source_records_source_native_id_source_revision_raw_sha256"
    op.drop_constraint(name, "source_records", type_="unique", schema="cq_app")
    op.create_unique_constraint(
        name,
        "source_records",
        ["source", "native_id", "source_revision", "raw_sha256"],
        schema="cq_app",
        postgresql_nulls_not_distinct=True,
    )
    op.create_foreign_key(
        "fk_source_records_raw_security_domain_raw_sha256_blobs",
        "source_records",
        "blobs",
        ["raw_security_domain", "raw_sha256"],
        ["security_domain", "sha256"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )


def downgrade() -> None:
    # Unknown cannot become a fabricated non-null revision during rollback.
    connection = op.get_bind()
    if connection.exec_driver_sql("SELECT count(*) FROM cq_app.source_records WHERE source_revision IS NULL").scalar():
        raise RuntimeError("Downgrade requires an explicit truthful source-revision backfill or backup restore")
    op.drop_constraint(
        "fk_source_records_raw_security_domain_raw_sha256_blobs", "source_records", type_="foreignkey", schema="cq_app"
    )
    op.drop_constraint(
        "uq_source_records_source_native_id_source_revision_raw_sha256",
        "source_records",
        type_="unique",
        schema="cq_app",
    )
    op.create_unique_constraint(
        "uq_source_records_source_native_id_source_revision_raw_sha256",
        "source_records",
        ["source", "native_id", "source_revision", "raw_sha256"],
        schema="cq_app",
    )
    op.alter_column("source_records", "source_revision", nullable=False, schema="cq_app")
    op.drop_column("source_records", "retrieved_at", schema="cq_app")
    op.drop_column("source_records", "raw_security_domain", schema="cq_app")
