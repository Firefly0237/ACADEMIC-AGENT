"""OIDC subject mapping and per-browser anonymous search scope.

Revision ID: s07_platform_0001
Revises: s06_search_0001
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "s07_platform_0001"
down_revision = "s06_search_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for name in ("oidc_identities", "guest_owners"):
        columns = [
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("owner_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("cq_app.owners.id"), nullable=False),
        ]
        if name == "oidc_identities":
            columns += [
                sa.Column("issuer", sa.String(1000), nullable=False),
                sa.Column("subject", sa.String(500), nullable=False),
                sa.UniqueConstraint("issuer", "subject"),
            ]
        else:
            columns += [sa.UniqueConstraint("owner_id")]
        op.create_table(name, *columns, schema="cq_app")
    op.create_index("ix_oidc_identities_owner_id", "oidc_identities", ["owner_id"], schema="cq_app")


def downgrade() -> None:
    raise RuntimeError("Identity rollback requires explicit data-preserving forward migration")
