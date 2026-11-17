"""Durable bounded search snapshots; migration does not create or publish indexes."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "s06_search_0001"
down_revision = "s05_index_0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("search_snapshots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("generation_id", sa.Uuid(), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("request", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False),
        sa.Column("query_sha256", sa.String(64), nullable=False),
        sa.Column("page_size", sa.Integer(), nullable=False),
        sa.Column("candidate_limit", sa.Integer(), nullable=False),
        sa.Column("candidate_count", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"), sa.UniqueConstraint("id", "generation_id"),
        sa.ForeignKeyConstraint(["owner_id"], ["cq_app.owners.id"]),
        sa.ForeignKeyConstraint(["generation_id"], ["cq_app.index_generations.id"]),
        sa.CheckConstraint("page_size BETWEEN 1 AND 50 AND candidate_limit BETWEEN 1 AND 200 AND candidate_count BETWEEN 0 AND 200 AND epoch >= 0", name="bounded_search_snapshot"), schema="cq_app")
    op.create_index("ix_search_snapshot_owner_expiry", "search_snapshots", ["owner_id", "expires_at", "id"], schema="cq_app")
    op.create_index("ix_search_snapshot_generation_expiry", "search_snapshots", ["generation_id", "expires_at"], schema="cq_app")
    op.create_table("search_snapshot_hits",
        sa.Column("snapshot_id", sa.Uuid(), nullable=False), sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("generation_id", sa.Uuid(), nullable=False), sa.Column("projection_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.String(64), nullable=False), sa.Column("score", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id", "ordinal"), sa.UniqueConstraint("snapshot_id", "projection_id"),
        sa.ForeignKeyConstraint(["snapshot_id", "generation_id"], ["cq_app.search_snapshots.id", "cq_app.search_snapshots.generation_id"]),
        sa.ForeignKeyConstraint(["generation_id", "projection_id"], ["cq_app.index_units.generation_id", "cq_app.index_units.projection_id"]),
        sa.ForeignKeyConstraint(["generation_id", "chunk_id"], ["cq_app.index_chunks.generation_id", "cq_app.index_chunks.chunk_id"]),
        sa.CheckConstraint("ordinal BETWEEN 1 AND 200 AND score >= -1.0e308 AND score <= 1.0e308", name="finite_bounded_search_hit"), schema="cq_app")


def downgrade():
    op.drop_table("search_snapshot_hits", schema="cq_app")
    op.drop_index("ix_search_snapshot_generation_expiry", table_name="search_snapshots", schema="cq_app")
    op.drop_index("ix_search_snapshot_owner_expiry", table_name="search_snapshots", schema="cq_app")
    op.drop_table("search_snapshots", schema="cq_app")
