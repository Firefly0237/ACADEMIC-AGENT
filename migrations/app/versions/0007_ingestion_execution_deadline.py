"""Separate artifact state from measured ingestion deadline/commit outcome."""
import sqlalchemy as sa
from alembic import op

revision = "s04_0002"
down_revision = "s04_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for name, kind in (("execution_id", sa.Uuid()), ("execution_started_at", sa.DateTime(timezone=True)),
                       ("deadline_at", sa.DateTime(timezone=True)), ("execution_budget_seconds", sa.Float()), ("active_elapsed_seconds", sa.Float())):
        op.add_column("material_ingestions", sa.Column(name, kind, nullable=True), schema="cq_app")
    op.add_column("material_ingestions", sa.Column("execution_outcome", sa.String(30), nullable=False, server_default="not_started"), schema="cq_app")
    op.execute("UPDATE cq_app.material_ingestions SET execution_outcome='legacy_not_measured' WHERE state <> 'queued'")


def downgrade() -> None:
    if op.get_bind().exec_driver_sql("SELECT EXISTS (SELECT 1 FROM cq_app.material_ingestions WHERE execution_id IS NOT NULL)").scalar():
        raise RuntimeError("Measured S04 execution history cannot be discarded; use verified backup restore")
    for name in ("execution_outcome", "active_elapsed_seconds", "execution_budget_seconds", "deadline_at", "execution_started_at", "execution_id"):
        op.drop_column("material_ingestions", name, schema="cq_app")
