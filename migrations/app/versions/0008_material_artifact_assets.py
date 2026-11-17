"""Retain immutable native source views, geometry, maps and resource receipts."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "s04_0003"
down_revision = "s04_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("material_artifact_assets",
                    sa.Column("owner_id", sa.Uuid(), nullable=False), sa.Column("representation_id", sa.String(64), nullable=False),
                    sa.Column("role", sa.String(60), nullable=False), sa.Column("security_domain", sa.String(100), nullable=False),
                    sa.Column("sha256", sa.String(64), nullable=False), sa.Column("size_bytes", sa.BigInteger(), nullable=False),
                    sa.Column("provenance", JSONB(none_as_null=True), nullable=False),
                    sa.PrimaryKeyConstraint("owner_id", "representation_id", "role"),
                    sa.ForeignKeyConstraint(["owner_id"], ["cq_app.owners.id"]),
                    sa.ForeignKeyConstraint(["owner_id", "representation_id"], ["cq_app.parsed_representations.owner_id", "cq_app.parsed_representations.id"]),
                    sa.ForeignKeyConstraint(["security_domain", "sha256"], ["cq_app.blobs.security_domain", "cq_app.blobs.sha256"]),
                    sa.CheckConstraint("size_bytes > 0", name="positive_asset_size"), schema="cq_app")
    op.execute("CREATE TRIGGER immutable_material_artifact_assets BEFORE UPDATE OR DELETE ON cq_app.material_artifact_assets FOR EACH ROW EXECUTE FUNCTION cq_app.keep_material_artifact()")


def downgrade() -> None:
    if op.get_bind().exec_driver_sql("SELECT EXISTS (SELECT 1 FROM cq_app.material_artifact_assets)").scalar():
        raise RuntimeError("S04 artifact evidence cannot be discarded; use verified backup restore")
    op.drop_table("material_artifact_assets", schema="cq_app")
