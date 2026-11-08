"""Enforce same-owner attachment associations at the PostgreSQL FK boundary."""

import sqlalchemy as sa
from alembic import op

revision = "s03_0002"
down_revision = "s03_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint("uq_files_owner_id_id", "files", ["owner_id", "id"], schema="cq_app")
    op.add_column("item_files", sa.Column("owner_id", sa.Uuid(), nullable=True), schema="cq_app")
    op.execute("UPDATE cq_app.item_files f SET owner_id=i.owner_id FROM cq_app.library_items i WHERE i.id=f.item_id")
    op.alter_column("item_files", "owner_id", nullable=False, schema="cq_app")
    op.drop_constraint("fk_item_files_item_id_library_items", "item_files", type_="foreignkey", schema="cq_app")
    op.drop_constraint("fk_item_files_file_id_files", "item_files", type_="foreignkey", schema="cq_app")
    op.drop_constraint("pk_item_files", "item_files", type_="primary", schema="cq_app")
    op.create_primary_key("pk_item_files", "item_files", ["owner_id", "item_id", "file_id"], schema="cq_app")
    op.create_foreign_key(
        "fk_item_files_owner_id_item_id_library_items",
        "item_files",
        "library_items",
        ["owner_id", "item_id"],
        ["owner_id", "id"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )
    op.create_foreign_key(
        "fk_item_files_owner_id_file_id_files",
        "item_files",
        "files",
        ["owner_id", "file_id"],
        ["owner_id", "id"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_item_files_owner_id_item_id_library_items", "item_files", type_="foreignkey", schema="cq_app"
    )
    op.drop_constraint("fk_item_files_owner_id_file_id_files", "item_files", type_="foreignkey", schema="cq_app")
    op.drop_constraint("pk_item_files", "item_files", type_="primary", schema="cq_app")
    op.create_primary_key("pk_item_files", "item_files", ["item_id", "file_id"], schema="cq_app")
    op.create_foreign_key(
        "fk_item_files_item_id_library_items",
        "item_files",
        "library_items",
        ["item_id"],
        ["id"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )
    op.create_foreign_key(
        "fk_item_files_file_id_files",
        "item_files",
        "files",
        ["file_id"],
        ["id"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )
    op.drop_column("item_files", "owner_id", schema="cq_app")
    op.drop_constraint("uq_files_owner_id_id", "files", type_="unique", schema="cq_app")
