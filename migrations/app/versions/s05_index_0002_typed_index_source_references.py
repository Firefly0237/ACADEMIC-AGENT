"""typed_index_source_references

Revision ID: s05_index_0002
Revises: s05_data_0002
Create Date: 2026-11-05 07:51:24.084388

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "s05_index_0002"
down_revision: str | Sequence[str] | None = "s05_data_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _guards() -> None:
    op.execute(r"""
CREATE FUNCTION cq_app.index_generation_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE s cq_app.ingest_snapshots%ROWTYPE; u cq_app.source_update_seals%ROWTYPE;
BEGIN
 IF TG_OP='DELETE' THEN RAISE EXCEPTION 'index generation history is append only'; END IF;
 IF NEW.source_root_kind IN ('metadata_snapshot','native_snapshot') THEN
  SELECT * INTO s FROM cq_app.ingest_snapshots WHERE id=NEW.snapshot_id;
  IF NOT FOUND OR s.security_domain<>NEW.security_domain OR
    (NEW.source_root_kind='metadata_snapshot' AND (s.purpose<>'standard' OR s.representation_profile<>'litsearch-clean-exact/1' OR NEW.catalog_name<>'standard_metadata')) OR
    (NEW.source_root_kind='native_snapshot' AND (s.purpose<>'application' OR s.representation_profile<>'litsearch-s2orc-native-text/1' OR NEW.catalog_name<>'application_papers')) THEN RAISE EXCEPTION 'index source snapshot family/domain mismatch'; END IF;
  IF TG_OP='INSERT' THEN
   NEW.source_namespace=COALESCE(NEW.source_namespace,s.source_namespace);
   NEW.source_manifest_sha256=COALESCE(NEW.source_manifest_sha256,s.source_manifest_sha256);
  END IF;
  IF NEW.source_namespace<>s.source_namespace OR NEW.source_manifest_sha256<>s.source_manifest_sha256 OR NEW.representation_profile<>s.representation_profile OR NEW.policy_sha256<>s.policy_sha256 OR NEW.evaluation_sha256<>s.evaluation_sha256 THEN RAISE EXCEPTION 'index source snapshot contract mismatch'; END IF;
 ELSE
  SELECT * INTO u FROM cq_app.source_update_seals WHERE id=NEW.update_seal_id;
  IF NOT FOUND OR u.security_domain<>NEW.security_domain OR u.source_namespace<>NEW.source_namespace OR u.seal_sha256<>NEW.source_manifest_sha256 OR NEW.catalog_name<>'managed_metadata' OR NEW.representation_profile<>'cq.protocol-metadata/1' THEN RAISE EXCEPTION 'index update seal contract mismatch'; END IF;
 END IF;
 IF TG_OP='UPDATE' THEN
  IF NOT ((OLD.state='preparing' AND NEW.state IN ('preparing','building','empty_unindexed','failed')) OR (OLD.state IN ('building','verifying') AND NEW.state IN ('building','verifying','verified','failed')) OR (OLD.state='verified' AND NEW.state IN ('verified','active','failed')) OR (OLD.state='active' AND NEW.state IN ('active','retired','failed')) OR (OLD.state='retired' AND NEW.state IN ('retired','active','failed')) OR (OLD.state IN ('failed','empty_unindexed') AND NEW.state=OLD.state)) THEN RAISE EXCEPTION 'index generation transition cannot reopen frozen membership'; END IF;
  IF (to_jsonb(NEW)-ARRAY['state','unit_count','empty_unit_count','denied_unit_count','chunk_count','membership_sha256','expected_manifest_sha256','source_seal_sha256','server_proof','readback_proof','error']) IS DISTINCT FROM (to_jsonb(OLD)-ARRAY['state','unit_count','empty_unit_count','denied_unit_count','chunk_count','membership_sha256','expected_manifest_sha256','source_seal_sha256','server_proof','readback_proof','error']) THEN RAISE EXCEPTION 'index generation identity immutable'; END IF;
  IF OLD.state<>'preparing' AND ROW(NEW.unit_count,NEW.empty_unit_count,NEW.denied_unit_count,NEW.chunk_count,NEW.membership_sha256,NEW.expected_manifest_sha256) IS DISTINCT FROM ROW(OLD.unit_count,OLD.empty_unit_count,OLD.denied_unit_count,OLD.chunk_count,OLD.membership_sha256,OLD.expected_manifest_sha256) THEN RAISE EXCEPTION 'index membership immutable after preparation'; END IF;
  IF OLD.state IN ('verified','active','retired') AND ROW(NEW.source_seal_sha256,NEW.server_proof,NEW.readback_proof) IS DISTINCT FROM ROW(OLD.source_seal_sha256,OLD.server_proof,OLD.readback_proof) THEN RAISE EXCEPTION 'verified index proofs immutable'; END IF;
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER index_generation_guard BEFORE INSERT OR UPDATE OR DELETE ON cq_app.index_generations FOR EACH ROW EXECUTE FUNCTION cq_app.index_generation_guard();
""")
    op.execute(r"""
CREATE FUNCTION cq_app.index_unit_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE g cq_app.index_generations%ROWTYPE; p cq_app.ingest_projections%ROWTYPE; r cq_app.ingest_rows%ROWTYPE; n cq_app.ingest_native_representations%ROWTYPE; o cq_app.source_update_observations%ROWTYPE; s cq_app.ingest_snapshots%ROWTYPE;
BEGIN
 SELECT * INTO g FROM cq_app.index_generations WHERE id=COALESCE(NEW.generation_id,OLD.generation_id) FOR SHARE;
 IF NOT FOUND OR g.state<>'preparing' THEN RAISE EXCEPTION 'index source units immutable after preparation'; END IF;
 IF TG_OP='DELETE' THEN RETURN OLD; END IF;
 IF TG_OP='UPDATE' THEN RAISE EXCEPTION 'staged index source unit is immutable'; END IF;
 IF NEW.source_reference_kind='metadata_projection' AND g.source_root_kind='metadata_snapshot' THEN
  NEW.metadata_projection_id=COALESCE(NEW.metadata_projection_id,NEW.projection_id);
  SELECT * INTO p FROM cq_app.ingest_projections WHERE id=NEW.metadata_projection_id;
  SELECT * INTO r FROM cq_app.ingest_rows WHERE id=p.source_row_id;
  SELECT * INTO s FROM cq_app.ingest_snapshots WHERE id=r.snapshot_id;
  IF p.id IS NULL OR NEW.projection_id<>p.id OR p.representation_profile<>g.representation_profile OR p.projection_sha256<>NEW.projection_sha256 THEN RAISE EXCEPTION 'index metadata reference mismatch'; END IF;
  NEW.source_namespace=COALESCE(NEW.source_namespace,s.source_namespace);
  NEW.source_revision=COALESCE(NEW.source_revision,s.source_revision);
 ELSIF NEW.source_reference_kind='native_representation' AND g.source_root_kind='native_snapshot' THEN
  SELECT * INTO n FROM cq_app.ingest_native_representations WHERE id=NEW.native_representation_id;
  SELECT * INTO r FROM cq_app.ingest_rows WHERE id=n.source_row_id;
  SELECT * INTO s FROM cq_app.ingest_snapshots WHERE id=r.snapshot_id;
  IF n.id IS NULL OR NEW.projection_id<>n.index_projection_id OR n.id<>NEW.projection_sha256 OR NEW.source_revision IS DISTINCT FROM s.source_revision THEN RAISE EXCEPTION 'index native reference mismatch'; END IF;
  IF NOT NEW.source_denied AND NOT EXISTS(SELECT 1 FROM cq_app.ingest_resolutions x WHERE x.source_row_id=r.id AND x.state='paper_identity_verified' AND x.representation_id=n.id AND x.work_id=NEW.work_id AND x.work_version_id=NEW.work_version_id AND cq_app.ingest_resolution_chain_valid(x) AND NOT EXISTS(SELECT 1 FROM cq_app.ingest_resolutions newer WHERE newer.source_row_id=x.source_row_id AND newer.revision>x.revision)) THEN RAISE EXCEPTION 'index native paper provenance mismatch'; END IF;
 ELSIF NEW.source_reference_kind='update_observation' AND g.source_root_kind='update_seal' THEN
  SELECT * INTO o FROM cq_app.source_update_observations WHERE id=NEW.update_observation_id;
  IF o.id IS NULL OR NEW.projection_id<>o.id OR o.security_domain<>g.security_domain OR NEW.native_id<>o.native_id OR NEW.source_namespace<>o.source_namespace OR NEW.source_revision IS DISTINCT FROM o.source_revision OR NEW.projection_sha256<>o.projection_sha256 OR NOT EXISTS(SELECT 1 FROM cq_app.source_update_seal_members sm WHERE sm.seal_id=g.update_seal_id AND sm.observation_id=o.id AND sm.projection_sha256=o.projection_sha256) THEN RAISE EXCEPTION 'index observation/seal member mismatch'; END IF;
  IF NOT NEW.source_denied AND (NEW.work_id IS DISTINCT FROM o.work_id OR NEW.work_version_id IS NOT NULL) THEN RAISE EXCEPTION 'observed metadata cannot invent paper version'; END IF;
 ELSE RAISE EXCEPTION 'index reference family mismatches source root'; END IF;
 IF r.id IS NOT NULL AND (r.snapshot_id<>g.snapshot_id OR r.security_domain<>g.security_domain OR r.native_id<>NEW.native_id OR r.source_pointer_sha256<>NEW.source_pointer_sha256 OR NEW.source_namespace<>g.source_namespace OR NEW.source_revision IS DISTINCT FROM s.source_revision) THEN RAISE EXCEPTION 'index source row lineage mismatch'; END IF;
 IF NEW.work_version_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM cq_app.work_versions v WHERE v.id=NEW.work_version_id AND v.work_id=NEW.work_id AND v.security_domain IN ('public',g.security_domain)) THEN RAISE EXCEPTION 'index WorkVersion lineage mismatch'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER index_unit_guard BEFORE INSERT OR UPDATE OR DELETE ON cq_app.index_units FOR EACH ROW EXECUTE FUNCTION cq_app.index_unit_guard();
CREATE FUNCTION cq_app.index_chunk_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE g cq_app.index_generations%ROWTYPE;
BEGIN
 SELECT * INTO g FROM cq_app.index_generations WHERE id=COALESCE(NEW.generation_id,OLD.generation_id) FOR SHARE;
 IF NOT FOUND OR g.state<>'preparing' OR TG_OP='UPDATE' THEN RAISE EXCEPTION 'staged index chunks immutable'; END IF;
 IF TG_OP='DELETE' THEN RETURN OLD; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER index_chunk_guard BEFORE INSERT OR UPDATE OR DELETE ON cq_app.index_chunks FOR EACH ROW EXECUTE FUNCTION cq_app.index_chunk_guard();
""")


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("index_generations", sa.Column("update_seal_id", sa.Uuid(), nullable=True), schema="cq_app")
    op.add_column(
        "index_generations",
        sa.Column("source_root_kind", sa.String(length=30), server_default="metadata_snapshot", nullable=False),
        schema="cq_app",
    )
    op.add_column(
        "index_generations",
        sa.Column("catalog_name", sa.String(length=100), server_default="standard_metadata", nullable=False),
        schema="cq_app",
    )
    op.add_column(
        "index_generations", sa.Column("source_namespace", sa.String(length=100), nullable=True), schema="cq_app"
    )
    op.add_column(
        "index_generations", sa.Column("source_manifest_sha256", sa.String(length=64), nullable=True), schema="cq_app"
    )
    op.add_column(
        "index_generations",
        sa.Column("denied_unit_count", sa.Integer(), server_default="0", nullable=False),
        schema="cq_app",
    )
    op.execute(
        "UPDATE cq_app.index_generations g SET source_namespace=s.source_namespace,source_manifest_sha256=s.source_manifest_sha256 FROM cq_app.ingest_snapshots s WHERE s.id=g.snapshot_id AND s.security_domain=g.security_domain"
    )
    op.alter_column("index_generations", "source_namespace", nullable=False, schema="cq_app")
    op.alter_column("index_generations", "source_manifest_sha256", nullable=False, schema="cq_app")
    op.alter_column("index_generations", "snapshot_id", existing_type=sa.UUID(), nullable=True, schema="cq_app")
    op.create_unique_constraint(
        op.f("uq_index_generations_id_security_domain_catalog_name"),
        "index_generations",
        ["id", "security_domain", "catalog_name"],
        schema="cq_app",
    )
    op.create_foreign_key(
        op.f("fk_index_catalogs_active_generation_id_security_domain_name_index_generations"),
        "index_catalogs",
        "index_generations",
        ["active_generation_id", "security_domain", "name"],
        ["id", "security_domain", "catalog_name"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )
    op.create_foreign_key(
        op.f("fk_index_generations_update_seal_id_source_update_seals"),
        "index_generations",
        "source_update_seals",
        ["update_seal_id"],
        ["id"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )
    op.add_column(
        "index_units",
        sa.Column("source_reference_kind", sa.String(length=30), server_default="metadata_projection", nullable=False),
        schema="cq_app",
    )
    op.add_column("index_units", sa.Column("metadata_projection_id", sa.Uuid(), nullable=True), schema="cq_app")
    op.add_column(
        "index_units", sa.Column("native_representation_id", sa.String(length=64), nullable=True), schema="cq_app"
    )
    op.add_column("index_units", sa.Column("update_observation_id", sa.Uuid(), nullable=True), schema="cq_app")
    op.add_column("index_units", sa.Column("source_namespace", sa.String(length=100), nullable=True), schema="cq_app")
    op.add_column("index_units", sa.Column("source_revision", sa.String(length=100), nullable=True), schema="cq_app")
    op.add_column("index_units", sa.Column("work_id", sa.Uuid(), nullable=True), schema="cq_app")
    op.add_column("index_units", sa.Column("work_version_id", sa.Uuid(), nullable=True), schema="cq_app")
    op.add_column(
        "index_units", sa.Column("source_denied", sa.Boolean(), server_default="false", nullable=False), schema="cq_app"
    )
    op.execute(
        "UPDATE cq_app.index_units u SET metadata_projection_id=u.projection_id,source_namespace=g.source_namespace,source_revision=s.source_revision FROM cq_app.index_generations g JOIN cq_app.ingest_snapshots s ON s.id=g.snapshot_id WHERE g.id=u.generation_id"
    )
    op.alter_column("index_units", "source_namespace", nullable=False, schema="cq_app")
    op.create_index(
        "ix_index_unit_current_source",
        "index_units",
        ["generation_id", "source_namespace", "native_id", "source_revision"],
        unique=False,
        schema="cq_app",
    )
    op.drop_constraint(
        op.f("fk_index_units_projection_id_ingest_projections"), "index_units", schema="cq_app", type_="foreignkey"
    )
    op.create_foreign_key(
        op.f("fk_index_units_work_version_id_work_versions"),
        "index_units",
        "work_versions",
        ["work_version_id"],
        ["id"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )
    op.create_foreign_key(
        op.f("fk_index_units_work_id_works"),
        "index_units",
        "works",
        ["work_id"],
        ["id"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )
    op.create_foreign_key(
        op.f("fk_index_units_metadata_projection_id_ingest_projections"),
        "index_units",
        "ingest_projections",
        ["metadata_projection_id"],
        ["id"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )
    op.create_foreign_key(
        op.f("fk_index_units_native_representation_id_ingest_native_representations"),
        "index_units",
        "ingest_native_representations",
        ["native_representation_id"],
        ["id"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )
    op.create_foreign_key(
        op.f("fk_index_units_update_observation_id_source_update_observations"),
        "index_units",
        "source_update_observations",
        ["update_observation_id"],
        ["id"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )
    op.create_check_constraint(
        op.f("ck_index_generations_exact_typed_index_source_root"),
        "index_generations",
        "(source_root_kind IN ('metadata_snapshot','native_snapshot') AND snapshot_id IS NOT NULL AND update_seal_id IS NULL) OR (source_root_kind='update_seal' AND snapshot_id IS NULL AND update_seal_id IS NOT NULL)",
        schema="cq_app",
    )
    op.create_check_constraint(
        op.f("ck_index_units_exact_typed_index_source_reference"),
        "index_units",
        "(source_reference_kind='metadata_projection' AND metadata_projection_id IS NOT NULL AND native_representation_id IS NULL AND update_observation_id IS NULL) OR (source_reference_kind='native_representation' AND metadata_projection_id IS NULL AND native_representation_id IS NOT NULL AND update_observation_id IS NULL) OR (source_reference_kind='update_observation' AND metadata_projection_id IS NULL AND native_representation_id IS NULL AND update_observation_id IS NOT NULL)",
        schema="cq_app",
    )
    op.drop_constraint(
        op.f("ck_index_generations_known_index_generation_state"), "index_generations", schema="cq_app", type_="check"
    )
    op.create_check_constraint(
        op.f("ck_index_generations_known_index_generation_state"),
        "index_generations",
        "state IN ('preparing','building','verifying','verified','active','retired','failed','empty_unindexed')",
        schema="cq_app",
    )
    _guards()


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        "DO $$ BEGIN IF EXISTS(SELECT 1 FROM cq_app.index_generations WHERE source_root_kind<>'metadata_snapshot' OR state='empty_unindexed') THEN RAISE EXCEPTION 'typed source history prevents lossy downgrade'; END IF; END $$"
    )
    for table, name in (
        ("index_chunks", "index_chunk_guard"),
        ("index_units", "index_unit_guard"),
        ("index_generations", "index_generation_guard"),
    ):
        op.execute(f"DROP TRIGGER {name} ON cq_app.{table}")
        op.execute(f"DROP FUNCTION cq_app.{name}()")
    op.drop_constraint(
        op.f("ck_index_generations_exact_typed_index_source_root"), "index_generations", schema="cq_app", type_="check"
    )
    op.drop_constraint(
        op.f("ck_index_units_exact_typed_index_source_reference"), "index_units", schema="cq_app", type_="check"
    )
    op.drop_constraint(
        op.f("ck_index_generations_known_index_generation_state"), "index_generations", schema="cq_app", type_="check"
    )
    op.create_check_constraint(
        op.f("ck_index_generations_known_index_generation_state"),
        "index_generations",
        "state IN ('preparing','building','verifying','verified','active','retired','failed')",
        schema="cq_app",
    )
    op.drop_constraint(
        op.f("fk_index_catalogs_active_generation_id_security_domain_name_index_generations"),
        "index_catalogs",
        schema="cq_app",
        type_="foreignkey",
    )
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(
        op.f("fk_index_units_update_observation_id_source_update_observations"),
        "index_units",
        schema="cq_app",
        type_="foreignkey",
    )
    op.drop_constraint(
        op.f("fk_index_units_native_representation_id_ingest_native_representations"),
        "index_units",
        schema="cq_app",
        type_="foreignkey",
    )
    op.drop_constraint(
        op.f("fk_index_units_metadata_projection_id_ingest_projections"),
        "index_units",
        schema="cq_app",
        type_="foreignkey",
    )
    op.drop_constraint(op.f("fk_index_units_work_id_works"), "index_units", schema="cq_app", type_="foreignkey")
    op.drop_constraint(
        op.f("fk_index_units_work_version_id_work_versions"), "index_units", schema="cq_app", type_="foreignkey"
    )
    op.create_foreign_key(
        op.f("fk_index_units_projection_id_ingest_projections"),
        "index_units",
        "ingest_projections",
        ["projection_id"],
        ["id"],
        source_schema="cq_app",
        referent_schema="cq_app",
    )
    op.drop_index("ix_index_unit_current_source", table_name="index_units", schema="cq_app")
    op.drop_column("index_units", "source_denied", schema="cq_app")
    op.drop_column("index_units", "work_version_id", schema="cq_app")
    op.drop_column("index_units", "work_id", schema="cq_app")
    op.drop_column("index_units", "source_revision", schema="cq_app")
    op.drop_column("index_units", "source_namespace", schema="cq_app")
    op.drop_column("index_units", "update_observation_id", schema="cq_app")
    op.drop_column("index_units", "native_representation_id", schema="cq_app")
    op.drop_column("index_units", "metadata_projection_id", schema="cq_app")
    op.drop_column("index_units", "source_reference_kind", schema="cq_app")
    op.drop_constraint(
        op.f("fk_index_generations_update_seal_id_source_update_seals"),
        "index_generations",
        schema="cq_app",
        type_="foreignkey",
    )
    op.drop_constraint(
        op.f("uq_index_generations_id_security_domain_catalog_name"),
        "index_generations",
        schema="cq_app",
        type_="unique",
    )
    op.alter_column("index_generations", "snapshot_id", existing_type=sa.UUID(), nullable=False, schema="cq_app")
    op.drop_column("index_generations", "denied_unit_count", schema="cq_app")
    op.drop_column("index_generations", "source_manifest_sha256", schema="cq_app")
    op.drop_column("index_generations", "source_namespace", schema="cq_app")
    op.drop_column("index_generations", "catalog_name", schema="cq_app")
    op.drop_column("index_generations", "source_root_kind", schema="cq_app")
    op.drop_column("index_generations", "update_seal_id", schema="cq_app")
    # ### end Alembic commands ###
