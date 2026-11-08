"""Fail-closed adoption for new numerical encoder backends; original CPU unchanged.

Revision ID: s05_index_0003
Revises: s05_data_0003
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "s05_index_0003"
down_revision = "s05_data_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "index_backend_bindings",
        sa.Column("generation_id", sa.Uuid(), nullable=False),
        sa.Column("backend_profile_sha256", sa.String(64), nullable=False),
        sa.Column("expected_manifest_sha256", sa.String(64), nullable=False),
        sa.Column("profile", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False),
        sa.Column("profile_canonical", sa.Text(), nullable=False),
        sa.Column("bound_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["generation_id"], ["cq_app.index_generations.id"]),
        sa.PrimaryKeyConstraint("generation_id"),
        schema="cq_app",
    )
    op.create_table(
        "index_backend_adoptions",
        sa.Column("generation_id", sa.Uuid(), nullable=False),
        sa.Column("backend_profile_sha256", sa.String(64), nullable=False),
        sa.Column("expected_manifest_sha256", sa.String(64), nullable=False),
        sa.Column("standard_quality_generation_id", sa.Uuid(), nullable=False),
        sa.Column("evaluation_sha256", sa.String(64), nullable=False),
        sa.Column("quality_report_sha256", sa.String(64), nullable=False),
        sa.Column("approval_sha256", sa.String(64), nullable=False),
        sa.Column("evaluation_queries", sa.Integer(), nullable=False),
        sa.Column("evaluation_qrels_edges", sa.Integer(), nullable=False),
        sa.Column("metrics", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False),
        sa.Column("adopted_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["generation_id"], ["cq_app.index_generations.id"]),
        sa.ForeignKeyConstraint(["standard_quality_generation_id"], ["cq_app.index_generations.id"]),
        sa.PrimaryKeyConstraint("generation_id"),
        schema="cq_app",
    )
    op.execute("REVOKE INSERT,UPDATE,DELETE,TRUNCATE ON cq_app.index_backend_adoptions FROM cq_s03_app")
    op.execute("GRANT SELECT ON cq_app.index_backend_adoptions TO cq_s03_app")
    op.execute("REVOKE INSERT,UPDATE,DELETE,TRUNCATE ON cq_app.index_backend_bindings FROM cq_s03_app")
    op.execute("GRANT SELECT ON cq_app.index_backend_bindings TO cq_s03_app")
    op.execute("""
    CREATE FUNCTION cq_app.index_full_source_publication_ready(generation uuid) RETURNS boolean
    LANGUAGE sql STABLE AS $$
      SELECT COALESCE(
        g.build_scope='full' AND g.state IN ('verified','active','retired')
        AND s.id=g.snapshot_id AND s.security_domain=g.security_domain
        AND s.status='staged' AND s.profile_adopted AND s.seal_sha256=g.source_seal_sha256
        AND g.server_proof IS NOT NULL AND g.server_proof->>'profile_sha256'=g.index_profile_sha256
        AND g.readback_proof->>'complete'='true'
        AND (g.readback_proof->>'matched')::bigint=g.chunk_count
        AND g.readback_proof->>'rows_sha256'=g.expected_manifest_sha256
        AND (SELECT count(*) FROM cq_app.index_units u WHERE u.generation_id=g.id)=g.unit_count
        AND (SELECT count(*) FROM cq_app.index_chunks c WHERE c.generation_id=g.id)=g.chunk_count
        AND NOT EXISTS(SELECT 1 FROM cq_app.index_batches b WHERE b.generation_id=g.id AND b.state<>'verified')
        AND ((g.source_root_kind='metadata_snapshot' AND s.purpose='standard'
              AND s.representation_profile='litsearch-clean-exact/1' AND g.unit_count=64183)
          OR (g.source_root_kind='native_snapshot' AND s.purpose='application'
              AND s.representation_profile='litsearch-s2orc-native-text/1' AND g.unit_count BETWEEN 500 AND 2000
              AND (SELECT count(*) FROM cq_app.application_corpus_members a WHERE a.snapshot_id=s.id)=g.unit_count
              AND NOT EXISTS(SELECT 1 FROM cq_app.index_units u LEFT JOIN cq_app.application_corpus_members a
                ON a.snapshot_id=s.id AND a.representation_id=u.native_representation_id
                WHERE u.generation_id=g.id AND a.representation_id IS NULL)))
      ,false) FROM cq_app.index_generations g JOIN cq_app.ingest_snapshots s ON s.id=g.snapshot_id WHERE g.id=generation
    $$;
    CREATE FUNCTION cq_app.index_backend_binding_guard() RETURNS trigger
    LANGUAGE plpgsql AS $$ DECLARE g cq_app.index_generations; BEGIN
      IF TG_OP <> 'INSERT' THEN RAISE EXCEPTION 'backend_binding_append_only'; END IF;
      IF current_user <> 'cq_s03_migrator' THEN RAISE EXCEPTION 'migrator_backend_binding_required'; END IF;
      SELECT * INTO STRICT g FROM cq_app.index_generations WHERE id=NEW.generation_id;
      IF g.state <> 'building' OR g.build_scope <> 'full' OR g.server_proof IS NOT NULL
         OR g.expected_manifest_sha256 IS DISTINCT FROM NEW.expected_manifest_sha256
         OR NEW.backend_profile_sha256 !~ '^[a-f0-9]{64}$'
         OR NEW.profile IS DISTINCT FROM NEW.profile_canonical::jsonb
         OR encode(sha256(convert_to(NEW.profile_canonical,'UTF8')),'hex') IS DISTINCT FROM NEW.backend_profile_sha256
      THEN RAISE EXCEPTION 'prepared_full_generation_backend_binding_required'; END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER index_backend_binding_immutable BEFORE INSERT OR UPDATE OR DELETE
      ON cq_app.index_backend_bindings FOR EACH ROW EXECUTE FUNCTION cq_app.index_backend_binding_guard();
    CREATE FUNCTION cq_app.index_backend_adoption_guard() RETURNS trigger
    LANGUAGE plpgsql AS $$
    DECLARE g cq_app.index_generations; q cq_app.index_generations; metric text; method text;
      lower_ci double precision; upper_ci double precision; delta double precision;
    BEGIN
      IF TG_OP <> 'INSERT' THEN RAISE EXCEPTION 'backend_adoption_append_only'; END IF;
      IF current_user <> 'cq_s03_migrator' THEN RAISE EXCEPTION 'migrator_backend_approval_required'; END IF;
      SELECT * INTO STRICT g FROM cq_app.index_generations WHERE id=NEW.generation_id;
      SELECT * INTO STRICT q FROM cq_app.index_generations WHERE id=NEW.standard_quality_generation_id;
      IF cq_app.index_full_source_publication_ready(g.id) IS NOT TRUE
         OR g.server_proof->>'backend_profile_sha256' IS DISTINCT FROM NEW.backend_profile_sha256
         OR g.expected_manifest_sha256 IS DISTINCT FROM NEW.expected_manifest_sha256
         OR g.evaluation_sha256 IS DISTINCT FROM NEW.evaluation_sha256
         OR g.readback_proof->>'complete' IS DISTINCT FROM 'true'
         OR (g.readback_proof->>'matched')::bigint IS DISTINCT FROM g.chunk_count
         OR g.readback_proof->>'rows_sha256' IS DISTINCT FROM g.expected_manifest_sha256
         OR q.source_root_kind <> 'metadata_snapshot' OR q.build_scope <> 'full' OR q.unit_count <> 64183
         OR q.state NOT IN ('verified','active','retired')
         OR q.readback_proof->>'complete' IS DISTINCT FROM 'true'
         OR (q.readback_proof->>'matched')::bigint IS DISTINCT FROM q.chunk_count
         OR q.readback_proof->>'rows_sha256' IS DISTINCT FROM q.expected_manifest_sha256
         OR q.server_proof->>'backend_profile_sha256' IS DISTINCT FROM NEW.backend_profile_sha256
         OR q.evaluation_sha256 IS DISTINCT FROM NEW.evaluation_sha256
         OR NEW.evaluation_queries <> 597 OR NEW.evaluation_qrels_edges <> 639
         OR NEW.backend_profile_sha256 !~ '^[a-f0-9]{64}$'
         OR NEW.quality_report_sha256 !~ '^[a-f0-9]{64}$' OR NEW.approval_sha256 !~ '^[a-f0-9]{64}$'
      THEN RAISE EXCEPTION 'complete_current_generation_and_fixed_standard_quality_required'; END IF;
      IF NOT EXISTS(SELECT 1 FROM cq_app.index_backend_bindings b WHERE b.generation_id=g.id
          AND b.backend_profile_sha256=NEW.backend_profile_sha256
          AND b.expected_manifest_sha256=NEW.expected_manifest_sha256)
      THEN RAISE EXCEPTION 'trusted_backend_binding_required'; END IF;
      IF g.id <> q.id AND NOT EXISTS(SELECT 1 FROM cq_app.index_backend_adoptions a
          WHERE a.generation_id=q.id AND a.backend_profile_sha256=NEW.backend_profile_sha256
          AND a.quality_report_sha256=NEW.quality_report_sha256)
      THEN RAISE EXCEPTION 'standard_backend_quality_must_be_adopted_first'; END IF;
      FOREACH method IN ARRAY ARRAY['bm25','dense','hybrid'] LOOP
        FOREACH metric IN ARRAY ARRAY['recall20','ndcg10'] LOOP
          IF jsonb_typeof(NEW.metrics->method->metric) IS DISTINCT FROM 'number'
             OR NOT ((NEW.metrics->method->>metric)::double precision BETWEEN 0 AND 1)
          THEN RAISE EXCEPTION 'all_three_methods_full_quality_metrics_required'; END IF;
        END LOOP;
      END LOOP;
      IF (NEW.metrics->'hybrid'->>'recall20')::double precision
           - (NEW.metrics->'bm25'->>'recall20')::double precision < -0.02
         OR NEW.metrics->>'excluded_queries' IS DISTINCT FROM '0'
         OR NEW.metrics->>'status' IS DISTINCT FROM 'pass'
         OR NEW.metrics->>'attempt_errors' IS DISTINCT FROM '0'
         OR NEW.metrics->>'query_count' IS DISTINCT FROM '597'
         OR NEW.metrics->>'qrels_edges' IS DISTINCT FROM '639'
         OR NEW.metrics->>'dense_enabled' IS DISTINCT FROM 'true'
         OR NEW.metrics->>'bm25_enabled' IS DISTINCT FROM 'true'
         OR jsonb_typeof(NEW.metrics->'recall20_delta_ci95') IS DISTINCT FROM 'array'
         OR jsonb_array_length(NEW.metrics->'recall20_delta_ci95') <> 2
         OR jsonb_typeof(NEW.metrics->'ndcg10_delta_ci95') IS DISTINCT FROM 'array'
         OR jsonb_array_length(NEW.metrics->'ndcg10_delta_ci95') <> 2
      THEN RAISE EXCEPTION 'full_hybrid_quality_gate_and_difference_intervals_required'; END IF;
      FOREACH metric IN ARRAY ARRAY['recall20','ndcg10'] LOOP
        IF jsonb_typeof(NEW.metrics->(metric||'_delta_ci95')->0) IS DISTINCT FROM 'number'
           OR jsonb_typeof(NEW.metrics->(metric||'_delta_ci95')->1) IS DISTINCT FROM 'number'
        THEN RAISE EXCEPTION 'numeric_difference_interval_required'; END IF;
        lower_ci=(NEW.metrics->(metric||'_delta_ci95')->>0)::double precision;
        upper_ci=(NEW.metrics->(metric||'_delta_ci95')->>1)::double precision;
        delta=(NEW.metrics->'hybrid'->>metric)::double precision - (NEW.metrics->'bm25'->>metric)::double precision;
        IF NOT (lower_ci BETWEEN -1 AND 1 AND upper_ci BETWEEN -1 AND 1
                AND lower_ci <= delta AND delta <= upper_ci)
        THEN RAISE EXCEPTION 'ordered_difference_interval_must_contain_point_estimate'; END IF;
      END LOOP;
      RETURN NEW;
    END $$;
    CREATE TRIGGER index_backend_adoption_immutable BEFORE INSERT OR UPDATE OR DELETE
      ON cq_app.index_backend_adoptions FOR EACH ROW EXECUTE FUNCTION cq_app.index_backend_adoption_guard();
    """)
    op.execute("""
    CREATE FUNCTION cq_app.index_backend_publication_allowed(generation uuid) RETURNS boolean
    LANGUAGE sql STABLE AS $$
      SELECT cq_app.index_full_source_publication_ready(g.id) AND CASE WHEN EXISTS(SELECT 1 FROM cq_app.index_backend_bindings b WHERE b.generation_id=g.id)
        OR g.server_proof ? 'backend_profile_sha256' THEN EXISTS(
        SELECT 1 FROM cq_app.index_backend_adoptions a WHERE a.generation_id=g.id
          AND a.backend_profile_sha256=g.server_proof->>'backend_profile_sha256'
          AND a.expected_manifest_sha256=g.expected_manifest_sha256 AND a.evaluation_sha256=g.evaluation_sha256
      ) ELSE true END FROM cq_app.index_generations g WHERE g.id=generation
    $$;
    CREATE FUNCTION cq_app.index_backend_publication_guard() RETURNS trigger
    LANGUAGE plpgsql AS $$ DECLARE target uuid; b cq_app.index_backend_bindings; BEGIN
      IF TG_TABLE_NAME='index_publish_intents' THEN
        IF NEW.state IN ('rolled_back','reconcile_required') THEN RETURN NEW; END IF;
        target=NEW.target_generation_id;
      ELSIF TG_TABLE_NAME='index_catalogs' THEN target=NEW.active_generation_id;
      ELSE
        SELECT * INTO b FROM cq_app.index_backend_bindings WHERE generation_id=NEW.id;
        IF FOUND AND NEW.server_proof IS NOT NULL AND NEW.server_proof->>'backend_profile_sha256' IS DISTINCT FROM b.backend_profile_sha256
        THEN RAISE EXCEPTION 'business_proof_cannot_erase_trusted_backend_binding'; END IF;
        IF NEW.server_proof ? 'backend_profile_sha256' AND NOT FOUND
        THEN RAISE EXCEPTION 'migrator_backend_binding_required_before_proof'; END IF;
        IF NEW.state <> 'active' THEN RETURN NEW; END IF; target=NEW.id;
      END IF;
      IF target IS NOT NULL AND cq_app.index_backend_publication_allowed(target) IS NOT TRUE
      THEN RAISE EXCEPTION 'candidate_backend_requires_migrator_quality_adoption'; END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER index_backend_publish_intent BEFORE INSERT OR UPDATE ON cq_app.index_publish_intents
      FOR EACH ROW EXECUTE FUNCTION cq_app.index_backend_publication_guard();
    CREATE TRIGGER index_backend_active_catalog BEFORE INSERT OR UPDATE ON cq_app.index_catalogs
      FOR EACH ROW EXECUTE FUNCTION cq_app.index_backend_publication_guard();
    CREATE TRIGGER index_backend_active_generation BEFORE UPDATE ON cq_app.index_generations
      FOR EACH ROW EXECUTE FUNCTION cq_app.index_backend_publication_guard();
    """)


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS(SELECT 1 FROM cq_app.index_backend_adoptions)
      OR EXISTS(SELECT 1 FROM cq_app.index_backend_bindings)
      OR EXISTS(SELECT 1 FROM cq_app.index_generations WHERE server_proof ? 'backend_profile_sha256')
      THEN RAISE EXCEPTION 'backend_history_requires_forward_only_recovery'; END IF; END $$""")
    for trigger, table in [
        ("index_backend_publish_intent", "index_publish_intents"),
        ("index_backend_active_catalog", "index_catalogs"),
        ("index_backend_active_generation", "index_generations"),
    ]:
        op.execute(f"DROP TRIGGER {trigger} ON cq_app.{table}")
    op.execute("DROP FUNCTION cq_app.index_backend_publication_guard()")
    op.execute("DROP FUNCTION cq_app.index_backend_publication_allowed(uuid)")
    op.drop_table("index_backend_adoptions", schema="cq_app")
    op.execute("DROP FUNCTION cq_app.index_backend_adoption_guard()")
    op.drop_table("index_backend_bindings", schema="cq_app")
    op.execute("DROP FUNCTION cq_app.index_backend_binding_guard()")
    op.execute("DROP FUNCTION cq_app.index_full_source_publication_ready(uuid)")
