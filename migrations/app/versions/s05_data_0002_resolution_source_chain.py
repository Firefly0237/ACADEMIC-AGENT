"""Validate coordinated resolution identities without rewriting source facts.

Revision ID: s05_data_0002
Revises: s05_update_0001
"""
from alembic import op

revision = "s05_data_0002"
down_revision = "s05_update_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
    CREATE FUNCTION cq_app.ingest_resolution_chain_valid(r cq_app.ingest_resolutions)
    RETURNS boolean LANGUAGE sql STABLE AS $$
      SELECT CASE WHEN r.state='unresolved' THEN true ELSE COALESCE(EXISTS (
        SELECT 1 FROM cq_app.ingest_rows sr
        JOIN cq_app.ingest_snapshots ss ON ss.id=sr.snapshot_id
        JOIN cq_app.source_identities si ON si.security_domain=sr.security_domain
          AND si.source='litsearch' AND si.native_id=sr.native_id AND si.work_id=r.work_id
        WHERE sr.id=r.source_row_id AND sr.security_domain=r.security_domain
          AND ss.source_namespace IN ('litsearch_clean','litsearch_s2orc')
          AND (r.state='metadata_resolved' OR (
            ss.source_namespace='litsearch_s2orc' AND ss.purpose='application'
            AND ss.representation_profile='litsearch-s2orc-native-text/1'
            AND r.proof->>'native_source_pointer_sha256'=sr.source_pointer_sha256
            AND r.proof->>'source_chain_manifest_sha256'='fd09876d6863e51f25f822b3bb03bf41ee3756314a22b952edfb3c5d3b309813'
            AND r.proof->>'rights_manifest_sha256'='e43140d1f1fc118d7692ae562cde983f3399ba05309cd4365a6a3b95647766d3'
            AND r.proof->>'copyright_policy_sha256'='0150e838394bae4268abc506fa5612f6ab3a6e7110b9718b52b4a591d754246f'
            AND r.proof->'source_reference'->>'representation_id'=r.representation_id
            AND r.proof->'source_reference'->>'work_version_id'=r.work_version_id::text
            AND r.proof->'source_reference'->>'source_snapshot_sha256'=ss.source_manifest_sha256
            AND EXISTS (
              SELECT 1 FROM cq_app.work_versions v
              JOIN cq_app.source_records s ON s.work_id=v.work_id
                AND s.security_domain=r.security_domain
                AND s.raw_security_domain='owner:' || ss.importer_owner_id::text
                AND s.id::text=r.proof->>'source_record_id'
              JOIN cq_app.ingest_native_representations n ON n.id=r.representation_id
                AND n.source_row_id=r.source_row_id AND n.security_domain=r.security_domain
              WHERE v.id=r.work_version_id AND v.work_id=r.work_id
                AND v.security_domain=r.security_domain AND v.source_sha256=s.raw_sha256
                AND s.raw_sha256=r.proof->>'raw_PDF_sha256'
                AND s.source='acl_anthology' AND s.rights_status='verified' AND s.license='CC-BY-4.0'
                AND s.native_id=r.proof->'attribution'->>'official_id'
                AND s.source_revision=r.proof->'attribution'->>'source_commit'
            )
          ))
      ), false) END
    $$;
    DO $$ BEGIN
      IF EXISTS(SELECT 1 FROM cq_app.ingest_resolutions r
                WHERE NOT cq_app.ingest_resolution_chain_valid(r)) THEN
        RAISE EXCEPTION 'Existing resolution source chain is inconsistent; migration refuses repair';
      END IF;
    END $$;
    CREATE FUNCTION cq_app.ingest_resolution_chain_guard() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF NOT cq_app.ingest_resolution_chain_valid(NEW) THEN
        RAISE EXCEPTION 'Resolution canonical source identity or trusted chain conflicts';
      END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER source_resolution_zchain BEFORE INSERT ON cq_app.ingest_resolutions
      FOR EACH ROW EXECUTE FUNCTION cq_app.ingest_resolution_chain_guard();
    """)


def downgrade() -> None:
    raise RuntimeError("Source-chain guard is a security boundary; restore an explicit verified backup instead")
