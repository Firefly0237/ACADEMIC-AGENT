"""Indexed immutable application membership and compact CAS-backed adoption seal.

Revision ID: s05_data_0003
Revises: s05_index_0002
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "s05_data_0003"
down_revision = "s05_index_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    uuid_columns = ("snapshot_id", "source_row_id", "resolution_id", "work_id", "work_version_id", "source_record_id")
    sha_columns = (
        "resolution_proof_sha256",
        "representation_id",
        "draft_sha256",
        "source_text_sha256",
        "source_pointer_sha256",
        "source_record_sha256",
        "raw_PDF_sha256",
        "member_sha256",
        "record_sha256",
    )
    op.create_table(
        "application_corpus_members",
        *(sa.Column(name, sa.Uuid(), nullable=False) for name in uuid_columns),
        *(sa.Column(name, sa.String(64), nullable=False) for name in sha_columns),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("native_id", sa.String(512), nullable=False),
        sa.Column("resolution_revision", sa.Integer(), nullable=False),
        sa.Column("source_reference", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id", "source_row_id"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["cq_app.ingest_snapshots.id"]),
        sa.ForeignKeyConstraint(["source_row_id"], ["cq_app.ingest_rows.id"]),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "native_id"], ["cq_app.ingest_rows.snapshot_id", "cq_app.ingest_rows.native_id"]
        ),
        sa.ForeignKeyConstraint(["resolution_id"], ["cq_app.ingest_resolutions.id"]),
        sa.ForeignKeyConstraint(["work_id"], ["cq_app.works.id"]),
        sa.ForeignKeyConstraint(["work_version_id"], ["cq_app.work_versions.id"]),
        sa.ForeignKeyConstraint(["source_record_id"], ["cq_app.source_records.id"]),
        sa.ForeignKeyConstraint(["representation_id"], ["cq_app.ingest_native_representations.id"]),
        *(
            sa.UniqueConstraint("snapshot_id", name)
            for name in (
                "rank",
                "native_id",
                "representation_id",
                "work_id",
                "work_version_id",
                "source_record_id",
                "raw_PDF_sha256",
            )
        ),
        sa.CheckConstraint(
            "rank >= 0 AND rank < 3000 AND resolution_revision >= 1", name="valid_fixed_application_position"
        ),
        schema="cq_app",
    )
    op.create_index(
        "ix_application_member_version",
        "application_corpus_members",
        ["work_version_id", "snapshot_id"],
        schema="cq_app",
    )
    op.execute(r"""
CREATE FUNCTION cq_app.application_member_record_sha(m cq_app.application_corpus_members)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
 SELECT encode(sha256(convert_to(concat_ws(E'\n',m.snapshot_id::text,m.source_row_id::text,m.rank::text,m.native_id,
   m.resolution_id::text,m.resolution_revision::text,m.resolution_proof_sha256,m.representation_id,m.work_id::text,
   m.work_version_id::text,m.source_record_id::text,m.draft_sha256,m.source_text_sha256,m.source_pointer_sha256,
   m.source_record_sha256,m."raw_PDF_sha256",m.member_sha256),'UTF8')),'hex')
$$;
CREATE FUNCTION cq_app.application_member_insert_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE snapshot cq_app.ingest_snapshots%ROWTYPE;
BEGIN
 SELECT * INTO snapshot FROM cq_app.ingest_snapshots WHERE id=NEW.snapshot_id FOR SHARE;
 IF snapshot.id IS NULL OR snapshot.status<>'landing' OR snapshot.profile_adopted OR
    snapshot.id<>'e7cabf2f-bf73-5cb9-8846-c1afb3b3ed35'::uuid OR
    snapshot.purpose<>'application' OR snapshot.source_namespace<>'litsearch_s2orc' OR
    snapshot.security_domain<>'public' THEN
   RAISE EXCEPTION 'Application membership preparation is outside the exact unadopted source';
 END IF;
 IF NEW.record_sha256<>cq_app.application_member_record_sha(NEW) OR NOT EXISTS(
  SELECT 1 FROM cq_app.ingest_rows sr
  JOIN cq_app.ingest_resolutions r ON r.source_row_id=sr.id AND r.id=NEW.resolution_id
  JOIN cq_app.ingest_native_representations nr ON nr.id=r.representation_id AND nr.source_row_id=sr.id
  JOIN cq_app.works w ON w.id=r.work_id
  JOIN cq_app.work_versions v ON v.id=r.work_version_id AND v.work_id=w.id
  JOIN cq_app.source_records source ON source.id=NEW.source_record_id AND source.work_id=w.id
  WHERE sr.id=NEW.source_row_id AND sr.snapshot_id=NEW.snapshot_id AND sr.native_id=NEW.native_id
   AND sr.security_domain=snapshot.security_domain AND sr.source_pointer_sha256=NEW.source_pointer_sha256
   AND r.state='paper_identity_verified' AND r.revision=NEW.resolution_revision
   AND r.work_id=NEW.work_id AND r.work_version_id=NEW.work_version_id
   AND r.representation_id=NEW.representation_id AND r.proof_sha256=NEW.resolution_proof_sha256
   AND r.proof->'source_reference'=NEW.source_reference AND r.proof->>'source_record_id'=source.id::text
   AND nr.draft_sha256=NEW.draft_sha256 AND nr.source_text_sha256=NEW.source_text_sha256
   AND source.raw_sha256=NEW."raw_PDF_sha256" AND v.source_sha256=source.raw_sha256
   AND source.rights_status='verified' AND source.license='CC-BY-4.0' AND w.public
   AND w.canonical_redirect_work_id IS NULL AND cq_app.ingest_resolution_chain_valid(r)
   AND NOT EXISTS(SELECT 1 FROM cq_app.ingest_resolutions later WHERE later.source_row_id=sr.id AND later.revision>r.revision)
 ) THEN RAISE EXCEPTION 'Application member does not match its exact source, rights and current resolution'; END IF;
 RETURN NEW;
END $$;
CREATE FUNCTION cq_app.application_membership_insert_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE snapshot_id uuid;
BEGIN
 IF TG_TABLE_NAME IN ('ingest_rows','ingest_artifacts') THEN snapshot_id:=NEW.snapshot_id;
 ELSE SELECT r.snapshot_id INTO snapshot_id FROM cq_app.ingest_rows r WHERE r.id=NEW.source_row_id;
 END IF;
 PERFORM 1 FROM cq_app.ingest_snapshots s WHERE s.id=snapshot_id FOR SHARE;
 IF EXISTS(SELECT 1 FROM cq_app.ingest_snapshots s WHERE s.id=snapshot_id
           AND s.purpose='application' AND s.profile_adopted) THEN
  RAISE EXCEPTION 'Adopted application source membership and resolution are immutable';
 END IF;
 RETURN NEW;
END $$;
CREATE FUNCTION cq_app.application_adoption_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE member_count integer; records_sha text;
BEGIN
 IF NEW.purpose<>'application' OR NOT NEW.profile_adopted OR OLD.profile_adopted THEN RETURN NEW; END IF;
 IF (NEW.id<>'e7cabf2f-bf73-5cb9-8846-c1afb3b3ed35'::uuid OR
    NEW.source_namespace<>'litsearch_s2orc' OR NEW.security_domain<>'public' OR
    NEW.policy_sha256<>'79d74a66007e3967a080f36703d470204ec5c5c875478bb9ba0f60221d8af17a' OR
    NEW.seal->>'schema_version'<>'cq.application-source-seal/2' OR
    NEW.seal->>'membership_table'<>'cq_app.application_corpus_members' OR
    NEW.seal->>'original_policy_sha256'<>NEW.policy_sha256 OR NEW.seal->>'evaluation_sha256'<>NEW.evaluation_sha256 OR
    NEW.seal->>'source_manifest_sha256'<>NEW.source_manifest_sha256 OR NEW.seal->>'snapshot_id'<>NEW.id::text OR
    NEW.seal->>'stop_rule_adoption_sha256'<>'20f0b743e94e538dad69152e7812c3a7f61270736b3f29d08522c90512b7db24' OR
    NEW.seal->>'source_chain_manifest_sha256'<>'fd09876d6863e51f25f822b3bb03bf41ee3756314a22b952edfb3c5d3b309813' OR
    NEW.seal->>'rights_manifest_sha256'<>'e43140d1f1fc118d7692ae562cde983f3399ba05309cd4365a6a3b95647766d3' OR
    NEW.seal->>'all_original_positions_preserved'<>'true' OR NEW.seal->>'all_source_candidates_assessed'<>'true' OR
    NEW.seal->>'eligible_to_freeze'<>'true' OR NEW.seal->>'QA_granted'<>'false' OR
    NEW.seal->>'S05_or_S09_gate_passed'<>'false' OR NEW.seal->'coverage'->>'original_positions'<>'3000' OR
    NEW.seal->'coverage'->>'component_candidates_assessed'<>'692' OR
    NEW.seal->'audit_blob'->>'sha256'<>NEW.seal->>'qualification_sha256' OR
    NEW.seal->'audit_blob'->>'security_domain'<>NEW.security_domain) IS DISTINCT FROM false THEN
  RAISE EXCEPTION 'Application adoption lacks fixed policy, stop rule and complete qualification';
 END IF;
 SELECT count(*), encode(sha256(convert_to(string_agg(m.record_sha256,E'\n' ORDER BY m.rank)||E'\n','UTF8')),'hex')
   INTO member_count, records_sha FROM cq_app.application_corpus_members m WHERE m.snapshot_id=NEW.id;
 IF (member_count NOT BETWEEN 500 AND 2000 OR (NEW.seal->'coverage'->>'qualified')::integer<>member_count OR
     NEW.seal->>'membership_records_sha256'<>records_sha) IS DISTINCT FROM false THEN
  RAISE EXCEPTION 'Application adoption count or indexed membership digest is invalid';
 END IF;
 IF NOT EXISTS(SELECT 1 FROM cq_app.blobs b WHERE b.security_domain=NEW.security_domain
    AND b.sha256=NEW.seal->'audit_blob'->>'sha256' AND b.size_bytes=(NEW.seal->'audit_blob'->>'size_bytes')::bigint
    AND b.size_bytes>0 AND b.size_bytes<=33554432) THEN
  RAISE EXCEPTION 'Application adoption lacks registered immutable qualification CAS evidence';
 END IF;
 IF EXISTS(SELECT 1 FROM cq_app.application_corpus_members member
   JOIN cq_app.ingest_resolutions r ON r.id=member.resolution_id
   JOIN cq_app.ingest_rows sr ON sr.id=member.source_row_id
   JOIN cq_app.works w ON w.id=member.work_id
   WHERE member.snapshot_id=NEW.id AND (NOT w.public OR w.canonical_redirect_work_id IS NOT NULL OR
    NOT cq_app.ingest_resolution_chain_valid(r) OR EXISTS(SELECT 1 FROM cq_app.ingest_resolutions later
      WHERE later.source_row_id=sr.id AND later.revision>r.revision) OR
    EXISTS(SELECT 1 FROM cq_app.source_update_denials d WHERE d.security_domain=sr.security_domain
      AND d.source_namespace='litsearch_s2orc' AND d.native_id=sr.native_id AND d.active
      AND d.source_revision IN ('*',NEW.source_revision)))) THEN
  RAISE EXCEPTION 'Application current identity or source denial changed before adoption';
 END IF;
 RETURN NEW;
END $$;
CREATE FUNCTION cq_app.application_version_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF EXISTS(SELECT 1 FROM cq_app.application_corpus_members member JOIN cq_app.ingest_snapshots snapshot
    ON snapshot.id=member.snapshot_id WHERE member.work_version_id=OLD.id AND snapshot.profile_adopted) THEN
   IF TG_OP='DELETE' OR NEW IS DISTINCT FROM OLD THEN RAISE EXCEPTION 'A sealed application WorkVersion is immutable'; END IF;
 END IF;
 IF TG_OP='DELETE' THEN RETURN OLD; END IF;
 RETURN NEW;
END $$;
""")
    for table in ("ingest_artifacts", "ingest_rows", "ingest_resolutions", "ingest_native_representations"):
        op.execute(
            f"CREATE TRIGGER application_membership_insert BEFORE INSERT ON cq_app.{table} "
            "FOR EACH ROW EXECUTE FUNCTION cq_app.application_membership_insert_guard()"
        )
    op.execute(
        "CREATE TRIGGER application_member_insert BEFORE INSERT ON cq_app.application_corpus_members "
        "FOR EACH ROW EXECUTE FUNCTION cq_app.application_member_insert_guard()"
    )
    op.execute(
        "CREATE TRIGGER application_member_immutable BEFORE UPDATE OR DELETE ON cq_app.application_corpus_members "
        "FOR EACH ROW EXECUTE FUNCTION cq_app.ingest_immutable_row()"
    )
    op.execute(
        "CREATE TRIGGER application_adoption BEFORE UPDATE ON cq_app.ingest_snapshots "
        "FOR EACH ROW EXECUTE FUNCTION cq_app.application_adoption_guard()"
    )
    op.execute(
        "CREATE TRIGGER application_version BEFORE UPDATE OR DELETE ON cq_app.work_versions "
        "FOR EACH ROW EXECUTE FUNCTION cq_app.application_version_guard()"
    )


def downgrade() -> None:
    raise RuntimeError("Application freeze history is irreversible; restore an explicit owned backup")
