"""Durable source receipts and a separate bounded observation publication root."""
from alembic import op

revision = "s16_observed_publication_0001"
down_revision = "s16_export_version_0001"
branch_labels = None
depends_on = None

DDL = r"""
CREATE TABLE cq_app.source_fetch_receipts(
 window_id uuid NOT NULL REFERENCES cq_app.source_update_windows(id), sequence integer NOT NULL CHECK(sequence>0),
 payload jsonb NOT NULL, payload_canonical text NOT NULL CHECK(octet_length(payload_canonical)<=2097152),
 payload_sha256 varchar(64) NOT NULL CHECK(payload_sha256 ~ '^[a-f0-9]{64}$'), PRIMARY KEY(window_id,sequence),
 CHECK(payload=payload_canonical::jsonb), CHECK(encode(sha256(convert_to(payload_canonical,'UTF8')),'hex')=payload_sha256));
CREATE TRIGGER source_receipt_immutable BEFORE UPDATE OR DELETE ON cq_app.source_fetch_receipts
 FOR EACH ROW EXECUTE FUNCTION cq_app.reject_source_fact_mutation();
CREATE TABLE cq_app.source_publication_snapshots(
 id uuid PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(),
 base_generation_id uuid NOT NULL REFERENCES cq_app.index_generations(id),
 base_snapshot_id uuid NOT NULL REFERENCES cq_app.ingest_snapshots(id), security_domain varchar(100) NOT NULL,
 catalog_name varchar(100) NOT NULL CHECK(catalog_name='standard_metadata'),
 profile_id varchar(100) NOT NULL CHECK(profile_id='cq.bounded-observed-metadata-union/1'),
 policy_sha256 varchar(64) NOT NULL, evaluation_sha256 varchar(64) NOT NULL, membership_sha256 varchar(64) NOT NULL,
 member_count integer NOT NULL CHECK(member_count BETWEEN 1 AND 100000), payload jsonb NOT NULL,
 payload_canonical text NOT NULL, seal_sha256 varchar(64) NOT NULL,
 created_txid bigint NOT NULL DEFAULT txid_current(),
 CHECK(payload=payload_canonical::jsonb), CHECK(encode(sha256(convert_to(payload_canonical,'UTF8')),'hex')=seal_sha256));
CREATE TABLE cq_app.source_publication_members(
 snapshot_id uuid NOT NULL REFERENCES cq_app.source_publication_snapshots(id), source_unit_id uuid NOT NULL,
 source_namespace varchar(100) NOT NULL, native_id varchar(512) NOT NULL, source_revision varchar(100),
 origin_generation_id uuid REFERENCES cq_app.index_generations(id), origin_seal_id uuid REFERENCES cq_app.source_update_seals(id),
 reference jsonb NOT NULL, projection_sha256 varchar(64) NOT NULL, source_pointer_sha256 varchar(64) NOT NULL,
 record_canonical text NOT NULL, PRIMARY KEY(snapshot_id,source_unit_id), UNIQUE(snapshot_id,source_namespace,native_id),
 CHECK((origin_generation_id IS NULL)<>(origin_seal_id IS NULL)));
CREATE INDEX ix_publication_member_origin ON cq_app.source_publication_members(source_unit_id,snapshot_id);
CREATE TRIGGER publication_snapshot_immutable BEFORE UPDATE OR DELETE ON cq_app.source_publication_snapshots
 FOR EACH ROW EXECUTE FUNCTION cq_app.reject_source_fact_mutation();
CREATE TRIGGER publication_member_immutable BEFORE UPDATE OR DELETE ON cq_app.source_publication_members
 FOR EACH ROW EXECUTE FUNCTION cq_app.reject_source_fact_mutation();
CREATE FUNCTION cq_app.publication_snapshot_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE g cq_app.index_generations; item jsonb;
BEGIN
 SELECT * INTO STRICT g FROM cq_app.index_generations WHERE id=NEW.base_generation_id FOR SHARE;
 IF (g.state NOT IN ('verified','active','retired') OR g.build_scope<>'full' OR g.catalog_name<>'standard_metadata'
  OR NEW.base_snapshot_id IS DISTINCT FROM g.snapshot_id OR NEW.security_domain<>g.security_domain
  OR NEW.policy_sha256<>g.policy_sha256 OR NEW.evaluation_sha256<>g.evaluation_sha256
  OR NEW.payload->>'schema_version'<>NEW.profile_id OR NEW.payload->>'snapshot_id'<>NEW.id::text
  OR NEW.payload->>'base_generation_id'<>g.id::text OR NEW.payload->>'base_source_manifest_sha256'<>g.source_manifest_sha256
  OR NEW.payload->>'membership_sha256'<>NEW.membership_sha256 OR (NEW.payload->>'member_count')::integer<>NEW.member_count
  OR NEW.payload->>'current_status_confirmed'<>'false' OR NEW.payload->>'complete_global_or_latest_sync_claimed'<>'false'
  OR jsonb_typeof(NEW.payload->'observations')<>'array' OR jsonb_array_length(NEW.payload->'observations') NOT BETWEEN 1 AND 100
 ) IS NOT FALSE THEN RAISE EXCEPTION 'bounded_publication_header_mismatch'; END IF;
 FOR item IN SELECT * FROM jsonb_array_elements(NEW.payload->'observations') LOOP
  IF NOT EXISTS(SELECT 1 FROM cq_app.source_update_seals s JOIN cq_app.source_update_windows w ON w.id=s.window_id
   WHERE s.id::text=item->>'seal_id' AND s.seal_sha256=item->>'seal_sha256' AND NOT s.publication_eligible
    AND s.source_namespace=item->>'source_namespace' AND s.scope_sha256=item->>'scope_sha256'
    AND s.security_domain=NEW.security_domain AND w.state='observed' AND w.config->>'mode'='production'
    AND w.upper_bound=(item->>'observed_watermark')::timestamptz AND w.lower_bound=(item->>'lower_bound')::timestamptz)
  THEN RAISE EXCEPTION 'publication_requires_actual_unchanged_bounded_observations'; END IF;
 END LOOP;
 RETURN NEW;
END $$;
CREATE TRIGGER publication_snapshot_guard BEFORE INSERT ON cq_app.source_publication_snapshots
 FOR EACH ROW EXECUTE FUNCTION cq_app.publication_snapshot_guard();
CREATE FUNCTION cq_app.publication_member_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE s cq_app.source_publication_snapshots;
BEGIN
 SELECT * INTO STRICT s FROM cq_app.source_publication_snapshots WHERE id=NEW.snapshot_id FOR SHARE;
 IF (s.created_txid<>txid_current() OR (NEW.record_canonical::jsonb)->>'source_unit_id'<>NEW.source_unit_id::text
  OR (NEW.record_canonical::jsonb)->>'source_namespace'<>NEW.source_namespace
  OR (NEW.record_canonical::jsonb)->>'native_id'<>NEW.native_id
  OR (NEW.record_canonical::jsonb)->'reference' IS DISTINCT FROM NEW.reference
  OR (NEW.record_canonical::jsonb)->>'projection_sha256'<>NEW.projection_sha256
  OR (NEW.record_canonical::jsonb)->>'source_pointer_sha256'<>NEW.source_pointer_sha256
  OR (NEW.record_canonical::jsonb)->>'source_revision' IS DISTINCT FROM NEW.source_revision
  OR (NEW.record_canonical::jsonb)->>'origin_generation_id' IS DISTINCT FROM NEW.origin_generation_id::text
  OR (NEW.record_canonical::jsonb)->>'origin_seal_id' IS DISTINCT FROM NEW.origin_seal_id::text
 ) IS NOT FALSE THEN RAISE EXCEPTION 'publication_member_identity_or_late_insert_rejected'; END IF;
 IF NEW.origin_generation_id IS NOT NULL THEN
  IF NOT EXISTS(SELECT 1 FROM cq_app.index_units u WHERE u.generation_id=s.base_generation_id
   AND u.projection_id=NEW.source_unit_id AND u.source_namespace=NEW.source_namespace AND u.native_id=NEW.native_id
   AND u.source_revision IS NOT DISTINCT FROM NEW.source_revision AND u.projection_sha256=NEW.projection_sha256
   AND u.source_pointer_sha256=NEW.source_pointer_sha256 AND u.projection->'source_reference'=NEW.reference)
  THEN RAISE EXCEPTION 'complete_base_member_reference_required'; END IF;
 ELSE
  IF NOT EXISTS(SELECT 1 FROM cq_app.source_update_seal_members sm JOIN cq_app.source_update_observations o ON o.id=sm.observation_id
   WHERE sm.seal_id=NEW.origin_seal_id AND o.id=NEW.source_unit_id AND o.security_domain=s.security_domain
    AND sm.projection_sha256=NEW.projection_sha256 AND o.native_id=NEW.native_id AND o.source_namespace=NEW.source_namespace
    AND o.source_revision IS NOT DISTINCT FROM NEW.source_revision
    AND NEW.reference=jsonb_build_object('kind','update_observation','update_observation_id',o.id::text))
   OR NOT (EXISTS(SELECT 1 FROM cq_app.index_units u WHERE u.generation_id=s.base_generation_id AND u.projection_id=NEW.source_unit_id)
    OR EXISTS(SELECT 1 FROM jsonb_array_elements(s.payload->'observations') x WHERE x->>'seal_id'=NEW.origin_seal_id::text))
  THEN RAISE EXCEPTION 'actual_admitted_observation_member_required'; END IF;
 END IF;
 IF EXISTS(SELECT 1 FROM cq_app.source_update_denials d WHERE d.security_domain=s.security_domain AND d.active
  AND d.source_namespace=NEW.source_namespace AND d.native_id=NEW.native_id AND d.source_revision IN ('*',NEW.source_revision))
 THEN RAISE EXCEPTION 'quarantined_source_cannot_enter_new_publication'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER publication_member_guard BEFORE INSERT ON cq_app.source_publication_members
 FOR EACH ROW EXECUTE FUNCTION cq_app.publication_member_guard();
CREATE FUNCTION cq_app.publication_membership_complete() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE identifier uuid; expected cq_app.source_publication_snapshots; actual_count bigint; actual_sha text;
BEGIN
 identifier:=CASE WHEN TG_TABLE_NAME='source_publication_snapshots' THEN NEW.id ELSE NEW.snapshot_id END;
 SELECT * INTO STRICT expected FROM cq_app.source_publication_snapshots WHERE id=identifier;
 SELECT count(*),encode(sha256(convert_to('['||COALESCE(string_agg(record_canonical,',' ORDER BY source_unit_id::text),'')||']','UTF8')),'hex')
  INTO actual_count,actual_sha FROM cq_app.source_publication_members WHERE snapshot_id=identifier;
 IF expected.member_count<>actual_count OR expected.membership_sha256<>actual_sha THEN RAISE EXCEPTION 'complete_publication_membership_required'; END IF;
 RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER publication_complete_header AFTER INSERT ON cq_app.source_publication_snapshots
 DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION cq_app.publication_membership_complete();
ALTER TABLE cq_app.index_generations ADD COLUMN publication_snapshot_id uuid REFERENCES cq_app.source_publication_snapshots(id);
ALTER TABLE cq_app.index_generations DROP CONSTRAINT ck_index_generations_exact_typed_index_source_root;
ALTER TABLE cq_app.index_generations ADD CONSTRAINT ck_index_generations_exact_typed_index_source_root CHECK(
 (source_root_kind IN ('metadata_snapshot','native_snapshot') AND snapshot_id IS NOT NULL AND update_seal_id IS NULL AND publication_snapshot_id IS NULL)
 OR (source_root_kind='update_seal' AND snapshot_id IS NULL AND update_seal_id IS NOT NULL AND publication_snapshot_id IS NULL)
 OR (source_root_kind='publication_snapshot' AND snapshot_id IS NOT NULL AND update_seal_id IS NULL AND publication_snapshot_id IS NOT NULL));
ALTER TABLE cq_app.index_units DROP CONSTRAINT uq_index_units_generation_id_native_id;
ALTER TABLE cq_app.index_units ADD CONSTRAINT uq_index_units_generation_id_source_namespace_native_id UNIQUE(generation_id,source_namespace,native_id);
GRANT SELECT,INSERT ON cq_app.source_fetch_receipts,cq_app.source_publication_snapshots,cq_app.source_publication_members TO cq_s03_app;
REVOKE UPDATE,DELETE,TRUNCATE ON cq_app.source_fetch_receipts,cq_app.source_publication_snapshots,cq_app.source_publication_members FROM cq_s03_app;
"""

# Existing-generation guards below retain every legacy branch and state/proof
# immutability check; only the separately identified publication root is added.
GUARDS = r"""

CREATE OR REPLACE FUNCTION cq_app.index_generation_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE s cq_app.ingest_snapshots%ROWTYPE; u cq_app.source_update_seals%ROWTYPE; p cq_app.source_publication_snapshots%ROWTYPE;
BEGIN
 IF TG_OP='DELETE' THEN RAISE EXCEPTION 'index generation history is append only'; END IF;
 IF NEW.source_root_kind='publication_snapshot' THEN
  SELECT * INTO p FROM cq_app.source_publication_snapshots WHERE id=NEW.publication_snapshot_id;
  IF p.id IS NULL OR p.base_snapshot_id IS DISTINCT FROM NEW.snapshot_id OR p.security_domain<>NEW.security_domain
   OR p.profile_id<>NEW.representation_profile OR p.seal_sha256<>NEW.source_manifest_sha256
   OR p.catalog_name<>NEW.catalog_name OR NEW.source_namespace<>'bounded_metadata_union'
   OR p.policy_sha256<>NEW.policy_sha256 OR p.evaluation_sha256<>NEW.evaluation_sha256
  THEN RAISE EXCEPTION 'immutable_bounded_publication_root_required'; END IF;
 ELSIF NEW.source_root_kind IN ('metadata_snapshot','native_snapshot') THEN
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


CREATE OR REPLACE FUNCTION cq_app.index_unit_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE g cq_app.index_generations%ROWTYPE; p cq_app.ingest_projections%ROWTYPE; r cq_app.ingest_rows%ROWTYPE; n cq_app.ingest_native_representations%ROWTYPE; o cq_app.source_update_observations%ROWTYPE; s cq_app.ingest_snapshots%ROWTYPE; pm cq_app.source_publication_members%ROWTYPE;
BEGIN
 SELECT * INTO g FROM cq_app.index_generations WHERE id=COALESCE(NEW.generation_id,OLD.generation_id) FOR SHARE;
 IF NOT FOUND OR g.state<>'preparing' THEN RAISE EXCEPTION 'index source units immutable after preparation'; END IF;
 IF TG_OP='DELETE' THEN RETURN OLD; END IF;
 IF TG_OP='UPDATE' THEN RAISE EXCEPTION 'staged index source unit is immutable'; END IF;
 IF g.source_root_kind='publication_snapshot' THEN
  SELECT * INTO pm FROM cq_app.source_publication_members WHERE snapshot_id=g.publication_snapshot_id AND source_unit_id=NEW.projection_id;
  IF (pm.source_unit_id IS NULL OR pm.source_namespace<>NEW.source_namespace OR pm.native_id<>NEW.native_id
   OR pm.source_revision IS DISTINCT FROM NEW.source_revision OR pm.projection_sha256<>NEW.projection_sha256
   OR pm.source_pointer_sha256<>NEW.source_pointer_sha256 OR pm.reference->>'kind'<>NEW.source_reference_kind
   OR NOT ((NEW.source_reference_kind='metadata_projection' AND NEW.metadata_projection_id=NEW.projection_id
     AND pm.reference->>'metadata_projection_id'=NEW.metadata_projection_id::text)
    OR (NEW.source_reference_kind='update_observation' AND NEW.update_observation_id=NEW.projection_id
     AND pm.reference->>'update_observation_id'=NEW.update_observation_id::text))
  ) IS NOT FALSE THEN RAISE EXCEPTION 'bounded_publication_unit_membership_mismatch'; END IF;
  IF NOT NEW.source_denied THEN
   IF pm.origin_generation_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM cq_app.index_units original
      WHERE original.generation_id=pm.origin_generation_id AND original.projection_id=NEW.projection_id
       AND original.work_id IS NOT DISTINCT FROM NEW.work_id AND original.work_version_id IS NOT DISTINCT FROM NEW.work_version_id)
    OR pm.origin_seal_id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM cq_app.source_update_observations original
      WHERE original.id=NEW.update_observation_id AND original.work_id IS NOT DISTINCT FROM NEW.work_id AND NEW.work_version_id IS NULL)
   THEN RAISE EXCEPTION 'bounded_publication_work_lineage_mismatch'; END IF;
  END IF;
  RETURN NEW;
 ELSIF NEW.source_reference_kind='metadata_projection' AND g.source_root_kind='metadata_snapshot' THEN
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

"""
READY = r"""
CREATE OR REPLACE FUNCTION cq_app.index_full_source_publication_ready(generation uuid) RETURNS boolean
LANGUAGE sql STABLE AS $$
 SELECT COALESCE(g.build_scope='full' AND g.state IN ('verified','active','retired')
  AND g.server_proof IS NOT NULL AND g.server_proof->>'profile_sha256'=g.index_profile_sha256
  AND g.readback_proof->>'complete'='true' AND (g.readback_proof->>'matched')::bigint=g.chunk_count
  AND g.readback_proof->>'rows_sha256'=g.expected_manifest_sha256
  AND (SELECT count(*) FROM cq_app.index_units u WHERE u.generation_id=g.id)=g.unit_count
  AND (SELECT count(*) FROM cq_app.index_chunks c WHERE c.generation_id=g.id)=g.chunk_count
  AND NOT EXISTS(SELECT 1 FROM cq_app.index_batches b WHERE b.generation_id=g.id AND b.state<>'verified')
  AND (
   (g.source_root_kind IN ('metadata_snapshot','native_snapshot') AND s.id=g.snapshot_id AND s.security_domain=g.security_domain
    AND s.status='staged' AND s.profile_adopted AND s.seal_sha256=g.source_seal_sha256
    AND ((g.source_root_kind='metadata_snapshot' AND s.purpose='standard'
      AND s.representation_profile='litsearch-clean-exact/1' AND g.unit_count=64183)
     OR (g.source_root_kind='native_snapshot' AND s.purpose='application'
      AND s.representation_profile='litsearch-s2orc-native-text/1' AND g.unit_count BETWEEN 500 AND 2000
      AND (SELECT count(*) FROM cq_app.application_corpus_members a WHERE a.snapshot_id=s.id)=g.unit_count
      AND NOT EXISTS(SELECT 1 FROM cq_app.index_units u LEFT JOIN cq_app.application_corpus_members a
       ON a.snapshot_id=s.id AND a.representation_id=u.native_representation_id WHERE u.generation_id=g.id AND a.representation_id IS NULL))))
   OR (g.source_root_kind='publication_snapshot' AND p.id=g.publication_snapshot_id
    AND p.base_snapshot_id=g.snapshot_id AND p.security_domain=g.security_domain AND p.catalog_name=g.catalog_name
    AND p.profile_id=g.representation_profile AND p.seal_sha256=g.source_seal_sha256
    AND p.policy_sha256=g.policy_sha256 AND p.evaluation_sha256=g.evaluation_sha256 AND p.member_count=g.unit_count
    AND NOT EXISTS(SELECT 1 FROM cq_app.source_publication_members member LEFT JOIN cq_app.index_units u
     ON u.generation_id=g.id AND u.projection_id=member.source_unit_id
     WHERE member.snapshot_id=p.id AND (u.projection_id IS NULL OR u.projection_sha256<>member.projection_sha256
      OR u.source_pointer_sha256<>member.source_pointer_sha256 OR u.source_namespace<>member.source_namespace OR u.native_id<>member.native_id)))
  ),false)
 FROM cq_app.index_generations g LEFT JOIN cq_app.ingest_snapshots s ON s.id=g.snapshot_id
 LEFT JOIN cq_app.source_publication_snapshots p ON p.id=g.publication_snapshot_id WHERE g.id=generation
$$;
"""


def upgrade() -> None:
    op.execute(DDL)
    op.execute(GUARDS)
    op.execute(READY)


def downgrade() -> None:
    raise RuntimeError("publication_history_requires_explicit_restore_not_lossy_downgrade")
