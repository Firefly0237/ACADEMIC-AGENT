"""Private-material native index generations; no product activation or fake papers."""
from alembic import op

revision = "s09_material_0001"
down_revision = "s08_agent_0001"
branch_labels = None
depends_on = None

DDL = """
CREATE TABLE cq_app.material_index_generations(
 id uuid PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(), owner_id uuid NOT NULL, file_id uuid NOT NULL,
 representation_id varchar(64) NOT NULL, material_version_id uuid NOT NULL,
 file_revision integer NOT NULL CHECK(file_revision>=1), source_file_sha256 varchar(64) NOT NULL,
 backend_profile_sha256 varchar(64) NOT NULL, schema_profile_sha256 varchar(64) NOT NULL,
 tokenizer_profile_sha256 varchar(64) NOT NULL, collection_name varchar(128) NOT NULL,
 manifest_sha256 varchar(64) NOT NULL, expected_rows integer NOT NULL CHECK(expected_rows BETWEEN 1 AND 5000),
 state varchar(30) NOT NULL CHECK(state IN ('ready','verified','adopted','superseded','cancelled','failed')), readback_proof jsonb,
 UNIQUE(id,owner_id,file_id), UNIQUE(id,owner_id,file_id,representation_id),
 UNIQUE(owner_id,file_id,representation_id,file_revision,backend_profile_sha256,schema_profile_sha256,tokenizer_profile_sha256,collection_name),
 FOREIGN KEY(owner_id,file_id) REFERENCES cq_app.files(owner_id,id),
 FOREIGN KEY(owner_id,representation_id) REFERENCES cq_app.parsed_representations(owner_id,id),
 FOREIGN KEY(owner_id,material_version_id) REFERENCES cq_app.material_versions(owner_id,id)
);
CREATE INDEX ix_material_index_generation_file ON cq_app.material_index_generations(owner_id,file_id,state);
CREATE TABLE cq_app.material_index_rows(
 generation_id uuid NOT NULL, row_id varchar(64) NOT NULL, owner_id uuid NOT NULL, file_id uuid NOT NULL,
 representation_id varchar(64) NOT NULL, fragment_id varchar(64) NOT NULL, ordinal integer NOT NULL CHECK(ordinal>=1),
 row_sha256 varchar(64) NOT NULL, payload jsonb NOT NULL,
 state varchar(20) NOT NULL CHECK(state IN ('ready','sending','unknown','verified','deleted')),
 PRIMARY KEY(generation_id,row_id), UNIQUE(generation_id,fragment_id),
 FOREIGN KEY(generation_id,owner_id,file_id,representation_id) REFERENCES cq_app.material_index_generations(id,owner_id,file_id,representation_id),
 FOREIGN KEY(owner_id,fragment_id) REFERENCES cq_app.material_fragments(owner_id,id)
);
CREATE INDEX ix_material_index_row_delivery ON cq_app.material_index_rows(generation_id,state,ordinal);
CREATE TABLE cq_app.material_index_jobs(
 id uuid PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(),
 generation_id uuid NOT NULL REFERENCES cq_app.material_index_generations(id), operation varchar(20) NOT NULL CHECK(operation IN ('upsert','delete')),
 state varchar(20) NOT NULL CHECK(state IN ('ready','sending','unknown','verified','failed','cancelled')),
 epoch integer NOT NULL CHECK(epoch>=0), worker_id varchar(100), lease_id uuid, lease_until timestamptz,
 batch_ids jsonb NOT NULL, attempts jsonb NOT NULL, error_code varchar(100), UNIQUE(generation_id,operation)
);
CREATE INDEX ix_material_index_job_queue ON cq_app.material_index_jobs(state,created_at,id);
CREATE TABLE cq_app.material_index_attempts(
 id uuid PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(), job_id uuid NOT NULL REFERENCES cq_app.material_index_jobs(id),
 epoch integer NOT NULL CHECK(epoch>=0), operation varchar(20) NOT NULL CHECK(operation IN ('upsert','delete','reconcile')),
 request_sha256 varchar(64) NOT NULL, batch_ids jsonb NOT NULL,
 write_outcome varchar(20) NOT NULL CHECK(write_outcome IN ('sending','acknowledged','unknown','read_only')), receipt jsonb
);
CREATE INDEX ix_material_index_attempt_job_outcome ON cq_app.material_index_attempts(job_id,write_outcome,created_at);
CREATE TABLE cq_app.material_index_pointers(
 owner_id uuid NOT NULL, file_id uuid NOT NULL, generation_id uuid, epoch integer NOT NULL CHECK(epoch>=0),
 PRIMARY KEY(owner_id,file_id), FOREIGN KEY(owner_id,file_id) REFERENCES cq_app.files(owner_id,id),
 FOREIGN KEY(generation_id,owner_id,file_id) REFERENCES cq_app.material_index_generations(id,owner_id,file_id)
);
CREATE FUNCTION cq_app.material_index_row_identity_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF (to_jsonb(NEW)-'state') IS DISTINCT FROM (to_jsonb(OLD)-'state') THEN
   RAISE EXCEPTION 'material_index_row_identity_immutable';
 END IF; RETURN NEW;
END $$;
CREATE TRIGGER material_index_row_identity BEFORE UPDATE ON cq_app.material_index_rows
 FOR EACH ROW EXECUTE FUNCTION cq_app.material_index_row_identity_guard();
CREATE FUNCTION cq_app.material_index_generation_identity_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF (to_jsonb(NEW)-'state'-'readback_proof') IS DISTINCT FROM (to_jsonb(OLD)-'state'-'readback_proof') THEN
   RAISE EXCEPTION 'material_index_generation_identity_immutable';
 END IF; RETURN NEW;
END $$;
CREATE TRIGGER material_index_generation_identity BEFORE UPDATE ON cq_app.material_index_generations
 FOR EACH ROW EXECUTE FUNCTION cq_app.material_index_generation_identity_guard();
CREATE FUNCTION cq_app.material_index_attempt_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF (to_jsonb(NEW)-'write_outcome'-'receipt') IS DISTINCT FROM (to_jsonb(OLD)-'write_outcome'-'receipt')
   OR OLD.write_outcome<>'sending' THEN RAISE EXCEPTION 'material_attempt_identity_or_settled_history_immutable'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER material_index_attempt_history BEFORE UPDATE ON cq_app.material_index_attempts
 FOR EACH ROW EXECUTE FUNCTION cq_app.material_index_attempt_guard();
CREATE FUNCTION cq_app.material_index_pointer_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE g cq_app.material_index_generations; f cq_app.files; ingestion cq_app.material_ingestions;
BEGIN
 IF TG_OP='UPDATE' AND NEW.generation_id IS DISTINCT FROM OLD.generation_id AND NEW.epoch<>OLD.epoch+1 THEN
   RAISE EXCEPTION 'material_index_pointer_epoch_must_advance';
 END IF;
 IF NEW.generation_id IS NULL THEN RETURN NEW; END IF;
 SELECT * INTO STRICT g FROM cq_app.material_index_generations WHERE id=NEW.generation_id AND owner_id=NEW.owner_id AND file_id=NEW.file_id;
 SELECT * INTO STRICT f FROM cq_app.files WHERE id=NEW.file_id AND owner_id=NEW.owner_id;
 SELECT * INTO STRICT ingestion FROM cq_app.material_ingestions WHERE file_id=NEW.file_id AND owner_id=NEW.owner_id;
 IF g.state NOT IN ('verified','adopted') OR g.readback_proof->>'complete' IS DISTINCT FROM 'true'
   OR (g.readback_proof->>'matched')::integer IS DISTINCT FROM g.expected_rows
   OR g.readback_proof->>'manifest_sha256' IS DISTINCT FROM g.manifest_sha256
   OR f.deleted_at IS NOT NULL OR f.status NOT IN ('ready','partial') OR f.rights_status='restricted'
   OR f.outbound_policy<>'selected_cloud_processing_allowed' OR f.sha256<>g.source_file_sha256 OR f.revision<>g.file_revision
   OR ingestion.representation_id IS DISTINCT FROM g.representation_id
   OR (SELECT count(*) FROM cq_app.material_index_rows r WHERE r.generation_id=g.id AND r.state='verified')<>g.expected_rows
   OR NOT EXISTS(SELECT 1 FROM cq_app.index_backend_adoptions a WHERE a.backend_profile_sha256=g.backend_profile_sha256
      AND a.metrics->>'status'='pass' AND a.metrics->>'attempt_errors'='0')
 THEN RAISE EXCEPTION 'complete_current_material_and_adopted_backend_required'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER material_index_pointer_current_source BEFORE INSERT OR UPDATE ON cq_app.material_index_pointers
 FOR EACH ROW EXECUTE FUNCTION cq_app.material_index_pointer_guard();
"""


def upgrade():
    op.execute(DDL)


def downgrade():
    raise RuntimeError("Private source/index delivery history requires an approved backup or forward migration")
