"""ADR0035 immutable Task catalog pins; explicit policy, no history backfill."""
from alembic import op

revision = 's15_snapshot_0001'
down_revision = 's15_blob_0001'
branch_labels = None
depends_on = None

DDL = r'''
CREATE TABLE cq_app.agent_catalog_pins(
 task_id uuid NOT NULL REFERENCES cq_app.agent_tasks(id) ON DELETE RESTRICT,
 catalog varchar(30) NOT NULL CHECK(catalog IN ('application_papers','standard_metadata')),
 owner_id uuid NOT NULL REFERENCES cq_app.owners(id),
 generation_id uuid NOT NULL REFERENCES cq_app.index_generations(id) ON DELETE RESTRICT,
 epoch bigint NOT NULL CHECK(epoch>=1), binding jsonb NOT NULL CHECK(jsonb_typeof(binding)='object'),
 binding_canonical text NOT NULL CHECK(octet_length(binding_canonical) BETWEEN 1 AND 16384),
 binding_sha256 varchar(64) NOT NULL CHECK(binding_sha256 ~ '^[a-f0-9]{64}$'),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), PRIMARY KEY(task_id,catalog)
);
CREATE INDEX ix_agent_catalog_pins_generation ON cq_app.agent_catalog_pins(generation_id,task_id);
CREATE FUNCTION cq_app.agent_catalog_pin_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE task cq_app.agent_tasks; generation cq_app.index_generations;
BEGIN
 IF TG_OP<>'INSERT' THEN RAISE EXCEPTION 'task_catalog_pin_immutable'; END IF;
 SELECT * INTO STRICT task FROM cq_app.agent_tasks WHERE id=NEW.task_id;
 IF task.owner_id<>NEW.owner_id OR task.profile->>'catalog_scope_policy' IS DISTINCT FROM 'task-catalog-snapshot/1'
  OR right(task.graph_version,length('+task-catalog-snapshot1'))<>'+task-catalog-snapshot1'
 THEN RAISE EXCEPTION 'task_catalog_task_identity_conflict'; END IF;
 SELECT * INTO STRICT generation FROM cq_app.index_generations WHERE id=NEW.generation_id;
 IF generation.catalog_name<>NEW.catalog OR generation.security_domain<>'public'
  OR NEW.binding_canonical::jsonb IS DISTINCT FROM NEW.binding
  OR encode(pg_catalog.sha256(convert_to(NEW.binding_canonical,'UTF8')),'hex')<>NEW.binding_sha256
  OR NEW.binding->>'version' IS DISTINCT FROM 'task-catalog-snapshot/1'
  OR NEW.binding->>'catalog' IS DISTINCT FROM NEW.catalog
  OR NEW.binding->>'generation_id' IS DISTINCT FROM NEW.generation_id::text
  OR (NEW.binding->>'epoch')::bigint IS DISTINCT FROM NEW.epoch
  OR NEW.binding->>'collection_name' IS DISTINCT FROM generation.collection_name
  OR NEW.binding->>'source_snapshot_id' IS DISTINCT FROM generation.snapshot_id::text
  OR NEW.binding->>'manifest_sha256' IS DISTINCT FROM generation.expected_manifest_sha256
  OR NEW.binding->>'source_seal_sha256' IS DISTINCT FROM generation.source_seal_sha256
  OR NEW.binding->>'index_profile_sha256' IS DISTINCT FROM generation.index_profile_sha256
  OR NEW.binding->>'tokenizer_profile_sha256' IS DISTINCT FROM generation.tokenizer_profile_sha256
  OR NEW.binding->>'policy_sha256' IS DISTINCT FROM generation.policy_sha256
  OR NEW.binding->>'evaluation_sha256' IS DISTINCT FROM generation.evaluation_sha256
  OR (NEW.binding->>'native_runtime_sha256' ~ '^[a-f0-9]{64}$') IS NOT TRUE
  OR (NEW.binding->>'server_proof_sha256' ~ '^[a-f0-9]{64}$') IS NOT TRUE
 THEN RAISE EXCEPTION 'task_catalog_pin_identity_or_binding_conflict'; END IF;
 IF NOT EXISTS(SELECT 1 FROM cq_app.index_publish_intents p WHERE p.state='committed'
  AND p.target_generation_id=NEW.generation_id AND p.catalog_name=NEW.catalog
  AND p.security_domain=generation.security_domain AND p.expected_epoch+1=NEW.epoch)
 THEN RAISE EXCEPTION 'task_catalog_epoch_not_published'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER agent_catalog_pin_immutable BEFORE INSERT OR UPDATE OR DELETE ON cq_app.agent_catalog_pins
 FOR EACH ROW EXECUTE FUNCTION cq_app.agent_catalog_pin_guard();
DO $$ BEGIN IF EXISTS(SELECT 1 FROM pg_roles WHERE rolname='cq_s03_app') THEN
 GRANT SELECT,INSERT ON cq_app.agent_catalog_pins TO cq_s03_app;
END IF; END $$;
'''


def upgrade() -> None:
    op.execute(DDL)


def downgrade() -> None:
    raise RuntimeError('Preserve Task scope/GC references and original history; no destructive downgrade')
