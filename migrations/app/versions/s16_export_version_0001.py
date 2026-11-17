"""Bind new export jobs to an exact completed research document version."""
from alembic import op

revision = "s16_export_version_0001"
down_revision = "s16_turn_index_0001"
branch_labels = None
depends_on = None

DDL = """
ALTER TABLE cq_app.agent_document_exports ADD COLUMN source_binding jsonb;
ALTER TABLE cq_app.agent_document_exports ADD CONSTRAINT export_source_binding_shape CHECK(
 source_binding IS NULL OR (
  jsonb_typeof(source_binding)='object' AND octet_length(source_binding::text)<=4096
  AND source_binding ?& ARRAY['source_task_id','source_command_id','command_seq','event_seq','task_revision','document_sha256','context_sha256']
  AND (source_binding-ARRAY['source_task_id','source_command_id','command_seq','event_seq','task_revision','document_sha256','context_sha256'])='{}'::jsonb
  AND source_binding->>'source_task_id'=task_id::text
  AND source_binding->>'document_sha256'=document_sha256
  AND source_binding->>'context_sha256'=context_sha256
  AND jsonb_typeof(source_binding->'command_seq')='number' AND source_binding->>'command_seq' ~ '^[1-9][0-9]*$'
  AND jsonb_typeof(source_binding->'event_seq')='number' AND source_binding->>'event_seq' ~ '^[1-9][0-9]*$'
  AND jsonb_typeof(source_binding->'task_revision')='number' AND source_binding->>'task_revision' ~ '^[1-9][0-9]*$'
  AND source_binding->>'source_command_id' ~ '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
  ) IS TRUE);
CREATE FUNCTION cq_app.agent_export_source_binding_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='UPDATE' AND NEW.source_binding IS DISTINCT FROM OLD.source_binding THEN
  RAISE EXCEPTION 'agent_export_source_binding_is_immutable';
 END IF;
 IF NEW.source_binding IS NOT NULL AND NOT EXISTS(
  SELECT 1 FROM cq_app.agent_commands c JOIN cq_app.agent_events e ON e.task_id=c.task_id
  WHERE c.task_id=NEW.task_id AND c.owner_id=NEW.owner_id AND c.id::text=NEW.source_binding->>'source_command_id'
    AND c.seq::text=NEW.source_binding->>'command_seq' AND c.state='done' AND c.kind<>'cancel'
    AND e.seq::text=NEW.source_binding->>'event_seq' AND e.event_type='conversation_turn_completed'
    AND e.payload->>'command_id'=c.id::text
    AND e.payload->>'command_seq'=NEW.source_binding->>'command_seq'
    AND e.payload->>'brief_revision'=NEW.source_binding->>'task_revision'
    AND e.payload->>'state'='succeeded' AND e.payload->'answer'->>'status'='school_document_ready'
    AND e.payload->'answer'->>'publication_status'='validated_ai_review'
 ) THEN RAISE EXCEPTION 'agent_export_exact_completed_source_required'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER agent_export_source_binding BEFORE INSERT OR UPDATE ON cq_app.agent_document_exports
 FOR EACH ROW EXECUTE FUNCTION cq_app.agent_export_source_binding_guard();
"""


def upgrade() -> None:
    op.execute(DDL)


def downgrade() -> None:
    raise RuntimeError("Preserve version-bound export jobs; explicit retention migration required")
