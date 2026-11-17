"""Candidate expand-only SQL, never executed on import or application startup."""

DDL = """
CREATE TABLE cq_app.agent_document_exports (
 id uuid PRIMARY KEY, owner_id uuid NOT NULL REFERENCES cq_app.owners(id),
 project_id uuid NOT NULL, task_id uuid NOT NULL REFERENCES cq_app.agent_tasks(id),
 document_sha256 varchar(64) NOT NULL CHECK(document_sha256 ~ '^[a-f0-9]{64}$'),
 context_sha256 varchar(64) NOT NULL CHECK(context_sha256 ~ '^[a-f0-9]{64}$'),
 requested_formats jsonb NOT NULL CHECK(jsonb_typeof(requested_formats)='array' AND jsonb_array_length(requested_formats) BETWEEN 1 AND 3),
 toolchain_profile varchar(120) NOT NULL,
 toolchain_sha256 varchar(64) NOT NULL CHECK(toolchain_sha256 ~ '^[a-f0-9]{64}$'), font_profile varchar(120) NOT NULL,
 state varchar(16) NOT NULL DEFAULT 'queued' CHECK(state IN ('queued','running','succeeded','failed','cancelled')),
 revision integer NOT NULL DEFAULT 1 CHECK(revision>=1),
 epoch bigint NOT NULL DEFAULT 0 CHECK(epoch>=0), worker_id varchar(100),
 lease_until timestamptz, deadline_at timestamptz, cancellation_requested boolean NOT NULL DEFAULT false,
 artifacts jsonb NOT NULL DEFAULT '{}' CHECK(jsonb_typeof(artifacts)='object'),
 receipt jsonb, error_code varchar(100),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 UNIQUE(owner_id,id), FOREIGN KEY(owner_id,project_id) REFERENCES cq_app.projects(owner_id,id)
);
CREATE INDEX ix_agent_document_exports_queue ON cq_app.agent_document_exports(created_at,id) WHERE state='queued';
CREATE INDEX ix_agent_document_exports_owner_task ON cq_app.agent_document_exports(owner_id,task_id,created_at DESC,id DESC);
CREATE INDEX ix_agent_document_exports_expiry ON cq_app.agent_document_exports(lease_until) WHERE state='running';
CREATE TABLE cq_app.agent_export_operations (
 owner_id uuid NOT NULL REFERENCES cq_app.owners(id), operation_id uuid NOT NULL,
 payload_sha256 varchar(64) NOT NULL CHECK(payload_sha256 ~ '^[a-f0-9]{64}$'),
 export_id uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(owner_id,operation_id),
 FOREIGN KEY(owner_id,export_id) REFERENCES cq_app.agent_document_exports(owner_id,id)
);
CREATE TABLE cq_app.agent_export_events (
 export_id uuid NOT NULL REFERENCES cq_app.agent_document_exports(id),
 seq integer NOT NULL CHECK(seq>=1), event_type varchar(60) NOT NULL,
 payload jsonb NOT NULL CHECK(jsonb_typeof(payload)='object'),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), PRIMARY KEY(export_id,seq)
);
CREATE FUNCTION cq_app.agent_export_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'agent_export_operation_or_event_immutable'; END $$;
CREATE TRIGGER agent_export_operations_immutable BEFORE UPDATE OR DELETE ON cq_app.agent_export_operations
 FOR EACH ROW EXECUTE FUNCTION cq_app.agent_export_immutable();
CREATE TRIGGER agent_export_events_immutable BEFORE UPDATE OR DELETE ON cq_app.agent_export_events
 FOR EACH ROW EXECUTE FUNCTION cq_app.agent_export_immutable();
CREATE FUNCTION cq_app.agent_export_task_owner_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM cq_app.agent_tasks t WHERE t.id=NEW.task_id AND t.owner_id=NEW.owner_id AND t.project_id=NEW.project_id) THEN
  RAISE EXCEPTION 'agent_export_task_owner_project_mismatch';
 END IF;
 IF TG_OP='UPDATE' AND (NEW.id,NEW.owner_id,NEW.project_id,NEW.task_id,NEW.document_sha256,NEW.context_sha256,NEW.requested_formats,NEW.toolchain_profile,NEW.toolchain_sha256,NEW.font_profile,NEW.created_at)
  IS DISTINCT FROM (OLD.id,OLD.owner_id,OLD.project_id,OLD.task_id,OLD.document_sha256,OLD.context_sha256,OLD.requested_formats,OLD.toolchain_profile,OLD.toolchain_sha256,OLD.font_profile,OLD.created_at) THEN
  RAISE EXCEPTION 'agent_export_identity_is_immutable';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER agent_export_task_owner BEFORE INSERT OR UPDATE ON cq_app.agent_document_exports
 FOR EACH ROW EXECUTE FUNCTION cq_app.agent_export_task_owner_guard();
"""
