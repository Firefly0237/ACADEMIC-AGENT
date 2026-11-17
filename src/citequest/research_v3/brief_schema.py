"""Expand-only migration candidate; executable only through an explicit migrator.

Root chooses the Alembic revision/parent after reconciling the unapplied S09
candidate. No application code imports this SQL or creates its own tables.
"""

DDL = """
CREATE TABLE cq_app.research_briefs (
 id uuid PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(), owner_id uuid NOT NULL,
 project_id uuid NOT NULL, revision integer NOT NULL CHECK(revision>=1), document jsonb NOT NULL,
 document_sha256 varchar(64) NOT NULL CHECK(document_sha256 ~ '^[a-f0-9]{64}$'), deleted_at timestamptz,
 UNIQUE(owner_id,id), FOREIGN KEY(owner_id,project_id) REFERENCES cq_app.projects(owner_id,id),
 CHECK(octet_length(document::text)<=131072)
);
CREATE INDEX ix_research_brief_owner_project_created
 ON cq_app.research_briefs(owner_id,project_id,created_at,id);
CREATE TABLE cq_app.research_brief_revisions (
 brief_id uuid NOT NULL, revision integer NOT NULL CHECK(revision>=1), owner_id uuid NOT NULL,
 operation_id uuid NOT NULL, document jsonb NOT NULL, document_sha256 varchar(64) NOT NULL,
 change_kind varchar(20) NOT NULL CHECK(change_kind IN ('created','turn','resolved','deleted','restored')),
 created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(brief_id,revision), FOREIGN KEY(owner_id,brief_id) REFERENCES cq_app.research_briefs(owner_id,id),
 CHECK(document_sha256 ~ '^[a-f0-9]{64}$')
);
CREATE FUNCTION cq_app.research_brief_revision_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'research_brief_revision_immutable'; END $$;
CREATE TRIGGER research_brief_revision_no_mutation BEFORE UPDATE OR DELETE
 ON cq_app.research_brief_revisions FOR EACH ROW EXECUTE FUNCTION cq_app.research_brief_revision_immutable();
CREATE FUNCTION cq_app.research_brief_identity_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF (NEW.id,NEW.owner_id,NEW.project_id,NEW.created_at) IS DISTINCT FROM
    (OLD.id,OLD.owner_id,OLD.project_id,OLD.created_at) OR NEW.revision<>OLD.revision+1 THEN
   RAISE EXCEPTION 'research_brief_identity_or_revision';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER research_brief_identity BEFORE UPDATE ON cq_app.research_briefs
 FOR EACH ROW EXECUTE FUNCTION cq_app.research_brief_identity_guard();
"""

APPLICATION_GRANTS = """
GRANT SELECT,INSERT,UPDATE ON cq_app.research_briefs TO {application_role};
GRANT SELECT,INSERT ON cq_app.research_brief_revisions TO {application_role};
REVOKE UPDATE,DELETE ON cq_app.research_brief_revisions FROM {application_role};
"""
