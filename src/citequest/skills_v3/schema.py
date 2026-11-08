"""Self-contained expand-only DDL candidate; the application never applies it.

Root captures this literal in a new migration after the S11 workstation head.
"""

DDL = """
CREATE TABLE cq_app.user_skills (
 id uuid PRIMARY KEY, owner_id uuid NOT NULL, project_id uuid NOT NULL,
 name varchar(64) NOT NULL CHECK(name ~ '^[a-z0-9]+(-[a-z0-9]+)*$'),
 head_revision integer NOT NULL CHECK(head_revision>=1), head_version_id uuid NOT NULL,
 deleted boolean NOT NULL DEFAULT false,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(owner_id,project_id,id), UNIQUE(owner_id,project_id,name),
 FOREIGN KEY(owner_id,project_id) REFERENCES cq_app.projects(owner_id,id)
);
CREATE INDEX ix_user_skills_catalog ON cq_app.user_skills(owner_id,project_id,deleted,name);
CREATE TABLE cq_app.user_skill_versions (
 id uuid PRIMARY KEY, skill_id uuid NOT NULL, owner_id uuid NOT NULL, project_id uuid NOT NULL,
 created_revision integer NOT NULL CHECK(created_revision>=1),
 content_sha256 varchar(64) NOT NULL CHECK(content_sha256 ~ '^[a-f0-9]{64}$'),
 markdown text NOT NULL CHECK(octet_length(markdown) BETWEEN 1 AND 16384),
 manifest jsonb NOT NULL CHECK(jsonb_typeof(manifest)='object' AND octet_length(manifest::text)<=16384),
 compatibility varchar(60) NOT NULL CHECK(compatibility='citequest-skill-md-instruction-only/1'),
 created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(skill_id,id), UNIQUE(skill_id,created_revision),
 FOREIGN KEY(owner_id,project_id,skill_id) REFERENCES cq_app.user_skills(owner_id,project_id,id)
);
ALTER TABLE cq_app.user_skills ADD CONSTRAINT user_skill_current_version_fk
 FOREIGN KEY(id,head_version_id) REFERENCES cq_app.user_skill_versions(skill_id,id) DEFERRABLE INITIALLY DEFERRED;
CREATE INDEX ix_user_skill_versions_history ON cq_app.user_skill_versions(skill_id,created_revision DESC);
CREATE TABLE cq_app.user_skill_operations (
 owner_id uuid NOT NULL, operation_id uuid NOT NULL, project_id uuid NOT NULL,
 payload_sha256 varchar(64) NOT NULL CHECK(payload_sha256 ~ '^[a-f0-9]{64}$'), result jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(owner_id,operation_id),
 FOREIGN KEY(owner_id,project_id) REFERENCES cq_app.projects(owner_id,id)
);
CREATE FUNCTION cq_app.user_skill_version_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'user_skill_version_or_operation_immutable'; END $$;
CREATE TRIGGER user_skill_versions_immutable BEFORE UPDATE OR DELETE ON cq_app.user_skill_versions
 FOR EACH ROW EXECUTE FUNCTION cq_app.user_skill_version_immutable();
CREATE TRIGGER user_skill_operations_immutable BEFORE UPDATE OR DELETE ON cq_app.user_skill_operations
 FOR EACH ROW EXECUTE FUNCTION cq_app.user_skill_version_immutable();
CREATE TRIGGER user_skills_no_delete BEFORE DELETE ON cq_app.user_skills
 FOR EACH ROW EXECUTE FUNCTION cq_app.user_skill_version_immutable();
CREATE FUNCTION cq_app.user_skill_identity_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF (NEW.id,NEW.owner_id,NEW.project_id,NEW.name,NEW.created_at) IS DISTINCT FROM
    (OLD.id,OLD.owner_id,OLD.project_id,OLD.name,OLD.created_at) OR NEW.head_revision<>OLD.head_revision+1 THEN
  RAISE EXCEPTION 'user_skill_identity_or_revision';
 END IF;
 IF NEW.head_version_id<>OLD.head_version_id AND NOT EXISTS(
  SELECT 1 FROM cq_app.user_skill_versions v WHERE v.skill_id=NEW.id AND v.id=NEW.head_version_id
   AND v.owner_id=NEW.owner_id AND v.project_id=NEW.project_id AND v.created_revision=NEW.head_revision) THEN
  RAISE EXCEPTION 'user_skill_new_version_revision';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER user_skill_identity BEFORE UPDATE ON cq_app.user_skills
 FOR EACH ROW EXECUTE FUNCTION cq_app.user_skill_identity_guard();
"""
