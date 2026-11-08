"""Literal expand-only DDL candidate, to merge into the unapplied S11 migration.

No runtime DDL and no import of this constant from an executed Alembic revision.
Root's backend owner captures the literal into one self-contained revision.
"""

DDL = """
CREATE TABLE cq_app.saved_paper_references(
 id uuid PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(), owner_id uuid NOT NULL REFERENCES cq_app.owners(id),
 paper_kind varchar(30) NOT NULL CHECK(paper_kind IN ('metadata_projection','native_representation')),
 paper_id varchar(64) NOT NULL, work_id uuid REFERENCES cq_app.works(id),
 source_snapshot_id uuid NOT NULL, captured_metadata jsonb NOT NULL,
 captured_metadata_sha256 varchar(64) NOT NULL CHECK(captured_metadata_sha256 ~ '^[a-f0-9]{64}$'),
 source_projection_sha256 varchar(64) NOT NULL CHECK(source_projection_sha256 ~ '^[a-f0-9]{64}$'),
 source_pointer_sha256 varchar(64) NOT NULL CHECK(source_pointer_sha256 ~ '^[a-f0-9]{64}$'),
 revision integer NOT NULL DEFAULT 1 CHECK(revision>=1), deleted_at timestamptz,
 UNIQUE(owner_id,id), UNIQUE(owner_id,paper_kind,paper_id),
 CHECK((paper_kind='native_representation' AND paper_id ~ '^[a-f0-9]{64}$') OR
       (paper_kind='metadata_projection' AND paper_id ~ '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$')),
 CHECK(octet_length(captured_metadata::text)<=131072)
);
CREATE INDEX ix_saved_paper_owner_lifecycle_created ON cq_app.saved_paper_references(owner_id,deleted_at,created_at,id);
CREATE TABLE cq_app.saved_paper_collections(
 owner_id uuid NOT NULL, saved_paper_id uuid NOT NULL, collection_id uuid NOT NULL,
 PRIMARY KEY(owner_id,saved_paper_id,collection_id),
 FOREIGN KEY(owner_id,saved_paper_id) REFERENCES cq_app.saved_paper_references(owner_id,id),
 FOREIGN KEY(owner_id,collection_id) REFERENCES cq_app.collections(owner_id,id)
);
CREATE INDEX ix_saved_paper_collection_members ON cq_app.saved_paper_collections(owner_id,collection_id,saved_paper_id);
CREATE TABLE cq_app.saved_paper_tags(
 owner_id uuid NOT NULL, saved_paper_id uuid NOT NULL, tag varchar(100) NOT NULL CHECK(length(tag)>0),
 PRIMARY KEY(owner_id,saved_paper_id,tag),
 FOREIGN KEY(owner_id,saved_paper_id) REFERENCES cq_app.saved_paper_references(owner_id,id)
);
CREATE INDEX ix_saved_paper_tag_members ON cq_app.saved_paper_tags(owner_id,tag,saved_paper_id);
CREATE TABLE cq_app.saved_paper_notes(
 id uuid PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(), owner_id uuid NOT NULL,
 saved_paper_id uuid NOT NULL, body text NOT NULL CHECK(length(body) BETWEEN 1 AND 15000),
 revision integer NOT NULL DEFAULT 1 CHECK(revision>=1), deleted_at timestamptz,
 UNIQUE(owner_id,id), FOREIGN KEY(owner_id,saved_paper_id) REFERENCES cq_app.saved_paper_references(owner_id,id)
);
CREATE INDEX ix_saved_paper_note_owner_created ON cq_app.saved_paper_notes(owner_id,saved_paper_id,deleted_at,created_at,id);
CREATE FUNCTION cq_app.saved_paper_identity_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF (to_jsonb(NEW)-'revision'-'deleted_at') IS DISTINCT FROM (to_jsonb(OLD)-'revision'-'deleted_at')
    OR NEW.revision<>OLD.revision+1 THEN RAISE EXCEPTION 'saved_paper_identity_immutable'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER saved_paper_identity BEFORE UPDATE ON cq_app.saved_paper_references
 FOR EACH ROW EXECUTE FUNCTION cq_app.saved_paper_identity_guard();
CREATE FUNCTION cq_app.saved_paper_note_identity_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF (NEW.id,NEW.owner_id,NEW.saved_paper_id,NEW.created_at) IS DISTINCT FROM
    (OLD.id,OLD.owner_id,OLD.saved_paper_id,OLD.created_at) OR NEW.revision<>OLD.revision+1 THEN
   RAISE EXCEPTION 'saved_paper_note_identity_or_revision'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER saved_paper_note_identity BEFORE UPDATE ON cq_app.saved_paper_notes
 FOR EACH ROW EXECUTE FUNCTION cq_app.saved_paper_note_identity_guard();
"""
