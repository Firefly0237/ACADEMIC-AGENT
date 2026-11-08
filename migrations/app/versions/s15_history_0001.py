"""Keep user search history independent of expiring index result snapshots."""
from alembic import op

revision = "s15_history_0001"
down_revision = "s15_dispatch_0001"
branch_labels = None
depends_on = None

DDL = """
CREATE TABLE cq_app.search_activities (
 history_id uuid PRIMARY KEY REFERENCES cq_app.history(id) ON DELETE CASCADE,
 owner_id uuid NOT NULL REFERENCES cq_app.owners(id),
 search_snapshot_id uuid NOT NULL, generation_id uuid NOT NULL,
 epoch integer NOT NULL CHECK(epoch>=0),
 request jsonb NOT NULL CHECK(jsonb_typeof(request)='object' AND octet_length(request::text)<=32768),
 request_sha256 varchar(64) NOT NULL CHECK(request_sha256 ~ '^[a-f0-9]{64}$'),
 UNIQUE(owner_id,search_snapshot_id)
);
CREATE FUNCTION cq_app.search_activity_identity_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='UPDATE' THEN RAISE EXCEPTION 'search_history_capture_is_immutable'; END IF;
 IF NOT EXISTS(SELECT 1 FROM cq_app.history h WHERE h.id=NEW.history_id AND h.owner_id=NEW.owner_id
  AND h.kind='search' AND h.target_id='search:'||NEW.search_snapshot_id::text) THEN
  RAISE EXCEPTION 'search_history_owner_or_target_mismatch';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER search_activity_identity BEFORE INSERT OR UPDATE ON cq_app.search_activities
 FOR EACH ROW EXECUTE FUNCTION cq_app.search_activity_identity_guard();
"""


def upgrade() -> None:
    op.execute(DDL)


def downgrade() -> None:
    raise RuntimeError("User search history requires explicit retained-data rollback planning")
