"""Bounded independent RAG chunk receipts; no change to paper result ranks."""
from alembic import op

revision = "s15_chunks_0001"
down_revision = "s15_snapshot_0001"
branch_labels = None
depends_on = None

DDL = """
CREATE TABLE cq_app.evidence_retrievals (
 id uuid PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 owner_id uuid NOT NULL REFERENCES cq_app.owners(id),
 generation_id uuid NOT NULL REFERENCES cq_app.index_generations(id),
 epoch integer NOT NULL CHECK(epoch>=1),
 request jsonb NOT NULL CHECK(jsonb_typeof(request)='object' AND octet_length(request::text)<=32768),
 request_sha256 varchar(64) NOT NULL CHECK(request_sha256 ~ '^[a-f0-9]{64}$'),
 catalog_binding_sha256 varchar(64) NOT NULL CHECK(catalog_binding_sha256 ~ '^[a-f0-9]{64}$'),
 candidate_count integer NOT NULL CHECK(candidate_count BETWEEN 0 AND 20),
 delivered_count integer NOT NULL CHECK(delivered_count>=0), masked_count integer NOT NULL CHECK(masked_count>=0),
 CHECK(candidate_count=delivered_count+masked_count),
 expires_at timestamptz NOT NULL, UNIQUE(id,generation_id)
);
CREATE INDEX ix_evidence_retrieval_owner_expiry ON cq_app.evidence_retrievals(owner_id,expires_at,id);
CREATE INDEX ix_evidence_retrieval_generation_expiry ON cq_app.evidence_retrievals(generation_id,expires_at);
CREATE TABLE cq_app.evidence_retrieval_hits (
 retrieval_id uuid NOT NULL, ordinal integer NOT NULL CHECK(ordinal BETWEEN 1 AND 20),
 generation_id uuid NOT NULL, projection_id uuid NOT NULL, chunk_id varchar(64) NOT NULL,
 score double precision NOT NULL CHECK(score>=-1.0e308 AND score<=1.0e308),
 PRIMARY KEY(retrieval_id,ordinal), UNIQUE(retrieval_id,chunk_id),
 FOREIGN KEY(retrieval_id,generation_id) REFERENCES cq_app.evidence_retrievals(id,generation_id) ON DELETE CASCADE,
 FOREIGN KEY(generation_id,projection_id) REFERENCES cq_app.index_units(generation_id,projection_id),
 FOREIGN KEY(generation_id,chunk_id) REFERENCES cq_app.index_chunks(generation_id,chunk_id)
);
CREATE FUNCTION cq_app.evidence_retrieval_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='UPDATE' THEN RAISE EXCEPTION 'evidence_retrieval_is_immutable'; END IF;
 -- Each expression is prepared for its table's concrete NEW record. Boolean
 -- AND short-circuiting cannot protect fields absent from the other table.
 IF TG_TABLE_NAME='evidence_retrievals' THEN
  IF NEW.request->>'retrieval_policy' IS DISTINCT FROM 'source-chunk-hybrid/1'
   OR NOT EXISTS(SELECT 1 FROM cq_app.index_publish_intents p JOIN cq_app.index_generations g
    ON g.id=p.target_generation_id WHERE p.state='committed' AND p.target_generation_id=NEW.generation_id
     AND p.expected_epoch+1=NEW.epoch AND p.catalog_name=g.catalog_name
     AND p.security_domain=g.security_domain AND g.catalog_name=NEW.request->'search'->>'catalog')
  THEN RAISE EXCEPTION 'evidence_retrieval_epoch_or_catalog_not_published'; END IF;
 ELSIF TG_TABLE_NAME='evidence_retrieval_hits' THEN
  IF NOT EXISTS(SELECT 1 FROM cq_app.index_chunks c WHERE c.generation_id=NEW.generation_id
   AND c.chunk_id=NEW.chunk_id AND c.projection_id=NEW.projection_id)
  THEN RAISE EXCEPTION 'evidence_retrieval_chunk_unit_mismatch'; END IF;
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER evidence_retrieval_immutable BEFORE INSERT OR UPDATE ON cq_app.evidence_retrievals
 FOR EACH ROW EXECUTE FUNCTION cq_app.evidence_retrieval_guard();
CREATE TRIGGER evidence_retrieval_hit_immutable BEFORE INSERT OR UPDATE ON cq_app.evidence_retrieval_hits
 FOR EACH ROW EXECUTE FUNCTION cq_app.evidence_retrieval_guard();
CREATE FUNCTION cq_app.evidence_retrieval_delivery_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE receipt_id uuid; expected integer; actual integer; first_ordinal integer; last_ordinal integer;
BEGIN
 IF TG_TABLE_NAME='evidence_retrievals' THEN receipt_id:=NEW.id;
 ELSIF TG_OP='DELETE' THEN receipt_id:=OLD.retrieval_id;
 ELSE receipt_id:=NEW.retrieval_id; END IF;
 SELECT delivered_count INTO expected FROM cq_app.evidence_retrievals WHERE id=receipt_id;
 IF NOT FOUND THEN RETURN NULL; END IF;
 SELECT count(*),min(ordinal),max(ordinal) INTO actual,first_ordinal,last_ordinal
  FROM cq_app.evidence_retrieval_hits WHERE retrieval_id=receipt_id;
 IF actual<>expected OR (actual>0 AND (first_ordinal<>1 OR last_ordinal<>actual))
 THEN RAISE EXCEPTION 'evidence_retrieval_delivery_count_or_order_mismatch'; END IF;
 RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER evidence_retrieval_delivery_complete AFTER INSERT ON cq_app.evidence_retrievals
 DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION cq_app.evidence_retrieval_delivery_guard();
CREATE CONSTRAINT TRIGGER evidence_retrieval_hit_delivery_complete AFTER INSERT OR DELETE ON cq_app.evidence_retrieval_hits
 DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION cq_app.evidence_retrieval_delivery_guard();
"""


def upgrade() -> None:
    op.execute(DDL)


def downgrade() -> None:
    raise RuntimeError("Preserve retrieval receipts and current Task references; explicit retention planning required")
