"""Expand-only candidate; application startup never executes this SQL."""

DDL = """
CREATE TABLE cq_app.business_blob_stores (
 store_id varchar(100) PRIMARY KEY CHECK(length(store_id)>0),
 binding_sha256 varchar(64) NOT NULL CHECK(binding_sha256 ~ '^[a-f0-9]{64}$'),
 backend varchar(16) NOT NULL CHECK(backend IN ('filesystem','gcs')), bucket varchar(100),
 registered_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 CHECK((backend='filesystem' AND bucket IS NULL) OR (backend='gcs' AND bucket IS NOT NULL AND length(bucket)>0)),
 UNIQUE(store_id,binding_sha256)
);
CREATE TABLE cq_app.business_blob_locations (
 security_domain varchar(100) NOT NULL,
 sha256 varchar(64) NOT NULL CHECK(sha256 ~ '^[a-f0-9]{64}$'),
 store_id varchar(100) NOT NULL CHECK(length(store_id)>0),
 size_bytes bigint NOT NULL CHECK(size_bytes>0),
 binding_sha256 varchar(64) NOT NULL CHECK(binding_sha256 ~ '^[a-f0-9]{64}$'),
 reference_sha256 varchar(64) NOT NULL CHECK(reference_sha256 ~ '^[a-f0-9]{64}$'),
 reference jsonb NOT NULL CHECK(jsonb_typeof(reference)='object' AND octet_length(reference::text)<=8192),
 verified_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(security_domain,sha256,store_id),
 UNIQUE(security_domain,sha256,store_id,reference_sha256),
 FOREIGN KEY(security_domain,sha256) REFERENCES cq_app.blobs(security_domain,sha256),
 FOREIGN KEY(store_id,binding_sha256) REFERENCES cq_app.business_blob_stores(store_id,binding_sha256)
);
CREATE TABLE cq_app.business_blob_admissions (
 id uuid PRIMARY KEY, owner_id uuid NOT NULL REFERENCES cq_app.owners(id), operation_id uuid NOT NULL,
 payload_sha256 varchar(64) NOT NULL CHECK(payload_sha256 ~ '^[a-f0-9]{64}$'),
 security_domain varchar(100) NOT NULL,
 sha256 varchar(64) NOT NULL CHECK(sha256 ~ '^[a-f0-9]{64}$'),
 size_bytes bigint NOT NULL CHECK(size_bytes>0), store_id varchar(100) NOT NULL CHECK(length(store_id)>0),
 binding_sha256 varchar(64) NOT NULL CHECK(binding_sha256 ~ '^[a-f0-9]{64}$'),
 state varchar(16) NOT NULL CHECK(state IN ('intent','sent','verified','unknown','failed')),
 revision integer NOT NULL DEFAULT 1 CHECK(revision>=1),
 location_reference_sha256 varchar(64), error_code varchar(100),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 deadline_at timestamptz NOT NULL, dispatch_started_at timestamptz, completed_at timestamptz,
 UNIQUE(owner_id,operation_id),
 CHECK((state='verified')=(location_reference_sha256 IS NOT NULL)),
 FOREIGN KEY(store_id,binding_sha256) REFERENCES cq_app.business_blob_stores(store_id,binding_sha256),
 CHECK((state='intent' AND dispatch_started_at IS NULL AND completed_at IS NULL)
    OR (state IN ('sent','unknown') AND dispatch_started_at IS NOT NULL AND completed_at IS NULL)
    OR (state='verified' AND dispatch_started_at IS NOT NULL AND completed_at IS NOT NULL)
    OR (state='failed' AND dispatch_started_at IS NULL AND completed_at IS NOT NULL)),
 FOREIGN KEY(security_domain,sha256,store_id,location_reference_sha256)
  REFERENCES cq_app.business_blob_locations(security_domain,sha256,store_id,reference_sha256)
);
CREATE UNIQUE INDEX uq_business_blob_admission_inflight
 ON cq_app.business_blob_admissions(security_domain,sha256,store_id) WHERE state IN ('intent','sent','unknown');
CREATE INDEX ix_business_blob_admission_owner ON cq_app.business_blob_admissions(owner_id,created_at,id);
CREATE TABLE cq_app.business_blob_admission_events (
 admission_id uuid NOT NULL REFERENCES cq_app.business_blob_admissions(id), seq integer NOT NULL CHECK(seq>=1),
 event_type varchar(60) NOT NULL, payload jsonb NOT NULL CHECK(jsonb_typeof(payload)='object' AND octet_length(payload::text)<=8192),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), PRIMARY KEY(admission_id,seq)
);
CREATE FUNCTION cq_app.business_blob_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'business_blob_location_or_event_is_immutable'; END $$;
CREATE TRIGGER business_blob_store_immutable BEFORE UPDATE OR DELETE ON cq_app.business_blob_stores
 FOR EACH ROW EXECUTE FUNCTION cq_app.business_blob_append_only();
CREATE TRIGGER business_blob_location_immutable BEFORE UPDATE OR DELETE ON cq_app.business_blob_locations
 FOR EACH ROW EXECUTE FUNCTION cq_app.business_blob_append_only();
CREATE TRIGGER business_blob_event_immutable BEFORE UPDATE OR DELETE ON cq_app.business_blob_admission_events
 FOR EACH ROW EXECUTE FUNCTION cq_app.business_blob_append_only();
CREATE FUNCTION cq_app.business_blob_location_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE configured_backend text; configured_bucket text;
BEGIN
 PERFORM 1 FROM cq_app.blobs b WHERE b.security_domain=NEW.security_domain AND b.sha256=NEW.sha256 AND b.size_bytes=NEW.size_bytes FOR SHARE;
 IF NOT FOUND THEN
  RAISE EXCEPTION 'business_blob_logical_size_conflict';
 END IF;
 IF NEW.reference->'identity'->>'security_domain' IS DISTINCT FROM NEW.security_domain
 OR NEW.reference->'identity'->>'sha256' IS DISTINCT FROM NEW.sha256
 OR (NEW.reference->'identity'->>'size_bytes')::bigint IS DISTINCT FROM NEW.size_bytes
 OR NEW.reference->>'store_id' IS DISTINCT FROM NEW.store_id
 OR NEW.reference->>'object_key' IS DISTINCT FROM ('blobs/'||encode(pg_catalog.sha256(convert_to(NEW.security_domain,'UTF8')),'hex')||'/'||NEW.sha256) THEN
  RAISE EXCEPTION 'business_blob_reference_identity_conflict';
 END IF;
 SELECT backend,bucket INTO configured_backend,configured_bucket FROM cq_app.business_blob_stores
  WHERE store_id=NEW.store_id AND binding_sha256=NEW.binding_sha256;
 IF NOT FOUND OR NEW.reference->>'backend' IS DISTINCT FROM configured_backend
  OR NEW.reference->>'bucket' IS DISTINCT FROM configured_bucket THEN
  RAISE EXCEPTION 'business_blob_reference_store_conflict';
 END IF;
 IF ((NEW.reference->>'backend'='filesystem' AND NEW.reference->'bucket'='null'::jsonb AND NEW.reference->'generation'='null'::jsonb)
  OR (NEW.reference->>'backend'='gcs' AND length(NEW.reference->>'bucket')>0
      AND jsonb_typeof(NEW.reference->'generation')='number' AND (NEW.reference->>'generation')::numeric>0
      AND mod((NEW.reference->>'generation')::numeric,1)=0)) IS NOT TRUE THEN
  RAISE EXCEPTION 'business_blob_reference_backend_invalid';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER business_blob_location_identity BEFORE INSERT ON cq_app.business_blob_locations
 FOR EACH ROW EXECUTE FUNCTION cq_app.business_blob_location_guard();
CREATE FUNCTION cq_app.business_blob_logical_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF EXISTS(SELECT 1 FROM cq_app.business_blob_locations l WHERE l.security_domain=OLD.security_domain AND l.sha256=OLD.sha256)
 AND (TG_OP='DELETE' OR (NEW.security_domain,NEW.sha256,NEW.size_bytes) IS DISTINCT FROM (OLD.security_domain,OLD.sha256,OLD.size_bytes)) THEN
  RAISE EXCEPTION 'business_blob_mapped_logical_identity_is_immutable';
 END IF;
 IF TG_OP='DELETE' THEN RETURN OLD; END IF; RETURN NEW;
END $$;
CREATE TRIGGER business_blob_logical_identity BEFORE UPDATE OR DELETE ON cq_app.blobs
 FOR EACH ROW EXECUTE FUNCTION cq_app.business_blob_logical_guard();
CREATE FUNCTION cq_app.business_blob_admission_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='DELETE' THEN RAISE EXCEPTION 'business_blob_admission_history_is_retained'; END IF;
 IF TG_OP='INSERT' AND (NEW.state<>'intent' OR NEW.revision<>1 OR NEW.location_reference_sha256 IS NOT NULL
    OR NEW.dispatch_started_at IS NOT NULL OR NEW.completed_at IS NOT NULL) THEN
  RAISE EXCEPTION 'business_blob_admission_must_start_as_intent';
 END IF;
 IF TG_OP='UPDATE' THEN
  IF (NEW.id,NEW.owner_id,NEW.operation_id,NEW.payload_sha256,NEW.security_domain,NEW.sha256,NEW.size_bytes,NEW.store_id,NEW.binding_sha256,NEW.created_at,NEW.deadline_at)
    IS DISTINCT FROM (OLD.id,OLD.owner_id,OLD.operation_id,OLD.payload_sha256,OLD.security_domain,OLD.sha256,OLD.size_bytes,OLD.store_id,OLD.binding_sha256,OLD.created_at,OLD.deadline_at)
  OR NEW.revision<>OLD.revision+1
  OR (OLD.dispatch_started_at IS NOT NULL AND NEW.dispatch_started_at IS DISTINCT FROM OLD.dispatch_started_at)
  OR NOT ((OLD.state='intent' AND NEW.state IN ('sent','failed'))
       OR (OLD.state='sent' AND NEW.state IN ('verified','unknown'))
       OR (OLD.state='unknown' AND NEW.state IN ('unknown','verified'))) THEN
   RAISE EXCEPTION 'business_blob_admission_transition_invalid';
  END IF;
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER business_blob_admission_state BEFORE INSERT OR UPDATE OR DELETE ON cq_app.business_blob_admissions
 FOR EACH ROW EXECUTE FUNCTION cq_app.business_blob_admission_guard();
"""
