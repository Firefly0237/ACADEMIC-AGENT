"""Optional durable FIFO resource admission; disabled until a drained activation."""
from alembic import op

revision = "s15_dispatch_0001"
down_revision = "s13_exports_0001"
branch_labels = None
depends_on = None

DDL = r"""
CREATE TABLE cq_app.agent_dispatch_control (
 singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton), enabled boolean NOT NULL DEFAULT false,
 policy jsonb, policy_sha256 varchar(64), activated_at timestamptz,
 CHECK(NOT enabled OR (jsonb_typeof(policy)='object' AND policy_sha256 ~ '^[a-f0-9]{64}$'))
);
INSERT INTO cq_app.agent_dispatch_control(singleton,enabled) VALUES(true,false);
CREATE TABLE cq_app.agent_command_queues (
 command_id uuid PRIMARY KEY REFERENCES cq_app.agent_commands(id),
 task_id uuid NOT NULL REFERENCES cq_app.agent_tasks(id), queued_at timestamptz NOT NULL,
 claimed_at timestamptz NOT NULL, queue_seconds double precision NOT NULL,
 CHECK(claimed_at>=queued_at AND queue_seconds>=0 AND queue_seconds<'Infinity'::double precision)
);
CREATE INDEX ix_agent_command_queues_task ON cq_app.agent_command_queues(task_id,command_id);
CREATE TABLE cq_app.agent_model_waiters (
 id bigserial PRIMARY KEY, task_id uuid NOT NULL REFERENCES cq_app.agent_tasks(id),
 command_id uuid NOT NULL REFERENCES cq_app.agent_commands(id), run_id uuid NOT NULL REFERENCES cq_app.agent_runs(id),
 epoch bigint NOT NULL CHECK(epoch>=1), logical_key text NOT NULL, request_hash varchar(64) NOT NULL CHECK(request_hash ~ '^[a-f0-9]{64}$'),
 enqueued_at timestamptz NOT NULL DEFAULT clock_timestamp(), expires_at timestamptz NOT NULL,
 state varchar(16) NOT NULL DEFAULT 'waiting' CHECK(state IN ('waiting','admitted','cancelled','expired')),
 reason varchar(100), attempt_id uuid REFERENCES cq_app.agent_attempts(id), admitted_at timestamptz,
 UNIQUE(task_id,logical_key), CHECK(expires_at>=enqueued_at),
 CHECK((state='admitted' AND attempt_id IS NOT NULL AND admitted_at IS NOT NULL) OR (state<>'admitted' AND attempt_id IS NULL AND admitted_at IS NULL))
);
CREATE INDEX ix_agent_model_waiters_head ON cq_app.agent_model_waiters(id) WHERE state='waiting';
CREATE INDEX ix_agent_model_waiters_task ON cq_app.agent_model_waiters(task_id,run_id,epoch,state);
CREATE INDEX ix_agent_tasks_research_dispatch ON cq_app.agent_tasks(kind,state,id) WHERE state IN ('queued','running','reconciling');
CREATE FUNCTION cq_app.agent_queue_receipt_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP<>'INSERT' THEN RAISE EXCEPTION 'command_queue_receipt_append_only'; END IF;
 IF NOT EXISTS(SELECT 1 FROM cq_app.agent_commands c WHERE c.id=NEW.command_id AND c.task_id=NEW.task_id AND c.created_at=NEW.queued_at)
 THEN RAISE EXCEPTION 'command_queue_receipt_identity_mismatch'; END IF;
 IF abs(NEW.queue_seconds-extract(epoch FROM NEW.claimed_at-NEW.queued_at))>0.000001
 THEN RAISE EXCEPTION 'command_queue_receipt_duration_mismatch'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER agent_command_queue_immutable BEFORE INSERT OR UPDATE OR DELETE ON cq_app.agent_command_queues
 FOR EACH ROW EXECUTE FUNCTION cq_app.agent_queue_receipt_guard();
CREATE FUNCTION cq_app.agent_waiter_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='DELETE' THEN RAISE EXCEPTION 'model_waiter_history_preserved'; END IF;
 IF TG_OP='UPDATE' THEN
  IF (NEW.id,NEW.task_id,NEW.command_id,NEW.logical_key,NEW.request_hash,NEW.enqueued_at,NEW.expires_at)
   IS DISTINCT FROM (OLD.id,OLD.task_id,OLD.command_id,OLD.logical_key,OLD.request_hash,OLD.enqueued_at,OLD.expires_at)
  THEN RAISE EXCEPTION 'model_waiter_identity_immutable'; END IF;
  IF OLD.state<>'waiting' THEN RAISE EXCEPTION 'model_waiter_closed'; END IF;
 END IF;
 IF NEW.state IN ('waiting','admitted') AND NOT EXISTS(
  SELECT 1 FROM cq_app.agent_tasks t JOIN cq_app.agent_commands c ON c.task_id=t.id
   JOIN cq_app.agent_runs r ON r.task_id=t.id AND r.command_id=c.id
  WHERE t.id=NEW.task_id AND t.epoch=NEW.epoch AND t.state='running' AND NOT t.cancellation_requested
   AND c.id=NEW.command_id AND c.state='running' AND r.id=NEW.run_id AND r.epoch=NEW.epoch AND r.state='running'
   AND NEW.logical_key LIKE NEW.command_id::text || ':%')
 THEN RAISE EXCEPTION 'model_waiter_current_execution_required'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER agent_model_waiter_immutable BEFORE INSERT OR UPDATE OR DELETE ON cq_app.agent_model_waiters
 FOR EACH ROW EXECUTE FUNCTION cq_app.agent_waiter_guard();
CREATE FUNCTION cq_app.agent_dispatch_head() RETURNS bigint LANGUAGE sql STABLE AS $$
 SELECT w.id FROM cq_app.agent_model_waiters w JOIN cq_app.agent_tasks t ON t.id=w.task_id
 JOIN cq_app.agent_commands c ON c.id=w.command_id AND c.task_id=t.id
 JOIN cq_app.agent_runs r ON r.id=w.run_id AND r.task_id=t.id AND r.command_id=c.id
 WHERE w.state='waiting' AND w.epoch=t.epoch AND r.epoch=w.epoch AND r.state='running'
  AND t.state='running' AND NOT t.cancellation_requested AND t.running_since IS NOT NULL
  AND c.state='running' AND t.lease_until>clock_timestamp() AND w.expires_at>clock_timestamp()
  AND t.active_seconds+greatest(0,extract(epoch FROM clock_timestamp()-t.running_since))
   +CASE WHEN t.kind='qa' THEN coalesce((SELECT sum(q.queue_seconds) FROM cq_app.agent_command_queues q WHERE q.task_id=t.id),0) ELSE 0 END
   <(t.profile->>'active_seconds')::double precision
 ORDER BY w.id LIMIT 1
$$;
CREATE FUNCTION cq_app.agent_dispatch_task_unlocked(task uuid) RETURNS boolean LANGUAGE sql STABLE AS $$
 SELECT NOT EXISTS(SELECT 1 FROM pg_locks l
  WHERE l.locktype='advisory' AND l.granted AND l.objsubid=1
   AND l.database=(SELECT oid FROM pg_database WHERE datname=current_database())
   AND l.classid::bigint=(('x'||substr(encode(sha256(convert_to(task::text,'UTF8')),'hex'),1,8))::bit(32)::bigint)
   AND l.objid::bigint=(('x'||substr(encode(sha256(convert_to(task::text,'UTF8')),'hex'),9,8))::bit(32)::bigint))
$$;
CREATE FUNCTION cq_app.agent_dispatch_attempt_before() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE candidate bigint; BEGIN
 IF NEW.kind<>'model' OR NOT EXISTS(SELECT 1 FROM cq_app.agent_dispatch_control WHERE singleton AND enabled) THEN RETURN NEW; END IF;
 PERFORM pg_advisory_xact_lock(7122828494682946860);
 IF EXISTS(SELECT 1 FROM cq_app.agent_attempts WHERE kind='model' AND status IN ('reserved','sent','unknown'))
 THEN RAISE EXCEPTION 'durable_model_dispatch_slot_occupied'; END IF;
 candidate:=cq_app.agent_dispatch_head();
 IF candidate IS NULL OR NOT EXISTS(SELECT 1 FROM cq_app.agent_model_waiters w WHERE w.id=candidate
  AND w.task_id=NEW.task_id AND w.run_id=NEW.run_id AND w.epoch=NEW.epoch
  AND w.logical_key=NEW.logical_key AND w.request_hash=NEW.request_hash AND w.state='waiting')
 THEN RAISE EXCEPTION 'durable_FIFO_model_waiter_required'; END IF;
 RETURN NEW;
END $$;
CREATE FUNCTION cq_app.agent_dispatch_attempt_after() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.kind='model' AND EXISTS(SELECT 1 FROM cq_app.agent_dispatch_control WHERE singleton AND enabled) THEN
  UPDATE cq_app.agent_model_waiters SET state='admitted',attempt_id=NEW.id,admitted_at=clock_timestamp()
   WHERE task_id=NEW.task_id AND logical_key=NEW.logical_key AND run_id=NEW.run_id AND epoch=NEW.epoch AND state='waiting';
  IF NOT FOUND THEN RAISE EXCEPTION 'durable_FIFO_waiter_admission_lost'; END IF;
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER agent_attempt_FIFO_before BEFORE INSERT ON cq_app.agent_attempts
 FOR EACH ROW EXECUTE FUNCTION cq_app.agent_dispatch_attempt_before();
CREATE TRIGGER agent_attempt_FIFO_after AFTER INSERT ON cq_app.agent_attempts
 FOR EACH ROW EXECUTE FUNCTION cq_app.agent_dispatch_attempt_after();
CREATE FUNCTION cq_app.agent_dispatch_control_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP<>'UPDATE' OR NEW.singleton IS DISTINCT FROM OLD.singleton THEN RAISE EXCEPTION 'dispatch_control_singleton_preserved'; END IF;
 PERFORM pg_advisory_xact_lock(7122828494682946860);
 IF (NEW.enabled,NEW.policy,NEW.policy_sha256) IS DISTINCT FROM (OLD.enabled,OLD.policy,OLD.policy_sha256) THEN
  IF EXISTS(SELECT 1 FROM cq_app.agent_tasks WHERE state IN ('running','reconciling'))
   OR EXISTS(SELECT 1 FROM cq_app.agent_attempts WHERE status IN ('reserved','sent') OR (kind='model' AND status='unknown'))
   OR EXISTS(SELECT 1 FROM cq_app.agent_model_waiters WHERE state='waiting')
  THEN RAISE EXCEPTION 'dispatch_policy_change_requires_drain'; END IF;
  IF NEW.enabled AND (NEW.policy IS NULL OR NEW.policy_sha256 IS NULL OR NEW.policy_sha256 !~ '^[a-f0-9]{64}$'
   OR NEW.policy->>'version' IS DISTINCT FROM 'cq.agent-dispatch-fifo/1'
   OR NEW.policy->>'model_concurrency' IS DISTINCT FROM '1'
   OR NEW.policy->>'research_active' IS DISTINCT FROM '1'
   OR NEW.policy->>'research_queued' IS DISTINCT FROM '2'
   OR NEW.policy->>'research_queue_seconds' IS DISTINCT FROM '300'
   OR NEW.policy->'short_kinds' IS DISTINCT FROM '["qa"]'::jsonb
   OR NEW.policy->'long_kinds' IS DISTINCT FROM '["summary","compare","stance","research"]'::jsonb
   OR (SELECT count(*) FROM cq_app.agent_tasks WHERE kind='research' AND state='queued')>2)
  THEN RAISE EXCEPTION 'fixed_dispatch_policy_or_capacity_required'; END IF;
  NEW.activated_at:=CASE WHEN NEW.enabled THEN clock_timestamp() ELSE NULL END;
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER agent_dispatch_control_protected BEFORE INSERT OR UPDATE OR DELETE ON cq_app.agent_dispatch_control
 FOR EACH ROW EXECUTE FUNCTION cq_app.agent_dispatch_control_guard();
DO $$ BEGIN IF EXISTS(SELECT 1 FROM pg_roles WHERE rolname='cq_s03_app') THEN
 GRANT SELECT ON cq_app.agent_dispatch_control TO cq_s03_app;
 GRANT SELECT,INSERT ON cq_app.agent_command_queues TO cq_s03_app;
 GRANT SELECT,INSERT,UPDATE ON cq_app.agent_model_waiters TO cq_s03_app;
 GRANT USAGE,SELECT ON SEQUENCE cq_app.agent_model_waiters_id_seq TO cq_s03_app;
END IF; END $$;
"""


def upgrade() -> None:
    op.execute(DDL)


def downgrade() -> None:
    raise RuntimeError("Preserve dispatch tickets, queue times and financial history; no destructive downgrade")
