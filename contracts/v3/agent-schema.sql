-- S08 candidate migration body. Root coordinates the Alembic parent and application.
-- No source imports or runtime DDL. Official Saver/Store setup is a separate migration step.
CREATE TABLE cq_app.agent_tasks (
 id uuid PRIMARY KEY, owner_id uuid NOT NULL REFERENCES cq_app.owners(id),
 project_id uuid NOT NULL, operation_id uuid NOT NULL, create_hash text NOT NULL,
 kind text NOT NULL CHECK(kind IN ('qa','summary','compare','stance')), brief jsonb NOT NULL,
 profile jsonb NOT NULL, profile_hash text NOT NULL, graph_version text NOT NULL,
 revision integer NOT NULL DEFAULT 1, state text NOT NULL DEFAULT 'queued',
 epoch bigint NOT NULL DEFAULT 0, worker_id text, lease_until timestamptz,
 running_since timestamptz, active_seconds double precision NOT NULL DEFAULT 0 CHECK(active_seconds>=0),
 model_calls integer NOT NULL DEFAULT 0, tool_calls integer NOT NULL DEFAULT 0,
 input_tokens bigint NOT NULL DEFAULT 0, output_tokens bigint NOT NULL DEFAULT 0,
 known_ncny bigint NOT NULL DEFAULT 0, reserved_ncny bigint NOT NULL DEFAULT 0,
 event_seq bigint NOT NULL DEFAULT 0, command_seq bigint NOT NULL DEFAULT 1,
 accepted_checkpoint text, result jsonb, error_code text, history jsonb NOT NULL DEFAULT '[]',
 cancellation_requested boolean NOT NULL DEFAULT false,
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 UNIQUE(owner_id,operation_id), FOREIGN KEY(owner_id,project_id) REFERENCES cq_app.projects(owner_id,id),
 CHECK(state IN ('queued','running','waiting_input','succeeded','partial','failed','cancelled','reconciling')),
 CHECK(model_calls>=0 AND tool_calls>=0 AND input_tokens>=0 AND output_tokens>=0 AND known_ncny>=0 AND reserved_ncny>=0)
);
CREATE TABLE cq_app.agent_commands (
 id uuid PRIMARY KEY, task_id uuid NOT NULL REFERENCES cq_app.agent_tasks(id),
 owner_id uuid NOT NULL REFERENCES cq_app.owners(id), operation_id uuid NOT NULL,
 seq bigint NOT NULL, kind text NOT NULL, payload jsonb NOT NULL, payload_hash text NOT NULL,
 state text NOT NULL DEFAULT 'queued' CHECK(state IN ('queued','running','done','rejected')),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), UNIQUE(task_id,seq), UNIQUE(owner_id,operation_id)
);
CREATE INDEX ix_agent_commands_queue ON cq_app.agent_commands(state,created_at,id);
CREATE TABLE cq_app.agent_runs (
 id uuid PRIMARY KEY, task_id uuid NOT NULL REFERENCES cq_app.agent_tasks(id),
 command_id uuid NOT NULL REFERENCES cq_app.agent_commands(id), epoch bigint NOT NULL,
 worker_id text NOT NULL, state text NOT NULL DEFAULT 'running', checkpoint_id text,
 started_at timestamptz NOT NULL DEFAULT clock_timestamp(), ended_at timestamptz
);
CREATE TABLE cq_app.agent_attempts (
 id uuid PRIMARY KEY, task_id uuid NOT NULL REFERENCES cq_app.agent_tasks(id),
 run_id uuid NOT NULL REFERENCES cq_app.agent_runs(id), epoch bigint NOT NULL,
 logical_key text NOT NULL, kind text NOT NULL CHECK(kind IN ('model','tool')),
 request_hash text NOT NULL, request jsonb NOT NULL, status text NOT NULL DEFAULT 'reserved',
 reserved_ncny bigint NOT NULL DEFAULT 0, output_limit integer NOT NULL DEFAULT 0,
 response jsonb, usage jsonb, cost_ncny bigint, error_code text,
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), sent_at timestamptz, settled_at timestamptz,
 UNIQUE(task_id,logical_key), CHECK(status IN ('reserved','sent','settled','unknown','not_sent')),
 CHECK(reserved_ncny>=0 AND (cost_ncny IS NULL OR cost_ncny>=0))
);
CREATE UNIQUE INDEX uq_agent_one_inflight ON cq_app.agent_attempts(task_id) WHERE status IN ('reserved','sent','unknown');
CREATE TABLE cq_app.agent_events (
 task_id uuid NOT NULL REFERENCES cq_app.agent_tasks(id), seq bigint NOT NULL,
 event_type text NOT NULL, payload jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(task_id,seq)
);
CREATE TABLE cq_app.agent_memories (
 owner_id uuid NOT NULL REFERENCES cq_app.owners(id), project_id uuid NOT NULL,
 id uuid NOT NULL, revision integer NOT NULL, kind text NOT NULL, value_hash text NOT NULL,
 state text NOT NULL CHECK(state IN ('pending','active','deleted')), source_refs jsonb NOT NULL,
 deletion_epoch bigint NOT NULL DEFAULT 0, updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(owner_id,project_id,id), FOREIGN KEY(owner_id,project_id) REFERENCES cq_app.projects(owner_id,id)
);
CREATE TABLE cq_app.agent_memory_outbox (
 id uuid PRIMARY KEY, owner_id uuid NOT NULL, project_id uuid NOT NULL, memory_id uuid NOT NULL,
 revision integer NOT NULL, operation_id uuid NOT NULL, payload_hash text NOT NULL,
 operation text NOT NULL CHECK(operation IN ('put','delete')), payload jsonb NOT NULL,
 state text NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','done')),
 UNIQUE(owner_id,operation_id), FOREIGN KEY(owner_id,project_id,memory_id) REFERENCES cq_app.agent_memories(owner_id,project_id,id)
);
CREATE TABLE cq_app.agent_evidence_refs (
 task_id uuid NOT NULL REFERENCES cq_app.agent_tasks(id), ref_id text NOT NULL,
 source_hash text NOT NULL, source_ref jsonb NOT NULL, payload jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(task_id,ref_id)
);
