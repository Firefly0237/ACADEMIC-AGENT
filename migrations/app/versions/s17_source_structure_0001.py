"""Immutable, domain-scoped pointers for derived source navigation metadata."""
from alembic import op

revision = "s17_source_structure_0001"
down_revision = "s16_observed_publication_0001"
branch_labels = None
depends_on = None

DDL = r"""
CREATE TABLE cq_app.source_structure_cache (
 security_domain varchar(100) NOT NULL,
 source_kind varchar(32) NOT NULL CHECK(source_kind IN ('native_representation','metadata_projection','material_file')),
 source_id varchar(64) NOT NULL CHECK(length(source_id)>0),
 source_version_sha256 varchar(64) NOT NULL CHECK(source_version_sha256 ~ '^[a-f0-9]{64}$'),
 policy_sha256 varchar(64) NOT NULL CHECK(policy_sha256 ~ '^[a-f0-9]{64}$'),
 sidecar_sha256 varchar(64) NOT NULL CHECK(sidecar_sha256 ~ '^[a-f0-9]{64}$'),
 sidecar_size_bytes bigint NOT NULL CHECK(sidecar_size_bytes BETWEEN 1 AND 16777216),
 source_binding_sha256 varchar(64) NOT NULL CHECK(source_binding_sha256 ~ '^[a-f0-9]{64}$'),
 schema_version varchar(100) NOT NULL CHECK(schema_version='citequest-source-structure/1'),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(security_domain,source_kind,source_id,source_version_sha256,policy_sha256),
 FOREIGN KEY(security_domain,sidecar_sha256) REFERENCES cq_app.blobs(security_domain,sha256)
);
CREATE TRIGGER source_structure_cache_immutable BEFORE UPDATE OR DELETE ON cq_app.source_structure_cache
 FOR EACH ROW EXECUTE FUNCTION cq_app.reject_source_fact_mutation();
CREATE TRIGGER source_structure_cache_no_truncate BEFORE TRUNCATE ON cq_app.source_structure_cache
 FOR EACH STATEMENT EXECUTE FUNCTION cq_app.reject_source_fact_mutation();
"""


def upgrade() -> None:
    op.execute(DDL)


def downgrade() -> None:
    raise RuntimeError("source_structure_history_requires_explicit_restore")
