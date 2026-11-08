"""S03 initial business schema; fixed SQL snapshot, no runtime ORM imports."""

from alembic import op

revision = "s03_0001"
down_revision = None
branch_labels = None
depends_on = None

STATEMENTS = [
    "\nCREATE TABLE cq_app.blobs (\n\tsecurity_domain VARCHAR(100) NOT NULL, \n\tsha256 VARCHAR(64) NOT NULL, \n\tsize_bytes BIGINT NOT NULL, \n\tCONSTRAINT pk_blobs PRIMARY KEY (security_domain, sha256), \n\tCONSTRAINT ck_blobs_positive_size CHECK (size_bytes > 0)\n)\n\n",
    "\nCREATE TABLE cq_app.outbox (\n\tidempotency_key VARCHAR(200) NOT NULL, \n\tentity_id VARCHAR(100) NOT NULL, \n\tentity_revision INTEGER NOT NULL, \n\toperation VARCHAR(20) NOT NULL, \n\tpayload_sha256 VARCHAR(64) NOT NULL, \n\tstate VARCHAR(20) NOT NULL, \n\tattempt_count INTEGER NOT NULL, \n\tid UUID NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tCONSTRAINT pk_outbox PRIMARY KEY (id), \n\tCONSTRAINT uq_outbox_idempotency_key UNIQUE (idempotency_key)\n)\n\n",
    "CREATE INDEX ix_outbox_state_created ON cq_app.outbox (state, created_at, id)",
    "\nCREATE TABLE cq_app.owners (\n\tname VARCHAR(500) NOT NULL, \n\tid UUID NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tCONSTRAINT pk_owners PRIMARY KEY (id)\n)\n\n",
    "\nCREATE TABLE cq_app.works (\n\ttitle TEXT NOT NULL, \n\ttitle_normalized TEXT NOT NULL, \n\tauthors JSONB, \n\tyear INTEGER, \n\tpublic BOOLEAN NOT NULL, \n\tmetadata_revision INTEGER NOT NULL, \n\tfield_provenance JSONB NOT NULL, \n\tcanonical_redirect_work_id UUID, \n\tid UUID NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tCONSTRAINT pk_works PRIMARY KEY (id), \n\tCONSTRAINT ck_works_year_range CHECK (year IS NULL OR year BETWEEN 1000 AND 2200), \n\tCONSTRAINT fk_works_canonical_redirect_work_id_works FOREIGN KEY(canonical_redirect_work_id) REFERENCES cq_app.works (id)\n)\n\n",
    "CREATE INDEX ix_works_title_normalized ON cq_app.works (title_normalized)",
    "\nCREATE TABLE cq_app.collections (\n\towner_id UUID NOT NULL, \n\tname VARCHAR(500) NOT NULL, \n\tparent_id UUID, \n\trevision INTEGER NOT NULL, \n\tdeleted_at TIMESTAMP WITH TIME ZONE, \n\tid UUID NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tCONSTRAINT pk_collections PRIMARY KEY (id), \n\tCONSTRAINT uq_collections_owner_id_id UNIQUE (owner_id, id), \n\tCONSTRAINT fk_collections_owner_id_parent_id_collections FOREIGN KEY(owner_id, parent_id) REFERENCES cq_app.collections (owner_id, id), \n\tCONSTRAINT fk_collections_owner_id_owners FOREIGN KEY(owner_id) REFERENCES cq_app.owners (id)\n)\n\n",
    "\nCREATE TABLE cq_app.history (\n\towner_id UUID NOT NULL, \n\tkind VARCHAR(40) NOT NULL, \n\ttarget_id VARCHAR(100) NOT NULL, \n\tsnapshot_id UUID, \n\tid UUID NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tCONSTRAINT pk_history PRIMARY KEY (id), \n\tCONSTRAINT fk_history_owner_id_owners FOREIGN KEY(owner_id) REFERENCES cq_app.owners (id)\n)\n\n",
    "CREATE INDEX ix_history_owner_created ON cq_app.history (owner_id, created_at, id)",
    "\nCREATE TABLE cq_app.idempotency_keys (\n\towner_id UUID NOT NULL, \n\tkey UUID NOT NULL, \n\tpayload_sha256 VARCHAR(64) NOT NULL, \n\tresult JSONB, \n\tCONSTRAINT pk_idempotency_keys PRIMARY KEY (owner_id, key), \n\tCONSTRAINT fk_idempotency_keys_owner_id_owners FOREIGN KEY(owner_id) REFERENCES cq_app.owners (id)\n)\n\n",
    "\nCREATE TABLE cq_app.library_items (\n\towner_id UUID NOT NULL, \n\twork_id UUID NOT NULL, \n\trevision INTEGER NOT NULL, \n\tdeleted_at TIMESTAMP WITH TIME ZONE, \n\tid UUID NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tCONSTRAINT pk_library_items PRIMARY KEY (id), \n\tCONSTRAINT uq_library_items_owner_id_work_id UNIQUE (owner_id, work_id), \n\tCONSTRAINT uq_library_items_owner_id_id UNIQUE (owner_id, id), \n\tCONSTRAINT fk_library_items_owner_id_owners FOREIGN KEY(owner_id) REFERENCES cq_app.owners (id), \n\tCONSTRAINT fk_library_items_work_id_works FOREIGN KEY(work_id) REFERENCES cq_app.works (id)\n)\n\n",
    "CREATE INDEX ix_library_owner_created ON cq_app.library_items (owner_id, created_at, id)",
    "\nCREATE TABLE cq_app.local_sessions (\n\towner_id UUID NOT NULL, \n\ttoken_sha256 VARCHAR(64) NOT NULL, \n\tcsrf_sha256 VARCHAR(64) NOT NULL, \n\texpires_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\trevoked BOOLEAN NOT NULL, \n\tid UUID NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tCONSTRAINT pk_local_sessions PRIMARY KEY (id), \n\tCONSTRAINT fk_local_sessions_owner_id_owners FOREIGN KEY(owner_id) REFERENCES cq_app.owners (id), \n\tCONSTRAINT uq_local_sessions_token_sha256 UNIQUE (token_sha256)\n)\n\n",
    "CREATE INDEX ix_local_sessions_owner_id ON cq_app.local_sessions (owner_id)",
    "\nCREATE TABLE cq_app.projects (\n\towner_id UUID NOT NULL, \n\ttitle VARCHAR(300) NOT NULL, \n\trevision INTEGER NOT NULL, \n\tid UUID NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tCONSTRAINT pk_projects PRIMARY KEY (id), \n\tCONSTRAINT uq_projects_owner_id_id UNIQUE (owner_id, id), \n\tCONSTRAINT fk_projects_owner_id_owners FOREIGN KEY(owner_id) REFERENCES cq_app.owners (id)\n)\n\n",
    "\nCREATE TABLE cq_app.source_records (\n\twork_id UUID NOT NULL, \n\tsource VARCHAR(100) NOT NULL, \n\tnative_id VARCHAR(512) NOT NULL, \n\tsource_revision VARCHAR(100) NOT NULL, \n\traw_sha256 VARCHAR(64) NOT NULL, \n\tsource_url TEXT, \n\tlicense TEXT, \n\trights_status VARCHAR(20) NOT NULL, \n\tpayload JSONB NOT NULL, \n\tid UUID NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tCONSTRAINT pk_source_records PRIMARY KEY (id), \n\tCONSTRAINT uq_source_records_source_native_id_source_revision_raw_sha256 UNIQUE (source, native_id, source_revision, raw_sha256), \n\tCONSTRAINT fk_source_records_work_id_works FOREIGN KEY(work_id) REFERENCES cq_app.works (id)\n)\n\n",
    "\nCREATE TABLE cq_app.work_access (\n\towner_id UUID NOT NULL, \n\twork_id UUID NOT NULL, \n\tacl_epoch INTEGER NOT NULL, \n\trevoked BOOLEAN NOT NULL, \n\tCONSTRAINT pk_work_access PRIMARY KEY (owner_id, work_id), \n\tCONSTRAINT fk_work_access_owner_id_owners FOREIGN KEY(owner_id) REFERENCES cq_app.owners (id), \n\tCONSTRAINT fk_work_access_work_id_works FOREIGN KEY(work_id) REFERENCES cq_app.works (id)\n)\n\n",
    "\nCREATE TABLE cq_app.work_identifiers (\n\tnamespace VARCHAR(40) NOT NULL, \n\tnormalized VARCHAR(512) NOT NULL, \n\twork_id UUID NOT NULL, \n\traw VARCHAR(512) NOT NULL, \n\tnormalizer_version VARCHAR(50) NOT NULL, \n\tCONSTRAINT pk_work_identifiers PRIMARY KEY (namespace, normalized), \n\tCONSTRAINT fk_work_identifiers_work_id_works FOREIGN KEY(work_id) REFERENCES cq_app.works (id)\n)\n\n",
    "CREATE INDEX ix_work_identifiers_work_id ON cq_app.work_identifiers (work_id)",
    "\nCREATE TABLE cq_app.work_versions (\n\twork_id UUID NOT NULL, \n\tsource_sha256 VARCHAR(64) NOT NULL, \n\tversion_label VARCHAR(100), \n\tkind VARCHAR(40) NOT NULL, \n\tid UUID NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tCONSTRAINT pk_work_versions PRIMARY KEY (id), \n\tCONSTRAINT fk_work_versions_work_id_works FOREIGN KEY(work_id) REFERENCES cq_app.works (id)\n)\n\n",
    "CREATE INDEX ix_work_versions_work_id ON cq_app.work_versions (work_id)",
    "\nCREATE TABLE cq_app.files (\n\towner_id UUID NOT NULL, \n\tversion_id UUID, \n\tsecurity_domain VARCHAR(100) NOT NULL, \n\tsha256 VARCHAR(64) NOT NULL, \n\tmedia_type VARCHAR(100) NOT NULL, \n\tstatus VARCHAR(20) NOT NULL, \n\trights_status VARCHAR(20) NOT NULL, \n\toutbound_policy VARCHAR(50) NOT NULL, \n\trevision INTEGER NOT NULL, \n\tdeleted_at TIMESTAMP WITH TIME ZONE, \n\tid UUID NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tCONSTRAINT pk_files PRIMARY KEY (id), \n\tCONSTRAINT fk_files_security_domain_sha256_blobs FOREIGN KEY(security_domain, sha256) REFERENCES cq_app.blobs (security_domain, sha256), \n\tCONSTRAINT fk_files_owner_id_owners FOREIGN KEY(owner_id) REFERENCES cq_app.owners (id), \n\tCONSTRAINT fk_files_version_id_work_versions FOREIGN KEY(version_id) REFERENCES cq_app.work_versions (id)\n)\n\n",
    "CREATE INDEX ix_files_owner_created ON cq_app.files (owner_id, created_at, id)",
    "\nCREATE TABLE cq_app.item_collections (\n\towner_id UUID NOT NULL, \n\titem_id UUID NOT NULL, \n\tcollection_id UUID NOT NULL, \n\tCONSTRAINT pk_item_collections PRIMARY KEY (owner_id, item_id, collection_id), \n\tCONSTRAINT fk_item_collections_owner_id_item_id_library_items FOREIGN KEY(owner_id, item_id) REFERENCES cq_app.library_items (owner_id, id), \n\tCONSTRAINT fk_item_collections_owner_id_collection_id_collections FOREIGN KEY(owner_id, collection_id) REFERENCES cq_app.collections (owner_id, id)\n)\n\n",
    "\nCREATE TABLE cq_app.item_tags (\n\titem_id UUID NOT NULL, \n\ttag VARCHAR(500) NOT NULL, \n\tCONSTRAINT pk_item_tags PRIMARY KEY (item_id, tag), \n\tCONSTRAINT fk_item_tags_item_id_library_items FOREIGN KEY(item_id) REFERENCES cq_app.library_items (id)\n)\n\n",
    "\nCREATE TABLE cq_app.notes (\n\towner_id UUID NOT NULL, \n\tlibrary_item_id UUID NOT NULL, \n\tbody TEXT NOT NULL, \n\trevision INTEGER NOT NULL, \n\tdeleted_at TIMESTAMP WITH TIME ZONE, \n\tid UUID NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tCONSTRAINT pk_notes PRIMARY KEY (id), \n\tCONSTRAINT fk_notes_owner_id_library_item_id_library_items FOREIGN KEY(owner_id, library_item_id) REFERENCES cq_app.library_items (owner_id, id), \n\tCONSTRAINT fk_notes_owner_id_owners FOREIGN KEY(owner_id) REFERENCES cq_app.owners (id)\n)\n\n",
    "CREATE INDEX ix_notes_owner_item ON cq_app.notes (owner_id, library_item_id)",
    "\nCREATE TABLE cq_app.item_files (\n\titem_id UUID NOT NULL, \n\tfile_id UUID NOT NULL, \n\tCONSTRAINT pk_item_files PRIMARY KEY (item_id, file_id), \n\tCONSTRAINT fk_item_files_item_id_library_items FOREIGN KEY(item_id) REFERENCES cq_app.library_items (id), \n\tCONSTRAINT fk_item_files_file_id_files FOREIGN KEY(file_id) REFERENCES cq_app.files (id)\n)\n\n",
]


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    op.execute("DROP TABLE cq_app.item_files")
    op.execute("DROP TABLE cq_app.notes")
    op.execute("DROP TABLE cq_app.item_tags")
    op.execute("DROP TABLE cq_app.item_collections")
    op.execute("DROP TABLE cq_app.files")
    op.execute("DROP TABLE cq_app.work_versions")
    op.execute("DROP TABLE cq_app.work_identifiers")
    op.execute("DROP TABLE cq_app.work_access")
    op.execute("DROP TABLE cq_app.source_records")
    op.execute("DROP TABLE cq_app.projects")
    op.execute("DROP TABLE cq_app.local_sessions")
    op.execute("DROP TABLE cq_app.library_items")
    op.execute("DROP TABLE cq_app.idempotency_keys")
    op.execute("DROP TABLE cq_app.history")
    op.execute("DROP TABLE cq_app.collections")
    op.execute("DROP TABLE cq_app.works")
    op.execute("DROP TABLE cq_app.owners")
    op.execute("DROP TABLE cq_app.outbox")
    op.execute("DROP TABLE cq_app.blobs")
