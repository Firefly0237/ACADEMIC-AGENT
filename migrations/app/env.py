"""Official Alembic environment, cq_app only; no checkpoint/memory reflection."""

import os

from alembic import context

from citequest.data_v2 import application_members as application_members
from citequest.data_v2 import models as data_models  # noqa: F401
from citequest.index_v2 import models as index_models  # noqa: F401
from citequest.index_v3 import models as backend_models  # noqa: F401
from citequest.materials_v2 import models as material_models  # noqa: F401
from citequest.updates_v2 import models as update_models  # noqa: F401
from citequest.v2.db import engine
from citequest.v2.models import Base
from citequest.web_v3 import models as web_models  # noqa: F401

config = context.config
url = os.environ.get("CITEQUEST_S03_MIGRATION_URL")


def include_name(name: str | None, type_: str, parent_names: dict[str, str | None]) -> bool:
    if type_ == "schema":
        return name == "cq_app"
    return parent_names.get("schema_name") == "cq_app"


options = {
    "target_metadata": Base.metadata,
    "version_table_schema": "cq_app",
    "include_schemas": True,
    "include_name": include_name,
    "compare_type": True,
}
if context.is_offline_mode():
    if url is None:
        raise RuntimeError("explicit_offline_migration_url_required")
    context.configure(url=url, literal_binds=True, **options)
    with context.begin_transaction():
        context.run_migrations()
else:
    supplied = config.attributes.get("connection")
    if supplied is not None:
        # Official Alembic shared-connection recipe: the management caller owns
        # the transaction, including exact post-DDL grants and verification.
        if not supplied.in_transaction():
            raise RuntimeError("external_migration_connection_requires_transaction")
        if not supplied.exec_driver_sql("SELECT 1 FROM pg_namespace WHERE nspname='cq_app'").scalar():
            raise RuntimeError("S03 schema must be provisioned by the separate admin bootstrap")
        context.configure(connection=supplied, **options)
        with context.begin_transaction():
            context.run_migrations()
    else:
        if url is None:
            raise RuntimeError("explicit_online_migration_url_required")
        db = engine(url)
        try:
            with db.connect() as connection:
                if not connection.exec_driver_sql("SELECT 1 FROM pg_namespace WHERE nspname='cq_app'").scalar():
                    raise RuntimeError("S03 schema must be provisioned by the separate admin bootstrap")
                connection.commit()
                context.configure(connection=connection, **options)
                with context.begin_transaction():
                    context.run_migrations()
        finally:
            db.dispose()
