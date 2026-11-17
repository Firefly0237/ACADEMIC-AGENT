"""Formal S05 material API wiring: shared fail-closed source gate is mandatory."""

from pathlib import Path

from citequest.materials_v2.api import create_app
from citequest.updates_v2.gate import PgSourceGate
from citequest.updates_v2.transport import SourceFetcher
from citequest.v2.storage_config import app_storage
from .locks import WINDOWS_PROFILE


def create_shared_source_app(session_factory, *, gate_engine, workspace: Path, cursor_key: bytes,
                             source_gate_profile: str = WINDOWS_PROFILE):
    store = app_storage(workspace).cas()
    gate = PgSourceGate(gate_engine, workspace / "data/runtime-s05-updates/source-locks", wait_for_slot=True,
        profile_id=source_gate_profile)
    fetcher = SourceFetcher(gate, references=True)
    app = create_app(session_factory, store=store, cursor_key=cursor_key,
                     contract_path=workspace / "contracts/v2/domain.schema.json",
                     policy_path=workspace / "contracts/v2/policy.json", reference_fetcher=fetcher)
    app.state.source_gate_mode = "shared_kernel_lock_and_PG_fail_closed"
    return app
