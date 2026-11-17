"""Task-list privacy and signed-filter pagination without external services."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from citequest.v2.domain import DomainError
from citequest.v2.pagination import Pager
from citequest.workstation_v3.service import WorkstationService
from citequest.workstation_v3.task_listing import task_category


class ReadDatabase:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def execute(self, statement, values):
        self.queries.append((str(statement), values))
        return SimpleNamespace(mappings=lambda: self.rows)


def listing(*, owner=None, rows=None):
    db = ReadDatabase(rows or [])
    catalog = SimpleNamespace(db=db, actor=SimpleNamespace(owner_id=owner or uuid4()), project=lambda _: None)
    return WorkstationService(catalog, Pager(b"task-list-test-key-32-bytes-long!!")), db


def row(**updates):
    created = datetime(2026, 11, 17, tzinfo=UTC)
    return {
        "id": uuid4(), "project_id": uuid4(), "kind": "research", "state": "waiting_input",
        "revision": 17, "created_at": created, "updated_at": created + timedelta(minutes=1),
        "task_profile_id": "research-v6", "research_action": "experiment", **updates,
    }


@pytest.mark.parametrize(("kind", "profile", "action", "expected"), [
    ("qa", "qa-v9", None, "qa"),
    ("summary", "workset-v6", None, "summary"),
    ("research", "research-v6", "experiment", "experiment_design"),
    ("research", "research-v6", "suggest", "research_suggestions"),
    ("research", "research-v6", "extract", "draft_intake"),
    ("research", "research-v6", "plan", "research_plan"),
    ("research", "research-v1", None, "research_plan"),
    ("research", "research-review-v1", "plan", "research_review"),
    ("research", "research-revision-v1", "plan", "research_revision"),
    ("research", "private-profile-data", "private-action-data", "research"),
    ("research", "private-profile-data", "plan", "research"),
    ("research", "private-profile-data", None, "research"),
    ("private-kind-data", "private-profile-data", "private-action-data", "other"),
])
def test_category_is_allowlisted_and_distinguishes_independent_research_work(kind, profile, action, expected):
    assert task_category(kind, profile, action) == expected


def test_list_exposes_only_metadata_not_raw_profile_brief_or_result():
    source = row()
    service, db = listing(rows=[source])
    item = service.tasks()["items"][0]
    assert item["category"] == "experiment_design"
    assert item["updated_at"] == source["updated_at"].isoformat()
    assert set(item) == {"task_id", "project_id", "kind", "category", "state", "revision", "created_at",
                         "updated_at", "detail_path", "artifact_content"}
    sql, values = db.queries[0]
    assert "SELECT t.*" not in sql and "t.result" not in sql
    assert "t.profile->>'profile_id'" in sql and "t.brief->>'research_action'" in sql
    assert "t.owner_id=:owner" in sql and values["owner"] == service.actor.owner_id
    assert "(p.id,p.owner_id)=(t.project_id,t.owner_id)" in sql


@pytest.mark.parametrize(("group", "states"), [
    ("needs_input", {"waiting_input"}),
    ("active", {"queued", "running", "reconciling"}),
    ("completed", {"succeeded"}),
    ("ended", {"partial", "failed", "cancelled"}),
])
def test_state_filter_is_bound_in_database_query_before_pagination(group, states):
    service, db = listing()
    service.tasks(state_group=group, limit=2)
    sql, values = db.queries[0]
    assert sql.index("t.state IN (") < sql.index("ORDER BY") < sql.index("LIMIT")
    assert {value for key, value in values.items() if key.startswith("task_state_")} == states
    assert values["limit"] == 3


def test_filtered_cursor_cannot_be_reused_for_another_filter_owner_project_or_page_size():
    service, db = listing(rows=[row(), row(), row()])
    cursor = service.tasks(state_group="active", limit=2)["next_cursor"]
    assert cursor
    service.tasks(state_group="active", limit=2, cursor=cursor)
    reads = len(db.queries)
    for changes in ({"state_group": "all"}, {"state_group": "needs_input"}, {"limit": 3}, {"project_id": uuid4()}):
        with pytest.raises(DomainError, match="invalid_or_expired_cursor"):
            service.tasks(**{"state_group": "active", "limit": 2, "cursor": cursor, **changes})
    assert len(db.queries) == reads
    other, other_db = listing()
    with pytest.raises(DomainError, match="invalid_or_expired_cursor"):
        other.tasks(state_group="active", limit=2, cursor=cursor)
    assert not other_db.queries


def test_legacy_all_tasks_cursor_remains_valid_only_for_all():
    service, db = listing()
    cursor = service.pager.encode({
        "identity": {"owner": str(service.actor.owner_id), "kind": "workstation_tasks", "project_id": None, "limit": 20},
        "created": datetime.now(UTC).isoformat(), "id": str(uuid4()),
        "expires": (datetime.now(UTC) + timedelta(minutes=5)).timestamp(),
    })
    service.tasks(cursor=cursor)
    assert len(db.queries) == 1
    with pytest.raises(DomainError, match="invalid_or_expired_cursor"):
        service.tasks(cursor=cursor, state_group="active")
    assert len(db.queries) == 1


def test_unknown_filter_is_rejected_without_reading_tasks():
    service, db = listing()
    with pytest.raises(DomainError, match="invalid_task_state_group"):
        service.tasks(state_group="running OR true")
    assert not db.queries
