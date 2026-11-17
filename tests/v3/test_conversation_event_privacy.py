"""The private answer snapshots must not enter the ordinary event/SSE API."""
from contextlib import contextmanager
from uuid import uuid4

from citequest.agents_v3.contracts import RuntimeLimits
from citequest.agents_v3.database import TaskService, public_event


def test_private_turn_event_has_a_default_deny_public_projection():
    private = {"seq":12,"event_type":"conversation_turn_completed","created_at":"fixture",
        "payload":{"version":"citequest-conversation-turn/1","command_seq":3,"brief_revision":3,
            "state":"partial","error_code":None,"command_id":"private-command",
            "brief":{"question":"PRIVATE_QUESTION"},"answer":{"quote":"PRIVATE_SOURCE"},
            "future_private_field":"PRIVATE_FUTURE"}}
    projected = public_event(private)
    assert projected["payload"] == {"version":"citequest-conversation-turn/1","command_seq":3,
        "brief_revision":3,"state":"partial","error_code":None}
    assert private["payload"]["answer"]["quote"] == "PRIVATE_SOURCE"
    assert projected["seq"] == private["seq"]


def test_task_service_events_always_filters_snapshot_payload():
    private = {"seq":1,"event_type":"conversation_turn_completed","created_at":"fixture",
        "payload":{"version":"citequest-conversation-turn/1","command_seq":1,"brief_revision":1,
            "state":"succeeded","error_code":None,"brief":{"question":"PRIVATE_QUESTION"},
            "answer":{"text":"PRIVATE_ANSWER"}}}
    ordinary = {"seq":2,"event_type":"task_succeeded","payload":{"state":"succeeded"},"created_at":"fixture"}
    class Cursor:
        def fetchone(self):
            return {"id":uuid4()}
        def fetchall(self):
            return [private,ordinary]
    class Connection:
        def execute(self,*args):
            return Cursor()
    class Database:
        @contextmanager
        def connect(self):
            yield Connection()
    service = TaskService(Database(),RuntimeLimits(mode="bounded"))
    result = service.events(uuid4(),uuid4())
    assert "PRIVATE_" not in str(result)
    assert result[1] == ordinary
    assert [row["seq"] for row in result] == [1,2]
