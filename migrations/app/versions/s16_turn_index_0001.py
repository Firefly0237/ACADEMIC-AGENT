"""Index owner-authorized conversation snapshot lookup by command identity."""
from alembic import op

revision = "s16_turn_index_0001"
down_revision = "s15_chunks_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Non-unique: historical duplicate completions remain detectable by readers.
    op.execute("""CREATE INDEX ix_agent_event_completed_command
        ON cq_app.agent_events(task_id, (payload->>'command_id'))
        WHERE event_type='conversation_turn_completed'""")


def downgrade() -> None:
    op.execute("DROP INDEX cq_app.ix_agent_event_completed_command")
