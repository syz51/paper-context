from __future__ import annotations

import pytest
from sqlalchemy import text

from paper_context.queue.pgmq import PgmqAdapter

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_postgres,
    pytest.mark.slow,
]


def test_pgmq_adapter_round_trip_against_real_postgres(
    migrated_postgres_engine,
    unique_queue_name: str,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"),
            {"queue_name": unique_queue_name},
        )
        adapter = PgmqAdapter(unique_queue_name)

        first_message_id = adapter.send(connection, {"kind": "archive-me"})
        second_message_id = adapter.send(connection, {"kind": "delete-me"})

        claimed = adapter.read_with_poll(
            connection,
            vt_seconds=30,
            max_poll_seconds=1,
            poll_interval_ms=10,
        )
        updated = adapter.set_vt(connection, first_message_id, 60)
        metrics = adapter.metrics(connection)
        archived = adapter.archive_message(connection, first_message_id)
        deleted = adapter.delete_message(connection, second_message_id)

    assert [message.msg_id for message in claimed] == [first_message_id]
    assert claimed[0].message == {"kind": "archive-me"}
    assert updated is not None
    assert updated.msg_id == first_message_id
    assert metrics.queue_name == unique_queue_name
    assert metrics.total_messages == 2
    assert archived is True
    assert deleted is True
