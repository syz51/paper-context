from __future__ import annotations

import time

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


def test_pgmq_adapter_redelivers_message_after_visibility_timeout_expires(
    migrated_postgres_engine,
    unique_queue_name: str,
) -> None:
    adapter = PgmqAdapter(unique_queue_name)

    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"),
            {"queue_name": unique_queue_name},
        )
        message_id = adapter.send(connection, {"kind": "redeliver-me"})

    with migrated_postgres_engine.begin() as connection:
        first_claim = adapter.read_with_poll(
            connection,
            vt_seconds=2,
            max_poll_seconds=1,
            poll_interval_ms=10,
        )

    with migrated_postgres_engine.begin() as connection:
        hidden_claim = adapter.read_with_poll(
            connection,
            vt_seconds=2,
            max_poll_seconds=1,
            poll_interval_ms=10,
        )

    time.sleep(2.2)

    with migrated_postgres_engine.begin() as connection:
        redelivered_claim = adapter.read_with_poll(
            connection,
            vt_seconds=2,
            max_poll_seconds=1,
            poll_interval_ms=10,
        )
        archived = adapter.archive_message(connection, message_id)

    assert [message.msg_id for message in first_claim] == [message_id]
    assert first_claim[0].message == {"kind": "redeliver-me"}
    assert hidden_claim == []
    assert [message.msg_id for message in redelivered_claim] == [message_id]
    assert redelivered_claim[0].message == {"kind": "redeliver-me"}
    assert redelivered_claim[0].read_ct > first_claim[0].read_ct
    assert archived is True


def test_pgmq_claim_stays_hidden_until_open_claim_transaction_ends(
    migrated_postgres_engine,
    unique_queue_name: str,
) -> None:
    adapter = PgmqAdapter(unique_queue_name)

    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"),
            {"queue_name": unique_queue_name},
        )
        message_id = adapter.send(connection, {"kind": "held-open"})

    claim_connection = migrated_postgres_engine.connect()
    claim_transaction = claim_connection.begin()
    try:
        first_claim = adapter.read_with_poll(
            claim_connection,
            vt_seconds=1,
            max_poll_seconds=1,
            poll_interval_ms=10,
        )

        time.sleep(1.2)

        with migrated_postgres_engine.begin() as connection:
            hidden_claim = adapter.read_with_poll(
                connection,
                vt_seconds=1,
                max_poll_seconds=1,
                poll_interval_ms=10,
            )

        assert [message.msg_id for message in first_claim] == [message_id]
        assert hidden_claim == []
    finally:
        claim_transaction.rollback()
        claim_connection.close()

    with migrated_postgres_engine.begin() as connection:
        redelivered_claim = adapter.read_with_poll(
            connection,
            vt_seconds=1,
            max_poll_seconds=1,
            poll_interval_ms=10,
        )
        archived = adapter.archive_message(connection, message_id)

    assert [message.msg_id for message in redelivered_claim] == [message_id]
    assert archived is True


def test_pgmq_claim_stays_hidden_until_claim_transaction_exits(
    migrated_postgres_engine,
    unique_queue_name: str,
) -> None:
    adapter = PgmqAdapter(unique_queue_name)

    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"),
            {"queue_name": unique_queue_name},
        )
        message_id = adapter.send(connection, {"kind": "tx-bound-claim"})

    claim_connection = migrated_postgres_engine.connect()
    claim_transaction = claim_connection.begin()
    try:
        claimed = adapter.read_with_poll(
            claim_connection,
            vt_seconds=2,
            max_poll_seconds=1,
            poll_interval_ms=10,
        )
        time.sleep(2.2)
        with migrated_postgres_engine.begin() as connection:
            hidden_claim = adapter.read_with_poll(
                connection,
                vt_seconds=2,
                max_poll_seconds=1,
                poll_interval_ms=10,
            )
        assert [message.msg_id for message in claimed] == [message_id]
        assert hidden_claim == []
    finally:
        claim_transaction.rollback()
        claim_connection.close()

    with migrated_postgres_engine.begin() as connection:
        redelivered = adapter.read_with_poll(
            connection,
            vt_seconds=2,
            max_poll_seconds=1,
            poll_interval_ms=10,
        )
        archived = adapter.archive_message(connection, message_id)

    assert [message.msg_id for message in redelivered] == [message_id]
    assert archived is True
