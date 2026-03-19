from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from paper_context.queue.pgmq import PgmqAdapter, QueueMetrics

pytestmark = pytest.mark.unit


def _message_row(**overrides: object) -> dict[str, object]:
    base = {
        "msg_id": 7,
        "read_ct": 1,
        "enqueued_at": datetime.now(UTC),
        "vt": datetime.now(UTC),
        "message": {"a": 1},
    }
    base.update(overrides)
    return base


def _metrics_row() -> dict[str, object]:
    return {
        "queue_name": "document_ingest",
        "queue_length": 1,
        "queue_visible_length": 1,
        "newest_msg_age_sec": 0,
        "oldest_msg_age_sec": 0,
        "total_messages": 1,
        "scrape_time": datetime.now(UTC),
    }


def test_send_returns_integer_message_id() -> None:
    connection = MagicMock()
    result = MagicMock()
    result.scalar_one.return_value = "42"
    connection.execute.return_value = result
    adapter = PgmqAdapter("document_ingest")

    assert adapter.send(connection, {"foo": "bar"}, delay_seconds=3) == 42
    connection.execute.assert_called_once()
    params = connection.execute.call_args[0][1]
    assert params["delay_seconds"] == 3


def test_read_with_poll_parses_json_string_messages() -> None:
    connection = MagicMock()
    result = MagicMock()
    mappings = MagicMock()
    row = _message_row(message=json.dumps({"foo": "bar"}))
    mappings.all.return_value = [row]
    result.mappings.return_value = mappings
    connection.execute.return_value = result
    adapter = PgmqAdapter("document_ingest")

    messages = adapter.read_with_poll(
        connection,
        vt_seconds=5,
        max_poll_seconds=1,
        poll_interval_ms=10,
    )

    assert len(messages) == 1
    assert isinstance(messages[0].message, dict)
    assert messages[0].message["foo"] == "bar"


def test_row_to_message_accepts_dict_payloads() -> None:
    adapter = PgmqAdapter("document_ingest")
    row = _message_row(message={"foo": "bar"})

    message = adapter._row_to_message(row)
    assert message.message == row["message"]
    assert message.msg_id == row["msg_id"]


def test_set_vt_returns_message_or_none() -> None:
    connection = MagicMock()
    adapter = PgmqAdapter("document_ingest")
    result = MagicMock()
    mappings = MagicMock()
    row = _message_row()
    mappings.one_or_none.return_value = row
    result.mappings.return_value = mappings
    connection.execute.return_value = result

    assert adapter.set_vt(connection, 1, 30) is not None
    mappings.one_or_none.return_value = None
    assert adapter.set_vt(connection, 1, 30) is None


def test_archive_and_delete_return_bool_flags() -> None:
    connection = MagicMock()
    adapter = PgmqAdapter("document_ingest")
    archive_result = MagicMock()
    delete_result = MagicMock()
    archive_result.scalar_one.return_value = 1
    delete_result.scalar_one.return_value = 0
    connection.execute.side_effect = [archive_result, delete_result]

    assert adapter.archive_message(connection, 1) is True
    assert adapter.delete_message(connection, 1) is False


def test_delete_messages_for_ingest_job_id_returns_deleted_message_ids() -> None:
    connection = MagicMock()
    adapter = PgmqAdapter("document_ingest")
    result = MagicMock()
    scalars = MagicMock()
    ingest_job_id = uuid4()
    scalars.all.return_value = [11, 12]
    result.scalars.return_value = scalars
    connection.execute.return_value = result

    deleted_message_ids = adapter.delete_messages_for_ingest_job_id(connection, ingest_job_id)

    assert deleted_message_ids == [11, 12]
    stmt = connection.execute.call_args.args[0]
    assert "DELETE FROM pgmq.q_document_ingest" in str(stmt)


def test_queue_metrics_returns_struct() -> None:
    connection = MagicMock()
    adapter = PgmqAdapter("document_ingest")
    result = MagicMock()
    mappings = MagicMock()
    metrics_row = _metrics_row()
    mappings.one.return_value = metrics_row
    result.mappings.return_value = mappings
    connection.execute.return_value = result

    metrics = adapter.metrics(connection)
    assert isinstance(metrics, QueueMetrics)
    assert metrics.queue_name == metrics_row["queue_name"]
    assert metrics.total_messages == metrics_row["total_messages"]
