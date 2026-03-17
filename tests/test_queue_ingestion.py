from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from paper_context.ingestion.queue import IngestionQueueService
from paper_context.ingestion.service import (
    IngestJobContext,
    LeaseExtender,
    SyntheticIngestProcessor,
)
from paper_context.queue.contracts import IngestionQueue, IngestQueuePayload
from paper_context.queue.pgmq import PgmqMessage


def _make_message(payload: Mapping[str, object]) -> PgmqMessage:
    return PgmqMessage(
        msg_id=1,
        read_ct=0,
        enqueued_at=datetime.now(UTC),
        vt=datetime.now(UTC),
        message=dict(payload),
    )


def _make_context() -> IngestJobContext:
    payload = IngestQueuePayload(ingest_job_id=uuid4(), document_id=uuid4())
    return IngestJobContext(message=_make_message({}), payload=payload)


def test_payload_parses_trace_dict() -> None:
    payload = {"ingest_job_id": str(uuid4()), "document_id": str(uuid4()), "trace": {"x": "y"}}
    parsed = IngestQueuePayload.from_message(_make_message(payload))
    assert parsed.trace == payload["trace"]


def test_payload_drops_non_dict_trace() -> None:
    payload = {"ingest_job_id": str(uuid4()), "document_id": str(uuid4()), "trace": "nope"}
    parsed = IngestQueuePayload.from_message(_make_message(payload))
    assert parsed.trace is None


def test_claim_ingest_returns_none_when_empty() -> None:
    queue = IngestionQueue("document_ingest")
    queue._queue = MagicMock()
    queue._queue.read_with_poll.return_value = []

    assert queue.claim_ingest(MagicMock(), 10, 1, poll_interval_ms=1) is None
    queue._queue.read_with_poll.assert_called_once()


def test_claim_ingest_returns_first_message() -> None:
    payload = {"ingest_job_id": str(uuid4()), "document_id": str(uuid4())}
    adapter = MagicMock()
    adapter.read_with_poll.return_value = [_make_message(payload)]
    queue = IngestionQueue("document_ingest")
    queue._queue = adapter

    claimed = queue.claim_ingest(MagicMock(), 3, 1, poll_interval_ms=1)
    assert claimed is not None
    assert str(claimed.payload.document_id) == payload["document_id"]


def test_enqueue_ingest_merges_trace_metadata_and_headers() -> None:
    adapter = MagicMock()
    queue = IngestionQueue("document_ingest")
    queue._queue = adapter
    ingest_job_id = uuid4()
    document_id = uuid4()

    queue.enqueue_ingest(
        MagicMock(),
        ingest_job_id=ingest_job_id,
        document_id=document_id,
        headers={"h": "1"},
        trace_metadata={"t": "2"},
    )

    sent_payload = adapter.send.call_args[0][1]
    assert sent_payload["ingest_job_id"] == str(ingest_job_id)
    assert sent_payload["document_id"] == str(document_id)
    assert sent_payload["trace"] == {"t": "2", "h": "1"}


def test_queue_delegates_extend_archive_delete_metrics() -> None:
    adapter = MagicMock()
    queue = IngestionQueue("document_ingest")
    queue._queue = adapter

    connection = MagicMock()
    queue.extend_lease(connection, 11, 60)
    adapter.set_vt.assert_called_once_with(connection, 11, 60)

    queue.archive_message(connection, 12)
    adapter.archive_message.assert_called_once_with(connection, 12)

    queue.delete_message(connection, 13)
    adapter.delete_message.assert_called_once_with(connection, 13)

    expected_metrics = MagicMock()
    adapter.metrics.return_value = expected_metrics
    assert queue.queue_metrics(connection) is expected_metrics


def test_ingestion_queue_service_enqueues_document_with_trace_headers() -> None:
    engine = MagicMock()
    connection = MagicMock()
    ctx = MagicMock()
    ctx.__enter__.return_value = connection
    ctx.__exit__.return_value = None
    engine.begin.return_value = ctx
    adapter = MagicMock()
    document_id = uuid4()
    ingest_job_id = uuid4()

    with patch(
        "paper_context.ingestion.queue.uuid.uuid4",
        side_effect=[document_id, ingest_job_id],
    ):
        service = IngestionQueueService(engine, adapter)
        returned_document_id, returned_ingest_job_id = service.enqueue_document(
            {"title": "doc"}, trace_headers={"x": "y"}
        )

    assert returned_document_id == document_id
    assert returned_ingest_job_id == ingest_job_id
    adapter.enqueue_ingest.assert_called_once_with(
        connection,
        ingest_job_id,
        document_id,
        headers={"x": "y"},
    )


def test_lease_extender_uses_default_and_overridden_vt() -> None:
    adapter = MagicMock()
    connection = MagicMock()
    message = MagicMock(msg_id=7)
    lease = LeaseExtender(connection, adapter, message, default_vt_seconds=60)

    lease.extend()
    lease.extend(vt_seconds=5)

    adapter.extend_lease.assert_any_call(connection, 7, 60)
    adapter.extend_lease.assert_any_call(connection, 7, 5)


def test_synthetic_processor_returns_on_terminal_status() -> None:
    processor = SyntheticIngestProcessor()
    connection = MagicMock()
    select_result = MagicMock()
    select_result.mappings.return_value.one_or_none.return_value = {"status": "ready"}
    connection.execute.return_value = select_result
    lease = MagicMock()
    context = _make_context()

    processor.process(connection, context, lease)
    assert connection.execute.call_count == 1
    lease.extend.assert_not_called()


def test_synthetic_processor_raises_when_job_missing() -> None:
    processor = SyntheticIngestProcessor()
    connection = MagicMock()
    select_result = MagicMock()
    select_result.mappings.return_value.one_or_none.return_value = None
    connection.execute.return_value = select_result
    lease = MagicMock()
    context = _make_context()

    with pytest.raises(LookupError):
        processor.process(connection, context, lease)


def test_synthetic_processor_updates_non_terminal_job() -> None:
    processor = SyntheticIngestProcessor()
    connection = MagicMock()
    select_result = MagicMock()
    select_result.mappings.return_value.one_or_none.return_value = {"status": "queued"}
    connection.execute.side_effect = [select_result, MagicMock(), MagicMock(), MagicMock()]
    lease = MagicMock()
    context = _make_context()

    processor.process(connection, context, lease)

    assert connection.execute.call_count == 4
    lease.extend.assert_called_once()
    parsing_args = connection.execute.call_args_list[1][0][1]
    assert parsing_args["ingest_job_id"] == context.payload.ingest_job_id
    ready_args = connection.execute.call_args_list[2][0][1]
    assert ready_args["ingest_job_id"] == context.payload.ingest_job_id
    document_args = connection.execute.call_args_list[3][0][1]
    assert document_args["document_id"] == context.payload.document_id
