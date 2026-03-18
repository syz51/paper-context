from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from unittest.mock import ANY, MagicMock
from uuid import uuid4

import pytest

from paper_context.queue.contracts import (
    ClaimedIngestMessage,
    IngestionQueue,
    IngestQueuePayload,
    LeaseLostError,
)
from paper_context.queue.pgmq import PgmqMessage
from paper_context.worker.loop import IngestWorker, WorkerConfig

pytestmark = pytest.mark.unit


def test_worker_returns_none_when_queue_empty() -> None:
    queue = MagicMock(spec=IngestionQueue)
    queue.claim_ingest.return_value = None
    processor = MagicMock()
    connection = MagicMock()
    worker = IngestWorker(
        connection_factory=lambda: contextlib.nullcontext(connection),
        queue_adapter=queue,
        processor=processor,
    )

    assert worker.run_once() is None
    processor.process.assert_not_called()
    queue.archive_message.assert_not_called()


def test_worker_processes_and_archives_message() -> None:
    payload = IngestQueuePayload(ingest_job_id=uuid4(), document_id=uuid4())
    message = PgmqMessage(
        msg_id=7,
        read_ct=1,
        enqueued_at=datetime.now(UTC),
        vt=datetime.now(UTC),
        message={
            "ingest_job_id": str(payload.ingest_job_id),
            "document_id": str(payload.document_id),
        },
    )
    queue = MagicMock(spec=IngestionQueue)
    queue.claim_ingest.return_value = ClaimedIngestMessage(message=message, payload=payload)
    processor = MagicMock()
    claim_connection = MagicMock()
    lease_connection = MagicMock()
    processing_connection = MagicMock()
    archive_connection = MagicMock()
    contexts = iter(
        [
            contextlib.nullcontext(claim_connection),
            contextlib.nullcontext(lease_connection),
            contextlib.nullcontext(processing_connection),
            contextlib.nullcontext(archive_connection),
        ]
    )
    worker = IngestWorker(
        connection_factory=lambda: next(contexts),
        queue_adapter=queue,
        processor=processor,
        config=WorkerConfig(vt_seconds=60, max_poll_seconds=2, poll_interval_ms=25),
    )

    handled = worker.run_once()

    assert handled is not None
    queue.claim_ingest.assert_called_once_with(
        claim_connection,
        vt_seconds=60,
        max_poll_seconds=2,
        poll_interval_ms=25,
    )
    queue.extend_lease.assert_called_once_with(lease_connection, 7, 60)
    processor.process.assert_called_once_with(
        processing_connection,
        ANY,
        ANY,
    )
    queue.archive_message.assert_called_once_with(archive_connection, 7)


def test_worker_stops_when_initial_lease_extension_is_lost() -> None:
    payload = IngestQueuePayload(ingest_job_id=uuid4(), document_id=uuid4())
    message = PgmqMessage(
        msg_id=7,
        read_ct=1,
        enqueued_at=datetime.now(UTC),
        vt=datetime.now(UTC),
        message={
            "ingest_job_id": str(payload.ingest_job_id),
            "document_id": str(payload.document_id),
        },
    )
    queue = MagicMock(spec=IngestionQueue)
    queue.claim_ingest.return_value = ClaimedIngestMessage(message=message, payload=payload)
    queue.extend_lease.side_effect = LeaseLostError("lost")
    processor = MagicMock()
    claim_connection = MagicMock()
    lease_connection = MagicMock()
    contexts = iter(
        [
            contextlib.nullcontext(claim_connection),
            contextlib.nullcontext(lease_connection),
        ]
    )
    worker = IngestWorker(
        connection_factory=lambda: next(contexts),
        queue_adapter=queue,
        processor=processor,
    )

    with pytest.raises(LeaseLostError, match="lost"):
        worker.run_once()

    processor.process.assert_not_called()
    queue.archive_message.assert_not_called()
