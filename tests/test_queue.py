from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

from paper_context.queue.contracts import IngestionQueue, IngestQueuePayload
from paper_context.queue.pgmq import PgmqAdapter, PgmqMessage


def test_payload_parses_message() -> None:
    payload = {"ingest_job_id": str(uuid4()), "document_id": str(uuid4())}
    message = PgmqMessage(
        msg_id=1,
        read_ct=0,
        enqueued_at=datetime.now(UTC),
        vt=datetime.now(UTC),
        message=payload,
    )

    parsed = IngestQueuePayload.from_message(message)

    assert str(parsed.ingest_job_id) == payload["ingest_job_id"]
    assert str(parsed.document_id) == payload["document_id"]


def test_claim_ingest_returns_first_message() -> None:
    adapter = MagicMock(spec=PgmqAdapter)
    payload = {"ingest_job_id": str(uuid4()), "document_id": str(uuid4())}
    message = PgmqMessage(
        msg_id=11,
        read_ct=1,
        enqueued_at=datetime.now(UTC),
        vt=datetime.now(UTC),
        message=payload,
    )
    adapter.read_with_poll.return_value = [message]
    queue = IngestionQueue("document_ingest")
    queue._queue = adapter

    claimed = queue.claim_ingest(
        MagicMock(), vt_seconds=60, max_poll_seconds=3, poll_interval_ms=100
    )

    assert claimed is not None
    assert claimed.message.msg_id == 11
    assert str(claimed.payload.ingest_job_id) == payload["ingest_job_id"]
