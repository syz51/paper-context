from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy.engine import Connection

from .pgmq import PgmqAdapter, PgmqMessage, QueueMetrics


class LeaseLostError(RuntimeError):
    """Raised when a claimed queue message can no longer be extended."""


@dataclass(frozen=True)
class IngestQueuePayload:
    ingest_job_id: UUID
    document_id: UUID
    trace: dict[str, Any] | None = None

    @classmethod
    def from_message(cls, message: PgmqMessage) -> IngestQueuePayload:
        trace = message.message.get("trace")
        return cls(
            ingest_job_id=UUID(str(message.message["ingest_job_id"])),
            document_id=UUID(str(message.message["document_id"])),
            trace=trace if isinstance(trace, dict) else None,
        )


@dataclass(frozen=True)
class ClaimedIngestMessage:
    message: PgmqMessage
    payload: IngestQueuePayload


class IngestionQueue:
    def __init__(self, queue_name: str) -> None:
        self._queue = PgmqAdapter(queue_name)

    def enqueue_ingest(
        self,
        conn: Connection,
        ingest_job_id: UUID,
        document_id: UUID,
        headers: Mapping[str, str] | None = None,
        trace_metadata: Mapping[str, str] | None = None,
        delay_seconds: int = 0,
    ) -> int:
        payload: dict[str, str | dict[str, str]] = {
            "ingest_job_id": str(ingest_job_id),
            "document_id": str(document_id),
        }
        if trace_metadata or headers:
            payload["trace"] = {**dict(trace_metadata or {}), **dict(headers or {})}
        return self._queue.send(conn, payload, delay_seconds=delay_seconds)

    def claim_ingest(
        self,
        conn: Connection,
        vt_seconds: int,
        max_poll_seconds: int,
        poll_interval_ms: int = 100,
    ) -> ClaimedIngestMessage | None:
        messages = self._queue.read_with_poll(
            conn,
            vt_seconds=vt_seconds,
            max_poll_seconds=max_poll_seconds,
            poll_interval_ms=poll_interval_ms,
            qty=1,
        )
        if not messages:
            return None
        message = messages[0]
        return ClaimedIngestMessage(
            message=message, payload=IngestQueuePayload.from_message(message)
        )

    def extend_lease(self, conn: Connection, msg_id: int, vt_seconds: int) -> None:
        if self._queue.set_vt(conn, msg_id, vt_seconds) is None:
            raise LeaseLostError(
                f"queue lease for message {msg_id} was lost before it could be extended"
            )

    def archive_message(self, conn: Connection, message_id: int) -> None:
        self._queue.archive_message(conn, message_id)

    def delete_message(self, conn: Connection, message_id: int) -> None:
        self._queue.delete_message(conn, message_id)

    def queue_metrics(self, conn: Connection) -> QueueMetrics:
        return self._queue.metrics(conn)
