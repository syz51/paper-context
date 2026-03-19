from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.engine import Connection

from paper_context.models import IngestJob

from .pgmq import PgmqAdapter, PgmqMessage, QueueMetrics

_TERMINAL_INGEST_JOB_STATUSES = frozenset({"ready", "failed"})


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
    already_archived: bool = False


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
        deadline = time.monotonic() + max_poll_seconds
        remaining_poll_seconds = max_poll_seconds
        archived_terminal_message: ClaimedIngestMessage | None = None
        while True:
            messages = self._queue.read_with_poll(
                conn,
                vt_seconds=vt_seconds,
                max_poll_seconds=max(1, remaining_poll_seconds),
                poll_interval_ms=poll_interval_ms,
                qty=1,
            )
            if not messages:
                return archived_terminal_message
            message = messages[0]
            payload = IngestQueuePayload.from_message(message)
            if self._ingest_job_is_terminal(conn, ingest_job_id=payload.ingest_job_id):
                self.archive_message(conn, message.msg_id)
                archived_terminal_message = ClaimedIngestMessage(
                    message=message,
                    payload=payload,
                    already_archived=True,
                )
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    return archived_terminal_message
                remaining_poll_seconds = max(1, math.ceil(remaining_seconds))
                continue
            return ClaimedIngestMessage(message=message, payload=payload)

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

    def _ingest_job_is_terminal(self, conn: Connection, *, ingest_job_id: UUID) -> bool:
        row = (
            conn.execute(select(IngestJob.status).where(IngestJob.id == ingest_job_id))
            .mappings()
            .one_or_none()
        )
        if row is None:
            return True
        return str(row["status"]) in _TERMINAL_INGEST_JOB_STATUSES
