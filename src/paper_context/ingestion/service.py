"""Shared ingestion service contracts used by the worker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.engine import Connection

from paper_context.queue.contracts import IngestionQueue, IngestQueuePayload
from paper_context.queue.pgmq import PgmqMessage


@dataclass(frozen=True)
class IngestJobContext:
    message: PgmqMessage
    payload: IngestQueuePayload


class LeaseExtender:
    def __init__(
        self,
        connection: Connection,
        queue_adapter: IngestionQueue,
        message: PgmqMessage,
        default_vt_seconds: int,
    ) -> None:
        self._connection = connection
        self._queue_adapter = queue_adapter
        self._message = message
        self._default_vt_seconds = default_vt_seconds

    def extend(self, vt_seconds: int | None = None) -> None:
        vt_seconds = vt_seconds or self._default_vt_seconds
        self._queue_adapter.extend_lease(self._connection, self._message.msg_id, vt_seconds)


class IngestProcessor(Protocol):
    def process(
        self,
        connection: Connection,
        context: IngestJobContext,
        lease: LeaseExtender,
    ) -> None:
        """Process an ingest job context while the lease remains granted."""
        ...


class SyntheticIngestProcessor:
    """Phase 0 worker processor that simulates terminal completion for smoke testing."""

    TERMINAL_STATUSES = {"ready", "failed"}

    def process(
        self,
        connection: Connection,
        context: IngestJobContext,
        lease: LeaseExtender,
    ) -> None:
        row = (
            connection.execute(
                text(
                    """
                SELECT status
                FROM ingest_jobs
                WHERE id = :ingest_job_id
                FOR UPDATE
                """
                ),
                {"ingest_job_id": context.payload.ingest_job_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise LookupError(f"missing ingest job {context.payload.ingest_job_id}")

        if row["status"] in self.TERMINAL_STATUSES:
            return

        now = datetime.now(UTC)
        connection.execute(
            text(
                """
                UPDATE ingest_jobs
                SET status = 'parsing',
                    started_at = COALESCE(started_at, :now)
                WHERE id = :ingest_job_id
                """
            ),
            {"ingest_job_id": context.payload.ingest_job_id, "now": now},
        )
        lease.extend()
        connection.execute(
            text(
                """
                UPDATE ingest_jobs
                SET status = 'ready',
                    finished_at = :now,
                    warnings = COALESCE(warnings, '[]'::jsonb)
                WHERE id = :ingest_job_id
                """
            ),
            {"ingest_job_id": context.payload.ingest_job_id, "now": now},
        )
        connection.execute(
            text(
                """
                UPDATE documents
                SET current_status = 'ready',
                    updated_at = :now
                WHERE id = :document_id
                """
            ),
            {"document_id": context.payload.document_id, "now": now},
        )
