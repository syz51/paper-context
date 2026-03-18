"""Minimal worker loop that claims queue messages, extends leases, and archives work."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass

from sqlalchemy.engine import Connection

from paper_context.ingestion.service import IngestJobContext, IngestProcessor, LeaseExtender
from paper_context.queue.contracts import ClaimedIngestMessage, IngestionQueue

ConnectionFactory = Callable[[], AbstractContextManager[Connection]]


@dataclass(frozen=True)
class WorkerConfig:
    vt_seconds: int = 60
    max_poll_seconds: int = 5
    poll_interval_ms: int = 250


class IngestWorker:
    def __init__(
        self,
        connection_factory: ConnectionFactory,
        queue_adapter: IngestionQueue,
        processor: IngestProcessor,
        config: WorkerConfig | None = None,
    ) -> None:
        self._connection_factory = connection_factory
        self._queue_adapter = queue_adapter
        self._processor = processor
        self._config = config or WorkerConfig()

    def run_once(self) -> ClaimedIngestMessage | None:
        with self._connection_factory() as claim_connection:
            task = self._queue_adapter.claim_ingest(
                claim_connection,
                vt_seconds=self._config.vt_seconds,
                max_poll_seconds=self._config.max_poll_seconds,
                poll_interval_ms=self._config.poll_interval_ms,
            )
        if task is None:
            return None

        lease = LeaseExtender(
            self._connection_factory,
            self._queue_adapter,
            task.message,
            self._config.vt_seconds,
        )
        lease.extend()
        with self._connection_factory() as processing_connection:
            self._processor.process(
                processing_connection,
                IngestJobContext(message=task.message, payload=task.payload),
                lease,
            )
        with self._connection_factory() as archive_connection:
            self._queue_adapter.archive_message(archive_connection, task.message.msg_id)
        return task
