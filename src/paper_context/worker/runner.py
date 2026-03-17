from __future__ import annotations

import time
from typing import Any

from sqlalchemy import text

from paper_context.config import get_settings
from paper_context.db.engine import get_engine
from paper_context.db.session import connection_scope
from paper_context.ingestion.queue import IngestionQueueService
from paper_context.ingestion.service import SyntheticIngestProcessor
from paper_context.queue.contracts import IngestionQueue

from .loop import IngestWorker, WorkerConfig


def build_worker() -> IngestWorker:
    settings = get_settings()
    queue = IngestionQueue(settings.queue.name)
    return IngestWorker(
        connection_factory=lambda: connection_scope(get_engine()),
        queue_adapter=queue,
        processor=SyntheticIngestProcessor(),
        config=WorkerConfig(
            vt_seconds=settings.queue.visibility_timeout_seconds,
            max_poll_seconds=settings.queue.max_poll_seconds,
            poll_interval_ms=settings.queue.poll_interval_ms,
        ),
    )


def run_worker(*, once: bool = False) -> None:
    settings = get_settings()
    worker = build_worker()
    while True:
        handled = worker.run_once()
        if once:
            return
        if handled is None:
            time.sleep(settings.runtime.worker_idle_sleep_seconds)


def run_synthetic_job_verification() -> dict[str, Any]:
    settings = get_settings()
    engine = get_engine()
    queue = IngestionQueue(settings.queue.name)
    service = IngestionQueueService(engine, queue)
    document_id, ingest_job_id = service.enqueue_document(
        {
            "title": "Phase 0 synthetic ingest job",
            "source_type": "synthetic",
            "trigger": "synthetic",
        }
    )
    worker = build_worker()
    handled = worker.run_once()
    with engine.begin() as connection:
        ingest_job = connection.execute(
            text(
                """
                SELECT status, started_at, finished_at
                FROM ingest_jobs
                WHERE id = :ingest_job_id
                """
            ),
            {"ingest_job_id": ingest_job_id},
        ).mappings().one()
        metrics = queue.queue_metrics(connection)
    return {
        "document_id": str(document_id),
        "ingest_job_id": str(ingest_job_id),
        "handled_message": handled is not None,
        "job_status": ingest_job["status"],
        "started_at": ingest_job["started_at"],
        "finished_at": ingest_job["finished_at"],
        "queue_metrics": {
            "queue_name": metrics.queue_name,
            "queue_length": metrics.queue_length,
            "total_messages": metrics.total_messages,
        },
    }
