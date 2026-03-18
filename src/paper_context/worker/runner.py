from __future__ import annotations

import time
from typing import Any

from sqlalchemy import text

from paper_context.config import get_settings
from paper_context.db.engine import get_engine
from paper_context.db.session import connection_scope
from paper_context.ingestion.enrichment import NullMetadataEnricher
from paper_context.ingestion.parsers import DoclingPdfParser, PdfPlumberPdfParser
from paper_context.ingestion.queue import IngestionQueueService
from paper_context.ingestion.service import DeterministicIngestProcessor, SyntheticIngestProcessor
from paper_context.queue.contracts import IngestionQueue
from paper_context.retrieval import (
    DeterministicEmbeddingClient,
    DocumentRetrievalIndexer,
    HeuristicRerankerClient,
    VoyageEmbeddingClient,
    ZeroEntropyRerankerClient,
)
from paper_context.storage.local_fs import LocalFilesystemStorage

from .loop import IngestWorker, WorkerConfig


def build_worker() -> IngestWorker:
    settings = get_settings()
    queue = IngestionQueue(settings.queue.name)
    storage = LocalFilesystemStorage(settings.storage.root_path)
    storage.ensure_root()
    voyage_api_key = getattr(settings.providers, "voyage_api_key", None)
    zero_entropy_api_key = getattr(settings.providers, "zero_entropy_api_key", None)
    embedding_client = (
        VoyageEmbeddingClient(
            api_key=voyage_api_key,
            model=settings.providers.voyage_model,
        )
        if voyage_api_key
        else DeterministicEmbeddingClient(model=settings.providers.voyage_model)
    )
    reranker_client = (
        ZeroEntropyRerankerClient(
            api_key=zero_entropy_api_key,
            model=settings.providers.reranker_model,
        )
        if zero_entropy_api_key
        else HeuristicRerankerClient(model=settings.providers.reranker_model)
    )
    return IngestWorker(
        connection_factory=lambda: connection_scope(get_engine()),
        queue_adapter=queue,
        processor=DeterministicIngestProcessor(
            storage=storage,
            primary_parser=DoclingPdfParser(),
            fallback_parser=PdfPlumberPdfParser(),
            metadata_enricher=NullMetadataEnricher(),
            index_version=settings.providers.index_version,
            chunking_version=settings.chunking.version,
            embedding_model=settings.providers.voyage_model,
            reranker_model=settings.providers.reranker_model,
            min_tokens=settings.chunking.min_tokens,
            max_tokens=settings.chunking.max_tokens,
            overlap_fraction=settings.chunking.overlap_fraction,
            retrieval_indexer=DocumentRetrievalIndexer(
                index_version=settings.providers.index_version,
                chunking_version=settings.chunking.version,
                embedding_model=settings.providers.voyage_model,
                reranker_model=settings.providers.reranker_model,
                embedding_client=embedding_client,
                reranker_client=reranker_client,
            ),
        ),
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
    worker = IngestWorker(
        connection_factory=lambda: connection_scope(engine),
        queue_adapter=queue,
        processor=SyntheticIngestProcessor(),
        config=WorkerConfig(
            vt_seconds=settings.queue.visibility_timeout_seconds,
            max_poll_seconds=settings.queue.max_poll_seconds,
            poll_interval_ms=settings.queue.poll_interval_ms,
        ),
    )
    handled = worker.run_once()
    with engine.begin() as connection:
        ingest_job = (
            connection.execute(
                text(
                    """
                SELECT status, started_at, finished_at
                FROM ingest_jobs
                WHERE id = :ingest_job_id
                """
                ),
                {"ingest_job_id": ingest_job_id},
            )
            .mappings()
            .one()
        )
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
