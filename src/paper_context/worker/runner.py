from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy import select

from paper_context.config import get_settings
from paper_context.db.engine import get_engine
from paper_context.db.session import connection_scope
from paper_context.ingestion.enrichment import NullMetadataEnricher
from paper_context.ingestion.parser_isolation import ParserIsolationConfig, build_pdf_parser
from paper_context.ingestion.queue import IngestionQueueService
from paper_context.ingestion.service import DeterministicIngestProcessor, SyntheticIngestProcessor
from paper_context.models import IngestJob
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

logger = logging.getLogger(__name__)


def build_worker() -> IngestWorker:
    settings = get_settings()
    queue = IngestionQueue(settings.queue.name)
    storage = LocalFilesystemStorage(settings.storage.root_path)
    storage.ensure_root()
    parser_settings = getattr(settings, "parser", None)
    parser_isolated = getattr(parser_settings, "execution_mode", "subprocess") == "subprocess"
    parser_config = ParserIsolationConfig(
        timeout_seconds=getattr(parser_settings, "timeout_seconds", 120),
        memory_limit_mb=getattr(parser_settings, "memory_limit_mb", 2_048),
        output_limit_mb=getattr(parser_settings, "output_limit_mb", 32),
    )
    primary_parser_name = getattr(parser_settings, "primary", "docling")
    fallback_parser_name = getattr(parser_settings, "fallback", "pdfplumber")
    voyage_api_key = getattr(settings.providers, "voyage_api_key", None)
    zero_entropy_api_key = getattr(settings.providers, "zero_entropy_api_key", None)
    if not voyage_api_key:
        logger.warning("voyage_api_key not configured; using deterministic embedding client")
    if not zero_entropy_api_key:
        logger.warning("zero_entropy_api_key not configured; using heuristic reranker")
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
            primary_parser=build_pdf_parser(
                primary_parser_name,
                isolated=parser_isolated,
                config=parser_config,
            ),
            fallback_parser=build_pdf_parser(
                fallback_parser_name,
                isolated=parser_isolated,
                config=parser_config,
            ),
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
    logger.info(
        "worker started",
        extra={
            "structured_data": {
                "event": "worker.started",
                "once": once,
                "queue_name": settings.queue.name,
                "vt_seconds": settings.queue.visibility_timeout_seconds,
            }
        },
    )
    while True:
        try:
            handled = worker.run_once()
        except Exception:
            if once:
                raise
            logger.exception("worker poll failed; continuing after backoff")
            time.sleep(settings.runtime.worker_idle_sleep_seconds)
            continue
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
                select(
                    IngestJob.status,
                    IngestJob.started_at,
                    IngestJob.finished_at,
                ).where(IngestJob.id == ingest_job_id)
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
            "queue_visible_length": metrics.queue_visible_length,
            "total_messages": metrics.total_messages,
        },
    }
