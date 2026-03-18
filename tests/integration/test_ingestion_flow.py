from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from paper_context.api.app import create_app
from paper_context.config import get_settings
from paper_context.db.engine import dispose_engine, get_engine
from paper_context.db.session import connection_scope
from paper_context.ingestion.queue import IngestionQueueService
from paper_context.ingestion.service import SyntheticIngestProcessor
from paper_context.queue.contracts import IngestionQueue
from paper_context.worker.loop import IngestWorker, WorkerConfig
from paper_context.worker.runner import build_worker

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_postgres,
    pytest.mark.slow,
]


class RecordingIngestionQueue(IngestionQueue):
    def __init__(self, queue_name: str) -> None:
        super().__init__(queue_name)
        self.extended: list[tuple[int, int]] = []
        self.archived: list[int] = []

    def extend_lease(self, conn, msg_id: int, vt_seconds: int) -> None:
        self.extended.append((msg_id, vt_seconds))
        super().extend_lease(conn, msg_id, vt_seconds)

    def archive_message(self, conn, message_id: int) -> None:
        self.archived.append(message_id)
        super().archive_message(conn, message_id)


def _live_pdf_path() -> Path:
    configured = os.environ.get("PAPER_CONTEXT_E2E_PDF_PATH", "~/Downloads/attention.pdf")
    pdf_path = Path(configured).expanduser()
    if not pdf_path.is_file():
        pytest.skip(f"live PDF fixture not found at {pdf_path}")
    return pdf_path


def test_readiness_reports_ready_against_real_postgres(
    monkeypatch: pytest.MonkeyPatch,
    migrated_postgres_url: str,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PAPER_CONTEXT_DATABASE__URL", migrated_postgres_url)
    monkeypatch.setenv("PAPER_CONTEXT_STORAGE__ROOT_PATH", str(tmp_path / "artifacts"))
    get_settings.cache_clear()
    get_engine.cache_clear()

    try:
        with TestClient(create_app()) as client:
            response = client.get("/readyz")
    finally:
        dispose_engine()
        get_settings.cache_clear()

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ready"
    assert payload["database_ready"] is True


def test_ingestion_service_and_worker_round_trip_against_real_postgres(
    migrated_postgres_engine,
    unique_queue_name: str,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"),
            {"queue_name": unique_queue_name},
        )

    queue = RecordingIngestionQueue(unique_queue_name)
    service = IngestionQueueService(migrated_postgres_engine, queue)
    worker = IngestWorker(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        queue_adapter=queue,
        processor=SyntheticIngestProcessor(),
        config=WorkerConfig(vt_seconds=45, max_poll_seconds=1, poll_interval_ms=10),
    )

    document_id, ingest_job_id = service.enqueue_document(
        {"title": "Integration paper", "source_type": "upload"},
        trace_headers={"x-trace-id": "integration"},
    )
    handled = worker.run_once()

    assert handled is not None
    assert handled.payload.document_id == document_id
    assert handled.payload.ingest_job_id == ingest_job_id
    assert handled.payload.trace == {"x-trace-id": "integration"}
    assert queue.extended == [(handled.message.msg_id, 45), (handled.message.msg_id, 45)]
    assert queue.archived == [handled.message.msg_id]

    with migrated_postgres_engine.begin() as connection:
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
        document = (
            connection.execute(
                text(
                    """
                    SELECT current_status
                    FROM documents
                    WHERE id = :document_id
                    """
                ),
                {"document_id": document_id},
            )
            .mappings()
            .one()
        )
        metrics = queue.queue_metrics(connection)

    assert ingest_job["status"] == "ready"
    assert ingest_job["started_at"] is not None
    assert ingest_job["finished_at"] is not None
    assert document["current_status"] == "ready"
    assert metrics.queue_length == 0
    assert metrics.total_messages == 1


def test_enqueue_document_rolls_back_when_queue_write_fails(
    migrated_postgres_engine,
    unique_queue_name: str,
) -> None:
    service = IngestionQueueService(migrated_postgres_engine, IngestionQueue(unique_queue_name))

    with pytest.raises(SQLAlchemyError):
        service.enqueue_document({"title": "will rollback"})

    with migrated_postgres_engine.begin() as connection:
        document_count = connection.execute(text("SELECT COUNT(*) FROM documents")).scalar_one()
        ingest_job_count = connection.execute(text("SELECT COUNT(*) FROM ingest_jobs")).scalar_one()

    assert document_count == 0
    assert ingest_job_count == 0


def test_documents_upload_and_worker_round_trip_against_real_postgres_with_live_pdf(
    monkeypatch: pytest.MonkeyPatch,
    migrated_postgres_engine,
    migrated_postgres_url: str,
    unique_queue_name: str,
    tmp_path: Path,
) -> None:
    pdf_path = _live_pdf_path()
    storage_root = tmp_path / "artifacts"

    monkeypatch.setenv("PAPER_CONTEXT_DATABASE__URL", migrated_postgres_url)
    monkeypatch.setenv("PAPER_CONTEXT_STORAGE__ROOT_PATH", str(storage_root))
    monkeypatch.setenv("PAPER_CONTEXT_QUEUE__NAME", unique_queue_name)
    get_settings.cache_clear()
    get_engine.cache_clear()

    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"),
            {"queue_name": unique_queue_name},
        )

    try:
        with TestClient(create_app()) as client:
            response = client.post(
                "/documents",
                data={"title": "Attention Is All You Need"},
                files={
                    "file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf"),
                },
            )

        assert response.status_code == 201
        upload_payload = response.json()
        document_id = UUID(upload_payload["document_id"])
        ingest_job_id = UUID(upload_payload["ingest_job_id"])

        worker = build_worker()
        handled = worker.run_once()

        assert handled is not None
        assert handled.payload.document_id == document_id
        assert handled.payload.ingest_job_id == ingest_job_id

        with migrated_postgres_engine.begin() as connection:
            ingest_job = (
                connection.execute(
                    text(
                        """
                        SELECT status, failure_code, failure_message,
                               started_at, finished_at, warnings
                        FROM ingest_jobs
                        WHERE id = :ingest_job_id
                        """
                    ),
                    {"ingest_job_id": ingest_job_id},
                )
                .mappings()
                .one()
            )
            document = (
                connection.execute(
                    text(
                        """
                        SELECT current_status, title, metadata_confidence
                        FROM documents
                        WHERE id = :document_id
                        """
                    ),
                    {"document_id": document_id},
                )
                .mappings()
                .one()
            )
            parser_artifacts = connection.execute(
                text(
                    """
                    SELECT artifact_type, parser, storage_ref, is_primary
                    FROM document_artifacts
                    WHERE document_id = :document_id
                    ORDER BY created_at
                    """
                ),
                {"document_id": document_id},
            ).mappings()
            section_count = connection.execute(
                text("SELECT COUNT(*) FROM document_sections WHERE document_id = :document_id"),
                {"document_id": document_id},
            ).scalar_one()
            passage_count = connection.execute(
                text("SELECT COUNT(*) FROM document_passages WHERE document_id = :document_id"),
                {"document_id": document_id},
            ).scalar_one()
            retrieval_index = (
                connection.execute(
                    text(
                        """
                        SELECT parser_source, status
                        FROM retrieval_index_runs
                        WHERE document_id = :document_id
                        """
                    ),
                    {"document_id": document_id},
                )
                .mappings()
                .one()
            )
            metrics = IngestionQueue(unique_queue_name).queue_metrics(connection)

        artifacts = list(parser_artifacts)
        primary_artifact = next(artifact for artifact in artifacts if artifact["is_primary"])
        source_artifact = next(
            artifact for artifact in artifacts if artifact["artifact_type"] == "source_pdf"
        )

        assert upload_payload["status"] == "queued"
        assert ingest_job["status"] == "ready"
        assert ingest_job["failure_code"] is None
        assert ingest_job["failure_message"] is None
        assert ingest_job["started_at"] is not None
        assert ingest_job["finished_at"] is not None
        assert document["current_status"] == "ready"
        assert document["title"] == "Attention Is All You Need"
        assert document["metadata_confidence"] is not None
        assert section_count > 0
        assert passage_count > 0
        assert retrieval_index["status"] == "ready"
        assert retrieval_index["parser_source"] in {"docling", "pdfplumber"}
        assert primary_artifact["parser"] == retrieval_index["parser_source"]
        assert (storage_root / source_artifact["storage_ref"]).is_file()
        assert (storage_root / primary_artifact["storage_ref"]).is_file()
        assert metrics.queue_length == 0
        assert metrics.total_messages == 1
    finally:
        dispose_engine()
        get_settings.cache_clear()
