from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert, text
from sqlalchemy.exc import SQLAlchemyError

from paper_context.api.app import create_app
from paper_context.config import get_settings
from paper_context.db.engine import dispose_engine, get_engine
from paper_context.db.session import connection_scope
from paper_context.ingestion.api import DocumentsApiService
from paper_context.ingestion.enrichment import NullMetadataEnricher
from paper_context.ingestion.identifiers import artifact_id, retrieval_index_run_id
from paper_context.ingestion.queue import IngestionQueueService
from paper_context.ingestion.service import DeterministicIngestProcessor, SyntheticIngestProcessor
from paper_context.ingestion.types import (
    GateStatus,
    ParsedDocument,
    ParsedParagraph,
    ParsedSection,
    ParserArtifact,
    ParserResult,
)
from paper_context.models import Document, DocumentArtifact, IngestJob, RetrievalIndexRun
from paper_context.queue.contracts import IngestionQueue
from paper_context.storage.local_fs import LocalFilesystemStorage
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


class ArchiveFailsOnceQueue(RecordingIngestionQueue):
    def __init__(self, queue_name: str) -> None:
        super().__init__(queue_name)
        self.archive_attempts: list[int] = []
        self._should_fail = True

    def archive_message(self, conn, message_id: int) -> None:
        self.archive_attempts.append(message_id)
        if self._should_fail:
            self._should_fail = False
            raise RuntimeError("archive failed")
        super().archive_message(conn, message_id)


class _BlockingProcessor:
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self._started = started
        self._release = release

    def process(self, connection, context, lease) -> None:
        self._started.set()
        assert self._release.wait(timeout=5)


class _StaticParser:
    def __init__(self, result: ParserResult) -> None:
        self.name = result.artifact.parser
        self.result = result
        self.calls: list[str] = []
        self.name = result.artifact.parser

    def parse(
        self, filename: str, content: bytes | None = None, *, source_path=None
    ) -> ParserResult:
        del content, source_path
        self.calls.append(filename)
        return self.result


class _CrashOnceProcessor(DeterministicIngestProcessor):
    def __init__(self, *args, crash_after: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._crash_after = crash_after
        self._crashed = False

    def _maybe_crash(self, stage: str) -> None:
        if self._crash_after == stage and not self._crashed:
            self._crashed = True
            raise RuntimeError(f"crash after {stage}")

    def _store_parser_artifact(self, **kwargs):
        staged_artifact = super()._store_parser_artifact(**kwargs)
        self._maybe_crash("parser_artifact")
        return staged_artifact

    def _record_parser_artifact(self, connection, **kwargs):
        parser_artifact_id = super()._record_parser_artifact(connection, **kwargs)
        self._maybe_crash("parser_artifact")
        return parser_artifact_id

    def _normalize_document(self, connection, **kwargs):
        section_ids = super()._normalize_document(connection, **kwargs)
        self._maybe_crash("normalization")
        return section_ids


def _live_pdf_path() -> Path:
    configured = os.environ.get("PAPER_CONTEXT_E2E_PDF_PATH", "~/Downloads/attention.pdf")
    pdf_path = Path(configured).expanduser()
    if not pdf_path.is_file():
        pytest.skip(f"live PDF fixture not found at {pdf_path}")
    return pdf_path


def _make_parsed_document() -> ParsedDocument:
    return ParsedDocument(
        title="Integration paper",
        authors=["Ada Lovelace"],
        abstract="A deterministic integration fixture.",
        publication_year=2024,
        metadata_confidence=0.9,
        sections=[
            ParsedSection(
                key="intro",
                heading="Introduction",
                heading_path=["Introduction"],
                level=1,
                page_start=1,
                page_end=1,
                paragraphs=[
                    ParsedParagraph(
                        text="This integration paragraph is long enough to create a chunk.",
                        page_start=1,
                        page_end=1,
                        provenance_offsets={"pages": [1], "charspans": [[0, 58]]},
                    )
                ],
            )
        ],
        tables=[],
        references=[],
    )


def _make_parser_result(
    gate_status: GateStatus = "pass", *, parser_name: str = "docling"
) -> ParserResult:
    return ParserResult(
        gate_status=gate_status,
        parsed_document=_make_parsed_document() if gate_status != "fail" else None,
        artifact=ParserArtifact(
            artifact_type=f"{parser_name}_parse",
            parser=parser_name,
            filename=f"{parser_name}.json",
            content=b"{}",
        ),
        warnings=["reduced_structure_confidence"] if gate_status == "degraded" else [],
        failure_code=f"{parser_name}_failed" if gate_status == "fail" else None,
        failure_message=f"{parser_name} failed" if gate_status == "fail" else None,
    )


def _build_processor(
    storage_root: Path,
    *,
    primary_result: ParserResult,
    fallback_result: ParserResult | None = None,
    crash_after: str | None = None,
) -> tuple[DeterministicIngestProcessor, _StaticParser, _StaticParser]:
    storage = LocalFilesystemStorage(storage_root)
    storage.ensure_root()
    primary_parser = _StaticParser(primary_result)
    fallback_parser = _StaticParser(
        fallback_result or _make_parser_result(parser_name="pdfplumber")
    )
    if crash_after is not None:
        processor = _CrashOnceProcessor(
            storage=storage,
            primary_parser=primary_parser,
            fallback_parser=fallback_parser,
            metadata_enricher=NullMetadataEnricher(),
            index_version="mvp-v1",
            chunking_version="phase1",
            embedding_model="voyage-4-large",
            reranker_model="zerank-2",
            min_tokens=1,
            max_tokens=20,
            overlap_fraction=0.1,
            crash_after=crash_after,
        )
    else:
        processor = DeterministicIngestProcessor(
            storage=storage,
            primary_parser=primary_parser,
            fallback_parser=fallback_parser,
            metadata_enricher=NullMetadataEnricher(),
            index_version="mvp-v1",
            chunking_version="phase1",
            embedding_model="voyage-4-large",
            reranker_model="zerank-2",
            min_tokens=1,
            max_tokens=20,
            overlap_fraction=0.1,
        )
    return processor, primary_parser, fallback_parser


def _make_worker(engine, queue: IngestionQueue, processor, *, vt_seconds: int = 1) -> IngestWorker:
    return IngestWorker(
        connection_factory=lambda: connection_scope(engine),
        queue_adapter=queue,
        processor=processor,
        config=WorkerConfig(vt_seconds=vt_seconds, max_poll_seconds=1, poll_interval_ms=10),
    )


def _create_uploaded_document(
    engine,
    queue: IngestionQueue,
    storage_root: Path,
    *,
    title: str = "Integration paper",
):
    service = DocumentsApiService(
        engine=engine,
        queue=queue,
        storage=LocalFilesystemStorage(storage_root),
        max_upload_bytes=5 * 1024 * 1024,
    )
    return service.create_document(
        filename="paper.pdf",
        content_type="application/pdf",
        upload=BytesIO(b"%PDF-1.4\nintegration"),
        title=title,
    )


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
            table_count = connection.execute(
                text("SELECT COUNT(*) FROM document_tables WHERE document_id = :document_id"),
                {"document_id": document_id},
            ).scalar_one()
            passage_count = connection.execute(
                text("SELECT COUNT(*) FROM document_passages WHERE document_id = :document_id"),
                {"document_id": document_id},
            ).scalar_one()
            retrieval_passage_count = connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM retrieval_passage_assets
                    WHERE document_id = :document_id
                    """
                ),
                {"document_id": document_id},
            ).scalar_one()
            retrieval_table_count = connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM retrieval_table_assets
                    WHERE document_id = :document_id
                    """
                ),
                {"document_id": document_id},
            ).scalar_one()
            retrieval_index = (
                connection.execute(
                    text(
                        """
                        SELECT parser_source, status, is_active
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
        assert retrieval_passage_count == passage_count
        assert retrieval_table_count == table_count
        assert retrieval_index["status"] == "ready"
        assert retrieval_index["is_active"] is True
        assert retrieval_index["parser_source"] in {"docling", "pdfplumber"}
        assert primary_artifact["parser"] == retrieval_index["parser_source"]
        assert (storage_root / source_artifact["storage_ref"]).is_file()
        assert (storage_root / primary_artifact["storage_ref"]).is_file()
        assert metrics.queue_length == 0
        assert metrics.total_messages == 1
    finally:
        dispose_engine()
        get_settings.cache_clear()


def test_worker_redelivers_stalled_message_after_visibility_timeout(
    migrated_postgres_engine,
    unique_queue_name: str,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"),
            {"queue_name": unique_queue_name},
        )

    queue = IngestionQueue(unique_queue_name)
    service = IngestionQueueService(migrated_postgres_engine, queue)
    service.enqueue_document({"title": "stall", "source_type": "upload"})

    started = threading.Event()
    release = threading.Event()
    worker = _make_worker(
        migrated_postgres_engine,
        queue,
        _BlockingProcessor(started, release),
        vt_seconds=1,
    )
    worker_error: list[Exception] = []

    def _run_worker() -> None:
        try:
            worker.run_once()
        except Exception as exc:  # pragma: no cover - asserted below
            worker_error.append(exc)

    thread = threading.Thread(target=_run_worker, daemon=True)
    thread.start()
    assert started.wait(timeout=5)
    time.sleep(1.2)

    with migrated_postgres_engine.begin() as connection:
        redelivered = queue.claim_ingest(
            connection,
            vt_seconds=1,
            max_poll_seconds=1,
            poll_interval_ms=10,
        )

    release.set()
    thread.join(timeout=5)

    assert worker_error == []
    assert redelivered is not None


def test_documents_upload_cleans_up_source_artifact_when_queue_write_fails(
    migrated_postgres_engine,
    unique_queue_name: str,
    tmp_path: Path,
) -> None:
    service = DocumentsApiService(
        engine=migrated_postgres_engine,
        queue=IngestionQueue(unique_queue_name),
        storage=LocalFilesystemStorage(tmp_path / "artifacts"),
        max_upload_bytes=1024,
    )

    with pytest.raises(SQLAlchemyError):
        service.create_document(
            filename="paper.pdf",
            content_type="application/pdf",
            upload=BytesIO(b"%PDF-1.4\ncleanup"),
            title="Cleanup",
        )

    assert list((tmp_path / "artifacts").rglob("*")) == []


def test_replay_after_parser_artifact_crash_is_idempotent(
    migrated_postgres_engine,
    unique_queue_name: str,
    tmp_path: Path,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"),
            {"queue_name": unique_queue_name},
        )

    queue = IngestionQueue(unique_queue_name)
    upload = _create_uploaded_document(migrated_postgres_engine, queue, tmp_path / "artifacts")
    document_id = upload.document_id
    ingest_job_id = upload.ingest_job_id
    processor, _, _ = _build_processor(
        tmp_path / "artifacts",
        primary_result=_make_parser_result(),
        crash_after="parser_artifact",
    )
    worker = _make_worker(migrated_postgres_engine, queue, processor, vt_seconds=1)

    with pytest.raises(RuntimeError, match="parser_artifact"):
        worker.run_once()

    parser_paths = [
        path
        for path in (tmp_path / "artifacts").rglob("*")
        if path.is_file() and path.name != "source.pdf"
    ]
    assert parser_paths == []

    time.sleep(1.2)
    handled = worker.run_once()
    assert handled is not None

    with migrated_postgres_engine.begin() as connection:
        ingest_job = (
            connection.execute(
                text("SELECT status FROM ingest_jobs WHERE id = :ingest_job_id"),
                {"ingest_job_id": ingest_job_id},
            )
            .mappings()
            .one()
        )
        parser_artifact_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM document_artifacts
                WHERE document_id = :document_id
                  AND artifact_type <> 'source_pdf'
                """
            ),
            {"document_id": document_id},
        ).scalar_one()
        retrieval_passage_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM retrieval_passage_assets
                WHERE document_id = :document_id
                """
            ),
            {"document_id": document_id},
        ).scalar_one()
        retrieval_table_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM retrieval_table_assets
                WHERE document_id = :document_id
                """
            ),
            {"document_id": document_id},
        ).scalar_one()

    assert ingest_job["status"] == "ready"
    assert parser_artifact_count == 1
    assert retrieval_passage_count == 1
    assert retrieval_table_count == 0


def test_replay_after_archive_failure_archives_terminal_job_on_redelivery(
    migrated_postgres_engine,
    unique_queue_name: str,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"),
            {"queue_name": unique_queue_name},
        )

    queue = ArchiveFailsOnceQueue(unique_queue_name)
    service = IngestionQueueService(migrated_postgres_engine, queue)
    _, ingest_job_id = service.enqueue_document({"title": "archive", "source_type": "upload"})
    worker = _make_worker(migrated_postgres_engine, queue, SyntheticIngestProcessor(), vt_seconds=1)

    with pytest.raises(RuntimeError, match="archive failed"):
        worker.run_once()

    with migrated_postgres_engine.begin() as connection:
        job = (
            connection.execute(
                text("SELECT status FROM ingest_jobs WHERE id = :ingest_job_id"),
                {"ingest_job_id": ingest_job_id},
            )
            .mappings()
            .one()
        )
    assert job["status"] == "ready"

    time.sleep(1.2)
    handled = worker.run_once()
    assert handled is not None
    assert len(queue.archive_attempts) == 2


def test_stale_redelivery_does_not_reprocess_when_newer_ingest_job_exists(
    migrated_postgres_engine,
    unique_queue_name: str,
    tmp_path: Path,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"),
            {"queue_name": unique_queue_name},
        )

    queue = IngestionQueue(unique_queue_name)
    document_id = uuid4()
    older_ingest_job_id = uuid4()
    newer_ingest_job_id = uuid4()
    older_source_artifact_id = artifact_id(
        ingest_job_id=older_ingest_job_id,
        artifact_type="source_pdf",
        parser="upload",
    )
    newer_source_artifact_id = artifact_id(
        ingest_job_id=newer_ingest_job_id,
        artifact_type="source_pdf",
        parser="upload",
    )
    newer_parse_artifact_id = artifact_id(
        ingest_job_id=newer_ingest_job_id,
        artifact_type="docling_parse",
        parser="docling",
    )
    now = datetime.now(UTC)

    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            insert(Document).values(
                id=document_id,
                title="Replay",
                source_type="upload",
                current_status="ready",
                created_at=now - timedelta(minutes=2),
                updated_at=now,
            )
        )
        connection.execute(
            insert(IngestJob).values(
                id=older_ingest_job_id,
                document_id=document_id,
                source_artifact_id=None,
                status="queued",
                trigger="upload",
                warnings=[],
                created_at=now - timedelta(minutes=2),
            )
        )
        connection.execute(
            insert(IngestJob).values(
                id=newer_ingest_job_id,
                document_id=document_id,
                source_artifact_id=None,
                status="ready",
                trigger="upload",
                warnings=[],
                started_at=now - timedelta(minutes=1),
                finished_at=now - timedelta(minutes=1),
                created_at=now - timedelta(minutes=1),
            )
        )
        connection.execute(
            insert(DocumentArtifact).values(
                id=older_source_artifact_id,
                document_id=document_id,
                ingest_job_id=older_ingest_job_id,
                artifact_type="source_pdf",
                parser="upload",
                storage_ref="documents/older/source.pdf",
                checksum="old",
                is_primary=False,
            )
        )
        connection.execute(
            insert(DocumentArtifact).values(
                id=newer_source_artifact_id,
                document_id=document_id,
                ingest_job_id=newer_ingest_job_id,
                artifact_type="source_pdf",
                parser="upload",
                storage_ref="documents/newer/source.pdf",
                checksum="new",
                is_primary=False,
            )
        )
        connection.execute(
            text(
                """
                UPDATE ingest_jobs
                SET source_artifact_id = :source_artifact_id
                WHERE id = :ingest_job_id
                """
            ),
            {
                "ingest_job_id": older_ingest_job_id,
                "source_artifact_id": older_source_artifact_id,
            },
        )
        connection.execute(
            text(
                """
                UPDATE ingest_jobs
                SET source_artifact_id = :source_artifact_id
                WHERE id = :ingest_job_id
                """
            ),
            {
                "ingest_job_id": newer_ingest_job_id,
                "source_artifact_id": newer_source_artifact_id,
            },
        )
        connection.execute(
            insert(DocumentArtifact).values(
                id=newer_parse_artifact_id,
                document_id=document_id,
                ingest_job_id=newer_ingest_job_id,
                artifact_type="docling_parse",
                parser="docling",
                storage_ref="documents/newer/docling.json",
                checksum="parse",
                is_primary=True,
            )
        )
        connection.execute(
            insert(RetrievalIndexRun).values(
                id=retrieval_index_run_id(ingest_job_id=newer_ingest_job_id),
                document_id=document_id,
                ingest_job_id=newer_ingest_job_id,
                index_version="mvp-v1",
                embedding_provider="voyage",
                embedding_model="voyage-4-large",
                embedding_dimensions=1024,
                reranker_provider="zero_entropy",
                reranker_model="zerank-2",
                chunking_version="phase1",
                parser_source="docling",
                status="ready",
                is_active=True,
                activated_at=now - timedelta(minutes=1),
                deactivated_at=None,
                created_at=now - timedelta(minutes=1),
            )
        )
        queue.enqueue_ingest(connection, older_ingest_job_id, document_id)

    processor, primary_parser, _ = _build_processor(
        tmp_path / "artifacts",
        primary_result=_make_parser_result(),
    )
    worker = _make_worker(migrated_postgres_engine, queue, processor, vt_seconds=1)
    handled = worker.run_once()

    assert handled is not None
    assert primary_parser.calls == []

    with migrated_postgres_engine.begin() as connection:
        older_job = (
            connection.execute(
                text(
                    """
                SELECT status, failure_code
                FROM ingest_jobs
                WHERE id = :ingest_job_id
                """
                ),
                {"ingest_job_id": older_ingest_job_id},
            )
            .mappings()
            .one()
        )
        retrieval_runs = connection.execute(
            text("SELECT COUNT(*) FROM retrieval_index_runs WHERE document_id = :document_id"),
            {"document_id": document_id},
        ).scalar_one()
        retrieval_passage_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM retrieval_passage_assets
                WHERE document_id = :document_id
                """
            ),
            {"document_id": document_id},
        ).scalar_one()
        retrieval_table_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM retrieval_table_assets
                WHERE document_id = :document_id
                """
            ),
            {"document_id": document_id},
        ).scalar_one()

    assert older_job["status"] == "failed"
    assert older_job["failure_code"] == "superseded_by_newer_ingest_job"
    assert retrieval_runs == 1
    assert retrieval_passage_count == 0
    assert retrieval_table_count == 0


def test_fallback_path_reaches_ready_with_pdfplumber_parser_source(
    migrated_postgres_engine,
    unique_queue_name: str,
    tmp_path: Path,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"),
            {"queue_name": unique_queue_name},
        )

    queue = IngestionQueue(unique_queue_name)
    upload = _create_uploaded_document(migrated_postgres_engine, queue, tmp_path / "artifacts")
    processor, _, fallback_parser = _build_processor(
        tmp_path / "artifacts",
        primary_result=_make_parser_result("degraded", parser_name="docling"),
        fallback_result=_make_parser_result("pass", parser_name="pdfplumber"),
    )
    worker = _make_worker(migrated_postgres_engine, queue, processor, vt_seconds=1)

    handled = worker.run_once()

    assert handled is not None
    assert len(fallback_parser.calls) == 1

    with migrated_postgres_engine.begin() as connection:
        ingest_job = (
            connection.execute(
                text("SELECT status, warnings FROM ingest_jobs WHERE id = :ingest_job_id"),
                {"ingest_job_id": upload.ingest_job_id},
            )
            .mappings()
            .one()
        )
        retrieval_passage_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM retrieval_passage_assets
                WHERE document_id = :document_id
                """
            ),
            {"document_id": upload.document_id},
        ).scalar_one()
        retrieval_index = (
            connection.execute(
                text(
                    """
                SELECT parser_source, status, is_active
                FROM retrieval_index_runs
                WHERE document_id = :document_id
                """
                ),
                {"document_id": upload.document_id},
            )
            .mappings()
            .one()
        )

    assert ingest_job["status"] == "ready"
    assert "parser_fallback_used" in ingest_job["warnings"]
    assert retrieval_index["status"] == "ready"
    assert retrieval_index["is_active"] is True
    assert retrieval_index["parser_source"] == "pdfplumber"
    assert retrieval_passage_count > 0


def test_failed_parse_clears_existing_retrieval_index_run(
    migrated_postgres_engine,
    unique_queue_name: str,
    tmp_path: Path,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"),
            {"queue_name": unique_queue_name},
        )

    queue = IngestionQueue(unique_queue_name)
    initial_upload = _create_uploaded_document(
        migrated_postgres_engine,
        queue,
        tmp_path / "artifacts-initial",
        title="Initial",
    )
    ready_processor, _, _ = _build_processor(
        tmp_path / "artifacts-initial",
        primary_result=_make_parser_result(),
    )
    _make_worker(migrated_postgres_engine, queue, ready_processor, vt_seconds=1).run_once()

    with migrated_postgres_engine.begin() as connection:
        new_ingest_job_id = uuid4()
        new_source_artifact_id = artifact_id(
            ingest_job_id=new_ingest_job_id,
            artifact_type="source_pdf",
            parser="upload",
        )
        connection.execute(
            insert(IngestJob).values(
                id=new_ingest_job_id,
                document_id=initial_upload.document_id,
                source_artifact_id=None,
                status="queued",
                trigger="upload",
                warnings=[],
            )
        )
        connection.execute(
            insert(DocumentArtifact).values(
                id=new_source_artifact_id,
                document_id=initial_upload.document_id,
                ingest_job_id=new_ingest_job_id,
                artifact_type="source_pdf",
                parser="upload",
                storage_ref="documents/reingest/source.pdf",
                checksum="reingest",
                is_primary=False,
            )
        )
        connection.execute(
            text(
                """
                UPDATE ingest_jobs
                SET source_artifact_id = :source_artifact_id
                WHERE id = :ingest_job_id
                """
            ),
            {
                "ingest_job_id": new_ingest_job_id,
                "source_artifact_id": new_source_artifact_id,
            },
        )
        queue.enqueue_ingest(connection, new_ingest_job_id, initial_upload.document_id)

    retry_storage = LocalFilesystemStorage(tmp_path / "artifacts-retry")
    retry_storage.store_bytes("documents/reingest/source.pdf", b"%PDF-1.4\nretry")
    fail_processor, _, _ = _build_processor(
        tmp_path / "artifacts-retry",
        primary_result=_make_parser_result("fail", parser_name="docling"),
    )
    _make_worker(migrated_postgres_engine, queue, fail_processor, vt_seconds=1).run_once()

    with migrated_postgres_engine.begin() as connection:
        failed_job = (
            connection.execute(
                text(
                    """
                SELECT status, failure_code
                FROM ingest_jobs
                WHERE id = :ingest_job_id
                """
                ),
                {"ingest_job_id": new_ingest_job_id},
            )
            .mappings()
            .one()
        )
        retrieval_runs = connection.execute(
            text("SELECT COUNT(*) FROM retrieval_index_runs WHERE document_id = :document_id"),
            {"document_id": initial_upload.document_id},
        ).scalar_one()

    assert failed_job["status"] == "failed"
    assert failed_job["failure_code"] == "docling_failed"
    assert retrieval_runs == 0
