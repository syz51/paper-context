from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import insert, text
from sqlalchemy.exc import SQLAlchemyError

from paper_context.db.session import connection_scope
from paper_context.ingestion.api import DocumentsApiService
from paper_context.ingestion.enrichment import NullMetadataEnricher
from paper_context.ingestion.identifiers import artifact_id
from paper_context.ingestion.service import DeterministicIngestProcessor
from paper_context.ingestion.types import (
    GateStatus,
    ParsedDocument,
    ParsedParagraph,
    ParsedSection,
    ParserArtifact,
    ParserResult,
)
from paper_context.models import DocumentArtifact, IngestJob
from paper_context.queue.contracts import IngestionQueue
from paper_context.storage.local_fs import LocalFilesystemStorage
from paper_context.worker.loop import IngestWorker, WorkerConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_postgres,
    pytest.mark.slow,
]


class StaticPdfParser:
    def __init__(self, result: ParserResult) -> None:
        self.name = result.artifact.parser
        self._result = result

    def parse(self, filename: str, content: bytes) -> ParserResult:
        del filename, content
        return self._result


class FailOnceArchiveQueue(IngestionQueue):
    def __init__(self, queue_name: str) -> None:
        super().__init__(queue_name)
        self._failed = False

    def archive_message(self, conn, message_id: int) -> None:
        if not self._failed:
            self._failed = True
            raise RuntimeError("archive failed")
        super().archive_message(conn, message_id)


class CrashAfterParserArtifactProcessor(DeterministicIngestProcessor):
    def _normalize_document(self, connection, *, document_id, parsed_document, artifact_id):
        raise RuntimeError("crash after parser artifact write")


class CrashAfterNormalizationProcessor(DeterministicIngestProcessor):
    def _apply_document_metadata(
        self,
        connection,
        *,
        document_id,
        title,
        authors,
        abstract,
        publication_year,
        metadata_confidence,
    ) -> None:
        del (
            connection,
            document_id,
            title,
            authors,
            abstract,
            publication_year,
            metadata_confidence,
        )
        raise RuntimeError("crash after normalization writes")


def _queue(queue_name: str) -> IngestionQueue:
    return IngestionQueue(queue_name)


def _worker(
    engine,
    *,
    queue: IngestionQueue,
    processor,
    vt_seconds: int = 1,
) -> IngestWorker:
    return IngestWorker(
        connection_factory=lambda: connection_scope(engine),
        queue_adapter=queue,
        processor=processor,
        config=WorkerConfig(vt_seconds=vt_seconds, max_poll_seconds=1, poll_interval_ms=10),
    )


def _parsed_document() -> ParsedDocument:
    return ParsedDocument(
        title="Hardening paper",
        authors=["Ada Lovelace"],
        abstract="Deterministic replay coverage.",
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
                        text="one two three four five six seven",
                        page_start=1,
                        page_end=1,
                        provenance_offsets={"pages": [1], "charspans": [[0, 33]]},
                    )
                ],
            )
        ],
        tables=[],
        references=[],
    )


def _parser_result(
    *,
    parser_name: str,
    gate_status: GateStatus,
) -> ParserResult:
    parsed_document = _parsed_document() if gate_status != "fail" else None
    return ParserResult(
        gate_status=gate_status,
        parsed_document=parsed_document,
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


def _processor(
    storage: LocalFilesystemStorage,
    *,
    primary_result: ParserResult,
    fallback_result: ParserResult | None = None,
    cls=DeterministicIngestProcessor,
) -> DeterministicIngestProcessor:
    return cls(
        storage=storage,
        primary_parser=StaticPdfParser(primary_result),
        fallback_parser=StaticPdfParser(
            fallback_result or _parser_result(parser_name="pdfplumber", gate_status="fail")
        ),
        metadata_enricher=NullMetadataEnricher(),
        index_version="index-v1",
        chunking_version="chunk-v1",
        embedding_model="emb-v1",
        reranker_model="rank-v1",
        min_tokens=1,
        max_tokens=20,
        overlap_fraction=0.1,
    )


def _create_upload(
    engine,
    *,
    queue_name: str,
    storage_root: Path,
    title: str = "Hardening upload",
):
    storage = LocalFilesystemStorage(storage_root)
    service = DocumentsApiService(
        engine=engine,
        queue=_queue(queue_name),
        storage=storage,
        max_upload_bytes=1024 * 1024,
    )
    return storage, service.create_document(
        filename="paper.pdf",
        content_type="application/pdf",
        upload=BytesIO(b"%PDF-1.4\nphase-1"),
        title=title,
    )


def _insert_superseding_job(
    engine,
    *,
    queue_name: str,
    storage: LocalFilesystemStorage,
    document_id: UUID,
) -> UUID:
    ingest_job_id = uuid4()
    source_artifact_id = artifact_id(
        ingest_job_id=ingest_job_id,
        artifact_type="source_pdf",
        parser="upload",
    )
    stored = storage.store_bytes(
        f"documents/{document_id}/{ingest_job_id}/source.pdf",
        b"%PDF-1.4\nnewer",
    )

    with engine.begin() as connection:
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                source_artifact_id=None,
                status="queued",
                trigger="upload",
                warnings=[],
            )
        )
        connection.execute(
            insert(DocumentArtifact).values(
                id=source_artifact_id,
                document_id=document_id,
                ingest_job_id=ingest_job_id,
                artifact_type="source_pdf",
                parser="upload",
                storage_ref=stored.storage_ref,
                checksum=stored.checksum,
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
                "ingest_job_id": ingest_job_id,
                "source_artifact_id": source_artifact_id,
            },
        )
        _queue(queue_name).enqueue_ingest(connection, ingest_job_id, document_id)

    return ingest_job_id


def test_documents_api_service_cleans_up_source_artifact_when_queue_write_fails(
    migrated_postgres_engine,
    unique_queue_name: str,
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "artifacts"
    storage = LocalFilesystemStorage(storage_root)
    service = DocumentsApiService(
        engine=migrated_postgres_engine,
        queue=_queue(unique_queue_name),
        storage=storage,
        max_upload_bytes=1024 * 1024,
    )

    with pytest.raises(SQLAlchemyError):
        service.create_document(
            filename="paper.pdf",
            content_type="application/pdf",
            upload=BytesIO(b"%PDF-1.4\nphase-1"),
            title="Should rollback",
        )

    with migrated_postgres_engine.begin() as connection:
        document_count = connection.execute(text("SELECT COUNT(*) FROM documents")).scalar_one()
        ingest_job_count = connection.execute(text("SELECT COUNT(*) FROM ingest_jobs")).scalar_one()

    assert document_count == 0
    assert ingest_job_count == 0
    assert list(storage_root.rglob("*")) == []


def test_fallback_ingestion_reaches_ready_with_pdfplumber_parser_source(
    migrated_postgres_engine,
    unique_queue_name: str,
    tmp_path: Path,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"), {"queue_name": unique_queue_name}
        )

    storage, upload = _create_upload(
        migrated_postgres_engine,
        queue_name=unique_queue_name,
        storage_root=tmp_path / "artifacts",
    )
    worker = _worker(
        migrated_postgres_engine,
        queue=_queue(unique_queue_name),
        processor=_processor(
            storage,
            primary_result=_parser_result(parser_name="docling", gate_status="degraded"),
            fallback_result=_parser_result(parser_name="pdfplumber", gate_status="pass"),
        ),
    )

    handled = worker.run_once()

    assert handled is not None
    with migrated_postgres_engine.begin() as connection:
        job = (
            connection.execute(
                text(
                    """
                SELECT status, warnings
                FROM ingest_jobs
                WHERE id = :ingest_job_id
                """
                ),
                {"ingest_job_id": upload.ingest_job_id},
            )
            .mappings()
            .one()
        )
        retrieval = (
            connection.execute(
                text(
                    """
                SELECT parser_source, status, is_active
                FROM retrieval_index_runs
                WHERE ingest_job_id = :ingest_job_id
                """
                ),
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

    assert job["status"] == "ready"
    assert "parser_fallback_used" in job["warnings"]
    assert retrieval["status"] == "ready"
    assert retrieval["is_active"] is True
    assert retrieval["parser_source"] == "pdfplumber"
    assert retrieval_passage_count > 0


def test_failed_parse_leaves_no_active_retrieval_index_run(
    migrated_postgres_engine,
    unique_queue_name: str,
    tmp_path: Path,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"), {"queue_name": unique_queue_name}
        )

    storage, upload = _create_upload(
        migrated_postgres_engine,
        queue_name=unique_queue_name,
        storage_root=tmp_path / "artifacts",
    )
    worker = _worker(
        migrated_postgres_engine,
        queue=_queue(unique_queue_name),
        processor=_processor(
            storage,
            primary_result=_parser_result(parser_name="docling", gate_status="fail"),
        ),
    )

    handled = worker.run_once()

    assert handled is not None
    with migrated_postgres_engine.begin() as connection:
        job = (
            connection.execute(
                text(
                    """
                SELECT status, failure_code
                FROM ingest_jobs
                WHERE id = :ingest_job_id
                """
                ),
                {"ingest_job_id": upload.ingest_job_id},
            )
            .mappings()
            .one()
        )
        retrieval_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM retrieval_index_runs
                WHERE ingest_job_id = :ingest_job_id
                """
            ),
            {"ingest_job_id": upload.ingest_job_id},
        ).scalar_one()
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

    assert job["status"] == "failed"
    assert job["failure_code"] == "docling_failed"
    assert retrieval_count == 0
    assert retrieval_passage_count == 0


def test_worker_replay_after_parser_artifact_crash_cleans_files_and_recovers(
    migrated_postgres_engine,
    unique_queue_name: str,
    tmp_path: Path,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"), {"queue_name": unique_queue_name}
        )

    storage_root = tmp_path / "artifacts"
    storage, upload = _create_upload(
        migrated_postgres_engine,
        queue_name=unique_queue_name,
        storage_root=storage_root,
    )
    crashing_worker = _worker(
        migrated_postgres_engine,
        queue=_queue(unique_queue_name),
        processor=_processor(
            storage,
            primary_result=_parser_result(parser_name="docling", gate_status="pass"),
            cls=CrashAfterParserArtifactProcessor,
        ),
    )

    with pytest.raises(RuntimeError, match="crash after parser artifact write"):
        crashing_worker.run_once()

    with migrated_postgres_engine.begin() as connection:
        parser_artifact_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM document_artifacts
                WHERE document_id = :document_id
                  AND artifact_type <> 'source_pdf'
                """
            ),
            {"document_id": upload.document_id},
        ).scalar_one()
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
        table_count = connection.execute(
            text("SELECT COUNT(*) FROM document_tables WHERE document_id = :document_id"),
            {"document_id": upload.document_id},
        ).scalar_one()
        retrieval_table_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM retrieval_table_assets
                WHERE document_id = :document_id
                """
            ),
            {"document_id": upload.document_id},
        ).scalar_one()

    assert parser_artifact_count == 0
    assert retrieval_passage_count == 0
    assert retrieval_table_count == 0
    assert not (
        storage_root / f"documents/{upload.document_id}/{upload.ingest_job_id}/docling.json"
    ).exists()

    time.sleep(1.2)
    recovering_worker = _worker(
        migrated_postgres_engine,
        queue=_queue(unique_queue_name),
        processor=_processor(
            storage,
            primary_result=_parser_result(parser_name="docling", gate_status="pass"),
        ),
    )
    handled = recovering_worker.run_once()

    assert handled is not None
    with migrated_postgres_engine.begin() as connection:
        job = (
            connection.execute(
                text(
                    """
                SELECT status
                FROM ingest_jobs
                WHERE id = :ingest_job_id
                """
                ),
                {"ingest_job_id": upload.ingest_job_id},
            )
            .mappings()
            .one()
        )
        retrieval_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM retrieval_index_runs
                WHERE ingest_job_id = :ingest_job_id
                """
            ),
            {"ingest_job_id": upload.ingest_job_id},
        ).scalar_one()
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
        retrieval_table_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM retrieval_table_assets
                WHERE document_id = :document_id
                """
            ),
            {"document_id": upload.document_id},
        ).scalar_one()

    assert job["status"] == "ready"
    assert retrieval_count == 1
    assert retrieval_passage_count > 0
    assert retrieval_table_count == table_count


def test_worker_replay_after_normalization_crash_rolls_back_partial_rows(
    migrated_postgres_engine,
    unique_queue_name: str,
    tmp_path: Path,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"), {"queue_name": unique_queue_name}
        )

    storage, upload = _create_upload(
        migrated_postgres_engine,
        queue_name=unique_queue_name,
        storage_root=tmp_path / "artifacts",
    )
    crashing_worker = _worker(
        migrated_postgres_engine,
        queue=_queue(unique_queue_name),
        processor=_processor(
            storage,
            primary_result=_parser_result(parser_name="docling", gate_status="pass"),
            cls=CrashAfterNormalizationProcessor,
        ),
    )

    with pytest.raises(RuntimeError, match="crash after normalization writes"):
        crashing_worker.run_once()

    with migrated_postgres_engine.begin() as connection:
        section_count = connection.execute(
            text("SELECT COUNT(*) FROM document_sections WHERE document_id = :document_id"),
            {"document_id": upload.document_id},
        ).scalar_one()
        passage_count = connection.execute(
            text("SELECT COUNT(*) FROM document_passages WHERE document_id = :document_id"),
            {"document_id": upload.document_id},
        ).scalar_one()
        retrieval_count = connection.execute(
            text("SELECT COUNT(*) FROM retrieval_index_runs WHERE document_id = :document_id"),
            {"document_id": upload.document_id},
        ).scalar_one()
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
        retrieval_table_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM retrieval_table_assets
                WHERE document_id = :document_id
                """
            ),
            {"document_id": upload.document_id},
        ).scalar_one()

    assert section_count == 0
    assert passage_count == 0
    assert retrieval_count == 0
    assert retrieval_passage_count == 0
    assert retrieval_table_count == 0


def test_worker_replay_after_archive_failure_archives_terminal_job_on_redelivery(
    migrated_postgres_engine,
    unique_queue_name: str,
    tmp_path: Path,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"), {"queue_name": unique_queue_name}
        )

    queue = FailOnceArchiveQueue(unique_queue_name)
    storage = LocalFilesystemStorage(tmp_path / "artifacts")
    service = DocumentsApiService(
        engine=migrated_postgres_engine,
        queue=queue,
        storage=storage,
        max_upload_bytes=1024 * 1024,
    )
    upload = service.create_document(
        filename="paper.pdf",
        content_type="application/pdf",
        upload=BytesIO(b"%PDF-1.4\nphase-1"),
        title="Archive crash",
    )
    worker = _worker(
        migrated_postgres_engine,
        queue=queue,
        processor=_processor(
            storage,
            primary_result=_parser_result(parser_name="docling", gate_status="pass"),
        ),
    )

    with pytest.raises(RuntimeError, match="archive failed"):
        worker.run_once()

    with migrated_postgres_engine.begin() as connection:
        job = (
            connection.execute(
                text(
                    """
                SELECT status
                FROM ingest_jobs
                WHERE id = :ingest_job_id
                """
                ),
                {"ingest_job_id": upload.ingest_job_id},
            )
            .mappings()
            .one()
        )
    assert job["status"] == "ready"

    time.sleep(1.2)
    handled = worker.run_once()

    assert handled is not None
    with migrated_postgres_engine.begin() as connection:
        metrics = queue.queue_metrics(connection)

    assert metrics.queue_length == 0


def test_stale_redelivery_after_newer_ingest_job_exists_is_failed_without_wiping_newer_data(
    migrated_postgres_engine,
    unique_queue_name: str,
    tmp_path: Path,
) -> None:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"), {"queue_name": unique_queue_name}
        )

    storage_root = tmp_path / "artifacts"
    storage, original_upload = _create_upload(
        migrated_postgres_engine,
        queue_name=unique_queue_name,
        storage_root=storage_root,
        title="Original upload",
    )

    with migrated_postgres_engine.begin() as connection:
        claimed = _queue(unique_queue_name).claim_ingest(
            connection,
            vt_seconds=1,
            max_poll_seconds=1,
            poll_interval_ms=10,
        )
    assert claimed is not None
    assert claimed.payload.ingest_job_id == original_upload.ingest_job_id

    newer_ingest_job_id = _insert_superseding_job(
        migrated_postgres_engine,
        queue_name=unique_queue_name,
        storage=storage,
        document_id=original_upload.document_id,
    )
    newer_worker = _worker(
        migrated_postgres_engine,
        queue=_queue(unique_queue_name),
        processor=_processor(
            storage,
            primary_result=_parser_result(parser_name="docling", gate_status="pass"),
        ),
    )

    handled_newer = newer_worker.run_once()
    assert handled_newer is not None
    assert handled_newer.payload.ingest_job_id == newer_ingest_job_id

    time.sleep(1.2)
    handled_stale = newer_worker.run_once()
    assert handled_stale is not None
    assert handled_stale.payload.ingest_job_id == original_upload.ingest_job_id

    with migrated_postgres_engine.begin() as connection:
        original_job = (
            connection.execute(
                text(
                    """
                SELECT status, failure_code
                FROM ingest_jobs
                WHERE id = :ingest_job_id
                """
                ),
                {"ingest_job_id": original_upload.ingest_job_id},
            )
            .mappings()
            .one()
        )
        current_document = (
            connection.execute(
                text(
                    """
                SELECT current_status
                FROM documents
                WHERE id = :document_id
                """
                ),
                {"document_id": original_upload.document_id},
            )
            .mappings()
            .one()
        )
        retrieval = (
            connection.execute(
                text(
                    """
                SELECT ingest_job_id, status, is_active
                FROM retrieval_index_runs
                WHERE document_id = :document_id
                """
                ),
                {"document_id": original_upload.document_id},
            )
            .mappings()
            .one()
        )

    assert original_job["status"] == "failed"
    assert original_job["failure_code"] == "superseded_by_newer_ingest_job"
    assert current_document["current_status"] == "ready"
    assert retrieval["ingest_job_id"] == newer_ingest_job_id
    assert retrieval["status"] == "ready"
    assert retrieval["is_active"] is True
