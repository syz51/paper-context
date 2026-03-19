from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from paper_context.ingestion.api import DocumentsApiService
from paper_context.ingestion.enrichment import NullMetadataEnricher
from paper_context.ingestion.parsers import PdfPlumberPdfParser
from paper_context.ingestion.service import (
    DeterministicIngestProcessor,
    IngestJobContext,
    IngestJobRow,
    SourceArtifactRow,
)
from paper_context.ingestion.types import (
    GateStatus,
    ParsedDocument,
    ParsedParagraph,
    ParsedSection,
    ParserArtifact,
    ParserResult,
)
from paper_context.queue.contracts import IngestQueuePayload
from paper_context.queue.pgmq import PgmqMessage

pytestmark = pytest.mark.unit


def _make_message(payload: Mapping[str, object]) -> PgmqMessage:
    return PgmqMessage(
        msg_id=1,
        read_ct=0,
        enqueued_at=datetime.now(UTC),
        vt=datetime.now(UTC),
        message=dict(payload),
    )


def _make_context() -> IngestJobContext:
    ingest_job_id = uuid4()
    document_id = uuid4()
    message = _make_message({"ingest_job_id": str(ingest_job_id), "document_id": str(document_id)})
    payload = IngestQueuePayload(ingest_job_id=ingest_job_id, document_id=document_id)
    return IngestJobContext(message=message, payload=payload)


def _make_parsed_document() -> ParsedDocument:
    return ParsedDocument(
        title="Phase 1 paper",
        authors=["Ada Lovelace"],
        abstract="A deterministic ingestion test document.",
        publication_year=2024,
        metadata_confidence=0.9,
        sections=[
            ParsedSection(
                key="s1",
                heading="Introduction",
                heading_path=["Introduction"],
                level=1,
                page_start=1,
                page_end=1,
                paragraphs=[
                    ParsedParagraph(
                        text=(
                            "This is a long enough paragraph to create at least one "
                            "chunk of content."
                        ),
                        page_start=1,
                        page_end=1,
                        provenance_offsets={"pages": [1], "charspans": [[0, 72]]},
                    )
                ],
            )
        ],
        tables=[],
        references=[],
    )


def _make_parser_result(
    gate_status: GateStatus = "pass",
    *,
    parser_name: str = "docling",
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


class _RecordingProcessor(DeterministicIngestProcessor):
    def __init__(
        self,
        *,
        primary_result: ParserResult,
        fallback_result: ParserResult | None = None,
    ) -> None:
        storage = MagicMock()
        resolved_path = Path("/tmp/source.pdf")
        storage.resolve.return_value = resolved_path
        primary_parser = MagicMock()
        primary_parser.parse.return_value = primary_result
        fallback_parser = MagicMock()
        fallback_parser.parse.return_value = fallback_result
        super().__init__(
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
        self.storage = storage
        self.primary_parser = primary_parser
        self.fallback_parser = fallback_parser
        self.stage_calls: list[tuple[str, list[str]]] = []
        self.failed_calls: list[tuple[str, str, list[str]]] = []
        self.ready_warnings: list[str] | None = None
        self.replaced_index = False
        self._retrieval_indexer.rebuild = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda *args, **kwargs: setattr(self, "replaced_index", True)
        )
        self._try_claim_processing_lock = MagicMock(return_value=True)  # type: ignore[method-assign]
        self._release_processing_lock = MagicMock()  # type: ignore[method-assign]

    def _lock_ingest_job(self, connection, ingest_job_id) -> IngestJobRow | None:
        return {
            "document_id": uuid4(),
            "created_at": datetime.now(UTC),
            "status": "queued",
            "warnings": [],
            "source_artifact_id": uuid4(),
        }

    def _load_source_artifact(
        self, connection, *, ingest_job_id, source_artifact_id
    ) -> SourceArtifactRow | None:
        return cast(
            SourceArtifactRow,
            {"id": source_artifact_id, "storage_ref": "documents/test/source.pdf"},
        )

    def _reset_document_state(self, connection, *, document_id, ingest_job_id) -> None:
        return None

    def _is_superseded(self, connection, *, ingest_job_id, document_id, created_at) -> bool:
        return False

    def _persist_parser_artifact(self, connection, **kwargs):
        return uuid4()

    def _mark_stage(self, connection, *, ingest_job_id, document_id, status, warnings) -> None:
        self.stage_calls.append((status, list(warnings)))

    def _normalize_document(self, connection, *, document_id, parsed_document, artifact_id):
        return {"s1": uuid4()}

    def _apply_document_metadata(self, connection, **kwargs) -> None:
        return None

    def _apply_enriched_document_metadata(self, connection, **kwargs) -> None:
        return None

    def _insert_passages(self, connection, **kwargs) -> None:
        return None

    def _mark_ready(self, connection, *, ingest_job_id, document_id, warnings) -> None:
        self.ready_warnings = list(warnings)

    def _mark_failed(
        self,
        connection,
        *,
        ingest_job_id,
        document_id,
        failure_code,
        failure_message,
        warnings,
    ) -> None:
        self.failed_calls.append((failure_code, failure_message, list(warnings)))


def test_documents_api_service_stores_source_artifact_and_enqueues_job(tmp_path: Path) -> None:
    engine = MagicMock()
    connection = MagicMock()
    engine.begin.return_value = nullcontext(connection)
    queue = MagicMock()
    storage = MagicMock()
    storage.store_file.return_value = SimpleNamespace(
        storage_ref="documents/source.pdf",
        checksum="abc123",
    )
    service = DocumentsApiService(engine=engine, queue=queue, storage=storage)

    response = service.create_document(
        filename="paper.pdf",
        content_type="application/pdf",
        upload=BytesIO(b"%PDF-1.4\nphase-1"),
        title="Phase 1 paper",
    )

    assert response.status == "queued"
    storage.store_file.assert_called_once()
    assert connection.execute.call_count == 4
    queue.enqueue_ingest.assert_called_once()


def test_replace_document_supersedes_older_queued_jobs_before_enqueuing_new_job() -> None:
    engine = MagicMock()
    connection = MagicMock()
    engine.begin.return_value = nullcontext(connection)
    update_result = MagicMock()
    update_result.rowcount = 1
    connection.execute.side_effect = [
        update_result,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]
    queue = MagicMock()
    storage = MagicMock()
    storage.store_file.return_value = SimpleNamespace(
        storage_ref="documents/source.pdf",
        checksum="abc123",
    )
    service = DocumentsApiService(engine=engine, queue=queue, storage=storage)

    response = service.replace_document(
        uuid4(),
        filename="replacement.pdf",
        content_type="application/pdf",
        upload=BytesIO(b"%PDF-1.4\nreplacement"),
        title="Replacement title",
    )

    assert response.status == "queued"
    supersede_calls = [
        call
        for call in connection.execute.call_args_list
        if len(call.args) > 1
        and isinstance(call.args[1], dict)
        and call.args[1].get("failure_code") == "superseded_by_newer_ingest_job"
    ]
    assert len(supersede_calls) == 1
    queue.enqueue_ingest.assert_called_once()


def test_documents_api_service_get_ingest_job_returns_schema() -> None:
    engine = MagicMock()
    connection = MagicMock()
    row = {
        "id": uuid4(),
        "document_id": uuid4(),
        "status": "failed",
        "failure_code": "parse_failed",
        "failure_message": "Docling failed",
        "warnings": ["parser_fallback_used"],
        "started_at": datetime(2026, 3, 18, 8, 0, tzinfo=UTC),
        "finished_at": datetime(2026, 3, 18, 8, 1, tzinfo=UTC),
        "trigger": "upload",
    }
    connection.execute.return_value.mappings.return_value.one_or_none.return_value = row
    engine.begin.return_value = nullcontext(connection)
    service = DocumentsApiService(engine=engine, queue=MagicMock(), storage=MagicMock())

    job = service.get_ingest_job(row["id"])

    assert job is not None
    assert job.id == row["id"]
    assert job.status == "failed"
    assert job.warnings == ["parser_fallback_used"]


def test_deterministic_processor_archives_terminal_jobs_without_work() -> None:
    processor = _RecordingProcessor(primary_result=_make_parser_result())
    processor._lock_ingest_job = MagicMock(return_value={"status": "ready", "warnings": []})  # type: ignore[method-assign]
    lease = MagicMock()

    processor.process(MagicMock(), _make_context(), lease)

    processor.primary_parser.parse.assert_not_called()
    assert processor.stage_calls == []
    assert processor.failed_calls == []


def test_deterministic_processor_uses_fallback_on_degraded_primary() -> None:
    processor = _RecordingProcessor(
        primary_result=_make_parser_result("degraded", parser_name="docling"),
        fallback_result=_make_parser_result("pass", parser_name="pdfplumber"),
    )
    lease = MagicMock()

    processor.process(MagicMock(), _make_context(), lease)

    assert [status for status, _ in processor.stage_calls] == [
        "parsing",
        "normalizing",
        "enriching_metadata",
        "chunking",
        "indexing",
    ]
    processor.primary_parser.parse.assert_called_once()
    processor.fallback_parser.parse.assert_called_once()
    processor.primary_parser.parse.assert_called_once_with(
        filename="documents/test/source.pdf",
        source_path=Path("/tmp/source.pdf"),
    )
    processor.fallback_parser.parse.assert_called_once_with(
        filename="documents/test/source.pdf",
        source_path=Path("/tmp/source.pdf"),
    )
    assert processor.replaced_index is True
    assert processor.ready_warnings is not None
    assert "parser_fallback_used" in processor.ready_warnings


def test_deterministic_processor_marks_failures_when_primary_parse_fails() -> None:
    processor = _RecordingProcessor(primary_result=_make_parser_result("fail"))
    lease = MagicMock()

    processor.process(MagicMock(), _make_context(), lease)

    assert processor.failed_calls == [("docling_failed", "docling failed", [])]
    processor.fallback_parser.parse.assert_not_called()


def test_pdfplumber_parser_returns_machine_readable_failure_for_invalid_bytes() -> None:
    result = PdfPlumberPdfParser().parse("broken.pdf", b"not-a-pdf")

    assert result.gate_status == "fail"
    assert result.failure_code == "pdfplumber_conversion_failed"
    assert result.parsed_document is None
