from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime

from sqlalchemy import insert, text
from sqlalchemy.engine import Engine

from paper_context.models import Document, DocumentArtifact, IngestJob
from paper_context.queue.contracts import IngestionQueue
from paper_context.schemas.api import DocumentUploadResponse, IngestJobResponse
from paper_context.storage.base import StorageInterface


class DocumentsApiService:
    def __init__(
        self,
        engine: Engine,
        queue: IngestionQueue,
        storage: StorageInterface,
    ) -> None:
        self._engine = engine
        self._queue = queue
        self._storage = storage

    def create_document(
        self,
        *,
        filename: str,
        content_type: str | None,
        body: bytes,
        title: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> DocumentUploadResponse:
        if not body:
            raise ValueError("uploaded file is empty")
        if content_type not in {None, "application/pdf"} and not filename.lower().endswith(".pdf"):
            raise ValueError("uploaded file must be a PDF")

        document_id = uuid.uuid4()
        ingest_job_id = uuid.uuid4()
        now = datetime.now(UTC)
        source_extension = "pdf"
        storage_path = f"documents/{document_id}/{ingest_job_id}/source.{source_extension}"
        stored_artifact = self._storage.store_bytes(storage_path, body)
        document_title = title or filename.rsplit(".", 1)[0] or None

        with self._engine.begin() as connection:
            connection.execute(
                insert(Document).values(
                    id=document_id,
                    title=document_title,
                    source_type="upload",
                    current_status="queued",
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.execute(
                insert(DocumentArtifact).values(
                    document_id=document_id,
                    artifact_type="source_pdf",
                    parser="upload",
                    storage_ref=stored_artifact.storage_ref,
                    checksum=stored_artifact.checksum,
                    is_primary=False,
                )
            )
            connection.execute(
                insert(IngestJob).values(
                    id=ingest_job_id,
                    document_id=document_id,
                    status="queued",
                    trigger="upload",
                    warnings=[],
                )
            )
            self._queue.enqueue_ingest(
                connection,
                ingest_job_id,
                document_id,
                headers=trace_headers,
            )

        return DocumentUploadResponse(
            document_id=document_id,
            ingest_job_id=ingest_job_id,
            status="queued",
        )

    def get_ingest_job(self, ingest_job_id: uuid.UUID) -> IngestJobResponse | None:
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        SELECT
                            id,
                            document_id,
                            status,
                            failure_code,
                            failure_message,
                            COALESCE(warnings, '[]'::jsonb) AS warnings,
                            started_at,
                            finished_at,
                            trigger
                        FROM ingest_jobs
                        WHERE id = :ingest_job_id
                        """
                    ),
                    {"ingest_job_id": ingest_job_id},
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            return None
        return IngestJobResponse.model_validate(dict(row))
