from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from tempfile import SpooledTemporaryFile
from typing import BinaryIO, cast

from sqlalchemy import insert, text
from sqlalchemy.engine import Engine

from paper_context.models import Document, DocumentArtifact, IngestJob
from paper_context.queue.contracts import IngestionQueue
from paper_context.schemas.api import DocumentUploadResponse, IngestJobResponse
from paper_context.storage.base import StorageInterface

from .identifiers import artifact_id

_PDF_MAGIC = b"%PDF-"
_UPLOAD_CHUNK_SIZE = 1024 * 1024


class UploadTooLargeError(ValueError):
    pass


class DocumentsApiService:
    def __init__(
        self,
        engine: Engine,
        queue: IngestionQueue,
        storage: StorageInterface,
        *,
        max_upload_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        self._engine = engine
        self._queue = queue
        self._storage = storage
        self._max_upload_bytes = max_upload_bytes

    def create_document(
        self,
        *,
        filename: str,
        content_type: str | None,
        upload: BinaryIO,
        title: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> DocumentUploadResponse:
        del content_type
        staged_upload = self._stage_upload(upload)
        try:
            document_id = uuid.uuid4()
            ingest_job_id = uuid.uuid4()
            source_artifact_id = artifact_id(
                ingest_job_id=ingest_job_id,
                artifact_type="source_pdf",
                parser="upload",
            )
            now = datetime.now(UTC)
            storage_path = f"documents/{document_id}/{ingest_job_id}/source.pdf"
            stored_artifact = self._storage.store_file(storage_path, staged_upload)
            document_title = title or filename.rsplit(".", 1)[0] or None

            try:
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
                            storage_ref=stored_artifact.storage_ref,
                            checksum=stored_artifact.checksum,
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
                    self._queue.enqueue_ingest(
                        connection,
                        ingest_job_id,
                        document_id,
                        headers=trace_headers,
                    )
            except Exception:
                self._storage.delete(stored_artifact.storage_ref)
                raise
        finally:
            staged_upload.close()

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

    def _stage_upload(self, upload: BinaryIO) -> BinaryIO:
        staged_upload = SpooledTemporaryFile(
            max_size=min(self._max_upload_bytes, _UPLOAD_CHUNK_SIZE),
            mode="w+b",
        )
        if hasattr(upload, "seek"):
            upload.seek(0)

        total_bytes = 0
        header = bytearray()
        while True:
            chunk = upload.read(_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break

            total_bytes += len(chunk)
            if total_bytes > self._max_upload_bytes:
                staged_upload.close()
                raise UploadTooLargeError(
                    f"uploaded file exceeds the {self._max_upload_bytes}-byte limit"
                )

            if len(header) < len(_PDF_MAGIC):
                header.extend(chunk[: len(_PDF_MAGIC) - len(header)])
            staged_upload.write(chunk)

        if total_bytes == 0:
            staged_upload.close()
            raise ValueError("uploaded file is empty")
        if bytes(header) != _PDF_MAGIC:
            staged_upload.close()
            raise ValueError("uploaded file must be a PDF")

        staged_upload.seek(0)
        return cast(BinaryIO, staged_upload)
