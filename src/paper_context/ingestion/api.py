from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from tempfile import SpooledTemporaryFile
from typing import Any, BinaryIO, TypeVar, cast

from sqlalchemy import insert, text
from sqlalchemy.engine import Engine

from paper_context.models import Document, DocumentArtifact, IngestJob
from paper_context.pagination import CursorError, decode_cursor, encode_cursor, fingerprint_payload
from paper_context.queue.contracts import IngestionQueue
from paper_context.schemas.api import (
    DocumentListResponse,
    DocumentOutlineNode,
    DocumentOutlineResponse,
    DocumentReplaceResponse,
    DocumentResult,
    DocumentTableRecord,
    DocumentTablesResponse,
    DocumentUploadResponse,
    IngestJobResponse,
    RetrievalFiltersInput,
    TableDetailResponse,
    TablePreviewModel,
)
from paper_context.storage.base import StorageInterface

from .identifiers import artifact_id

_PDF_MAGIC = b"%PDF-"
_UPLOAD_CHUNK_SIZE = 1024 * 1024
_CursorItem = TypeVar("_CursorItem")
_SUPERSEDED_FAILURE_CODE = "superseded_by_newer_ingest_job"
_SUPERSEDED_FAILURE_MESSAGE = "A newer ingest job superseded this run before processing began."


class UploadTooLargeError(ValueError):
    pass


class DocumentNotFoundError(LookupError):
    pass


class InvalidCursorError(ValueError):
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
            document_title = title or filename.rsplit(".", 1)[0] or None
            ingest_job = self._store_and_enqueue_document(
                document_id=document_id,
                filename=filename,
                staged_upload=staged_upload,
                title=document_title,
                trigger="upload",
                trace_headers=trace_headers,
                create_document_row=True,
            )
        finally:
            staged_upload.close()

        return DocumentUploadResponse(
            document_id=document_id,
            ingest_job_id=ingest_job.ingest_job_id,
            status=ingest_job.status,
        )

    def replace_document(
        self,
        document_id: uuid.UUID,
        *,
        filename: str,
        content_type: str | None,
        upload: BinaryIO,
        title: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
    ) -> DocumentReplaceResponse:
        del content_type
        staged_upload = self._stage_upload(upload)
        try:
            ingest_job = self._store_and_enqueue_document(
                document_id=document_id,
                filename=filename,
                staged_upload=staged_upload,
                title=title,
                trigger="replace",
                trace_headers=trace_headers,
                create_document_row=False,
            )
        finally:
            staged_upload.close()

        return DocumentReplaceResponse(
            document_id=document_id,
            ingest_job_id=ingest_job.ingest_job_id,
            status=ingest_job.status,
        )

    def list_documents(self, *, limit: int = 20, cursor: str | None = None) -> DocumentListResponse:
        with self._engine.begin() as connection:
            rows = (
                connection.execute(
                    text(
                        """
                        SELECT
                            documents.id AS document_id,
                            COALESCE(documents.title, 'Untitled document') AS title,
                            COALESCE(documents.authors, '[]'::jsonb) AS authors,
                            documents.publication_year,
                            COALESCE(documents.quant_tags, '{{}}'::jsonb) AS quant_tags,
                            documents.current_status,
                            active_run.index_version AS active_index_version,
                            documents.updated_at
                        FROM documents
                        LEFT JOIN LATERAL (
                            SELECT runs.index_version
                            FROM retrieval_index_runs runs
                            WHERE runs.document_id = documents.id
                              AND runs.status = 'ready'
                              AND runs.is_active = true
                            ORDER BY COALESCE(runs.activated_at, runs.created_at) DESC, runs.id DESC
                            LIMIT 1
                        ) active_run ON true
                        ORDER BY documents.updated_at DESC NULLS LAST, documents.id DESC
                        """
                    )
                )
                .mappings()
                .all()
            )
        documents = [self._row_to_document_result(row) for row in rows]
        page, next_cursor = self._page_items(
            items=documents,
            limit=limit,
            cursor=cursor,
            kind="documents:list",
            fingerprint=fingerprint_payload({"kind": "documents:list"}),
        )
        return DocumentListResponse(documents=list(page), next_cursor=next_cursor)

    def search_documents(
        self,
        *,
        query: str,
        filters: RetrievalFiltersInput | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> DocumentListResponse:
        filters = filters or RetrievalFiltersInput()
        stripped_query = query.strip()
        params: dict[str, object] = {
            "query": f"%{stripped_query}%" if stripped_query else None,
            "apply_document_ids": bool(filters.document_ids),
            "document_ids": list(filters.document_ids),
            "apply_publication_years": bool(filters.publication_years),
            "publication_years": list(filters.publication_years),
        }

        with self._engine.begin() as connection:
            rows = (
                connection.execute(
                    text(
                        """
                        SELECT
                            documents.id AS document_id,
                            COALESCE(documents.title, 'Untitled document') AS title,
                            COALESCE(documents.authors, '[]'::jsonb) AS authors,
                            documents.publication_year,
                            COALESCE(documents.quant_tags, '{{}}'::jsonb) AS quant_tags,
                            documents.current_status,
                            active_run.index_version AS active_index_version,
                            documents.updated_at
                        FROM documents
                        LEFT JOIN LATERAL (
                            SELECT runs.index_version
                            FROM retrieval_index_runs runs
                            WHERE runs.document_id = documents.id
                              AND runs.status = 'ready'
                              AND runs.is_active = true
                            ORDER BY COALESCE(runs.activated_at, runs.created_at) DESC, runs.id DESC
                            LIMIT 1
                        ) active_run ON true
                        WHERE (
                            :query IS NULL
                            OR (
                                COALESCE(documents.title, '') ILIKE :query
                                OR COALESCE(documents.abstract, '') ILIKE :query
                                OR COALESCE(documents.authors::text, '') ILIKE :query
                            )
                        )
                          AND (
                            :apply_document_ids = false
                            OR documents.id = ANY(CAST(:document_ids AS uuid[]))
                        )
                          AND (
                            :apply_publication_years = false
                            OR documents.publication_year = ANY(
                                CAST(:publication_years AS integer[])
                            )
                        )
                        ORDER BY documents.updated_at DESC NULLS LAST, documents.id DESC
                        """
                    ),
                    params,
                )
                .mappings()
                .all()
            )
        documents = [self._row_to_document_result(row) for row in rows]
        page, next_cursor = self._page_items(
            items=documents,
            limit=limit,
            cursor=cursor,
            kind="documents:search",
            fingerprint=fingerprint_payload(
                {
                    "kind": "documents:search",
                    "query": stripped_query,
                    "filters": filters.model_dump(mode="json"),
                }
            ),
        )
        return DocumentListResponse(documents=list(page), next_cursor=next_cursor)

    def get_document(self, document_id: uuid.UUID) -> DocumentResult | None:
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        SELECT
                            documents.id AS document_id,
                            COALESCE(documents.title, 'Untitled document') AS title,
                            COALESCE(documents.authors, '[]'::jsonb) AS authors,
                            documents.publication_year,
                            COALESCE(documents.quant_tags, '{}'::jsonb) AS quant_tags,
                            documents.current_status,
                            active_run.index_version AS active_index_version
                        FROM documents
                        LEFT JOIN LATERAL (
                            SELECT runs.index_version
                            FROM retrieval_index_runs runs
                            WHERE runs.document_id = documents.id
                              AND runs.status = 'ready'
                              AND runs.is_active = true
                            ORDER BY COALESCE(runs.activated_at, runs.created_at) DESC, runs.id DESC
                            LIMIT 1
                        ) active_run ON true
                        WHERE documents.id = :document_id
                        """
                    ),
                    {"document_id": document_id},
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            return None
        return self._row_to_document_result(row)

    def get_document_outline(self, document_id: uuid.UUID) -> DocumentOutlineResponse | None:
        with self._engine.begin() as connection:
            document_row = (
                connection.execute(
                    text(
                        """
                        SELECT id AS document_id, COALESCE(title, 'Untitled document') AS title
                        FROM documents
                        WHERE id = :document_id
                        """
                    ),
                    {"document_id": document_id},
                )
                .mappings()
                .one_or_none()
            )
            if document_row is None:
                return None
            section_rows = (
                connection.execute(
                    text(
                        """
                        SELECT
                            id AS section_id,
                            parent_section_id,
                            heading,
                            COALESCE(heading_path, '[]'::jsonb) AS section_path,
                            ordinal,
                            page_start,
                            page_end
                        FROM document_sections
                        WHERE document_id = :document_id
                        ORDER BY ordinal NULLS LAST, id
                        """
                    ),
                    {"document_id": document_id},
                )
                .mappings()
                .all()
            )

        nodes = {
            cast(uuid.UUID, row["section_id"]): DocumentOutlineNode.model_validate(dict(row))
            for row in section_rows
        }
        roots: list[DocumentOutlineNode] = []
        for row in section_rows:
            node = nodes[cast(uuid.UUID, row["section_id"])]
            parent_section_id = cast(uuid.UUID | None, row["parent_section_id"])
            if parent_section_id is None:
                roots.append(node)
                continue
            parent = nodes.get(parent_section_id)
            if parent is None:
                roots.append(node)
                continue
            parent.children.append(node)

        return DocumentOutlineResponse(
            document_id=cast(uuid.UUID, document_row["document_id"]),
            title=cast(str, document_row["title"]),
            sections=roots,
        )

    def get_document_tables(
        self,
        document_id: uuid.UUID,
    ) -> DocumentTablesResponse | None:
        with self._engine.begin() as connection:
            document_row = (
                connection.execute(
                    text(
                        """
                        SELECT id AS document_id, COALESCE(title, 'Untitled document') AS title
                        FROM documents
                        WHERE id = :document_id
                        """
                    ),
                    {"document_id": document_id},
                )
                .mappings()
                .one_or_none()
            )
            if document_row is None:
                return None
            rows = (
                connection.execute(
                    text(
                        """
                        SELECT
                            tables.id AS table_id,
                            tables.document_id,
                            tables.section_id,
                            COALESCE(documents.title, 'Untitled document') AS document_title,
                            COALESCE(sections.heading_path, '[]'::jsonb) AS section_path,
                            tables.caption,
                            tables.table_type,
                            COALESCE(tables.headers_json, '[]'::jsonb) AS headers_json,
                            COALESCE(tables.rows_json, '[]'::jsonb) AS rows_json,
                            tables.page_start,
                            tables.page_end,
                            sections.ordinal AS section_ordinal
                        FROM document_tables tables
                        JOIN documents
                            ON documents.id = tables.document_id
                        JOIN document_sections sections
                            ON sections.id = tables.section_id
                        WHERE tables.document_id = :document_id
                        ORDER BY sections.ordinal NULLS LAST, tables.id
                        """
                    ),
                    {"document_id": document_id},
                )
                .mappings()
                .all()
            )
        return DocumentTablesResponse(
            document_id=cast(uuid.UUID, document_row["document_id"]),
            title=cast(str, document_row["title"]),
            tables=[self._row_to_document_table_record(row) for row in rows],
        )

    def get_table(self, table_id: uuid.UUID) -> TableDetailResponse | None:
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        SELECT
                            tables.id AS table_id,
                            tables.document_id,
                            tables.section_id,
                            COALESCE(documents.title, 'Untitled document') AS document_title,
                            COALESCE(sections.heading_path, '[]'::jsonb) AS section_path,
                            tables.caption,
                            tables.table_type,
                            COALESCE(tables.headers_json, '[]'::jsonb) AS headers_json,
                            COALESCE(tables.rows_json, '[]'::jsonb) AS rows_json,
                            tables.page_start,
                            tables.page_end
                        FROM document_tables tables
                        JOIN documents
                            ON documents.id = tables.document_id
                        JOIN document_sections sections
                            ON sections.id = tables.section_id
                        WHERE tables.id = :table_id
                        """
                    ),
                    {"table_id": table_id},
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            return None
        headers = [str(value) for value in cast(list[object], row["headers_json"] or [])]
        rows = [
            [str(cell) for cell in cast(list[object], table_row)]
            for table_row in cast(list[list[object]], row["rows_json"] or [])
        ]
        return TableDetailResponse(
            table_id=cast(uuid.UUID, row["table_id"]),
            document_id=cast(uuid.UUID, row["document_id"]),
            section_id=cast(uuid.UUID, row["section_id"]),
            document_title=cast(str, row["document_title"]),
            section_path=cast(list[str], row["section_path"] or []),
            caption=cast(str | None, row["caption"]),
            table_type=cast(str | None, row["table_type"]),
            headers=headers,
            rows=rows,
            row_count=len(rows),
            page_start=cast(int | None, row["page_start"]),
            page_end=cast(int | None, row["page_end"]),
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

    def _store_and_enqueue_document(
        self,
        *,
        document_id: uuid.UUID,
        filename: str,
        staged_upload: BinaryIO,
        title: str | None,
        trigger: str,
        trace_headers: Mapping[str, str] | None,
        create_document_row: bool,
    ) -> DocumentUploadResponse:
        ingest_job_id = uuid.uuid4()
        source_artifact_id = artifact_id(
            ingest_job_id=ingest_job_id,
            artifact_type="source_pdf",
            parser="upload",
        )
        now = datetime.now(UTC)
        storage_path = f"documents/{document_id}/{ingest_job_id}/source.pdf"
        stored_artifact = self._storage.store_file(storage_path, staged_upload)

        try:
            with self._engine.begin() as connection:
                if create_document_row:
                    connection.execute(
                        insert(Document).values(
                            id=document_id,
                            title=title,
                            source_type="upload",
                            current_status="queued",
                            created_at=now,
                            updated_at=now,
                        )
                    )
                else:
                    updated_rows = connection.execute(
                        text(
                            """
                            UPDATE documents
                            SET current_status = 'queued',
                                updated_at = :updated_at,
                                title = COALESCE(:title, title)
                            WHERE id = :document_id
                            """
                        ),
                        {
                            "document_id": document_id,
                            "updated_at": now,
                            "title": title,
                        },
                    ).rowcount
                    if updated_rows == 0:
                        raise DocumentNotFoundError("document not found")
                    self._supersede_queued_jobs(
                        connection,
                        document_id=document_id,
                        now=now,
                    )

                connection.execute(
                    insert(IngestJob).values(
                        id=ingest_job_id,
                        document_id=document_id,
                        source_artifact_id=None,
                        status="queued",
                        trigger=trigger,
                        warnings=[],
                        created_at=now,
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
                        created_at=now,
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

        return DocumentUploadResponse(
            document_id=document_id,
            ingest_job_id=ingest_job_id,
            status="queued",
        )

    def _supersede_queued_jobs(
        self,
        connection,
        *,
        document_id: uuid.UUID,
        now: datetime,
    ) -> None:
        connection.execute(
            text(
                """
                UPDATE ingest_jobs
                SET status = 'failed',
                    failure_code = :failure_code,
                    failure_message = :failure_message,
                    started_at = COALESCE(started_at, :now),
                    finished_at = :now
                WHERE document_id = :document_id
                  AND status = 'queued'
                """
            ),
            {
                "document_id": document_id,
                "failure_code": _SUPERSEDED_FAILURE_CODE,
                "failure_message": _SUPERSEDED_FAILURE_MESSAGE,
                "now": now,
            },
        )

    def _page_items(
        self,
        *,
        items: Sequence[_CursorItem],
        limit: int,
        cursor: str | None,
        kind: str,
        fingerprint: str,
    ) -> tuple[tuple[_CursorItem, ...], str | None]:
        page_size = max(1, limit)
        start = 0
        if cursor is not None:
            try:
                payload = decode_cursor(cursor)
            except CursorError as exc:
                raise InvalidCursorError(str(exc)) from exc
            if payload.get("kind") != kind or payload.get("fingerprint") != fingerprint:
                raise InvalidCursorError("cursor does not match request")
            position = payload.get("position")
            if not isinstance(position, int) or position < 0:
                raise InvalidCursorError("cursor does not match request")
            start = position

        page = tuple(items[start : start + page_size])
        next_position = start + len(page)
        if next_position >= len(items):
            return page, None
        return page, encode_cursor(
            {
                "kind": kind,
                "fingerprint": fingerprint,
                "position": next_position,
            }
        )

    def _row_to_document_result(self, row: Any) -> DocumentResult:
        return DocumentResult(
            document_id=row["document_id"],
            title=row["title"],
            authors=cast(list[str], row["authors"] or []),
            publication_year=row["publication_year"],
            quant_tags=cast(dict[str, object], row["quant_tags"] or {}),
            current_status=row["current_status"],
            active_index_version=row["active_index_version"],
        )

    def _row_to_document_table_record(self, row: Any) -> DocumentTableRecord:
        headers = [str(value) for value in cast(list[object], row["headers_json"] or [])]
        rows = [
            [str(cell) for cell in cast(list[object], table_row)]
            for table_row in cast(list[list[object]], row["rows_json"] or [])
        ]
        return DocumentTableRecord(
            table_id=row["table_id"],
            document_id=row["document_id"],
            section_id=row["section_id"],
            document_title=row["document_title"],
            section_path=cast(list[str], row["section_path"] or []),
            caption=row["caption"],
            table_type=row["table_type"],
            preview=TablePreviewModel(
                headers=headers,
                rows=rows[:3],
                row_count=len(rows),
            ),
            page_start=row["page_start"],
            page_end=row["page_end"],
        )

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
