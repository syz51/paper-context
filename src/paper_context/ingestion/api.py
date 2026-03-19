from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from tempfile import SpooledTemporaryFile
from typing import Any, BinaryIO, cast

from sqlalchemy import Text as SqlText
from sqlalchemy import and_, bindparam, func, insert, lateral, literal, or_, select, true, update
from sqlalchemy import cast as sa_cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql.elements import ColumnElement

from paper_context.models import (
    Document,
    DocumentArtifact,
    DocumentRevision,
    DocumentSection,
    DocumentTable,
    IngestJob,
    RetrievalIndexRun,
)
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
_SUPERSEDED_FAILURE_CODE = "superseded_by_newer_ingest_job"
_SUPERSEDED_FAILURE_MESSAGE = "A newer ingest job superseded this run before processing began."
_MAX_DOCUMENT_PAGE_SIZE = 100


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
        page_size = self._normalize_document_limit(limit)
        fingerprint = fingerprint_payload({"kind": "documents:list"})
        cursor_state = self._decode_document_cursor(
            cursor=cursor,
            kind="documents:list",
            fingerprint=fingerprint,
        )
        statement = self._document_projection_statement(include_updated_at=True)
        statement = self._apply_document_cursor(statement, cursor_state)
        statement = statement.limit(page_size + 1)
        with self._engine.begin() as connection:
            rows = connection.execute(statement).mappings().all()
        return self._document_page_response(
            rows=rows,
            limit=page_size,
            kind="documents:list",
            fingerprint=fingerprint,
        )

    def search_documents(
        self,
        *,
        query: str,
        filters: RetrievalFiltersInput | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> DocumentListResponse:
        filters = filters or RetrievalFiltersInput()
        page_size = self._normalize_document_limit(limit)
        stripped_query = query.strip()
        fingerprint = fingerprint_payload(
            {
                "kind": "documents:search",
                "query": stripped_query,
                "filters": filters.model_dump(mode="json"),
            }
        )
        cursor_state = self._decode_document_cursor(
            cursor=cursor,
            kind="documents:search",
            fingerprint=fingerprint,
        )
        predicates: list[ColumnElement[bool]] = []
        if stripped_query:
            predicates.append(
                func.to_tsvector("english", self._document_search_text()).op("@@")(
                    func.websearch_to_tsquery("english", stripped_query)
                )
            )
        if filters.document_ids:
            predicates.append(Document.id.in_(filters.document_ids))
        if filters.publication_years:
            predicates.append(Document.publication_year.in_(filters.publication_years))

        statement = self._document_projection_statement(
            *predicates,
            include_updated_at=True,
        )
        statement = self._apply_document_cursor(statement, cursor_state)
        statement = statement.limit(page_size + 1)
        with self._engine.begin() as connection:
            rows = connection.execute(statement).mappings().all()
        return self._document_page_response(
            rows=rows,
            limit=page_size,
            kind="documents:search",
            fingerprint=fingerprint,
        )

    def get_document(self, document_id: uuid.UUID) -> DocumentResult | None:
        statement = self._document_projection_statement(Document.id == document_id)
        with self._engine.begin() as connection:
            row = connection.execute(statement).mappings().one_or_none()
        if row is None:
            return None
        return self._row_to_document_result(row)

    def get_document_outline(self, document_id: uuid.UUID) -> DocumentOutlineResponse | None:
        active_revision_id = (
            select(Document.active_revision_id).where(Document.id == document_id).scalar_subquery()
        )
        with self._engine.begin() as connection:
            document_row = (
                connection.execute(self._document_title_statement(document_id=document_id))
                .mappings()
                .one_or_none()
            )
            if document_row is None:
                return None
            section_rows = (
                connection.execute(
                    select(
                        DocumentSection.id.label("section_id"),
                        DocumentSection.parent_section_id,
                        DocumentSection.heading,
                        func.coalesce(
                            DocumentSection.heading_path,
                            sa_cast([], JSONB),
                        ).label("section_path"),
                        DocumentSection.ordinal,
                        DocumentSection.page_start,
                        DocumentSection.page_end,
                    )
                    .where(
                        DocumentSection.document_id == document_id,
                        DocumentSection.revision_id == active_revision_id,
                    )
                    .order_by(DocumentSection.ordinal.asc().nullslast(), DocumentSection.id)
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
        active_revision_id = (
            select(Document.active_revision_id).where(Document.id == document_id).scalar_subquery()
        )
        with self._engine.begin() as connection:
            document_row = (
                connection.execute(self._document_title_statement(document_id=document_id))
                .mappings()
                .one_or_none()
            )
            if document_row is None:
                return None
            rows = (
                connection.execute(
                    self._document_table_projection_statement(
                        DocumentTable.document_id == document_id,
                        DocumentTable.revision_id == active_revision_id,
                    )
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
                    self._document_table_projection_statement(
                        DocumentTable.id == table_id,
                        DocumentTable.revision_id
                        == select(Document.active_revision_id)
                        .where(Document.id == DocumentTable.document_id)
                        .scalar_subquery(),
                    )
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
                    select(
                        IngestJob.id,
                        IngestJob.document_id,
                        IngestJob.status,
                        IngestJob.failure_code,
                        IngestJob.failure_message,
                        func.coalesce(IngestJob.warnings, sa_cast([], JSONB)).label("warnings"),
                        IngestJob.started_at,
                        IngestJob.finished_at,
                        IngestJob.trigger,
                    ).where(IngestJob.id == ingest_job_id)
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
        revision_id = uuid.uuid4()
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
                    revision_number = 1
                else:
                    updated_rows = connection.execute(
                        update(Document)
                        .where(Document.id == document_id)
                        .values(
                            current_status="queued",
                            updated_at=now,
                        )
                    ).rowcount
                    if updated_rows == 0:
                        raise DocumentNotFoundError("document not found")
                    self._supersede_queued_jobs(
                        connection,
                        document_id=document_id,
                        now=now,
                    )
                    revision_number = self._next_revision_number(
                        connection,
                        document_id=document_id,
                    )

                connection.execute(
                    insert(DocumentRevision).values(
                        id=revision_id,
                        document_id=document_id,
                        revision_number=revision_number,
                        title=title,
                        authors=[],
                        abstract=None,
                        publication_year=None,
                        source_type="upload",
                        metadata_confidence=None,
                        quant_tags={},
                        status="queued",
                        source_artifact_id=None,
                        ingest_job_id=None,
                        activated_at=None,
                        superseded_at=None,
                        created_at=now,
                        updated_at=now,
                    )
                )

                connection.execute(
                    insert(IngestJob).values(
                        id=ingest_job_id,
                        document_id=document_id,
                        revision_id=revision_id,
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
                        revision_id=revision_id,
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
                    update(IngestJob)
                    .where(IngestJob.id == ingest_job_id)
                    .values(source_artifact_id=source_artifact_id)
                )
                connection.execute(
                    update(DocumentRevision)
                    .where(DocumentRevision.id == revision_id)
                    .values(
                        source_artifact_id=source_artifact_id,
                        ingest_job_id=ingest_job_id,
                    )
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

    def _next_revision_number(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
    ) -> int:
        max_revision = connection.execute(
            select(func.max(DocumentRevision.revision_number)).where(
                DocumentRevision.document_id == document_id
            )
        ).scalar_one()
        return int(max_revision or 0) + 1

    def _supersede_queued_jobs(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        now: datetime,
    ) -> None:
        superseded_rows = connection.execute(
            update(IngestJob)
            .where(
                IngestJob.document_id == bindparam("b_document_id"),
                IngestJob.status == "queued",
            )
            .values(
                status="failed",
                failure_code=bindparam("b_failure_code"),
                failure_message=bindparam("b_failure_message"),
                started_at=func.coalesce(IngestJob.started_at, bindparam("b_now")),
                finished_at=bindparam("b_now"),
            )
            .returning(IngestJob.id, IngestJob.revision_id),
            {
                "document_id": document_id,
                "b_document_id": document_id,
                "failure_code": _SUPERSEDED_FAILURE_CODE,
                "b_failure_code": _SUPERSEDED_FAILURE_CODE,
                "failure_message": _SUPERSEDED_FAILURE_MESSAGE,
                "b_failure_message": _SUPERSEDED_FAILURE_MESSAGE,
                "now": now,
                "b_now": now,
            },
        ).all()
        superseded_job_ids = tuple(cast(uuid.UUID, row[0]) for row in superseded_rows)
        superseded_revision_ids = tuple(cast(uuid.UUID, row[1]) for row in superseded_rows)
        if not superseded_revision_ids:
            return
        connection.execute(
            update(DocumentRevision)
            .where(DocumentRevision.id.in_(superseded_revision_ids))
            .values(
                status="failed",
                superseded_at=bindparam("b_now"),
                updated_at=bindparam("b_now"),
            ),
            {
                "now": now,
                "b_now": now,
            },
        )
        for ingest_job_id in superseded_job_ids:
            self._queue.delete_messages_for_ingest_job_id(connection, ingest_job_id)

    def _document_projection_statement(
        self,
        *predicates: ColumnElement[bool],
        include_updated_at: bool = False,
    ):
        active_run = lateral(
            select(
                RetrievalIndexRun.index_version.label("active_index_version"),
            )
            .where(
                RetrievalIndexRun.revision_id == Document.active_revision_id,
                RetrievalIndexRun.status == "ready",
                RetrievalIndexRun.is_active.is_(True),
            )
            .order_by(
                func.coalesce(
                    RetrievalIndexRun.activated_at,
                    RetrievalIndexRun.created_at,
                ).desc(),
                RetrievalIndexRun.id.desc(),
            )
            .limit(1)
        ).alias("active_run")
        columns = [
            Document.id.label("document_id"),
            func.coalesce(Document.title, "Untitled document").label("title"),
            func.coalesce(Document.authors, sa_cast([], JSONB)).label("authors"),
            Document.publication_year,
            func.coalesce(Document.quant_tags, sa_cast({}, JSONB)).label("quant_tags"),
            Document.current_status,
            active_run.c.active_index_version,
        ]
        if include_updated_at:
            columns.append(Document.updated_at)
        statement = (
            select(*columns)
            .select_from(Document)
            .outerjoin(active_run, true())
            .order_by(Document.updated_at.desc().nullslast(), Document.id.desc())
        )
        for predicate in predicates:
            statement = statement.where(predicate)
        return statement

    def _document_search_text(self):
        return (
            func.coalesce(Document.title, "")
            + literal(" ")
            + func.coalesce(Document.abstract, "")
            + literal(" ")
            + func.translate(
                func.coalesce(sa_cast(Document.authors, SqlText), ""),
                '[]"',
                "   ",
            )
        )

    def _document_title_statement(self, *, document_id: uuid.UUID):
        return select(
            Document.id.label("document_id"),
            func.coalesce(Document.title, "Untitled document").label("title"),
        ).where(Document.id == document_id)

    def _document_table_projection_statement(self, *predicates: ColumnElement[bool]):
        statement = (
            select(
                DocumentTable.id.label("table_id"),
                DocumentTable.document_id,
                DocumentTable.section_id,
                func.coalesce(Document.title, "Untitled document").label("document_title"),
                func.coalesce(DocumentSection.heading_path, sa_cast([], JSONB)).label(
                    "section_path"
                ),
                DocumentTable.caption,
                DocumentTable.table_type,
                func.coalesce(DocumentTable.headers_json, sa_cast([], JSONB)).label("headers_json"),
                func.coalesce(DocumentTable.rows_json, sa_cast([], JSONB)).label("rows_json"),
                DocumentTable.page_start,
                DocumentTable.page_end,
                DocumentSection.ordinal.label("section_ordinal"),
            )
            .select_from(DocumentTable)
            .join(Document, Document.id == DocumentTable.document_id)
            .join(DocumentSection, DocumentSection.id == DocumentTable.section_id)
        )
        for predicate in predicates:
            statement = statement.where(predicate)
        return statement.order_by(DocumentSection.ordinal.asc().nullslast(), DocumentTable.id)

    def _normalize_document_limit(self, limit: int) -> int:
        return max(1, min(limit, _MAX_DOCUMENT_PAGE_SIZE))

    def _decode_document_cursor(
        self,
        *,
        cursor: str | None,
        kind: str,
        fingerprint: str,
    ) -> tuple[datetime, uuid.UUID] | None:
        if cursor is None:
            return None
        try:
            payload = decode_cursor(cursor)
        except CursorError as exc:
            raise InvalidCursorError(str(exc)) from exc
        if payload.get("kind") != kind or payload.get("fingerprint") != fingerprint:
            raise InvalidCursorError("cursor does not match request")
        updated_at_raw = payload.get("updated_at")
        document_id_raw = payload.get("document_id")
        if not isinstance(updated_at_raw, str) or not isinstance(document_id_raw, str):
            raise InvalidCursorError("cursor does not match request")
        try:
            updated_at = datetime.fromisoformat(updated_at_raw)
            document_id = uuid.UUID(document_id_raw)
        except ValueError as exc:
            raise InvalidCursorError("cursor does not match request") from exc
        return updated_at, document_id

    def _apply_document_cursor(
        self,
        statement,
        cursor_state: tuple[datetime, uuid.UUID] | None,
    ):
        if cursor_state is None:
            return statement
        updated_at, document_id = cursor_state
        return statement.where(
            or_(
                Document.updated_at < updated_at,
                and_(Document.updated_at == updated_at, Document.id < document_id),
            )
        )

    def _document_page_response(
        self,
        *,
        rows: Sequence[Any],
        limit: int,
        kind: str,
        fingerprint: str,
    ) -> DocumentListResponse:
        page_rows = rows[:limit]
        documents = [self._row_to_document_result(row) for row in page_rows]
        next_cursor: str | None = None
        if len(rows) > limit and page_rows:
            last_row = page_rows[-1]
            updated_at = cast(datetime, last_row["updated_at"])
            document_id = cast(uuid.UUID, last_row["document_id"])
            next_cursor = encode_cursor(
                {
                    "kind": kind,
                    "fingerprint": fingerprint,
                    "updated_at": updated_at.isoformat(),
                    "document_id": str(document_id),
                }
            )
        return DocumentListResponse(documents=documents, next_cursor=next_cursor)

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
