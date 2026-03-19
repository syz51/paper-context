from __future__ import annotations

import json
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast

from sqlalchemy import and_, bindparam, column, delete, func, insert, or_, select, table, update
from sqlalchemy import cast as sa_cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection
from sqlalchemy.sql import text

from paper_context.models import (
    Document,
    DocumentPassage,
    DocumentRevision,
    DocumentSection,
    DocumentTable,
    IngestJob,
    RetrievalIndexRun,
    RetrievalPassageAsset,
    RetrievalTableAsset,
)
from paper_context.pagination import CursorError, decode_cursor, encode_cursor, fingerprint_payload

from .clients import (
    DeterministicEmbeddingClient,
    HeuristicRerankerClient,
)
from .types import (
    ContextPackProvenance,
    ContextPackResult,
    ContextPassage,
    DocumentSummary,
    EmbeddingClient,
    MixedIndexVersionError,
    ParentSectionResult,
    PassageContextResult,
    PassageContextTarget,
    PassageResult,
    RerankerClient,
    RetrievalError,
    RetrievalFilters,
    RetrievalMode,
    SearchPage,
    TableDetailResult,
    TablePreview,
    TableResult,
)

ConnectionFactory = Callable[[], AbstractContextManager[Connection]]

PASSAGE_SPARSE_CANDIDATES = 30
PASSAGE_DENSE_CANDIDATES = 30
PASSAGE_FUSED_CANDIDATES = 40
PASSAGE_RESULT_LIMIT = 8
TABLE_SPARSE_CANDIDATES = 20
TABLE_FUSED_CANDIDATES = 20
TABLE_RESULT_LIMIT = 5
RRF_K = 60
PARENT_SIBLINGS_BEFORE = 1
PARENT_SIBLINGS_AFTER = 1
TABLE_PREVIEW_ROWS = 3
EMBEDDING_DIMENSIONS = 1024
INDEX_BUILD_BATCH_SIZE = 128
_RETRIEVAL_NAMESPACE = uuid.UUID("68b1ce8d-1a1e-49f5-b1db-47052d7ece6c")

_DOCUMENTS_SQL = table(
    "documents",
    column("id"),
    column("active_revision_id"),
    column("title"),
    column("authors"),
    column("abstract"),
    column("publication_year"),
    column("quant_tags"),
    column("current_status"),
    column("updated_at"),
)
_DOCUMENT_REVISIONS_SQL = table(
    "document_revisions",
    column("id"),
    column("document_id"),
    column("title"),
    column("authors"),
    column("abstract"),
    column("publication_year"),
    column("quant_tags"),
    column("status"),
    column("updated_at"),
)
_DOCUMENT_SECTIONS_SQL = table(
    "document_sections",
    column("id"),
    column("document_id"),
    column("revision_id"),
    column("parent_section_id"),
    column("heading"),
    column("heading_path"),
    column("ordinal"),
    column("page_start"),
    column("page_end"),
    column("artifact_id"),
)
_DOCUMENT_PASSAGES_SQL = table(
    "document_passages",
    column("id"),
    column("document_id"),
    column("revision_id"),
    column("section_id"),
    column("chunk_ordinal"),
    column("body_text"),
    column("contextualized_text"),
    column("token_count"),
    column("page_start"),
    column("page_end"),
    column("provenance_offsets"),
    column("quant_tags"),
    column("artifact_id"),
)
_DOCUMENT_TABLES_SQL = table(
    "document_tables",
    column("id"),
    column("document_id"),
    column("revision_id"),
    column("section_id"),
    column("caption"),
    column("table_type"),
    column("headers_json"),
    column("rows_json"),
    column("page_start"),
    column("page_end"),
    column("quant_tags"),
    column("artifact_id"),
)
_RETRIEVAL_INDEX_RUNS_SQL = table(
    "retrieval_index_runs",
    column("id"),
    column("document_id"),
    column("revision_id"),
    column("ingest_job_id"),
    column("index_version"),
    column("embedding_provider"),
    column("embedding_model"),
    column("embedding_dimensions"),
    column("reranker_provider"),
    column("reranker_model"),
    column("chunking_version"),
    column("parser_source"),
    column("status"),
    column("is_active"),
    column("activated_at"),
    column("deactivated_at"),
    column("created_at"),
)
_RETRIEVAL_PASSAGE_ASSETS_SQL = table(
    "retrieval_passage_assets",
    column("id"),
    column("retrieval_index_run_id"),
    column("revision_id"),
    column("document_id"),
    column("passage_id"),
    column("section_id"),
    column("publication_year"),
    column("search_text"),
    column("search_tsvector"),
    column("embedding"),
    column("created_at"),
)
_RETRIEVAL_TABLE_ASSETS_SQL = table(
    "retrieval_table_assets",
    column("id"),
    column("retrieval_index_run_id"),
    column("revision_id"),
    column("document_id"),
    column("table_id"),
    column("section_id"),
    column("publication_year"),
    column("search_text"),
    column("search_tsvector"),
    column("created_at"),
)


def _json_dumps(value: object) -> str:
    return json.dumps(value)


def _vector_literal(values: tuple[float, ...] | list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _retrieval_index_run_id(*, ingest_job_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(_RETRIEVAL_NAMESPACE, str(ingest_job_id))


def _normalize_modes(modes: set[str]) -> tuple[RetrievalMode, ...]:
    ordered: list[RetrievalMode] = []
    for mode in ("sparse", "dense"):
        if mode in modes:
            ordered.append(cast(RetrievalMode, mode))
    return tuple(ordered)


def _dedupe_warnings(warnings: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for warning in warnings:
        if warning not in seen:
            seen.add(warning)
            ordered.append(warning)
    return tuple(ordered)


def _revision_match(left_revision_id, right_revision_id):
    return or_(
        left_revision_id == right_revision_id,
        and_(left_revision_id.is_(None), right_revision_id.is_(None)),
    )


@dataclass
class _PassageIndexRow:
    passage_id: uuid.UUID
    document_id: uuid.UUID
    revision_id: uuid.UUID
    section_id: uuid.UUID
    chunk_ordinal: int
    body_text: str
    contextualized_text: str
    page_start: int | None
    page_end: int | None
    document_title: str
    authors: tuple[str, ...]
    abstract: str | None
    publication_year: int | None
    section_path: tuple[str, ...]


@dataclass
class _TableIndexRow:
    table_id: uuid.UUID
    document_id: uuid.UUID
    revision_id: uuid.UUID
    section_id: uuid.UUID
    caption: str | None
    table_type: str | None
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    page_start: int | None
    page_end: int | None
    document_title: str
    publication_year: int | None
    section_path: tuple[str, ...]


@dataclass
class _Candidate:
    entity_kind: Literal["passage", "table"]
    entity_id: uuid.UUID
    document_id: uuid.UUID
    section_id: uuid.UUID
    document_title: str
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    retrieval_index_run_id: uuid.UUID
    index_version: str
    warnings: tuple[str, ...]
    rerank_text: str
    parser_source: str | None = None
    retrieval_modes: set[str] = field(default_factory=set)
    fused_score: float = 0.0
    score: float = 0.0
    passage_id: uuid.UUID | None = None
    body_text: str | None = None
    chunk_ordinal: int | None = None
    table_id: uuid.UUID | None = None
    caption: str | None = None
    table_type: str | None = None
    preview: TablePreview | None = None


@dataclass(frozen=True)
class _ActiveRunSelection:
    run_ids: tuple[uuid.UUID, ...]
    index_versions: tuple[str, ...]


class DocumentRetrievalIndexer:
    def __init__(
        self,
        *,
        index_version: str,
        chunking_version: str,
        embedding_model: str,
        reranker_model: str,
        embedding_client: EmbeddingClient | None = None,
        reranker_client: RerankerClient | None = None,
    ) -> None:
        self.index_version = index_version
        self.chunking_version = chunking_version
        self.embedding_client = embedding_client or DeterministicEmbeddingClient(
            model=embedding_model
        )
        self.reranker_client = reranker_client or HeuristicRerankerClient(model=reranker_model)

    def rebuild(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        ingest_job_id: uuid.UUID,
        parser_source: str,
    ) -> uuid.UUID:
        run_id = _retrieval_index_run_id(ingest_job_id=ingest_job_id)
        now = datetime.now(UTC)
        self._upsert_build_run(
            connection,
            run_id=run_id,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            parser_source=parser_source,
            created_at=now,
        )
        embedding_dimensions: int | None = None
        self._clear_existing_assets(connection, run_id=run_id)
        for passages in self._iter_passage_row_batches(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            batch_size=INDEX_BUILD_BATCH_SIZE,
        ):
            passage_embeddings = self.embedding_client.embed(
                [row.contextualized_text for row in passages],
                input_type="document",
            )
            batch_dimensions = passage_embeddings.dimensions
            if batch_dimensions != EMBEDDING_DIMENSIONS:
                raise RetrievalError(
                    "embedding dimension mismatch for retrieval assets: "
                    f"expected {EMBEDDING_DIMENSIONS}, got {batch_dimensions}"
                )
            if embedding_dimensions is None:
                embedding_dimensions = batch_dimensions
            elif embedding_dimensions != batch_dimensions:
                raise RetrievalError(
                    "embedding dimension mismatch across retrieval asset batches: "
                    f"expected {embedding_dimensions}, got {batch_dimensions}"
                )
            self._insert_passage_asset_batch(
                connection,
                run_id=run_id,
                rows=passages,
                embeddings=passage_embeddings.embeddings,
                created_at=now,
            )
        for tables in self._iter_table_row_batches(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            batch_size=INDEX_BUILD_BATCH_SIZE,
        ):
            self._insert_table_asset_batch(
                connection,
                run_id=run_id,
                rows=tables,
                created_at=now,
            )
        self._activate_build_run(
            connection,
            run_id=run_id,
            revision_id=revision_id,
            embedding_dimensions=embedding_dimensions,
            activated_at=now,
        )
        return run_id

    def _upsert_build_run(
        self,
        connection: Connection,
        *,
        run_id: uuid.UUID,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        ingest_job_id: uuid.UUID,
        parser_source: str,
        created_at: datetime,
    ) -> None:
        statement = pg_insert(RetrievalIndexRun).values(
            id=bindparam("b_id"),
            document_id=bindparam("b_document_id"),
            revision_id=bindparam("b_revision_id"),
            ingest_job_id=bindparam("b_ingest_job_id"),
            index_version=bindparam("b_index_version"),
            embedding_provider=bindparam("b_embedding_provider"),
            embedding_model=bindparam("b_embedding_model"),
            embedding_dimensions=None,
            reranker_provider=bindparam("b_reranker_provider"),
            reranker_model=bindparam("b_reranker_model"),
            chunking_version=bindparam("b_chunking_version"),
            parser_source=bindparam("b_parser_source"),
            status="building",
            is_active=False,
            activated_at=None,
            created_at=bindparam("b_created_at"),
        )
        connection.execute(
            statement.on_conflict_do_update(
                index_elements=[RetrievalIndexRun.id],
                set_={
                    "index_version": statement.excluded.index_version,
                    "embedding_provider": statement.excluded.embedding_provider,
                    "embedding_model": statement.excluded.embedding_model,
                    "reranker_provider": statement.excluded.reranker_provider,
                    "reranker_model": statement.excluded.reranker_model,
                    "chunking_version": statement.excluded.chunking_version,
                    "parser_source": statement.excluded.parser_source,
                    "status": "building",
                    "is_active": False,
                    "activated_at": None,
                },
            ),
            {
                "id": run_id,
                "b_id": run_id,
                "document_id": document_id,
                "b_document_id": document_id,
                "revision_id": revision_id,
                "b_revision_id": revision_id,
                "ingest_job_id": ingest_job_id,
                "b_ingest_job_id": ingest_job_id,
                "index_version": self.index_version,
                "b_index_version": self.index_version,
                "embedding_provider": self.embedding_client.provider,
                "b_embedding_provider": self.embedding_client.provider,
                "embedding_model": self.embedding_client.model,
                "b_embedding_model": self.embedding_client.model,
                "reranker_provider": self.reranker_client.provider,
                "b_reranker_provider": self.reranker_client.provider,
                "reranker_model": self.reranker_client.model,
                "b_reranker_model": self.reranker_client.model,
                "chunking_version": self.chunking_version,
                "b_chunking_version": self.chunking_version,
                "parser_source": parser_source,
                "b_parser_source": parser_source,
                "created_at": created_at,
                "b_created_at": created_at,
            },
        )

    def _clear_existing_assets(
        self,
        connection: Connection,
        *,
        run_id: uuid.UUID,
    ) -> None:
        connection.execute(
            delete(RetrievalPassageAsset).where(
                RetrievalPassageAsset.retrieval_index_run_id == run_id
            )
        )
        connection.execute(
            delete(RetrievalTableAsset).where(RetrievalTableAsset.retrieval_index_run_id == run_id)
        )

    def _activate_build_run(
        self,
        connection: Connection,
        *,
        run_id: uuid.UUID,
        revision_id: uuid.UUID,
        embedding_dimensions: int | None,
        activated_at: datetime,
    ) -> None:
        connection.execute(
            update(RetrievalIndexRun)
            .where(RetrievalIndexRun.revision_id == revision_id)
            .values(
                is_active=False,
                deactivated_at=activated_at,
            )
        )
        connection.execute(
            update(RetrievalIndexRun)
            .where(RetrievalIndexRun.id == run_id)
            .values(
                embedding_dimensions=embedding_dimensions,
                status="ready",
                is_active=True,
                activated_at=activated_at,
                deactivated_at=None,
            )
        )

    def _iter_passage_row_batches(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        batch_size: int,
    ) -> Iterator[list[_PassageIndexRow]]:
        offset = 0
        while True:
            rows = (
                connection.execute(
                    select(
                        DocumentPassage.id.label("passage_id"),
                        DocumentPassage.document_id,
                        DocumentPassage.revision_id,
                        DocumentPassage.section_id,
                        DocumentPassage.chunk_ordinal,
                        DocumentPassage.body_text,
                        DocumentPassage.contextualized_text,
                        DocumentPassage.page_start,
                        DocumentPassage.page_end,
                        func.coalesce(
                            DocumentRevision.title,
                            Document.title,
                            "Untitled document",
                        ).label("document_title"),
                        func.coalesce(
                            DocumentRevision.authors,
                            Document.authors,
                            sa_cast([], JSONB),
                        ).label("authors"),
                        func.coalesce(DocumentRevision.abstract, Document.abstract).label(
                            "abstract"
                        ),
                        func.coalesce(
                            DocumentRevision.publication_year,
                            Document.publication_year,
                        ).label("publication_year"),
                        func.coalesce(DocumentSection.heading_path, sa_cast([], JSONB)).label(
                            "section_path"
                        ),
                    )
                    .select_from(DocumentPassage)
                    .join(Document, Document.id == DocumentPassage.document_id)
                    .outerjoin(DocumentRevision, DocumentRevision.id == DocumentPassage.revision_id)
                    .join(DocumentSection, DocumentSection.id == DocumentPassage.section_id)
                    .where(
                        DocumentPassage.document_id == document_id,
                        DocumentPassage.revision_id == revision_id,
                    )
                    .order_by(
                        DocumentSection.ordinal,
                        DocumentPassage.chunk_ordinal,
                        DocumentPassage.id,
                    )
                    .limit(batch_size)
                    .offset(offset)
                )
                .mappings()
                .all()
            )
            if not rows:
                return
            yield [self._row_to_passage_index_row(row) for row in rows]
            offset += len(rows)

    def _iter_table_row_batches(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        batch_size: int,
    ) -> Iterator[list[_TableIndexRow]]:
        offset = 0
        while True:
            rows = (
                connection.execute(
                    select(
                        DocumentTable.id.label("table_id"),
                        DocumentTable.document_id,
                        DocumentTable.revision_id,
                        DocumentTable.section_id,
                        DocumentTable.caption,
                        DocumentTable.table_type,
                        func.coalesce(DocumentTable.headers_json, sa_cast([], JSONB)).label(
                            "headers_json"
                        ),
                        func.coalesce(DocumentTable.rows_json, sa_cast([], JSONB)).label(
                            "rows_json"
                        ),
                        DocumentTable.page_start,
                        DocumentTable.page_end,
                        func.coalesce(
                            DocumentRevision.title,
                            Document.title,
                            "Untitled document",
                        ).label("document_title"),
                        func.coalesce(
                            DocumentRevision.publication_year,
                            Document.publication_year,
                        ).label("publication_year"),
                        func.coalesce(DocumentSection.heading_path, sa_cast([], JSONB)).label(
                            "section_path"
                        ),
                    )
                    .select_from(DocumentTable)
                    .join(Document, Document.id == DocumentTable.document_id)
                    .outerjoin(DocumentRevision, DocumentRevision.id == DocumentTable.revision_id)
                    .join(DocumentSection, DocumentSection.id == DocumentTable.section_id)
                    .where(
                        DocumentTable.document_id == document_id,
                        DocumentTable.revision_id == revision_id,
                    )
                    .order_by(DocumentSection.ordinal, DocumentTable.id)
                    .limit(batch_size)
                    .offset(offset)
                )
                .mappings()
                .all()
            )
            if not rows:
                return
            yield [self._row_to_table_index_row(row) for row in rows]
            offset += len(rows)

    def _insert_passage_asset_batch(
        self,
        connection: Connection,
        *,
        run_id: uuid.UUID,
        rows: list[_PassageIndexRow],
        embeddings: tuple[tuple[float, ...], ...],
        created_at: datetime,
    ) -> None:
        statement = insert(RetrievalPassageAsset).values(
            id=bindparam("b_id"),
            retrieval_index_run_id=bindparam("b_retrieval_index_run_id"),
            passage_id=bindparam("b_passage_id"),
            document_id=bindparam("b_document_id"),
            revision_id=bindparam("b_revision_id"),
            section_id=bindparam("b_section_id"),
            publication_year=bindparam("b_publication_year"),
            search_text=bindparam("b_search_text"),
            search_tsvector=func.to_tsvector("english", bindparam("b_search_text")),
            embedding=bindparam("b_embedding"),
            created_at=bindparam("b_created_at"),
        )
        payloads: list[dict[str, object]] = []
        for row, embedding in zip(rows, embeddings, strict=True):
            asset_id = uuid.uuid4()
            search_text = self._build_passage_search_text(row)
            embedding_literal = _vector_literal(embedding)
            payloads.append(
                {
                    "id": asset_id,
                    "b_id": asset_id,
                    "retrieval_index_run_id": run_id,
                    "b_retrieval_index_run_id": run_id,
                    "passage_id": row.passage_id,
                    "b_passage_id": row.passage_id,
                    "document_id": row.document_id,
                    "b_document_id": row.document_id,
                    "revision_id": row.revision_id,
                    "b_revision_id": row.revision_id,
                    "section_id": row.section_id,
                    "b_section_id": row.section_id,
                    "publication_year": row.publication_year,
                    "b_publication_year": row.publication_year,
                    "search_text": search_text,
                    "b_search_text": search_text,
                    "embedding": embedding_literal,
                    "b_embedding": embedding_literal,
                    "created_at": created_at,
                    "b_created_at": created_at,
                }
            )
        connection.execute(
            statement,
            payloads,
        )

    def _insert_table_asset_batch(
        self,
        connection: Connection,
        *,
        run_id: uuid.UUID,
        rows: list[_TableIndexRow],
        created_at: datetime,
    ) -> None:
        statement = insert(RetrievalTableAsset).values(
            id=bindparam("b_id"),
            retrieval_index_run_id=bindparam("b_retrieval_index_run_id"),
            table_id=bindparam("b_table_id"),
            document_id=bindparam("b_document_id"),
            revision_id=bindparam("b_revision_id"),
            section_id=bindparam("b_section_id"),
            publication_year=bindparam("b_publication_year"),
            search_text=bindparam("b_search_text"),
            search_tsvector=func.to_tsvector("english", bindparam("b_search_text")),
            created_at=bindparam("b_created_at"),
        )
        payloads: list[dict[str, object]] = []
        for row in rows:
            asset_id = uuid.uuid4()
            search_text = self._build_table_search_text(row)
            payloads.append(
                {
                    "id": asset_id,
                    "b_id": asset_id,
                    "retrieval_index_run_id": run_id,
                    "b_retrieval_index_run_id": run_id,
                    "table_id": row.table_id,
                    "b_table_id": row.table_id,
                    "document_id": row.document_id,
                    "b_document_id": row.document_id,
                    "revision_id": row.revision_id,
                    "b_revision_id": row.revision_id,
                    "section_id": row.section_id,
                    "b_section_id": row.section_id,
                    "publication_year": row.publication_year,
                    "b_publication_year": row.publication_year,
                    "search_text": search_text,
                    "b_search_text": search_text,
                    "created_at": created_at,
                    "b_created_at": created_at,
                }
            )
        connection.execute(
            statement,
            payloads,
        )

    def _row_to_passage_index_row(self, row: Any) -> _PassageIndexRow:
        return _PassageIndexRow(
            passage_id=row["passage_id"],
            document_id=row["document_id"],
            revision_id=row["revision_id"],
            section_id=row["section_id"],
            chunk_ordinal=row["chunk_ordinal"],
            body_text=row["body_text"],
            contextualized_text=row["contextualized_text"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            document_title=row["document_title"],
            authors=tuple(cast(list[str], row["authors"] or [])),
            abstract=row["abstract"],
            publication_year=row["publication_year"],
            section_path=tuple(cast(list[str], row["section_path"] or [])),
        )

    def _row_to_table_index_row(self, row: Any) -> _TableIndexRow:
        return _TableIndexRow(
            table_id=row["table_id"],
            document_id=row["document_id"],
            revision_id=row["revision_id"],
            section_id=row["section_id"],
            caption=row["caption"],
            table_type=row["table_type"],
            headers=tuple(str(header) for header in cast(list[object], row["headers_json"] or [])),
            rows=tuple(
                tuple(str(cell) for cell in cast(list[object], table_row))
                for table_row in cast(list[list[object]], row["rows_json"] or [])
            ),
            page_start=row["page_start"],
            page_end=row["page_end"],
            document_title=row["document_title"],
            publication_year=row["publication_year"],
            section_path=tuple(cast(list[str], row["section_path"] or [])),
        )

    def _build_passage_search_text(self, row: _PassageIndexRow) -> str:
        metadata_lines: list[str] = [row.contextualized_text]
        if row.authors:
            metadata_lines.append(f"Document authors: {', '.join(row.authors)}")
        if row.publication_year is not None:
            metadata_lines.append(f"Publication year: {row.publication_year}")
        if row.abstract:
            metadata_lines.append(f"Abstract: {row.abstract}")
        return "\n".join(metadata_lines)

    def _build_table_search_text(self, row: _TableIndexRow) -> str:
        header_text = " | ".join(row.headers)
        row_lines = [" | ".join(table_row) for table_row in row.rows]
        section_path = " > ".join(row.section_path) if row.section_path else "Body"
        lines = [
            f"Document title: {row.document_title}",
            f"Section path: {section_path}",
        ]
        if row.caption:
            lines.append(f"Table caption: {row.caption}")
        if header_text:
            lines.append(f"Table headers: {header_text}")
        if row_lines:
            lines.append("Table rows:")
            lines.extend(row_lines)
        return "\n".join(lines)


class RetrievalService:
    def __init__(
        self,
        *,
        connection_factory: ConnectionFactory | None = None,
        active_index_version: str | None = None,
        embedding_client: EmbeddingClient | None = None,
        reranker_client: RerankerClient | None = None,
    ) -> None:
        self._connection_factory = connection_factory
        self._active_index_version = active_index_version
        self._embedding_client = embedding_client
        self._reranker_client = reranker_client

    def health_summary(self) -> dict[str, str]:
        if self._connection_factory is None:
            return {"status": "not-configured"}
        if self._embedding_client is None or self._reranker_client is None:
            return {"status": "partially-configured"}
        return {
            "status": "configured",
            "embedding_provider": self._embedding_client.provider,
            "reranker_provider": self._reranker_client.provider,
            "active_index_version": self._active_index_version or "auto",
        }

    def search_passages(
        self,
        *,
        query: str,
        filters: RetrievalFilters | None = None,
        limit: int = PASSAGE_RESULT_LIMIT,
    ) -> list[PassageResult]:
        filters = filters or RetrievalFilters()
        with self._connection() as connection:
            return self._search_passages_with_connection(
                connection,
                query=query,
                filters=filters,
                limit=limit,
            )

    def search_tables(
        self,
        *,
        query: str,
        filters: RetrievalFilters | None = None,
        limit: int = TABLE_RESULT_LIMIT,
    ) -> list[TableResult]:
        filters = filters or RetrievalFilters()
        with self._connection() as connection:
            return self._search_tables_with_connection(
                connection,
                query=query,
                filters=filters,
                limit=limit,
            )

    def search_passages_page(
        self,
        *,
        query: str,
        filters: RetrievalFilters | None = None,
        cursor: str | None = None,
        limit: int = PASSAGE_RESULT_LIMIT,
    ) -> SearchPage[PassageResult]:
        filters = filters or RetrievalFilters()
        fingerprint = fingerprint_payload(
            {
                "kind": "passages",
                "query": query.strip(),
                "filters": {
                    "document_ids": [str(document_id) for document_id in filters.document_ids],
                    "publication_years": list(filters.publication_years),
                },
            }
        )
        with self._connection() as connection:
            filtered_document_ids = self._resolve_filtered_document_ids(
                connection,
                filters=filters,
            )
            active_runs = self._resolve_active_run_selection(
                connection,
                filtered_document_ids=filtered_document_ids,
            )
            results = tuple(
                self._search_passages_with_connection(
                    connection,
                    query=query,
                    filters=filters,
                    limit=PASSAGE_FUSED_CANDIDATES,
                    filtered_document_ids=filtered_document_ids,
                    active_runs=active_runs,
                )
            )
            return cast(
                SearchPage[PassageResult],
                self._paginate_ranked_results(
                    kind="passages",
                    results=results,
                    limit=limit,
                    cursor=cursor,
                    active_runs=active_runs,
                    fingerprint=fingerprint,
                ),
            )

    def search_tables_page(
        self,
        *,
        query: str,
        filters: RetrievalFilters | None = None,
        cursor: str | None = None,
        limit: int = TABLE_RESULT_LIMIT,
    ) -> SearchPage[TableResult]:
        filters = filters or RetrievalFilters()
        fingerprint = fingerprint_payload(
            {
                "kind": "tables",
                "query": query.strip(),
                "filters": {
                    "document_ids": [str(document_id) for document_id in filters.document_ids],
                    "publication_years": list(filters.publication_years),
                },
            }
        )
        with self._connection() as connection:
            filtered_document_ids = self._resolve_filtered_document_ids(
                connection,
                filters=filters,
            )
            active_runs = self._resolve_active_run_selection(
                connection,
                filtered_document_ids=filtered_document_ids,
            )
            results = tuple(
                self._search_tables_with_connection(
                    connection,
                    query=query,
                    filters=filters,
                    limit=TABLE_FUSED_CANDIDATES,
                    filtered_document_ids=filtered_document_ids,
                    active_runs=active_runs,
                )
            )
            return cast(
                SearchPage[TableResult],
                self._paginate_ranked_results(
                    kind="tables",
                    results=results,
                    limit=limit,
                    cursor=cursor,
                    active_runs=active_runs,
                    fingerprint=fingerprint,
                ),
            )

    def build_context_pack(
        self,
        *,
        query: str,
        filters: RetrievalFilters | None = None,
        cursor: str | None = None,
        limit: int = PASSAGE_RESULT_LIMIT,
    ) -> ContextPackResult:
        filters = filters or RetrievalFilters()
        fingerprint = fingerprint_payload(
            {
                "kind": "context_pack",
                "query": query.strip(),
                "filters": {
                    "document_ids": [str(document_id) for document_id in filters.document_ids],
                    "publication_years": list(filters.publication_years),
                },
            }
        )
        with self._connection() as connection:
            filtered_document_ids = self._resolve_filtered_document_ids(
                connection,
                filters=filters,
            )
            active_runs = self._resolve_active_run_selection(
                connection,
                filtered_document_ids=filtered_document_ids,
            )
            all_passages = tuple(
                self._search_passages_with_connection(
                    connection,
                    query=query,
                    filters=filters,
                    limit=PASSAGE_FUSED_CANDIDATES,
                    filtered_document_ids=filtered_document_ids,
                    active_runs=active_runs,
                )
            )
            passage_page = cast(
                SearchPage[PassageResult],
                self._paginate_ranked_results(
                    kind="context_pack",
                    results=all_passages,
                    limit=limit,
                    cursor=cursor,
                    active_runs=active_runs,
                    fingerprint=fingerprint,
                ),
            )
            passages = passage_page.items
            pack_document_ids = tuple(dict.fromkeys(result.document_id for result in passages))
            tables = tuple(
                self._search_tables_with_connection(
                    connection,
                    query=query,
                    filters=filters,
                    limit=TABLE_RESULT_LIMIT,
                    filtered_document_ids=pack_document_ids or filtered_document_ids,
                    active_runs=active_runs,
                )
            )
            active_index_version = self._ensure_single_index_version([*passages, *tables])
            document_ids = tuple(
                dict.fromkeys(
                    [
                        *(result.document_id for result in passages),
                        *(result.document_id for result in tables),
                    ]
                )
            )
            retrieval_index_run_ids = tuple(
                dict.fromkeys(
                    [
                        *(result.retrieval_index_run_id for result in passages),
                        *(result.retrieval_index_run_id for result in tables),
                    ]
                )
            )
            parent_sections = self._load_parent_sections(
                connection,
                passages=passages,
                tables=tables,
            )
            documents = self._load_document_summaries(
                connection,
                document_ids=document_ids,
                active_index_version=active_index_version,
            )

        warnings = _dedupe_warnings(
            [
                *(warning for result in passages for warning in result.warnings),
                *(warning for result in tables for warning in result.warnings),
                *(warning for section in parent_sections for warning in section.warnings),
            ]
        )
        retrieval_modes = _dedupe_modes_from_results(passages=passages, tables=tables)
        provenance = ContextPackProvenance(
            active_index_version=active_index_version,
            retrieval_index_run_ids=retrieval_index_run_ids,
            retrieval_modes=retrieval_modes,
        )
        return ContextPackResult(
            context_pack_id=uuid.uuid4(),
            query=query,
            passages=passages,
            tables=tables,
            parent_sections=parent_sections,
            documents=documents,
            provenance=provenance,
            warnings=warnings,
            next_cursor=passage_page.next_cursor,
        )

    def get_table(self, *, table_id: uuid.UUID) -> TableDetailResult | None:
        run_join_conditions = [
            RetrievalIndexRun.document_id == DocumentTable.document_id,
            RetrievalIndexRun.revision_id == DocumentTable.revision_id,
            RetrievalIndexRun.status == "ready",
            RetrievalIndexRun.is_active.is_(True),
        ]
        if self._active_index_version is not None:
            run_join_conditions.append(
                RetrievalIndexRun.index_version == self._active_index_version
            )
        with self._connection() as connection:
            row = (
                connection.execute(
                    select(
                        DocumentTable.id.label("table_id"),
                        DocumentTable.document_id,
                        DocumentTable.section_id,
                        func.coalesce(
                            DocumentRevision.title,
                            Document.title,
                            "Untitled document",
                        ).label("document_title"),
                        func.coalesce(DocumentSection.heading_path, sa_cast([], JSONB)).label(
                            "section_path"
                        ),
                        DocumentTable.caption,
                        DocumentTable.table_type,
                        func.coalesce(DocumentTable.headers_json, sa_cast([], JSONB)).label(
                            "headers_json"
                        ),
                        func.coalesce(DocumentTable.rows_json, sa_cast([], JSONB)).label(
                            "rows_json"
                        ),
                        DocumentTable.page_start,
                        DocumentTable.page_end,
                        RetrievalIndexRun.id.label("retrieval_index_run_id"),
                        RetrievalIndexRun.index_version,
                        RetrievalIndexRun.parser_source,
                        func.coalesce(IngestJob.warnings, sa_cast([], JSONB)).label("warnings"),
                    )
                    .select_from(DocumentTable)
                    .join(Document, Document.id == DocumentTable.document_id)
                    .join(DocumentSection, DocumentSection.id == DocumentTable.section_id)
                    .outerjoin(
                        DocumentRevision,
                        _revision_match(DocumentRevision.id, DocumentTable.revision_id),
                    )
                    .outerjoin(RetrievalIndexRun, and_(*run_join_conditions))
                    .outerjoin(IngestJob, IngestJob.id == RetrievalIndexRun.ingest_job_id)
                    .where(
                        DocumentTable.id == table_id,
                        DocumentTable.revision_id == Document.active_revision_id,
                    )
                    .order_by(
                        func.coalesce(
                            RetrievalIndexRun.activated_at,
                            RetrievalIndexRun.created_at,
                        )
                        .desc()
                        .nullslast()
                    )
                    .limit(1)
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            return None
        headers = tuple(str(value) for value in cast(list[object], row["headers_json"] or []))
        rows = tuple(
            tuple(str(cell) for cell in cast(list[object], table_row))
            for table_row in cast(list[list[object]], row["rows_json"] or [])
        )
        return TableDetailResult(
            table_id=row["table_id"],
            document_id=row["document_id"],
            section_id=row["section_id"],
            document_title=row["document_title"],
            section_path=tuple(cast(list[str], row["section_path"] or [])),
            caption=row["caption"],
            table_type=row["table_type"],
            headers=headers,
            rows=rows,
            row_count=len(rows),
            page_start=row["page_start"],
            page_end=row["page_end"],
            index_version=row["index_version"],
            retrieval_index_run_id=row["retrieval_index_run_id"],
            parser_source=row["parser_source"],
            warnings=tuple(cast(list[str], row["warnings"] or [])),
        )

    def get_passage_context(
        self,
        *,
        passage_id: uuid.UUID,
        before: int = 1,
        after: int = 1,
    ) -> PassageContextResult | None:
        before = max(0, before)
        after = max(0, after)
        run_join_conditions = [
            RetrievalIndexRun.document_id == DocumentPassage.document_id,
            RetrievalIndexRun.revision_id == DocumentPassage.revision_id,
            RetrievalIndexRun.status == "ready",
            RetrievalIndexRun.is_active.is_(True),
        ]
        if self._active_index_version is not None:
            run_join_conditions.append(
                RetrievalIndexRun.index_version == self._active_index_version
            )
        with self._connection() as connection:
            row = (
                connection.execute(
                    select(
                        DocumentPassage.id.label("passage_id"),
                        DocumentPassage.document_id,
                        DocumentPassage.section_id,
                        DocumentPassage.body_text,
                        DocumentPassage.chunk_ordinal,
                        DocumentPassage.page_start,
                        DocumentPassage.page_end,
                        DocumentPassage.revision_id,
                        func.coalesce(
                            DocumentRevision.title,
                            Document.title,
                            "Untitled document",
                        ).label("document_title"),
                        func.coalesce(DocumentSection.heading_path, sa_cast([], JSONB)).label(
                            "section_path"
                        ),
                        RetrievalIndexRun.id.label("retrieval_index_run_id"),
                        RetrievalIndexRun.index_version,
                        RetrievalIndexRun.parser_source,
                        func.coalesce(IngestJob.warnings, sa_cast([], JSONB)).label("warnings"),
                    )
                    .select_from(DocumentPassage)
                    .join(DocumentSection, DocumentSection.id == DocumentPassage.section_id)
                    .join(Document, Document.id == DocumentPassage.document_id)
                    .outerjoin(
                        DocumentRevision,
                        _revision_match(DocumentRevision.id, DocumentPassage.revision_id),
                    )
                    .outerjoin(RetrievalIndexRun, and_(*run_join_conditions))
                    .outerjoin(IngestJob, IngestJob.id == RetrievalIndexRun.ingest_job_id)
                    .where(
                        DocumentPassage.id == passage_id,
                        DocumentPassage.revision_id == Document.active_revision_id,
                    )
                    .order_by(
                        func.coalesce(
                            RetrievalIndexRun.activated_at,
                            RetrievalIndexRun.created_at,
                        )
                        .desc()
                        .nullslast()
                    )
                    .limit(1)
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                return None

            section_rows = (
                connection.execute(
                    select(
                        DocumentPassage.id.label("passage_id"),
                        DocumentPassage.body_text,
                        DocumentPassage.chunk_ordinal,
                        DocumentPassage.page_start,
                        DocumentPassage.page_end,
                    )
                    .where(
                        DocumentPassage.section_id == row["section_id"],
                        DocumentPassage.revision_id == row["revision_id"],
                    )
                    .order_by(DocumentPassage.chunk_ordinal, DocumentPassage.id)
                )
                .mappings()
                .all()
            )

        selected_index = next(
            (
                index
                for index, section_row in enumerate(section_rows)
                if section_row["passage_id"] == passage_id
            ),
            None,
        )
        if selected_index is None:
            return None

        start = max(0, selected_index - before)
        end = min(len(section_rows), selected_index + after + 1)
        context_passages = tuple(
            ContextPassage(
                passage_id=section_row["passage_id"],
                text=section_row["body_text"],
                chunk_ordinal=section_row["chunk_ordinal"],
                page_start=section_row["page_start"],
                page_end=section_row["page_end"],
                relationship=cast(
                    Literal["selected", "sibling"],
                    "selected" if index == selected_index else "sibling",
                ),
            )
            for index, section_row in enumerate(section_rows[start:end], start=start)
        )
        warnings: list[str] = list(cast(list[str], row["warnings"] or []))
        if start > 0 or end < len(section_rows):
            warnings.append("parent_context_truncated")
        return PassageContextResult(
            passage=PassageContextTarget(
                passage_id=row["passage_id"],
                document_id=row["document_id"],
                section_id=row["section_id"],
                document_title=row["document_title"],
                section_path=tuple(cast(list[str], row["section_path"] or [])),
                text=row["body_text"],
                chunk_ordinal=row["chunk_ordinal"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                index_version=row["index_version"],
                retrieval_index_run_id=row["retrieval_index_run_id"],
                parser_source=row["parser_source"],
                warnings=tuple(cast(list[str], row["warnings"] or [])),
            ),
            context_passages=context_passages,
            warnings=_dedupe_warnings(warnings),
        )

    def _connection(self) -> AbstractContextManager[Connection]:
        if self._connection_factory is None:
            raise RetrievalError("retrieval service has no connection factory configured")
        return self._connection_factory()

    def _search_passages_with_connection(
        self,
        connection: Connection,
        *,
        query: str,
        filters: RetrievalFilters,
        limit: int,
        filtered_document_ids: tuple[uuid.UUID, ...] | None = None,
        active_runs: _ActiveRunSelection | None = None,
    ) -> list[PassageResult]:
        if not query.strip():
            return []
        if filtered_document_ids is None:
            filtered_document_ids = self._resolve_filtered_document_ids(
                connection,
                filters=filters,
            )
        if filtered_document_ids == ():
            return []
        if active_runs is None:
            active_runs = self._resolve_active_run_selection(
                connection,
                filtered_document_ids=filtered_document_ids,
            )
        if not active_runs.run_ids:
            return []
        sparse_candidates = self._load_sparse_passage_candidates(
            connection,
            query=query,
            retrieval_index_run_ids=active_runs.run_ids,
            filtered_document_ids=filtered_document_ids,
            limit=PASSAGE_SPARSE_CANDIDATES,
        )
        dense_candidates = self._load_dense_passage_candidates(
            connection,
            query=query,
            retrieval_index_run_ids=active_runs.run_ids,
            filtered_document_ids=filtered_document_ids,
            limit=PASSAGE_DENSE_CANDIDATES,
        )
        fused = self._fuse_candidates(
            sparse_candidates,
            dense_candidates,
            fused_limit=PASSAGE_FUSED_CANDIDATES,
        )
        reranked = self._rerank_candidates(query=query, candidates=fused, limit=limit)
        return [self._candidate_to_passage_result(candidate) for candidate in reranked[:limit]]

    def _search_tables_with_connection(
        self,
        connection: Connection,
        *,
        query: str,
        filters: RetrievalFilters,
        limit: int,
        filtered_document_ids: tuple[uuid.UUID, ...] | None = None,
        active_runs: _ActiveRunSelection | None = None,
    ) -> list[TableResult]:
        if not query.strip():
            return []
        if filtered_document_ids is None:
            filtered_document_ids = self._resolve_filtered_document_ids(
                connection,
                filters=filters,
            )
        if filtered_document_ids == ():
            return []
        if active_runs is None:
            active_runs = self._resolve_active_run_selection(
                connection,
                filtered_document_ids=filtered_document_ids,
            )
        if not active_runs.run_ids:
            return []
        sparse_candidates = self._load_sparse_table_candidates(
            connection,
            query=query,
            retrieval_index_run_ids=active_runs.run_ids,
            filtered_document_ids=filtered_document_ids,
            limit=TABLE_SPARSE_CANDIDATES,
        )
        fused = self._fuse_candidates(
            sparse_candidates,
            [],
            fused_limit=TABLE_FUSED_CANDIDATES,
        )
        reranked = self._rerank_candidates(query=query, candidates=fused, limit=limit)
        return [self._candidate_to_table_result(candidate) for candidate in reranked[:limit]]

    def _resolve_active_run_selection(
        self,
        connection: Connection,
        *,
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
    ) -> _ActiveRunSelection:
        if self._active_index_version is not None:
            run_ids = self._resolve_active_run_ids(
                connection,
                index_version=self._active_index_version,
                filtered_document_ids=filtered_document_ids,
            )
            index_versions = (self._active_index_version,) if run_ids else ()
            return _ActiveRunSelection(run_ids=run_ids, index_versions=index_versions)
        params = {
            "apply_document_filter": filtered_document_ids is not None,
            "document_ids": list(filtered_document_ids or ()),
        }
        statement = (
            select(
                _RETRIEVAL_INDEX_RUNS_SQL.c.id,
                _RETRIEVAL_INDEX_RUNS_SQL.c.index_version,
            )
            .select_from(
                _RETRIEVAL_INDEX_RUNS_SQL.join(
                    _DOCUMENTS_SQL,
                    _DOCUMENTS_SQL.c.id == _RETRIEVAL_INDEX_RUNS_SQL.c.document_id,
                )
            )
            .where(
                _RETRIEVAL_INDEX_RUNS_SQL.c.status == "ready",
                _RETRIEVAL_INDEX_RUNS_SQL.c.is_active.is_(True),
                or_(
                    bindparam("apply_document_filter") == False,  # noqa: E712
                    _DOCUMENTS_SQL.c.id.in_(bindparam("document_ids", expanding=True)),
                ),
                _revision_match(
                    _RETRIEVAL_INDEX_RUNS_SQL.c.revision_id,
                    _DOCUMENTS_SQL.c.active_revision_id,
                ),
            )
            .order_by(
                func.coalesce(
                    _RETRIEVAL_INDEX_RUNS_SQL.c.activated_at,
                    _RETRIEVAL_INDEX_RUNS_SQL.c.created_at,
                ).desc(),
                _RETRIEVAL_INDEX_RUNS_SQL.c.id,
            )
        )
        rows = connection.execute(statement, params).mappings().all()
        run_ids = tuple(cast(uuid.UUID, row["id"]) for row in rows)
        index_versions = tuple(dict.fromkeys(cast(str, row["index_version"]) for row in rows))
        return _ActiveRunSelection(run_ids=run_ids, index_versions=index_versions)

    def _resolve_filtered_document_ids(
        self,
        connection: Connection,
        *,
        filters: RetrievalFilters,
    ) -> tuple[uuid.UUID, ...] | None:
        if not filters.document_ids and not filters.publication_years:
            return None
        params = {
            "apply_document_ids": bool(filters.document_ids),
            "document_ids": list(filters.document_ids),
            "apply_publication_years": bool(filters.publication_years),
            "publication_years": list(filters.publication_years),
        }
        statement = text(
            """
            SELECT doc.id
            FROM documents doc
            LEFT JOIN document_revisions rev
              ON rev.id = doc.active_revision_id
            WHERE (
                    :apply_document_ids = FALSE
                    OR doc.id = ANY(CAST(:document_ids AS uuid[]))
                  )
              AND (
                    :apply_publication_years = FALSE
                    OR COALESCE(rev.publication_year, doc.publication_year)
                       = ANY(CAST(:publication_years AS int[]))
                  )
            ORDER BY doc.id
            """
        )
        rows = connection.execute(statement, params).scalars().all()
        return tuple(cast(list[uuid.UUID], rows))

    def _resolve_active_run_ids(
        self,
        connection: Connection,
        *,
        index_version: str,
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
    ) -> tuple[uuid.UUID, ...]:
        params = {
            "index_version": index_version,
            "apply_document_filter": filtered_document_ids is not None,
            "document_ids": list(filtered_document_ids or ()),
        }
        statement = (
            select(_RETRIEVAL_INDEX_RUNS_SQL.c.id)
            .select_from(
                _RETRIEVAL_INDEX_RUNS_SQL.join(
                    _DOCUMENTS_SQL,
                    _DOCUMENTS_SQL.c.id == _RETRIEVAL_INDEX_RUNS_SQL.c.document_id,
                )
            )
            .where(
                _RETRIEVAL_INDEX_RUNS_SQL.c.status == "ready",
                _RETRIEVAL_INDEX_RUNS_SQL.c.is_active.is_(True),
                _RETRIEVAL_INDEX_RUNS_SQL.c.index_version == bindparam("index_version"),
                or_(
                    bindparam("apply_document_filter") == False,  # noqa: E712
                    _DOCUMENTS_SQL.c.id.in_(bindparam("document_ids", expanding=True)),
                ),
                _revision_match(
                    _RETRIEVAL_INDEX_RUNS_SQL.c.revision_id,
                    _DOCUMENTS_SQL.c.active_revision_id,
                ),
            )
            .order_by(
                func.coalesce(
                    _RETRIEVAL_INDEX_RUNS_SQL.c.activated_at,
                    _RETRIEVAL_INDEX_RUNS_SQL.c.created_at,
                ).desc(),
                _RETRIEVAL_INDEX_RUNS_SQL.c.id,
            )
        )
        rows = connection.execute(statement, params).scalars().all()
        return tuple(cast(list[uuid.UUID], rows))

    def _load_sparse_passage_candidates(
        self,
        connection: Connection,
        *,
        query: str,
        retrieval_index_run_ids: tuple[uuid.UUID, ...],
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
        limit: int,
    ) -> list[_Candidate]:
        params: dict[str, object] = {
            "query": query,
            "retrieval_index_run_ids": list(retrieval_index_run_ids),
            "candidate_limit": limit,
            "apply_document_filter": filtered_document_ids is not None,
            "document_ids": list(filtered_document_ids or ()),
        }
        query_sql = """
            WITH candidate_assets AS (
                SELECT
                    assets.id,
                    assets.retrieval_index_run_id,
                    assets.passage_id,
                    ts_rank_cd(
                        assets.search_tsvector,
                        websearch_to_tsquery('english', :query)
                    ) AS rank_score
                FROM retrieval_passage_assets assets
                WHERE assets.retrieval_index_run_id = ANY(CAST(:retrieval_index_run_ids AS uuid[]))
                  AND assets.search_tsvector @@ websearch_to_tsquery('english', :query)
                  AND (
                      :apply_document_filter = false
                      OR assets.document_id = ANY(CAST(:document_ids AS uuid[]))
                  )
                ORDER BY rank_score DESC, assets.passage_id
                LIMIT :candidate_limit
            )
            SELECT
                passages.id AS passage_id,
                passages.document_id,
                passages.section_id,
                passages.chunk_ordinal,
                passages.body_text,
                passages.contextualized_text,
                passages.page_start,
                passages.page_end,
                COALESCE(revisions.title, documents.title, 'Untitled document') AS document_title,
                COALESCE(sections.heading_path, '[]'::jsonb) AS section_path,
                runs.id AS retrieval_index_run_id,
                runs.index_version,
                runs.parser_source,
                COALESCE(jobs.warnings, '[]'::jsonb) AS warnings,
                candidate_assets.rank_score
            FROM candidate_assets
            JOIN retrieval_passage_assets assets
                ON assets.id = candidate_assets.id
            JOIN retrieval_index_runs runs
                ON runs.id = assets.retrieval_index_run_id
            JOIN document_passages passages
                ON passages.id = assets.passage_id
               AND (
                   passages.revision_id = runs.revision_id
                   OR (passages.revision_id IS NULL AND runs.revision_id IS NULL)
               )
            JOIN document_sections sections
                ON sections.id = passages.section_id
               AND (
                   sections.revision_id = passages.revision_id
                   OR (sections.revision_id IS NULL AND passages.revision_id IS NULL)
               )
            JOIN documents
                ON documents.id = passages.document_id
            LEFT JOIN document_revisions revisions
                ON revisions.id = documents.active_revision_id
            JOIN ingest_jobs jobs
                ON jobs.id = runs.ingest_job_id
            ORDER BY candidate_assets.rank_score DESC, passages.id
            """
        rows = (
            connection.execute(
                text(query_sql),
                params,
            )
            .mappings()
            .all()
        )
        return [self._row_to_passage_candidate(row) for row in rows]

    def _load_dense_passage_candidates(
        self,
        connection: Connection,
        *,
        query: str,
        retrieval_index_run_ids: tuple[uuid.UUID, ...],
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
        limit: int,
    ) -> list[_Candidate]:
        if self._embedding_client is None:
            return []
        batch = self._embedding_client.embed([query], input_type="query")
        if not batch.embeddings:
            return []
        if batch.dimensions != EMBEDDING_DIMENSIONS:
            raise RetrievalError(
                "query embedding dimension mismatch: "
                f"expected {EMBEDDING_DIMENSIONS}, got {batch.dimensions}"
            )
        params: dict[str, object] = {
            "query_embedding": _vector_literal(batch.embeddings[0]),
            "retrieval_index_run_ids": list(retrieval_index_run_ids),
            "candidate_limit": limit,
            "apply_document_filter": filtered_document_ids is not None,
            "document_ids": list(filtered_document_ids or ()),
        }
        query_sql = """
            WITH candidate_assets AS (
                SELECT
                    assets.id,
                    assets.retrieval_index_run_id,
                    assets.passage_id,
                    1 - (assets.embedding <=> CAST(:query_embedding AS vector)) AS dense_score
                FROM retrieval_passage_assets assets
                WHERE assets.retrieval_index_run_id = ANY(CAST(:retrieval_index_run_ids AS uuid[]))
                  AND assets.embedding IS NOT NULL
                  AND (
                      :apply_document_filter = false
                      OR assets.document_id = ANY(CAST(:document_ids AS uuid[]))
                  )
                ORDER BY assets.embedding <=> CAST(:query_embedding AS vector), assets.passage_id
                LIMIT :candidate_limit
            )
            SELECT
                passages.id AS passage_id,
                passages.document_id,
                passages.section_id,
                passages.chunk_ordinal,
                passages.body_text,
                passages.contextualized_text,
                passages.page_start,
                passages.page_end,
                COALESCE(revisions.title, documents.title, 'Untitled document') AS document_title,
                COALESCE(sections.heading_path, '[]'::jsonb) AS section_path,
                runs.id AS retrieval_index_run_id,
                runs.index_version,
                runs.parser_source,
                COALESCE(jobs.warnings, '[]'::jsonb) AS warnings,
                candidate_assets.dense_score
            FROM candidate_assets
            JOIN retrieval_passage_assets assets
                ON assets.id = candidate_assets.id
            JOIN retrieval_index_runs runs
                ON runs.id = assets.retrieval_index_run_id
            JOIN document_passages passages
                ON passages.id = assets.passage_id
               AND (
                   passages.revision_id = runs.revision_id
                   OR (passages.revision_id IS NULL AND runs.revision_id IS NULL)
               )
            JOIN document_sections sections
                ON sections.id = passages.section_id
               AND (
                   sections.revision_id = passages.revision_id
                   OR (sections.revision_id IS NULL AND passages.revision_id IS NULL)
               )
            JOIN documents
                ON documents.id = passages.document_id
            LEFT JOIN document_revisions revisions
                ON revisions.id = documents.active_revision_id
            JOIN ingest_jobs jobs
                ON jobs.id = runs.ingest_job_id
            ORDER BY assets.embedding <=> CAST(:query_embedding AS vector), passages.id
            """
        rows = (
            connection.execute(
                text(query_sql),
                params,
            )
            .mappings()
            .all()
        )
        return [self._row_to_passage_candidate(row) for row in rows]

    def _load_sparse_table_candidates(
        self,
        connection: Connection,
        *,
        query: str,
        retrieval_index_run_ids: tuple[uuid.UUID, ...],
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
        limit: int,
    ) -> list[_Candidate]:
        params: dict[str, object] = {
            "query": query,
            "retrieval_index_run_ids": list(retrieval_index_run_ids),
            "candidate_limit": limit,
            "apply_document_filter": filtered_document_ids is not None,
            "document_ids": list(filtered_document_ids or ()),
        }
        query_sql = """
            WITH candidate_assets AS (
                SELECT
                    assets.id,
                    assets.retrieval_index_run_id,
                    assets.table_id,
                    assets.search_text,
                    ts_rank_cd(
                        assets.search_tsvector,
                        websearch_to_tsquery('english', :query)
                    ) AS rank_score
                FROM retrieval_table_assets assets
                WHERE assets.retrieval_index_run_id = ANY(CAST(:retrieval_index_run_ids AS uuid[]))
                  AND assets.search_tsvector @@ websearch_to_tsquery('english', :query)
                  AND (
                      :apply_document_filter = false
                      OR assets.document_id = ANY(CAST(:document_ids AS uuid[]))
                  )
                ORDER BY rank_score DESC, assets.table_id
                LIMIT :candidate_limit
            )
            SELECT
                tables.id AS table_id,
                tables.document_id,
                tables.section_id,
                tables.caption,
                tables.table_type,
                COALESCE(tables.headers_json, '[]'::jsonb) AS headers_json,
                COALESCE(tables.rows_json, '[]'::jsonb) AS rows_json,
                tables.page_start,
                tables.page_end,
                COALESCE(revisions.title, documents.title, 'Untitled document') AS document_title,
                COALESCE(sections.heading_path, '[]'::jsonb) AS section_path,
                runs.id AS retrieval_index_run_id,
                runs.index_version,
                runs.parser_source,
                COALESCE(jobs.warnings, '[]'::jsonb) AS warnings,
                candidate_assets.search_text,
                candidate_assets.rank_score
            FROM candidate_assets
            JOIN retrieval_table_assets assets
                ON assets.id = candidate_assets.id
            JOIN retrieval_index_runs runs
                ON runs.id = assets.retrieval_index_run_id
            JOIN document_tables tables
                ON tables.id = assets.table_id
               AND (
                   tables.revision_id = runs.revision_id
                   OR (tables.revision_id IS NULL AND runs.revision_id IS NULL)
               )
            JOIN document_sections sections
                ON sections.id = tables.section_id
               AND (
                   sections.revision_id = tables.revision_id
                   OR (sections.revision_id IS NULL AND tables.revision_id IS NULL)
               )
            JOIN documents
                ON documents.id = tables.document_id
            LEFT JOIN document_revisions revisions
                ON revisions.id = documents.active_revision_id
            JOIN ingest_jobs jobs
                ON jobs.id = runs.ingest_job_id
            ORDER BY candidate_assets.rank_score DESC, tables.id
            """
        rows = (
            connection.execute(
                text(query_sql),
                params,
            )
            .mappings()
            .all()
        )
        return [self._row_to_table_candidate(row) for row in rows]

    def _fuse_candidates(
        self,
        sparse_candidates: list[_Candidate],
        dense_candidates: list[_Candidate],
        *,
        fused_limit: int,
    ) -> list[_Candidate]:
        merged: dict[uuid.UUID, _Candidate] = {}
        for rank, candidate in enumerate(sparse_candidates, start=1):
            stored = merged.setdefault(candidate.entity_id, candidate)
            stored.retrieval_modes.add("sparse")
            stored.fused_score += 1 / (RRF_K + rank)
        for rank, candidate in enumerate(dense_candidates, start=1):
            stored = merged.setdefault(candidate.entity_id, candidate)
            stored.retrieval_modes.add("dense")
            stored.fused_score += 1 / (RRF_K + rank)
        fused = sorted(
            merged.values(),
            key=lambda candidate: (-candidate.fused_score, str(candidate.entity_id)),
        )
        return fused[:fused_limit]

    def _rerank_candidates(
        self,
        *,
        query: str,
        candidates: list[_Candidate],
        limit: int,
    ) -> list[_Candidate]:
        if not candidates:
            return []
        if self._reranker_client is None:
            for candidate in candidates:
                candidate.score = candidate.fused_score
            return sorted(
                candidates,
                key=lambda candidate: (-candidate.score, str(candidate.entity_id)),
            )[:limit]

        reranked_items = self._reranker_client.rerank(
            query=query,
            documents=[candidate.rerank_text for candidate in candidates],
            top_n=min(limit, len(candidates)),
        )
        ordered: list[_Candidate] = []
        seen_indexes: set[int] = set()
        for item in reranked_items:
            if item.index < 0 or item.index >= len(candidates):
                raise RetrievalError("reranker returned an out-of-range candidate index")
            candidate = candidates[item.index]
            candidate.score = item.score
            ordered.append(candidate)
            seen_indexes.add(item.index)
        remainder = [
            candidate for index, candidate in enumerate(candidates) if index not in seen_indexes
        ]
        for candidate in remainder:
            candidate.score = candidate.fused_score
        ranked = [*ordered, *remainder]
        ranked.sort(key=lambda candidate: (-candidate.score, str(candidate.entity_id)))
        return ranked[:limit]

    def _row_to_passage_candidate(self, row: Any) -> _Candidate:
        warnings = tuple(cast(list[str], row["warnings"] or []))
        return _Candidate(
            entity_kind="passage",
            entity_id=row["passage_id"],
            document_id=row["document_id"],
            section_id=row["section_id"],
            document_title=row["document_title"],
            section_path=tuple(cast(list[str], row["section_path"] or [])),
            page_start=row["page_start"],
            page_end=row["page_end"],
            retrieval_index_run_id=row["retrieval_index_run_id"],
            index_version=row["index_version"],
            parser_source=row.get("parser_source"),
            warnings=warnings,
            rerank_text=row["contextualized_text"],
            passage_id=row["passage_id"],
            body_text=row["body_text"],
            chunk_ordinal=row["chunk_ordinal"],
        )

    def _row_to_table_candidate(self, row: Any) -> _Candidate:
        warnings = tuple(cast(list[str], row["warnings"] or []))
        headers = tuple(str(value) for value in cast(list[object], row["headers_json"] or []))
        table_rows = tuple(
            tuple(str(cell) for cell in cast(list[object], values))
            for values in cast(list[list[object]], row["rows_json"] or [])
        )
        return _Candidate(
            entity_kind="table",
            entity_id=row["table_id"],
            document_id=row["document_id"],
            section_id=row["section_id"],
            document_title=row["document_title"],
            section_path=tuple(cast(list[str], row["section_path"] or [])),
            page_start=row["page_start"],
            page_end=row["page_end"],
            retrieval_index_run_id=row["retrieval_index_run_id"],
            index_version=row["index_version"],
            parser_source=row.get("parser_source"),
            warnings=warnings,
            rerank_text=row["search_text"],
            table_id=row["table_id"],
            caption=row["caption"],
            table_type=row["table_type"],
            preview=_build_table_preview(headers=headers, rows=table_rows),
        )

    def _candidate_to_passage_result(self, candidate: _Candidate) -> PassageResult:
        if candidate.passage_id is None or candidate.body_text is None:
            raise RetrievalError("passage candidate is missing passage payload")
        return PassageResult(
            passage_id=candidate.passage_id,
            document_id=candidate.document_id,
            section_id=candidate.section_id,
            document_title=candidate.document_title,
            section_path=candidate.section_path,
            text=candidate.body_text,
            score=candidate.score,
            retrieval_modes=_normalize_modes(candidate.retrieval_modes),
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            index_version=candidate.index_version,
            retrieval_index_run_id=candidate.retrieval_index_run_id,
            parser_source=candidate.parser_source,
            warnings=candidate.warnings,
        )

    def _candidate_to_table_result(self, candidate: _Candidate) -> TableResult:
        if candidate.table_id is None or candidate.preview is None:
            raise RetrievalError("table candidate is missing table payload")
        return TableResult(
            table_id=candidate.table_id,
            document_id=candidate.document_id,
            section_id=candidate.section_id,
            document_title=candidate.document_title,
            section_path=candidate.section_path,
            caption=candidate.caption,
            table_type=candidate.table_type,
            preview=candidate.preview,
            score=candidate.score,
            retrieval_modes=_normalize_modes(candidate.retrieval_modes),
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            index_version=candidate.index_version,
            retrieval_index_run_id=candidate.retrieval_index_run_id,
            parser_source=candidate.parser_source,
            warnings=candidate.warnings,
        )

    def _load_parent_sections(
        self,
        connection: Connection,
        *,
        passages: tuple[PassageResult, ...],
        tables: tuple[TableResult, ...],
    ) -> tuple[ParentSectionResult, ...]:
        section_ids = tuple(
            dict.fromkeys(
                [
                    *(result.section_id for result in passages),
                    *(result.section_id for result in tables),
                ]
            )
        )
        if not section_ids:
            return ()
        rows = (
            connection.execute(
                select(
                    _DOCUMENT_PASSAGES_SQL.c.id.label("passage_id"),
                    _DOCUMENT_PASSAGES_SQL.c.section_id,
                    _DOCUMENT_PASSAGES_SQL.c.chunk_ordinal,
                    _DOCUMENT_PASSAGES_SQL.c.body_text,
                    _DOCUMENT_PASSAGES_SQL.c.page_start,
                    _DOCUMENT_PASSAGES_SQL.c.page_end,
                    _DOCUMENT_SECTIONS_SQL.c.document_id,
                    func.coalesce(
                        _DOCUMENT_REVISIONS_SQL.c.title,
                        _DOCUMENTS_SQL.c.title,
                        "Untitled document",
                    ).label("document_title"),
                    _DOCUMENT_SECTIONS_SQL.c.heading,
                    func.coalesce(_DOCUMENT_SECTIONS_SQL.c.heading_path, sa_cast([], JSONB)).label(
                        "section_path"
                    ),
                    _DOCUMENT_SECTIONS_SQL.c.page_start.label("section_page_start"),
                    _DOCUMENT_SECTIONS_SQL.c.page_end.label("section_page_end"),
                )
                .select_from(_DOCUMENT_PASSAGES_SQL)
                .join(
                    _DOCUMENT_SECTIONS_SQL,
                    and_(
                        _DOCUMENT_SECTIONS_SQL.c.id == _DOCUMENT_PASSAGES_SQL.c.section_id,
                        _revision_match(
                            _DOCUMENT_SECTIONS_SQL.c.revision_id,
                            _DOCUMENT_PASSAGES_SQL.c.revision_id,
                        ),
                    ),
                )
                .join(_DOCUMENTS_SQL, _DOCUMENTS_SQL.c.id == _DOCUMENT_SECTIONS_SQL.c.document_id)
                .outerjoin(
                    _DOCUMENT_REVISIONS_SQL,
                    _revision_match(
                        _DOCUMENT_REVISIONS_SQL.c.id,
                        _DOCUMENTS_SQL.c.active_revision_id,
                    ),
                )
                .where(
                    _DOCUMENT_PASSAGES_SQL.c.section_id.in_(section_ids),
                    _revision_match(
                        _DOCUMENT_PASSAGES_SQL.c.revision_id,
                        _DOCUMENT_SECTIONS_SQL.c.revision_id,
                    ),
                )
                .order_by(
                    _DOCUMENT_PASSAGES_SQL.c.section_id,
                    _DOCUMENT_PASSAGES_SQL.c.chunk_ordinal,
                    _DOCUMENT_PASSAGES_SQL.c.id,
                )
            )
            .mappings()
            .all()
        )
        grouped_rows: dict[uuid.UUID, list[Any]] = defaultdict(list)
        for row in rows:
            grouped_rows[row["section_id"]].append(row)

        selected_passages_by_section: dict[uuid.UUID, set[uuid.UUID]] = defaultdict(set)
        for passage in passages:
            selected_passages_by_section[passage.section_id].add(passage.passage_id)

        parent_sections: list[ParentSectionResult] = []
        for section_id in section_ids:
            section_rows = grouped_rows.get(section_id, [])
            if not section_rows:
                continue
            selected_ids = selected_passages_by_section.get(section_id, set())
            included_indexes: set[int] = set()
            if selected_ids:
                for index, row in enumerate(section_rows):
                    if row["passage_id"] not in selected_ids:
                        continue
                    start = max(0, index - PARENT_SIBLINGS_BEFORE)
                    end = min(len(section_rows), index + PARENT_SIBLINGS_AFTER + 1)
                    included_indexes.update(range(start, end))
            else:
                included_indexes.update(range(min(len(section_rows), PARENT_SIBLINGS_AFTER + 1)))

            supporting_passages: list[ContextPassage] = []
            for index in sorted(included_indexes):
                row = section_rows[index]
                relationship = "selected" if row["passage_id"] in selected_ids else "sibling"
                supporting_passages.append(
                    ContextPassage(
                        passage_id=row["passage_id"],
                        text=row["body_text"],
                        chunk_ordinal=row["chunk_ordinal"],
                        page_start=row["page_start"],
                        page_end=row["page_end"],
                        relationship=cast(Literal["selected", "sibling"], relationship),
                    )
                )
            warnings: list[str] = []
            if len(section_rows) > len(included_indexes):
                warnings.append("parent_context_truncated")
            first_row = section_rows[0]
            parent_sections.append(
                ParentSectionResult(
                    section_id=section_id,
                    document_id=first_row["document_id"],
                    document_title=first_row["document_title"],
                    heading=first_row["heading"],
                    section_path=tuple(cast(list[str], first_row["section_path"] or [])),
                    page_start=first_row["section_page_start"],
                    page_end=first_row["section_page_end"],
                    supporting_passages=tuple(supporting_passages),
                    warnings=_dedupe_warnings(warnings),
                )
            )
        return tuple(parent_sections)

    def _load_document_summaries(
        self,
        connection: Connection,
        *,
        document_ids: tuple[uuid.UUID, ...],
        active_index_version: str | None,
    ) -> tuple[DocumentSummary, ...]:
        if not document_ids:
            return ()
        rows = (
            connection.execute(
                select(
                    _DOCUMENTS_SQL.c.id,
                    func.coalesce(
                        _DOCUMENT_REVISIONS_SQL.c.title,
                        _DOCUMENTS_SQL.c.title,
                        "Untitled document",
                    ).label("title"),
                    func.coalesce(
                        _DOCUMENT_REVISIONS_SQL.c.authors,
                        _DOCUMENTS_SQL.c.authors,
                        sa_cast([], JSONB),
                    ).label("authors"),
                    func.coalesce(
                        _DOCUMENT_REVISIONS_SQL.c.publication_year,
                        _DOCUMENTS_SQL.c.publication_year,
                    ).label("publication_year"),
                    func.coalesce(
                        _DOCUMENT_REVISIONS_SQL.c.quant_tags,
                        _DOCUMENTS_SQL.c.quant_tags,
                        sa_cast({}, JSONB),
                    ).label("quant_tags"),
                    _DOCUMENTS_SQL.c.current_status.label("current_status"),
                )
                .select_from(_DOCUMENTS_SQL)
                .outerjoin(
                    _DOCUMENT_REVISIONS_SQL,
                    _revision_match(
                        _DOCUMENT_REVISIONS_SQL.c.id,
                        _DOCUMENTS_SQL.c.active_revision_id,
                    ),
                )
                .where(_DOCUMENTS_SQL.c.id.in_(document_ids))
                .order_by(_DOCUMENTS_SQL.c.id)
            )
            .mappings()
            .all()
        )
        return tuple(
            DocumentSummary(
                document_id=row["id"],
                title=row["title"],
                authors=tuple(cast(list[str], row["authors"] or [])),
                publication_year=row["publication_year"],
                quant_tags=cast(dict[str, object], row["quant_tags"] or {}),
                current_status=row["current_status"],
                active_index_version=active_index_version,
            )
            for row in rows
        )

    def _ensure_single_index_version(
        self,
        results: list[PassageResult | TableResult] | list[_Candidate],
    ) -> str | None:
        versions = {result.index_version for result in results}
        if not versions:
            return None
        if len(versions) > 1:
            raise MixedIndexVersionError(
                f"retrieval response mixed index versions: {sorted(versions)}"
            )
        return next(iter(versions))

    def _paginate_ranked_results(
        self,
        *,
        kind: str,
        results: tuple[PassageResult, ...] | tuple[TableResult, ...],
        limit: int,
        cursor: str | None,
        active_runs: _ActiveRunSelection,
        fingerprint: str,
    ) -> SearchPage[PassageResult] | SearchPage[TableResult]:
        index_version = self._page_index_version(results=results, active_runs=active_runs)
        ordered = tuple(
            sorted(
                results,
                key=lambda result: (-result.score, self._result_identity(result)),
            )
        )
        start_index = 0
        if cursor is not None:
            try:
                payload = decode_cursor(cursor)
            except CursorError as exc:
                raise RetrievalError(str(exc)) from exc
            if payload.get("kind") != kind or payload.get("fingerprint") != fingerprint:
                raise RetrievalError("cursor does not match request")
            cursor_version = payload.get("index_version")
            if cursor_version != index_version:
                raise RetrievalError("cursor index version is no longer active")
            cursor_score = float(payload["score"])
            cursor_entity_id = str(payload["entity_id"])
            start_index = len(ordered)
            for index, result in enumerate(ordered):
                result_score = result.score
                result_entity_id = self._result_identity(result)
                if result_score < cursor_score or (
                    result_score == cursor_score and result_entity_id > cursor_entity_id
                ):
                    start_index = index
                    break

        page_items = ordered[start_index : start_index + max(0, limit)]
        next_cursor: str | None = None
        if start_index + limit < len(ordered) and page_items:
            last_result = page_items[-1]
            next_cursor = encode_cursor(
                {
                    "kind": kind,
                    "fingerprint": fingerprint,
                    "index_version": index_version,
                    "score": repr(last_result.score),
                    "entity_id": self._result_identity(last_result),
                }
            )
        return cast(
            SearchPage[PassageResult] | SearchPage[TableResult],
            SearchPage(
                items=page_items,
                next_cursor=next_cursor,
                index_version=index_version,
            ),
        )

    def _page_index_version(
        self,
        *,
        results: tuple[PassageResult, ...] | tuple[TableResult, ...],
        active_runs: _ActiveRunSelection,
    ) -> str | None:
        if len(active_runs.index_versions) > 1:
            return self._ensure_single_index_version(list(results))
        if active_runs.index_versions:
            return active_runs.index_versions[0]
        return self._ensure_single_index_version(list(results))

    def _result_identity(self, result: PassageResult | TableResult) -> str:
        if isinstance(result, PassageResult):
            return str(result.passage_id)
        return str(result.table_id)


def _build_table_preview(
    *,
    headers: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
) -> TablePreview:
    return TablePreview(
        headers=headers,
        rows=rows[:TABLE_PREVIEW_ROWS],
        row_count=len(rows),
    )


def _dedupe_modes_from_results(
    *,
    passages: tuple[PassageResult, ...],
    tables: tuple[TableResult, ...],
) -> tuple[RetrievalMode, ...]:
    modes: set[str] = set()
    for result in passages:
        modes.update(result.retrieval_modes)
    for result in tables:
        modes.update(result.retrieval_modes)
    return _normalize_modes(modes)
