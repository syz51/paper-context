from __future__ import annotations

import json
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast

from sqlalchemy.engine import Connection
from sqlalchemy.sql import text

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
    PassageResult,
    RerankerClient,
    RetrievalError,
    RetrievalFilters,
    RetrievalMode,
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


@dataclass
class _PassageIndexRow:
    passage_id: uuid.UUID
    document_id: uuid.UUID
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
        ingest_job_id: uuid.UUID,
        parser_source: str,
    ) -> uuid.UUID:
        run_id = _retrieval_index_run_id(ingest_job_id=ingest_job_id)
        now = datetime.now(UTC)
        self._upsert_build_run(
            connection,
            run_id=run_id,
            document_id=document_id,
            ingest_job_id=ingest_job_id,
            parser_source=parser_source,
            created_at=now,
        )
        embedding_dimensions: int | None = None
        self._clear_existing_assets(connection, run_id=run_id)
        for passages in self._iter_passage_row_batches(
            connection,
            document_id=document_id,
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
            document_id=document_id,
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
        ingest_job_id: uuid.UUID,
        parser_source: str,
        created_at: datetime,
    ) -> None:
        connection.execute(
            text(
                """
                INSERT INTO retrieval_index_runs (
                    id,
                    document_id,
                    ingest_job_id,
                    index_version,
                    embedding_provider,
                    embedding_model,
                    embedding_dimensions,
                    reranker_provider,
                    reranker_model,
                    chunking_version,
                    parser_source,
                    status,
                    is_active,
                    activated_at,
                    created_at
                )
                VALUES (
                    :id,
                    :document_id,
                    :ingest_job_id,
                    :index_version,
                    :embedding_provider,
                    :embedding_model,
                    NULL,
                    :reranker_provider,
                    :reranker_model,
                    :chunking_version,
                    :parser_source,
                    'building',
                    false,
                    NULL,
                    :created_at
                )
                ON CONFLICT (id) DO UPDATE
                SET index_version = EXCLUDED.index_version,
                    embedding_provider = EXCLUDED.embedding_provider,
                    embedding_model = EXCLUDED.embedding_model,
                    reranker_provider = EXCLUDED.reranker_provider,
                    reranker_model = EXCLUDED.reranker_model,
                    chunking_version = EXCLUDED.chunking_version,
                    parser_source = EXCLUDED.parser_source,
                    status = 'building',
                    is_active = false,
                    activated_at = NULL
                """
            ),
            {
                "id": run_id,
                "document_id": document_id,
                "ingest_job_id": ingest_job_id,
                "index_version": self.index_version,
                "embedding_provider": self.embedding_client.provider,
                "embedding_model": self.embedding_client.model,
                "reranker_provider": self.reranker_client.provider,
                "reranker_model": self.reranker_client.model,
                "chunking_version": self.chunking_version,
                "parser_source": parser_source,
                "created_at": created_at,
            },
        )

    def _clear_existing_assets(
        self,
        connection: Connection,
        *,
        run_id: uuid.UUID,
    ) -> None:
        connection.execute(
            text(
                """
                DELETE FROM retrieval_passage_assets
                WHERE retrieval_index_run_id = :retrieval_index_run_id
                """
            ),
            {"retrieval_index_run_id": run_id},
        )
        connection.execute(
            text(
                """
                DELETE FROM retrieval_table_assets
                WHERE retrieval_index_run_id = :retrieval_index_run_id
                """
            ),
            {"retrieval_index_run_id": run_id},
        )

    def _activate_build_run(
        self,
        connection: Connection,
        *,
        run_id: uuid.UUID,
        document_id: uuid.UUID,
        embedding_dimensions: int | None,
        activated_at: datetime,
    ) -> None:
        connection.execute(
            text(
                """
                UPDATE retrieval_index_runs
                SET is_active = false,
                    deactivated_at = :deactivated_at
                WHERE document_id = :document_id
                """
            ),
            {
                "document_id": document_id,
                "deactivated_at": activated_at,
            },
        )
        connection.execute(
            text(
                """
                UPDATE retrieval_index_runs
                SET embedding_dimensions = :embedding_dimensions,
                    status = 'ready',
                    is_active = true,
                    activated_at = :activated_at,
                    deactivated_at = NULL
                WHERE id = :retrieval_index_run_id
                """
            ),
            {
                "retrieval_index_run_id": run_id,
                "embedding_dimensions": embedding_dimensions,
                "activated_at": activated_at,
            },
        )

    def _iter_passage_row_batches(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        batch_size: int,
    ) -> Iterator[list[_PassageIndexRow]]:
        offset = 0
        while True:
            rows = (
                connection.execute(
                    text(
                        """
                    SELECT
                        passages.id AS passage_id,
                        passages.document_id,
                        passages.section_id,
                        passages.chunk_ordinal,
                        passages.body_text,
                        passages.contextualized_text,
                        passages.page_start,
                        passages.page_end,
                        COALESCE(documents.title, 'Untitled document') AS document_title,
                        COALESCE(documents.authors, '[]'::jsonb) AS authors,
                        documents.abstract,
                        documents.publication_year,
                        COALESCE(sections.heading_path, '[]'::jsonb) AS section_path
                    FROM document_passages passages
                    JOIN documents ON documents.id = passages.document_id
                    JOIN document_sections sections ON sections.id = passages.section_id
                    WHERE passages.document_id = :document_id
                    ORDER BY sections.ordinal, passages.chunk_ordinal, passages.id
                    LIMIT :batch_size
                    OFFSET :offset
                    """
                    ),
                    {
                        "document_id": document_id,
                        "batch_size": batch_size,
                        "offset": offset,
                    },
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
        batch_size: int,
    ) -> Iterator[list[_TableIndexRow]]:
        offset = 0
        while True:
            rows = (
                connection.execute(
                    text(
                        """
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
                        COALESCE(documents.title, 'Untitled document') AS document_title,
                        documents.publication_year,
                        COALESCE(sections.heading_path, '[]'::jsonb) AS section_path
                    FROM document_tables tables
                    JOIN documents ON documents.id = tables.document_id
                    JOIN document_sections sections ON sections.id = tables.section_id
                    WHERE tables.document_id = :document_id
                    ORDER BY sections.ordinal, tables.id
                    LIMIT :batch_size
                    OFFSET :offset
                    """
                    ),
                    {
                        "document_id": document_id,
                        "batch_size": batch_size,
                        "offset": offset,
                    },
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
        connection.execute(
            text(
                """
                INSERT INTO retrieval_passage_assets (
                    id,
                    retrieval_index_run_id,
                    passage_id,
                    document_id,
                    section_id,
                    publication_year,
                    search_text,
                    search_tsvector,
                    embedding,
                    created_at
                )
                VALUES (
                    :id,
                    :retrieval_index_run_id,
                    :passage_id,
                    :document_id,
                    :section_id,
                    :publication_year,
                    :search_text,
                    to_tsvector('english', :search_text),
                    CAST(:embedding AS vector),
                    :created_at
                )
                """
            ),
            [
                {
                    "id": uuid.uuid4(),
                    "retrieval_index_run_id": run_id,
                    "passage_id": row.passage_id,
                    "document_id": row.document_id,
                    "section_id": row.section_id,
                    "publication_year": row.publication_year,
                    "search_text": self._build_passage_search_text(row),
                    "embedding": _vector_literal(embedding),
                    "created_at": created_at,
                }
                for row, embedding in zip(rows, embeddings, strict=True)
            ],
        )

    def _insert_table_asset_batch(
        self,
        connection: Connection,
        *,
        run_id: uuid.UUID,
        rows: list[_TableIndexRow],
        created_at: datetime,
    ) -> None:
        connection.execute(
            text(
                """
                INSERT INTO retrieval_table_assets (
                    id,
                    retrieval_index_run_id,
                    table_id,
                    document_id,
                    section_id,
                    publication_year,
                    search_text,
                    search_tsvector,
                    created_at
                )
                VALUES (
                    :id,
                    :retrieval_index_run_id,
                    :table_id,
                    :document_id,
                    :section_id,
                    :publication_year,
                    :search_text,
                    to_tsvector('english', :search_text),
                    :created_at
                )
                """
            ),
            [
                {
                    "id": uuid.uuid4(),
                    "retrieval_index_run_id": run_id,
                    "table_id": row.table_id,
                    "document_id": row.document_id,
                    "section_id": row.section_id,
                    "publication_year": row.publication_year,
                    "search_text": self._build_table_search_text(row),
                    "created_at": created_at,
                }
                for row in rows
            ],
        )

    def _row_to_passage_index_row(self, row: Any) -> _PassageIndexRow:
        return _PassageIndexRow(
            passage_id=row["passage_id"],
            document_id=row["document_id"],
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

    def build_context_pack(
        self,
        *,
        query: str,
        filters: RetrievalFilters | None = None,
    ) -> ContextPackResult:
        filters = filters or RetrievalFilters()
        with self._connection() as connection:
            filtered_document_ids = self._resolve_filtered_document_ids(
                connection,
                filters=filters,
            )
            active_runs = self._resolve_active_run_selection(
                connection,
                filtered_document_ids=filtered_document_ids,
            )
            passages = tuple(
                self._search_passages_with_connection(
                    connection,
                    query=query,
                    filters=filters,
                    limit=PASSAGE_RESULT_LIMIT,
                    filtered_document_ids=filtered_document_ids,
                    active_runs=active_runs,
                )
            )
            tables = tuple(
                self._search_tables_with_connection(
                    connection,
                    query=query,
                    filters=filters,
                    limit=TABLE_RESULT_LIMIT,
                    filtered_document_ids=filtered_document_ids,
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
        query_sql = """
            SELECT runs.id, runs.index_version
            FROM retrieval_index_runs runs
            WHERE runs.status = 'ready'
              AND runs.is_active = true
              {document_filter_sql}
            ORDER BY
                COALESCE(runs.activated_at, runs.created_at) DESC,
                runs.id
            """
        document_filter_sql = ""
        params: dict[str, object] = {}
        if filtered_document_ids is not None:
            document_filter_sql = "AND runs.document_id = ANY(CAST(:document_ids AS uuid[]))"
            params["document_ids"] = list(filtered_document_ids)
        rows = (
            connection.execute(
                text(query_sql.format(document_filter_sql=document_filter_sql)),
                params,
            )
            .mappings()
            .all()
        )
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

        clauses = ["1 = 1"]
        params: dict[str, object] = {}
        if filters.document_ids:
            clauses.append("documents.id = ANY(CAST(:document_ids AS uuid[]))")
            params["document_ids"] = list(filters.document_ids)
        if filters.publication_years:
            clauses.append(
                "documents.publication_year = ANY(CAST(:publication_years AS integer[]))"
            )
            params["publication_years"] = list(filters.publication_years)

        rows = (
            connection.execute(
                text(
                    f"""
                    SELECT documents.id
                    FROM documents
                    WHERE {" AND ".join(clauses)}
                    ORDER BY documents.id
                    """  # nosec B608
                ),
                params,
            )
            .scalars()
            .all()
        )
        return tuple(cast(list[uuid.UUID], rows))

    def _resolve_active_run_ids(
        self,
        connection: Connection,
        *,
        index_version: str,
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
    ) -> tuple[uuid.UUID, ...]:
        document_filter_sql = ""
        params: dict[str, object] = {
            "index_version": index_version,
        }
        if filtered_document_ids is not None:
            document_filter_sql = "AND document_id = ANY(CAST(:document_ids AS uuid[]))"
            params["document_ids"] = list(filtered_document_ids)
        rows = (
            connection.execute(
                text(
                    f"""
                    SELECT id
                    FROM retrieval_index_runs
                    WHERE status = 'ready'
                      AND is_active = true
                      AND index_version = :index_version
                      {document_filter_sql}
                    ORDER BY COALESCE(activated_at, created_at) DESC, id
                    """  # nosec B608
                ),
                params,
            )
            .scalars()
            .all()
        )
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
        document_filter_sql = ""
        params: dict[str, object] = {
            "query": query,
            "retrieval_index_run_ids": list(retrieval_index_run_ids),
            "candidate_limit": limit,
        }
        if filtered_document_ids is not None:
            document_filter_sql = "AND assets.document_id = ANY(CAST(:document_ids AS uuid[]))"
            params["document_ids"] = list(filtered_document_ids)
        query_sql = f"""
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
                  {document_filter_sql}
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
                COALESCE(documents.title, 'Untitled document') AS document_title,
                COALESCE(sections.heading_path, '[]'::jsonb) AS section_path,
                runs.id AS retrieval_index_run_id,
                runs.index_version,
                COALESCE(jobs.warnings, '[]'::jsonb) AS warnings,
                candidate_assets.rank_score
            FROM candidate_assets
            JOIN retrieval_passage_assets assets
                ON assets.id = candidate_assets.id
            JOIN retrieval_index_runs runs
                ON runs.id = assets.retrieval_index_run_id
            JOIN document_passages passages
                ON passages.id = assets.passage_id
            JOIN document_sections sections
                ON sections.id = passages.section_id
            JOIN documents
                ON documents.id = passages.document_id
            JOIN ingest_jobs jobs
                ON jobs.id = runs.ingest_job_id
            ORDER BY candidate_assets.rank_score DESC, passages.id
            """  # nosec B608
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
        document_filter_sql = ""
        params: dict[str, object] = {
            "query_embedding": _vector_literal(batch.embeddings[0]),
            "retrieval_index_run_ids": list(retrieval_index_run_ids),
            "candidate_limit": limit,
        }
        if filtered_document_ids is not None:
            document_filter_sql = "AND assets.document_id = ANY(CAST(:document_ids AS uuid[]))"
            params["document_ids"] = list(filtered_document_ids)
        query_sql = f"""
            WITH candidate_assets AS (
                SELECT
                    assets.id,
                    assets.retrieval_index_run_id,
                    assets.passage_id,
                    1 - (assets.embedding <=> CAST(:query_embedding AS vector)) AS dense_score
                FROM retrieval_passage_assets assets
                WHERE assets.retrieval_index_run_id = ANY(CAST(:retrieval_index_run_ids AS uuid[]))
                  AND assets.embedding IS NOT NULL
                  {document_filter_sql}
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
                COALESCE(documents.title, 'Untitled document') AS document_title,
                COALESCE(sections.heading_path, '[]'::jsonb) AS section_path,
                runs.id AS retrieval_index_run_id,
                runs.index_version,
                COALESCE(jobs.warnings, '[]'::jsonb) AS warnings,
                candidate_assets.dense_score
            FROM candidate_assets
            JOIN retrieval_passage_assets assets
                ON assets.id = candidate_assets.id
            JOIN retrieval_index_runs runs
                ON runs.id = assets.retrieval_index_run_id
            JOIN document_passages passages
                ON passages.id = assets.passage_id
            JOIN document_sections sections
                ON sections.id = passages.section_id
            JOIN documents
                ON documents.id = passages.document_id
            JOIN ingest_jobs jobs
                ON jobs.id = runs.ingest_job_id
            ORDER BY assets.embedding <=> CAST(:query_embedding AS vector), passages.id
            """  # nosec B608
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
        document_filter_sql = ""
        params: dict[str, object] = {
            "query": query,
            "retrieval_index_run_ids": list(retrieval_index_run_ids),
            "candidate_limit": limit,
        }
        if filtered_document_ids is not None:
            document_filter_sql = "AND assets.document_id = ANY(CAST(:document_ids AS uuid[]))"
            params["document_ids"] = list(filtered_document_ids)
        query_sql = f"""
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
                  {document_filter_sql}
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
                COALESCE(documents.title, 'Untitled document') AS document_title,
                COALESCE(sections.heading_path, '[]'::jsonb) AS section_path,
                runs.id AS retrieval_index_run_id,
                runs.index_version,
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
            JOIN document_sections sections
                ON sections.id = tables.section_id
            JOIN documents
                ON documents.id = tables.document_id
            JOIN ingest_jobs jobs
                ON jobs.id = runs.ingest_job_id
            ORDER BY candidate_assets.rank_score DESC, tables.id
            """  # nosec B608
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
        remainder.sort(key=lambda candidate: (-candidate.score, str(candidate.entity_id)))
        return [*ordered, *remainder][:limit]

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
                text(
                    """
                    SELECT
                        passages.id AS passage_id,
                        passages.section_id,
                        passages.chunk_ordinal,
                        passages.body_text,
                        passages.page_start,
                        passages.page_end,
                        sections.document_id,
                        COALESCE(documents.title, 'Untitled document') AS document_title,
                        sections.heading,
                        COALESCE(sections.heading_path, '[]'::jsonb) AS section_path,
                        sections.page_start AS section_page_start,
                        sections.page_end AS section_page_end
                    FROM document_passages passages
                    JOIN document_sections sections
                        ON sections.id = passages.section_id
                    JOIN documents
                        ON documents.id = sections.document_id
                    WHERE passages.section_id = ANY(CAST(:section_ids AS uuid[]))
                    ORDER BY passages.section_id, passages.chunk_ordinal, passages.id
                    """
                ),
                {"section_ids": list(section_ids)},
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
                text(
                    """
                    SELECT
                        id,
                        COALESCE(title, 'Untitled document') AS title,
                        COALESCE(authors, '[]'::jsonb) AS authors,
                        publication_year,
                        COALESCE(quant_tags, '{}'::jsonb) AS quant_tags,
                        current_status
                    FROM documents
                    WHERE id = ANY(CAST(:document_ids AS uuid[]))
                    ORDER BY id
                    """
                ),
                {"document_ids": list(document_ids)},
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
