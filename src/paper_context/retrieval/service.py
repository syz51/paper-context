from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import zip_longest
from typing import Any, Literal, cast

from sqlalchemy import (
    and_,
    bindparam,
    column,
    delete,
    func,
    insert,
    or_,
    select,
    table,
    update,
)
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
from paper_context.observability import observe_operation
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
logger = logging.getLogger(__name__)

PASSAGE_SPARSE_CANDIDATES = 30
PASSAGE_DENSE_CANDIDATES = 30
PASSAGE_FUSED_CANDIDATES = 40
PASSAGE_RESULT_LIMIT = 8
TABLE_SPARSE_CANDIDATES = 20
TABLE_DENSE_CANDIDATES = 20
TABLE_FUSED_CANDIDATES = 24
TABLE_RESULT_LIMIT = 5
RRF_K = 60
PARENT_SIBLINGS_BEFORE = 1
PARENT_SIBLINGS_AFTER = 1
TABLE_PREVIEW_ROWS = 3
TABLE_SEMANTIC_SAMPLE_ROWS = 3
EMBEDDING_DIMENSIONS = 1024
INDEX_BUILD_BATCH_SIZE = 128
DENSE_EF_SEARCH_MIN = 40
DENSE_EF_SEARCH_MAX = 400
DENSE_EF_SEARCH_MULTIPLIER = 6
DENSE_FILTER_EF_SEARCH_MULTIPLIER = 10
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


def _accumulate_stage_timing(
    stage_timings: dict[str, float | int],
    stage_name: str,
    timing_payload: Mapping[str, float],
) -> None:
    duration_seconds = timing_payload.get("duration_seconds")
    if duration_seconds is None:
        return
    current = float(stage_timings.get(stage_name, 0.0))
    stage_timings[stage_name] = round(current + duration_seconds, 6)


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
        stage_timings: dict[str, float | int] | None = None,
    ) -> uuid.UUID:
        run_id = _retrieval_index_run_id(ingest_job_id=ingest_job_id)
        now = datetime.now(UTC)
        stage_timings = stage_timings if stage_timings is not None else {}
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
            with observe_operation(
                "ingest.embedding",
                logger=logger,
                fields={
                    "document_id": str(document_id),
                    "revision_id": str(revision_id),
                    "ingest_job_id": str(ingest_job_id),
                    "batch_size": len(passages),
                },
            ) as embedding_timing:
                passage_embeddings = self.embedding_client.embed(
                    [row.contextualized_text for row in passages],
                    input_type="document",
                )
            _accumulate_stage_timing(stage_timings, "embedding_seconds", embedding_timing)
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
            table_semantic_texts = [self._build_table_semantic_text(row) for row in tables]
            with observe_operation(
                "ingest.embedding",
                logger=logger,
                fields={
                    "document_id": str(document_id),
                    "revision_id": str(revision_id),
                    "ingest_job_id": str(ingest_job_id),
                    "batch_size": len(tables),
                    "entity_kind": "table",
                },
            ) as embedding_timing:
                table_embeddings = self.embedding_client.embed(
                    table_semantic_texts,
                    input_type="document",
                )
            _accumulate_stage_timing(stage_timings, "embedding_seconds", embedding_timing)
            batch_dimensions = table_embeddings.dimensions
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
            self._insert_table_asset_batch(
                connection,
                run_id=run_id,
                rows=tables,
                embeddings=table_embeddings.embeddings,
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
        last_key: tuple[int, int, uuid.UUID] | None = None
        while True:
            statement = (
                select(
                    DocumentPassage.id.label("passage_id"),
                    DocumentPassage.document_id,
                    DocumentPassage.revision_id,
                    DocumentPassage.section_id,
                    DocumentSection.ordinal.label("section_ordinal"),
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
                    func.coalesce(DocumentRevision.abstract, Document.abstract).label("abstract"),
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
            )
            if last_key is not None:
                last_section_ordinal, last_chunk_ordinal, last_passage_id = last_key
                statement = statement.where(
                    or_(
                        DocumentSection.ordinal > last_section_ordinal,
                        and_(
                            DocumentSection.ordinal == last_section_ordinal,
                            or_(
                                DocumentPassage.chunk_ordinal > last_chunk_ordinal,
                                and_(
                                    DocumentPassage.chunk_ordinal == last_chunk_ordinal,
                                    DocumentPassage.id > last_passage_id,
                                ),
                            ),
                        ),
                    )
                )
            rows = (
                connection.execute(
                    statement.order_by(
                        DocumentSection.ordinal,
                        DocumentPassage.chunk_ordinal,
                        DocumentPassage.id,
                    ).limit(batch_size)
                )
                .mappings()
                .all()
            )
            if not rows:
                return
            yield [self._row_to_passage_index_row(row) for row in rows]
            last_row = rows[-1]
            last_key = (
                cast(int, last_row["section_ordinal"]),
                cast(int, last_row["chunk_ordinal"]),
                cast(uuid.UUID, last_row["passage_id"]),
            )

    def _iter_table_row_batches(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        batch_size: int,
    ) -> Iterator[list[_TableIndexRow]]:
        last_key: tuple[int, uuid.UUID] | None = None
        while True:
            statement = (
                select(
                    DocumentTable.id.label("table_id"),
                    DocumentTable.document_id,
                    DocumentTable.revision_id,
                    DocumentTable.section_id,
                    DocumentSection.ordinal.label("section_ordinal"),
                    DocumentTable.caption,
                    DocumentTable.table_type,
                    func.coalesce(DocumentTable.headers_json, sa_cast([], JSONB)).label(
                        "headers_json"
                    ),
                    func.coalesce(DocumentTable.rows_json, sa_cast([], JSONB)).label("rows_json"),
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
            )
            if last_key is not None:
                last_section_ordinal, last_table_id = last_key
                statement = statement.where(
                    or_(
                        DocumentSection.ordinal > last_section_ordinal,
                        and_(
                            DocumentSection.ordinal == last_section_ordinal,
                            DocumentTable.id > last_table_id,
                        ),
                    )
                )
            rows = (
                connection.execute(
                    statement.order_by(DocumentSection.ordinal, DocumentTable.id).limit(batch_size)
                )
                .mappings()
                .all()
            )
            if not rows:
                return
            yield [self._row_to_table_index_row(row) for row in rows]
            last_row = rows[-1]
            last_key = (
                cast(int, last_row["section_ordinal"]),
                cast(uuid.UUID, last_row["table_id"]),
            )

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
        embeddings: tuple[tuple[float, ...], ...],
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
            semantic_text=bindparam("b_semantic_text"),
            search_tsvector=func.to_tsvector("english", bindparam("b_search_text")),
            embedding=bindparam("b_embedding"),
            created_at=bindparam("b_created_at"),
        )
        payloads: list[dict[str, object]] = []
        for row, embedding in zip(rows, embeddings, strict=True):
            asset_id = uuid.uuid4()
            search_text = self._build_table_search_text(row)
            semantic_text = self._build_table_semantic_text(row)
            embedding_literal = _vector_literal(embedding)
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
                    "semantic_text": semantic_text,
                    "b_semantic_text": semantic_text,
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

    def _build_table_semantic_text(self, row: _TableIndexRow) -> str:
        section_path = " > ".join(row.section_path) if row.section_path else "Body"
        lines = [
            f"Document title: {row.document_title}",
            f"Section path: {section_path}",
        ]
        if row.publication_year is not None:
            lines.append(f"Publication year: {row.publication_year}")
        if row.table_type:
            lines.append(f"Table type: {row.table_type}")
        if row.caption:
            lines.append(f"Table caption: {row.caption}")
        if row.headers:
            lines.append(f"Columns: {', '.join(row.headers)}")
        examples = [
            example
            for example in (
                self._build_table_semantic_example(row.headers, table_row)
                for table_row in row.rows[:TABLE_SEMANTIC_SAMPLE_ROWS]
            )
            if example
        ]
        if examples:
            lines.append("Example values:")
            lines.extend(examples)
        return "\n".join(lines)

    def _build_table_semantic_example(
        self,
        headers: tuple[str, ...],
        table_row: tuple[str, ...],
    ) -> str:
        parts: list[str] = []
        for index, (header, value) in enumerate(
            zip_longest(headers, table_row, fillvalue=""), start=1
        ):
            if not value:
                continue
            label = header or f"Column {index}"
            parts.append(f"{label}={value}")
        return "; ".join(parts)


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
            return self._search_passages_page_with_connection(
                connection,
                query=query,
                filters=filters,
                cursor=cursor,
                limit=limit,
                filtered_document_ids=filtered_document_ids,
                active_runs=active_runs,
                fingerprint=fingerprint,
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
            return self._search_tables_page_with_connection(
                connection,
                query=query,
                filters=filters,
                cursor=cursor,
                limit=limit,
                filtered_document_ids=filtered_document_ids,
                active_runs=active_runs,
                fingerprint=fingerprint,
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
        with observe_operation(
            "retrieval.context_pack_assembly",
            logger=logger,
            fields={
                "query": query.strip(),
                "cursor_present": cursor is not None,
                "limit": limit,
            },
        ):
            with self._connection() as connection:
                filtered_document_ids = self._resolve_filtered_document_ids(
                    connection,
                    filters=filters,
                )
                active_runs = self._resolve_active_run_selection(
                    connection,
                    filtered_document_ids=filtered_document_ids,
                )
                passage_page = self._search_passages_page_with_connection(
                    connection,
                    query=query,
                    filters=filters,
                    cursor=cursor,
                    limit=limit,
                    filtered_document_ids=filtered_document_ids,
                    active_runs=active_runs,
                    fingerprint=fingerprint,
                )
                passages = passage_page.items
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
                    .join(RetrievalIndexRun, and_(*run_join_conditions))
                    .join(IngestJob, IngestJob.id == RetrievalIndexRun.ingest_job_id)
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
                    .join(RetrievalIndexRun, and_(*run_join_conditions))
                    .join(IngestJob, IngestJob.id == RetrievalIndexRun.ingest_job_id)
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

    def _initial_sparse_candidate_limit(
        self,
        *,
        base_limit: int,
        result_limit: int | None,
    ) -> int:
        if result_limit is None:
            return base_limit
        return max(base_limit, result_limit * 4)

    def _initial_dense_candidate_limit(
        self,
        *,
        base_limit: int,
        result_limit: int | None,
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
    ) -> int:
        limit = self._initial_sparse_candidate_limit(
            base_limit=base_limit,
            result_limit=result_limit,
        )
        if result_limit is not None and filtered_document_ids is not None:
            limit = max(limit, result_limit * 8)
        return limit

    def _search_passages_with_connection(
        self,
        connection: Connection,
        *,
        query: str,
        filters: RetrievalFilters,
        limit: int | None,
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
        sparse_limit = self._initial_sparse_candidate_limit(
            base_limit=PASSAGE_SPARSE_CANDIDATES,
            result_limit=limit,
        )
        dense_limit = self._initial_dense_candidate_limit(
            base_limit=PASSAGE_DENSE_CANDIDATES,
            result_limit=limit,
            filtered_document_ids=filtered_document_ids,
        )
        results, _, _ = self._load_ranked_passage_results(
            connection,
            query=query,
            limit=limit,
            filtered_document_ids=filtered_document_ids,
            active_runs=active_runs,
            sparse_candidate_limit=sparse_limit,
            dense_candidate_limit=dense_limit,
        )
        return results

    def _search_tables_with_connection(
        self,
        connection: Connection,
        *,
        query: str,
        filters: RetrievalFilters,
        limit: int | None,
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
        sparse_limit = self._initial_sparse_candidate_limit(
            base_limit=TABLE_SPARSE_CANDIDATES,
            result_limit=limit,
        )
        dense_limit = self._initial_dense_candidate_limit(
            base_limit=TABLE_DENSE_CANDIDATES,
            result_limit=limit,
            filtered_document_ids=filtered_document_ids,
        )
        results, _, _ = self._load_ranked_table_results(
            connection,
            query=query,
            limit=limit,
            filtered_document_ids=filtered_document_ids,
            active_runs=active_runs,
            sparse_candidate_limit=sparse_limit,
            dense_candidate_limit=dense_limit,
        )
        return results

    def _search_passages_page_with_connection(
        self,
        connection: Connection,
        *,
        query: str,
        filters: RetrievalFilters,
        cursor: str | None,
        limit: int,
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
        active_runs: _ActiveRunSelection,
        fingerprint: str,
    ) -> SearchPage[PassageResult]:
        sparse_limit = self._initial_sparse_candidate_limit(
            base_limit=PASSAGE_SPARSE_CANDIDATES,
            result_limit=limit,
        )
        dense_limit = self._initial_dense_candidate_limit(
            base_limit=PASSAGE_DENSE_CANDIDATES,
            result_limit=limit,
            filtered_document_ids=filtered_document_ids,
        )
        previous_signature: tuple[tuple[str, ...], str | None] | None = None
        while True:
            results, sparse_count, dense_count = self._load_ranked_passage_results(
                connection,
                query=query,
                limit=None,
                filtered_document_ids=filtered_document_ids,
                active_runs=active_runs,
                sparse_candidate_limit=sparse_limit,
                dense_candidate_limit=dense_limit,
            )
            page = cast(
                SearchPage[PassageResult],
                self._paginate_ranked_results(
                    kind="passages",
                    results=tuple(results),
                    limit=limit,
                    cursor=cursor,
                    active_runs=active_runs,
                    fingerprint=fingerprint,
                ),
            )
            signature = (
                tuple(self._result_identity(item) for item in page.items),
                page.next_cursor,
            )
            sparse_exhausted = sparse_count < sparse_limit
            dense_exhausted = self._embedding_client is None or dense_count < dense_limit
            if sparse_exhausted and dense_exhausted:
                return page
            if (
                page.next_cursor is not None
                and len(page.items) == limit
                and signature == previous_signature
            ):
                return page
            previous_signature = signature
            if not sparse_exhausted:
                sparse_limit *= 2
            if self._embedding_client is not None and not dense_exhausted:
                dense_limit *= 2

    def _search_tables_page_with_connection(
        self,
        connection: Connection,
        *,
        query: str,
        filters: RetrievalFilters,
        cursor: str | None,
        limit: int,
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
        active_runs: _ActiveRunSelection,
        fingerprint: str,
    ) -> SearchPage[TableResult]:
        sparse_limit = self._initial_sparse_candidate_limit(
            base_limit=TABLE_SPARSE_CANDIDATES,
            result_limit=limit,
        )
        dense_limit = self._initial_dense_candidate_limit(
            base_limit=TABLE_DENSE_CANDIDATES,
            result_limit=limit,
            filtered_document_ids=filtered_document_ids,
        )
        previous_signature: tuple[tuple[str, ...], str | None] | None = None
        while True:
            results, sparse_count, dense_count = self._load_ranked_table_results(
                connection,
                query=query,
                limit=None,
                filtered_document_ids=filtered_document_ids,
                active_runs=active_runs,
                sparse_candidate_limit=sparse_limit,
                dense_candidate_limit=dense_limit,
            )
            page = cast(
                SearchPage[TableResult],
                self._paginate_ranked_results(
                    kind="tables",
                    results=tuple(results),
                    limit=limit,
                    cursor=cursor,
                    active_runs=active_runs,
                    fingerprint=fingerprint,
                ),
            )
            signature = (
                tuple(self._result_identity(item) for item in page.items),
                page.next_cursor,
            )
            sparse_exhausted = sparse_count < sparse_limit
            dense_exhausted = self._embedding_client is None or dense_count < dense_limit
            if sparse_exhausted and dense_exhausted:
                return page
            if (
                page.next_cursor is not None
                and len(page.items) == limit
                and signature == previous_signature
            ):
                return page
            previous_signature = signature
            if not sparse_exhausted:
                sparse_limit *= 2
            if self._embedding_client is not None and not dense_exhausted:
                dense_limit *= 2

    def _load_ranked_passage_results(
        self,
        connection: Connection,
        *,
        query: str,
        limit: int | None,
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
        active_runs: _ActiveRunSelection,
        sparse_candidate_limit: int | None,
        dense_candidate_limit: int | None,
    ) -> tuple[list[PassageResult], int, int]:
        if not query.strip() or filtered_document_ids == () or not active_runs.run_ids:
            return [], 0, 0
        sparse_candidates = self._load_sparse_passage_candidates(
            connection,
            query=query,
            retrieval_index_run_ids=active_runs.run_ids,
            filtered_document_ids=filtered_document_ids,
            limit=sparse_candidate_limit,
        )
        dense_candidates = self._load_dense_passage_candidates(
            connection,
            query=query,
            retrieval_index_run_ids=active_runs.run_ids,
            filtered_document_ids=filtered_document_ids,
            limit=dense_candidate_limit,
        )
        fused = self._fuse_candidates(
            sparse_candidates,
            dense_candidates,
            fused_limit=None if limit is None else PASSAGE_FUSED_CANDIDATES,
        )
        reranked = self._rerank_candidates(query=query, candidates=fused, limit=limit)
        return (
            [self._candidate_to_passage_result(candidate) for candidate in reranked],
            len(sparse_candidates),
            len(dense_candidates),
        )

    def _load_ranked_table_results(
        self,
        connection: Connection,
        *,
        query: str,
        limit: int | None,
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
        active_runs: _ActiveRunSelection,
        sparse_candidate_limit: int | None,
        dense_candidate_limit: int | None,
    ) -> tuple[list[TableResult], int, int]:
        if not query.strip() or filtered_document_ids == () or not active_runs.run_ids:
            return [], 0, 0
        sparse_candidates = self._load_sparse_table_candidates(
            connection,
            query=query,
            retrieval_index_run_ids=active_runs.run_ids,
            filtered_document_ids=filtered_document_ids,
            limit=sparse_candidate_limit,
        )
        dense_candidates = self._load_dense_table_candidates(
            connection,
            query=query,
            retrieval_index_run_ids=active_runs.run_ids,
            filtered_document_ids=filtered_document_ids,
            limit=dense_candidate_limit,
        )
        fused = self._fuse_candidates(
            sparse_candidates,
            dense_candidates,
            fused_limit=None if limit is None else TABLE_FUSED_CANDIDATES,
        )
        reranked = self._rerank_candidates(query=query, candidates=fused, limit=limit)
        return (
            [self._candidate_to_table_result(candidate) for candidate in reranked],
            len(sparse_candidates),
            len(dense_candidates),
        )

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
        if not rows:
            return _ActiveRunSelection(run_ids=(), index_versions=())
        if self._active_index_version is None:
            selected_index_version = cast(str, rows[0]["index_version"])
            rows = [row for row in rows if row["index_version"] == selected_index_version]
            run_ids = tuple(cast(uuid.UUID, row["id"]) for row in rows)
            return _ActiveRunSelection(
                run_ids=run_ids,
                index_versions=(selected_index_version,),
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

    def _configure_dense_vector_search(
        self,
        connection: Connection,
        *,
        candidate_limit: int | None,
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
    ) -> None:
        ef_search = self._dense_ef_search(
            candidate_limit=candidate_limit,
            filtered_document_ids=filtered_document_ids,
        )
        connection.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))
        connection.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))

    def _dense_ef_search(
        self,
        *,
        candidate_limit: int | None,
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
    ) -> int:
        effective_limit = max(candidate_limit or 0, 1)
        multiplier = (
            DENSE_FILTER_EF_SEARCH_MULTIPLIER
            if filtered_document_ids is not None
            else DENSE_EF_SEARCH_MULTIPLIER
        )
        return max(
            DENSE_EF_SEARCH_MIN,
            min(DENSE_EF_SEARCH_MAX, effective_limit * multiplier),
        )

    def _should_retry_exact_dense_query(
        self,
        *,
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
        returned_count: int,
        requested_limit: int | None,
    ) -> bool:
        if filtered_document_ids is None or requested_limit is None:
            return False
        return returned_count < requested_limit

    def _load_sparse_passage_candidates(
        self,
        connection: Connection,
        *,
        query: str,
        retrieval_index_run_ids: tuple[uuid.UUID, ...],
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
        limit: int | None,
    ) -> list[_Candidate]:
        params: dict[str, object] = {
            "query": query,
            "retrieval_index_run_ids": list(retrieval_index_run_ids),
            "apply_document_filter": filtered_document_ids is not None,
            "document_ids": list(filtered_document_ids or ()),
        }
        limit_sql = ""
        if limit is not None:
            params["candidate_limit"] = limit
            limit_sql = "\n                LIMIT :candidate_limit"
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
            """
        query_sql += limit_sql
        query_sql += """
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
        limit: int | None,
    ) -> list[_Candidate]:
        if self._embedding_client is None:
            return []
        with observe_operation(
            "retrieval.embedding",
            logger=logger,
            fields={"query": query.strip(), "candidate_limit": limit},
        ):
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
            "apply_document_filter": filtered_document_ids is not None,
            "document_ids": list(filtered_document_ids or ()),
            "candidate_limit": limit,
        }
        limit_sql = ""
        if limit is not None:
            limit_sql = "\n                LIMIT :candidate_limit"
        self._configure_dense_vector_search(
            connection,
            candidate_limit=limit,
            filtered_document_ids=filtered_document_ids,
        )
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
            """
        query_sql += limit_sql
        query_sql += """
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
        if self._should_retry_exact_dense_query(
            filtered_document_ids=filtered_document_ids,
            returned_count=len(rows),
            requested_limit=limit,
        ):
            exact_rows = (
                connection.execute(
                    text(
                        """
                        WITH filtered_assets AS MATERIALIZED (
                            SELECT
                                assets.id,
                                assets.retrieval_index_run_id,
                                assets.passage_id,
                                1 - (
                                    assets.embedding <=> CAST(:query_embedding AS vector)
                                ) AS dense_score
                            FROM retrieval_passage_assets assets
                            WHERE assets.retrieval_index_run_id = ANY(
                                CAST(:retrieval_index_run_ids AS uuid[])
                            )
                              AND assets.embedding IS NOT NULL
                              AND (
                                  :apply_document_filter = false
                                  OR assets.document_id = ANY(CAST(:document_ids AS uuid[]))
                              )
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
                            COALESCE(
                                revisions.title,
                                documents.title,
                                'Untitled document'
                            ) AS document_title,
                            COALESCE(sections.heading_path, '[]'::jsonb) AS section_path,
                            runs.id AS retrieval_index_run_id,
                            runs.index_version,
                            runs.parser_source,
                            COALESCE(jobs.warnings, '[]'::jsonb) AS warnings,
                            filtered_assets.dense_score
                        FROM filtered_assets
                        JOIN retrieval_index_runs runs
                            ON runs.id = filtered_assets.retrieval_index_run_id
                        JOIN document_passages passages
                            ON passages.id = filtered_assets.passage_id
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
                        ORDER BY filtered_assets.dense_score DESC, passages.id
                        LIMIT COALESCE(CAST(:candidate_limit AS integer), 2147483647)
                        """
                    ),
                    params,
                )
                .mappings()
                .all()
            )
            rows = exact_rows
        return [self._row_to_passage_candidate(row) for row in rows]

    def _load_sparse_table_candidates(
        self,
        connection: Connection,
        *,
        query: str,
        retrieval_index_run_ids: tuple[uuid.UUID, ...],
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
        limit: int | None,
    ) -> list[_Candidate]:
        params: dict[str, object] = {
            "query": query,
            "retrieval_index_run_ids": list(retrieval_index_run_ids),
            "apply_document_filter": filtered_document_ids is not None,
            "document_ids": list(filtered_document_ids or ()),
        }
        limit_sql = ""
        if limit is not None:
            params["candidate_limit"] = limit
            limit_sql = "\n                LIMIT :candidate_limit"
        query_sql = """
            WITH candidate_assets AS (
                SELECT
                    assets.id,
                    assets.retrieval_index_run_id,
                    assets.table_id,
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
            """
        query_sql += limit_sql
        query_sql += """
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
                assets.semantic_text,
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

    def _load_dense_table_candidates(
        self,
        connection: Connection,
        *,
        query: str,
        retrieval_index_run_ids: tuple[uuid.UUID, ...],
        filtered_document_ids: tuple[uuid.UUID, ...] | None,
        limit: int | None,
    ) -> list[_Candidate]:
        if self._embedding_client is None:
            return []
        with observe_operation(
            "retrieval.embedding",
            logger=logger,
            fields={
                "query": query.strip(),
                "candidate_limit": limit,
                "entity_kind": "table",
            },
        ):
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
            "apply_document_filter": filtered_document_ids is not None,
            "document_ids": list(filtered_document_ids or ()),
            "candidate_limit": limit,
        }
        limit_sql = ""
        if limit is not None:
            limit_sql = "\n                LIMIT :candidate_limit"
        self._configure_dense_vector_search(
            connection,
            candidate_limit=limit,
            filtered_document_ids=filtered_document_ids,
        )
        query_sql = """
            WITH candidate_assets AS (
                SELECT
                    assets.id,
                    assets.retrieval_index_run_id,
                    assets.table_id,
                    1 - (assets.embedding <=> CAST(:query_embedding AS vector)) AS dense_score
                FROM retrieval_table_assets assets
                WHERE assets.retrieval_index_run_id = ANY(CAST(:retrieval_index_run_ids AS uuid[]))
                  AND assets.embedding IS NOT NULL
                  AND (
                      :apply_document_filter = false
                      OR assets.document_id = ANY(CAST(:document_ids AS uuid[]))
                  )
                ORDER BY assets.embedding <=> CAST(:query_embedding AS vector), assets.table_id
            """
        query_sql += limit_sql
        query_sql += """
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
                assets.semantic_text,
                candidate_assets.dense_score
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
            ORDER BY assets.embedding <=> CAST(:query_embedding AS vector), tables.id
            """
        rows = (
            connection.execute(
                text(query_sql),
                params,
            )
            .mappings()
            .all()
        )
        if self._should_retry_exact_dense_query(
            filtered_document_ids=filtered_document_ids,
            returned_count=len(rows),
            requested_limit=limit,
        ):
            exact_rows = (
                connection.execute(
                    text(
                        """
                        WITH filtered_assets AS MATERIALIZED (
                            SELECT
                                assets.id,
                                assets.retrieval_index_run_id,
                                assets.table_id,
                                1 - (
                                    assets.embedding <=> CAST(:query_embedding AS vector)
                                ) AS dense_score
                            FROM retrieval_table_assets assets
                            WHERE assets.retrieval_index_run_id = ANY(
                                CAST(:retrieval_index_run_ids AS uuid[])
                            )
                              AND assets.embedding IS NOT NULL
                              AND (
                                  :apply_document_filter = false
                                  OR assets.document_id = ANY(CAST(:document_ids AS uuid[]))
                              )
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
                            COALESCE(
                                revisions.title,
                                documents.title,
                                'Untitled document'
                            ) AS document_title,
                            COALESCE(sections.heading_path, '[]'::jsonb) AS section_path,
                            runs.id AS retrieval_index_run_id,
                            runs.index_version,
                            runs.parser_source,
                            COALESCE(jobs.warnings, '[]'::jsonb) AS warnings,
                            assets.semantic_text,
                            filtered_assets.dense_score
                        FROM filtered_assets
                        JOIN retrieval_table_assets assets
                            ON assets.id = filtered_assets.id
                        JOIN retrieval_index_runs runs
                            ON runs.id = filtered_assets.retrieval_index_run_id
                        JOIN document_tables tables
                            ON tables.id = filtered_assets.table_id
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
                        ORDER BY filtered_assets.dense_score DESC, tables.id
                        LIMIT COALESCE(CAST(:candidate_limit AS integer), 2147483647)
                        """
                    ),
                    params,
                )
                .mappings()
                .all()
            )
            rows = exact_rows
        return [self._row_to_table_candidate(row) for row in rows]

    def _fuse_candidates(
        self,
        sparse_candidates: list[_Candidate],
        dense_candidates: list[_Candidate],
        *,
        fused_limit: int | None,
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
        return fused if fused_limit is None else fused[:fused_limit]

    def _rerank_candidates(
        self,
        *,
        query: str,
        candidates: list[_Candidate],
        limit: int | None,
    ) -> list[_Candidate]:
        if not candidates:
            return []
        if self._reranker_client is None:
            for candidate in candidates:
                candidate.score = candidate.fused_score
            ranked = sorted(
                candidates,
                key=lambda candidate: (-candidate.score, str(candidate.entity_id)),
            )
            return ranked if limit is None else ranked[:limit]

        with observe_operation(
            "retrieval.rerank",
            logger=logger,
            fields={
                "query": query.strip(),
                "candidate_count": len(candidates),
                "result_limit": limit,
            },
        ):
            reranked_items = self._reranker_client.rerank(
                query=query,
                documents=[candidate.rerank_text for candidate in candidates],
                top_n=None if limit is None else min(limit, len(candidates)),
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
        return ranked if limit is None else ranked[:limit]

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
            rerank_text=row.get("semantic_text") or row.get("search_text") or "",
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
        selected_passages_by_section: dict[uuid.UUID, set[uuid.UUID]] = defaultdict(set)
        for passage in passages:
            selected_passages_by_section[passage.section_id].add(passage.passage_id)

        selected_passage_ids = tuple(dict.fromkeys(passage.passage_id for passage in passages))
        passage_windows_query = self._parent_section_passage_windows_subquery(
            section_ids=section_ids,
        )
        section_row_counts = self._load_parent_section_row_counts(
            connection,
            passage_windows_query=passage_windows_query,
        )
        if not section_row_counts:
            return ()
        passage_windows = self._load_parent_section_passage_windows(
            connection,
            passage_windows_query=passage_windows_query,
            selected_passage_ids=selected_passage_ids,
        )

        requested_ranges_by_section: dict[uuid.UUID, list[tuple[int, int]]] = defaultdict(list)
        for section_id in section_ids:
            section_row_count = section_row_counts.get(section_id, 0)
            if section_row_count <= 0:
                continue
            selected_row_numbers = [
                cast(int, row["row_number"])
                for row in passage_windows
                if cast(uuid.UUID, row["section_id"]) == section_id
                and cast(uuid.UUID, row["passage_id"])
                in selected_passages_by_section.get(section_id, set())
            ]
            if selected_row_numbers:
                for row_number in selected_row_numbers:
                    requested_ranges_by_section[section_id].append(
                        (
                            max(1, row_number - PARENT_SIBLINGS_BEFORE),
                            min(section_row_count, row_number + PARENT_SIBLINGS_AFTER),
                        )
                    )
                continue
            requested_ranges_by_section[section_id].append(
                (1, min(section_row_count, PARENT_SIBLINGS_AFTER + 1))
            )

        requested_ranges_by_section = {
            section_id: list(self._merge_parent_section_ranges(ranges))
            for section_id, ranges in requested_ranges_by_section.items()
            if ranges
        }
        if not requested_ranges_by_section:
            return ()

        rows = self._load_parent_section_rows(
            connection,
            passage_windows_query=passage_windows_query,
            requested_ranges_by_section=requested_ranges_by_section,
        )
        grouped_rows: dict[uuid.UUID, list[Any]] = defaultdict(list)
        for row in rows:
            grouped_rows[cast(uuid.UUID, row["section_id"])].append(row)

        parent_sections: list[ParentSectionResult] = []
        for section_id in section_ids:
            section_rows = grouped_rows.get(section_id, [])
            if not section_rows:
                continue
            selected_ids = selected_passages_by_section.get(section_id, set())
            supporting_passages: list[ContextPassage] = []
            for row in section_rows:
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
            if section_row_counts.get(section_id, 0) > len(section_rows):
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

    def _parent_section_passage_windows_subquery(
        self,
        *,
        section_ids: tuple[uuid.UUID, ...],
    ):
        return (
            select(
                _DOCUMENT_PASSAGES_SQL.c.id.label("passage_id"),
                _DOCUMENT_PASSAGES_SQL.c.section_id,
                _DOCUMENT_PASSAGES_SQL.c.chunk_ordinal,
                func.row_number()
                .over(
                    partition_by=_DOCUMENT_PASSAGES_SQL.c.section_id,
                    order_by=(
                        _DOCUMENT_PASSAGES_SQL.c.chunk_ordinal,
                        _DOCUMENT_PASSAGES_SQL.c.id,
                    ),
                )
                .label("row_number"),
                func.count()
                .over(partition_by=_DOCUMENT_PASSAGES_SQL.c.section_id)
                .label("section_row_count"),
            )
            .where(_DOCUMENT_PASSAGES_SQL.c.section_id.in_(section_ids))
            .subquery("parent_section_passage_windows")
        )

    def _load_parent_section_passage_windows(
        self,
        connection: Connection,
        *,
        passage_windows_query: Any,
        selected_passage_ids: tuple[uuid.UUID, ...],
    ) -> tuple[Any, ...]:
        if not selected_passage_ids:
            return ()
        selected_rows = (
            connection.execute(
                select(
                    passage_windows_query.c.passage_id,
                    passage_windows_query.c.section_id,
                    passage_windows_query.c.row_number,
                    passage_windows_query.c.section_row_count,
                ).where(passage_windows_query.c.passage_id.in_(selected_passage_ids))
            )
            .mappings()
            .all()
        )
        return tuple(selected_rows)

    def _load_parent_section_row_counts(
        self,
        connection: Connection,
        *,
        passage_windows_query: Any,
    ) -> dict[uuid.UUID, int]:
        rows = (
            connection.execute(
                select(
                    passage_windows_query.c.section_id,
                    func.max(passage_windows_query.c.section_row_count).label("section_row_count"),
                ).group_by(passage_windows_query.c.section_id)
            )
            .mappings()
            .all()
        )
        return {
            cast(uuid.UUID, row["section_id"]): cast(int, row["section_row_count"]) for row in rows
        }

    def _load_parent_section_rows(
        self,
        connection: Connection,
        *,
        passage_windows_query: Any,
        requested_ranges_by_section: dict[uuid.UUID, list[tuple[int, int]]],
    ) -> list[Any]:
        range_clauses = [
            and_(
                passage_windows_query.c.section_id == section_id,
                passage_windows_query.c.row_number.between(start, end),
            )
            for section_id, ranges in requested_ranges_by_section.items()
            for start, end in ranges
        ]
        if not range_clauses:
            return []
        return list(
            connection.execute(
                select(
                    passage_windows_query.c.passage_id.label("passage_id"),
                    passage_windows_query.c.section_id,
                    passage_windows_query.c.chunk_ordinal,
                    passage_windows_query.c.section_row_count,
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
                .select_from(passage_windows_query)
                .join(
                    _DOCUMENT_PASSAGES_SQL,
                    _DOCUMENT_PASSAGES_SQL.c.id == passage_windows_query.c.passage_id,
                )
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
                .where(or_(*range_clauses))
                .order_by(
                    passage_windows_query.c.section_id,
                    passage_windows_query.c.row_number,
                    passage_windows_query.c.passage_id,
                )
            )
            .mappings()
            .all()
        )

    def _merge_parent_section_ranges(
        self,
        ranges: list[tuple[int, int]],
    ) -> tuple[tuple[int, int], ...]:
        merged_ranges: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if not merged_ranges or start > merged_ranges[-1][1] + 1:
                merged_ranges.append((start, end))
                continue
            previous_start, previous_end = merged_ranges[-1]
            merged_ranges[-1] = (previous_start, max(previous_end, end))
        return tuple(merged_ranges)

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
