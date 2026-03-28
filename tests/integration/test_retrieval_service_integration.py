from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import insert, text
from sqlalchemy.exc import IntegrityError

from paper_context.db.session import connection_scope
from paper_context.models import (
    Document,
    DocumentPassage,
    DocumentSection,
    DocumentTable,
    IngestJob,
    RetrievalIndexRun,
)
from paper_context.pagination import encode_cursor, fingerprint_payload
from paper_context.retrieval import DocumentRetrievalIndexer, RetrievalService
from paper_context.retrieval import service as retrieval_service_module
from paper_context.retrieval.clients import EmbeddingBatch
from paper_context.retrieval.types import RerankItem, RetrievalError, RetrievalFilters

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_postgres,
    pytest.mark.slow,
]

VECTOR_DIMENSIONS = 1024


def _vector_values(index: int) -> list[float]:
    vector = [0.0] * VECTOR_DIMENSIONS
    vector[index] = 1.0
    return vector


def _vector_string(index: int) -> str:
    return (
        "["
        + ",".join("1.0" if position == index else "0.0" for position in range(VECTOR_DIMENSIONS))
        + "]"
    )


class FixedEmbeddingClient:
    provider = "fake"

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.model = "fixed-embedding"
        self._mapping = mapping

    def embed(
        self,
        texts: list[str],
        *,
        input_type: str,
    ) -> EmbeddingBatch:
        del input_type
        embeddings = tuple(tuple(self._mapping.get(text, _vector_values(999))) for text in texts)
        return EmbeddingBatch(
            provider=self.provider,
            model=self.model,
            dimensions=VECTOR_DIMENSIONS,
            embeddings=embeddings,
        )


class CountingEmbeddingClient(FixedEmbeddingClient):
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        super().__init__(mapping)
        self.batch_sizes: list[int] = []

    def embed(
        self,
        texts: list[str],
        *,
        input_type: str,
    ) -> EmbeddingBatch:
        self.batch_sizes.append(len(texts))
        return super().embed(texts, input_type=input_type)


class IdentityReranker:
    provider = "fake"
    model = "identity"

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankItem]:
        del query
        indexes = list(range(len(documents)))
        if top_n is not None:
            indexes = indexes[:top_n]
        return [
            RerankItem(index=index, score=float(len(indexes) - position))
            for position, index in enumerate(indexes)
        ]


class CountingReranker(IdentityReranker):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankItem]:
        self.calls.append(list(documents))
        return super().rerank(query=query, documents=documents, top_n=top_n)


def _insert_run(
    connection,
    *,
    run_id: UUID | None = None,
    document_id: UUID,
    revision_id: UUID,
    ingest_job_id: UUID,
    index_version: str,
    parser_source: str,
    is_active: bool,
    activated_at: datetime,
    created_at: datetime,
) -> UUID:
    run_id = run_id or uuid4()
    connection.execute(
        insert(RetrievalIndexRun).values(
            id=run_id,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version=index_version,
            embedding_provider="fake",
            embedding_model="fixed-embedding",
            embedding_dimensions=VECTOR_DIMENSIONS,
            reranker_provider="fake",
            reranker_model="identity",
            chunking_version="phase2",
            parser_source=parser_source,
            status="ready",
            is_active=is_active,
            activated_at=activated_at,
            deactivated_at=None if is_active else activated_at,
            created_at=created_at,
        )
    )
    return run_id


def _insert_passage_asset(
    connection,
    *,
    run_id: UUID,
    revision_id: UUID,
    passage_id: UUID,
    document_id: UUID,
    section_id: UUID,
    publication_year: int | None,
    search_text: str,
    embedding: str,
) -> None:
    connection.execute(
        text(
            """
            INSERT INTO retrieval_passage_assets (
                id,
                retrieval_index_run_id,
                revision_id,
                passage_id,
                document_id,
                section_id,
                publication_year,
                search_text,
                search_tsvector,
                embedding
            )
            VALUES (
                :id,
                :retrieval_index_run_id,
                :revision_id,
                :passage_id,
                :document_id,
                :section_id,
                :publication_year,
                :search_text,
                to_tsvector('english', :search_text),
                CAST(:embedding AS vector)
            )
            """
        ),
        {
            "id": uuid4(),
            "retrieval_index_run_id": run_id,
            "revision_id": revision_id,
            "passage_id": passage_id,
            "document_id": document_id,
            "section_id": section_id,
            "publication_year": publication_year,
            "search_text": search_text,
            "embedding": embedding,
        },
    )


def _insert_table_asset(
    connection,
    *,
    run_id: UUID,
    revision_id: UUID,
    table_id: UUID,
    document_id: UUID,
    section_id: UUID,
    publication_year: int | None,
    search_text: str,
    semantic_text: str | None = None,
    embedding: str | None = None,
) -> None:
    table_semantic_text = semantic_text or search_text
    table_embedding = embedding or _vector_string(999)
    connection.execute(
        text(
            """
            INSERT INTO retrieval_table_assets (
                id,
                retrieval_index_run_id,
                revision_id,
                table_id,
                document_id,
                section_id,
                publication_year,
                search_text,
                semantic_text,
                search_tsvector,
                embedding
            )
            VALUES (
                :id,
                :retrieval_index_run_id,
                :revision_id,
                :table_id,
                :document_id,
                :section_id,
                :publication_year,
                :search_text,
                :semantic_text,
                to_tsvector('english', :search_text),
                CAST(:embedding AS vector)
            )
            """
        ),
        {
            "id": uuid4(),
            "retrieval_index_run_id": run_id,
            "revision_id": revision_id,
            "table_id": table_id,
            "document_id": document_id,
            "section_id": section_id,
            "publication_year": publication_year,
            "search_text": search_text,
            "semantic_text": table_semantic_text,
            "embedding": table_embedding,
        },
    )


def _insert_revisioned_document(
    connection,
    *,
    document_id: UUID,
    revision_id: UUID,
    revision_number: int,
    title: str,
    authors: list[str] | None = None,
    abstract: str | None = None,
    publication_year: int | None = None,
    source_type: str = "upload",
    quant_tags: dict[str, object] | None = None,
    current_status: str = "ready",
    created_at: datetime,
    updated_at: datetime,
    create_document: bool = True,
    activate: bool = True,
) -> None:
    if create_document:
        connection.execute(
            insert(Document).values(
                id=document_id,
                title=title,
                source_type=source_type,
                current_status=current_status,
                created_at=created_at,
                updated_at=updated_at,
            )
        )
    connection.execute(
        text(
            """
            INSERT INTO document_revisions (
                id,
                document_id,
                revision_number,
                status,
                title,
                authors,
                abstract,
                publication_year,
                source_type,
                metadata_confidence,
                quant_tags,
                source_artifact_id,
                ingest_job_id,
                activated_at,
                superseded_at,
                created_at,
                updated_at
            ) VALUES (
                :id,
                :document_id,
                :revision_number,
                :status,
                :title,
                CAST(:authors AS jsonb),
                :abstract,
                :publication_year,
                :source_type,
                NULL,
                CAST(:quant_tags AS jsonb),
                NULL,
                NULL,
                NULL,
                NULL,
                :created_at,
                :updated_at
            )
            """
        ),
        {
            "id": revision_id,
            "document_id": document_id,
            "revision_number": revision_number,
            "status": "ready",
            "title": title,
            "authors": json.dumps(authors or []),
            "abstract": abstract,
            "publication_year": publication_year,
            "source_type": source_type,
            "quant_tags": json.dumps(quant_tags or {}),
            "created_at": created_at,
            "updated_at": updated_at,
        },
    )
    if activate:
        connection.execute(
            text(
                """
                UPDATE documents
                SET active_revision_id = :active_revision_id,
                    updated_at = :updated_at
                WHERE id = :id
                """
            ),
            {
                "id": document_id,
                "active_revision_id": revision_id,
                "updated_at": updated_at,
            },
        )


def _insert_revisioned_job(
    connection,
    *,
    ingest_job_id: UUID,
    document_id: UUID,
    revision_id: UUID,
    trigger: str,
    warnings: list[str],
    created_at: datetime,
) -> None:
    connection.execute(
        insert(IngestJob).values(
            id=ingest_job_id,
            document_id=document_id,
            revision_id=revision_id,
            status="ready",
            trigger=trigger,
            warnings=warnings,
            created_at=created_at,
            started_at=created_at,
            finished_at=created_at,
        )
    )


def _insert_revisioned_section(
    connection,
    *,
    section_id: UUID,
    document_id: UUID,
    revision_id: UUID,
    heading: str,
    heading_path: list[str],
    ordinal: int = 1,
    page_start: int = 1,
    page_end: int = 1,
) -> None:
    connection.execute(
        insert(DocumentSection).values(
            id=section_id,
            document_id=document_id,
            revision_id=revision_id,
            heading=heading,
            heading_path=heading_path,
            ordinal=ordinal,
            page_start=page_start,
            page_end=page_end,
        )
    )


def _insert_revisioned_passage(
    connection,
    *,
    passage_id: UUID,
    document_id: UUID,
    revision_id: UUID,
    section_id: UUID,
    chunk_ordinal: int,
    body_text: str,
    contextualized_text: str,
    page_start: int,
    page_end: int,
    token_count: int | None = None,
    provenance_offsets: dict[str, object] | None = None,
) -> None:
    connection.execute(
        insert(DocumentPassage).values(
            id=passage_id,
            document_id=document_id,
            revision_id=revision_id,
            section_id=section_id,
            chunk_ordinal=chunk_ordinal,
            body_text=body_text,
            contextualized_text=contextualized_text,
            token_count=token_count,
            page_start=page_start,
            page_end=page_end,
            provenance_offsets=provenance_offsets,
            artifact_id=None,
        )
    )


def _insert_revisioned_table(
    connection,
    *,
    table_id: UUID,
    document_id: UUID,
    revision_id: UUID,
    section_id: UUID,
    caption: str,
    headers_json: list[object],
    rows_json: list[list[object]],
    page_start: int,
    page_end: int,
) -> None:
    connection.execute(
        insert(DocumentTable).values(
            id=table_id,
            document_id=document_id,
            revision_id=revision_id,
            section_id=section_id,
            caption=caption,
            table_type="lexical",
            headers_json=headers_json,
            rows_json=rows_json,
            page_start=page_start,
            page_end=page_end,
            artifact_id=None,
        )
    )


def test_search_passages_and_tables_return_only_active_index_version_rows(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    active_index_version = "mvp-v2"
    legacy_index_version = "mvp-v1"

    document_id = uuid4()
    legacy_revision_id = uuid4()
    active_revision_id = uuid4()
    legacy_section_id = uuid4()
    active_section_id = uuid4()
    legacy_passage_id = uuid4()
    active_passage_id = uuid4()
    legacy_table_id = uuid4()
    active_table_id = uuid4()
    legacy_ingest_job_id = uuid4()
    active_ingest_job_id = uuid4()
    legacy_run_id = uuid4()
    active_run_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=legacy_revision_id,
            revision_number=1,
            title="Retrieval paper",
            publication_year=2023,
            created_at=now - timedelta(minutes=2),
            updated_at=now - timedelta(minutes=2),
            activate=False,
        )
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=active_revision_id,
            revision_number=2,
            title="Retrieval paper",
            publication_year=2024,
            created_at=now - timedelta(minutes=1),
            updated_at=now - timedelta(minutes=1),
            create_document=False,
            activate=True,
        )
        for (
            revision_id,
            section_id,
            passage_id,
            table_id,
            ingest_job_id,
            run_id,
            is_active,
            activated_at,
            created_at,
        ) in [
            (
                legacy_revision_id,
                legacy_section_id,
                legacy_passage_id,
                legacy_table_id,
                legacy_ingest_job_id,
                legacy_run_id,
                False,
                now - timedelta(minutes=2),
                now - timedelta(minutes=2),
            ),
            (
                active_revision_id,
                active_section_id,
                active_passage_id,
                active_table_id,
                active_ingest_job_id,
                active_run_id,
                True,
                now - timedelta(minutes=1),
                now - timedelta(minutes=1),
            ),
        ]:
            connection.execute(
                insert(IngestJob).values(
                    id=ingest_job_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    status="ready",
                    trigger="upload",
                    warnings=[],
                    created_at=created_at,
                    started_at=created_at,
                    finished_at=created_at,
                )
            )
            connection.execute(
                insert(DocumentSection).values(
                    id=section_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    heading="Methods",
                    heading_path=["Methods"],
                    ordinal=1,
                    page_start=1,
                    page_end=1,
                )
            )
            connection.execute(
                insert(DocumentPassage).values(
                    id=passage_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    chunk_ordinal=1,
                    body_text="Shared methods alpha signal is present here.",
                    contextualized_text=(
                        "Document title: Retrieval paper\n"
                        "Section path: Methods\n"
                        "Local heading context: Methods\n\n"
                        "Shared methods alpha signal is present here."
                    ),
                    token_count=7,
                    page_start=1,
                    page_end=1,
                    provenance_offsets={"pages": [1], "charspans": [[0, 43]]},
                    artifact_id=None,
                )
            )
            connection.execute(
                insert(DocumentTable).values(
                    id=table_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    caption="Shared methods table",
                    table_type="lexical",
                    headers_json=["A", "B"],
                    rows_json=[["1", "2"], ["3", "4"]],
                    page_start=1,
                    page_end=1,
                    artifact_id=None,
                )
            )
            run_id = _insert_run(
                connection,
                run_id=run_id,
                document_id=document_id,
                revision_id=revision_id,
                ingest_job_id=ingest_job_id,
                index_version=legacy_index_version if not is_active else active_index_version,
                parser_source="docling",
                is_active=is_active,
                activated_at=activated_at,
                created_at=created_at,
            )
            _insert_passage_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                passage_id=passage_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text="Shared methods alpha signal is present here.",
                embedding=_vector_string(10 if not is_active else 11),
            )
            _insert_table_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                table_id=table_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text="Shared methods table A B 1 2 3 4",
                embedding=_vector_string(100),
            )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        embedding_client=FixedEmbeddingClient(
            {
                "shared methods": _vector_values(100),
            }
        ),
        reranker_client=IdentityReranker(),
    )

    passages = service.search_passages(query="shared methods")
    tables = service.search_tables(query="shared methods")

    assert len(passages) == 1
    assert passages[0].index_version == active_index_version
    assert passages[0].retrieval_index_run_id == active_run_id
    assert passages[0].text == "Shared methods alpha signal is present here."
    assert passages[0].retrieval_modes == ("sparse", "dense")

    assert len(tables) == 1
    assert tables[0].index_version == active_index_version
    assert tables[0].retrieval_index_run_id == active_run_id
    assert tables[0].preview.headers == ("A", "B")
    assert tables[0].preview.rows == (("1", "2"), ("3", "4"))
    assert tables[0].retrieval_modes == ("sparse", "dense")


def test_active_revision_controls_reads_and_hides_previous_revision(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    legacy_revision_id = uuid4()
    active_revision_id = uuid4()
    legacy_section_id = uuid4()
    active_section_id = uuid4()
    legacy_passage_id = uuid4()
    active_passage_id = uuid4()
    legacy_table_id = uuid4()
    active_table_id = uuid4()
    legacy_ingest_job_id = uuid4()
    active_ingest_job_id = uuid4()
    legacy_run_id = uuid4()
    active_run_id = uuid4()

    def _execute(connection, sql: str, params: dict[str, object]) -> None:
        connection.execute(text(sql), params)

    with migrated_postgres_engine.begin() as connection:
        _execute(
            connection,
            """
            INSERT INTO documents (
                id,
                title,
                authors,
                abstract,
                publication_year,
                source_type,
                metadata_confidence,
                quant_tags,
                current_status,
                active_revision_id,
                created_at,
                updated_at
            ) VALUES (
                :id,
                :title,
                CAST(:authors AS jsonb),
                :abstract,
                :publication_year,
                :source_type,
                :metadata_confidence,
                CAST(:quant_tags AS jsonb),
                :current_status,
                :active_revision_id,
                :created_at,
                :updated_at
            )
            """,
            {
                "id": document_id,
                "title": "Document shell",
                "authors": json.dumps(["Shell Author"]),
                "abstract": "Shell abstract",
                "publication_year": 2023,
                "source_type": "upload",
                "metadata_confidence": 0.6,
                "quant_tags": json.dumps({"asset_universe": "equities"}),
                "current_status": "ready",
                "active_revision_id": None,
                "created_at": now - timedelta(minutes=2),
                "updated_at": now,
            },
        )

        revisions = [
            {
                "revision_id": legacy_revision_id,
                "revision_number": 1,
                "title": "Legacy title",
                "authors": ["Legacy title Author"],
                "abstract": "Legacy title abstract",
                "publication_year": 2023,
                "quant_tags": {"revision": "legacy"},
                "section_id": legacy_section_id,
                "passage_id": legacy_passage_id,
                "table_id": legacy_table_id,
                "ingest_job_id": legacy_ingest_job_id,
                "index_version": "mvp-v1",
                "is_active": False,
                "activated_at": now - timedelta(minutes=2),
                "deactivated_at": now - timedelta(minutes=1),
                "created_at": now - timedelta(minutes=2),
                "updated_at": now - timedelta(minutes=2),
                "passage_text": "Legacy title legacy signal stays hidden.",
                "table_search_text": "Legacy title legacy revision table",
                "embedding": _vector_string(41),
            },
            {
                "revision_id": active_revision_id,
                "revision_number": 2,
                "title": "Active title",
                "authors": ["Active title Author"],
                "abstract": "Active title abstract",
                "publication_year": 2024,
                "quant_tags": {"revision": "active"},
                "section_id": active_section_id,
                "passage_id": active_passage_id,
                "table_id": active_table_id,
                "ingest_job_id": active_ingest_job_id,
                "index_version": "mvp-v2",
                "is_active": True,
                "activated_at": now - timedelta(minutes=1),
                "deactivated_at": None,
                "created_at": now - timedelta(minutes=1),
                "updated_at": now - timedelta(minutes=1),
                "passage_text": "Active title active revision signal stays visible.",
                "table_search_text": "Active title active revision table",
                "embedding": _vector_string(42),
            },
        ]

        for revision in revisions:
            _execute(
                connection,
                """
                INSERT INTO document_revisions (
                    id,
                    document_id,
                    revision_number,
                    status,
                    title,
                    authors,
                    abstract,
                    publication_year,
                    quant_tags,
                    created_at,
                    updated_at
                ) VALUES (
                    :id,
                    :document_id,
                    :revision_number,
                    :status,
                    :title,
                    CAST(:authors AS jsonb),
                    :abstract,
                    :publication_year,
                    CAST(:quant_tags AS jsonb),
                    :created_at,
                    :updated_at
                )
                """,
                {
                    "id": revision["revision_id"],
                    "document_id": document_id,
                    "revision_number": revision["revision_number"],
                    "status": "ready",
                    "title": revision["title"],
                    "authors": json.dumps(revision["authors"]),
                    "abstract": revision["abstract"],
                    "publication_year": revision["publication_year"],
                    "quant_tags": json.dumps(revision["quant_tags"]),
                    "created_at": revision["created_at"],
                    "updated_at": revision["updated_at"],
                },
            )
            connection.execute(
                insert(IngestJob).values(
                    id=revision["ingest_job_id"],
                    document_id=document_id,
                    revision_id=revision["revision_id"],
                    status="ready",
                    trigger="upload" if not revision["is_active"] else "replace",
                    warnings=[],
                    created_at=revision["created_at"],
                    started_at=revision["created_at"],
                    finished_at=revision["created_at"],
                )
            )
            _execute(
                connection,
                """
                INSERT INTO document_sections (
                    id,
                    document_id,
                    revision_id,
                    parent_section_id,
                    heading,
                    heading_path,
                    ordinal,
                    page_start,
                    page_end,
                    artifact_id
                ) VALUES (
                    :id,
                    :document_id,
                    :revision_id,
                    NULL,
                    :heading,
                    CAST(:heading_path AS jsonb),
                    1,
                    1,
                    1,
                    NULL
                )
                """,
                {
                    "id": revision["section_id"],
                    "document_id": document_id,
                    "revision_id": revision["revision_id"],
                    "heading": f"{revision['title']} methods",
                    "heading_path": json.dumps([f"{revision['title']} methods"]),
                },
            )
            _execute(
                connection,
                """
                INSERT INTO document_passages (
                    id,
                    document_id,
                    revision_id,
                    section_id,
                    chunk_ordinal,
                    body_text,
                    contextualized_text,
                    token_count,
                    page_start,
                    page_end,
                    provenance_offsets,
                    quant_tags,
                    artifact_id
                ) VALUES (
                    :id,
                    :document_id,
                    :revision_id,
                    :section_id,
                    1,
                    :body_text,
                    :contextualized_text,
                    6,
                    1,
                    1,
                    CAST(:provenance_offsets AS jsonb),
                    CAST(:quant_tags AS jsonb),
                    NULL
                )
                """,
                {
                    "id": revision["passage_id"],
                    "document_id": document_id,
                    "revision_id": revision["revision_id"],
                    "section_id": revision["section_id"],
                    "body_text": revision["passage_text"],
                    "contextualized_text": (
                        "Document title: "
                        + revision["title"]
                        + "\nSection path: "
                        + f"{revision['title']} methods"
                        + "\nLocal heading context: "
                        + f"{revision['title']} methods"
                        + "\n\n"
                        + revision["passage_text"]
                    ),
                    "provenance_offsets": json.dumps(
                        {
                            "pages": [1],
                            "charspans": [[0, len(revision["passage_text"])]],
                        }
                    ),
                    "quant_tags": json.dumps({}),
                },
            )
            _execute(
                connection,
                """
                INSERT INTO document_tables (
                    id,
                    document_id,
                    revision_id,
                    section_id,
                    caption,
                    table_type,
                    headers_json,
                    rows_json,
                    page_start,
                    page_end,
                    quant_tags,
                    artifact_id
                ) VALUES (
                    :id,
                    :document_id,
                    :revision_id,
                    :section_id,
                    :caption,
                    :table_type,
                    CAST(:headers_json AS jsonb),
                    CAST(:rows_json AS jsonb),
                    1,
                    1,
                    CAST(:quant_tags AS jsonb),
                    NULL
                )
                """,
                {
                    "id": revision["table_id"],
                    "document_id": document_id,
                    "revision_id": revision["revision_id"],
                    "section_id": revision["section_id"],
                    "caption": f"{revision['title']} table",
                    "table_type": "lexical",
                    "headers_json": json.dumps(["A", "B"]),
                    "rows_json": json.dumps([["1", "2"], ["3", "4"]]),
                    "quant_tags": json.dumps({}),
                },
            )
            _execute(
                connection,
                """
                INSERT INTO retrieval_index_runs (
                    id,
                    document_id,
                    revision_id,
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
                    deactivated_at,
                    created_at
                ) VALUES (
                    :id,
                    :document_id,
                    :revision_id,
                    :ingest_job_id,
                    :index_version,
                    :embedding_provider,
                    :embedding_model,
                    :embedding_dimensions,
                    :reranker_provider,
                    :reranker_model,
                    :chunking_version,
                    :parser_source,
                    :status,
                    :is_active,
                    :activated_at,
                    :deactivated_at,
                    :created_at
                )
                """,
                {
                    "id": legacy_run_id if not revision["is_active"] else active_run_id,
                    "document_id": document_id,
                    "revision_id": revision["revision_id"],
                    "ingest_job_id": revision["ingest_job_id"],
                    "index_version": revision["index_version"],
                    "embedding_provider": "fake",
                    "embedding_model": "fixed-embedding",
                    "embedding_dimensions": VECTOR_DIMENSIONS,
                    "reranker_provider": "fake",
                    "reranker_model": "identity",
                    "chunking_version": "phase2",
                    "parser_source": "docling",
                    "status": "ready",
                    "is_active": revision["is_active"],
                    "activated_at": revision["activated_at"],
                    "deactivated_at": revision["deactivated_at"],
                    "created_at": revision["created_at"],
                },
            )
            _execute(
                connection,
                """
                INSERT INTO retrieval_passage_assets (
                    id,
                    retrieval_index_run_id,
                    revision_id,
                    document_id,
                    passage_id,
                    section_id,
                    publication_year,
                    search_text,
                    search_tsvector,
                    embedding,
                    created_at
                    ) VALUES (
                    :id,
                    :retrieval_index_run_id,
                    :revision_id,
                    :document_id,
                    :passage_id,
                    :section_id,
                    :publication_year,
                    :search_text,
                    to_tsvector('english', :search_text),
                    CAST(:embedding AS vector),
                    :created_at
                    )
                """,
                {
                    "id": uuid4(),
                    "retrieval_index_run_id": legacy_run_id
                    if not revision["is_active"]
                    else active_run_id,
                    "revision_id": revision["revision_id"],
                    "document_id": document_id,
                    "passage_id": revision["passage_id"],
                    "section_id": revision["section_id"],
                    "publication_year": revision["publication_year"],
                    "search_text": revision["passage_text"],
                    "embedding": revision["embedding"],
                    "created_at": revision["created_at"],
                },
            )
            _execute(
                connection,
                """
                INSERT INTO retrieval_table_assets (
                    id,
                    retrieval_index_run_id,
                    revision_id,
                    document_id,
                    table_id,
                    section_id,
                    publication_year,
                    search_text,
                    semantic_text,
                    search_tsvector,
                    embedding,
                    created_at
                ) VALUES (
                    :id,
                    :retrieval_index_run_id,
                    :revision_id,
                    :document_id,
                    :table_id,
                    :section_id,
                    :publication_year,
                    :search_text,
                    :semantic_text,
                    to_tsvector('english', :search_text),
                    CAST(:embedding AS vector),
                    :created_at
                )
                """,
                {
                    "id": uuid4(),
                    "retrieval_index_run_id": legacy_run_id
                    if not revision["is_active"]
                    else active_run_id,
                    "revision_id": revision["revision_id"],
                    "document_id": document_id,
                    "table_id": revision["table_id"],
                    "section_id": revision["section_id"],
                    "publication_year": revision["publication_year"],
                    "search_text": revision["table_search_text"],
                    "semantic_text": revision["table_search_text"],
                    "embedding": revision["embedding"],
                    "created_at": revision["created_at"],
                },
            )

        _execute(
            connection,
            """
            UPDATE documents
            SET active_revision_id = :active_revision_id,
                updated_at = :updated_at
            WHERE id = :id
            """,
            {
                "id": document_id,
                "active_revision_id": active_revision_id,
                "updated_at": now,
            },
        )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
        embedding_client=FixedEmbeddingClient(
            {
                "active revision": _vector_values(42),
                "active revision signal": _vector_values(42),
            }
        ),
        reranker_client=IdentityReranker(),
    )

    passages = service.search_passages(query="active revision")
    tables = service.search_tables(query="active revision")
    context = service.get_passage_context(passage_id=active_passage_id)
    old_context = service.get_passage_context(passage_id=legacy_passage_id)
    table = service.get_table(table_id=active_table_id)
    old_table = service.get_table(table_id=legacy_table_id)
    pack = service.build_context_pack(query="active revision")

    assert len(passages) == 1
    assert passages[0].passage_id == active_passage_id
    assert passages[0].document_title == "Active title"
    assert passages[0].index_version == "mvp-v2"

    assert len(tables) == 1
    assert tables[0].table_id == active_table_id
    assert tables[0].document_title == "Active title"
    assert tables[0].index_version == "mvp-v2"
    assert tables[0].retrieval_modes == ("sparse", "dense")

    assert context is not None
    assert context.passage.document_title == "Active title"
    assert old_context is None

    assert table is not None
    assert table.document_title == "Active title"
    assert old_table is None

    assert pack.documents[0].title == "Active title"
    assert pack.documents[0].publication_year == 2024
    assert pack.provenance.active_index_version == "mvp-v2"


def test_search_passages_and_tables_include_all_active_versions_during_rollout(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    newer_document_id = uuid4()
    older_document_id = uuid4()
    newer_section_id = uuid4()
    older_section_id = uuid4()
    newer_passage_id = uuid4()
    older_passage_id = uuid4()
    newer_table_id = uuid4()
    older_table_id = uuid4()
    newer_ingest_job_id = uuid4()
    older_ingest_job_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        for (
            document_id,
            section_id,
            ingest_job_id,
            title,
            index_version,
            activation_offset_minutes,
        ) in [
            (
                newer_document_id,
                newer_section_id,
                newer_ingest_job_id,
                "New rollout paper",
                "mvp-v2",
                0,
            ),
            (
                older_document_id,
                older_section_id,
                older_ingest_job_id,
                "Older rollout paper",
                "mvp-v1",
                1,
            ),
        ]:
            created_at = now - timedelta(minutes=activation_offset_minutes)
            revision_id = uuid4()
            _insert_revisioned_document(
                connection,
                document_id=document_id,
                revision_id=revision_id,
                revision_number=1,
                title=title,
                created_at=created_at,
                updated_at=created_at,
            )
            connection.execute(
                insert(IngestJob).values(
                    id=ingest_job_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    status="ready",
                    trigger="upload",
                    warnings=[],
                    created_at=created_at,
                    started_at=created_at,
                    finished_at=created_at,
                )
            )
            connection.execute(
                insert(DocumentSection).values(
                    id=section_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    heading="Methods",
                    heading_path=["Methods"],
                    ordinal=1,
                    page_start=1,
                    page_end=1,
                )
            )
            run_id = _insert_run(
                connection,
                document_id=document_id,
                revision_id=revision_id,
                ingest_job_id=ingest_job_id,
                index_version=index_version,
                parser_source="docling",
                is_active=True,
                activated_at=created_at,
                created_at=created_at,
            )
            passage_id = newer_passage_id if document_id == newer_document_id else older_passage_id
            table_id = newer_table_id if document_id == newer_document_id else older_table_id
            connection.execute(
                insert(DocumentPassage).values(
                    id=passage_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    chunk_ordinal=1,
                    body_text=f"Rollout keyword stays searchable for {title}.",
                    contextualized_text=f"Rollout keyword stays searchable for {title}.",
                    token_count=6,
                    page_start=1,
                    page_end=1,
                    provenance_offsets={"pages": [1], "charspans": [[0, 20]]},
                    artifact_id=None,
                )
            )
            connection.execute(
                insert(DocumentTable).values(
                    id=table_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    caption=f"{title} table",
                    table_type="lexical",
                    headers_json=["A"],
                    rows_json=[["1"]],
                    page_start=1,
                    page_end=1,
                    artifact_id=None,
                )
            )
            _insert_passage_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                passage_id=passage_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text=f"Rollout keyword stays searchable for {title}.",
                embedding=_vector_string(20 if document_id == newer_document_id else 21),
            )
            _insert_table_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                table_id=table_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text=f"Rollout keyword {title} table",
                embedding=_vector_string(20 if document_id == newer_document_id else 21),
            )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        embedding_client=FixedEmbeddingClient({"rollout keyword": _vector_values(20)}),
        reranker_client=IdentityReranker(),
    )

    passages = service.search_passages(query="rollout keyword")
    tables = service.search_tables(query="rollout keyword")
    pack = service.build_context_pack(query="rollout keyword")

    assert {result.document_id for result in passages} == {newer_document_id}
    assert {result.index_version for result in passages} == {"mvp-v2"}
    assert {result.document_id for result in tables} == {newer_document_id}
    assert {result.index_version for result in tables} == {"mvp-v2"}
    assert {result.retrieval_modes for result in tables} == {("sparse", "dense")}
    assert pack.provenance.active_index_version == "mvp-v2"
    assert {result.document_id for result in pack.passages} == {newer_document_id}
    assert {result.index_version for result in pack.passages} == {"mvp-v2"}


def test_build_context_pack_includes_table_led_documents(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    passage_document_id = uuid4()
    table_document_id = uuid4()
    passage_revision_id = uuid4()
    table_revision_id = uuid4()
    passage_section_id = uuid4()
    table_section_id = uuid4()
    passage_ingest_job_id = uuid4()
    table_ingest_job_id = uuid4()
    passage_run_id = uuid4()
    table_run_id = uuid4()
    passage_id = uuid4()
    table_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=passage_document_id,
            revision_id=passage_revision_id,
            revision_number=1,
            title="Passage-led paper",
            created_at=now,
            updated_at=now,
        )
        _insert_revisioned_document(
            connection,
            document_id=table_document_id,
            revision_id=table_revision_id,
            revision_number=1,
            title="Table-led paper",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=passage_ingest_job_id,
                document_id=passage_document_id,
                revision_id=passage_revision_id,
                source_artifact_id=None,
                status="ready",
                failure_code=None,
                failure_message=None,
                warnings=[],
                stage_timings={},
                started_at=now,
                finished_at=now,
                trigger="upload",
                created_at=now,
            )
        )
        connection.execute(
            insert(IngestJob).values(
                id=table_ingest_job_id,
                document_id=table_document_id,
                revision_id=table_revision_id,
                source_artifact_id=None,
                status="ready",
                failure_code=None,
                failure_message=None,
                warnings=[],
                stage_timings={},
                started_at=now,
                finished_at=now,
                trigger="upload",
                created_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=passage_section_id,
                document_id=passage_document_id,
                revision_id=passage_revision_id,
                parent_section_id=None,
                heading="Methods",
                heading_path=["Methods"],
                ordinal=1,
                page_start=1,
                page_end=1,
                artifact_id=None,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=table_section_id,
                document_id=table_document_id,
                revision_id=table_revision_id,
                parent_section_id=None,
                heading="Results",
                heading_path=["Results"],
                ordinal=1,
                page_start=2,
                page_end=2,
                artifact_id=None,
            )
        )
        connection.execute(
            insert(DocumentPassage).values(
                id=passage_id,
                document_id=passage_document_id,
                revision_id=passage_revision_id,
                section_id=passage_section_id,
                chunk_ordinal=0,
                body_text="Table-led query passage.",
                contextualized_text="Table-led query passage.",
                token_count=4,
                page_start=1,
                page_end=1,
                provenance_offsets={},
                artifact_id=None,
            )
        )
        connection.execute(
            insert(DocumentTable).values(
                id=table_id,
                document_id=table_document_id,
                revision_id=table_revision_id,
                section_id=table_section_id,
                caption="Unrelated table",
                table_type="lexical",
                headers_json=["A"],
                rows_json=[["1"]],
                page_start=2,
                page_end=2,
                artifact_id=None,
            )
        )
        _insert_run(
            connection,
            run_id=passage_run_id,
            document_id=passage_document_id,
            revision_id=passage_revision_id,
            ingest_job_id=passage_ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        _insert_run(
            connection,
            run_id=table_run_id,
            document_id=table_document_id,
            revision_id=table_revision_id,
            ingest_job_id=table_ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        _insert_passage_asset(
            connection,
            run_id=passage_run_id,
            revision_id=passage_revision_id,
            passage_id=passage_id,
            document_id=passage_document_id,
            section_id=passage_section_id,
            publication_year=None,
            search_text="Table-led query passage.",
            embedding=_vector_string(7),
        )
        _insert_table_asset(
            connection,
            run_id=table_run_id,
            revision_id=table_revision_id,
            table_id=table_id,
            document_id=table_document_id,
            section_id=table_section_id,
            publication_year=None,
            search_text="Table-led query table content",
            embedding=_vector_string(7),
        )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
        embedding_client=FixedEmbeddingClient({"table-led query": _vector_values(7)}),
        reranker_client=IdentityReranker(),
    )

    pack = service.build_context_pack(query="table-led query")

    assert {result.document_id for result in pack.passages} == {passage_document_id}
    assert {result.document_id for result in pack.tables} == {table_document_id}
    assert {document.document_id for document in pack.documents} == {
        passage_document_id,
        table_document_id,
    }
    assert pack.tables[0].retrieval_modes == ("sparse", "dense")


def test_search_passages_dense_path_uses_embeddings(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    passage_id = uuid4()
    ingest_job_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Dense retrieval paper",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Results",
                heading_path=["Results"],
                ordinal=1,
                page_start=2,
                page_end=2,
            )
        )
        connection.execute(
            insert(DocumentPassage).values(
                id=passage_id,
                document_id=document_id,
                revision_id=revision_id,
                section_id=section_id,
                chunk_ordinal=1,
                body_text="Latent signal stays hidden from lexical matching.",
                contextualized_text=(
                    "Document title: Dense retrieval paper\n"
                    "Section path: Results\n"
                    "Local heading context: Results\n\n"
                    "Latent signal stays hidden from lexical matching."
                ),
                token_count=7,
                page_start=2,
                page_end=2,
                provenance_offsets={"pages": [2], "charspans": [[0, 49]]},
                artifact_id=None,
            )
        )
        run_id = _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        _insert_passage_asset(
            connection,
            run_id=run_id,
            revision_id=revision_id,
            passage_id=passage_id,
            document_id=document_id,
            section_id=section_id,
            publication_year=None,
            search_text="Latent signal stays hidden from lexical matching.",
            embedding=_vector_string(1),
        )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        embedding_client=FixedEmbeddingClient(
            {
                "dense only": _vector_values(1),
            }
        ),
        reranker_client=IdentityReranker(),
    )

    results = service.search_passages(query="dense only")

    assert len(results) == 1
    assert results[0].passage_id == passage_id
    assert results[0].index_version == "mvp-v2"
    assert results[0].retrieval_modes == ("dense",)


def test_build_context_pack_propagates_parent_warning_and_provenance(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    ingest_job_id = uuid4()
    passage_ids = [uuid4(), uuid4(), uuid4()]

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Context pack paper",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=["parser_fallback_used"],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Methods",
                heading_path=["Methods"],
                ordinal=1,
                page_start=1,
                page_end=1,
            )
        )
        run_id = _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        for position, passage_id in enumerate(passage_ids):
            connection.execute(
                insert(DocumentPassage).values(
                    id=passage_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    chunk_ordinal=position + 1,
                    body_text=(
                        "Selected sparse context."
                        if position == 0
                        else f"Sibling context {position}."
                    ),
                    contextualized_text=(
                        "Document title: Context pack paper\n"
                        "Section path: Methods\n"
                        "Local heading context: Methods\n\n"
                        "Selected sparse context."
                        if position == 0
                        else (
                            "Document title: Context pack paper\n"
                            "Section path: Methods\n"
                            "Local heading context: Methods\n\n"
                            f"Sibling context {position}."
                        )
                    ),
                    token_count=4,
                    page_start=1,
                    page_end=1,
                    provenance_offsets={"pages": [1], "charspans": [[0, 20]]},
                    artifact_id=None,
                )
            )
            _insert_passage_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                passage_id=passage_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text=(
                    "Selected sparse context." if position == 0 else f"Sibling context {position}."
                ),
                embedding=_vector_string(20 + position),
            )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        reranker_client=IdentityReranker(),
    )

    pack = service.build_context_pack(query="Selected sparse context")

    assert pack.query == "Selected sparse context"
    assert len(pack.passages) == 1
    assert pack.passages[0].warnings == ("parser_fallback_used",)
    assert pack.provenance.active_index_version == "mvp-v2"
    assert pack.provenance.retrieval_modes == ("sparse",)
    assert pack.documents[0].active_index_version == "mvp-v2"
    assert pack.warnings == ("parser_fallback_used", "parent_context_truncated")
    assert len(pack.parent_sections) == 1
    assert len(pack.parent_sections[0].supporting_passages) == 2


def test_document_retrieval_indexer_batches_passage_embeddings(
    migrated_postgres_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retrieval_service_module, "INDEX_BUILD_BATCH_SIZE", 2)
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    ingest_job_id = uuid4()
    section_id = uuid4()

    contextualized_texts = [
        (
            "Document title: Retrieval paper\n"
            "Section path: Methods\n"
            "Local heading context: Methods\n\n"
            f"passage {index}"
        )
        for index in range(5)
    ]
    embedding_client = CountingEmbeddingClient(
        {text: _vector_values(index) for index, text in enumerate(contextualized_texts)}
    )
    indexer = DocumentRetrievalIndexer(
        index_version="mvp-v2",
        chunking_version="phase2",
        embedding_model="fixed-embedding",
        reranker_model="identity",
        embedding_client=embedding_client,
        reranker_client=IdentityReranker(),
    )

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Retrieval paper",
            created_at=now - timedelta(minutes=2),
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now - timedelta(minutes=1),
                started_at=now - timedelta(minutes=1),
                finished_at=now - timedelta(minutes=1),
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Methods",
                heading_path=["Methods"],
                ordinal=1,
                page_start=1,
                page_end=1,
            )
        )
        for index, contextualized_text in enumerate(contextualized_texts, start=1):
            connection.execute(
                insert(DocumentPassage).values(
                    id=uuid4(),
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    chunk_ordinal=index,
                    body_text=f"passage {index}",
                    contextualized_text=contextualized_text,
                    token_count=3,
                    page_start=1,
                    page_end=1,
                    provenance_offsets={"pages": [1], "charspans": [[0, 9]]},
                    artifact_id=None,
                )
            )

        run_id = indexer.rebuild(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            parser_source="docling",
        )

        passage_asset_count = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM retrieval_passage_assets
                WHERE retrieval_index_run_id = :run_id
                """
            ),
            {"run_id": run_id},
        ).scalar_one()
        retrieval_run = (
            connection.execute(
                text(
                    """
                SELECT status, is_active, embedding_dimensions
                FROM retrieval_index_runs
                WHERE id = :run_id
                """
                ),
                {"run_id": run_id},
            )
            .mappings()
            .one()
        )

    assert embedding_client.batch_sizes == [2, 2, 1]
    assert passage_asset_count == 5
    assert retrieval_run["status"] == "ready"
    assert retrieval_run["is_active"] is True
    assert retrieval_run["embedding_dimensions"] == VECTOR_DIMENSIONS


def test_search_retrieval_filters_apply_via_document_scope(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    matching_document_id = uuid4()
    other_document_id = uuid4()
    matching_section_id = uuid4()
    other_section_id = uuid4()
    matching_passage_id = uuid4()
    other_passage_id = uuid4()
    matching_table_id = uuid4()
    other_table_id = uuid4()
    matching_ingest_job_id = uuid4()
    other_ingest_job_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        documents = [
            (
                matching_document_id,
                matching_ingest_job_id,
                matching_section_id,
                matching_passage_id,
                matching_table_id,
                2024,
                "match",
                uuid4(),
            ),
            (
                other_document_id,
                other_ingest_job_id,
                other_section_id,
                other_passage_id,
                other_table_id,
                2023,
                "other",
                uuid4(),
            ),
        ]
        for (
            document_id,
            ingest_job_id,
            section_id,
            passage_id,
            table_id,
            publication_year,
            label,
            revision_id,
        ) in documents:
            _insert_revisioned_document(
                connection,
                document_id=document_id,
                revision_id=revision_id,
                revision_number=1,
                title=f"{label.title()} retrieval paper",
                publication_year=publication_year,
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                insert(IngestJob).values(
                    id=ingest_job_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    status="ready",
                    trigger="upload",
                    warnings=[],
                    created_at=now,
                    started_at=now,
                    finished_at=now,
                )
            )
            connection.execute(
                insert(DocumentSection).values(
                    id=section_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    heading="Results",
                    heading_path=["Results"],
                    ordinal=1,
                    page_start=1,
                    page_end=1,
                )
            )

            connection.execute(
                insert(DocumentPassage).values(
                    id=passage_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    chunk_ordinal=1,
                    body_text=f"Shared result token for the {label} document.",
                    contextualized_text=f"Shared result token for the {label} document.",
                    token_count=7,
                    page_start=1,
                    page_end=1,
                    provenance_offsets={"pages": [1], "charspans": [[0, 10]]},
                    artifact_id=None,
                )
            )
            connection.execute(
                insert(DocumentTable).values(
                    id=table_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    caption=f"{label.title()} table",
                    table_type="lexical",
                    headers_json=["A"],
                    rows_json=[["1" if label == "match" else "2"]],
                    page_start=1,
                    page_end=1,
                    artifact_id=None,
                )
            )
            run_id = _insert_run(
                connection,
                document_id=document_id,
                revision_id=revision_id,
                ingest_job_id=ingest_job_id,
                index_version="mvp-v2",
                parser_source="docling",
                is_active=True,
                activated_at=now,
                created_at=now,
            )
            _insert_passage_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                passage_id=passage_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text=f"Shared result token for the {label} document.",
                embedding=_vector_string(4 if label == "match" else 5),
            )
            _insert_table_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                table_id=table_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text=f"{label.title()} table A {1 if label == 'match' else 2}",
                embedding=_vector_string(4 if label == "match" else 5),
            )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        embedding_client=FixedEmbeddingClient({"shared result token": _vector_values(4)}),
        reranker_client=IdentityReranker(),
    )

    filters = service.search_passages(
        query="shared result token",
        filters=RetrievalFilters(
            document_ids=(matching_document_id,),
            publication_years=(2024,),
        ),
    )
    table_filters = service.search_tables(
        query="matching table",
        filters=RetrievalFilters(
            document_ids=(matching_document_id,),
            publication_years=(2024,),
        ),
    )

    assert [result.document_id for result in filters] == [matching_document_id]
    assert [result.document_id for result in table_filters] == [matching_document_id]
    assert table_filters[0].retrieval_modes == ("sparse", "dense")


def test_active_retrieval_run_unique_index_blocks_duplicate_active_rows(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    first_ingest_job_id = uuid4()
    second_ingest_job_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Unique index paper",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=first_ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(IngestJob).values(
                id=second_ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=first_ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )

        with pytest.raises(IntegrityError):
            _insert_run(
                connection,
                document_id=document_id,
                revision_id=revision_id,
                ingest_job_id=second_ingest_job_id,
                index_version="mvp-v2",
                parser_source="docling",
                is_active=True,
                activated_at=now,
                created_at=now,
            )


def test_phase3_retrieval_helpers_support_cursor_table_detail_and_passage_context(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    first_passage_id = uuid4()
    second_passage_id = uuid4()
    table_id = uuid4()
    ingest_job_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Phase 3 retrieval paper",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=["parser_fallback_used"],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Methods",
                heading_path=["Methods"],
                ordinal=1,
                page_start=1,
                page_end=2,
            )
        )
        connection.execute(
            insert(DocumentPassage).values(
                id=first_passage_id,
                document_id=document_id,
                revision_id=revision_id,
                section_id=section_id,
                chunk_ordinal=1,
                body_text="Shared cursor keyword first passage.",
                contextualized_text="Shared cursor keyword first passage.",
                token_count=5,
                page_start=1,
                page_end=1,
                provenance_offsets={"pages": [1], "charspans": [[0, 10]]},
                artifact_id=None,
            )
        )
        connection.execute(
            insert(DocumentPassage).values(
                id=second_passage_id,
                document_id=document_id,
                revision_id=revision_id,
                section_id=section_id,
                chunk_ordinal=2,
                body_text="Shared cursor keyword second passage.",
                contextualized_text="Shared cursor keyword second passage.",
                token_count=5,
                page_start=2,
                page_end=2,
                provenance_offsets={"pages": [2], "charspans": [[0, 10]]},
                artifact_id=None,
            )
        )
        connection.execute(
            insert(DocumentTable).values(
                id=table_id,
                document_id=document_id,
                revision_id=revision_id,
                section_id=section_id,
                caption="Methods table",
                table_type="lexical",
                headers_json=["A", "B"],
                rows_json=[["1", "2"], ["3", "4"]],
                page_start=2,
                page_end=2,
                artifact_id=None,
            )
        )
        run_id = _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        _insert_passage_asset(
            connection,
            run_id=run_id,
            revision_id=revision_id,
            passage_id=first_passage_id,
            document_id=document_id,
            section_id=section_id,
            publication_year=None,
            search_text="Shared cursor keyword first passage.",
            embedding=_vector_string(30),
        )
        _insert_passage_asset(
            connection,
            run_id=run_id,
            revision_id=revision_id,
            passage_id=second_passage_id,
            document_id=document_id,
            section_id=section_id,
            publication_year=None,
            search_text="Shared cursor keyword second passage.",
            embedding=_vector_string(31),
        )
        _insert_table_asset(
            connection,
            run_id=run_id,
            revision_id=revision_id,
            table_id=table_id,
            document_id=document_id,
            section_id=section_id,
            publication_year=None,
            search_text="Methods table A B 1 2 3 4",
            embedding=_vector_string(30),
        )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
        embedding_client=FixedEmbeddingClient({"shared cursor keyword": _vector_values(30)}),
        reranker_client=IdentityReranker(),
    )

    first_page = service.search_passages_page(query="shared cursor keyword", limit=1)

    assert len(first_page.items) == 1
    assert first_page.next_cursor is not None
    assert first_page.index_version == "mvp-v2"

    second_page = service.search_passages_page(
        query="shared cursor keyword",
        limit=1,
        cursor=first_page.next_cursor,
    )

    assert len(second_page.items) == 1
    assert {first_page.items[0].passage_id, second_page.items[0].passage_id} == {
        first_passage_id,
        second_passage_id,
    }

    table = service.get_table(table_id=table_id)

    assert table is not None
    assert table.retrieval_index_run_id == run_id
    assert table.index_version == "mvp-v2"
    assert table.parser_source == "docling"
    assert table.rows == (("1", "2"), ("3", "4"))
    assert table.warnings == ("parser_fallback_used",)

    context = service.get_passage_context(passage_id=first_passage_id, before=0, after=1)

    assert context is not None
    assert context.passage.parser_source == "docling"
    assert context.passage.retrieval_index_run_id == run_id
    assert [passage.relationship for passage in context.context_passages] == [
        "selected",
        "sibling",
    ]
    assert context.warnings == ("parser_fallback_used",)


def test_search_passages_page_reuses_ranked_snapshot_for_later_pages(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    ingest_job_id = uuid4()
    first_passage_id = uuid4()
    second_passage_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Snapshot reuse passages",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Body",
                heading_path=["Body"],
                ordinal=1,
                page_start=1,
                page_end=1,
            )
        )
        run_id = _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        for chunk_ordinal, passage_id in enumerate((first_passage_id, second_passage_id), start=1):
            connection.execute(
                insert(DocumentPassage).values(
                    id=passage_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    chunk_ordinal=chunk_ordinal,
                    body_text=f"Snapshot passage {chunk_ordinal}",
                    contextualized_text=f"Snapshot passage {chunk_ordinal}",
                    token_count=3,
                    page_start=1,
                    page_end=1,
                    provenance_offsets={"pages": [1], "charspans": [[0, 10]]},
                    artifact_id=None,
                )
            )
            _insert_passage_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                passage_id=passage_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text="snapshot passage keyword",
                embedding=_vector_string(12),
            )

    reranker = CountingReranker()
    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
        embedding_client=FixedEmbeddingClient({"snapshot passage keyword": _vector_values(12)}),
        reranker_client=reranker,
    )

    first_page = service.search_passages_page(query="snapshot passage keyword", limit=1)
    second_page = service.search_passages_page(
        query="snapshot passage keyword",
        limit=1,
        cursor=first_page.next_cursor,
    )

    assert len(first_page.items) == 1
    assert len(second_page.items) == 1
    assert {first_page.items[0].passage_id, second_page.items[0].passage_id} == {
        first_passage_id,
        second_passage_id,
    }
    assert len(reranker.calls) == 1


def test_search_tables_page_reuses_ranked_snapshot_for_later_pages(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    ingest_job_id = uuid4()
    first_table_id = uuid4()
    second_table_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Snapshot reuse tables",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Tables",
                heading_path=["Tables"],
                ordinal=1,
                page_start=1,
                page_end=1,
            )
        )
        run_id = _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        for index, table_id in enumerate((first_table_id, second_table_id), start=1):
            connection.execute(
                insert(DocumentTable).values(
                    id=table_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    caption=f"Snapshot table {index}",
                    table_type="lexical",
                    headers_json=["A"],
                    rows_json=[[str(index)]],
                    page_start=1,
                    page_end=1,
                    artifact_id=None,
                )
            )
            _insert_table_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                table_id=table_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text="snapshot table keyword",
                embedding=_vector_string(13),
            )

    reranker = CountingReranker()
    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
        embedding_client=FixedEmbeddingClient({"snapshot table keyword": _vector_values(13)}),
        reranker_client=reranker,
    )

    first_page = service.search_tables_page(query="snapshot table keyword", limit=1)
    second_page = service.search_tables_page(
        query="snapshot table keyword",
        limit=1,
        cursor=first_page.next_cursor,
    )

    assert len(first_page.items) == 1
    assert len(second_page.items) == 1
    assert {first_page.items[0].table_id, second_page.items[0].table_id} == {
        first_table_id,
        second_table_id,
    }
    assert len(reranker.calls) == 1


def test_search_passages_page_exact_mode_reranks_only_needed_prefix_per_page_chain(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    ingest_job_id = uuid4()
    total_passages = 6
    passage_ids = tuple(UUID(int=30_000 + index) for index in range(total_passages))

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Exact pagination prefix",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Body",
                heading_path=["Body"],
                ordinal=1,
                page_start=1,
                page_end=1,
            )
        )
        run_id = _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        for index, passage_id in enumerate(passage_ids, start=1):
            connection.execute(
                insert(DocumentPassage).values(
                    id=passage_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    chunk_ordinal=index,
                    body_text=f"Exact pagination passage {index}",
                    contextualized_text=f"Exact pagination passage {index}",
                    token_count=3,
                    page_start=1,
                    page_end=1,
                    provenance_offsets={"pages": [1], "charspans": [[0, 10]]},
                    artifact_id=None,
                )
            )
            _insert_passage_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                passage_id=passage_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text="exact pagination keyword",
                embedding=_vector_string(16),
            )

    reranker = CountingReranker()
    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
        embedding_client=FixedEmbeddingClient({"exact pagination keyword": _vector_values(16)}),
        reranker_client=reranker,
    )

    first_page = service.search_passages_page(query="exact pagination keyword", limit=2)
    second_page = service.search_passages_page(
        query="exact pagination keyword",
        limit=2,
        cursor=first_page.next_cursor,
    )

    assert [item.passage_id for item in first_page.items] == list(passage_ids[:2])
    assert [item.passage_id for item in second_page.items] == list(passage_ids[2:4])
    assert first_page.next_cursor is not None
    assert second_page.next_cursor is not None
    assert [len(call) for call in reranker.calls] == [3, 5]


def test_search_passages_page_reaches_results_beyond_sparse_candidate_cap(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    ingest_job_id = uuid4()
    total_passages = retrieval_service_module.PASSAGE_SPARSE_CANDIDATES + 15
    expected_passage_ids = tuple(UUID(int=index + 1) for index in range(total_passages))

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Deep pagination passages",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Body",
                heading_path=["Body"],
                ordinal=1,
                page_start=1,
                page_end=1,
            )
        )
        run_id = _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        for index, passage_id in enumerate(expected_passage_ids, start=1):
            connection.execute(
                insert(DocumentPassage).values(
                    id=passage_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    chunk_ordinal=index,
                    body_text=f"Pagination passage {index}",
                    contextualized_text=f"Pagination passage {index}",
                    token_count=3,
                    page_start=1,
                    page_end=1,
                    provenance_offsets={"pages": [1], "charspans": [[0, 10]]},
                    artifact_id=None,
                )
            )
            _insert_passage_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                passage_id=passage_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text="deep pagination keyword",
                embedding=_vector_string(0),
            )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
    )

    seen_passage_ids: list[UUID] = []
    cursor: str | None = None
    while True:
        page = service.search_passages_page(
            query="deep pagination keyword",
            limit=7,
            cursor=cursor,
        )
        seen_passage_ids.extend(item.passage_id for item in page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor

    assert tuple(seen_passage_ids) == expected_passage_ids
    assert len(seen_passage_ids) == total_passages


def test_search_passages_page_bounded_mode_exposes_truncation_metadata(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    ingest_job_id = uuid4()
    total_passages = retrieval_service_module.PASSAGE_SPARSE_CANDIDATES + 10

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Bounded pagination passages",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Body",
                heading_path=["Body"],
                ordinal=1,
                page_start=1,
                page_end=1,
            )
        )
        run_id = _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        for index in range(total_passages):
            passage_id = UUID(int=20_000 + index)
            connection.execute(
                insert(DocumentPassage).values(
                    id=passage_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    chunk_ordinal=index + 1,
                    body_text=f"Bounded passage {index + 1}",
                    contextualized_text=f"Bounded passage {index + 1}",
                    token_count=3,
                    page_start=1,
                    page_end=1,
                    provenance_offsets={"pages": [1], "charspans": [[0, 10]]},
                    artifact_id=None,
                )
            )
            _insert_passage_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                passage_id=passage_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text="bounded pagination keyword",
                embedding=_vector_string(8),
            )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
        embedding_client=FixedEmbeddingClient({"bounded pagination keyword": _vector_values(8)}),
        reranker_client=IdentityReranker(),
    )

    page = service.search_passages_page(
        query="bounded pagination keyword",
        limit=2,
        pagination_mode="bounded",
        max_rerank_candidates=3,
        max_expansion_rounds=1,
    )

    assert len(page.items) == 2
    assert page.exact is False
    assert page.truncated is True
    assert page.warnings == ("bounded_pagination_truncated",)
    assert page.next_cursor is not None


def test_search_tables_page_bounded_mode_exposes_truncation_metadata(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    ingest_job_id = uuid4()
    total_tables = retrieval_service_module.TABLE_SPARSE_CANDIDATES + 10

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Bounded pagination tables",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Tables",
                heading_path=["Tables"],
                ordinal=1,
                page_start=1,
                page_end=1,
            )
        )
        run_id = _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        for index in range(total_tables):
            table_id = UUID(int=30_000 + index)
            connection.execute(
                insert(DocumentTable).values(
                    id=table_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    caption=f"Bounded table {index + 1}",
                    table_type="lexical",
                    headers_json=["A"],
                    rows_json=[[str(index + 1)]],
                    page_start=1,
                    page_end=1,
                    artifact_id=None,
                )
            )
            _insert_table_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                table_id=table_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text="bounded table pagination keyword",
            )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
    )

    page = service.search_tables_page(
        query="bounded table pagination keyword",
        limit=2,
        pagination_mode="bounded",
        max_rerank_candidates=3,
        max_expansion_rounds=1,
    )

    assert len(page.items) == 2
    assert page.exact is False
    assert page.truncated is True
    assert page.warnings == ("bounded_pagination_truncated",)
    assert page.next_cursor is not None


def test_search_passages_page_rejects_legacy_cursor_integration(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    ingest_job_id = uuid4()
    passage_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Legacy cursor passages",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Body",
                heading_path=["Body"],
                ordinal=1,
                page_start=1,
                page_end=1,
            )
        )
        connection.execute(
            insert(DocumentPassage).values(
                id=passage_id,
                document_id=document_id,
                revision_id=revision_id,
                section_id=section_id,
                chunk_ordinal=1,
                body_text="Legacy cursor passage",
                contextualized_text="Legacy cursor passage",
                token_count=3,
                page_start=1,
                page_end=1,
                provenance_offsets={"pages": [1], "charspans": [[0, 10]]},
                artifact_id=None,
            )
        )
        run_id = _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        _insert_passage_asset(
            connection,
            run_id=run_id,
            revision_id=revision_id,
            passage_id=passage_id,
            document_id=document_id,
            section_id=section_id,
            publication_year=None,
            search_text="legacy cursor keyword",
            embedding=_vector_string(14),
        )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
    )
    fingerprint = fingerprint_payload(
        {
            "kind": "passages",
            "query": "legacy cursor keyword",
            "pagination_mode": "exact",
            "max_rerank_candidates": None,
            "max_expansion_rounds": None,
            "filters": {"document_ids": [], "publication_years": []},
        }
    )
    legacy_cursor = encode_cursor(
        {
            "kind": "passages",
            "fingerprint": fingerprint,
            "index_version": "mvp-v2",
            "score": "3.0",
            "entity_id": str(passage_id),
        }
    )

    with pytest.raises(RetrievalError, match="cursor is no longer supported"):
        service.search_passages_page(
            query="legacy cursor keyword",
            limit=1,
            cursor=legacy_cursor,
        )


def test_search_tables_page_reaches_results_beyond_sparse_candidate_cap(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    ingest_job_id = uuid4()
    total_tables = retrieval_service_module.TABLE_SPARSE_CANDIDATES + 7
    expected_table_ids = tuple(UUID(int=10_000 + index) for index in range(total_tables))

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Deep pagination tables",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Tables",
                heading_path=["Tables"],
                ordinal=1,
                page_start=1,
                page_end=1,
            )
        )
        run_id = _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        for index, table_id in enumerate(expected_table_ids, start=1):
            connection.execute(
                insert(DocumentTable).values(
                    id=table_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    caption=f"Pagination table {index}",
                    table_type="lexical",
                    headers_json=["A"],
                    rows_json=[[str(index)]],
                    page_start=1,
                    page_end=1,
                    artifact_id=None,
                )
            )
            _insert_table_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                table_id=table_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text="deep table pagination keyword",
            )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
    )

    seen_table_ids: list[UUID] = []
    cursor: str | None = None
    while True:
        page = service.search_tables_page(
            query="deep table pagination keyword",
            limit=4,
            cursor=cursor,
        )
        seen_table_ids.extend(item.table_id for item in page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor

    assert tuple(seen_table_ids) == expected_table_ids
    assert len(seen_table_ids) == total_tables


def test_search_passages_page_reaches_results_beyond_dense_candidate_cap(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    ingest_job_id = uuid4()
    total_passages = retrieval_service_module.PASSAGE_DENSE_CANDIDATES + 7
    expected_passage_ids = tuple(UUID(int=40_000 + index) for index in range(total_passages))

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Dense-only pagination passages",
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Body",
                heading_path=["Body"],
                ordinal=1,
                page_start=1,
                page_end=1,
            )
        )
        run_id = _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        for index, passage_id in enumerate(expected_passage_ids, start=1):
            connection.execute(
                insert(DocumentPassage).values(
                    id=passage_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    section_id=section_id,
                    chunk_ordinal=index,
                    body_text=f"Dense page passage {index}",
                    contextualized_text=f"Dense page passage {index}",
                    token_count=4,
                    page_start=1,
                    page_end=1,
                    provenance_offsets={"pages": [1], "charspans": [[0, 10]]},
                    artifact_id=None,
                )
            )
            _insert_passage_asset(
                connection,
                run_id=run_id,
                revision_id=revision_id,
                passage_id=passage_id,
                document_id=document_id,
                section_id=section_id,
                publication_year=None,
                search_text=f"latent vector candidate {index}",
                embedding=_vector_string(21),
            )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
        embedding_client=FixedEmbeddingClient(
            {"semantic-only pagination query": _vector_values(21)}
        ),
        reranker_client=IdentityReranker(),
    )

    seen_passage_ids: list[UUID] = []
    cursor: str | None = None
    while True:
        page = service.search_passages_page(
            query="semantic-only pagination query",
            limit=4,
            cursor=cursor,
        )
        seen_passage_ids.extend(item.passage_id for item in page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor

    assert tuple(seen_passage_ids) == expected_passage_ids
    assert len(seen_passage_ids) == total_passages


def test_detail_helpers_require_active_retrieval_provenance(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    passage_id = uuid4()
    table_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Provenance paper",
            created_at=now,
            updated_at=now,
        )
        _insert_revisioned_section(
            connection,
            section_id=section_id,
            document_id=document_id,
            revision_id=revision_id,
            heading="Methods",
            heading_path=["Methods"],
        )
        _insert_revisioned_passage(
            connection,
            passage_id=passage_id,
            document_id=document_id,
            revision_id=revision_id,
            section_id=section_id,
            chunk_ordinal=1,
            body_text="Shared provenance keyword.",
            contextualized_text="Shared provenance keyword.",
            token_count=3,
            page_start=1,
            page_end=1,
            provenance_offsets={"pages": [1], "charspans": [[0, 26]]},
        )
        _insert_revisioned_table(
            connection,
            table_id=table_id,
            document_id=document_id,
            revision_id=revision_id,
            section_id=section_id,
            caption="Provenance table",
            headers_json=["A"],
            rows_json=[["1"]],
            page_start=1,
            page_end=1,
        )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
    )

    assert service.get_table(table_id=table_id) is None
    assert service.get_passage_context(passage_id=passage_id) is None


def test_default_read_paths_handle_large_sections_and_table_payloads(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    selected_passage_id = uuid4()
    sibling_passage_id = uuid4()
    table_id = uuid4()
    ingest_job_id = uuid4()

    selected_text = "Selected large payload section. " + ("alpha " * 12_000)
    sibling_text = "Sibling large payload section. " + ("beta " * 10_000)
    headers_json: list[object] = [f"Header {index} " + ("H" * 128) for index in range(4)]
    rows_json: list[list[object]] = [
        [f"row-{row}-col-{column}-" + ("X" * 256) for column in range(4)] for row in range(40)
    ]

    with migrated_postgres_engine.begin() as connection:
        _insert_revisioned_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            revision_number=1,
            title="Large payload paper",
            created_at=now,
            updated_at=now,
        )
        _insert_revisioned_job(
            connection,
            ingest_job_id=ingest_job_id,
            document_id=document_id,
            revision_id=revision_id,
            trigger="upload",
            warnings=["parser_fallback_used"],
            created_at=now,
        )
        _insert_revisioned_section(
            connection,
            section_id=section_id,
            document_id=document_id,
            revision_id=revision_id,
            heading="Results",
            heading_path=["Results", "Large payload section"],
            page_start=10,
            page_end=12,
        )
        _insert_revisioned_passage(
            connection,
            passage_id=selected_passage_id,
            document_id=document_id,
            revision_id=revision_id,
            section_id=section_id,
            chunk_ordinal=1,
            body_text=selected_text,
            contextualized_text=selected_text,
            token_count=8_000,
            page_start=10,
            page_end=11,
            provenance_offsets={"pages": [10, 11], "charspans": [[0, len(selected_text)]]},
        )
        _insert_revisioned_passage(
            connection,
            passage_id=sibling_passage_id,
            document_id=document_id,
            revision_id=revision_id,
            section_id=section_id,
            chunk_ordinal=2,
            body_text=sibling_text,
            contextualized_text=sibling_text,
            token_count=7_000,
            page_start=11,
            page_end=12,
            provenance_offsets={"pages": [11, 12], "charspans": [[0, len(sibling_text)]]},
        )
        _insert_revisioned_table(
            connection,
            table_id=table_id,
            document_id=document_id,
            revision_id=revision_id,
            section_id=section_id,
            caption="Large payload table",
            headers_json=headers_json,
            rows_json=rows_json,
            page_start=12,
            page_end=12,
        )
        run_id = _insert_run(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        _insert_passage_asset(
            connection,
            run_id=run_id,
            revision_id=revision_id,
            passage_id=selected_passage_id,
            document_id=document_id,
            section_id=section_id,
            publication_year=None,
            search_text="large payload keyword selected section",
            embedding=_vector_string(0),
        )
        _insert_table_asset(
            connection,
            run_id=run_id,
            revision_id=revision_id,
            table_id=table_id,
            document_id=document_id,
            section_id=section_id,
            publication_year=None,
            search_text="large payload keyword table",
            embedding=_vector_string(0),
        )

    service = RetrievalService(
        connection_factory=lambda: connection_scope(migrated_postgres_engine),
        active_index_version="mvp-v2",
    )

    table = service.get_table(table_id=table_id)
    context = service.get_passage_context(
        passage_id=selected_passage_id,
        before=0,
        after=1,
    )
    pack = service.build_context_pack(query="large payload keyword", limit=8)

    assert table is not None
    assert table.headers == tuple(headers_json)
    assert table.row_count == len(rows_json)
    assert len(table.rows) == len(rows_json)
    assert table.rows[0] == tuple(rows_json[0])
    assert table.rows[-1] == tuple(rows_json[-1])

    assert context is not None
    assert context.passage.text == selected_text
    assert [item.text for item in context.context_passages] == [selected_text, sibling_text]
    assert context.warnings == ("parser_fallback_used",)

    assert [item.text for item in pack.passages] == [selected_text]
    assert pack.tables[0].preview.row_count == len(rows_json)
    assert pack.tables[0].preview.rows == tuple(tuple(row) for row in rows_json[:3])
    assert [item.text for item in pack.parent_sections[0].supporting_passages] == [
        selected_text,
        sibling_text,
    ]
    assert pack.provenance.active_index_version == "mvp-v2"
