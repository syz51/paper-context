from __future__ import annotations

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
from paper_context.retrieval import RetrievalService
from paper_context.retrieval.clients import EmbeddingBatch
from paper_context.retrieval.types import RerankItem, RetrievalFilters

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


def _insert_run(
    connection,
    *,
    document_id: UUID,
    ingest_job_id: UUID,
    index_version: str,
    parser_source: str,
    is_active: bool,
    activated_at: datetime,
    created_at: datetime,
) -> UUID:
    run_id = uuid4()
    connection.execute(
        insert(RetrievalIndexRun).values(
            id=run_id,
            document_id=document_id,
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
    table_id: UUID,
    document_id: UUID,
    section_id: UUID,
    publication_year: int | None,
    search_text: str,
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
                search_tsvector
            )
            VALUES (
                :id,
                :retrieval_index_run_id,
                :table_id,
                :document_id,
                :section_id,
                :publication_year,
                :search_text,
                to_tsvector('english', :search_text)
            )
            """
        ),
        {
            "id": uuid4(),
            "retrieval_index_run_id": run_id,
            "table_id": table_id,
            "document_id": document_id,
            "section_id": section_id,
            "publication_year": publication_year,
            "search_text": search_text,
        },
    )


def test_search_passages_and_tables_return_only_active_index_version_rows(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    active_index_version = "mvp-v2"
    legacy_index_version = "mvp-v1"

    document_id = uuid4()
    section_id = uuid4()
    passage_id = uuid4()
    table_id = uuid4()
    legacy_ingest_job_id = uuid4()
    active_ingest_job_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            insert(Document).values(
                id=document_id,
                title="Retrieval paper",
                source_type="upload",
                current_status="ready",
                created_at=now - timedelta(minutes=2),
                updated_at=now,
            )
        )
        connection.execute(
            insert(IngestJob).values(
                id=legacy_ingest_job_id,
                document_id=document_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now - timedelta(minutes=2),
                started_at=now - timedelta(minutes=2),
                finished_at=now - timedelta(minutes=2),
            )
        )
        connection.execute(
            insert(IngestJob).values(
                id=active_ingest_job_id,
                document_id=document_id,
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
        legacy_run_id = _insert_run(
            connection,
            document_id=document_id,
            ingest_job_id=legacy_ingest_job_id,
            index_version=legacy_index_version,
            parser_source="docling",
            is_active=False,
            activated_at=now - timedelta(minutes=2),
            created_at=now - timedelta(minutes=2),
        )
        active_run_id = _insert_run(
            connection,
            document_id=document_id,
            ingest_job_id=active_ingest_job_id,
            index_version=active_index_version,
            parser_source="docling",
            is_active=True,
            activated_at=now - timedelta(minutes=1),
            created_at=now - timedelta(minutes=1),
        )
        _insert_passage_asset(
            connection,
            run_id=legacy_run_id,
            passage_id=passage_id,
            document_id=document_id,
            section_id=section_id,
            publication_year=None,
            search_text="Shared methods alpha signal is present here.",
            embedding=_vector_string(10),
        )
        _insert_passage_asset(
            connection,
            run_id=active_run_id,
            passage_id=passage_id,
            document_id=document_id,
            section_id=section_id,
            publication_year=None,
            search_text="Shared methods alpha signal is present here.",
            embedding=_vector_string(11),
        )
        _insert_table_asset(
            connection,
            run_id=legacy_run_id,
            table_id=table_id,
            document_id=document_id,
            section_id=section_id,
            publication_year=None,
            search_text="Shared methods table A B 1 2 3 4",
        )
        _insert_table_asset(
            connection,
            run_id=active_run_id,
            table_id=table_id,
            document_id=document_id,
            section_id=section_id,
            publication_year=None,
            search_text="Shared methods table A B 1 2 3 4",
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
    assert tables[0].retrieval_modes == ("sparse",)


def test_search_passages_dense_path_uses_embeddings(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    section_id = uuid4()
    passage_id = uuid4()
    ingest_job_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            insert(Document).values(
                id=document_id,
                title="Dense retrieval paper",
                source_type="upload",
                current_status="ready",
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
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
    section_id = uuid4()
    ingest_job_id = uuid4()
    passage_ids = [uuid4(), uuid4(), uuid4()]

    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            insert(Document).values(
                id=document_id,
                title="Context pack paper",
                source_type="upload",
                current_status="ready",
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
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
        for document_id, ingest_job_id, section_id, publication_year, label in [
            (matching_document_id, matching_ingest_job_id, matching_section_id, 2024, "match"),
            (other_document_id, other_ingest_job_id, other_section_id, 2023, "other"),
        ]:
            connection.execute(
                insert(Document).values(
                    id=document_id,
                    title=f"{label.title()} retrieval paper",
                    source_type="upload",
                    publication_year=publication_year,
                    current_status="ready",
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.execute(
                insert(IngestJob).values(
                    id=ingest_job_id,
                    document_id=document_id,
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
                    heading="Results",
                    heading_path=["Results"],
                    ordinal=1,
                    page_start=1,
                    page_end=1,
                )
            )

        connection.execute(
            insert(DocumentPassage).values(
                id=matching_passage_id,
                document_id=matching_document_id,
                section_id=matching_section_id,
                chunk_ordinal=1,
                body_text="Shared result token for the matching document.",
                contextualized_text="Shared result token for the matching document.",
                token_count=7,
                page_start=1,
                page_end=1,
                provenance_offsets={"pages": [1], "charspans": [[0, 10]]},
                artifact_id=None,
            )
        )
        connection.execute(
            insert(DocumentPassage).values(
                id=other_passage_id,
                document_id=other_document_id,
                section_id=other_section_id,
                chunk_ordinal=1,
                body_text="Shared result token for the other document.",
                contextualized_text="Shared result token for the other document.",
                token_count=7,
                page_start=1,
                page_end=1,
                provenance_offsets={"pages": [1], "charspans": [[0, 10]]},
                artifact_id=None,
            )
        )
        connection.execute(
            insert(DocumentTable).values(
                id=matching_table_id,
                document_id=matching_document_id,
                section_id=matching_section_id,
                caption="Matching table",
                table_type="lexical",
                headers_json=["A"],
                rows_json=[["1"]],
                page_start=1,
                page_end=1,
                artifact_id=None,
            )
        )
        connection.execute(
            insert(DocumentTable).values(
                id=other_table_id,
                document_id=other_document_id,
                section_id=other_section_id,
                caption="Other table",
                table_type="lexical",
                headers_json=["A"],
                rows_json=[["2"]],
                page_start=1,
                page_end=1,
                artifact_id=None,
            )
        )

        matching_run_id = _insert_run(
            connection,
            document_id=matching_document_id,
            ingest_job_id=matching_ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        other_run_id = _insert_run(
            connection,
            document_id=other_document_id,
            ingest_job_id=other_ingest_job_id,
            index_version="mvp-v2",
            parser_source="docling",
            is_active=True,
            activated_at=now,
            created_at=now,
        )
        _insert_passage_asset(
            connection,
            run_id=matching_run_id,
            passage_id=matching_passage_id,
            document_id=matching_document_id,
            section_id=matching_section_id,
            publication_year=None,
            search_text="Shared result token for the matching document.",
            embedding=_vector_string(4),
        )
        _insert_passage_asset(
            connection,
            run_id=other_run_id,
            passage_id=other_passage_id,
            document_id=other_document_id,
            section_id=other_section_id,
            publication_year=None,
            search_text="Shared result token for the other document.",
            embedding=_vector_string(5),
        )
        _insert_table_asset(
            connection,
            run_id=matching_run_id,
            table_id=matching_table_id,
            document_id=matching_document_id,
            section_id=matching_section_id,
            publication_year=None,
            search_text="Matching table A 1",
        )
        _insert_table_asset(
            connection,
            run_id=other_run_id,
            table_id=other_table_id,
            document_id=other_document_id,
            section_id=other_section_id,
            publication_year=None,
            search_text="Other table A 2",
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


def test_active_retrieval_run_unique_index_blocks_duplicate_active_rows(
    migrated_postgres_engine,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    first_ingest_job_id = uuid4()
    second_ingest_job_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            insert(Document).values(
                id=document_id,
                title="Unique index paper",
                source_type="upload",
                current_status="ready",
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            insert(IngestJob).values(
                id=first_ingest_job_id,
                document_id=document_id,
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
                ingest_job_id=second_ingest_job_id,
                index_version="mvp-v2",
                parser_source="docling",
                is_active=True,
                activated_at=now,
                created_at=now,
            )
