from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

pytestmark = [
    pytest.mark.integration,
    pytest.mark.migration,
    pytest.mark.requires_postgres,
    pytest.mark.slow,
]

EXPECTED_PHASE0_TABLE_COLUMNS = {
    "documents": {
        "id",
        "title",
        "authors",
        "abstract",
        "publication_year",
        "source_type",
        "metadata_confidence",
        "quant_tags",
        "current_status",
        "created_at",
        "updated_at",
    },
    "document_artifacts": {
        "id",
        "document_id",
        "ingest_job_id",
        "artifact_type",
        "parser",
        "storage_ref",
        "checksum",
        "is_primary",
        "created_at",
    },
    "document_sections": {
        "id",
        "document_id",
        "parent_section_id",
        "heading",
        "heading_path",
        "ordinal",
        "page_start",
        "page_end",
        "artifact_id",
    },
    "document_passages": {
        "id",
        "document_id",
        "section_id",
        "chunk_ordinal",
        "body_text",
        "contextualized_text",
        "token_count",
        "page_start",
        "page_end",
        "provenance_offsets",
        "quant_tags",
        "artifact_id",
    },
    "document_tables": {
        "id",
        "document_id",
        "section_id",
        "caption",
        "table_type",
        "headers_json",
        "rows_json",
        "page_start",
        "page_end",
        "quant_tags",
        "artifact_id",
    },
    "document_references": {
        "id",
        "document_id",
        "raw_citation",
        "normalized_title",
        "authors",
        "publication_year",
        "doi",
        "source_confidence",
        "artifact_id",
    },
    "ingest_jobs": {
        "id",
        "document_id",
        "source_artifact_id",
        "status",
        "failure_code",
        "failure_message",
        "warnings",
        "started_at",
        "finished_at",
        "trigger",
        "created_at",
    },
    "retrieval_index_runs": {
        "id",
        "document_id",
        "ingest_job_id",
        "index_version",
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "reranker_provider",
        "reranker_model",
        "chunking_version",
        "parser_source",
        "status",
        "created_at",
    },
}

EXPECTED_PHASE0_INDEX_COLUMNS = {
    "ix_documents_current_status": ["current_status"],
    "ix_document_artifacts_document_id": ["document_id"],
    "ix_document_artifacts_ingest_job_id": ["ingest_job_id"],
    "ix_document_sections_document_id": ["document_id"],
    "ix_document_passages_document_id": ["document_id"],
    "ix_document_passages_section_id": ["section_id"],
    "ix_document_tables_document_id": ["document_id"],
    "ix_document_tables_section_id": ["section_id"],
    "ix_document_references_document_id": ["document_id"],
    "ix_ingest_jobs_document_id": ["document_id"],
    "ix_ingest_jobs_status": ["status"],
    "ix_retrieval_index_runs_document_id": ["document_id"],
    "ix_retrieval_index_runs_document_version": ["document_id", "index_version"],
}


def test_alembic_upgrade_succeeds_on_fresh_database(
    postgres_test_database_url: str,
    run_alembic_upgrade,
) -> None:
    run_alembic_upgrade(postgres_test_database_url)

    engine = create_engine(postgres_test_database_url, future=True, pool_pre_ping=True)
    try:
        with engine.begin() as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                )
            }
            table_columns: dict[str, set[str]] = {}
            for table_name in EXPECTED_PHASE0_TABLE_COLUMNS:
                table_columns[table_name] = {
                    row[0]
                    for row in connection.execute(
                        text(
                            """
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_schema = 'public' AND table_name = :table_name
                            """
                        ),
                        {"table_name": table_name},
                    )
                }
            index_columns = {
                row.index_name: list(row.columns)
                for row in connection.execute(
                    text(
                        """
                        SELECT
                            idx.relname AS index_name,
                            array_agg(att.attname ORDER BY keys.ord) AS columns
                        FROM pg_class idx
                        JOIN pg_index i ON i.indexrelid = idx.oid
                        JOIN pg_class tbl ON tbl.oid = i.indrelid
                        JOIN pg_namespace ns ON ns.oid = tbl.relnamespace
                        JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS keys(attnum, ord) ON true
                        JOIN pg_attribute att
                            ON att.attrelid = tbl.oid
                            AND att.attnum = keys.attnum
                        WHERE ns.nspname = 'public'
                          AND idx.relname LIKE 'ix_%'
                        GROUP BY idx.relname
                        """
                    )
                ).mappings()
            }
            extensions = {
                row[0] for row in connection.execute(text("SELECT extname FROM pg_extension"))
            }
            queue_name = connection.execute(
                text("SELECT queue_name FROM pgmq.metrics(:queue_name)"),
                {"queue_name": "document_ingest"},
            ).scalar_one()

        assert set(EXPECTED_PHASE0_TABLE_COLUMNS) <= tables
        for table_name, expected_columns in EXPECTED_PHASE0_TABLE_COLUMNS.items():
            assert table_columns[table_name] == expected_columns
        assert index_columns == EXPECTED_PHASE0_INDEX_COLUMNS
        assert {"pgmq", "vector"} <= extensions
        assert queue_name == "document_ingest"
    finally:
        engine.dispose()
