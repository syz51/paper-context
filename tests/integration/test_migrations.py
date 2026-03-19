from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

pytestmark = [
    pytest.mark.integration,
    pytest.mark.migration,
    pytest.mark.requires_postgres,
    pytest.mark.slow,
]

EXPECTED_SCHEMA_TABLE_COLUMNS = {
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
        "active_revision_id",
        "created_at",
        "updated_at",
    },
    "document_revisions": {
        "id",
        "document_id",
        "revision_number",
        "status",
        "title",
        "authors",
        "abstract",
        "publication_year",
        "source_type",
        "metadata_confidence",
        "quant_tags",
        "source_artifact_id",
        "ingest_job_id",
        "activated_at",
        "superseded_at",
        "created_at",
        "updated_at",
    },
    "document_artifacts": {
        "id",
        "document_id",
        "revision_id",
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
        "revision_id",
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
        "revision_id",
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
        "revision_id",
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
        "revision_id",
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
        "revision_id",
        "source_artifact_id",
        "status",
        "failure_code",
        "failure_message",
        "warnings",
        "stage_timings",
        "started_at",
        "finished_at",
        "trigger",
        "created_at",
    },
    "retrieval_index_runs": {
        "id",
        "document_id",
        "revision_id",
        "ingest_job_id",
        "index_version",
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "reranker_provider",
        "reranker_model",
        "chunking_version",
        "parser_source",
        "is_active",
        "activated_at",
        "deactivated_at",
        "status",
        "created_at",
    },
    "retrieval_passage_assets": {
        "id",
        "retrieval_index_run_id",
        "document_id",
        "revision_id",
        "passage_id",
        "section_id",
        "publication_year",
        "search_text",
        "search_tsvector",
        "embedding",
        "created_at",
    },
    "retrieval_table_assets": {
        "id",
        "retrieval_index_run_id",
        "document_id",
        "revision_id",
        "table_id",
        "section_id",
        "publication_year",
        "search_text",
        "search_tsvector",
        "created_at",
    },
}

EXPECTED_SCHEMA_INDEX_COLUMNS = {
    "ix_documents_current_status": ["current_status"],
    "ix_documents_active_revision_id": ["active_revision_id"],
    "ix_documents_updated_at_id": ["updated_at", "id"],
    "ix_document_revisions_document_id": ["document_id"],
    "ix_document_revisions_document_status": ["document_id", "status"],
    "ix_document_artifacts_document_id": ["document_id"],
    "ix_document_artifacts_revision_id": ["revision_id"],
    "ix_document_artifacts_ingest_job_id": ["ingest_job_id"],
    "ix_document_artifacts_revision_ingest_job_id": ["revision_id", "ingest_job_id"],
    "ix_document_sections_document_id": ["document_id"],
    "ix_document_sections_revision_id": ["revision_id"],
    "ix_document_sections_revision_artifact_id": ["revision_id", "artifact_id"],
    "ix_document_passages_document_id": ["document_id"],
    "ix_document_passages_revision_id": ["revision_id"],
    "ix_document_passages_section_id": ["section_id"],
    "ix_document_passages_revision_section_id": ["revision_id", "section_id"],
    "ix_document_tables_document_id": ["document_id"],
    "ix_document_tables_revision_id": ["revision_id"],
    "ix_document_tables_section_id": ["section_id"],
    "ix_document_tables_revision_section_id": ["revision_id", "section_id"],
    "ix_document_references_document_id": ["document_id"],
    "ix_document_references_revision_id": ["revision_id"],
    "ix_document_references_revision_artifact_id": ["revision_id", "artifact_id"],
    "ix_ingest_jobs_document_id": ["document_id"],
    "ix_ingest_jobs_document_created_at_id": ["document_id", "created_at", "id"],
    "ix_ingest_jobs_revision_id": ["revision_id"],
    "ix_ingest_jobs_revision_created_at_id": ["revision_id", "created_at", "id"],
    "ix_ingest_jobs_status": ["status"],
    "ix_retrieval_index_runs_document_id": ["document_id"],
    "ix_retrieval_index_runs_document_version": ["document_id", "index_version"],
    "ix_retrieval_index_runs_document_active_state": ["document_id", "is_active"],
    "ix_retrieval_index_runs_revision_id": ["revision_id"],
    "ix_retrieval_index_runs_revision_active_state": ["revision_id", "is_active"],
    "ix_retrieval_index_runs_revision_version": ["revision_id", "index_version"],
    "ix_retrieval_index_runs_one_active_per_revision": ["revision_id"],
    "ix_retrieval_passage_assets_retrieval_index_run_id": ["retrieval_index_run_id"],
    "ix_retrieval_passage_assets_document_run": ["document_id", "retrieval_index_run_id"],
    "ix_retrieval_passage_assets_revision_run": ["revision_id", "retrieval_index_run_id"],
    "ix_retrieval_passage_assets_publication_year_run": [
        "publication_year",
        "retrieval_index_run_id",
    ],
    "ix_retrieval_passage_assets_revision_publication_year_run": [
        "revision_id",
        "publication_year",
        "retrieval_index_run_id",
    ],
    "ix_retrieval_passage_assets_search_tsvector": ["search_tsvector"],
    "ix_retrieval_passage_assets_embedding": ["embedding"],
    "ix_retrieval_table_assets_retrieval_index_run_id": ["retrieval_index_run_id"],
    "ix_retrieval_table_assets_document_run": ["document_id", "retrieval_index_run_id"],
    "ix_retrieval_table_assets_revision_run": ["revision_id", "retrieval_index_run_id"],
    "ix_retrieval_table_assets_publication_year_run": [
        "publication_year",
        "retrieval_index_run_id",
    ],
    "ix_retrieval_table_assets_revision_publication_year_run": [
        "revision_id",
        "publication_year",
        "retrieval_index_run_id",
    ],
    "ix_retrieval_table_assets_search_tsvector": ["search_tsvector"],
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
            for table_name in EXPECTED_SCHEMA_TABLE_COLUMNS:
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
            unique_active_index = (
                connection.execute(
                    text(
                        """
                    SELECT
                        idx.relname AS index_name,
                        i.indisunique AS is_unique,
                        pg_get_expr(i.indpred, i.indrelid) AS predicate
                    FROM pg_class idx
                    JOIN pg_index i ON i.indexrelid = idx.oid
                    WHERE idx.relname = 'ix_retrieval_index_runs_one_active_per_revision'
                    """
                    )
                )
                .mappings()
                .one()
            )
            extensions = {
                row[0] for row in connection.execute(text("SELECT extname FROM pg_extension"))
            }
            queue_name = connection.execute(
                text("SELECT queue_name FROM pgmq.metrics(:queue_name)"),
                {"queue_name": "document_ingest"},
            ).scalar_one()

        assert set(EXPECTED_SCHEMA_TABLE_COLUMNS) <= tables
        for table_name, expected_columns in EXPECTED_SCHEMA_TABLE_COLUMNS.items():
            assert table_columns[table_name] == expected_columns
        assert index_columns == EXPECTED_SCHEMA_INDEX_COLUMNS
        assert unique_active_index["is_unique"] is True
        assert unique_active_index["predicate"] == "(is_active = true)"
        assert {"pgmq", "vector"} <= extensions
        assert queue_name == "document_ingest"
    finally:
        engine.dispose()
