from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

pytestmark = [
    pytest.mark.integration,
    pytest.mark.migration,
    pytest.mark.requires_postgres,
    pytest.mark.slow,
]


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
            extensions = {
                row[0] for row in connection.execute(text("SELECT extname FROM pg_extension"))
            }
            queue_name = connection.execute(
                text("SELECT queue_name FROM pgmq.metrics(:queue_name)"),
                {"queue_name": "document_ingest"},
            ).scalar_one()

        assert {"documents", "ingest_jobs", "retrieval_index_runs"} <= tables
        assert {"pgmq", "vector"} <= extensions
        assert queue_name == "document_ingest"
    finally:
        engine.dispose()
