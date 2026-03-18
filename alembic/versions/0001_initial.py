"""Initial migration for Paper Context MVP schema and PGMQ queue."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgmq")

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("title", sa.String(1024), nullable=True),
        sa.Column("authors", postgresql.JSONB, nullable=True),
        sa.Column("abstract", sa.Text, nullable=True),
        sa.Column("publication_year", sa.Integer, nullable=True),
        sa.Column("source_type", sa.String(128), nullable=True),
        sa.Column("metadata_confidence", sa.Float, nullable=True),
        sa.Column("quant_tags", postgresql.JSONB, nullable=True),
        sa.Column(
            "current_status", sa.String(32), nullable=False, server_default=sa.text("'queued'")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_documents_current_status", "documents", ["current_status"])

    op.create_table(
        "ingest_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_artifact_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("status", sa.String(32), nullable=False, server_default=sa.text("'queued'")),
        sa.Column("failure_code", sa.String(128), nullable=True),
        sa.Column("failure_message", sa.Text, nullable=True),
        sa.Column("warnings", postgresql.JSONB, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("trigger", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_ingest_jobs_document_id", "ingest_jobs", ["document_id"])
    op.create_index("ix_ingest_jobs_status", "ingest_jobs", ["status"])

    op.create_table(
        "document_artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "ingest_job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("artifact_type", sa.String(64), nullable=False),
        sa.Column("parser", sa.String(64), nullable=False),
        sa.Column("storage_ref", sa.String(2048), nullable=False),
        sa.Column("checksum", sa.String(128), nullable=True),
        sa.Column("is_primary", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_document_artifacts_document_id", "document_artifacts", ["document_id"])
    op.create_index("ix_document_artifacts_ingest_job_id", "document_artifacts", ["ingest_job_id"])
    op.create_foreign_key(
        "fk_ingest_jobs_source_artifact_id_document_artifacts",
        "ingest_jobs",
        "document_artifacts",
        ["source_artifact_id"],
        ["id"],
    )

    op.create_table(
        "document_sections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_section_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_sections.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("heading", sa.String(512), nullable=True),
        sa.Column("heading_path", postgresql.JSONB, nullable=True),
        sa.Column("ordinal", sa.Integer, nullable=True),
        sa.Column("page_start", sa.Integer, nullable=True),
        sa.Column("page_end", sa.Integer, nullable=True),
        sa.Column(
            "artifact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_artifacts.id"),
            nullable=True,
        ),
    )
    op.create_index("ix_document_sections_document_id", "document_sections", ["document_id"])

    op.create_table(
        "document_passages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "section_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_sections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_ordinal", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("body_text", sa.Text, nullable=False),
        sa.Column("contextualized_text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=True),
        sa.Column("page_start", sa.Integer, nullable=True),
        sa.Column("page_end", sa.Integer, nullable=True),
        sa.Column("provenance_offsets", postgresql.JSONB, nullable=True),
        sa.Column("quant_tags", postgresql.JSONB, nullable=True),
        sa.Column(
            "artifact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_artifacts.id"),
            nullable=True,
        ),
    )
    op.create_index("ix_document_passages_document_id", "document_passages", ["document_id"])
    op.create_index("ix_document_passages_section_id", "document_passages", ["section_id"])

    op.create_table(
        "document_tables",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "section_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_sections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("caption", sa.String(1024), nullable=True),
        sa.Column("table_type", sa.String(128), nullable=True),
        sa.Column("headers_json", postgresql.JSONB, nullable=True),
        sa.Column("rows_json", postgresql.JSONB, nullable=True),
        sa.Column("page_start", sa.Integer, nullable=True),
        sa.Column("page_end", sa.Integer, nullable=True),
        sa.Column("quant_tags", postgresql.JSONB, nullable=True),
        sa.Column(
            "artifact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_artifacts.id"),
            nullable=True,
        ),
    )
    op.create_index("ix_document_tables_document_id", "document_tables", ["document_id"])
    op.create_index("ix_document_tables_section_id", "document_tables", ["section_id"])

    op.create_table(
        "document_references",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("raw_citation", sa.Text, nullable=True),
        sa.Column("normalized_title", sa.String(1024), nullable=True),
        sa.Column("authors", postgresql.JSONB, nullable=True),
        sa.Column("publication_year", sa.Integer, nullable=True),
        sa.Column("doi", sa.String(256), nullable=True),
        sa.Column("source_confidence", sa.Float, nullable=True),
        sa.Column(
            "artifact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_artifacts.id"),
            nullable=True,
        ),
    )
    op.create_index("ix_document_references_document_id", "document_references", ["document_id"])

    op.create_table(
        "retrieval_index_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "ingest_job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("index_version", sa.String(128), nullable=False),
        sa.Column("embedding_provider", sa.String(128), nullable=True),
        sa.Column("embedding_model", sa.String(128), nullable=True),
        sa.Column("embedding_dimensions", sa.Integer, nullable=True),
        sa.Column("reranker_provider", sa.String(128), nullable=True),
        sa.Column("reranker_model", sa.String(128), nullable=True),
        sa.Column("chunking_version", sa.String(64), nullable=True),
        sa.Column("parser_source", sa.String(64), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default=sa.text("'building'")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_retrieval_index_runs_document_id", "retrieval_index_runs", ["document_id"])
    op.create_index(
        "ix_retrieval_index_runs_document_version",
        "retrieval_index_runs",
        ["document_id", "index_version"],
    )

    op.execute("SELECT pgmq.create('document_ingest')")


def downgrade() -> None:
    op.execute("SELECT pgmq.drop_queue('document_ingest')")
    op.drop_index("ix_retrieval_index_runs_document_version", table_name="retrieval_index_runs")
    op.drop_index("ix_retrieval_index_runs_document_id", table_name="retrieval_index_runs")
    op.drop_table("retrieval_index_runs")
    op.drop_index("ix_document_artifacts_ingest_job_id", table_name="document_artifacts")
    op.drop_index("ix_document_artifacts_document_id", table_name="document_artifacts")
    op.drop_constraint(
        "fk_ingest_jobs_source_artifact_id_document_artifacts",
        "ingest_jobs",
        type_="foreignkey",
    )
    op.drop_table("document_artifacts")
    op.drop_index("ix_ingest_jobs_status", table_name="ingest_jobs")
    op.drop_index("ix_ingest_jobs_document_id", table_name="ingest_jobs")
    op.drop_table("ingest_jobs")
    op.drop_index("ix_document_references_document_id", table_name="document_references")
    op.drop_table("document_references")
    op.drop_index("ix_document_tables_section_id", table_name="document_tables")
    op.drop_index("ix_document_tables_document_id", table_name="document_tables")
    op.drop_table("document_tables")
    op.drop_index("ix_document_passages_section_id", table_name="document_passages")
    op.drop_index("ix_document_passages_document_id", table_name="document_passages")
    op.drop_table("document_passages")
    op.drop_index("ix_document_sections_document_id", table_name="document_sections")
    op.drop_table("document_sections")
    op.drop_index("ix_documents_current_status", table_name="documents")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS pgmq CASCADE")
    op.execute("DROP EXTENSION IF EXISTS vector")
