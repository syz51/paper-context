"""Revisioned documents schema for replace-safe historical provenance."""

from __future__ import annotations

import uuid

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0003_document_revisions"
down_revision = "0002_phase2_retrieval_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_revisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default=sa.text("'queued'")),
        sa.Column("title", sa.String(1024), nullable=True),
        sa.Column("authors", postgresql.JSONB, nullable=True),
        sa.Column("abstract", sa.Text, nullable=True),
        sa.Column("publication_year", sa.Integer, nullable=True),
        sa.Column("source_type", sa.String(128), nullable=True),
        sa.Column("metadata_confidence", sa.Float, nullable=True),
        sa.Column("quant_tags", postgresql.JSONB, nullable=True),
        sa.Column(
            "source_artifact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_artifacts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "ingest_job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ingest_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "document_id",
            "revision_number",
            name="uq_document_revisions_document_revision_number",
        ),
    )

    op.add_column(
        "documents",
        sa.Column(
            "active_revision_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.add_column(
        "document_artifacts",
        sa.Column("revision_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "document_sections",
        sa.Column("revision_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "document_passages",
        sa.Column("revision_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "document_tables",
        sa.Column("revision_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "document_references",
        sa.Column("revision_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "ingest_jobs",
        sa.Column("revision_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "retrieval_index_runs",
        sa.Column("revision_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "retrieval_passage_assets",
        sa.Column("revision_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "retrieval_table_assets",
        sa.Column("revision_id", postgresql.UUID(as_uuid=True), nullable=True),
    )

    connection = op.get_bind()
    _backfill_document_revisions(connection)
    _backfill_revision_links(connection)

    op.create_index(
        "ix_document_revisions_document_id",
        "document_revisions",
        ["document_id"],
    )
    op.create_index(
        "ix_document_revisions_document_status",
        "document_revisions",
        ["document_id", "status"],
    )
    op.create_index("ix_documents_active_revision_id", "documents", ["active_revision_id"])
    op.create_index("ix_document_artifacts_revision_id", "document_artifacts", ["revision_id"])
    op.create_index(
        "ix_document_artifacts_revision_ingest_job_id",
        "document_artifacts",
        ["revision_id", "ingest_job_id"],
    )
    op.create_index("ix_document_sections_revision_id", "document_sections", ["revision_id"])
    op.create_index(
        "ix_document_sections_revision_artifact_id",
        "document_sections",
        ["revision_id", "artifact_id"],
    )
    op.create_index("ix_document_passages_revision_id", "document_passages", ["revision_id"])
    op.create_index(
        "ix_document_passages_revision_section_id",
        "document_passages",
        ["revision_id", "section_id"],
    )
    op.create_index("ix_document_tables_revision_id", "document_tables", ["revision_id"])
    op.create_index(
        "ix_document_tables_revision_section_id",
        "document_tables",
        ["revision_id", "section_id"],
    )
    op.create_index("ix_document_references_revision_id", "document_references", ["revision_id"])
    op.create_index(
        "ix_document_references_revision_artifact_id",
        "document_references",
        ["revision_id", "artifact_id"],
    )
    op.create_index("ix_ingest_jobs_revision_id", "ingest_jobs", ["revision_id"])
    op.create_index(
        "ix_ingest_jobs_revision_created_at_id",
        "ingest_jobs",
        ["revision_id", "created_at", "id"],
    )
    op.create_index("ix_retrieval_index_runs_revision_id", "retrieval_index_runs", ["revision_id"])
    op.create_index(
        "ix_retrieval_index_runs_revision_active_state",
        "retrieval_index_runs",
        ["revision_id", "is_active"],
    )
    op.create_index(
        "ix_retrieval_index_runs_revision_version",
        "retrieval_index_runs",
        ["revision_id", "index_version"],
    )
    op.create_index(
        "ix_retrieval_index_runs_one_active_per_revision",
        "retrieval_index_runs",
        ["revision_id"],
        unique=True,
        postgresql_where=sa.text("is_active = true"),
    )
    op.create_index(
        "ix_retrieval_passage_assets_revision_run",
        "retrieval_passage_assets",
        ["revision_id", "retrieval_index_run_id"],
    )
    op.create_index(
        "ix_retrieval_passage_assets_revision_publication_year_run",
        "retrieval_passage_assets",
        ["revision_id", "publication_year", "retrieval_index_run_id"],
    )
    op.create_index(
        "ix_retrieval_table_assets_revision_run",
        "retrieval_table_assets",
        ["revision_id", "retrieval_index_run_id"],
    )
    op.create_index(
        "ix_retrieval_table_assets_revision_publication_year_run",
        "retrieval_table_assets",
        ["revision_id", "publication_year", "retrieval_index_run_id"],
    )

    op.drop_index(
        "ix_retrieval_index_runs_one_active_per_document", table_name="retrieval_index_runs"
    )

    op.create_foreign_key(
        "fk_documents_active_revision_id_document_revisions",
        "documents",
        "document_revisions",
        ["active_revision_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_document_artifacts_revision_id_document_revisions",
        "document_artifacts",
        "document_revisions",
        ["revision_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_document_sections_revision_id_document_revisions",
        "document_sections",
        "document_revisions",
        ["revision_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_document_passages_revision_id_document_revisions",
        "document_passages",
        "document_revisions",
        ["revision_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_document_tables_revision_id_document_revisions",
        "document_tables",
        "document_revisions",
        ["revision_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_document_references_revision_id_document_revisions",
        "document_references",
        "document_revisions",
        ["revision_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_ingest_jobs_revision_id_document_revisions",
        "ingest_jobs",
        "document_revisions",
        ["revision_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_retrieval_index_runs_revision_id_document_revisions",
        "retrieval_index_runs",
        "document_revisions",
        ["revision_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_retrieval_passage_assets_revision_id_document_revisions",
        "retrieval_passage_assets",
        "document_revisions",
        ["revision_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_retrieval_table_assets_revision_id_document_revisions",
        "retrieval_table_assets",
        "document_revisions",
        ["revision_id"],
        ["id"],
        ondelete="CASCADE",
    )

    for table_name in (
        "document_artifacts",
        "document_sections",
        "document_passages",
        "document_tables",
        "document_references",
        "ingest_jobs",
        "retrieval_index_runs",
        "retrieval_passage_assets",
        "retrieval_table_assets",
    ):
        op.alter_column(
            table_name,
            "revision_id",
            existing_type=postgresql.UUID(as_uuid=True),
            nullable=False,
        )


def downgrade() -> None:
    for table_name, index_name in (
        ("retrieval_table_assets", "ix_retrieval_table_assets_revision_publication_year_run"),
        ("retrieval_table_assets", "ix_retrieval_table_assets_revision_run"),
        ("retrieval_passage_assets", "ix_retrieval_passage_assets_revision_publication_year_run"),
        ("retrieval_passage_assets", "ix_retrieval_passage_assets_revision_run"),
        ("retrieval_index_runs", "ix_retrieval_index_runs_one_active_per_revision"),
        ("retrieval_index_runs", "ix_retrieval_index_runs_revision_version"),
        ("retrieval_index_runs", "ix_retrieval_index_runs_revision_active_state"),
        ("retrieval_index_runs", "ix_retrieval_index_runs_revision_id"),
        ("ingest_jobs", "ix_ingest_jobs_revision_created_at_id"),
        ("ingest_jobs", "ix_ingest_jobs_revision_id"),
        ("document_references", "ix_document_references_revision_artifact_id"),
        ("document_references", "ix_document_references_revision_id"),
        ("document_tables", "ix_document_tables_revision_section_id"),
        ("document_tables", "ix_document_tables_revision_id"),
        ("document_passages", "ix_document_passages_revision_section_id"),
        ("document_passages", "ix_document_passages_revision_id"),
        ("document_sections", "ix_document_sections_revision_artifact_id"),
        ("document_sections", "ix_document_sections_revision_id"),
        ("document_artifacts", "ix_document_artifacts_revision_ingest_job_id"),
        ("document_artifacts", "ix_document_artifacts_revision_id"),
        ("documents", "ix_documents_active_revision_id"),
        ("document_revisions", "ix_document_revisions_document_status"),
        ("document_revisions", "ix_document_revisions_document_id"),
    ):
        op.drop_index(index_name, table_name=table_name)

    for constraint_name, table_name in (
        ("fk_retrieval_table_assets_revision_id_document_revisions", "retrieval_table_assets"),
        ("fk_retrieval_passage_assets_revision_id_document_revisions", "retrieval_passage_assets"),
        ("fk_retrieval_index_runs_revision_id_document_revisions", "retrieval_index_runs"),
        ("fk_ingest_jobs_revision_id_document_revisions", "ingest_jobs"),
        ("fk_document_references_revision_id_document_revisions", "document_references"),
        ("fk_document_tables_revision_id_document_revisions", "document_tables"),
        ("fk_document_passages_revision_id_document_revisions", "document_passages"),
        ("fk_document_sections_revision_id_document_revisions", "document_sections"),
        ("fk_document_artifacts_revision_id_document_revisions", "document_artifacts"),
        ("fk_documents_active_revision_id_document_revisions", "documents"),
    ):
        op.drop_constraint(constraint_name, table_name, type_="foreignkey")

    for table_name in (
        "retrieval_table_assets",
        "retrieval_passage_assets",
        "retrieval_index_runs",
        "ingest_jobs",
        "document_references",
        "document_tables",
        "document_passages",
        "document_sections",
        "document_artifacts",
    ):
        op.drop_column(table_name, "revision_id")

    op.drop_column("documents", "active_revision_id")
    op.drop_table("document_revisions")
    op.create_index(
        "ix_retrieval_index_runs_one_active_per_document",
        "retrieval_index_runs",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("is_active = true"),
    )


def _backfill_document_revisions(connection: sa.engine.Connection) -> None:
    documents = (
        connection.execute(
            sa.text(
                """
                SELECT
                    id,
                    title,
                    authors,
                    abstract,
                    publication_year,
                    source_type,
                    metadata_confidence,
                    quant_tags,
                    current_status,
                    created_at,
                    updated_at
                FROM documents
                ORDER BY id
                """
            )
        )
        .mappings()
        .all()
    )
    latest_jobs = {
        row["document_id"]: row
        for row in connection.execute(
            sa.text(
                """
                SELECT DISTINCT ON (document_id)
                    document_id,
                    id,
                    source_artifact_id
                FROM ingest_jobs
                ORDER BY document_id, created_at DESC, id DESC
                """
            )
        ).mappings()
    }
    latest_source_artifacts = {
        row["document_id"]: row["source_artifact_id"]
        for row in connection.execute(
            sa.text(
                """
                SELECT DISTINCT ON (document_id)
                    document_id,
                    id AS source_artifact_id
                FROM document_artifacts
                WHERE artifact_type = 'source_pdf'
                ORDER BY document_id, created_at DESC, id DESC
                """
            )
        ).mappings()
    }

    revision_rows: list[dict[str, object]] = []
    for document in documents:
        document_id = document["id"]
        revision_id = uuid.uuid4()
        latest_job = latest_jobs.get(document_id)
        source_artifact_id = None
        if latest_job is not None:
            source_artifact_id = latest_job.get("source_artifact_id")
        if source_artifact_id is None:
            source_artifact_id = latest_source_artifacts.get(document_id)
        revision_rows.append(
            {
                "id": revision_id,
                "document_id": document_id,
                "revision_number": 1,
                "status": document["current_status"],
                "title": document["title"],
                "authors": document["authors"],
                "abstract": document["abstract"],
                "publication_year": document["publication_year"],
                "source_type": document["source_type"],
                "metadata_confidence": document["metadata_confidence"],
                "quant_tags": document["quant_tags"],
                "source_artifact_id": source_artifact_id,
                "ingest_job_id": latest_job["id"] if latest_job is not None else None,
                "activated_at": document["updated_at"]
                if document["current_status"] == "ready"
                else None,
                "superseded_at": None,
                "created_at": document["created_at"],
                "updated_at": document["updated_at"],
            }
        )

    if revision_rows:
        connection.execute(
            sa.text(
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
                    :authors,
                    :abstract,
                    :publication_year,
                    :source_type,
                    :metadata_confidence,
                    :quant_tags,
                    :source_artifact_id,
                    :ingest_job_id,
                    :activated_at,
                    :superseded_at,
                    :created_at,
                    :updated_at
                )
                """
            ),
            revision_rows,
        )

        connection.execute(
            sa.text(
                """
                UPDATE documents
                SET active_revision_id = revisions.id
                FROM document_revisions revisions
                WHERE revisions.document_id = documents.id
                """
            )
        )


def _backfill_revision_links(connection: sa.engine.Connection) -> None:
    revisions = sa.table(
        "document_revisions",
        sa.column("id"),
        sa.column("document_id"),
    )
    table_names = (
        "document_artifacts",
        "document_sections",
        "document_passages",
        "document_tables",
        "document_references",
        "ingest_jobs",
        "retrieval_index_runs",
        "retrieval_passage_assets",
        "retrieval_table_assets",
    )
    for table_name in table_names:
        target = sa.table(
            table_name,
            sa.column("revision_id"),
            sa.column("document_id"),
        )
        connection.execute(
            sa.update(target)
            .values(revision_id=revisions.c.id)
            .where(revisions.c.document_id == target.c.document_id)
        )
