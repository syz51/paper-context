"""Phase 2 retrieval schema for derived assets and activation metadata."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import UserDefinedType

from alembic import op

revision = "0002_phase2_retrieval_schema"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


class VectorType(UserDefinedType):
    cache_ok = True

    def __init__(self, dimensions: int | None = None) -> None:
        self.dimensions = dimensions

    def get_col_spec(self, **kw: object) -> str:
        if self.dimensions is None:
            return "vector"
        return f"vector({self.dimensions})"


def upgrade() -> None:
    op.add_column(
        "retrieval_index_runs",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "retrieval_index_runs",
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "retrieval_index_runs",
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_retrieval_index_runs_document_active_state",
        "retrieval_index_runs",
        ["document_id", "is_active"],
    )
    op.create_index(
        "ix_retrieval_index_runs_one_active_per_document",
        "retrieval_index_runs",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("is_active = true"),
    )

    op.create_table(
        "retrieval_passage_assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "retrieval_index_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("retrieval_index_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "passage_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_passages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "section_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_sections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("publication_year", sa.Integer(), nullable=True),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column("search_tsvector", postgresql.TSVECTOR(), nullable=False),
        sa.Column("embedding", VectorType(1024), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "retrieval_index_run_id",
            "passage_id",
            name="uq_retrieval_passage_assets_run_passage",
        ),
    )
    op.create_index(
        "ix_retrieval_passage_assets_retrieval_index_run_id",
        "retrieval_passage_assets",
        ["retrieval_index_run_id"],
    )
    op.create_index(
        "ix_retrieval_passage_assets_document_run",
        "retrieval_passage_assets",
        ["document_id", "retrieval_index_run_id"],
    )
    op.create_index(
        "ix_retrieval_passage_assets_publication_year_run",
        "retrieval_passage_assets",
        ["publication_year", "retrieval_index_run_id"],
    )
    op.create_index(
        "ix_retrieval_passage_assets_search_tsvector",
        "retrieval_passage_assets",
        ["search_tsvector"],
        postgresql_using="gin",
    )
    op.create_index(
        "ix_retrieval_passage_assets_embedding",
        "retrieval_passage_assets",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "retrieval_table_assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "retrieval_index_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("retrieval_index_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "table_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_tables.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "section_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_sections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("publication_year", sa.Integer(), nullable=True),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column("search_tsvector", postgresql.TSVECTOR(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "retrieval_index_run_id",
            "table_id",
            name="uq_retrieval_table_assets_run_table",
        ),
    )
    op.create_index(
        "ix_retrieval_table_assets_retrieval_index_run_id",
        "retrieval_table_assets",
        ["retrieval_index_run_id"],
    )
    op.create_index(
        "ix_retrieval_table_assets_document_run",
        "retrieval_table_assets",
        ["document_id", "retrieval_index_run_id"],
    )
    op.create_index(
        "ix_retrieval_table_assets_publication_year_run",
        "retrieval_table_assets",
        ["publication_year", "retrieval_index_run_id"],
    )
    op.create_index(
        "ix_retrieval_table_assets_search_tsvector",
        "retrieval_table_assets",
        ["search_tsvector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_retrieval_table_assets_search_tsvector", table_name="retrieval_table_assets")
    op.drop_index(
        "ix_retrieval_table_assets_publication_year_run",
        table_name="retrieval_table_assets",
    )
    op.drop_index("ix_retrieval_table_assets_document_run", table_name="retrieval_table_assets")
    op.drop_index(
        "ix_retrieval_table_assets_retrieval_index_run_id",
        table_name="retrieval_table_assets",
    )
    op.drop_table("retrieval_table_assets")

    op.drop_index("ix_retrieval_passage_assets_embedding", table_name="retrieval_passage_assets")
    op.drop_index(
        "ix_retrieval_passage_assets_search_tsvector",
        table_name="retrieval_passage_assets",
    )
    op.drop_index(
        "ix_retrieval_passage_assets_publication_year_run",
        table_name="retrieval_passage_assets",
    )
    op.drop_index(
        "ix_retrieval_passage_assets_document_run",
        table_name="retrieval_passage_assets",
    )
    op.drop_index(
        "ix_retrieval_passage_assets_retrieval_index_run_id",
        table_name="retrieval_passage_assets",
    )
    op.drop_table("retrieval_passage_assets")

    op.drop_index(
        "ix_retrieval_index_runs_one_active_per_document",
        table_name="retrieval_index_runs",
    )
    op.drop_index(
        "ix_retrieval_index_runs_document_active_state",
        table_name="retrieval_index_runs",
    )
    op.drop_column("retrieval_index_runs", "deactivated_at")
    op.drop_column("retrieval_index_runs", "activated_at")
    op.drop_column("retrieval_index_runs", "is_active")
