"""Add semantic retrieval fields for table hybrid search."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.types import UserDefinedType

from alembic import op

revision = "0006_table_hybrid_embeds"
down_revision = "0005_phase4_observability"
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
        "retrieval_table_assets",
        sa.Column("semantic_text", sa.Text(), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE retrieval_table_assets
            SET semantic_text = search_text
            WHERE semantic_text IS NULL
            """
        )
    )
    op.alter_column("retrieval_table_assets", "semantic_text", nullable=False)
    op.add_column(
        "retrieval_table_assets",
        sa.Column("embedding", VectorType(1024), nullable=True),
    )
    op.create_index(
        "ix_retrieval_table_assets_embedding",
        "retrieval_table_assets",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_retrieval_table_assets_embedding", table_name="retrieval_table_assets")
    op.drop_column("retrieval_table_assets", "embedding")
    op.drop_column("retrieval_table_assets", "semantic_text")
