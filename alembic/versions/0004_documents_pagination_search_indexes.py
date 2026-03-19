"""Add document listing and search indexes."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0004_doc_search_idx"
down_revision = "0003_document_revisions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_documents_updated_at_id",
        "documents",
        ["updated_at", "id"],
    )
    op.create_index(
        "ix_documents_search_fts",
        "documents",
        [
            sa.text(
                """
                to_tsvector(
                    'english',
                    coalesce(title, '')
                    || ' '
                    || coalesce(abstract, '')
                    || ' '
                    || translate(coalesce(CAST(authors AS text), ''), '[]"', '   ')
                )
                """
            )
        ],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_documents_search_fts", table_name="documents")
    op.drop_index("ix_documents_updated_at_id", table_name="documents")
