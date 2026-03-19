"""Add phase 4 observability metadata for ingest jobs."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0005_phase4_observability"
down_revision = "0004_doc_search_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ingest_jobs", sa.Column("stage_timings", postgresql.JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("ingest_jobs", "stage_timings")
