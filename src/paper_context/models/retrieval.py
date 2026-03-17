from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .base import Base


class RetrievalIndexRun(Base):
    __tablename__ = "retrieval_index_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    ingest_job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ingest_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    index_version: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_dimensions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reranker_provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reranker_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    chunking_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parser_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="building",
        server_default=text("'building'"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
