from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.types import UserDefinedType

from .base import Base


class PgVector(UserDefinedType):
    """Lightweight SQLAlchemy type for the pgvector `vector` column type."""

    cache_ok = True

    def __init__(self, dimensions: int | None = None) -> None:
        self.dimensions = dimensions

    def get_col_spec(self, **kw: object) -> str:
        if self.dimensions is None:
            return "vector"
        return f"vector({self.dimensions})"

    def bind_processor(self, dialect):  # type: ignore[override]
        def process(value: Sequence[float] | str | None) -> str | None:
            if value is None or isinstance(value, str):
                return value
            return "[" + ",".join(str(component) for component in value) + "]"

        return process

    def result_processor(self, dialect, coltype):  # type: ignore[override]
        def process(
            value: str | list[float] | tuple[float, ...] | None,
        ) -> list[float] | None:
            if value is None or isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                stripped = stripped[1:-1]
            if not stripped:
                return []
            return [float(component) for component in stripped.split(",")]

        return process


class RetrievalIndexRun(Base):
    __tablename__ = "retrieval_index_runs"
    __table_args__ = (
        Index("ix_retrieval_index_runs_document_id", "document_id"),
        Index("ix_retrieval_index_runs_document_active_state", "document_id", "is_active"),
        Index("ix_retrieval_index_runs_revision_id", "revision_id"),
        Index("ix_retrieval_index_runs_revision_active_state", "revision_id", "is_active"),
        Index("ix_retrieval_index_runs_document_version", "document_id", "index_version"),
        Index("ix_retrieval_index_runs_revision_version", "revision_id", "index_version"),
        Index(
            "ix_retrieval_index_runs_one_active_per_revision",
            "revision_id",
            unique=True,
            postgresql_where=text("is_active = true"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_revisions.id", ondelete="CASCADE"),
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
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deactivated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RetrievalPassageAsset(Base):
    __tablename__ = "retrieval_passage_assets"
    __table_args__ = (
        UniqueConstraint(
            "retrieval_index_run_id",
            "passage_id",
            name="uq_retrieval_passage_assets_run_passage",
        ),
        Index(
            "ix_retrieval_passage_assets_retrieval_index_run_id",
            "retrieval_index_run_id",
        ),
        Index(
            "ix_retrieval_passage_assets_document_run",
            "document_id",
            "retrieval_index_run_id",
        ),
        Index(
            "ix_retrieval_passage_assets_revision_run",
            "revision_id",
            "retrieval_index_run_id",
        ),
        Index(
            "ix_retrieval_passage_assets_publication_year_run",
            "publication_year",
            "retrieval_index_run_id",
        ),
        Index(
            "ix_retrieval_passage_assets_revision_publication_year_run",
            "revision_id",
            "publication_year",
            "retrieval_index_run_id",
        ),
        Index(
            "ix_retrieval_passage_assets_search_tsvector",
            "search_tsvector",
            postgresql_using="gin",
        ),
        Index(
            "ix_retrieval_passage_assets_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    retrieval_index_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("retrieval_index_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_revisions.id", ondelete="CASCADE"),
        nullable=False,
    )
    passage_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_passages.id", ondelete="CASCADE"),
        nullable=False,
    )
    section_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_sections.id", ondelete="CASCADE"),
        nullable=False,
    )
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    search_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_tsvector: Mapped[str] = mapped_column(TSVECTOR, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(PgVector(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RetrievalTableAsset(Base):
    __tablename__ = "retrieval_table_assets"
    __table_args__ = (
        UniqueConstraint(
            "retrieval_index_run_id",
            "table_id",
            name="uq_retrieval_table_assets_run_table",
        ),
        Index(
            "ix_retrieval_table_assets_retrieval_index_run_id",
            "retrieval_index_run_id",
        ),
        Index(
            "ix_retrieval_table_assets_document_run",
            "document_id",
            "retrieval_index_run_id",
        ),
        Index(
            "ix_retrieval_table_assets_revision_run",
            "revision_id",
            "retrieval_index_run_id",
        ),
        Index(
            "ix_retrieval_table_assets_publication_year_run",
            "publication_year",
            "retrieval_index_run_id",
        ),
        Index(
            "ix_retrieval_table_assets_revision_publication_year_run",
            "revision_id",
            "publication_year",
            "retrieval_index_run_id",
        ),
        Index(
            "ix_retrieval_table_assets_search_tsvector",
            "search_tsvector",
            postgresql_using="gin",
        ),
        Index(
            "ix_retrieval_table_assets_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    retrieval_index_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("retrieval_index_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_revisions.id", ondelete="CASCADE"),
        nullable=False,
    )
    table_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_tables.id", ondelete="CASCADE"),
        nullable=False,
    )
    section_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_sections.id", ondelete="CASCADE"),
        nullable=False,
    )
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    search_text: Mapped[str] = mapped_column(Text, nullable=False)
    semantic_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_tsvector: Mapped[str] = mapped_column(TSVECTOR, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(PgVector(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
