from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    authors: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metadata_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    quant_tags: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    current_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
        server_default=text("'queued'"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class DocumentArtifact(Base):
    __tablename__ = "document_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    parser: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_ref: Mapped[str] = mapped_column(String(2048), nullable=False)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DocumentSection(Base):
    __tablename__ = "document_sections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_section_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_sections.id", ondelete="CASCADE"), nullable=True
    )
    heading: Mapped[str | None] = mapped_column(String(512), nullable=True)
    heading_path: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    ordinal: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_artifacts.id"), nullable=True
    )


class DocumentPassage(Base):
    __tablename__ = "document_passages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    section_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_sections.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    body_text: Mapped[str] = mapped_column(Text, nullable=False)
    contextualized_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provenance_offsets: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    quant_tags: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_artifacts.id"), nullable=True
    )


class DocumentTable(Base):
    __tablename__ = "document_tables"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    section_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_sections.id", ondelete="CASCADE"),
        nullable=False,
    )
    caption: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    table_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    headers_json: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    rows_json: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quant_tags: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_artifacts.id"), nullable=True
    )


class DocumentReference(Base):
    __tablename__ = "document_references"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    raw_citation: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_title: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    authors: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(256), nullable=True)
    source_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_artifacts.id"), nullable=True
    )
