from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import insert
from sqlalchemy.engine import Engine

from paper_context.models import Document, DocumentRevision, IngestJob
from paper_context.queue.contracts import IngestionQueue


class IngestionQueueService:
    def __init__(self, engine: Engine, adapter: IngestionQueue) -> None:
        self._engine = engine
        self._adapter = adapter

    def enqueue_document(
        self,
        document: Mapping[str, Any],
        trace_headers: Mapping[str, str] | None = None,
    ) -> tuple[uuid.UUID, uuid.UUID]:
        document_id = document.get("id") or uuid.uuid4()
        revision_id = uuid.uuid4()
        ingest_job_id = uuid.uuid4()
        now = datetime.now(UTC)
        with self._engine.begin() as conn:
            conn.execute(
                insert(Document).values(
                    id=document_id,
                    title=document.get("title"),
                    authors=document.get("authors"),
                    source_type=document.get("source_type"),
                    current_status="queued",
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                insert(DocumentRevision).values(
                    id=revision_id,
                    document_id=document_id,
                    revision_number=1,
                    title=document.get("title"),
                    authors=document.get("authors") or [],
                    abstract=document.get("abstract"),
                    publication_year=document.get("publication_year"),
                    source_type=document.get("source_type"),
                    metadata_confidence=document.get("metadata_confidence"),
                    quant_tags=document.get("quant_tags") or {},
                    status="queued",
                    activated_at=None,
                    superseded_at=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.execute(
                insert(IngestJob).values(
                    id=ingest_job_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    status="queued",
                    trigger=document.get("trigger", "upload"),
                    warnings=[],
                )
            )
            self._adapter.enqueue_ingest(
                conn,
                ingest_job_id,
                document_id,
                headers=trace_headers,
            )
        return document_id, ingest_job_id
