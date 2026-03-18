from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .common import HealthResponse, ReadinessResponse


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    ingest_job_id: UUID
    status: str


class IngestJobResponse(BaseModel):
    id: UUID
    document_id: UUID
    status: str
    failure_code: str | None = None
    failure_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    trigger: str | None = None


__all__ = [
    "DocumentUploadResponse",
    "HealthResponse",
    "IngestJobResponse",
    "ReadinessResponse",
]
