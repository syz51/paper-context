from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    service: str
    status: str
    version: str


class QueueMetricsResponse(BaseModel):
    queue_name: str
    queue_length: int
    queue_visible_length: int
    newest_msg_age_sec: int | None
    oldest_msg_age_sec: int | None
    total_messages: int
    scrape_time: datetime


class OperationTimingResponse(BaseModel):
    operation: str
    count: int
    total_ms: float
    avg_ms: float
    max_ms: float
    last_ms: float | None
    last_recorded_at: datetime | None


class ReadinessResponse(BaseModel):
    service: str
    status: str
    version: str
    database_ready: bool
    storage_root: Path
    storage_ready: bool
    queue_name: str
    queue_ready: bool
    queue_metrics: QueueMetricsResponse | None = None
    operation_timings: list[OperationTimingResponse] = Field(default_factory=list)
