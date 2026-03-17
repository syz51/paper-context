from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class HealthResponse(BaseModel):
    service: str
    status: str
    version: str


class ReadinessResponse(BaseModel):
    service: str
    status: str
    version: str
    database_ready: bool
    storage_root: Path
    queue_name: str
