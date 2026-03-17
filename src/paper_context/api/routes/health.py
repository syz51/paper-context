from __future__ import annotations

from fastapi import APIRouter

from paper_context import __version__
from paper_context.config import get_settings
from paper_context.db.engine import database_is_ready
from paper_context.schemas.common import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(service="app", status="ok", version=__version__)


@router.get("/readyz", response_model=ReadinessResponse)
def readiness() -> ReadinessResponse:
    settings = get_settings()
    db_ready = database_is_ready()
    return ReadinessResponse(
        service="app",
        status="ready" if db_ready else "degraded",
        version=__version__,
        database_ready=db_ready,
        storage_root=settings.storage.root_path,
        queue_name=settings.queue.name,
    )
