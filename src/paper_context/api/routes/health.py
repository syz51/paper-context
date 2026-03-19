from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from paper_context import __version__
from paper_context.config import get_settings
from paper_context.db.engine import database_is_ready, get_engine
from paper_context.observability import get_metrics_registry
from paper_context.queue.contracts import IngestionQueue
from paper_context.schemas.common import (
    HealthResponse,
    OperationTimingResponse,
    QueueMetricsResponse,
    ReadinessResponse,
)

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(service="app", status="ok", version=__version__)


@router.get("/readyz", response_model=ReadinessResponse)
def readiness() -> ReadinessResponse:
    settings = get_settings()
    db_ready = database_is_ready()
    storage_root = settings.storage.root_path
    storage_ready = _storage_ready(storage_root)
    queue_ready = False
    queue_metrics = None
    if db_ready:
        queue_metrics = _queue_metrics(settings.queue.name)
        queue_ready = queue_metrics is not None
    return ReadinessResponse(
        service="app",
        status="ready" if db_ready and storage_ready and queue_ready else "degraded",
        version=__version__,
        database_ready=db_ready,
        storage_root=storage_root,
        storage_ready=storage_ready,
        queue_name=settings.queue.name,
        queue_ready=queue_ready,
        queue_metrics=queue_metrics,
        operation_timings=[
            OperationTimingResponse.model_validate(snapshot.__dict__)
            for snapshot in get_metrics_registry().timing_snapshots(limit=20)
        ],
    )


def _storage_ready(storage_root: Path) -> bool:
    return storage_root.exists() and storage_root.is_dir()


def _queue_metrics(queue_name: str) -> QueueMetricsResponse | None:
    try:
        with get_engine().begin() as connection:
            metrics = IngestionQueue(queue_name).queue_metrics(connection)
    except Exception:
        return None
    return QueueMetricsResponse.model_validate(metrics.__dict__)
