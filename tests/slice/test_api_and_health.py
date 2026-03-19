from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from paper_context import __version__
from paper_context.api import app as api_app_module
from paper_context.api.app import create_app
from paper_context.api.routes import health as health_module
from paper_context.schemas.common import QueueMetricsResponse

pytestmark = pytest.mark.slice


def _patch_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    storage = SimpleNamespace(root_path=tmp_path / "artifacts")
    storage.root_path.mkdir(parents=True, exist_ok=True)
    queue = SimpleNamespace(name="document_ingest")
    runtime = SimpleNamespace(worker_idle_sleep_seconds=0.1)
    settings = SimpleNamespace(
        log_level="INFO",
        storage=storage,
        queue=queue,
        runtime=runtime,
        providers=SimpleNamespace(
            voyage_model="voyage-4-large",
            reranker_model="zerank-2",
            index_version="mvp-v1",
        ),
    )
    monkeypatch.setattr(api_app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(health_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        health_module,
        "get_metrics_registry",
        lambda: SimpleNamespace(timing_snapshots=lambda limit=20: []),
    )
    return settings


def test_lifespan_invokes_storage_and_dispose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _patch_settings(monkeypatch, tmp_path)
    calls: list[str | tuple[str, Path]] = []

    class StubStorage:
        def __init__(self, root_path: Path) -> None:
            calls.append(("storage-init", root_path))

        def ensure_root(self) -> None:
            calls.append("ensure-root")

    monkeypatch.setattr(api_app_module, "LocalFilesystemStorage", StubStorage)
    monkeypatch.setattr(api_app_module, "dispose_engine", lambda: calls.append("dispose"))
    monkeypatch.setattr(
        api_app_module,
        "configure_logging",
        lambda level: calls.append(("log", level)),
    )

    with TestClient(create_app()) as client:
        client.get("/healthz")

    assert ("storage-init", settings.storage.root_path) in calls
    assert "ensure-root" in calls
    assert "dispose" in calls


def test_health_endpoint_reports_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(
        api_app_module,
        "LocalFilesystemStorage",
        lambda root_path: SimpleNamespace(ensure_root=lambda: None),
    )
    monkeypatch.setattr(api_app_module, "configure_logging", lambda level: None)
    monkeypatch.setattr(api_app_module, "dispose_engine", lambda: None)

    with TestClient(create_app()) as client:
        response = client.get("/healthz")

    payload = response.json()
    assert payload["service"] == "app"
    assert payload["status"] == "ok"
    assert payload["version"] == __version__


@pytest.mark.parametrize("db_ready,status", [(True, "ready"), (False, "degraded")])
def test_readiness_reflects_database_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db_ready: bool, status: str
) -> None:
    _patch_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(
        api_app_module,
        "LocalFilesystemStorage",
        lambda root_path: SimpleNamespace(ensure_root=lambda: None),
    )
    monkeypatch.setattr(api_app_module, "configure_logging", lambda level: None)
    monkeypatch.setattr(api_app_module, "dispose_engine", lambda: None)
    monkeypatch.setattr(health_module, "database_is_ready", lambda: db_ready)
    monkeypatch.setattr(
        health_module,
        "_queue_metrics",
        lambda queue_name: (
            QueueMetricsResponse(
                queue_name=queue_name,
                queue_length=3,
                queue_visible_length=2,
                newest_msg_age_sec=5,
                oldest_msg_age_sec=10,
                total_messages=3,
                scrape_time=datetime.now(UTC),
            )
            if db_ready
            else None
        ),
    )

    with TestClient(create_app()) as client:
        response = client.get("/readyz")

    payload = response.json()
    assert payload["status"] == status
    assert payload["database_ready"] is db_ready
    assert payload["storage_ready"] is True
    assert payload["queue_name"] == "document_ingest"
    assert payload["queue_ready"] is db_ready
    if db_ready:
        assert payload["queue_metrics"]["queue_visible_length"] == 2
    else:
        assert payload["queue_metrics"] is None
    assert payload["operation_timings"] == []
    assert payload["service"] == "app"
