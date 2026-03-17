from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from paper_context import __version__
from paper_context.api import app as api_app_module
from paper_context.api.app import create_app
from paper_context.api.routes import health as health_module

pytestmark = pytest.mark.slice


def _patch_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    storage = SimpleNamespace(root_path=tmp_path / "artifacts")
    queue = SimpleNamespace(name="document_ingest")
    runtime = SimpleNamespace(worker_idle_sleep_seconds=0.1)
    settings = SimpleNamespace(
        log_level="INFO",
        storage=storage,
        queue=queue,
        runtime=runtime,
    )
    monkeypatch.setattr(api_app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(health_module, "get_settings", lambda: settings)
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

    with TestClient(create_app()) as client:
        response = client.get("/readyz")

    payload = response.json()
    assert payload["status"] == status
    assert payload["database_ready"] is db_ready
    assert payload["queue_name"] == "document_ingest"
    assert payload["service"] == "app"
