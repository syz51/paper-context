from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from paper_context import __version__
from paper_context.mcp import create_app as exported_create_app, create_server as exported_create_server
from paper_context.retrieval import RetrievalService
from paper_context.schemas import api as api_schemas
from paper_context.schemas import mcp as mcp_schemas
from paper_context.schemas.common import HealthResponse, ReadinessResponse

import paper_context.mcp.server as mcp_module


class FakeMcpApp:
    def __init__(self) -> None:
        self.events: list[str] = []

    @asynccontextmanager
    async def lifespan(self, app):  # pragma: no cover - exercised by TestClient startup
        self.events.append("lifespan-enter")
        yield
        self.events.append("lifespan-exit")

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})


def make_settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        log_level="INFO",
        storage=SimpleNamespace(root_path=tmp_path / "artifacts"),
        queue=SimpleNamespace(name="document_ingest"),
    )


def test_create_server_uses_fastmcp(monkeypatch: pytest.MonkeyPatch) -> None:
    fastmcp = pytest.importorskip("fastmcp")
    server = object()
    constructor = pytest.MonkeyPatch()
    constructor.setattr(mcp_module, "FastMCP", lambda name: server)
    try:
        assert exported_create_server() is server
    finally:
        constructor.undo()
    assert fastmcp is not None


def test_mcp_create_app_exposes_health_and_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    fake_mcp_app = FakeMcpApp()
    storage_events: list[str | tuple[str, Path]] = []

    class FakeServer:
        def http_app(self, *, path: str, transport: str) -> FakeMcpApp:
            assert path == "/"
            assert transport == "streamable-http"
            return fake_mcp_app

    monkeypatch.setattr(mcp_module, "get_settings", lambda: settings)
    monkeypatch.setattr(mcp_module, "create_server", lambda: FakeServer())
    monkeypatch.setattr(mcp_module, "configure_logging", lambda level: storage_events.append(level))
    monkeypatch.setattr(mcp_module, "dispose_engine", lambda: storage_events.append("dispose"))
    monkeypatch.setattr(mcp_module, "database_is_ready", lambda: True)

    class StubStorage:
        def __init__(self, root_path: Path) -> None:
            storage_events.append(("storage-init", root_path))

        def ensure_root(self) -> None:
            storage_events.append("ensure-root")

    monkeypatch.setattr(mcp_module, "LocalFilesystemStorage", StubStorage)

    with TestClient(exported_create_app()) as client:
        health = client.get("/healthz")
        ready = client.get("/readyz")

    assert health.json() == {"service": "mcp", "status": "ok", "version": __version__}
    assert ready.json() == {
        "service": "mcp",
        "status": "ready",
        "version": __version__,
        "database_ready": True,
        "storage_root": str(settings.storage.root_path),
        "queue_name": "document_ingest",
    }
    assert ("storage-init", settings.storage.root_path) in storage_events
    assert "ensure-root" in storage_events
    assert fake_mcp_app.events == ["lifespan-enter", "lifespan-exit"]
    assert "dispose" in storage_events


def test_mcp_create_app_reports_degraded_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    fake_mcp_app = FakeMcpApp()
    monkeypatch.setattr(mcp_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        mcp_module,
        "create_server",
        lambda: SimpleNamespace(http_app=lambda **kwargs: fake_mcp_app),
    )
    monkeypatch.setattr(
        mcp_module,
        "LocalFilesystemStorage",
        lambda root_path: SimpleNamespace(ensure_root=lambda: None),
    )
    monkeypatch.setattr(mcp_module, "configure_logging", lambda level: None)
    monkeypatch.setattr(mcp_module, "dispose_engine", lambda: None)
    monkeypatch.setattr(mcp_module, "database_is_ready", lambda: False)

    with TestClient(exported_create_app()) as client:
        ready = client.get("/readyz")
        mounted = client.get("/mcp")

    assert ready.json()["status"] == "degraded"
    assert mounted.text == "ok"


def test_retrieval_service_health_summary() -> None:
    assert RetrievalService().health_summary() == {"status": "not-configured"}


def test_schema_re_exports() -> None:
    assert api_schemas.HealthResponse is HealthResponse
    assert api_schemas.ReadinessResponse is ReadinessResponse
    assert mcp_schemas.HealthResponse is HealthResponse
    assert mcp_schemas.ReadinessResponse is ReadinessResponse
