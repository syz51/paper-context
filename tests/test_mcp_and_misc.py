from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import paper_context.mcp.server as mcp_module
from paper_context.api import app as api_app_module
from paper_context.api.app import create_app
from paper_context.mcp import create_server as exported_create_server
from paper_context.retrieval import RetrievalService
from paper_context.schemas import api as api_schemas
from paper_context.schemas import mcp as mcp_schemas
from paper_context.schemas.common import HealthResponse, ReadinessResponse


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


def test_mcp_http_app_mounts_and_runs_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp_app = FakeMcpApp()
    settings = SimpleNamespace(
        log_level="INFO",
        storage=SimpleNamespace(root_path="."),
    )
    monkeypatch.setattr(api_app_module, "create_http_app", lambda: fake_mcp_app)
    monkeypatch.setattr(api_app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        api_app_module,
        "LocalFilesystemStorage",
        lambda root_path: SimpleNamespace(ensure_root=lambda: None),
    )
    monkeypatch.setattr(api_app_module, "configure_logging", lambda level: None)
    monkeypatch.setattr(api_app_module, "dispose_engine", lambda: None)

    with TestClient(create_app()) as client:
        response = client.get("/mcp")

    assert response.text == "ok"
    assert fake_mcp_app.events == ["lifespan-enter", "lifespan-exit"]


def test_retrieval_service_health_summary() -> None:
    assert RetrievalService().health_summary() == {"status": "not-configured"}


def test_schema_re_exports() -> None:
    assert api_schemas.HealthResponse is HealthResponse
    assert api_schemas.ReadinessResponse is ReadinessResponse
    assert mcp_schemas.HealthResponse is HealthResponse
    assert mcp_schemas.ReadinessResponse is ReadinessResponse
