from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from paper_context.api import app as api_app_module
from paper_context.api.app import create_app

pytestmark = pytest.mark.slice


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


def test_mcp_http_app_mounts_and_runs_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
