from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from paper_context.api import app as api_app_module
from paper_context.api.app import create_app
from paper_context.mcp.server import create_http_app as create_real_mcp_http_app
from paper_context.schemas.mcp import DocumentListResponse

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


class _DocumentsServiceStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search_documents(
        self,
        *,
        query: str,
        filters,
        cursor: str | None = None,
        limit: int = 20,
    ) -> DocumentListResponse:
        self.calls.append(
            {
                "query": query,
                "filters": filters,
                "cursor": cursor,
                "limit": limit,
            }
        )
        return DocumentListResponse()

    def get_document_outline(self, document_id):
        del document_id
        return None


class _RetrievalServiceStub:
    def search_passages_page(
        self, *, query: str, filters, cursor: str | None = None, limit: int = 8
    ):
        del query, filters, cursor, limit
        return SimpleNamespace(items=(), next_cursor=None)

    def search_tables_page(self, *, query: str, filters, cursor: str | None = None, limit: int = 5):
        del query, filters, cursor, limit
        return SimpleNamespace(items=(), next_cursor=None)

    def get_table(self, *, table_id):
        del table_id
        return None

    def get_passage_context(self, *, passage_id, before: int = 1, after: int = 1):
        del passage_id, before, after
        return None

    def build_context_pack(
        self,
        *,
        query: str,
        filters,
        cursor: str | None = None,
        limit: int = 8,
    ):
        del query, filters, cursor, limit
        return SimpleNamespace(
            context_pack_id="00000000-0000-0000-0000-000000000000",
            query="query",
            passages=(),
            tables=(),
            parent_sections=(),
            documents=(),
            provenance=SimpleNamespace(
                active_index_version="mvp-v1",
                retrieval_index_run_ids=(),
                retrieval_modes=(),
            ),
            warnings=(),
            next_cursor=None,
        )


def _sse_json(response_text: str) -> dict[str, Any]:
    for line in response_text.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"missing SSE data line in response: {response_text!r}")


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


def test_mcp_mount_serves_real_streamable_http_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = _DocumentsServiceStub()
    retrieval = _RetrievalServiceStub()
    settings = SimpleNamespace(
        log_level="INFO",
        storage=SimpleNamespace(root_path="."),
    )
    monkeypatch.setattr(
        api_app_module,
        "create_http_app",
        lambda: create_real_mcp_http_app(
            documents_service=cast(Any, documents),
            retrieval_service=cast(Any, retrieval),
        ),
    )
    monkeypatch.setattr(api_app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        api_app_module,
        "LocalFilesystemStorage",
        lambda root_path: SimpleNamespace(ensure_root=lambda: None),
    )
    monkeypatch.setattr(api_app_module, "configure_logging", lambda level: None)
    monkeypatch.setattr(api_app_module, "dispose_engine", lambda: None)

    headers = {
        "accept": "application/json, text/event-stream",
        "content-type": "application/json",
    }

    with TestClient(create_app()) as client:
        initialize = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "slice-test", "version": "0"},
                },
            },
        )
        session_id = initialize.headers.get("mcp-session-id")
        tool_call = client.post(
            "/mcp",
            headers={**headers, "mcp-session-id": session_id or ""},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "search_documents",
                    "arguments": {"query": "alpha", "limit": 999},
                },
            },
        )

    initialize_payload = _sse_json(initialize.text)
    tool_payload = _sse_json(tool_call.text)

    assert initialize.status_code == 200
    assert session_id
    assert initialize_payload["result"]["serverInfo"]["name"] == "paper-context"
    assert tool_call.status_code == 200
    assert tool_payload["result"]["isError"] is False
    assert tool_payload["result"]["structuredContent"] == {"documents": [], "next_cursor": None}
    assert documents.calls == [
        {
            "query": "alpha",
            "filters": None,
            "cursor": None,
            "limit": 100,
        }
    ]
