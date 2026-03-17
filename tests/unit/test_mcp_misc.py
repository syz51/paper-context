from __future__ import annotations

import pytest

import paper_context.mcp.server as mcp_module
from paper_context.mcp import create_server as exported_create_server
from paper_context.retrieval import RetrievalService

pytestmark = pytest.mark.unit


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


def test_retrieval_service_health_summary() -> None:
    assert RetrievalService().health_summary() == {"status": "not-configured"}
