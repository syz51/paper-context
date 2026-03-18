from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

import paper_context.mcp.server as mcp_module
from paper_context.mcp import create_server as exported_create_server
from paper_context.retrieval import (
    DeterministicEmbeddingClient,
    HeuristicRerankerClient,
    RetrievalService,
)

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


def test_retrieval_service_health_summary_reports_configured_state() -> None:
    service = RetrievalService(
        connection_factory=lambda: nullcontext(MagicMock()),
        active_index_version="mvp-v1",
        embedding_client=DeterministicEmbeddingClient(model="voyage-4-large"),
        reranker_client=HeuristicRerankerClient(model="zerank-2"),
    )

    assert service.health_summary() == {
        "status": "configured",
        "embedding_provider": "deterministic",
        "reranker_provider": "deterministic",
        "active_index_version": "mvp-v1",
    }
