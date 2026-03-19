from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest

import paper_context.mcp.server as mcp_module
from paper_context.mcp import create_http_app as exported_create_http_app
from paper_context.mcp import create_server as exported_create_server
from paper_context.retrieval import (
    DeterministicEmbeddingClient,
    HeuristicRerankerClient,
    RetrievalFilters,
    RetrievalService,
)
from paper_context.schemas.api import (
    ContextPackResponse,
    DocumentListResponse,
    DocumentOutlineResponse,
    PassageContextResponse,
    PassageSearchResponse,
    TableDetailResponse,
    TableSearchResponse,
)

pytestmark = pytest.mark.unit


class _FakeFastMcp:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: dict[str, Callable[..., object]] = {}
        self.http_app_calls: list[dict[str, object]] = []

    def tool(self, fn=None, **kwargs):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        if fn is None:
            return decorator
        return decorator(fn)

    def http_app(self, *, path=None, transport=None, **kwargs):
        self.http_app_calls.append({"path": path, "transport": transport, "kwargs": kwargs})
        return self


class _DocumentsServiceStub:
    def __init__(self) -> None:
        self.search_documents_calls: list[dict[str, object]] = []
        self.outline_calls: list[UUID] = []
        self.search_documents_response = DocumentListResponse()
        self.outline_response = DocumentOutlineResponse(
            document_id=UUID("11111111-1111-1111-1111-111111111111"),
            title="Paper",
            sections=[],
        )

    def search_documents(
        self,
        *,
        query: str,
        filters,
        cursor: str | None = None,
        limit: int = 20,
    ) -> DocumentListResponse:
        self.search_documents_calls.append(
            {
                "query": query,
                "filters": filters,
                "cursor": cursor,
                "limit": limit,
            }
        )
        return self.search_documents_response

    def get_document_outline(self, document_id: UUID) -> DocumentOutlineResponse | None:
        self.outline_calls.append(document_id)
        return self.outline_response


class _RetrievalServiceStub:
    def __init__(self) -> None:
        self.search_passages_calls: list[dict[str, object]] = []
        self.search_tables_calls: list[dict[str, object]] = []
        self.get_table_calls: list[UUID] = []
        self.get_passage_context_calls: list[dict[str, object]] = []
        self.build_context_pack_calls: list[dict[str, object]] = []
        self.search_passages_response = SimpleNamespace(items=(), next_cursor="passages-cursor")
        self.search_tables_response = SimpleNamespace(items=(), next_cursor="tables-cursor")
        self.table_response = SimpleNamespace(
            table_id=UUID("22222222-2222-2222-2222-222222222222"),
            document_id=UUID("11111111-1111-1111-1111-111111111111"),
            section_id=UUID("33333333-3333-3333-3333-333333333333"),
            document_title="Paper",
            section_path=("Methods",),
            caption="Results table",
            table_type="lexical",
            headers=("A", "B"),
            rows=(("1", "2"),),
            row_count=1,
            page_start=1,
            page_end=1,
            index_version="mvp-v1",
            retrieval_index_run_id=UUID("44444444-4444-4444-4444-444444444444"),
            parser_source="docling",
            warnings=(),
        )
        self.passage_context_response = SimpleNamespace(
            passage=SimpleNamespace(
                passage_id=UUID("55555555-5555-5555-5555-555555555555"),
                document_id=UUID("11111111-1111-1111-1111-111111111111"),
                section_id=UUID("33333333-3333-3333-3333-333333333333"),
                document_title="Paper",
                section_path=("Methods",),
                text="context",
                chunk_ordinal=1,
                page_start=1,
                page_end=1,
                index_version="mvp-v1",
                retrieval_index_run_id=UUID("44444444-4444-4444-4444-444444444444"),
                parser_source="docling",
                warnings=(),
            ),
            context_passages=(),
            warnings=(),
        )
        self.context_pack_response = ContextPackResponse.model_validate(
            {
                "context_pack_id": "66666666-6666-6666-6666-666666666666",
                "query": "query",
                "passages": [],
                "tables": [],
                "parent_sections": [],
                "documents": [],
                "provenance": {
                    "active_index_version": "mvp-v1",
                    "retrieval_index_run_ids": [],
                    "retrieval_modes": [],
                },
                "warnings": [],
                "next_cursor": None,
            }
        )

    def search_passages_page(
        self, *, query: str, filters, cursor: str | None = None, limit: int = 8
    ):
        self.search_passages_calls.append(
            {"query": query, "filters": filters, "cursor": cursor, "limit": limit}
        )
        return self.search_passages_response

    def search_tables_page(self, *, query: str, filters, cursor: str | None = None, limit: int = 5):
        self.search_tables_calls.append(
            {"query": query, "filters": filters, "cursor": cursor, "limit": limit}
        )
        return self.search_tables_response

    def get_table(self, *, table_id: UUID):
        self.get_table_calls.append(table_id)
        return self.table_response

    def get_passage_context(self, *, passage_id: UUID, before: int = 1, after: int = 1):
        self.get_passage_context_calls.append(
            {"passage_id": passage_id, "before": before, "after": after}
        )
        return self.passage_context_response

    def build_context_pack(self, *, query: str, filters, cursor: str | None = None, limit: int = 8):
        self.build_context_pack_calls.append(
            {"query": query, "filters": filters, "cursor": cursor, "limit": limit}
        )
        return self.context_pack_response


def test_create_server_registers_all_expected_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_fastmcp = _FakeFastMcp("paper-context")
    documents = _DocumentsServiceStub()
    retrieval = _RetrievalServiceStub()

    monkeypatch.setattr(mcp_module, "FastMCP", lambda name: fake_fastmcp)
    monkeypatch.setattr(mcp_module, "_build_documents_service", lambda: documents)
    monkeypatch.setattr(mcp_module, "_build_retrieval_service", lambda: retrieval)

    server = exported_create_server()

    assert server is fake_fastmcp
    assert set(fake_fastmcp.tools) == {
        "search_documents",
        "search_passages",
        "search_tables",
        "get_document_outline",
        "get_table",
        "get_passage_context",
        "build_context_pack",
    }


def test_registered_mcp_tools_delegate_to_shared_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_fastmcp = _FakeFastMcp("paper-context")
    documents = _DocumentsServiceStub()
    retrieval = _RetrievalServiceStub()

    monkeypatch.setattr(mcp_module, "FastMCP", lambda name: fake_fastmcp)
    monkeypatch.setattr(mcp_module, "_build_documents_service", lambda: documents)
    monkeypatch.setattr(mcp_module, "_build_retrieval_service", lambda: retrieval)

    exported_create_server()

    docs_response = cast(
        DocumentListResponse,
        fake_fastmcp.tools["search_documents"](query="alpha"),
    )
    outline_response = fake_fastmcp.tools["get_document_outline"](
        "11111111-1111-1111-1111-111111111111"
    )
    passages_response = cast(
        PassageSearchResponse,
        fake_fastmcp.tools["search_passages"](query="beta"),
    )
    tables_response = cast(
        TableSearchResponse,
        fake_fastmcp.tools["search_tables"](query="gamma"),
    )
    table_response = cast(
        TableDetailResponse,
        fake_fastmcp.tools["get_table"]("22222222-2222-2222-2222-222222222222"),
    )
    context_response = cast(
        PassageContextResponse,
        fake_fastmcp.tools["get_passage_context"]("55555555-5555-5555-5555-555555555555"),
    )
    pack_response = cast(
        ContextPackResponse,
        fake_fastmcp.tools["build_context_pack"](query="delta"),
    )

    assert docs_response == documents.search_documents_response
    assert outline_response == documents.outline_response
    assert passages_response.query == "beta"
    assert passages_response.next_cursor == "passages-cursor"
    assert tables_response.query == "gamma"
    assert tables_response.next_cursor == "tables-cursor"
    assert table_response.table_id == retrieval.table_response.table_id
    assert (
        context_response.passage.passage_id == retrieval.passage_context_response.passage.passage_id
    )
    assert pack_response.context_pack_id == retrieval.context_pack_response.context_pack_id
    assert documents.search_documents_calls == [
        {"query": "alpha", "filters": None, "cursor": None, "limit": 20}
    ]
    assert documents.outline_calls == [UUID("11111111-1111-1111-1111-111111111111")]
    assert retrieval.search_passages_calls == [
        {
            "query": "beta",
            "filters": RetrievalFilters(),
            "cursor": None,
            "limit": 8,
        }
    ]
    assert retrieval.search_tables_calls == [
        {
            "query": "gamma",
            "filters": RetrievalFilters(),
            "cursor": None,
            "limit": 5,
        }
    ]
    assert retrieval.get_table_calls == [UUID("22222222-2222-2222-2222-222222222222")]
    assert retrieval.get_passage_context_calls == [
        {"passage_id": UUID("55555555-5555-5555-5555-555555555555"), "before": 1, "after": 1}
    ]
    assert retrieval.build_context_pack_calls == [
        {
            "query": "delta",
            "filters": RetrievalFilters(),
            "cursor": None,
            "limit": 8,
        }
    ]


def test_create_http_app_uses_streamable_http_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_fastmcp = _FakeFastMcp("paper-context")
    monkeypatch.setattr(mcp_module, "create_server", lambda **kwargs: fake_fastmcp)

    app = exported_create_http_app()

    assert app is fake_fastmcp
    assert fake_fastmcp.http_app_calls == [
        {"path": "/", "transport": "streamable-http", "kwargs": {}}
    ]


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
