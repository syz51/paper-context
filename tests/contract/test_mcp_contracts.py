from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from paper_context.api import app as api_app_module
from paper_context.api.app import create_app
from paper_context.mcp.server import create_http_app as create_real_mcp_http_app
from paper_context.schemas.mcp import (
    ContextPackResponse,
    DocumentListResponse,
    DocumentOutlineResponse,
    PassageContextResponse,
    PassageSearchResponse,
    TableDetailResponse,
    TableSearchResponse,
)
from paper_context.schemas.public import (
    ContextPackResponse as PublicContextPackResponse,
)
from paper_context.schemas.public import (
    DocumentListResponse as PublicDocumentListResponse,
)
from paper_context.schemas.public import (
    DocumentOutlineNode,
    DocumentResult,
    RetrievalFiltersInput,
)
from paper_context.schemas.public import (
    DocumentOutlineResponse as PublicDocumentOutlineResponse,
)

pytestmark = pytest.mark.contract

GOLDEN_DIR = Path(__file__).with_name("golden")


class _DocumentsServiceStub:
    def __init__(self) -> None:
        self.document_id = UUID("11111111-1111-1111-1111-111111111111")
        self.search_documents_response = PublicDocumentListResponse(
            documents=[
                DocumentResult(
                    document_id=self.document_id,
                    title="Phase 4 paper",
                    authors=["Ada Lovelace"],
                    publication_year=2024,
                    quant_tags={"asset_universe": "rates"},
                    current_status="ready",
                    active_index_version="mvp-v1",
                )
            ],
            next_cursor="docs-cursor",
        )
        self.outline_response = PublicDocumentOutlineResponse(
            document_id=self.document_id,
            title="Phase 4 paper",
            sections=[
                DocumentOutlineNode(
                    section_id=UUID("22222222-2222-2222-2222-222222222222"),
                    heading="Methods",
                    section_path=["Methods"],
                    ordinal=1,
                    page_start=1,
                    page_end=2,
                    children=[],
                )
            ],
        )

    def search_documents(
        self,
        *,
        query: str,
        filters: RetrievalFiltersInput | None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> PublicDocumentListResponse:
        del query, filters, cursor, limit
        return self.search_documents_response

    def get_document_outline(self, document_id: UUID) -> PublicDocumentOutlineResponse | None:
        return self.outline_response if document_id == self.document_id else None


class _RetrievalServiceStub:
    def __init__(self) -> None:
        self.document_id = UUID("11111111-1111-1111-1111-111111111111")
        self.section_id = UUID("22222222-2222-2222-2222-222222222222")
        self.passage_id = UUID("33333333-3333-3333-3333-333333333333")
        self.table_id = UUID("44444444-4444-4444-4444-444444444444")
        self.run_id = UUID("55555555-5555-5555-5555-555555555555")
        self.search_passages_response = SimpleNamespace(
            items=(
                SimpleNamespace(
                    passage_id=self.passage_id,
                    document_id=self.document_id,
                    section_id=self.section_id,
                    document_title="Phase 4 paper",
                    section_path=("Methods",),
                    text="A phase-4 passage.",
                    score=0.91,
                    retrieval_modes=("sparse", "dense"),
                    page_start=1,
                    page_end=1,
                    index_version="mvp-v1",
                    retrieval_index_run_id=self.run_id,
                    parser_source="docling",
                    warnings=("parser_fallback_used",),
                ),
            ),
            next_cursor="passages-cursor",
            exact=True,
            truncated=False,
            warnings=(),
        )
        preview = SimpleNamespace(
            headers=("Metric", "Value"), rows=(("Sharpe", "1.2"),), row_count=1
        )
        self.search_tables_response = SimpleNamespace(
            items=(
                SimpleNamespace(
                    table_id=self.table_id,
                    document_id=self.document_id,
                    section_id=self.section_id,
                    document_title="Phase 4 paper",
                    section_path=("Results",),
                    caption="Metrics",
                    table_type="lexical",
                    preview=preview,
                    score=0.88,
                    retrieval_modes=("sparse", "dense"),
                    page_start=2,
                    page_end=2,
                    index_version="mvp-v1",
                    retrieval_index_run_id=self.run_id,
                    parser_source="docling",
                    warnings=(),
                ),
            ),
            next_cursor="tables-cursor",
            exact=True,
            truncated=False,
            warnings=(),
        )
        self.table_response = SimpleNamespace(
            table_id=self.table_id,
            document_id=self.document_id,
            section_id=self.section_id,
            document_title="Phase 4 paper",
            section_path=("Results",),
            caption="Metrics",
            table_type="lexical",
            headers=("Metric", "Value"),
            rows=(("Sharpe", "1.2"), ("Sortino", "1.7")),
            row_count=2,
            page_start=2,
            page_end=2,
            index_version="mvp-v1",
            retrieval_index_run_id=self.run_id,
            parser_source="docling",
            warnings=(),
        )
        self.passage_context_response = SimpleNamespace(
            passage=SimpleNamespace(
                passage_id=self.passage_id,
                document_id=self.document_id,
                section_id=self.section_id,
                document_title="Phase 4 paper",
                section_path=("Methods",),
                text="A phase-4 passage.",
                chunk_ordinal=0,
                page_start=1,
                page_end=1,
                index_version="mvp-v1",
                retrieval_index_run_id=self.run_id,
                parser_source="docling",
                warnings=(),
            ),
            context_passages=(
                SimpleNamespace(
                    passage_id=UUID("66666666-6666-6666-6666-666666666666"),
                    text="Neighbor passage.",
                    chunk_ordinal=1,
                    page_start=1,
                    page_end=1,
                    relationship="after",
                ),
            ),
            warnings=("parent_context_truncated",),
        )
        self.context_pack_response = PublicContextPackResponse.model_validate(
            {
                "context_pack_id": "77777777-7777-7777-7777-777777777777",
                "query": "phase 4",
                "passages": [
                    {
                        "passage_id": str(self.passage_id),
                        "document_id": str(self.document_id),
                        "section_id": str(self.section_id),
                        "document_title": "Phase 4 paper",
                        "section_path": ["Methods"],
                        "text": "A phase-4 passage.",
                        "score": 0.91,
                        "retrieval_modes": ["sparse", "dense"],
                        "page_start": 1,
                        "page_end": 1,
                        "index_version": "mvp-v1",
                        "retrieval_index_run_id": str(self.run_id),
                        "parser_source": "docling",
                        "warnings": ["parser_fallback_used"],
                    }
                ],
                "tables": [
                    {
                        "table_id": str(self.table_id),
                        "document_id": str(self.document_id),
                        "section_id": str(self.section_id),
                        "document_title": "Phase 4 paper",
                        "section_path": ["Results"],
                        "caption": "Metrics",
                        "table_type": "lexical",
                        "preview": {
                            "headers": ["Metric", "Value"],
                            "rows": [["Sharpe", "1.2"]],
                            "row_count": 1,
                        },
                        "score": 0.88,
                        "retrieval_modes": ["sparse", "dense"],
                        "page_start": 2,
                        "page_end": 2,
                        "index_version": "mvp-v1",
                        "retrieval_index_run_id": str(self.run_id),
                        "parser_source": "docling",
                        "warnings": [],
                    }
                ],
                "parent_sections": [],
                "documents": [
                    {
                        "document_id": str(self.document_id),
                        "title": "Phase 4 paper",
                        "authors": ["Ada Lovelace"],
                        "publication_year": 2024,
                        "quant_tags": {"asset_universe": "rates"},
                        "current_status": "ready",
                        "active_index_version": "mvp-v1",
                    }
                ],
                "provenance": {
                    "active_index_version": "mvp-v1",
                    "retrieval_index_run_ids": [str(self.run_id)],
                    "retrieval_modes": ["sparse", "dense"],
                },
                "warnings": ["parser_fallback_used"],
                "next_cursor": "pack-cursor",
            }
        )

    def search_passages_page(
        self,
        *,
        query: str,
        filters,
        cursor: str | None = None,
        limit: int = 8,
        pagination_mode: str = "exact",
        max_rerank_candidates: int | None = None,
        max_expansion_rounds: int | None = None,
    ):
        del (
            query,
            filters,
            cursor,
            limit,
            pagination_mode,
            max_rerank_candidates,
            max_expansion_rounds,
        )
        return self.search_passages_response

    def search_tables_page(
        self,
        *,
        query: str,
        filters,
        cursor: str | None = None,
        limit: int = 5,
        pagination_mode: str = "exact",
        max_rerank_candidates: int | None = None,
        max_expansion_rounds: int | None = None,
    ):
        del (
            query,
            filters,
            cursor,
            limit,
            pagination_mode,
            max_rerank_candidates,
            max_expansion_rounds,
        )
        return self.search_tables_response

    def get_table(self, *, table_id: UUID):
        return self.table_response if table_id == self.table_id else None

    def get_passage_context(self, *, passage_id: UUID, before: int = 1, after: int = 1):
        del before, after
        return self.passage_context_response if passage_id == self.passage_id else None

    def build_context_pack(self, *, query: str, filters, cursor: str | None = None, limit: int = 8):
        del query, filters, cursor, limit
        return self.context_pack_response


def _golden_payload(name: str) -> dict[str, object]:
    return json.loads((GOLDEN_DIR / name).read_text())


def _sse_json(response_text: str) -> dict[str, Any]:
    for line in response_text.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"missing SSE data line in response: {response_text!r}")


def _build_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    settings = SimpleNamespace(
        log_level="INFO",
        storage=SimpleNamespace(root_path=tmp_path / "artifacts"),
        queue=SimpleNamespace(name="document_ingest"),
        providers=SimpleNamespace(
            voyage_model="voyage-4-large",
            reranker_model="zerank-2",
            index_version="mvp-v1",
        ),
    )
    monkeypatch.setattr(
        api_app_module,
        "create_http_app",
        lambda: create_real_mcp_http_app(
            documents_service=cast(Any, _DocumentsServiceStub()),
            retrieval_service=cast(Any, _RetrievalServiceStub()),
        ),
    )
    monkeypatch.setattr(api_app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        api_app_module,
        "LocalFilesystemStorage",
        lambda root_path: SimpleNamespace(
            ensure_root=lambda: Path(root_path).mkdir(parents=True, exist_ok=True)
        ),
    )
    monkeypatch.setattr(api_app_module, "configure_logging", lambda level: None)
    monkeypatch.setattr(api_app_module, "dispose_engine", lambda: None)
    return TestClient(create_app())


def _call_tool(
    client: TestClient,
    *,
    name: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    headers = {
        "accept": "application/json, text/event-stream",
        "content-type": "application/json",
    }
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
                "clientInfo": {"name": "contract-test", "version": "0"},
            },
        },
    )
    session_id = initialize.headers["mcp-session-id"]
    response = client.post(
        "/mcp",
        headers={**headers, "mcp-session-id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    payload = _sse_json(response.text)
    assert response.status_code == 200
    assert payload["result"]["isError"] is False
    return cast(dict[str, object], payload["result"]["structuredContent"])


@pytest.mark.parametrize(
    ("tool_name", "arguments", "golden_name", "schema"),
    [
        (
            "search_documents",
            {"query": "phase 4"},
            "mcp-search-documents.json",
            DocumentListResponse,
        ),
        (
            "search_passages",
            {"query": "phase 4"},
            "mcp-search-passages.json",
            PassageSearchResponse,
        ),
        ("search_tables", {"query": "phase 4"}, "mcp-search-tables.json", TableSearchResponse),
        (
            "get_document_outline",
            {"document_id": "11111111-1111-1111-1111-111111111111"},
            "mcp-get-document-outline.json",
            DocumentOutlineResponse,
        ),
        (
            "get_table",
            {"table_id": "44444444-4444-4444-4444-444444444444"},
            "mcp-get-table.json",
            TableDetailResponse,
        ),
        (
            "get_passage_context",
            {"passage_id": "33333333-3333-3333-3333-333333333333"},
            "mcp-get-passage-context.json",
            PassageContextResponse,
        ),
        (
            "build_context_pack",
            {"query": "phase 4"},
            "mcp-build-context-pack.json",
            ContextPackResponse,
        ),
    ],
)
def test_mcp_tools_match_golden_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tool_name: str,
    arguments: dict[str, object],
    golden_name: str,
    schema,
) -> None:
    with _build_client(monkeypatch, tmp_path) as client:
        payload = _call_tool(client, name=tool_name, arguments=arguments)

    expected = _golden_payload(golden_name)
    assert payload == expected
    assert schema.model_validate(payload).model_dump(mode="json") == expected
