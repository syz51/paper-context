from __future__ import annotations

import importlib
import json
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from paper_context.schemas.api import (
    ContextPackResponse,
    DocumentListResponse,
    DocumentOutlineNode,
    DocumentOutlineResponse,
    DocumentResult,
    PassageContextResponse,
    PassageResultModel,
    TableDetailResponse,
    TablePreviewModel,
    TableResultModel,
)
from paper_context.schemas.public import (
    ContextPackProvenanceModel,
    ContextPassageModel,
    ParentSectionResultModel,
    PassageContextTarget,
)

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
    def __init__(
        self,
        *,
        search_documents_response: DocumentListResponse | None = None,
        outline_response: DocumentOutlineResponse | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.search_documents_response = search_documents_response or DocumentListResponse()
        self.outline_response = outline_response or DocumentOutlineResponse(
            document_id=UUID("11111111-1111-1111-1111-111111111111"),
            title="Phase 4 paper",
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
        self.calls.append(
            {
                "query": query,
                "filters": filters,
                "cursor": cursor,
                "limit": limit,
            }
        )
        return self.search_documents_response

    def get_document_outline(self, document_id):
        self.calls.append({"document_id": document_id})
        return self.outline_response


class _RetrievalServiceStub:
    def __init__(
        self,
        *,
        search_passages_response: SimpleNamespace | None = None,
        search_tables_response: SimpleNamespace | None = None,
        table_response: TableDetailResponse | None = None,
        passage_context_response: PassageContextResponse | None = None,
        context_pack_response: ContextPackResponse | None = None,
    ) -> None:
        self.search_passages_response = search_passages_response or SimpleNamespace(
            items=(),
            next_cursor=None,
            exact=True,
            truncated=False,
            warnings=(),
        )
        self.search_tables_response = search_tables_response or SimpleNamespace(
            items=(),
            next_cursor=None,
            exact=True,
            truncated=False,
            warnings=(),
        )
        self.table_response = table_response
        self.passage_context_response = passage_context_response
        self.context_pack_response = context_pack_response
        self.search_passages_calls: list[dict[str, object]] = []
        self.search_tables_calls: list[dict[str, object]] = []
        self.get_table_calls: list[UUID] = []
        self.get_passage_context_calls: list[dict[str, object]] = []
        self.build_context_pack_calls: list[dict[str, object]] = []

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
        self.search_passages_calls.append(
            {
                "query": query,
                "filters": filters,
                "cursor": cursor,
                "limit": limit,
                "pagination_mode": pagination_mode,
                "max_rerank_candidates": max_rerank_candidates,
                "max_expansion_rounds": max_expansion_rounds,
            }
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
        self.search_tables_calls.append(
            {
                "query": query,
                "filters": filters,
                "cursor": cursor,
                "limit": limit,
                "pagination_mode": pagination_mode,
                "max_rerank_candidates": max_rerank_candidates,
                "max_expansion_rounds": max_expansion_rounds,
            }
        )
        return self.search_tables_response

    def get_table(self, *, table_id):
        self.get_table_calls.append(table_id)
        return self.table_response

    def get_passage_context(self, *, passage_id, before: int = 1, after: int = 1):
        self.get_passage_context_calls.append(
            {"passage_id": passage_id, "before": before, "after": after}
        )
        return self.passage_context_response

    def build_context_pack(
        self,
        *,
        query: str,
        filters,
        cursor: str | None = None,
        limit: int = 8,
    ):
        self.build_context_pack_calls.append(
            {"query": query, "filters": filters, "cursor": cursor, "limit": limit}
        )
        return self.context_pack_response


def _sse_json(response_text: str) -> dict[str, Any]:
    for line in response_text.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"missing SSE data line in response: {response_text!r}")


def _load_app_modules(monkeypatch: pytest.MonkeyPatch):
    observability_module = importlib.import_module("paper_context.observability")
    monkeypatch.setattr(
        observability_module,
        "observe_operation",
        lambda *args, **kwargs: nullcontext(),
        raising=False,
    )
    monkeypatch.setattr(
        observability_module,
        "get_metrics_registry",
        lambda: SimpleNamespace(timing_snapshots=lambda limit=20: []),
        raising=False,
    )
    api_app_module = importlib.import_module("paper_context.api.app")
    mcp_server_module = importlib.import_module("paper_context.mcp.server")
    return api_app_module, mcp_server_module


def test_mcp_http_app_mounts_and_runs_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_app_module, _ = _load_app_modules(monkeypatch)
    fake_mcp_app = FakeMcpApp()
    settings = SimpleNamespace(
        log_level="INFO",
        storage=SimpleNamespace(root_path="."),
        queue=SimpleNamespace(name="document_ingest"),
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

    with TestClient(api_app_module.create_app()) as client:
        response = client.get("/mcp")

    assert response.text == "ok"
    assert fake_mcp_app.events == ["lifespan-enter", "lifespan-exit"]


def test_mcp_mount_serves_real_streamable_http_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_app_module, mcp_server_module = _load_app_modules(monkeypatch)
    from paper_context.retrieval import RetrievalFilters

    document_id = UUID("11111111-1111-1111-1111-111111111111")
    section_id = UUID("33333333-3333-3333-3333-333333333333")
    table_id = UUID("22222222-2222-2222-2222-222222222222")
    passage_id = UUID("66666666-6666-6666-6666-666666666666")
    run_id = UUID("44444444-4444-4444-4444-444444444444")
    documents = _DocumentsServiceStub(
        search_documents_response=DocumentListResponse(
            documents=[
                DocumentResult(
                    document_id=document_id,
                    title="Phase 4 paper",
                    authors=["Ada Lovelace", "Grace Hopper"],
                    publication_year=2025,
                    quant_tags={"asset_universe": "rates"},
                    current_status="ready",
                    active_index_version="mvp-v2",
                )
            ],
            next_cursor="documents-cursor-1",
        ),
        outline_response=DocumentOutlineResponse(
            document_id=document_id,
            title="Phase 4 paper",
            sections=[
                DocumentOutlineNode(
                    section_id=section_id,
                    heading="Methods",
                    section_path=["Methods"],
                    ordinal=1,
                    page_start=1,
                    page_end=2,
                    children=[
                        DocumentOutlineNode(
                            section_id=UUID("77777777-7777-7777-7777-777777777777"),
                            parent_section_id=section_id,
                            heading="Calibration",
                            section_path=["Methods", "Calibration"],
                            ordinal=2,
                            page_start=2,
                            page_end=3,
                            children=[],
                        )
                    ],
                )
            ],
        ),
    )
    passage_item = SimpleNamespace(
        passage_id=passage_id,
        document_id=document_id,
        section_id=section_id,
        document_title="Phase 4 paper",
        section_path=("Methods", "Calibration"),
        text="Contextualized passage",
        score=0.94,
        retrieval_modes=("sparse", "dense"),
        page_start=2,
        page_end=3,
        index_version="mvp-v2",
        retrieval_index_run_id=run_id,
        parser_source="docling",
        warnings=("parser_fallback_used",),
    )
    table_preview = TablePreviewModel(
        headers=["Horizon", "Loss"],
        rows=[["1m", "0.5"], ["3m", "0.7"]],
        row_count=2,
    )
    table_item = SimpleNamespace(
        table_id=table_id,
        document_id=document_id,
        section_id=section_id,
        document_title="Phase 4 paper",
        section_path=("Results",),
        caption="Loss by horizon",
        table_type="lexical",
        preview=table_preview,
        score=0.91,
        retrieval_modes=("lexical",),
        page_start=4,
        page_end=4,
        index_version="mvp-v2",
        retrieval_index_run_id=run_id,
        parser_source="pdfplumber",
        warnings=("metadata_low_confidence",),
    )
    passage_context_response = PassageContextResponse(
        passage=PassageContextTarget(
            passage_id=passage_id,
            document_id=document_id,
            section_id=section_id,
            document_title="Phase 4 paper",
            section_path=["Methods", "Calibration"],
            text="Contextualized passage",
            chunk_ordinal=1,
            page_start=2,
            page_end=3,
            index_version="mvp-v2",
            retrieval_index_run_id=run_id,
            parser_source="docling",
            warnings=["parser_fallback_used"],
        ),
        context_passages=[
            ContextPassageModel(
                passage_id=UUID("88888888-8888-8888-8888-888888888888"),
                text="Previous context",
                chunk_ordinal=0,
                page_start=1,
                page_end=2,
                relationship="before",
            ),
            ContextPassageModel(
                passage_id=UUID("99999999-9999-9999-9999-999999999999"),
                text="Next context",
                chunk_ordinal=2,
                page_start=3,
                page_end=4,
                relationship="after",
            ),
        ],
        warnings=["parent_context_truncated"],
    )
    context_pack_response = ContextPackResponse(
        context_pack_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        query="delta",
        passages=[
            PassageResultModel(
                passage_id=passage_id,
                document_id=document_id,
                section_id=section_id,
                document_title="Phase 4 paper",
                section_path=["Methods", "Calibration"],
                text="Contextualized passage",
                score=0.94,
                retrieval_modes=["sparse", "dense"],
                page_start=2,
                page_end=3,
                index_version="mvp-v2",
                retrieval_index_run_id=run_id,
                parser_source="docling",
                warnings=["parser_fallback_used"],
            )
        ],
        tables=[
            TableResultModel(
                table_id=table_id,
                document_id=document_id,
                section_id=section_id,
                document_title="Phase 4 paper",
                section_path=["Results"],
                caption="Loss by horizon",
                table_type="lexical",
                preview=table_preview,
                score=0.91,
                retrieval_modes=["lexical"],
                page_start=4,
                page_end=4,
                index_version="mvp-v2",
                retrieval_index_run_id=run_id,
                parser_source="pdfplumber",
                warnings=["metadata_low_confidence"],
            )
        ],
        parent_sections=[
            ParentSectionResultModel(
                section_id=section_id,
                document_id=document_id,
                document_title="Phase 4 paper",
                heading="Methods",
                section_path=["Methods"],
                page_start=1,
                page_end=2,
                supporting_passages=[
                    ContextPassageModel(
                        passage_id=passage_id,
                        text="Contextualized passage",
                        chunk_ordinal=1,
                        page_start=2,
                        page_end=3,
                        relationship="seed",
                    )
                ],
                warnings=["parent_context_truncated"],
            )
        ],
        documents=[
            DocumentResult(
                document_id=document_id,
                title="Phase 4 paper",
                authors=["Ada Lovelace", "Grace Hopper"],
                publication_year=2025,
                quant_tags={"asset_universe": "rates"},
                current_status="ready",
                active_index_version="mvp-v2",
            )
        ],
        provenance=ContextPackProvenanceModel(
            active_index_version="mvp-v2",
            retrieval_index_run_ids=[run_id],
            retrieval_modes=["sparse", "dense", "lexical"],
        ),
        warnings=["parent_context_truncated", "parser_fallback_used"],
        next_cursor="context-cursor-1",
    )
    retrieval = _RetrievalServiceStub(
        search_passages_response=SimpleNamespace(
            items=(passage_item,),
            next_cursor="passages-cursor-1",
            exact=True,
            truncated=False,
            warnings=(),
        ),
        search_tables_response=SimpleNamespace(
            items=(table_item,),
            next_cursor="tables-cursor-1",
            exact=True,
            truncated=False,
            warnings=(),
        ),
        table_response=TableDetailResponse(
            table_id=table_id,
            document_id=document_id,
            section_id=section_id,
            document_title="Phase 4 paper",
            section_path=["Results"],
            caption="Loss by horizon",
            table_type="lexical",
            headers=["Horizon", "Loss"],
            rows=[["1m", "0.5"], ["3m", "0.7"]],
            row_count=2,
            page_start=4,
            page_end=4,
            index_version="mvp-v2",
            retrieval_index_run_id=run_id,
            parser_source="pdfplumber",
            warnings=["metadata_low_confidence"],
        ),
        passage_context_response=passage_context_response,
        context_pack_response=context_pack_response,
    )
    assert retrieval.table_response is not None
    assert retrieval.context_pack_response is not None
    settings = SimpleNamespace(
        log_level="INFO",
        storage=SimpleNamespace(root_path="."),
        queue=SimpleNamespace(name="document_ingest"),
    )
    monkeypatch.setattr(
        api_app_module,
        "create_http_app",
        lambda: mcp_server_module.create_http_app(
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

    with TestClient(api_app_module.create_app()) as client:
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
        tool_cases = [
            (
                2,
                "search_documents",
                {"query": "alpha", "limit": 999},
                documents.search_documents_response.model_dump(mode="json"),
            ),
            (
                3,
                "get_document_outline",
                {"document_id": str(document_id)},
                documents.outline_response.model_dump(mode="json"),
            ),
            (
                4,
                "search_passages",
                {"query": "beta", "limit": 999},
                {
                    "query": "beta",
                    "passages": [
                        {
                            "passage_id": str(passage_id),
                            "document_id": str(document_id),
                            "section_id": str(section_id),
                            "document_title": "Phase 4 paper",
                            "section_path": ["Methods", "Calibration"],
                            "text": "Contextualized passage",
                            "score": 0.94,
                            "retrieval_modes": ["sparse", "dense"],
                            "page_start": 2,
                            "page_end": 3,
                            "index_version": "mvp-v2",
                            "retrieval_index_run_id": str(run_id),
                            "parser_source": "docling",
                            "warnings": ["parser_fallback_used"],
                        }
                    ],
                    "next_cursor": "passages-cursor-1",
                    "exact": True,
                    "truncated": False,
                    "warnings": [],
                },
            ),
            (
                5,
                "search_tables",
                {"query": "gamma", "limit": 999},
                {
                    "query": "gamma",
                    "tables": [
                        {
                            "table_id": str(table_id),
                            "document_id": str(document_id),
                            "section_id": str(section_id),
                            "document_title": "Phase 4 paper",
                            "section_path": ["Results"],
                            "caption": "Loss by horizon",
                            "table_type": "lexical",
                            "preview": {
                                "headers": ["Horizon", "Loss"],
                                "rows": [["1m", "0.5"], ["3m", "0.7"]],
                                "row_count": 2,
                            },
                            "score": 0.91,
                            "retrieval_modes": ["lexical"],
                            "page_start": 4,
                            "page_end": 4,
                            "index_version": "mvp-v2",
                            "retrieval_index_run_id": str(run_id),
                            "parser_source": "pdfplumber",
                            "warnings": ["metadata_low_confidence"],
                        }
                    ],
                    "next_cursor": "tables-cursor-1",
                    "exact": True,
                    "truncated": False,
                    "warnings": [],
                },
            ),
            (
                6,
                "get_table",
                {"table_id": str(table_id)},
                retrieval.table_response.model_dump(mode="json"),
            ),
            (
                7,
                "get_passage_context",
                {"passage_id": str(passage_id), "before": 2, "after": 3},
                passage_context_response.model_dump(mode="json"),
            ),
            (
                8,
                "build_context_pack",
                {"query": "delta", "limit": 999},
                retrieval.context_pack_response.model_dump(mode="json"),
            ),
        ]
        responses: list[dict[str, object]] = []
        for request_id, tool_name, arguments, expected in tool_cases:
            response = client.post(
                "/mcp",
                headers={**headers, "mcp-session-id": session_id or ""},
                json={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                },
            )
            payload = _sse_json(response.text)
            responses.append(payload)
            assert response.status_code == 200
            assert payload["result"]["isError"] is False
            assert payload["result"]["structuredContent"] == expected
            assert json.loads(payload["result"]["content"][0]["text"]) == expected

    initialize_payload = _sse_json(initialize.text)
    assert len(responses) == 7

    assert initialize.status_code == 200
    assert session_id
    assert initialize_payload["result"]["serverInfo"]["name"] == "paper-context"
    assert documents.calls == [
        {"query": "alpha", "filters": None, "cursor": None, "limit": 100},
        {"document_id": document_id},
    ]
    assert retrieval.search_passages_calls == [
        {
            "query": "beta",
            "filters": RetrievalFilters(),
            "cursor": None,
            "limit": 8,
            "pagination_mode": "exact",
            "max_rerank_candidates": None,
            "max_expansion_rounds": None,
        }
    ]
    assert retrieval.search_tables_calls == [
        {
            "query": "gamma",
            "filters": RetrievalFilters(),
            "cursor": None,
            "limit": 5,
            "pagination_mode": "exact",
            "max_rerank_candidates": None,
            "max_expansion_rounds": None,
        }
    ]
    assert retrieval.get_table_calls == [table_id]
    assert retrieval.get_passage_context_calls == [
        {"passage_id": passage_id, "before": 2, "after": 3}
    ]
    assert retrieval.build_context_pack_calls == [
        {"query": "delta", "filters": RetrievalFilters(), "cursor": None, "limit": 8}
    ]


def test_mcp_transport_surfaces_not_found_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    api_app_module, mcp_server_module = _load_app_modules(monkeypatch)
    documents = _DocumentsServiceStub()
    retrieval = _RetrievalServiceStub(table_response=None)
    settings = SimpleNamespace(
        log_level="INFO",
        storage=SimpleNamespace(root_path="."),
        queue=SimpleNamespace(name="document_ingest"),
    )
    monkeypatch.setattr(
        api_app_module,
        "create_http_app",
        lambda: mcp_server_module.create_http_app(
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

    with TestClient(api_app_module.create_app()) as client:
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
        response = client.post(
            "/mcp",
            headers={**headers, "mcp-session-id": session_id or ""},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "get_table",
                    "arguments": {"table_id": "22222222-2222-2222-2222-222222222222"},
                },
            },
        )

    payload = _sse_json(response.text)

    assert initialize.status_code == 200
    assert response.status_code == 200
    assert payload["result"]["isError"] is True
    assert (
        payload["result"]["content"][0]["text"] == "Error calling tool 'get_table': table not found"
    )
