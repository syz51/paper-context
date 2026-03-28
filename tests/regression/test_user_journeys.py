from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import Engine

from paper_context.api.app import create_app
from paper_context.config import get_settings
from paper_context.db.engine import dispose_engine, get_engine
from paper_context.db.session import connection_scope
from paper_context.ingestion.enrichment import NullMetadataEnricher
from paper_context.ingestion.service import DeterministicIngestProcessor
from paper_context.ingestion.types import (
    GateStatus,
    ParsedDocument,
    ParsedParagraph,
    ParsedSection,
    ParsedTable,
    ParserArtifact,
    ParserResult,
)
from paper_context.pagination import encode_cursor, fingerprint_payload
from paper_context.queue.contracts import IngestionQueue
from paper_context.storage.local_fs import LocalFilesystemStorage
from paper_context.worker.loop import IngestWorker, WorkerConfig

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_postgres,
    pytest.mark.slow,
]

_PDF_BYTES = b"%PDF-1.4\nregression\n"
_MCP_HEADERS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}


@dataclass(frozen=True)
class _RegressionRuntime:
    engine: Engine
    queue_name: str
    storage_root: Path


class _StaticParser:
    def __init__(self, result: ParserResult) -> None:
        self.name = result.artifact.parser
        self.result = result
        self.calls: list[str] = []

    def parse(
        self, filename: str, content: bytes | None = None, *, source_path: Path | None = None
    ) -> ParserResult:
        del content, source_path
        self.calls.append(filename)
        return self.result


class _McpSession:
    def __init__(self, client: TestClient) -> None:
        initialize_request_id = 1
        initialize = client.post(
            "/mcp",
            headers=_MCP_HEADERS,
            json={
                "jsonrpc": "2.0",
                "id": initialize_request_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "regression-test", "version": "0"},
                },
            },
        )
        payload = _sse_json(initialize.text, expected_id=initialize_request_id)
        assert initialize.status_code == 200
        assert payload["result"]["serverInfo"]["name"] == "paper-context"
        self._client = client
        self._session_id = initialize.headers["mcp-session-id"]
        self._next_id = 2

    def call_tool(
        self,
        *,
        name: str,
        arguments: dict[str, object],
        expect_error: bool = False,
    ) -> Any:
        request_id = self._next_id
        response = self._client.post(
            "/mcp",
            headers={**_MCP_HEADERS, "mcp-session-id": self._session_id},
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        self._next_id += 1
        payload = _sse_json(response.text, expected_id=request_id)
        assert response.status_code == 200
        assert payload["result"]["isError"] is expect_error
        if expect_error:
            return payload["result"]["content"][0]["text"]
        return payload["result"]["structuredContent"]


@pytest.fixture
def regression_runtime(
    monkeypatch: pytest.MonkeyPatch,
    migrated_postgres_engine: Engine,
    migrated_postgres_url: str,
    unique_queue_name: str,
    tmp_path: Path,
) -> Iterator[_RegressionRuntime]:
    with migrated_postgres_engine.begin() as connection:
        connection.execute(
            text("SELECT pgmq.create(:queue_name)"),
            {"queue_name": unique_queue_name},
        )

    storage_root = tmp_path / "artifacts"
    monkeypatch.setenv("PAPER_CONTEXT_DATABASE__URL", migrated_postgres_url)
    monkeypatch.setenv("PAPER_CONTEXT_STORAGE__ROOT_PATH", str(storage_root))
    monkeypatch.setenv("PAPER_CONTEXT_QUEUE__NAME", unique_queue_name)
    monkeypatch.setenv("PAPER_CONTEXT_PROVIDERS__INDEX_VERSION", "mvp-v1")
    get_settings.cache_clear()
    get_engine.cache_clear()

    try:
        yield _RegressionRuntime(
            engine=migrated_postgres_engine,
            queue_name=unique_queue_name,
            storage_root=storage_root,
        )
    finally:
        dispose_engine()
        get_settings.cache_clear()
        get_engine.cache_clear()


def _sse_json(response_text: str, *, expected_id: int | None = None) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for line in response_text.splitlines():
        if line.startswith("data: "):
            payloads.append(json.loads(line.removeprefix("data: ")))
    if expected_id is not None:
        for payload in reversed(payloads):
            if payload.get("id") == expected_id:
                return payload
    if not payloads:
        raise AssertionError(f"missing SSE payload: {response_text!r}")
    return payloads[-1]


def _make_parsed_document(
    *,
    title: str,
    keyword: str,
    table_caption: str,
    publication_year: int = 2024,
    metadata_confidence: float = 0.91,
    paragraph_count: int = 2,
    table_count: int = 1,
) -> ParsedDocument:
    paragraphs = [
        ParsedParagraph(
            text=(
                f"{keyword} passage {index + 1} preserves stable narrative tokens for "
                "end to end regression retrieval."
            ),
            page_start=index + 1,
            page_end=index + 1,
            provenance_offsets={"pages": [index + 1], "charspans": [[0, 64]]},
        )
        for index in range(paragraph_count)
    ]
    tables = [
        ParsedTable(
            section_key="methods",
            caption=table_caption if table_count == 1 else f"{table_caption} {index + 1}",
            headers=["metric", "value"],
            rows=[[f"{keyword} ratio {index + 1}", "1.0"], ["samples", "42"]],
            page_start=paragraph_count + index + 1,
            page_end=paragraph_count + index + 1,
        )
        for index in range(table_count)
    ]
    return ParsedDocument(
        title=title,
        authors=["Ada Lovelace"],
        abstract=f"{keyword} abstract for regression coverage.",
        publication_year=publication_year,
        metadata_confidence=metadata_confidence,
        sections=[
            ParsedSection(
                key="methods",
                heading="Methods",
                heading_path=["Methods"],
                level=1,
                page_start=1,
                page_end=max(paragraph_count, 1),
                paragraphs=paragraphs,
            )
        ],
        tables=tables,
        references=[],
    )


def _make_parser_result(
    gate_status: GateStatus = "pass",
    *,
    parser_name: str = "docling",
    parsed_document: ParsedDocument | None = None,
) -> ParserResult:
    is_failure = gate_status == "fail"
    return ParserResult(
        gate_status=gate_status,
        parsed_document=None if is_failure else parsed_document,
        artifact=ParserArtifact(
            artifact_type=f"{parser_name}_parse",
            parser=parser_name,
            filename=f"{parser_name}.json",
            content=b"{}",
        ),
        warnings=["reduced_structure_confidence"] if gate_status == "degraded" else [],
        failure_code=f"{parser_name}_failed" if is_failure else None,
        failure_message=f"{parser_name} failed" if is_failure else None,
    )


def _make_paginated_document(
    *,
    title: str,
    passage_keyword: str,
    table_keyword: str,
    paragraph_count: int,
    table_count: int,
) -> ParsedDocument:
    paragraphs = []
    for index in range(paragraph_count):
        keyword_repeats = 5 if index == 0 else 4 if index == 1 else 1
        paragraphs.append(
            ParsedParagraph(
                text=(
                    ((passage_keyword + " ") * keyword_repeats)
                    + f"ranked passage {index + 1} preserves deterministic pagination."
                ).strip(),
                page_start=index + 1,
                page_end=index + 1,
                provenance_offsets={"pages": [index + 1], "charspans": [[0, 64]]},
            )
        )

    tables = []
    for index in range(table_count):
        keyword_repeats = 5 if index == 0 else 4 if index == 1 else 1
        tables.append(
            ParsedTable(
                section_key="methods",
                caption=(
                    ((table_keyword + " ") * keyword_repeats) + f"ranked metrics {index + 1}"
                ).strip(),
                headers=["metric", "value"],
                rows=[[f"row {index + 1}", str(index + 1)], ["samples", "42"]],
                page_start=paragraph_count + index + 1,
                page_end=paragraph_count + index + 1,
            )
        )

    return ParsedDocument(
        title=title,
        authors=["Ada Lovelace"],
        abstract=f"{passage_keyword} and {table_keyword} pagination regression coverage.",
        publication_year=2024,
        metadata_confidence=0.91,
        sections=[
            ParsedSection(
                key="methods",
                heading="Methods",
                heading_path=["Methods"],
                level=1,
                page_start=1,
                page_end=max(paragraph_count, 1),
                paragraphs=paragraphs,
            )
        ],
        tables=tables,
        references=[],
    )


def _build_processor(
    storage_root: Path,
    *,
    primary_result: ParserResult,
    fallback_result: ParserResult | None = None,
) -> tuple[DeterministicIngestProcessor, _StaticParser, _StaticParser]:
    storage = LocalFilesystemStorage(storage_root)
    storage.ensure_root()
    primary_parser = _StaticParser(primary_result)
    fallback_parser = _StaticParser(
        fallback_result
        or _make_parser_result(
            parsed_document=_make_parsed_document(
                title="Fallback paper",
                keyword="fallback beacon",
                table_caption="fallback beacon metrics",
            ),
            parser_name="pdfplumber",
        )
    )
    processor = DeterministicIngestProcessor(
        storage=storage,
        primary_parser=primary_parser,
        fallback_parser=fallback_parser,
        metadata_enricher=NullMetadataEnricher(),
        index_version="mvp-v1",
        chunking_version="phase1",
        embedding_model="voyage-4-large",
        reranker_model="zerank-2",
        min_tokens=1,
        max_tokens=12,
        overlap_fraction=0.0,
    )
    return processor, primary_parser, fallback_parser


def _run_worker(
    runtime: _RegressionRuntime,
    *,
    primary_result: ParserResult,
    fallback_result: ParserResult | None = None,
):
    queue = IngestionQueue(runtime.queue_name)
    processor, primary_parser, fallback_parser = _build_processor(
        runtime.storage_root,
        primary_result=primary_result,
        fallback_result=fallback_result,
    )
    worker = IngestWorker(
        connection_factory=lambda: connection_scope(runtime.engine),
        queue_adapter=queue,
        processor=processor,
        config=WorkerConfig(vt_seconds=30, max_poll_seconds=1, poll_interval_ms=10),
    )
    handled = worker.run_once()
    assert handled is not None
    return handled, primary_parser, fallback_parser


def _upload_document(client: TestClient, *, title: str) -> dict[str, object]:
    response = client.post(
        "/documents",
        data={"title": title},
        files={"file": ("paper.pdf", _PDF_BYTES, "application/pdf")},
    )
    assert response.status_code == 201
    return response.json()


def _replace_document(client: TestClient, *, document_id: UUID, title: str) -> dict[str, object]:
    response = client.post(
        f"/documents/{document_id}/replace",
        data={"title": title},
        files={"file": ("replacement.pdf", _PDF_BYTES, "application/pdf")},
    )
    assert response.status_code == 202
    return response.json()


def test_operational_and_validation_journeys(regression_runtime: _RegressionRuntime) -> None:
    with TestClient(create_app()) as client:
        health = client.get("/healthz")
        readiness = client.get("/readyz")
        empty_upload = client.post(
            "/documents",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        non_pdf_upload = client.post(
            "/documents",
            files={"file": ("note.txt", b"plain text", "text/plain")},
        )
        missing_replace = client.post(
            f"/documents/{uuid4()}/replace",
            files={"file": ("paper.pdf", _PDF_BYTES, "application/pdf")},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "ready"
    assert readiness.json()["queue_name"] == regression_runtime.queue_name
    assert empty_upload.status_code == 400
    assert empty_upload.json()["detail"] == "uploaded file is empty"
    assert non_pdf_upload.status_code == 400
    assert non_pdf_upload.json()["detail"] == "uploaded file must be a PDF"
    assert missing_replace.status_code == 404
    assert missing_replace.json()["detail"] == "document not found"


def test_upload_limit_rejection_returns_413(
    regression_runtime: _RegressionRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del regression_runtime
    monkeypatch.setenv("PAPER_CONTEXT_UPLOAD__MAX_BYTES", "8")
    get_settings.cache_clear()

    try:
        with TestClient(create_app()) as client:
            response = client.post(
                "/documents",
                files={"file": ("paper.pdf", _PDF_BYTES, "application/pdf")},
            )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 413
    assert response.json()["detail"] == "uploaded file exceeds the 8-byte limit"


def test_readiness_turns_degraded_when_storage_root_disappears(
    regression_runtime: _RegressionRuntime,
) -> None:
    with TestClient(create_app()) as client:
        shutil.rmtree(regression_runtime.storage_root)
        readiness = client.get("/readyz")

    assert readiness.status_code == 200
    assert readiness.json()["status"] == "degraded"
    assert readiness.json()["storage_ready"] is False
    assert readiness.json()["queue_ready"] is True


def test_upload_to_ready_http_and_mcp_journey(regression_runtime: _RegressionRuntime) -> None:
    parsed_document = _make_parsed_document(
        title="Journey Regression Paper",
        keyword="journeybeacon",
        table_caption="journeybeacon metrics",
    )

    with TestClient(create_app()) as client:
        upload = _upload_document(client, title="Queued upload title")
        document_id = UUID(str(upload["document_id"]))
        ingest_job_id = UUID(str(upload["ingest_job_id"]))

        queued_job = client.get(f"/ingest-jobs/{ingest_job_id}")
        assert queued_job.status_code == 200
        assert queued_job.json()["status"] == "queued"

        handled, _, fallback_parser = _run_worker(
            regression_runtime,
            primary_result=_make_parser_result(
                "degraded",
                parser_name="docling",
                parsed_document=parsed_document,
            ),
            fallback_result=_make_parser_result(
                parsed_document=parsed_document,
                parser_name="pdfplumber",
            ),
        )

        assert handled.payload.document_id == document_id
        assert handled.payload.ingest_job_id == ingest_job_id
        assert len(fallback_parser.calls) == 1

        ready_job = client.get(f"/ingest-jobs/{ingest_job_id}")
        documents = client.get("/documents")
        detail = client.get(f"/documents/{document_id}")
        outline = client.get(f"/documents/{document_id}/outline")
        tables = client.get(f"/documents/{document_id}/tables")
        mcp = _McpSession(client)
        mcp_documents = mcp.call_tool(
            name="search_documents",
            arguments={"query": "Journey Regression Paper"},
        )
        mcp_outline = mcp.call_tool(
            name="get_document_outline",
            arguments={"document_id": str(document_id)},
        )
        mcp_passages = mcp.call_tool(
            name="search_passages",
            arguments={"query": "journeybeacon", "limit": 8},
        )
        mcp_tables = mcp.call_tool(
            name="search_tables",
            arguments={"query": "journeybeacon metrics", "limit": 5},
        )

        passage_id = str(mcp_passages["passages"][0]["passage_id"])
        table_id = str(mcp_tables["tables"][0]["table_id"])
        passage_context = mcp.call_tool(
            name="get_passage_context",
            arguments={"passage_id": passage_id, "before": 0, "after": 1},
        )
        table_detail = mcp.call_tool(
            name="get_table",
            arguments={"table_id": table_id},
        )
        context_pack = mcp.call_tool(
            name="build_context_pack",
            arguments={"query": "journeybeacon", "limit": 8},
        )

    ready_job_payload = ready_job.json()
    documents_payload = documents.json()
    detail_payload = detail.json()
    outline_payload = outline.json()
    tables_payload = tables.json()

    assert ready_job.status_code == 200
    assert ready_job_payload["status"] == "ready"
    assert "parser_fallback_used" in ready_job_payload["warnings"]
    assert documents.status_code == 200
    assert documents_payload["documents"][0]["document_id"] == str(document_id)
    assert documents_payload["documents"][0]["active_index_version"] == "mvp-v1"
    assert detail.status_code == 200
    assert detail_payload["title"] == "Journey Regression Paper"
    assert detail_payload["current_status"] == "ready"
    assert outline.status_code == 200
    assert outline_payload["sections"][0]["heading"] == "Methods"
    assert tables.status_code == 200
    assert tables_payload["tables"][0]["caption"] == "journeybeacon metrics"
    assert mcp_documents["documents"][0]["document_id"] == str(document_id)
    assert mcp_outline["document_id"] == str(document_id)
    assert mcp_passages["passages"]
    assert mcp_passages["passages"][0]["document_id"] == str(document_id)
    assert mcp_passages["passages"][0]["parser_source"] == "pdfplumber"
    assert "parser_fallback_used" in mcp_passages["passages"][0]["warnings"]
    assert mcp_tables["tables"]
    assert mcp_tables["tables"][0]["caption"] == "journeybeacon metrics"
    assert table_detail["table_id"] == table_id
    assert table_detail["caption"] == "journeybeacon metrics"
    assert passage_context["passage"]["passage_id"] == passage_id
    assert passage_context["context_passages"][0]["relationship"] == "selected"
    assert "parser_fallback_used" in passage_context["warnings"]
    assert context_pack["passages"]
    assert context_pack["tables"]
    assert context_pack["documents"][0]["document_id"] == str(document_id)
    assert context_pack["provenance"]["active_index_version"] == "mvp-v1"
    assert "parser_fallback_used" in context_pack["warnings"]


def test_document_listing_and_search_journeys_cover_filters_and_cursors(
    regression_runtime: _RegressionRuntime,
) -> None:
    first_document = _make_parsed_document(
        title="Filter Regression Alpha",
        keyword="filterbeacon-alpha",
        table_caption="filterbeacon metrics alpha",
        publication_year=2024,
    )
    second_document = _make_parsed_document(
        title="Filter Regression Beta",
        keyword="filterbeacon-beta",
        table_caption="filterbeacon metrics beta",
        publication_year=2025,
    )

    with TestClient(create_app()) as client:
        first_upload = _upload_document(client, title="Alpha upload title")
        second_upload = _upload_document(client, title="Beta upload title")
        first_document_id = UUID(str(first_upload["document_id"]))
        second_document_id = UUID(str(second_upload["document_id"]))

        first_handled, _, _ = _run_worker(
            regression_runtime,
            primary_result=_make_parser_result(parsed_document=first_document),
        )
        second_handled, _, _ = _run_worker(
            regression_runtime,
            primary_result=_make_parser_result(parsed_document=second_document),
        )

        assert first_handled.payload.document_id == first_document_id
        assert second_handled.payload.document_id == second_document_id

        first_page = client.get("/documents", params={"limit": 1})
        first_cursor = first_page.json()["next_cursor"]
        second_page = client.get("/documents", params={"limit": 1, "cursor": first_cursor})

        mcp = _McpSession(client)
        filtered_documents = mcp.call_tool(
            name="search_documents",
            arguments={
                "query": "   ",
                "filters": {
                    "document_ids": [str(first_document_id), str(second_document_id)],
                    "publication_years": [2024],
                },
                "limit": 5,
            },
        )
        search_first_page = mcp.call_tool(
            name="search_documents",
            arguments={"query": "Filter Regression", "limit": 1},
        )
        search_second_page = mcp.call_tool(
            name="search_documents",
            arguments={
                "query": "Filter Regression",
                "limit": 1,
                "cursor": search_first_page["next_cursor"],
            },
        )

    assert first_page.status_code == 200
    assert second_page.status_code == 200
    listed_document_ids = {
        first_page.json()["documents"][0]["document_id"],
        second_page.json()["documents"][0]["document_id"],
    }
    assert listed_document_ids == {str(first_document_id), str(second_document_id)}
    assert second_page.json()["next_cursor"] is None
    assert [document["document_id"] for document in filtered_documents["documents"]] == [
        str(first_document_id)
    ]
    searched_document_ids = {
        search_first_page["documents"][0]["document_id"],
        search_second_page["documents"][0]["document_id"],
    }
    assert searched_document_ids == {str(first_document_id), str(second_document_id)}
    assert search_second_page["next_cursor"] is None


def test_mcp_pagination_modes_and_cursor_errors_regressions(
    regression_runtime: _RegressionRuntime,
) -> None:
    passage_query = "pagebeacon"
    table_query = "tablebeacon"
    paginated_document = _make_paginated_document(
        title="Pagination Regression Paper",
        passage_keyword=passage_query,
        table_keyword=table_query,
        paragraph_count=41,
        table_count=24,
    )

    with TestClient(create_app()) as client:
        upload = _upload_document(client, title="Pagination upload title")
        document_id = UUID(str(upload["document_id"]))
        handled, _, _ = _run_worker(
            regression_runtime,
            primary_result=_make_parser_result(parsed_document=paginated_document),
        )
        assert handled.payload.document_id == document_id

        mcp = _McpSession(client)
        first_passage_page = mcp.call_tool(
            name="search_passages",
            arguments={"query": passage_query, "limit": 1},
        )
        second_passage_page = mcp.call_tool(
            name="search_passages",
            arguments={
                "query": passage_query,
                "limit": 1,
                "cursor": first_passage_page["next_cursor"],
            },
        )
        first_table_page = mcp.call_tool(
            name="search_tables",
            arguments={"query": table_query, "limit": 1},
        )
        second_table_page = mcp.call_tool(
            name="search_tables",
            arguments={
                "query": table_query,
                "limit": 1,
                "cursor": first_table_page["next_cursor"],
            },
        )
        bounded_passage_page = mcp.call_tool(
            name="search_passages",
            arguments={
                "query": passage_query,
                "limit": 2,
                "pagination_mode": "bounded",
                "max_rerank_candidates": 3,
                "max_expansion_rounds": 1,
            },
        )
        bounded_table_page = mcp.call_tool(
            name="search_tables",
            arguments={
                "query": table_query,
                "limit": 2,
                "pagination_mode": "bounded",
                "max_rerank_candidates": 3,
                "max_expansion_rounds": 1,
            },
        )
        fingerprint = fingerprint_payload(
            {
                "kind": "passages",
                "query": passage_query,
                "pagination_mode": "exact",
                "max_rerank_candidates": None,
                "max_expansion_rounds": None,
                "filters": {"document_ids": [], "publication_years": []},
            }
        )
        legacy_cursor = encode_cursor(
            {
                "kind": "passages",
                "fingerprint": fingerprint,
                "index_version": "mvp-v1",
                "score": "3.0",
                "entity_id": str(first_passage_page["passages"][0]["passage_id"]),
            }
        )
        legacy_cursor_error = mcp.call_tool(
            name="search_passages",
            arguments={"query": passage_query, "limit": 1, "cursor": legacy_cursor},
            expect_error=True,
        )

    assert first_passage_page["next_cursor"] is not None
    assert (
        second_passage_page["passages"][0]["passage_id"]
        != first_passage_page["passages"][0]["passage_id"]
    )
    assert first_table_page["next_cursor"] is not None
    assert second_table_page["tables"][0]["table_id"] != first_table_page["tables"][0]["table_id"]
    assert bounded_passage_page["exact"] is False
    assert bounded_passage_page["truncated"] is True
    assert "bounded_pagination_truncated" in bounded_passage_page["warnings"]
    assert bounded_table_page["exact"] is False
    assert bounded_table_page["truncated"] is True
    assert "bounded_pagination_truncated" in bounded_table_page["warnings"]
    assert "cursor is no longer supported" in legacy_cursor_error


def test_successful_replacement_switches_http_and_mcp_reads(
    regression_runtime: _RegressionRuntime,
) -> None:
    initial_document = _make_parsed_document(
        title="Original Regression Paper",
        keyword="origamiquartz",
        table_caption="origamiquartz metrics",
    )
    replacement_document = _make_parsed_document(
        title="Replacement Regression Paper",
        keyword="nebulafjord",
        table_caption="nebulafjord metrics",
    )

    with TestClient(create_app()) as client:
        upload = _upload_document(client, title="Initial upload title")
        document_id = UUID(str(upload["document_id"]))
        initial_job_id = UUID(str(upload["ingest_job_id"]))

        initial_handled, _, _ = _run_worker(
            regression_runtime,
            primary_result=_make_parser_result(parsed_document=initial_document),
        )
        assert initial_handled.payload.ingest_job_id == initial_job_id

        initial_mcp = _McpSession(client)
        initial_passages = initial_mcp.call_tool(
            name="search_passages",
            arguments={"query": "origamiquartz", "limit": 8},
        )
        initial_tables = initial_mcp.call_tool(
            name="search_tables",
            arguments={"query": "origamiquartz metrics", "limit": 5},
        )
        initial_passage_id = str(initial_passages["passages"][0]["passage_id"])
        initial_table_id = str(initial_tables["tables"][0]["table_id"])

        replacement = _replace_document(
            client,
            document_id=document_id,
            title="Replacement upload title",
        )
        replacement_job_id = UUID(str(replacement["ingest_job_id"]))
        queued_replacement_job = client.get(f"/ingest-jobs/{replacement_job_id}")
        assert queued_replacement_job.status_code == 200
        assert queued_replacement_job.json()["status"] == "queued"

        replacement_handled, _, _ = _run_worker(
            regression_runtime,
            primary_result=_make_parser_result(parsed_document=replacement_document),
        )
        assert replacement_handled.payload.ingest_job_id == replacement_job_id

        detail = client.get(f"/documents/{document_id}")
        tables = client.get(f"/documents/{document_id}/tables")
        replacement_job = client.get(f"/ingest-jobs/{replacement_job_id}")
        mcp = _McpSession(client)
        old_document_search = mcp.call_tool(
            name="search_documents",
            arguments={"query": "Original Regression Paper"},
        )
        new_document_search = mcp.call_tool(
            name="search_documents",
            arguments={"query": "Replacement Regression Paper"},
        )
        new_passages = mcp.call_tool(
            name="search_passages",
            arguments={"query": "nebulafjord", "limit": 8},
        )
        new_tables = mcp.call_tool(
            name="search_tables",
            arguments={"query": "nebulafjord metrics", "limit": 5},
        )
        old_passage_error = mcp.call_tool(
            name="get_passage_context",
            arguments={"passage_id": initial_passage_id},
            expect_error=True,
        )
        old_table_error = mcp.call_tool(
            name="get_table",
            arguments={"table_id": initial_table_id},
            expect_error=True,
        )
        replacement_pack = mcp.call_tool(
            name="build_context_pack",
            arguments={"query": "nebulafjord", "limit": 8},
        )

    assert detail.status_code == 200
    assert detail.json()["title"] == "Replacement Regression Paper"
    assert detail.json()["current_status"] == "ready"
    assert tables.status_code == 200
    assert tables.json()["tables"][0]["caption"] == "nebulafjord metrics"
    assert replacement_job.status_code == 200
    assert replacement_job.json()["status"] == "ready"
    assert old_document_search["documents"] == []
    assert len(new_document_search["documents"]) == 1
    assert new_document_search["documents"][0]["document_id"] == str(document_id)
    assert len(new_passages["passages"]) >= 1
    assert new_passages["passages"][0]["document_id"] == str(document_id)
    assert len(new_tables["tables"]) == 1
    assert new_tables["tables"][0]["caption"] == "nebulafjord metrics"
    assert old_passage_error == "Error calling tool 'get_passage_context': passage not found"
    assert old_table_error == "Error calling tool 'get_table': table not found"
    assert replacement_pack["documents"][0]["title"] == "Replacement Regression Paper"


def test_newer_replacement_supersedes_older_queued_job(
    regression_runtime: _RegressionRuntime,
) -> None:
    initial_document = _make_parsed_document(
        title="Supersede Regression Original",
        keyword="supersede-origin",
        table_caption="supersede-origin metrics",
    )
    newest_document = _make_parsed_document(
        title="Supersede Regression Final",
        keyword="supersede-final",
        table_caption="supersede-final metrics",
    )

    with TestClient(create_app()) as client:
        upload = _upload_document(client, title="Initial upload title")
        document_id = UUID(str(upload["document_id"]))
        initial_job_id = UUID(str(upload["ingest_job_id"]))

        initial_handled, _, _ = _run_worker(
            regression_runtime,
            primary_result=_make_parser_result(parsed_document=initial_document),
        )
        assert initial_handled.payload.ingest_job_id == initial_job_id

        first_replacement = _replace_document(
            client,
            document_id=document_id,
            title="Superseded replacement upload title",
        )
        second_replacement = _replace_document(
            client,
            document_id=document_id,
            title="Newest replacement upload title",
        )
        first_replacement_job_id = UUID(str(first_replacement["ingest_job_id"]))
        second_replacement_job_id = UUID(str(second_replacement["ingest_job_id"]))

        first_replacement_job = client.get(f"/ingest-jobs/{first_replacement_job_id}")
        second_replacement_job = client.get(f"/ingest-jobs/{second_replacement_job_id}")
        pre_promotion_detail = client.get(f"/documents/{document_id}")

        handled, _, _ = _run_worker(
            regression_runtime,
            primary_result=_make_parser_result(parsed_document=newest_document),
        )
        assert handled.payload.ingest_job_id == second_replacement_job_id

        final_detail = client.get(f"/documents/{document_id}")
        final_tables = client.get(f"/documents/{document_id}/tables")
        final_second_replacement_job = client.get(f"/ingest-jobs/{second_replacement_job_id}")
        mcp = _McpSession(client)
        superseded_search = mcp.call_tool(
            name="search_documents",
            arguments={"query": "Superseded replacement upload title"},
        )
        final_search = mcp.call_tool(
            name="search_documents",
            arguments={"query": "Supersede Regression Final"},
        )

    assert first_replacement_job.status_code == 200
    assert first_replacement_job.json()["status"] == "failed"
    assert first_replacement_job.json()["failure_code"] == "superseded_by_newer_ingest_job"
    assert second_replacement_job.status_code == 200
    assert second_replacement_job.json()["status"] == "queued"
    assert pre_promotion_detail.status_code == 200
    assert pre_promotion_detail.json()["title"] == "Supersede Regression Original"
    assert final_detail.status_code == 200
    assert final_detail.json()["title"] == "Supersede Regression Final"
    assert final_tables.json()["tables"][0]["caption"] == "supersede-final metrics"
    assert final_second_replacement_job.json()["status"] == "ready"
    assert superseded_search["documents"] == []
    assert final_search["documents"][0]["document_id"] == str(document_id)


def test_failed_replacement_preserves_previous_live_revision(
    regression_runtime: _RegressionRuntime,
) -> None:
    initial_document = _make_parsed_document(
        title="Preserved Regression Paper",
        keyword="preservium",
        table_caption="preservium metrics",
    )

    with TestClient(create_app()) as client:
        upload = _upload_document(client, title="Initial upload title")
        document_id = UUID(str(upload["document_id"]))
        initial_job_id = UUID(str(upload["ingest_job_id"]))

        initial_handled, _, _ = _run_worker(
            regression_runtime,
            primary_result=_make_parser_result(parsed_document=initial_document),
        )
        assert initial_handled.payload.ingest_job_id == initial_job_id

        initial_mcp = _McpSession(client)
        initial_passages = initial_mcp.call_tool(
            name="search_passages",
            arguments={"query": "preservium", "limit": 8},
        )
        initial_tables = initial_mcp.call_tool(
            name="search_tables",
            arguments={"query": "preservium metrics", "limit": 5},
        )
        initial_passage_id = str(initial_passages["passages"][0]["passage_id"])
        initial_table_id = str(initial_tables["tables"][0]["table_id"])

        replacement = _replace_document(
            client,
            document_id=document_id,
            title="Broken replacement upload title",
        )
        replacement_job_id = UUID(str(replacement["ingest_job_id"]))

        failed_handled, _, _ = _run_worker(
            regression_runtime,
            primary_result=_make_parser_result("fail", parser_name="docling"),
        )
        assert failed_handled.payload.ingest_job_id == replacement_job_id

        replacement_job = client.get(f"/ingest-jobs/{replacement_job_id}")
        detail = client.get(f"/documents/{document_id}")
        tables = client.get(f"/documents/{document_id}/tables")
        mcp = _McpSession(client)
        original_document_search = mcp.call_tool(
            name="search_documents",
            arguments={"query": "Preserved Regression Paper"},
        )
        replacement_document_search = mcp.call_tool(
            name="search_documents",
            arguments={"query": "Broken replacement upload title"},
        )
        original_passages = mcp.call_tool(
            name="search_passages",
            arguments={"query": "preservium", "limit": 8},
        )
        original_passage_context = mcp.call_tool(
            name="get_passage_context",
            arguments={"passage_id": initial_passage_id, "before": 0, "after": 1},
        )
        original_table_detail = mcp.call_tool(
            name="get_table",
            arguments={"table_id": initial_table_id},
        )
        preserved_pack = mcp.call_tool(
            name="build_context_pack",
            arguments={"query": "preservium", "limit": 8},
        )

    assert replacement_job.status_code == 200
    assert replacement_job.json()["status"] == "failed"
    assert replacement_job.json()["failure_code"] == "docling_failed"
    assert detail.status_code == 200
    assert detail.json()["title"] == "Preserved Regression Paper"
    assert detail.json()["current_status"] == "ready"
    assert tables.status_code == 200
    assert tables.json()["tables"][0]["caption"] == "preservium metrics"
    assert len(original_document_search["documents"]) == 1
    assert replacement_document_search["documents"] == []
    assert len(original_passages["passages"]) >= 1
    assert original_passage_context["passage"]["passage_id"] == initial_passage_id
    assert original_table_detail["table_id"] == initial_table_id
    assert preserved_pack["documents"][0]["title"] == "Preserved Regression Paper"
