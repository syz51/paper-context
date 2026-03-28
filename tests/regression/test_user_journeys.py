from __future__ import annotations

import json
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
        initialize = client.post(
            "/mcp",
            headers=_MCP_HEADERS,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "regression-test", "version": "0"},
                },
            },
        )
        payload = _sse_json(initialize.text)
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
        response = self._client.post(
            "/mcp",
            headers={**_MCP_HEADERS, "mcp-session-id": self._session_id},
            json={
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        self._next_id += 1
        payload = _sse_json(response.text)
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


def _sse_json(response_text: str) -> dict[str, Any]:
    for line in response_text.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"missing SSE payload: {response_text!r}")


def _make_parsed_document(
    *,
    title: str,
    keyword: str,
    table_caption: str,
) -> ParsedDocument:
    return ParsedDocument(
        title=title,
        authors=["Ada Lovelace"],
        abstract=f"{keyword} abstract for regression coverage.",
        publication_year=2024,
        metadata_confidence=0.91,
        sections=[
            ParsedSection(
                key="methods",
                heading="Methods",
                heading_path=["Methods"],
                level=1,
                page_start=1,
                page_end=2,
                paragraphs=[
                    ParsedParagraph(
                        text=(
                            f"{keyword} passage one preserves stable narrative tokens for "
                            "end to end regression retrieval."
                        ),
                        page_start=1,
                        page_end=1,
                        provenance_offsets={"pages": [1], "charspans": [[0, 64]]},
                    ),
                    ParsedParagraph(
                        text=(
                            f"{keyword} passage two adds sibling context so passage context "
                            "lookups stay regression tested."
                        ),
                        page_start=2,
                        page_end=2,
                        provenance_offsets={"pages": [2], "charspans": [[0, 63]]},
                    ),
                ],
            )
        ],
        tables=[
            ParsedTable(
                section_key="methods",
                caption=table_caption,
                headers=["metric", "value"],
                rows=[[f"{keyword} ratio", "1.0"], ["samples", "42"]],
                page_start=2,
                page_end=2,
            )
        ],
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
