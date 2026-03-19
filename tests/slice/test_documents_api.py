from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from paper_context.api import app as api_app_module
from paper_context.api.app import create_app
from paper_context.api.routes.documents import get_documents_service
from paper_context.ingestion.api import DocumentNotFoundError, InvalidCursorError
from paper_context.schemas.api import (
    DocumentListResponse,
    DocumentOutlineNode,
    DocumentOutlineResponse,
    DocumentReplaceResponse,
    DocumentResult,
    DocumentTableRecord,
    DocumentTablesResponse,
    DocumentUploadResponse,
    IngestJobResponse,
    TablePreviewModel,
)

pytestmark = pytest.mark.slice


class _DocumentsApiService:
    def __init__(
        self,
        upload_response: DocumentUploadResponse,
        ingest_jobs: dict[UUID, IngestJobResponse],
        *,
        replace_response: DocumentReplaceResponse | None = None,
        documents: DocumentListResponse | None = None,
        document: DocumentResult | None = None,
        outline: DocumentOutlineResponse | None = None,
        tables: DocumentTablesResponse | None = None,
    ) -> None:
        self.upload_response = upload_response
        self.ingest_jobs = ingest_jobs
        self.replace_response = replace_response or DocumentReplaceResponse(
            document_id=upload_response.document_id,
            ingest_job_id=upload_response.ingest_job_id,
            status="queued",
        )
        self.documents = documents or DocumentListResponse()
        self.document = document
        self.outline = outline
        self.tables = tables
        self.upload_calls: list[dict[str, object]] = []
        self.replace_calls: list[dict[str, object]] = []
        self.job_calls: list[UUID] = []
        self.documents_calls: list[dict[str, object]] = []
        self.document_calls: list[UUID] = []
        self.outline_calls: list[UUID] = []
        self.tables_calls: list[dict[str, object]] = []

    def create_document(
        self,
        *,
        filename: str,
        content_type: str | None,
        upload,
        title: str | None,
        trace_headers=None,
    ) -> DocumentUploadResponse:
        self.upload_calls.append(
            {
                "filename": filename,
                "content_type": content_type,
                "body": upload.read(),
                "title": title,
                "trace_headers": trace_headers,
            }
        )
        return self.upload_response

    def replace_document(
        self,
        document_id: UUID,
        *,
        filename: str,
        content_type: str | None,
        upload,
        title: str | None,
        trace_headers=None,
    ) -> DocumentReplaceResponse:
        self.replace_calls.append(
            {
                "document_id": document_id,
                "filename": filename,
                "content_type": content_type,
                "body": upload.read(),
                "title": title,
                "trace_headers": trace_headers,
            }
        )
        return self.replace_response

    def list_documents(self, *, limit: int, cursor: str | None) -> DocumentListResponse:
        self.documents_calls.append({"limit": limit, "cursor": cursor})
        return self.documents

    def get_document(self, document_id: UUID) -> DocumentResult | None:
        self.document_calls.append(document_id)
        return self.document

    def get_document_outline(self, document_id: UUID) -> DocumentOutlineResponse | None:
        self.outline_calls.append(document_id)
        return self.outline

    def get_document_tables(
        self,
        document_id: UUID,
    ) -> DocumentTablesResponse | None:
        self.tables_calls.append({"document_id": document_id})
        return self.tables

    def get_ingest_job(self, ingest_job_id: UUID) -> IngestJobResponse | None:
        self.job_calls.append(ingest_job_id)
        return self.ingest_jobs.get(ingest_job_id)


def _patch_documents_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    service: _DocumentsApiService,
) -> TestClient:
    settings = SimpleNamespace(
        log_level="INFO",
        storage=SimpleNamespace(root_path=tmp_path / "artifacts"),
        queue=SimpleNamespace(name="document_ingest"),
        runtime=SimpleNamespace(worker_idle_sleep_seconds=0.1),
    )
    monkeypatch.setattr(api_app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        api_app_module,
        "LocalFilesystemStorage",
        lambda root_path: SimpleNamespace(ensure_root=lambda: None),
    )
    monkeypatch.setattr(api_app_module, "configure_logging", lambda level: None)
    monkeypatch.setattr(api_app_module, "dispose_engine", lambda: None)

    app = create_app()
    app.dependency_overrides[get_documents_service] = lambda: service
    return TestClient(app)


def test_post_documents_creates_document_and_returns_initial_job_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    upload_response = DocumentUploadResponse(
        document_id=UUID("11111111-1111-1111-1111-111111111111"),
        ingest_job_id=UUID("22222222-2222-2222-2222-222222222222"),
        status="queued",
    )
    service = _DocumentsApiService(upload_response, {})

    with _patch_documents_app(monkeypatch, tmp_path, service) as client:
        response = client.post(
            "/documents",
            files={"file": ("paper.pdf", b"%PDF-1.4\nphase-1", "application/pdf")},
            data={"title": "Phase 1 paper"},
        )

    assert response.status_code == 201
    assert response.json() == upload_response.model_dump(mode="json")
    assert service.upload_calls == [
        {
            "filename": "paper.pdf",
            "content_type": "application/pdf",
            "body": b"%PDF-1.4\nphase-1",
            "title": "Phase 1 paper",
            "trace_headers": {},
        }
    ]


def test_get_ingest_job_returns_status_and_failure_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ingest_job_id = UUID("33333333-3333-3333-3333-333333333333")
    job_response = IngestJobResponse(
        id=ingest_job_id,
        document_id=UUID("44444444-4444-4444-4444-444444444444"),
        status="failed",
        failure_code="parse_failed",
        failure_message="Docling and fallback parsing could not recover stable structure.",
        warnings=["parser_fallback_used", "reduced_structure_confidence"],
        started_at=datetime(2026, 3, 18, 10, 0, tzinfo=UTC),
        finished_at=datetime(2026, 3, 18, 10, 2, tzinfo=UTC),
        trigger="upload",
    )
    service = _DocumentsApiService(
        DocumentUploadResponse(
            document_id=UUID("11111111-1111-1111-1111-111111111111"),
            ingest_job_id=ingest_job_id,
            status="queued",
        ),
        {ingest_job_id: job_response},
    )

    with _patch_documents_app(monkeypatch, tmp_path, service) as client:
        response = client.get(f"/ingest-jobs/{ingest_job_id}")

    assert response.status_code == 200
    assert response.json() == job_response.model_dump(mode="json")
    assert service.job_calls == [ingest_job_id]


def test_get_ingest_job_returns_404_for_missing_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _DocumentsApiService(
        DocumentUploadResponse(
            document_id=UUID("11111111-1111-1111-1111-111111111111"),
            ingest_job_id=UUID("22222222-2222-2222-2222-222222222222"),
            status="queued",
        ),
        {},
    )

    with _patch_documents_app(monkeypatch, tmp_path, service) as client:
        response = client.get("/ingest-jobs/33333333-3333-3333-3333-333333333333")

    assert response.status_code == 404
    assert response.json() == {"detail": "ingest job not found"}


def test_get_documents_returns_paginated_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    document = DocumentResult(
        document_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        title="Phase 3 paper",
        authors=["Ada Lovelace"],
        publication_year=2024,
        quant_tags={"asset_universe": "rates"},
        current_status="ready",
        active_index_version="mvp-v1",
    )
    service = _DocumentsApiService(
        DocumentUploadResponse(
            document_id=document.document_id,
            ingest_job_id=UUID("22222222-2222-2222-2222-222222222222"),
            status="queued",
        ),
        {},
        documents=DocumentListResponse(documents=[document], next_cursor="cursor-1"),
    )

    with _patch_documents_app(monkeypatch, tmp_path, service) as client:
        response = client.get("/documents", params={"limit": 5, "cursor": "cursor-0"})

    assert response.status_code == 200
    assert response.json() == {
        "documents": [document.model_dump(mode="json")],
        "next_cursor": "cursor-1",
    }
    assert service.documents_calls == [{"limit": 5, "cursor": "cursor-0"}]


def test_document_read_surface_returns_document_outline_and_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    document = DocumentResult(
        document_id=document_id,
        title="Phase 3 paper",
        authors=["Ada Lovelace"],
        publication_year=2024,
        quant_tags={},
        current_status="ready",
        active_index_version="mvp-v1",
    )
    outline = DocumentOutlineResponse(
        document_id=document_id,
        title="Phase 3 paper",
        sections=[
            DocumentOutlineNode(
                section_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                heading="Methods",
                section_path=["Methods"],
                ordinal=1,
                page_start=1,
                page_end=2,
                children=[],
            )
        ],
    )
    tables = DocumentTablesResponse(
        document_id=document_id,
        title="Phase 3 paper",
        tables=[
            DocumentTableRecord(
                table_id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
                document_id=document_id,
                section_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                document_title="Phase 3 paper",
                section_path=["Methods"],
                caption="Results table",
                table_type="lexical",
                preview=TablePreviewModel(
                    headers=["A"],
                    rows=[["1"]],
                    row_count=1,
                ),
                page_start=2,
                page_end=2,
            )
        ],
    )
    service = _DocumentsApiService(
        DocumentUploadResponse(
            document_id=document_id,
            ingest_job_id=UUID("22222222-2222-2222-2222-222222222222"),
            status="queued",
        ),
        {},
        document=document,
        outline=outline,
        tables=tables,
    )

    with _patch_documents_app(monkeypatch, tmp_path, service) as client:
        detail = client.get(f"/documents/{document_id}")
        outline_response = client.get(f"/documents/{document_id}/outline")
        tables_response = client.get(f"/documents/{document_id}/tables")

    assert detail.status_code == 200
    assert detail.json() == document.model_dump(mode="json")
    assert outline_response.status_code == 200
    assert outline_response.json() == outline.model_dump(mode="json")
    assert tables_response.status_code == 200
    assert tables_response.json() == tables.model_dump(mode="json")
    assert service.document_calls == [document_id]
    assert service.outline_calls == [document_id]
    assert service.tables_calls == [{"document_id": document_id}]


def test_document_list_invalid_cursor_returns_400(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _DocumentsApiService(
        DocumentUploadResponse(
            document_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            ingest_job_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status="queued",
        ),
        {},
    )

    class _InvalidCursorService(_DocumentsApiService):
        def list_documents(self, *, limit: int, cursor: str | None) -> DocumentListResponse:
            raise InvalidCursorError("invalid cursor")

    with _patch_documents_app(
        monkeypatch,
        tmp_path,
        _InvalidCursorService(
            service.upload_response,
            {},
        ),
    ) as client:
        response = client.get("/documents", params={"cursor": "bad"})

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid cursor"}


def test_document_list_limit_above_route_bound_returns_422(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _DocumentsApiService(
        DocumentUploadResponse(
            document_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            ingest_job_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status="queued",
        ),
        {},
    )

    with _patch_documents_app(monkeypatch, tmp_path, service) as client:
        response = client.get("/documents", params={"limit": 101})

    assert response.status_code == 422


def test_post_document_replace_creates_new_ingest_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    document_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    replace_response = DocumentReplaceResponse(
        document_id=document_id,
        ingest_job_id=UUID("33333333-3333-3333-3333-333333333333"),
        status="queued",
    )
    service = _DocumentsApiService(
        DocumentUploadResponse(
            document_id=document_id,
            ingest_job_id=UUID("22222222-2222-2222-2222-222222222222"),
            status="queued",
        ),
        {},
        replace_response=replace_response,
    )

    with _patch_documents_app(monkeypatch, tmp_path, service) as client:
        response = client.post(
            f"/documents/{document_id}/replace",
            files={"file": ("paper.pdf", b"%PDF-1.4\nphase-3", "application/pdf")},
            data={"title": "Phase 3 replacement"},
        )

    assert response.status_code == 202
    assert response.json() == replace_response.model_dump(mode="json")
    assert service.replace_calls == [
        {
            "document_id": document_id,
            "filename": "paper.pdf",
            "content_type": "application/pdf",
            "body": b"%PDF-1.4\nphase-3",
            "title": "Phase 3 replacement",
            "trace_headers": {},
        }
    ]


def test_post_document_replace_missing_document_returns_404(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _MissingDocumentService(_DocumentsApiService):
        def replace_document(
            self,
            document_id: UUID,
            *,
            filename: str,
            content_type: str | None,
            upload,
            title: str | None,
            trace_headers=None,
        ) -> DocumentReplaceResponse:
            del document_id, filename, content_type, upload, title, trace_headers
            raise DocumentNotFoundError("document not found")

    service = _MissingDocumentService(
        DocumentUploadResponse(
            document_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            ingest_job_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            status="queued",
        ),
        {},
    )

    with _patch_documents_app(monkeypatch, tmp_path, service) as client:
        response = client.post(
            "/documents/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/replace",
            files={"file": ("paper.pdf", b"%PDF-1.4\nphase-3", "application/pdf")},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "document not found"}
