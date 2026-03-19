from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from paper_context.api import app as api_app_module
from paper_context.api.app import create_app
from paper_context.api.routes.documents import get_documents_service
from paper_context.ingestion.api import DocumentNotFoundError
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

pytestmark = pytest.mark.contract

GOLDEN_DIR = Path(__file__).with_name("golden")


class _DocumentsApiService:
    def __init__(
        self,
        upload_response: DocumentUploadResponse,
        ingest_jobs: dict[UUID, IngestJobResponse],
    ) -> None:
        self.upload_response = upload_response
        self.ingest_jobs = ingest_jobs
        self.document_id = UUID("11111111-1111-1111-1111-111111111111")
        self.document = DocumentResult(
            document_id=self.document_id,
            title="Phase 3 paper",
            authors=["Ada Lovelace"],
            publication_year=2024,
            quant_tags={"asset_universe": "rates"},
            current_status="ready",
            active_index_version="mvp-v1",
        )
        self.documents = DocumentListResponse(
            documents=[self.document],
            next_cursor="cursor-1",
        )
        self.outline = DocumentOutlineResponse(
            document_id=self.document_id,
            title=self.document.title,
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
        self.tables = DocumentTablesResponse(
            document_id=self.document_id,
            title=self.document.title,
            tables=[
                DocumentTableRecord(
                    table_id=UUID("33333333-3333-3333-3333-333333333333"),
                    document_id=self.document_id,
                    section_id=UUID("22222222-2222-2222-2222-222222222222"),
                    document_title=self.document.title,
                    section_path=["Methods"],
                    caption="Results table",
                    table_type="lexical",
                    preview=TablePreviewModel(
                        headers=["A", "B"],
                        rows=[["1", "2"], ["3", "4"]],
                        row_count=2,
                    ),
                    page_start=1,
                    page_end=2,
                )
            ],
        )
        self.replace_response = DocumentReplaceResponse(
            document_id=self.document_id,
            ingest_job_id=UUID("44444444-4444-4444-4444-444444444444"),
            status="queued",
        )

    def create_document(self, **kwargs) -> DocumentUploadResponse:
        del kwargs
        return self.upload_response

    def get_ingest_job(self, ingest_job_id: UUID) -> IngestJobResponse | None:
        return self.ingest_jobs.get(ingest_job_id)

    def list_documents(self, *, limit: int = 20, cursor: str | None = None) -> DocumentListResponse:
        del limit, cursor
        return self.documents

    def get_document(self, document_id: UUID) -> DocumentResult | None:
        return self.document if document_id == self.document_id else None

    def get_document_outline(self, document_id: UUID) -> DocumentOutlineResponse | None:
        return self.outline if document_id == self.document_id else None

    def get_document_tables(self, document_id: UUID) -> DocumentTablesResponse | None:
        return self.tables if document_id == self.document_id else None

    def replace_document(self, document_id: UUID, **kwargs) -> DocumentReplaceResponse:
        del kwargs
        if document_id != self.document_id:
            raise DocumentNotFoundError("document not found")
        return self.replace_response


def _golden_payload(name: str) -> dict[str, object]:
    return json.loads((GOLDEN_DIR / name).read_text())


def _build_client(
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


def test_documents_routes_expose_expected_openapi_contract(
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

    with _build_client(monkeypatch, tmp_path, service) as client:
        schema = client.get("/openapi.json").json()

    assert schema["paths"]["/documents"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/DocumentListResponse"}
    assert schema["paths"]["/documents/{document_id}"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/DocumentResult"}
    assert schema["paths"]["/documents/{document_id}/outline"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"] == {"$ref": "#/components/schemas/DocumentOutlineResponse"}
    assert schema["paths"]["/documents/{document_id}/tables"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/DocumentTablesResponse"}
    assert schema["paths"]["/documents/{document_id}/replace"]["post"]["responses"]["202"][
        "content"
    ]["application/json"]["schema"] == {"$ref": "#/components/schemas/DocumentReplaceResponse"}


def test_document_list_response_matches_golden_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = _golden_payload("documents-list.json")
    service = _DocumentsApiService(
        DocumentUploadResponse(
            document_id=UUID("11111111-1111-1111-1111-111111111111"),
            ingest_job_id=UUID("22222222-2222-2222-2222-222222222222"),
            status="queued",
        ),
        {},
    )

    with _build_client(monkeypatch, tmp_path, service) as client:
        payload = client.get("/documents").json()

    assert payload == expected
    assert DocumentListResponse.model_validate(payload).model_dump(mode="json") == expected


def test_document_detail_response_matches_golden_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = _golden_payload("document-detail.json")
    service = _DocumentsApiService(
        DocumentUploadResponse(
            document_id=UUID("11111111-1111-1111-1111-111111111111"),
            ingest_job_id=UUID("22222222-2222-2222-2222-222222222222"),
            status="queued",
        ),
        {},
    )

    with _build_client(monkeypatch, tmp_path, service) as client:
        payload = client.get("/documents/11111111-1111-1111-1111-111111111111").json()

    assert payload == expected
    assert DocumentResult.model_validate(payload).model_dump(mode="json") == expected
