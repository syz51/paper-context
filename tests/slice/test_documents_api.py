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
from paper_context.schemas.api import DocumentUploadResponse, IngestJobResponse

pytestmark = pytest.mark.slice


class _DocumentsApiService:
    def __init__(
        self,
        upload_response: DocumentUploadResponse,
        ingest_jobs: dict[UUID, IngestJobResponse],
    ) -> None:
        self.upload_response = upload_response
        self.ingest_jobs = ingest_jobs
        self.upload_calls: list[dict[str, object]] = []
        self.job_calls: list[UUID] = []

    def create_document(
        self,
        *,
        filename: str,
        content_type: str | None,
        body: bytes,
        title: str | None,
        trace_headers=None,
    ) -> DocumentUploadResponse:
        self.upload_calls.append(
            {
                "filename": filename,
                "content_type": content_type,
                "body": body,
                "title": title,
            }
        )
        return self.upload_response

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
