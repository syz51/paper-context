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
from paper_context.schemas.api import DocumentUploadResponse, IngestJobResponse

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

    def create_document(
        self,
        *,
        filename: str,
        content_type: str | None,
        body: bytes,
        title: str | None,
        trace_headers=None,
    ) -> DocumentUploadResponse:
        return self.upload_response

    def get_ingest_job(self, ingest_job_id: UUID) -> IngestJobResponse | None:
        return self.ingest_jobs.get(ingest_job_id)


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


def test_documents_upload_openapi_contract_exposes_multipart_request(
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

    upload_path = schema["paths"]["/documents"]["post"]
    job_path = schema["paths"]["/ingest-jobs/{ingest_job_id}"]["get"]

    assert "multipart/form-data" in upload_path["requestBody"]["content"]
    assert upload_path["responses"]["201"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/DocumentUploadResponse"
    }
    assert job_path["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/IngestJobResponse"
    }


def test_document_upload_response_matches_golden_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = _golden_payload("documents-upload.json")
    service = _DocumentsApiService(DocumentUploadResponse.model_validate(expected), {})

    with _build_client(monkeypatch, tmp_path, service) as client:
        payload = client.post(
            "/documents",
            files={"file": ("paper.pdf", b"%PDF-1.4\nphase-1", "application/pdf")},
            data={"title": "Phase 1 paper"},
        ).json()

    assert payload == expected
    assert DocumentUploadResponse.model_validate(payload).model_dump(mode="json") == expected


@pytest.mark.parametrize(
    ("golden_name", "requested_job_id"),
    [
        ("ingest-job-ready.json", "33333333-3333-3333-3333-333333333333"),
        ("ingest-job-failed.json", "44444444-4444-4444-4444-444444444444"),
    ],
)
def test_ingest_job_response_matches_golden_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    golden_name: str,
    requested_job_id: str,
) -> None:
    expected = _golden_payload(golden_name)
    ingest_job_id = UUID(str(expected["id"]))
    service = _DocumentsApiService(
        DocumentUploadResponse(
            document_id=UUID("11111111-1111-1111-1111-111111111111"),
            ingest_job_id=ingest_job_id,
            status="queued",
        ),
        {ingest_job_id: IngestJobResponse.model_validate(expected)},
    )

    with _build_client(monkeypatch, tmp_path, service) as client:
        payload = client.get(f"/ingest-jobs/{requested_job_id}").json()

    assert payload == expected
    assert IngestJobResponse.model_validate(payload).model_dump(mode="json") == expected
