from __future__ import annotations

import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException, UploadFile
from starlette.requests import Request

from paper_context.api.routes import documents as documents_route_module
from paper_context.ingestion.api import DocumentsApiService

pytestmark = pytest.mark.unit


def test_get_documents_service_builds_service_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        storage=SimpleNamespace(root_path="/tmp/storage"),
        queue=SimpleNamespace(name="document_ingest"),
        upload=SimpleNamespace(max_bytes=1024),
    )
    engine = object()
    storage = object()
    queue = object()
    service = object()
    service_cls = MagicMock(return_value=service)
    queue_cls = MagicMock(return_value=queue)
    storage_cls = MagicMock(return_value=storage)
    monkeypatch.setattr(documents_route_module, "get_settings", lambda: settings)
    monkeypatch.setattr(documents_route_module, "get_engine", lambda: engine)
    monkeypatch.setattr(documents_route_module, "LocalFilesystemStorage", storage_cls)
    monkeypatch.setattr(documents_route_module, "IngestionQueue", queue_cls)
    monkeypatch.setattr(documents_route_module, "DocumentsApiService", service_cls)

    resolved = documents_route_module.get_documents_service()

    assert resolved is service
    storage_cls.assert_called_once_with("/tmp/storage")
    queue_cls.assert_called_once_with("document_ingest")
    service_cls.assert_called_once_with(
        engine=engine,
        queue=queue,
        storage=storage,
        max_upload_bytes=1024,
    )


def test_upload_document_translates_value_error_to_http_400() -> None:
    service = MagicMock()
    service.create_document.side_effect = ValueError("uploaded file is empty")
    upload = UploadFile(filename="paper.pdf", file=BytesIO(b""))
    upload_document = documents_route_module.upload_document
    request = Request({"type": "http", "headers": []})

    with pytest.raises(HTTPException, match="uploaded file is empty") as exc_info:
        asyncio.run(upload_document(request=request, file=upload, title=None, service=service))

    assert exc_info.value.status_code == 400


def test_create_document_rejects_empty_body_before_storage_work() -> None:
    service = DocumentsApiService(engine=MagicMock(), queue=MagicMock(), storage=MagicMock())

    with pytest.raises(ValueError, match="uploaded file is empty"):
        service.create_document(
            filename="paper.pdf",
            content_type="application/pdf",
            upload=BytesIO(b""),
        )


def test_create_document_rejects_non_pdf_uploads() -> None:
    service = DocumentsApiService(engine=MagicMock(), queue=MagicMock(), storage=MagicMock())

    with pytest.raises(ValueError, match="uploaded file must be a PDF"):
        service.create_document(
            filename="paper.txt",
            content_type="text/plain",
            upload=BytesIO(b"plain text"),
        )


def test_create_document_forwards_trace_headers_to_queue() -> None:
    engine = MagicMock()
    connection = MagicMock()
    engine.begin.return_value.__enter__.return_value = connection
    engine.begin.return_value.__exit__.return_value = None
    queue = MagicMock()
    storage = MagicMock()
    storage.store_file.return_value = SimpleNamespace(
        storage_ref="documents/source.pdf",
        checksum="abc123",
    )
    service = DocumentsApiService(engine=engine, queue=queue, storage=storage)

    response = service.create_document(
        filename="paper.pdf",
        content_type="application/pdf",
        upload=BytesIO(b"%PDF-1.4"),
        trace_headers={"x-trace-id": "trace-123"},
    )

    assert response.status == "queued"
    queue.enqueue_ingest.assert_called_once()
    assert queue.enqueue_ingest.call_args.kwargs["headers"] == {"x-trace-id": "trace-123"}


def test_get_ingest_job_returns_none_when_not_found() -> None:
    ingest_job_id = uuid4()
    engine = MagicMock()
    connection = MagicMock()
    connection.execute.return_value.mappings.return_value.one_or_none.return_value = None
    engine.begin.return_value.__enter__.return_value = connection
    engine.begin.return_value.__exit__.return_value = None
    service = DocumentsApiService(engine=engine, queue=MagicMock(), storage=MagicMock())

    assert service.get_ingest_job(ingest_job_id) is None


def test_route_get_ingest_job_returns_404_when_missing() -> None:
    service = MagicMock()
    service.get_ingest_job.return_value = None

    with pytest.raises(HTTPException, match="ingest job not found") as exc_info:
        documents_route_module.get_ingest_job(uuid4(), service=service)

    assert exc_info.value.status_code == 404
