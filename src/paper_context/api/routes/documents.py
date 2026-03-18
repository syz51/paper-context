from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from paper_context.config import get_settings
from paper_context.db.engine import get_engine
from paper_context.ingestion.api import DocumentsApiService
from paper_context.queue.contracts import IngestionQueue
from paper_context.schemas.api import DocumentUploadResponse, IngestJobResponse
from paper_context.storage.local_fs import LocalFilesystemStorage

router = APIRouter()


def get_documents_service() -> DocumentsApiService:
    settings = get_settings()
    storage = LocalFilesystemStorage(settings.storage.root_path)
    return DocumentsApiService(
        engine=get_engine(),
        queue=IngestionQueue(settings.queue.name),
        storage=storage,
    )


@router.post("/documents", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),  # noqa: B008
    title: str | None = Form(default=None),  # noqa: B008
    service: DocumentsApiService = Depends(get_documents_service),  # noqa: B008
) -> DocumentUploadResponse:
    try:
        return service.create_document(
            filename=file.filename or "document.pdf",
            content_type=file.content_type,
            body=await file.read(),
            title=title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/ingest-jobs/{ingest_job_id}", response_model=IngestJobResponse)
def get_ingest_job(
    ingest_job_id: UUID,
    service: DocumentsApiService = Depends(get_documents_service),  # noqa: B008
) -> IngestJobResponse:
    job = service.get_ingest_job(ingest_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ingest job not found")
    return job
