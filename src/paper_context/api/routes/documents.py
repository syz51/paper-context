from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from paper_context.config import get_settings
from paper_context.db.engine import get_engine
from paper_context.ingestion.api import (
    DocumentNotFoundError,
    DocumentsApiService,
    InvalidCursorError,
    UploadTooLargeError,
)
from paper_context.queue.contracts import IngestionQueue
from paper_context.schemas.api import (
    DocumentListResponse,
    DocumentOutlineResponse,
    DocumentReplaceResponse,
    DocumentResult,
    DocumentTablesResponse,
    DocumentUploadResponse,
    IngestJobResponse,
)
from paper_context.storage.local_fs import LocalFilesystemStorage

router = APIRouter()
_TRACE_HEADER_NAMES = {"traceparent", "tracestate", "baggage", "x-request-id", "x-trace-id"}
_TRACE_HEADER_PREFIXES = ("x-b3-",)


def get_documents_service() -> DocumentsApiService:
    settings = get_settings()
    storage = LocalFilesystemStorage(settings.storage.root_path)
    return DocumentsApiService(
        engine=get_engine(),
        queue=IngestionQueue(settings.queue.name),
        storage=storage,
        max_upload_bytes=getattr(getattr(settings, "upload", None), "max_bytes", 25 * 1024 * 1024),
    )


def _trace_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lowered = key.lower()
        if lowered in _TRACE_HEADER_NAMES or lowered.startswith(_TRACE_HEADER_PREFIXES):
            headers[lowered] = value
    return headers


def _translate_document_error(exc: Exception) -> HTTPException:
    if isinstance(exc, DocumentNotFoundError):
        return HTTPException(status_code=404, detail="document not found")
    if isinstance(exc, UploadTooLargeError):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, InvalidCursorError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    raise exc


@router.post("/documents", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    title: str | None = Form(default=None),  # noqa: B008
    service: DocumentsApiService = Depends(get_documents_service),  # noqa: B008
) -> DocumentUploadResponse:
    try:
        return await run_in_threadpool(
            service.create_document,
            filename=file.filename or "document.pdf",
            content_type=file.content_type,
            upload=file.file,
            title=title,
            trace_headers=_trace_headers(request),
        )
    except Exception as exc:
        raise _translate_document_error(exc) from exc


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
    service: DocumentsApiService = Depends(get_documents_service),  # noqa: B008
) -> DocumentListResponse:
    try:
        return service.list_documents(limit=limit, cursor=cursor)
    except InvalidCursorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/documents/{document_id}", response_model=DocumentResult)
def get_document(
    document_id: UUID,
    service: DocumentsApiService = Depends(get_documents_service),  # noqa: B008
) -> DocumentResult:
    document = service.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return document


@router.get("/documents/{document_id}/outline", response_model=DocumentOutlineResponse)
def get_document_outline(
    document_id: UUID,
    service: DocumentsApiService = Depends(get_documents_service),  # noqa: B008
) -> DocumentOutlineResponse:
    outline = service.get_document_outline(document_id)
    if outline is None:
        raise HTTPException(status_code=404, detail="document not found")
    return outline


@router.get("/documents/{document_id}/tables", response_model=DocumentTablesResponse)
def get_document_tables(
    document_id: UUID,
    service: DocumentsApiService = Depends(get_documents_service),  # noqa: B008
) -> DocumentTablesResponse:
    tables = service.get_document_tables(document_id)
    if tables is None:
        raise HTTPException(status_code=404, detail="document not found")
    return tables


@router.post(
    "/documents/{document_id}/replace",
    response_model=DocumentReplaceResponse,
    status_code=202,
)
async def replace_document(
    document_id: UUID,
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    title: str | None = Form(default=None),  # noqa: B008
    service: DocumentsApiService = Depends(get_documents_service),  # noqa: B008
) -> DocumentReplaceResponse:
    try:
        return await run_in_threadpool(
            service.replace_document,
            document_id,
            filename=file.filename or "document.pdf",
            content_type=file.content_type,
            upload=file.file,
            title=title,
            trace_headers=_trace_headers(request),
        )
    except Exception as exc:
        raise _translate_document_error(exc) from exc


@router.get("/ingest-jobs/{ingest_job_id}", response_model=IngestJobResponse)
def get_ingest_job(
    ingest_job_id: UUID,
    service: DocumentsApiService = Depends(get_documents_service),  # noqa: B008
) -> IngestJobResponse:
    job = service.get_ingest_job(ingest_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ingest job not found")
    return job
