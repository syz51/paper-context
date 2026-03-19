from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from paper_context.ingestion.api import DocumentNotFoundError, DocumentsApiService
from paper_context.pagination import decode_cursor
from paper_context.schemas.api import RetrievalFiltersInput

pytestmark = pytest.mark.unit


def _result(
    *,
    one_or_none: dict[str, object] | None = None,
    all_rows: list[dict[str, object]] | None = None,
) -> MagicMock:
    result = MagicMock()
    result.mappings.return_value.one_or_none.return_value = one_or_none
    result.mappings.return_value.all.return_value = all_rows or []
    return result


def _make_service(*, max_upload_bytes: int = 25 * 1024 * 1024):
    engine = MagicMock()
    connection = MagicMock()
    ctx = MagicMock()
    ctx.__enter__.return_value = connection
    ctx.__exit__.return_value = None
    engine.begin.return_value = ctx
    queue = MagicMock()
    storage = MagicMock()
    service = DocumentsApiService(
        engine=engine,
        queue=queue,
        storage=storage,
        max_upload_bytes=max_upload_bytes,
    )
    return service, connection, queue, storage


def test_search_documents_applies_query_filters_and_paginates() -> None:
    service, connection, _, _ = _make_service()
    first_document_id = uuid4()
    second_document_id = uuid4()
    requested_document_id = uuid4()
    updated_at_1 = datetime(2026, 3, 18, 10, 0, tzinfo=UTC)
    updated_at_2 = datetime(2026, 3, 18, 9, 0, tzinfo=UTC)
    rows = [
        {
            "document_id": first_document_id,
            "title": "Alpha",
            "authors": ["Ada"],
            "publication_year": 2024,
            "quant_tags": {"theme": "rates"},
            "current_status": "ready",
            "active_index_version": "index-v1",
            "updated_at": updated_at_1,
        },
        {
            "document_id": second_document_id,
            "title": "Beta",
            "authors": ["Grace"],
            "publication_year": 2025,
            "quant_tags": {"theme": "fx"},
            "current_status": "queued",
            "active_index_version": None,
            "updated_at": updated_at_2,
        },
    ]
    connection.execute.side_effect = [
        _result(all_rows=rows[:2]),
        _result(all_rows=rows[1:2]),
    ]

    first_page = service.search_documents(
        query="  alpha  ",
        filters=RetrievalFiltersInput(
            document_ids=[requested_document_id],
            publication_years=[2024],
        ),
        limit=1,
    )

    assert [document.document_id for document in first_page.documents] == [first_document_id]
    assert first_page.next_cursor is not None
    assert decode_cursor(first_page.next_cursor)["updated_at"] == updated_at_1.isoformat()
    assert decode_cursor(first_page.next_cursor)["document_id"] == str(first_document_id)

    second_page = service.search_documents(
        query="alpha",
        filters=RetrievalFiltersInput(
            document_ids=[requested_document_id],
            publication_years=[2024],
        ),
        limit=1,
        cursor=first_page.next_cursor,
    )

    assert [document.document_id for document in second_page.documents] == [second_document_id]
    assert second_page.next_cursor is None


def test_search_documents_supports_blank_query_without_filters() -> None:
    service, connection, _, _ = _make_service()
    connection.execute.return_value = _result(all_rows=[])

    response = service.search_documents(query="   ", filters=None, limit=5)

    assert response.documents == []
    assert response.next_cursor is None


def test_document_search_limit_is_clamped() -> None:
    service, connection, _, _ = _make_service()
    rows = [
        {
            "document_id": uuid4(),
            "title": "Alpha",
            "authors": ["Ada"],
            "publication_year": 2024,
            "quant_tags": {"theme": "rates"},
            "current_status": "ready",
            "active_index_version": "index-v1",
            "updated_at": datetime(2026, 3, 18, 10, 0, tzinfo=UTC),
        }
    ]
    connection.execute.return_value = _result(all_rows=rows)

    response = service.search_documents(query="alpha", filters=None, limit=10_000)

    assert len(response.documents) == 1
    executed_statement = connection.execute.call_args.args[0]
    assert executed_statement._limit_clause.value == 101


def test_get_document_returns_model_when_row_exists() -> None:
    service, connection, _, _ = _make_service()
    document_id = uuid4()
    connection.execute.return_value = _result(
        one_or_none={
            "document_id": document_id,
            "title": "Model paper",
            "authors": ["Ada", "Grace"],
            "publication_year": 2024,
            "quant_tags": {"desk": "macro"},
            "current_status": "ready",
            "active_index_version": "index-v1",
        }
    )

    document = service.get_document(document_id)

    assert document is not None
    assert document.document_id == document_id
    assert document.authors == ["Ada", "Grace"]
    assert document.quant_tags == {"desk": "macro"}


def test_get_document_outline_builds_tree_and_keeps_orphan_as_root() -> None:
    service, connection, _, _ = _make_service()
    document_id = uuid4()
    root_id = uuid4()
    child_id = uuid4()
    missing_parent_id = uuid4()
    orphan_id = uuid4()

    connection.execute.side_effect = [
        _result(one_or_none={"document_id": document_id, "title": "Outline paper"}),
        _result(
            all_rows=[
                {
                    "section_id": root_id,
                    "parent_section_id": None,
                    "heading": "Intro",
                    "section_path": ["Intro"],
                    "ordinal": 1,
                    "page_start": 1,
                    "page_end": 1,
                },
                {
                    "section_id": child_id,
                    "parent_section_id": root_id,
                    "heading": "Background",
                    "section_path": ["Intro", "Background"],
                    "ordinal": 2,
                    "page_start": 1,
                    "page_end": 2,
                },
                {
                    "section_id": orphan_id,
                    "parent_section_id": missing_parent_id,
                    "heading": "Appendix",
                    "section_path": ["Appendix"],
                    "ordinal": 3,
                    "page_start": 8,
                    "page_end": 9,
                },
            ]
        ),
    ]

    outline = service.get_document_outline(document_id)

    assert outline is not None
    assert outline.document_id == document_id
    assert [section.heading for section in outline.sections] == ["Intro", "Appendix"]
    assert [child.heading for child in outline.sections[0].children] == ["Background"]


def test_get_ingest_job_returns_model_when_row_exists() -> None:
    service, connection, _, _ = _make_service()
    ingest_job_id = uuid4()
    document_id = uuid4()
    started_at = datetime(2026, 3, 18, 10, 0, tzinfo=UTC)
    finished_at = datetime(2026, 3, 18, 10, 2, tzinfo=UTC)
    connection.execute.return_value = _result(
        one_or_none={
            "id": ingest_job_id,
            "document_id": document_id,
            "status": "failed",
            "failure_code": "parse_failed",
            "failure_message": "parser broke",
            "warnings": ["parser_fallback_used"],
            "started_at": started_at,
            "finished_at": finished_at,
            "trigger": "replace",
        }
    )

    job = service.get_ingest_job(ingest_job_id)

    assert job is not None
    assert job.id == ingest_job_id
    assert job.document_id == document_id
    assert job.warnings == ["parser_fallback_used"]
    assert job.trigger == "replace"


def test_replace_document_raises_not_found_and_cleans_up_stored_artifact() -> None:
    service, connection, queue, storage = _make_service()
    document_id = uuid4()
    update_result = MagicMock()
    update_result.rowcount = 0
    connection.execute.return_value = update_result
    storage.store_file.return_value = SimpleNamespace(
        storage_ref="documents/missing/source.pdf",
        checksum="abc123",
    )

    with pytest.raises(DocumentNotFoundError, match="document not found"):
        service.replace_document(
            document_id,
            filename="replacement.pdf",
            content_type="application/pdf",
            upload=BytesIO(b"%PDF-1.4\nreplacement"),
            title="Replacement",
        )

    storage.delete.assert_called_once_with("documents/missing/source.pdf")
    queue.enqueue_ingest.assert_not_called()


def test_create_document_uses_filename_stem_as_default_title() -> None:
    service, _, _, _ = _make_service()
    ingest_job_id = uuid4()

    with patch.object(
        service,
        "_store_and_enqueue_document",
        return_value=SimpleNamespace(ingest_job_id=ingest_job_id, status="queued"),
    ) as store_and_enqueue:
        response = service.create_document(
            filename="macro-outlook.pdf",
            content_type="application/pdf",
            upload=BytesIO(b"%PDF-1.4\nbody"),
        )

    assert response.ingest_job_id == ingest_job_id
    assert response.status == "queued"
    assert store_and_enqueue.call_args.kwargs["title"] == "macro-outlook"
    assert store_and_enqueue.call_args.kwargs["trigger"] == "upload"
