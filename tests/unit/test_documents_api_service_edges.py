from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from paper_context.ingestion.api import DocumentsApiService, InvalidCursorError
from paper_context.pagination import decode_cursor, encode_cursor, fingerprint_payload

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


def _make_service() -> tuple[DocumentsApiService, MagicMock, MagicMock, MagicMock, MagicMock]:
    engine = MagicMock()
    connection = MagicMock()
    ctx = MagicMock()
    ctx.__enter__.return_value = connection
    ctx.__exit__.return_value = None
    engine.begin.return_value = ctx
    queue = MagicMock()
    storage = MagicMock()
    service = DocumentsApiService(engine=engine, queue=queue, storage=storage)
    return service, engine, connection, queue, storage


def test_decode_document_cursor_accepts_keyset_cursor() -> None:
    service, _, _, _, _ = _make_service()
    updated_at = datetime(2026, 3, 18, 10, 0, tzinfo=UTC)
    fingerprint = fingerprint_payload({"kind": "documents:list"})
    cursor = encode_cursor(
        {
            "kind": "documents:list",
            "fingerprint": fingerprint,
            "updated_at": updated_at.isoformat(),
            "document_id": "11111111-1111-1111-1111-111111111111",
        }
    )

    assert service._decode_document_cursor(
        cursor=cursor,
        kind="documents:list",
        fingerprint=fingerprint,
    ) == (updated_at, UUID("11111111-1111-1111-1111-111111111111"))


def test_decode_document_cursor_rejects_invalid_and_mismatched_cursor() -> None:
    service, _, _, _, _ = _make_service()
    fingerprint = fingerprint_payload({"kind": "documents:list"})
    bad_kind_cursor = encode_cursor(
        {
            "kind": "documents:search",
            "fingerprint": fingerprint,
            "updated_at": datetime.now(UTC).isoformat(),
            "document_id": str(uuid4()),
        }
    )

    with pytest.raises(InvalidCursorError, match="invalid cursor"):
        service._decode_document_cursor(
            cursor="not-base64",
            kind="documents:list",
            fingerprint=fingerprint,
        )

    with pytest.raises(InvalidCursorError, match="cursor does not match request"):
        service._decode_document_cursor(
            cursor=bad_kind_cursor,
            kind="documents:list",
            fingerprint=fingerprint,
        )


def test_list_documents_returns_paginated_pages() -> None:
    service, _, connection, _, _ = _make_service()
    document_id_1 = uuid4()
    document_id_2 = uuid4()
    updated_at_1 = datetime(2026, 3, 18, 10, 0, tzinfo=UTC)
    updated_at_2 = datetime(2026, 3, 18, 9, 0, tzinfo=UTC)
    rows = [
        {
            "document_id": document_id_1,
            "title": "First paper",
            "authors": ["Ada Lovelace"],
            "publication_year": 2024,
            "quant_tags": {"theme": "phase-1"},
            "current_status": "queued",
            "active_index_version": "index-v1",
            "updated_at": updated_at_1,
        },
        {
            "document_id": document_id_2,
            "title": "Second paper",
            "authors": [],
            "publication_year": 2025,
            "quant_tags": {},
            "current_status": "ready",
            "active_index_version": None,
            "updated_at": updated_at_2,
        },
    ]
    connection.execute.side_effect = [
        _result(all_rows=rows[:2]),
        _result(all_rows=rows[1:2]),
    ]

    first_page = service.list_documents(limit=1)
    assert [document.document_id for document in first_page.documents] == [document_id_1]
    assert first_page.next_cursor is not None
    assert decode_cursor(first_page.next_cursor)["updated_at"] == updated_at_1.isoformat()
    assert decode_cursor(first_page.next_cursor)["document_id"] == str(document_id_1)

    second_page = service.list_documents(limit=1, cursor=first_page.next_cursor)
    assert [document.document_id for document in second_page.documents] == [document_id_2]
    assert second_page.next_cursor is None


def test_list_documents_rejects_invalid_cursor() -> None:
    service, _, connection, _, _ = _make_service()
    connection.execute.return_value = _result(all_rows=[])

    with pytest.raises(InvalidCursorError, match="invalid cursor"):
        service.list_documents(limit=20, cursor="bad-cursor")


def test_get_document_tables_handles_missing_and_present_documents() -> None:
    document_id = uuid4()
    table_id = uuid4()
    section_id = uuid4()

    missing_service, _, missing_connection, _, _ = _make_service()
    missing_connection.execute.return_value = _result(one_or_none=None)

    assert missing_service.get_document_tables(document_id) is None

    present_service, _, present_connection, _, _ = _make_service()
    present_connection.execute.side_effect = [
        _result(one_or_none={"document_id": document_id, "title": "Tables paper"}),
        _result(
            all_rows=[
                {
                    "table_id": table_id,
                    "document_id": document_id,
                    "section_id": section_id,
                    "document_title": "Tables paper",
                    "section_path": ["Results", "Table 1"],
                    "caption": "Performance summary",
                    "table_type": "summary",
                    "headers_json": ["A", "B"],
                    "rows_json_count": 4,
                    "rows_json_preview_0": [1, 2],
                    "rows_json_preview_1": [3, 4],
                    "rows_json_preview_2": [5, 6],
                    "page_start": 3,
                    "page_end": 4,
                    "section_ordinal": 2,
                }
            ]
        ),
    ]

    tables = present_service.get_document_tables(document_id)
    assert tables is not None
    assert tables.document_id == document_id
    assert tables.title == "Tables paper"
    assert len(tables.tables) == 1
    record = tables.tables[0]
    assert record.table_id == table_id
    assert record.preview.row_count == 4
    assert record.preview.rows == [["1", "2"], ["3", "4"], ["5", "6"]]
    assert record.section_path == ["Results", "Table 1"]


def test_get_table_handles_missing_and_present_tables() -> None:
    document_id = uuid4()
    table_id = uuid4()
    section_id = uuid4()

    missing_service, _, missing_connection, _, _ = _make_service()
    missing_connection.execute.return_value = _result(one_or_none=None)

    assert missing_service.get_table(table_id) is None

    present_service, _, present_connection, _, _ = _make_service()
    present_connection.execute.return_value = _result(
        one_or_none={
            "table_id": table_id,
            "document_id": document_id,
            "section_id": section_id,
            "document_title": "Tables paper",
            "section_path": ["Appendix"],
            "caption": "Appendix table",
            "table_type": "appendix",
            "headers_json": ["Col A", "Col B"],
            "rows_json": [["x", "y"], ["z", "w"]],
            "page_start": 10,
            "page_end": 11,
        }
    )

    table = present_service.get_table(table_id)
    assert table is not None
    assert table.table_id == table_id
    assert table.document_id == document_id
    assert table.row_count == 2
    assert table.headers == ["Col A", "Col B"]
    assert table.rows == [["x", "y"], ["z", "w"]]


def test_create_document_cleans_up_storage_when_enqueue_fails() -> None:
    service, _, connection, queue, storage = _make_service()
    storage.store_file.return_value = SimpleNamespace(
        storage_ref="documents/abc/source.pdf",
        checksum="abc123",
    )
    queue.enqueue_ingest.side_effect = RuntimeError("queue unavailable")

    with pytest.raises(RuntimeError, match="queue unavailable"):
        service.create_document(
            filename="paper.pdf",
            content_type="application/pdf",
            upload=BytesIO(b"%PDF-1.4\nbody"),
            title="Cleanup case",
        )

    storage.delete.assert_called_once_with("documents/abc/source.pdf")
    assert connection.execute.call_count > 0
