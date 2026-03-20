from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from paper_context.pagination import decode_cursor, encode_cursor
from paper_context.retrieval.service import RetrievalService, _ActiveRunSelection
from paper_context.retrieval.types import PassageResult, RetrievalError, TablePreview, TableResult

pytestmark = pytest.mark.unit


def _service(connection: MagicMock | None = None) -> RetrievalService:
    return RetrievalService(
        connection_factory=lambda: nullcontext(connection or MagicMock()),
        active_index_version="mvp-v1",
    )


def _passage(
    *,
    passage_id: UUID,
    score: float,
    retrieval_index_run_id: UUID,
) -> PassageResult:
    return PassageResult(
        passage_id=passage_id,
        document_id=UUID("11111111-1111-1111-1111-111111111111"),
        section_id=UUID("22222222-2222-2222-2222-222222222222"),
        document_title="Paper",
        section_path=("Methods",),
        text=f"passage-{passage_id}",
        score=score,
        retrieval_modes=("sparse",),
        page_start=1,
        page_end=1,
        index_version="mvp-v1",
        retrieval_index_run_id=retrieval_index_run_id,
        parser_source="docling",
        warnings=("parser_fallback_used",),
    )


def test_search_passages_page_returns_cursor_bound_to_index_version() -> None:
    service = _service()
    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_ids=(uuid4(),), index_versions=("mvp-v1",))
    )
    passage_a = _passage(
        passage_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        score=3.0,
        retrieval_index_run_id=UUID("44444444-4444-4444-4444-444444444444"),
    )
    passage_b = _passage(
        passage_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        score=2.0,
        retrieval_index_run_id=UUID("44444444-4444-4444-4444-444444444444"),
    )
    passage_c = _passage(
        passage_id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
        score=1.0,
        retrieval_index_run_id=UUID("44444444-4444-4444-4444-444444444444"),
    )
    service._load_ranked_passage_results = MagicMock(  # type: ignore[method-assign]
        return_value=([passage_a, passage_b, passage_c], 0, 0)
    )

    first_page = service.search_passages_page(query="alpha", limit=2)
    second_page = service.search_passages_page(
        query="alpha", limit=2, cursor=first_page.next_cursor
    )

    assert [item.passage_id for item in first_page.items] == [
        passage_a.passage_id,
        passage_b.passage_id,
    ]
    assert first_page.next_cursor is not None
    assert decode_cursor(first_page.next_cursor)["index_version"] == "mvp-v1"
    assert decode_cursor(first_page.next_cursor)["kind"] == "passages"
    assert [item.passage_id for item in second_page.items] == [passage_c.passage_id]
    assert all(
        call.kwargs["limit"] is None for call in service._load_ranked_passage_results.call_args_list
    )


def test_search_tables_page_returns_cursor_bound_to_index_version() -> None:
    service = _service()
    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_ids=(uuid4(),), index_versions=("mvp-v1",))
    )
    table_a = TableResult(
        table_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        document_id=UUID("11111111-1111-1111-1111-111111111111"),
        section_id=UUID("22222222-2222-2222-2222-222222222222"),
        document_title="Paper",
        section_path=("Results",),
        caption="table-a",
        table_type="lexical",
        preview=TablePreview(headers=("A",), rows=(("1",),), row_count=1),
        score=3.0,
        retrieval_modes=("sparse",),
        page_start=1,
        page_end=1,
        index_version="mvp-v1",
        retrieval_index_run_id=UUID("55555555-5555-5555-5555-555555555555"),
        parser_source="docling",
        warnings=(),
    )
    table_b = TableResult(
        table_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        document_id=UUID("11111111-1111-1111-1111-111111111111"),
        section_id=UUID("22222222-2222-2222-2222-222222222222"),
        document_title="Paper",
        section_path=("Results",),
        caption="table-b",
        table_type="lexical",
        preview=TablePreview(headers=("A",), rows=(("2",),), row_count=1),
        score=2.0,
        retrieval_modes=("sparse",),
        page_start=1,
        page_end=1,
        index_version="mvp-v1",
        retrieval_index_run_id=UUID("55555555-5555-5555-5555-555555555555"),
        parser_source="docling",
        warnings=(),
    )
    service._load_ranked_table_results = MagicMock(  # type: ignore[method-assign]
        return_value=([table_a, table_b], 0, 0)
    )

    first_page = service.search_tables_page(query="beta", limit=1)
    second_page = service.search_tables_page(query="beta", limit=1, cursor=first_page.next_cursor)

    assert [item.table_id for item in first_page.items] == [table_a.table_id]
    assert first_page.next_cursor is not None
    assert decode_cursor(first_page.next_cursor)["index_version"] == "mvp-v1"
    assert decode_cursor(first_page.next_cursor)["kind"] == "tables"
    assert [item.table_id for item in second_page.items] == [table_b.table_id]
    assert first_page.items[0].preview.headers == ("A",)
    assert all(
        call.kwargs["limit"] is None for call in service._load_ranked_table_results.call_args_list
    )


def test_get_table_returns_full_structured_payload() -> None:
    connection = MagicMock()
    result = MagicMock()
    result.mappings.return_value.one_or_none.return_value = {
        "table_id": UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        "document_id": UUID("11111111-1111-1111-1111-111111111111"),
        "section_id": UUID("22222222-2222-2222-2222-222222222222"),
        "document_title": "Paper",
        "section_path": ["Results"],
        "caption": "Results table",
        "table_type": "lexical",
        "headers_json": ["A", "B"],
        "rows_json": [["1", "2"], ["3", "4"]],
        "page_start": 1,
        "page_end": 2,
        "retrieval_index_run_id": UUID("55555555-5555-5555-5555-555555555555"),
        "index_version": "mvp-v1",
        "parser_source": "docling",
        "warnings": ["parser_fallback_used"],
    }
    connection.execute.return_value = result
    service = _service(connection)

    table = service.get_table(table_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"))

    assert table is not None
    assert table.headers == ("A", "B")
    assert table.rows == (("1", "2"), ("3", "4"))
    assert table.row_count == 2
    assert table.parser_source == "docling"
    assert table.warnings == ("parser_fallback_used",)


def test_get_table_returns_none_when_table_is_missing() -> None:
    connection = MagicMock()
    result = MagicMock()
    result.mappings.return_value.one_or_none.return_value = None
    connection.execute.return_value = result
    service = _service(connection)

    assert service.get_table(table_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")) is None


def test_get_passage_context_returns_neighbors_and_warning() -> None:
    connection = MagicMock()
    first_result = MagicMock()
    first_result.mappings.return_value.one_or_none.return_value = {
        "passage_id": UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        "document_id": UUID("11111111-1111-1111-1111-111111111111"),
        "section_id": UUID("22222222-2222-2222-2222-222222222222"),
        "body_text": "selected",
        "chunk_ordinal": 1,
        "page_start": 1,
        "page_end": 1,
        "revision_id": UUID("33333333-3333-3333-3333-333333333333"),
        "document_title": "Paper",
        "section_path": ["Methods"],
        "retrieval_index_run_id": UUID("55555555-5555-5555-5555-555555555555"),
        "index_version": "mvp-v1",
        "parser_source": "docling",
        "warnings": ["parser_fallback_used"],
    }
    second_result = MagicMock()
    second_result.mappings.return_value.all.return_value = [
        {
            "passage_id": UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            "body_text": "selected",
            "chunk_ordinal": 1,
            "page_start": 1,
            "page_end": 1,
        },
        {
            "passage_id": UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            "body_text": "sibling",
            "chunk_ordinal": 2,
            "page_start": 1,
            "page_end": 1,
        },
        {
            "passage_id": UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
            "body_text": "sibling-2",
            "chunk_ordinal": 3,
            "page_start": 2,
            "page_end": 2,
        },
    ]
    connection.execute.side_effect = [first_result, second_result]
    service = _service(connection)

    context = service.get_passage_context(
        passage_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        before=0,
        after=1,
    )

    assert context is not None
    assert context.passage.passage_id == UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    assert [item.passage_id for item in context.context_passages] == [
        UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
    ]
    assert context.warnings == ("parser_fallback_used", "parent_context_truncated")


def test_get_passage_context_returns_none_when_target_or_selected_row_is_missing() -> None:
    connection = MagicMock()
    missing_target = MagicMock()
    missing_target.mappings.return_value.one_or_none.return_value = None
    connection.execute.return_value = missing_target
    service = _service(connection)

    assert (
        service.get_passage_context(
            passage_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            before=1,
            after=1,
        )
        is None
    )

    selected_row = MagicMock()
    selected_row.mappings.return_value.one_or_none.return_value = {
        "passage_id": UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        "document_id": UUID("11111111-1111-1111-1111-111111111111"),
        "section_id": UUID("22222222-2222-2222-2222-222222222222"),
        "body_text": "selected",
        "chunk_ordinal": 1,
        "page_start": 1,
        "page_end": 1,
        "revision_id": UUID("33333333-3333-3333-3333-333333333333"),
        "document_title": "Paper",
        "section_path": ["Methods"],
        "retrieval_index_run_id": UUID("55555555-5555-5555-5555-555555555555"),
        "index_version": "mvp-v1",
        "parser_source": "docling",
        "warnings": [],
    }
    section_rows = MagicMock()
    section_rows.mappings.return_value.all.return_value = [
        {
            "passage_id": UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
            "body_text": "sibling",
            "chunk_ordinal": 2,
            "page_start": 1,
            "page_end": 1,
        }
    ]
    connection.execute.side_effect = [selected_row, section_rows]

    assert (
        service.get_passage_context(
            passage_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            before=1,
            after=1,
        )
        is None
    )


def test_paginate_ranked_results_rejects_bad_cursor_variants() -> None:
    service = _service()
    active_runs = _ActiveRunSelection(
        run_ids=(UUID("44444444-4444-4444-4444-444444444444"),),
        index_versions=("mvp-v1",),
    )
    results = (
        _passage(
            passage_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            score=2.0,
            retrieval_index_run_id=UUID("44444444-4444-4444-4444-444444444444"),
        ),
    )

    with pytest.raises(RetrievalError, match="cursor"):
        service._paginate_ranked_results(
            kind="passages",
            results=results,
            limit=1,
            cursor="not-a-valid-cursor",
            active_runs=active_runs,
            fingerprint="fingerprint",
        )

    mismatched_cursor = encode_cursor(
        {
            "kind": "tables",
            "fingerprint": "fingerprint",
            "index_version": "mvp-v1",
            "score": "2.0",
            "entity_id": str(results[0].passage_id),
        }
    )
    with pytest.raises(RetrievalError, match="cursor does not match request"):
        service._paginate_ranked_results(
            kind="passages",
            results=results,
            limit=1,
            cursor=mismatched_cursor,
            active_runs=active_runs,
            fingerprint="fingerprint",
        )

    stale_cursor = encode_cursor(
        {
            "kind": "passages",
            "fingerprint": "fingerprint",
            "index_version": "mvp-v0",
            "score": "2.0",
            "entity_id": str(results[0].passage_id),
        }
    )
    with pytest.raises(RetrievalError, match="index version is no longer active"):
        service._paginate_ranked_results(
            kind="passages",
            results=results,
            limit=1,
            cursor=stale_cursor,
            active_runs=active_runs,
            fingerprint="fingerprint",
        )
