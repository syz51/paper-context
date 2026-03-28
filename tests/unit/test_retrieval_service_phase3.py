from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from paper_context.pagination import decode_cursor, encode_cursor, fingerprint_payload
from paper_context.retrieval.service import (
    RetrievalService,
    _ActiveRunSelection,
    _Candidate,
    _PaginationComputation,
)
from paper_context.retrieval.types import (
    PassageResult,
    RerankItem,
    RetrievalError,
    TablePreview,
)

pytestmark = pytest.mark.unit


def _service(
    connection: MagicMock | None = None,
    *,
    reranker_client: object | None = None,
) -> RetrievalService:
    return RetrievalService(
        connection_factory=lambda: nullcontext(connection or MagicMock()),
        active_index_version="mvp-v1",
        reranker_client=reranker_client,  # type: ignore[arg-type]
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


def _passage_candidate(
    *,
    passage_id: UUID,
    retrieval_index_run_id: UUID,
    score: float = 0.0,
) -> _Candidate:
    candidate = _Candidate(
        entity_kind="passage",
        entity_id=passage_id,
        document_id=UUID("11111111-1111-1111-1111-111111111111"),
        section_id=UUID("22222222-2222-2222-2222-222222222222"),
        document_title="Paper",
        section_path=("Methods",),
        page_start=1,
        page_end=1,
        retrieval_index_run_id=retrieval_index_run_id,
        index_version="mvp-v1",
        parser_source="docling",
        warnings=("parser_fallback_used",),
        rerank_text=f"passage-{passage_id}",
        passage_id=passage_id,
        body_text=f"passage-{passage_id}",
        chunk_ordinal=1,
    )
    candidate.score = score
    candidate.sparse_rank_score = score
    return candidate


def _table_candidate(
    *,
    table_id: UUID,
    retrieval_index_run_id: UUID,
    score: float = 0.0,
) -> _Candidate:
    candidate = _Candidate(
        entity_kind="table",
        entity_id=table_id,
        document_id=UUID("11111111-1111-1111-1111-111111111111"),
        section_id=UUID("22222222-2222-2222-2222-222222222222"),
        document_title="Paper",
        section_path=("Results",),
        page_start=1,
        page_end=1,
        retrieval_index_run_id=retrieval_index_run_id,
        index_version="mvp-v1",
        parser_source="docling",
        warnings=(),
        rerank_text=f"table-{table_id}",
        table_id=table_id,
        caption=f"table-{table_id}",
        table_type="lexical",
        preview=TablePreview(headers=("A",), rows=(("1",),), row_count=1),
    )
    candidate.score = score
    candidate.sparse_rank_score = score
    return candidate


class _RecordingReranker:
    provider = "recording"
    model = "recording-v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankItem]:
        del query, top_n
        self.calls.append(list(documents))
        return [
            RerankItem(index=index, score=float(len(documents) - index))
            for index in range(len(documents))
        ]


def test_search_passages_page_returns_cursor_bound_to_index_version() -> None:
    service = _service()
    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_ids=(uuid4(),), index_versions=("mvp-v1",))
    )
    run_id = UUID("44444444-4444-4444-4444-444444444444")
    passage_a = _passage_candidate(
        passage_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        score=3.0,
        retrieval_index_run_id=run_id,
    )
    passage_b = _passage_candidate(
        passage_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        score=2.0,
        retrieval_index_run_id=run_id,
    )
    passage_c = _passage_candidate(
        passage_id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
        score=1.0,
        retrieval_index_run_id=run_id,
    )
    service._compute_ranked_page_results = MagicMock(  # type: ignore[method-assign]
        return_value=_PaginationComputation(
            results=tuple(
                service._candidate_to_passage_result(candidate)
                for candidate in (passage_a, passage_b, passage_c)
            ),
            exact=True,
            truncated=False,
            warnings=(),
            stop_reason="streams_exhausted",
        )
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
    first_cursor = decode_cursor(first_page.next_cursor)
    assert first_cursor["cursor_version"] == 2
    assert first_cursor["index_version"] == "mvp-v1"
    assert first_cursor["kind"] == "passages"
    assert first_cursor["offset"] == 2
    assert [item.passage_id for item in second_page.items] == [passage_c.passage_id]
    service._compute_ranked_page_results.assert_called_once()


def test_search_tables_page_returns_cursor_bound_to_index_version() -> None:
    service = _service()
    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_ids=(uuid4(),), index_versions=("mvp-v1",))
    )
    run_id = UUID("55555555-5555-5555-5555-555555555555")
    table_a = _table_candidate(
        table_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        score=3.0,
        retrieval_index_run_id=run_id,
    )
    table_b = _table_candidate(
        table_id=UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        score=2.0,
        retrieval_index_run_id=run_id,
    )
    service._compute_ranked_page_results = MagicMock(  # type: ignore[method-assign]
        return_value=_PaginationComputation(
            results=tuple(
                service._candidate_to_table_result(candidate) for candidate in (table_a, table_b)
            ),
            exact=True,
            truncated=False,
            warnings=(),
            stop_reason="streams_exhausted",
        )
    )

    first_page = service.search_tables_page(query="beta", limit=1)
    second_page = service.search_tables_page(query="beta", limit=1, cursor=first_page.next_cursor)

    assert [item.table_id for item in first_page.items] == [table_a.table_id]
    assert first_page.next_cursor is not None
    first_cursor = decode_cursor(first_page.next_cursor)
    assert first_cursor["cursor_version"] == 2
    assert first_cursor["index_version"] == "mvp-v1"
    assert first_cursor["kind"] == "tables"
    assert first_cursor["offset"] == 1
    assert [item.table_id for item in second_page.items] == [table_b.table_id]
    assert first_page.items[0].preview.headers == ("A",)
    service._compute_ranked_page_results.assert_called_once()


def test_search_passages_page_rejects_legacy_cursor() -> None:
    service = _service()
    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_ids=(uuid4(),), index_versions=("mvp-v1",))
    )
    fingerprint = fingerprint_payload(
        {
            "kind": "passages",
            "query": "alpha",
            "pagination_mode": "exact",
            "max_rerank_candidates": None,
            "max_expansion_rounds": None,
            "filters": {"document_ids": [], "publication_years": []},
        }
    )
    legacy_cursor = encode_cursor(
        {
            "kind": "passages",
            "fingerprint": fingerprint,
            "index_version": "mvp-v1",
            "score": "3.0",
            "entity_id": str(UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")),
        }
    )

    with pytest.raises(RetrievalError, match="cursor is no longer supported"):
        service.search_passages_page(query="alpha", limit=1, cursor=legacy_cursor)


def test_search_passages_page_reuses_ranked_snapshot_on_later_pages() -> None:
    reranker = _RecordingReranker()
    service = _service(reranker_client=reranker)
    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_ids=(uuid4(),), index_versions=("mvp-v1",))
    )
    run_id = UUID("66666666-6666-6666-6666-666666666666")
    candidates = [
        _passage_candidate(
            passage_id=UUID(int=index + 1),
            retrieval_index_run_id=run_id,
        )
        for index in range(3)
    ]
    service._load_sparse_passage_candidates = MagicMock(return_value=candidates)  # type: ignore[method-assign]

    first_page = service.search_passages_page(query="alpha", limit=2)
    second_page = service.search_passages_page(
        query="alpha", limit=2, cursor=first_page.next_cursor
    )

    assert [item.passage_id for item in first_page.items] == [
        candidates[0].passage_id,
        candidates[1].passage_id,
    ]
    assert [item.passage_id for item in second_page.items] == [candidates[2].passage_id]
    assert len(reranker.calls) == 1
    assert len(reranker.calls[0]) == 3
    service._load_sparse_passage_candidates.assert_called_once()


def test_search_passages_page_bounded_mode_reranks_only_bounded_shortlist() -> None:
    reranker = _RecordingReranker()
    service = _service(reranker_client=reranker)
    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_ids=(uuid4(),), index_versions=("mvp-v1",))
    )
    run_id = UUID("66666666-6666-6666-6666-666666666666")
    candidates = [
        _passage_candidate(
            passage_id=UUID(int=index + 1),
            retrieval_index_run_id=run_id,
            score=float(100 - index),
        )
        for index in range(30)
    ]

    service._load_sparse_passage_candidates = MagicMock(  # type: ignore[method-assign]
        return_value=candidates
    )

    page = service.search_passages_page(
        query="alpha",
        limit=2,
        pagination_mode="bounded",
        max_rerank_candidates=3,
        max_expansion_rounds=1,
    )

    assert [item.passage_id for item in page.items] == [
        candidates[0].passage_id,
        candidates[1].passage_id,
    ]
    assert len(reranker.calls) == 1
    assert len(reranker.calls[0]) == 3
    assert page.exact is False
    assert page.truncated is True
    assert page.warnings == ("bounded_pagination_truncated",)


def test_search_tables_page_streams_sparse_expansion_with_search_after() -> None:
    service = _service()
    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_ids=(uuid4(),), index_versions=("mvp-v1",))
    )
    run_id = UUID("77777777-7777-7777-7777-777777777777")
    candidates = [
        _table_candidate(
            table_id=UUID(int=10_000 + index),
            retrieval_index_run_id=run_id,
            score=float(100 - index),
        )
        for index in range(25)
    ]

    first_anchor = candidates[19]

    def _slice_tables(*args, **kwargs):
        del args
        limit = kwargs["limit"]
        after_score = kwargs["after_score"]
        after_entity_id = kwargs["after_entity_id"]
        if after_score is None:
            return candidates[:limit]
        assert after_score == first_anchor.sparse_rank_score
        assert after_entity_id == first_anchor.entity_id
        return candidates[20 : 20 + limit]

    service._load_sparse_table_candidates = MagicMock(side_effect=_slice_tables)  # type: ignore[method-assign]
    page = service.search_tables_page(query="beta", limit=1)

    assert [item.table_id for item in page.items] == [candidates[0].table_id]
    assert [
        call.kwargs["after_score"] for call in service._load_sparse_table_candidates.call_args_list
    ] == [
        None,
        first_anchor.sparse_rank_score,
    ]
    assert page.next_cursor is not None
    assert decode_cursor(page.next_cursor)["offset"] == 1


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
            exact=True,
            truncated=False,
            warnings=(),
        )

    mismatched_cursor = encode_cursor(
        {
            "cursor_version": 2,
            "kind": "tables",
            "fingerprint": "fingerprint",
            "index_version": "mvp-v1",
            "offset": 0,
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
            exact=True,
            truncated=False,
            warnings=(),
        )

    stale_cursor = encode_cursor(
        {
            "cursor_version": 2,
            "kind": "passages",
            "fingerprint": "fingerprint",
            "index_version": "mvp-v0",
            "offset": 0,
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
            exact=True,
            truncated=False,
            warnings=(),
        )

    invalid_offset_cursor = encode_cursor(
        {
            "cursor_version": 2,
            "kind": "passages",
            "fingerprint": "fingerprint",
            "index_version": "mvp-v1",
            "offset": -1,
        }
    )
    with pytest.raises(RetrievalError, match="cursor offset is invalid"):
        service._paginate_ranked_results(
            kind="passages",
            results=results,
            limit=1,
            cursor=invalid_offset_cursor,
            active_runs=active_runs,
            fingerprint="fingerprint",
            exact=True,
            truncated=False,
            warnings=(),
        )
