from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from paper_context.retrieval.service import (
    DocumentRetrievalIndexer,
    RetrievalService,
    _ActiveRunSelection,
    _build_table_preview,
    _Candidate,
    _CandidateExpansionState,
    _dedupe_modes_from_results,
    _dedupe_warnings,
    _json_dumps,
    _normalize_modes,
    _PassageIndexRow,
    _retrieval_index_run_id,
    _TableIndexRow,
    _vector_literal,
)
from paper_context.retrieval.types import (
    DocumentSummary,
    EmbeddingBatch,
    EmbeddingClient,
    MixedIndexVersionError,
    PassageResult,
    RerankerClient,
    RerankItem,
    RetrievalError,
    RetrievalFilters,
    TablePreview,
    TableResult,
)

pytestmark = pytest.mark.unit


class _StubEmbeddingClient:
    provider = "stub-embed"

    def __init__(self, batches: list[EmbeddingBatch]) -> None:
        self.model = "stub-embed-model"
        self._batches = iter(batches)

    def embed(self, texts: list[str], *, input_type: str) -> EmbeddingBatch:
        del texts, input_type
        return next(self._batches)


class _StubRerankerClient:
    provider = "stub-reranker"

    def __init__(self, results: list[list[RerankItem]]) -> None:
        self.model = "stub-reranker-model"
        self._results = iter(results)

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankItem]:
        del query, documents, top_n
        return next(self._results)


def _mock_result(
    *,
    rows: list[dict[str, object]] | None = None,
    scalar_rows: list[object] | None = None,
) -> MagicMock:
    result = MagicMock()
    mappings = MagicMock()
    mappings.all.return_value = rows or []
    result.mappings.return_value = mappings
    scalars = MagicMock()
    scalars.all.return_value = scalar_rows or []
    result.scalars.return_value = scalars
    return result


def _connection(*results: MagicMock) -> MagicMock:
    connection = MagicMock()
    connection.execute.side_effect = list(results)
    return connection


def _service(
    *,
    connection: MagicMock | None = None,
    active_index_version: str | None = "mvp-v1",
    embedding_client: EmbeddingClient | None = None,
    reranker_client: RerankerClient | None = None,
) -> RetrievalService:
    factory_connection = connection or MagicMock()
    return RetrievalService(
        connection_factory=lambda: nullcontext(factory_connection),
        active_index_version=active_index_version,
        embedding_client=embedding_client,
        reranker_client=reranker_client,
    )


def _passage_candidate(
    *,
    entity_id: UUID,
    retrieval_index_run_id: UUID,
    index_version: str,
    text: str,
    modes: set[str],
    score_hint: float = 0.0,
) -> _Candidate:
    candidate = _Candidate(
        entity_kind="passage",
        entity_id=entity_id,
        document_id=uuid4(),
        section_id=uuid4(),
        document_title="Paper",
        section_path=("Methods",),
        page_start=1,
        page_end=1,
        retrieval_index_run_id=retrieval_index_run_id,
        index_version=index_version,
        warnings=("parser_fallback_used",),
        rerank_text=text,
        passage_id=entity_id,
        body_text=text,
        chunk_ordinal=1,
    )
    candidate.retrieval_modes.update(modes)
    candidate.fused_score = score_hint
    return candidate


def _table_candidate(
    *,
    entity_id: UUID,
    retrieval_index_run_id: UUID,
    index_version: str,
    text: str,
    modes: set[str],
) -> _Candidate:
    candidate = _Candidate(
        entity_kind="table",
        entity_id=entity_id,
        document_id=uuid4(),
        section_id=uuid4(),
        document_title="Paper",
        section_path=("Results",),
        page_start=2,
        page_end=2,
        retrieval_index_run_id=retrieval_index_run_id,
        index_version=index_version,
        warnings=("parent_context_truncated",),
        rerank_text=text,
        table_id=entity_id,
        caption="Results table",
        table_type="lexical",
        preview=TablePreview(headers=("A", "B"), rows=(("1", "2"), ("3", "4")), row_count=2),
    )
    candidate.retrieval_modes.update(modes)
    return candidate


def test_small_pure_helpers_cover_expected_shape() -> None:
    ingest_job_id = uuid4()

    assert _json_dumps({"a": 1}) == '{"a": 1}'
    assert _vector_literal((1.0, 2.5)) == "[1.00000000,2.50000000]"
    assert _retrieval_index_run_id(ingest_job_id=ingest_job_id) == _retrieval_index_run_id(
        ingest_job_id=ingest_job_id
    )
    assert _normalize_modes({"dense", "sparse", "other"}) == ("sparse", "dense")
    assert _dedupe_warnings(["a", "b", "a", "c"]) == ("a", "b", "c")


def test_health_summary_reports_configuration_states() -> None:
    empty = RetrievalService()
    partial = RetrievalService(
        connection_factory=lambda: nullcontext(MagicMock()),
        embedding_client=_StubEmbeddingClient(
            [
                EmbeddingBatch(
                    provider="embed",
                    model="embed-v1",
                    dimensions=1024,
                    embeddings=((0.0,) * 1024,),
                )
            ]
        ),
    )
    configured = RetrievalService(
        connection_factory=lambda: nullcontext(MagicMock()),
        active_index_version="index-v2",
        embedding_client=_StubEmbeddingClient(
            [
                EmbeddingBatch(
                    provider="embed",
                    model="embed-v1",
                    dimensions=1024,
                    embeddings=((0.0,) * 1024,),
                )
            ]
        ),
        reranker_client=_StubRerankerClient([[RerankItem(index=0, score=1.0)]]),
    )

    assert empty.health_summary() == {"status": "not-configured"}
    assert partial.health_summary() == {"status": "partially-configured"}
    assert configured.health_summary() == {
        "status": "configured",
        "embedding_provider": "stub-embed",
        "reranker_provider": "stub-reranker",
        "active_index_version": "index-v2",
    }


def test_connection_requires_factory() -> None:
    service = RetrievalService()

    with pytest.raises(RetrievalError, match="no connection factory configured"):
        service._connection()


def test_auto_active_run_selection_collapses_to_newest_index_version() -> None:
    service = _service(active_index_version=None)
    newest_run_id = uuid4()
    second_run_id = uuid4()
    legacy_run_id = uuid4()

    selection = service._resolve_active_run_selection(
        _connection(
            _mock_result(
                rows=[
                    {"id": newest_run_id, "index_version": "mvp-v2"},
                    {"id": second_run_id, "index_version": "mvp-v2"},
                    {"id": legacy_run_id, "index_version": "mvp-v1"},
                ]
            )
        ),
        filtered_document_ids=None,
    )

    assert selection.run_ids == (newest_run_id, second_run_id)
    assert selection.index_versions == ("mvp-v2",)


@pytest.mark.parametrize(
    "method_name",
    ["_search_passages_with_connection", "_search_tables_with_connection"],
)
def test_search_helpers_return_empty_on_blank_query(method_name: str) -> None:
    service = _service()
    connection = MagicMock()

    assert (
        getattr(service, method_name)(connection, query="   ", filters=RetrievalFilters(), limit=5)
        == []
    )


@pytest.mark.parametrize(
    "method_name",
    ["_search_passages_with_connection", "_search_tables_with_connection"],
)
def test_search_helpers_return_empty_when_filters_eliminate_candidates(method_name: str) -> None:
    service = _service()
    connection = MagicMock()

    assert (
        getattr(service, method_name)(
            connection,
            query="query",
            filters=RetrievalFilters(),
            limit=5,
            filtered_document_ids=(),
        )
        == []
    )


@pytest.mark.parametrize(
    "method_name",
    ["_search_passages_with_connection", "_search_tables_with_connection"],
)
def test_search_helpers_return_empty_when_active_runs_missing(method_name: str) -> None:
    service = _service()
    connection = MagicMock()

    assert (
        getattr(service, method_name)(
            connection,
            query="query",
            filters=RetrievalFilters(),
            limit=5,
            filtered_document_ids=None,
            active_runs=_ActiveRunSelection(run_ids=(), index_versions=()),
        )
        == []
    )


def test_resolve_active_run_selection_fixed_version_and_query_branch() -> None:
    connection = MagicMock()
    service = _service(active_index_version="index-v2")
    service._resolve_active_run_ids = MagicMock(  # type: ignore[method-assign]
        return_value=(uuid4(), uuid4())
    )

    fixed = service._resolve_active_run_selection(connection, filtered_document_ids=None)

    assert fixed.index_versions == ("index-v2",)
    assert len(fixed.run_ids) == 2
    service._resolve_active_run_ids.assert_called_once_with(
        connection,
        index_version="index-v2",
        filtered_document_ids=None,
    )

    rows = [
        {"id": uuid4(), "index_version": "mvp-v2"},
        {"id": uuid4(), "index_version": "mvp-v1"},
        {"id": uuid4(), "index_version": "mvp-v2"},
    ]
    query_connection = _connection(_mock_result(rows=rows))
    query_service = _service(active_index_version=None)
    filtered_document_ids = (uuid4(),)

    selection = query_service._resolve_active_run_selection(
        query_connection,
        filtered_document_ids=filtered_document_ids,
    )

    assert selection.run_ids == (rows[0]["id"], rows[2]["id"])
    assert selection.index_versions == ("mvp-v2",)
    assert query_connection.execute.call_args.args[1] == {
        "apply_document_filter": True,
        "document_ids": [filtered_document_ids[0]],
    }


def test_resolve_filtered_document_ids_and_active_run_ids() -> None:
    document_ids = (uuid4(), uuid4())
    years = (2023, 2024)
    filters = RetrievalFilters(document_ids=document_ids, publication_years=years)
    connection = _connection(_mock_result(scalar_rows=[document_ids[1], document_ids[0]]))
    service = _service(active_index_version=None)

    resolved = service._resolve_filtered_document_ids(connection, filters=filters)

    assert resolved == (document_ids[1], document_ids[0])
    assert "document_ids" in connection.execute.call_args.args[1]
    assert "publication_years" in connection.execute.call_args.args[1]

    active_connection = _connection(_mock_result(scalar_rows=[uuid4(), uuid4()]))
    active = service._resolve_active_run_ids(
        active_connection,
        index_version="mvp-v1",
        filtered_document_ids=document_ids,
    )

    assert len(active) == 2
    assert active_connection.execute.call_args.args[1] == {
        "apply_document_filter": True,
        "index_version": "mvp-v1",
        "document_ids": [document_ids[0], document_ids[1]],
    }


def test_sparse_and_dense_candidate_loaders_cover_rows_and_empty_results() -> None:
    service = _service(
        embedding_client=_StubEmbeddingClient(
            [
                EmbeddingBatch(
                    provider="stub",
                    model="stub",
                    dimensions=1024,
                    embeddings=((0.1,) * 1024,),
                )
            ]
        )
    )
    passage_row = {
        "passage_id": uuid4(),
        "document_id": uuid4(),
        "revision_id": uuid4(),
        "section_id": uuid4(),
        "chunk_ordinal": 1,
        "body_text": "Body",
        "contextualized_text": "Context",
        "page_start": 1,
        "page_end": 2,
        "document_title": "Paper",
        "section_path": ["Intro"],
        "retrieval_index_run_id": uuid4(),
        "index_version": "mvp-v1",
        "warnings": ["warn"],
    }
    table_row = {
        "table_id": uuid4(),
        "document_id": uuid4(),
        "section_id": uuid4(),
        "caption": "Caption",
        "table_type": "lexical",
        "headers_json": ["A", 2],
        "rows_json": [["x", 1], ["y", 2]],
        "page_start": 3,
        "page_end": 4,
        "document_title": "Paper",
        "section_path": ["Results"],
        "retrieval_index_run_id": uuid4(),
        "index_version": "mvp-v1",
        "warnings": ["warn"],
        "search_text": "Table text",
        "rank_score": 1.0,
    }

    sparse_passages = service._load_sparse_passage_candidates(
        _connection(_mock_result(rows=[passage_row])),
        query="search",
        retrieval_index_run_ids=(uuid4(),),
        filtered_document_ids=(uuid4(),),
        limit=5,
    )
    sparse_tables = service._load_sparse_table_candidates(
        _connection(_mock_result(rows=[table_row])),
        query="search",
        retrieval_index_run_ids=(uuid4(),),
        filtered_document_ids=(uuid4(),),
        limit=5,
    )
    assert sparse_passages[0].rerank_text == "Context"
    assert sparse_tables[0].preview is not None
    assert sparse_tables[0].preview.headers == ("A", "2")
    assert sparse_tables[0].preview.rows == (("x", "1"), ("y", "2"))

    unlimited_passages = service._load_sparse_passage_candidates(
        _connection(_mock_result(rows=[passage_row])),
        query="search",
        retrieval_index_run_ids=(uuid4(),),
        filtered_document_ids=None,
        limit=None,
    )
    unlimited_tables = service._load_sparse_table_candidates(
        _connection(_mock_result(rows=[table_row])),
        query="search",
        retrieval_index_run_ids=(uuid4(),),
        filtered_document_ids=None,
        limit=None,
    )
    assert len(unlimited_passages) == 1
    assert len(unlimited_tables) == 1

    assert (
        service._load_dense_passage_candidates(
            MagicMock(),
            query="search",
            retrieval_index_run_ids=(uuid4(),),
            filtered_document_ids=None,
            limit=5,
        )
        == []
    )

    empty_dense_service = _service(
        embedding_client=_StubEmbeddingClient(
            [
                EmbeddingBatch(
                    provider="stub",
                    model="stub",
                    dimensions=1024,
                    embeddings=(),
                )
            ]
        )
    )
    assert (
        empty_dense_service._load_dense_passage_candidates(
            MagicMock(),
            query="search",
            retrieval_index_run_ids=(uuid4(),),
            filtered_document_ids=None,
            limit=5,
        )
        == []
    )


def test_ranked_table_results_fuse_sparse_and_dense_candidates() -> None:
    service = _service(reranker_client=None)
    run_id = uuid4()
    shared = _table_candidate(
        entity_id=uuid4(),
        retrieval_index_run_id=run_id,
        index_version="mvp-v1",
        text="shared table",
        modes={"sparse"},
    )
    sparse_only = _table_candidate(
        entity_id=uuid4(),
        retrieval_index_run_id=run_id,
        index_version="mvp-v1",
        text="sparse table",
        modes={"sparse"},
    )
    shared_dense = _table_candidate(
        entity_id=shared.entity_id,
        retrieval_index_run_id=run_id,
        index_version="mvp-v1",
        text="shared table",
        modes={"dense"},
    )

    service._load_sparse_table_candidates = MagicMock(return_value=[shared, sparse_only])  # type: ignore[method-assign]
    service._load_dense_table_candidates = MagicMock(return_value=[shared_dense])  # type: ignore[method-assign]

    results, sparse_count, dense_count = service._load_ranked_table_results(
        MagicMock(),
        query="search",
        limit=5,
        filtered_document_ids=None,
        active_runs=_ActiveRunSelection(run_ids=(run_id,), index_versions=("mvp-v1",)),
        sparse_candidate_limit=5,
        dense_candidate_limit=5,
    )

    assert sparse_count == 2
    assert dense_count == 1
    assert len(results) == 2
    assert results[0].table_id == shared.table_id
    assert results[0].retrieval_modes == ("sparse", "dense")
    assert results[1].table_id == sparse_only.table_id
    assert results[1].retrieval_modes == ("sparse",)


def test_dense_loader_rejects_bad_embedding_dimensions() -> None:
    service = _service(
        embedding_client=_StubEmbeddingClient(
            [
                EmbeddingBatch(
                    provider="stub",
                    model="stub",
                    dimensions=128,
                    embeddings=((0.1,) * 128,),
                )
            ]
        )
    )

    with pytest.raises(RetrievalError, match="query embedding dimension mismatch"):
        service._load_dense_passage_candidates(
            MagicMock(),
            query="search",
            retrieval_index_run_ids=(uuid4(),),
            filtered_document_ids=None,
            limit=5,
        )


def test_dense_loader_returns_empty_without_embedding_client() -> None:
    assert (
        _service(embedding_client=None)._load_dense_passage_candidates(
            MagicMock(),
            query="search",
            retrieval_index_run_ids=(uuid4(),),
            filtered_document_ids=None,
            limit=5,
        )
        == []
    )


def test_fuse_candidates_and_rerank_helpers() -> None:
    service = _service(
        reranker_client=_StubRerankerClient(
            [[RerankItem(index=1, score=2.0), RerankItem(index=0, score=1.0)]]
        )
    )
    run_id = uuid4()
    shared = _passage_candidate(
        entity_id=uuid4(),
        retrieval_index_run_id=run_id,
        index_version="mvp-v1",
        text="shared",
        modes={"sparse"},
        score_hint=0.1,
    )
    sparse_only = _passage_candidate(
        entity_id=uuid4(),
        retrieval_index_run_id=run_id,
        index_version="mvp-v1",
        text="sparse",
        modes={"sparse"},
        score_hint=0.05,
    )
    dense_only = _passage_candidate(
        entity_id=uuid4(),
        retrieval_index_run_id=run_id,
        index_version="mvp-v1",
        text="dense",
        modes={"dense"},
        score_hint=0.07,
    )

    fused = service._fuse_candidates(
        [shared, sparse_only],
        [shared, dense_only],
        fused_limit=10,
    )

    assert fused[0].entity_id == shared.entity_id
    assert fused[0].retrieval_modes == {"sparse", "dense"}
    assert {candidate.entity_id for candidate in fused} == {
        shared.entity_id,
        sparse_only.entity_id,
        dense_only.entity_id,
    }

    reranked = service._rerank_candidates(
        query="query",
        candidates=fused,
        limit=2,
    )
    assert reranked[0] is fused[1]
    assert reranked[1] is fused[0]
    assert reranked[0].score == 2.0
    assert reranked[1].score == 1.0


def test_rerank_helpers_handle_empty_and_bad_indexes() -> None:
    service = _service(reranker_client=None)
    assert service._rerank_candidates(query="query", candidates=[], limit=3) == []

    candidate = _passage_candidate(
        entity_id=uuid4(),
        retrieval_index_run_id=uuid4(),
        index_version="mvp-v1",
        text="text",
        modes={"sparse"},
        score_hint=0.42,
    )
    reranked = service._rerank_candidates(query="query", candidates=[candidate], limit=1)
    assert reranked[0].score == pytest.approx(0.42)

    bad_service = _service(reranker_client=_StubRerankerClient([[RerankItem(index=9, score=1.0)]]))
    with pytest.raises(RetrievalError, match="out-of-range candidate index"):
        bad_service._rerank_candidates(query="query", candidates=[candidate], limit=1)


def test_certify_fused_shortlist_stops_before_sparse_exhaustion_when_boundary_is_safe() -> None:
    service = _service(reranker_client=None)
    run_id = uuid4()
    state = _CandidateExpansionState(dense_exhausted=True)
    batch = [
        _passage_candidate(
            entity_id=UUID(int=index + 1),
            retrieval_index_run_id=run_id,
            index_version="mvp-v1",
            text=f"candidate-{index}",
            modes={"sparse"},
        )
        for index in range(8)
    ]

    service._merge_candidate_batch(state=state, mode="sparse", offset=0, batch=batch)
    state.sparse_count = len(batch)

    shortlist = service._certify_fused_shortlist(state=state, target_count=3)

    assert shortlist is not None
    assert [candidate.entity_id for candidate in shortlist] == [
        UUID(int=1),
        UUID(int=2),
        UUID(int=3),
    ]


def test_certify_fused_shortlist_waits_when_partial_candidate_can_still_enter() -> None:
    service = _service(reranker_client=None)
    run_id = uuid4()
    state = _CandidateExpansionState()
    sparse_batch = [
        _passage_candidate(
            entity_id=UUID(int=index + 1),
            retrieval_index_run_id=run_id,
            index_version="mvp-v1",
            text=f"sparse-{index}",
            modes={"sparse"},
        )
        for index in range(3)
    ]
    dense_only = _passage_candidate(
        entity_id=UUID(int=10),
        retrieval_index_run_id=run_id,
        index_version="mvp-v1",
        text="dense-only",
        modes={"dense"},
    )

    service._merge_candidate_batch(state=state, mode="sparse", offset=0, batch=sparse_batch)
    state.sparse_count = len(sparse_batch)
    service._merge_candidate_batch(state=state, mode="dense", offset=0, batch=[dense_only])
    state.dense_count = 1

    assert service._certify_fused_shortlist(state=state, target_count=2) is None


def test_row_and_result_conversions_round_trip() -> None:
    passage_row = {
        "passage_id": uuid4(),
        "document_id": uuid4(),
        "section_id": uuid4(),
        "chunk_ordinal": 2,
        "body_text": "Body",
        "contextualized_text": "Context",
        "page_start": 1,
        "page_end": 2,
        "document_title": "Paper",
        "authors": ["Ada"],
        "abstract": "Abs",
        "publication_year": 2024,
        "section_path": ["Intro"],
        "warnings": ["warn"],
        "retrieval_index_run_id": uuid4(),
        "index_version": "mvp-v1",
    }
    table_row = {
        "table_id": uuid4(),
        "document_id": uuid4(),
        "section_id": uuid4(),
        "caption": "Caption",
        "table_type": "lexical",
        "headers_json": ["A", 2],
        "rows_json": [["x", 1], ["y", 2]],
        "page_start": 3,
        "page_end": 4,
        "document_title": "Paper",
        "section_path": ["Results"],
        "warnings": ["warn"],
        "retrieval_index_run_id": uuid4(),
        "index_version": "mvp-v1",
        "search_text": "Table text",
    }

    passage_candidate = _service()._row_to_passage_candidate(passage_row)
    table_candidate = _service()._row_to_table_candidate(table_row)

    assert passage_candidate.warnings == ("warn",)
    assert passage_candidate.section_path == ("Intro",)
    assert table_candidate.preview == TablePreview(
        headers=("A", "2"),
        rows=(("x", "1"), ("y", "2")),
        row_count=2,
    )

    passage_result = _service()._candidate_to_passage_result(passage_candidate)
    table_result = _service()._candidate_to_table_result(table_candidate)

    assert passage_result.text == "Body"
    assert passage_result.retrieval_modes == ()
    assert table_result.caption == "Caption"
    assert table_result.retrieval_modes == ()

    broken_passage = _Candidate(
        entity_kind="passage",
        entity_id=uuid4(),
        document_id=uuid4(),
        section_id=uuid4(),
        document_title="Paper",
        section_path=(),
        page_start=None,
        page_end=None,
        retrieval_index_run_id=uuid4(),
        index_version="mvp-v1",
        warnings=(),
        rerank_text="text",
    )
    broken_table = _Candidate(
        entity_kind="table",
        entity_id=uuid4(),
        document_id=uuid4(),
        section_id=uuid4(),
        document_title="Paper",
        section_path=(),
        page_start=None,
        page_end=None,
        retrieval_index_run_id=uuid4(),
        index_version="mvp-v1",
        warnings=(),
        rerank_text="text",
    )

    with pytest.raises(RetrievalError, match="passage payload"):
        _service()._candidate_to_passage_result(broken_passage)
    with pytest.raises(RetrievalError, match="table payload"):
        _service()._candidate_to_table_result(broken_table)


def test_parent_sections_and_document_summaries() -> None:
    section_id = uuid4()
    document_id = uuid4()
    passage_id = uuid4()
    rows = [
        {
            "passage_id": uuid4(),
            "section_id": section_id,
            "chunk_ordinal": 1,
            "body_text": "A",
            "page_start": 1,
            "page_end": 1,
            "document_id": document_id,
            "document_title": "Paper",
            "heading": "Methods",
            "section_path": ["Methods"],
            "section_page_start": 1,
            "section_page_end": 2,
        },
        {
            "passage_id": passage_id,
            "section_id": section_id,
            "chunk_ordinal": 2,
            "body_text": "B",
            "page_start": 2,
            "page_end": 2,
            "document_id": document_id,
            "document_title": "Paper",
            "heading": "Methods",
            "section_path": ["Methods"],
            "section_page_start": 1,
            "section_page_end": 2,
        },
        {
            "passage_id": uuid4(),
            "section_id": section_id,
            "chunk_ordinal": 3,
            "body_text": "C",
            "page_start": 3,
            "page_end": 3,
            "document_id": document_id,
            "document_title": "Paper",
            "heading": "Methods",
            "section_path": ["Methods"],
            "section_page_start": 1,
            "section_page_end": 2,
        },
        {
            "passage_id": uuid4(),
            "section_id": section_id,
            "chunk_ordinal": 4,
            "body_text": "D",
            "page_start": 4,
            "page_end": 4,
            "document_id": document_id,
            "document_title": "Paper",
            "heading": "Methods",
            "section_path": ["Methods"],
            "section_page_start": 1,
            "section_page_end": 2,
        },
    ]
    service = _service()

    assert (
        service._load_parent_sections(
            MagicMock(),
            passages=(),
            tables=(),
        )
        == ()
    )

    counts_result = _mock_result(
        rows=[
            {
                "section_id": section_id,
                "section_row_count": len(rows),
            }
        ]
    )
    selected_rows_result = _mock_result(
        rows=[
            {
                "passage_id": passage_id,
                "section_id": section_id,
                "row_number": 2,
                "section_row_count": len(rows),
            }
        ]
    )
    window_rows_result = _mock_result(rows=rows[0:3])
    connection = _connection(counts_result, selected_rows_result, window_rows_result)

    parent_sections = service._load_parent_sections(
        connection,
        passages=(
            PassageResult(
                passage_id=passage_id,
                document_id=document_id,
                section_id=section_id,
                document_title="Paper",
                section_path=("Methods",),
                text="Selected",
                score=1.0,
                retrieval_modes=("sparse",),
                page_start=2,
                page_end=2,
                index_version="mvp-v1",
                retrieval_index_run_id=uuid4(),
            ),
        ),
        tables=(),
    )

    assert len(parent_sections) == 1
    assert parent_sections[0].warnings == ("parent_context_truncated",)
    assert [passage.relationship for passage in parent_sections[0].supporting_passages] == [
        "sibling",
        "selected",
        "sibling",
    ]
    assert connection.execute.call_count == 3

    summaries = service._load_document_summaries(
        _connection(
            _mock_result(
                rows=[
                    {
                        "id": document_id,
                        "title": "Paper",
                        "authors": ["Ada", "Grace"],
                        "publication_year": 2024,
                        "quant_tags": {"asset_universe": "equities"},
                        "current_status": "ready",
                    }
                ]
            )
        ),
        document_ids=(document_id,),
        active_index_version="mvp-v1",
    )
    assert summaries == (
        DocumentSummary(
            document_id=document_id,
            title="Paper",
            authors=("Ada", "Grace"),
            publication_year=2024,
            quant_tags={"asset_universe": "equities"},
            current_status="ready",
            active_index_version="mvp-v1",
        ),
    )
    assert (
        service._load_document_summaries(
            MagicMock(),
            document_ids=(),
            active_index_version=None,
        )
        == ()
    )


def test_single_index_version_and_table_preview_helpers() -> None:
    passage = PassageResult(
        passage_id=uuid4(),
        document_id=uuid4(),
        section_id=uuid4(),
        document_title="Paper",
        section_path=("Methods",),
        text="Text",
        score=1.0,
        retrieval_modes=("sparse",),
        page_start=1,
        page_end=1,
        index_version="mvp-v1",
        retrieval_index_run_id=uuid4(),
    )
    table = TableResult(
        table_id=uuid4(),
        document_id=uuid4(),
        section_id=uuid4(),
        document_title="Paper",
        section_path=("Results",),
        caption="Caption",
        table_type="lexical",
        preview=TablePreview(headers=("A",), rows=(("1",),), row_count=1),
        score=1.0,
        retrieval_modes=("dense",),
        page_start=1,
        page_end=1,
        index_version="mvp-v1",
        retrieval_index_run_id=uuid4(),
    )

    assert _build_table_preview(headers=("A", "B"), rows=(("1", "2"), ("3", "4"))) == TablePreview(
        headers=("A", "B"),
        rows=(("1", "2"), ("3", "4")),
        row_count=2,
    )
    assert _dedupe_modes_from_results(passages=(passage,), tables=(table,)) == (
        "sparse",
        "dense",
    )
    assert _service()._ensure_single_index_version([]) is None
    assert _service()._ensure_single_index_version([passage, table]) == "mvp-v1"
    with pytest.raises(MixedIndexVersionError, match="mixed index versions"):
        _service()._ensure_single_index_version(
            [
                passage,
                TableResult(
                    table_id=uuid4(),
                    document_id=uuid4(),
                    section_id=uuid4(),
                    document_title="Paper",
                    section_path=("Results",),
                    caption="Caption",
                    table_type="lexical",
                    preview=TablePreview(headers=("A",), rows=(("1",),), row_count=1),
                    score=1.0,
                    retrieval_modes=("dense",),
                    page_start=1,
                    page_end=1,
                    index_version="mvp-v2",
                    retrieval_index_run_id=uuid4(),
                ),
            ]
        )


def test_indexer_helper_methods_and_rebuild_control_flow() -> None:
    connection = MagicMock()
    indexer = DocumentRetrievalIndexer(
        index_version="mvp-v1",
        chunking_version="chunk-v1",
        embedding_model="embed-v1",
        reranker_model="rank-v1",
        embedding_client=_StubEmbeddingClient(
            [
                EmbeddingBatch(
                    provider="stub",
                    model="stub",
                    dimensions=1024,
                    embeddings=((0.1,) * 1024,),
                ),
                EmbeddingBatch(
                    provider="stub",
                    model="stub",
                    dimensions=1024,
                    embeddings=((0.1,) * 1024,),
                ),
            ]
        ),
        reranker_client=_StubRerankerClient([[RerankItem(index=0, score=1.0)]]),
    )

    run_id = uuid4()
    document_id = uuid4()
    revision_id = uuid4()
    ingest_job_id = uuid4()
    indexer._upsert_build_run(
        connection,
        run_id=run_id,
        document_id=document_id,
        revision_id=revision_id,
        ingest_job_id=ingest_job_id,
        parser_source="docling",
        created_at=datetime.now(UTC),
    )
    params = connection.execute.call_args_list[0].args[1]
    assert params["parser_source"] == "docling"
    assert params["embedding_provider"] == "stub-embed"

    connection.reset_mock()
    indexer._clear_existing_assets(connection, run_id=run_id)
    assert connection.execute.call_count == 2

    connection.reset_mock()
    indexer._activate_build_run(
        connection,
        run_id=run_id,
        revision_id=revision_id,
        embedding_dimensions=1024,
        activated_at=datetime.now(UTC),
    )
    assert connection.execute.call_count == 2

    passage_row = {
        "passage_id": uuid4(),
        "document_id": uuid4(),
        "revision_id": revision_id,
        "section_id": uuid4(),
        "chunk_ordinal": 1,
        "body_text": "Body",
        "contextualized_text": "Context",
        "page_start": 1,
        "page_end": 2,
        "document_title": "Paper",
        "authors": ["Ada"],
        "abstract": "Abs",
        "publication_year": 2024,
        "section_path": ["Intro"],
    }
    table_row = {
        "table_id": uuid4(),
        "document_id": uuid4(),
        "revision_id": revision_id,
        "section_id": uuid4(),
        "caption": "Caption",
        "table_type": "lexical",
        "headers_json": ["A", 2],
        "rows_json": [["x", 1]],
        "page_start": 3,
        "page_end": 4,
        "document_title": "Paper",
        "publication_year": 2024,
        "section_path": ["Results"],
    }
    passage_index_row = indexer._row_to_passage_index_row(passage_row)
    table_index_row = indexer._row_to_table_index_row(table_row)
    assert passage_index_row.authors == ("Ada",)
    assert table_index_row.headers == ("A", "2")
    assert "Document authors: Ada" in indexer._build_passage_search_text(passage_index_row)
    assert "Section path: Results" in indexer._build_table_search_text(table_index_row)

    indexer._insert_passage_asset_batch(
        connection,
        run_id=run_id,
        rows=[passage_index_row],
        embeddings=((0.1,) * 1024,),
        created_at=datetime.now(UTC),
    )
    passage_params = connection.execute.call_args_list[-1].args[1][0]
    assert passage_params["search_text"].startswith("Context")
    assert passage_params["embedding"] == _vector_literal((0.1,) * 1024)

    indexer._insert_table_asset_batch(
        connection,
        run_id=run_id,
        rows=[table_index_row],
        embeddings=((0.2,) * 1024,),
        created_at=datetime.now(UTC),
    )
    table_params = connection.execute.call_args_list[-1].args[1][0]
    assert "Table headers: A | 2" in table_params["search_text"]
    assert table_params["embedding"] == _vector_literal((0.2,) * 1024)

    passage_batches = iter([[passage_index_row]])
    table_batches = iter([[table_index_row]])
    indexer._iter_passage_row_batches = MagicMock(return_value=passage_batches)  # type: ignore[method-assign]
    indexer._iter_table_row_batches = MagicMock(return_value=table_batches)  # type: ignore[method-assign]
    indexer._upsert_build_run = MagicMock()  # type: ignore[method-assign]
    indexer._clear_existing_assets = MagicMock()  # type: ignore[method-assign]
    indexer._insert_passage_asset_batch = MagicMock()  # type: ignore[method-assign]
    indexer._insert_table_asset_batch = MagicMock()  # type: ignore[method-assign]
    indexer._activate_build_run = MagicMock()  # type: ignore[method-assign]

    rebuilt_run_id = indexer.rebuild(
        connection,
        document_id=uuid4(),
        revision_id=revision_id,
        ingest_job_id=uuid4(),
        parser_source="docling",
    )

    assert isinstance(rebuilt_run_id, UUID)
    indexer._upsert_build_run.assert_called_once()
    indexer._clear_existing_assets.assert_called_once()
    indexer._insert_passage_asset_batch.assert_called_once()
    indexer._insert_table_asset_batch.assert_called_once()
    assert indexer._insert_table_asset_batch.call_args.kwargs["embeddings"] == ((0.1,) * 1024,)
    indexer._activate_build_run.assert_called_once()


def test_indexer_rebuild_rejects_unexpected_embedding_dimensions() -> None:
    connection = MagicMock()
    revision_id = uuid4()
    indexer = DocumentRetrievalIndexer(
        index_version="mvp-v1",
        chunking_version="chunk-v1",
        embedding_model="embed-v1",
        reranker_model="rank-v1",
        embedding_client=_StubEmbeddingClient(
            [
                EmbeddingBatch(
                    provider="stub",
                    model="stub",
                    dimensions=64,
                    embeddings=((0.1,) * 64,),
                )
            ]
        ),
        reranker_client=_StubRerankerClient([[RerankItem(index=0, score=1.0)]]),
    )
    indexer._iter_passage_row_batches = MagicMock(  # type: ignore[method-assign]
        return_value=iter([[SimpleNamespace(contextualized_text="Context")]])
    )
    indexer._iter_table_row_batches = MagicMock(return_value=iter([]))  # type: ignore[method-assign]
    indexer._upsert_build_run = MagicMock()  # type: ignore[method-assign]
    indexer._clear_existing_assets = MagicMock()  # type: ignore[method-assign]
    indexer._insert_passage_asset_batch = MagicMock()  # type: ignore[method-assign]
    indexer._insert_table_asset_batch = MagicMock()  # type: ignore[method-assign]
    indexer._activate_build_run = MagicMock()  # type: ignore[method-assign]

    with pytest.raises(RetrievalError, match="embedding dimension mismatch for retrieval assets"):
        indexer.rebuild(
            connection,
            document_id=uuid4(),
            revision_id=revision_id,
            ingest_job_id=uuid4(),
            parser_source="docling",
        )

    indexer._insert_passage_asset_batch.assert_not_called()
    indexer._insert_table_asset_batch.assert_not_called()
    indexer._activate_build_run.assert_not_called()


def test_indexer_search_text_helpers_cover_optional_metadata_and_table_parts() -> None:
    indexer = DocumentRetrievalIndexer(
        index_version="mvp-v1",
        chunking_version="chunk-v1",
        embedding_model="embed-v1",
        reranker_model="rank-v1",
    )

    minimal_passage_text = indexer._build_passage_search_text(
        _PassageIndexRow(
            passage_id=uuid4(),
            document_id=uuid4(),
            revision_id=uuid4(),
            section_id=uuid4(),
            chunk_ordinal=0,
            body_text="Context only",
            contextualized_text="Context only",
            page_start=None,
            page_end=None,
            document_title="Paper",
            authors=(),
            abstract=None,
            publication_year=None,
            section_path=(),
        )
    )
    assert minimal_passage_text == "Context only"

    table_text = indexer._build_table_search_text(
        _TableIndexRow(
            table_id=uuid4(),
            document_id=uuid4(),
            revision_id=uuid4(),
            section_id=uuid4(),
            table_type=None,
            page_start=None,
            page_end=None,
            publication_year=None,
            document_title="Paper",
            section_path=(),
            caption=None,
            headers=(),
            rows=(),
        )
    )
    assert table_text == "Document title: Paper\nSection path: Body"


def test_ranked_result_loaders_cover_guard_clauses_and_missing_active_runs() -> None:
    service = _service()
    active_runs = _ActiveRunSelection(run_ids=(uuid4(),), index_versions=("mvp-v1",))

    assert service._load_ranked_passage_results(
        MagicMock(),
        query="   ",
        limit=5,
        filtered_document_ids=None,
        active_runs=active_runs,
        sparse_candidate_limit=5,
        dense_candidate_limit=5,
    ) == ([], 0, 0)
    assert service._load_ranked_table_results(
        MagicMock(),
        query="query",
        limit=5,
        filtered_document_ids=(),
        active_runs=active_runs,
        sparse_candidate_limit=5,
        dense_candidate_limit=5,
    ) == ([], 0, 0)

    empty_selection = _service(active_index_version=None)._resolve_active_run_selection(
        _connection(_mock_result(rows=[])),
        filtered_document_ids=None,
    )
    assert empty_selection == _ActiveRunSelection(run_ids=(), index_versions=())


def test_page_index_version_uses_active_runs_or_result_fallback() -> None:
    service = _service(active_index_version=None)
    result = PassageResult(
        passage_id=uuid4(),
        document_id=uuid4(),
        section_id=uuid4(),
        document_title="Paper",
        section_path=("Methods",),
        text="Text",
        score=1.0,
        retrieval_modes=("sparse",),
        page_start=1,
        page_end=1,
        index_version="mvp-v1",
        retrieval_index_run_id=uuid4(),
    )

    assert (
        service._page_index_version(
            results=(result,),
            active_runs=_ActiveRunSelection(run_ids=(uuid4(),), index_versions=("mvp-v1",)),
        )
        == "mvp-v1"
    )
    assert (
        service._page_index_version(
            results=(result,),
            active_runs=_ActiveRunSelection(
                run_ids=(uuid4(), uuid4()),
                index_versions=("mvp-v1", "mvp-v2"),
            ),
        )
        == "mvp-v1"
    )
    assert (
        service._page_index_version(
            results=(result,),
            active_runs=_ActiveRunSelection(run_ids=(), index_versions=()),
        )
        == "mvp-v1"
    )


def test_parent_sections_cover_unselected_and_empty_section_paths() -> None:
    service = _service()
    populated_section_id = uuid4()
    empty_section_id = uuid4()
    document_id = uuid4()
    table = TableResult(
        table_id=uuid4(),
        document_id=document_id,
        section_id=populated_section_id,
        document_title="Paper",
        section_path=("Results",),
        caption="Caption",
        table_type="lexical",
        preview=TablePreview(headers=("A",), rows=(("1",),), row_count=1),
        score=1.0,
        retrieval_modes=("sparse",),
        page_start=1,
        page_end=1,
        index_version="mvp-v1",
        retrieval_index_run_id=uuid4(),
    )

    counts_result = _mock_result(
        rows=[
            {
                "section_id": populated_section_id,
                "section_row_count": 3,
            }
        ]
    )
    window_rows_result = _mock_result(
        rows=[
            {
                "passage_id": uuid4(),
                "section_id": populated_section_id,
                "chunk_ordinal": 1,
                "body_text": "A",
                "page_start": 1,
                "page_end": 1,
                "document_id": document_id,
                "document_title": "Paper",
                "heading": "Results",
                "section_path": ["Results"],
                "section_page_start": 1,
                "section_page_end": 3,
            },
            {
                "passage_id": uuid4(),
                "section_id": populated_section_id,
                "chunk_ordinal": 2,
                "body_text": "B",
                "page_start": 2,
                "page_end": 2,
                "document_id": document_id,
                "document_title": "Paper",
                "heading": "Results",
                "section_path": ["Results"],
                "section_page_start": 1,
                "section_page_end": 3,
            },
        ]
    )
    connection = _connection(counts_result, window_rows_result)
    parent_sections = service._load_parent_sections(
        connection,
        passages=(),
        tables=(
            table,
            TableResult(
                table_id=uuid4(),
                document_id=document_id,
                section_id=empty_section_id,
                document_title="Paper",
                section_path=("Appendix",),
                caption="Missing",
                table_type="lexical",
                preview=TablePreview(headers=("A",), rows=(("1",),), row_count=1),
                score=1.0,
                retrieval_modes=("sparse",),
                page_start=4,
                page_end=4,
                index_version="mvp-v1",
                retrieval_index_run_id=uuid4(),
            ),
        ),
    )

    assert len(parent_sections) == 1
    assert [passage.relationship for passage in parent_sections[0].supporting_passages] == [
        "sibling",
        "sibling",
    ]
    assert parent_sections[0].warnings == ("parent_context_truncated",)
    assert connection.execute.call_count == 2
