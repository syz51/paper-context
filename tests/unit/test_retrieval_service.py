from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from paper_context.retrieval import RetrievalService
from paper_context.retrieval.service import _Candidate
from paper_context.retrieval.types import (
    ContextPackProvenance,
    ContextPackResult,
    ContextPassage,
    DocumentSummary,
    MixedIndexVersionError,
    ParentSectionResult,
    PassageResult,
    RerankItem,
    SearchPage,
    TablePreview,
    TableResult,
)

pytestmark = pytest.mark.unit


class _FixedReranker:
    provider = "deterministic"

    def __init__(self, order_by_query: dict[str, list[int]] | None = None) -> None:
        self.model = "fixed-reranker"
        self._order_by_query = order_by_query or {}

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankItem]:
        order = self._order_by_query.get(query, list(range(len(documents))))
        items = [
            RerankItem(index=index, score=float(len(order) - position))
            for position, index in enumerate(order)
        ]
        if top_n is not None:
            items = items[:top_n]
        return items


def _service() -> RetrievalService:
    return RetrievalService(
        connection_factory=lambda: nullcontext(MagicMock()),
        active_index_version="mvp-v1",
        reranker_client=_FixedReranker(),
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
        warnings=(),
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
        warnings=(),
        rerank_text=text,
        table_id=entity_id,
        caption="Results table",
        table_type="lexical",
        preview=TablePreview(headers=("A", "B"), rows=(("1", "2"), ("3", "4")), row_count=2),
    )
    candidate.retrieval_modes.update(modes)
    return candidate


def test_search_passages_fuses_sparse_and_dense_candidates() -> None:
    service = _service()
    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_ids=(uuid4(),), index_versions=("mvp-v1",))
    )

    run_id = uuid4()
    passage_a = _passage_candidate(
        entity_id=uuid4(),
        retrieval_index_run_id=run_id,
        index_version="mvp-v1",
        text="Sparse only result",
        modes={"sparse"},
    )
    passage_b = _passage_candidate(
        entity_id=uuid4(),
        retrieval_index_run_id=run_id,
        index_version="mvp-v1",
        text="Shared sparse and dense result",
        modes={"sparse"},
    )
    passage_c = _passage_candidate(
        entity_id=uuid4(),
        retrieval_index_run_id=run_id,
        index_version="mvp-v1",
        text="Dense only result",
        modes={"dense"},
    )

    service._load_sparse_passage_candidates = MagicMock(return_value=[passage_a, passage_b])  # type: ignore[method-assign]
    service._load_dense_passage_candidates = MagicMock(return_value=[passage_b, passage_c])  # type: ignore[method-assign]

    results = service.search_passages(query="fusion")

    assert len(results) == 3
    assert results[0].passage_id == passage_b.passage_id
    assert {result.passage_id for result in results} == {
        passage_a.passage_id,
        passage_b.passage_id,
        passage_c.passage_id,
    }
    assert next(
        result for result in results if result.passage_id == passage_b.passage_id
    ).retrieval_modes == ("sparse", "dense")
    assert next(
        result for result in results if result.passage_id == passage_a.passage_id
    ).retrieval_modes == ("sparse",)
    assert next(
        result for result in results if result.passage_id == passage_c.passage_id
    ).retrieval_modes == ("dense",)
    assert all(result.index_version == "mvp-v1" for result in results)


def test_search_passages_dense_only_hit_survives_rerank() -> None:
    service = _service()
    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_ids=(uuid4(),), index_versions=("mvp-v1",))
    )

    run_id = uuid4()
    dense_candidate = _passage_candidate(
        entity_id=uuid4(),
        retrieval_index_run_id=run_id,
        index_version="mvp-v1",
        text="Dense only result",
        modes={"dense"},
    )

    service._load_sparse_passage_candidates = MagicMock(return_value=[])  # type: ignore[method-assign]
    service._load_dense_passage_candidates = MagicMock(return_value=[dense_candidate])  # type: ignore[method-assign]

    results = service.search_passages(query="dense only")

    assert len(results) == 1
    assert results[0].passage_id == dense_candidate.passage_id
    assert results[0].retrieval_modes == ("dense",)
    assert results[0].retrieval_index_run_id == run_id


def test_search_tables_returns_structured_preview() -> None:
    service = _service()
    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_ids=(uuid4(),), index_versions=("mvp-v1",))
    )

    run_id = uuid4()
    table_candidate = _table_candidate(
        entity_id=uuid4(),
        retrieval_index_run_id=run_id,
        index_version="mvp-v1",
        text="Table query result",
        modes={"sparse"},
    )

    service._load_sparse_table_candidates = MagicMock(return_value=[table_candidate])  # type: ignore[method-assign]

    results = service.search_tables(query="table query")

    assert len(results) == 1
    assert results[0].table_id == table_candidate.table_id
    assert results[0].preview.headers == ("A", "B")
    assert results[0].preview.rows == (("1", "2"), ("3", "4"))
    assert results[0].preview.row_count == 2
    assert results[0].retrieval_modes == ("sparse",)


def test_search_tables_skips_dense_candidate_loader() -> None:
    service = _service()
    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(run_ids=(uuid4(),), index_versions=("mvp-v1",))
    )
    service._load_sparse_table_candidates = MagicMock(return_value=[])  # type: ignore[method-assign]

    assert service.search_tables(query="table query") == []

    service._load_sparse_table_candidates.assert_called_once()


def test_build_context_pack_propagates_warnings_and_provenance() -> None:
    service = _service()
    passage = PassageResult(
        passage_id=uuid4(),
        document_id=uuid4(),
        section_id=uuid4(),
        document_title="Paper",
        section_path=("Methods",),
        text="Selected context",
        score=1.0,
        retrieval_modes=("sparse",),
        page_start=1,
        page_end=1,
        index_version="mvp-v1",
        retrieval_index_run_id=uuid4(),
        warnings=("parser_fallback_used",),
    )
    parent_section = ParentSectionResult(
        section_id=passage.section_id,
        document_id=passage.document_id,
        document_title="Paper",
        heading="Methods",
        section_path=("Methods",),
        page_start=1,
        page_end=2,
        supporting_passages=(
            ContextPassage(
                passage_id=passage.passage_id,
                text="Selected context",
                chunk_ordinal=1,
                page_start=1,
                page_end=1,
                relationship="selected",
            ),
        ),
        warnings=("parent_context_truncated",),
    )
    document = DocumentSummary(
        document_id=passage.document_id,
        title="Paper",
        authors=("Ada Lovelace",),
        publication_year=2024,
        quant_tags={"asset_universe": "equities"},
        current_status="ready",
        active_index_version="mvp-v1",
    )

    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            run_ids=(passage.retrieval_index_run_id,),
            index_versions=("mvp-v1",),
        )
    )
    service._search_passages_page_with_connection = MagicMock(  # type: ignore[method-assign]
        return_value=SearchPage(items=(passage,), next_cursor=None, index_version="mvp-v1")
    )
    service._search_tables_with_connection = MagicMock(return_value=[])  # type: ignore[method-assign]
    service._load_parent_sections = MagicMock(return_value=(parent_section,))  # type: ignore[method-assign]
    service._load_document_summaries = MagicMock(return_value=(document,))  # type: ignore[method-assign]

    pack = service.build_context_pack(query="pack query")

    assert isinstance(pack, ContextPackResult)
    assert pack.query == "pack query"
    assert pack.passages == (passage,)
    assert pack.tables == ()
    assert pack.parent_sections == (parent_section,)
    assert pack.documents == (document,)
    assert pack.provenance == ContextPackProvenance(
        active_index_version="mvp-v1",
        retrieval_index_run_ids=(passage.retrieval_index_run_id,),
        retrieval_modes=("sparse",),
    )
    assert pack.warnings == ("parser_fallback_used", "parent_context_truncated")


def test_build_context_pack_rejects_mixed_index_versions() -> None:
    service = _service()
    passage = PassageResult(
        passage_id=uuid4(),
        document_id=uuid4(),
        section_id=uuid4(),
        document_title="Paper",
        section_path=("Methods",),
        text="Selected context",
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
        caption="Result table",
        table_type="lexical",
        preview=TablePreview(headers=("A",), rows=(("1",),), row_count=1),
        score=1.0,
        retrieval_modes=("sparse",),
        page_start=1,
        page_end=1,
        index_version="mvp-v2",
        retrieval_index_run_id=uuid4(),
    )

    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            run_ids=(passage.retrieval_index_run_id, table.retrieval_index_run_id),
            index_versions=("mvp-v1", "mvp-v2"),
        )
    )
    service._search_passages_page_with_connection = MagicMock(  # type: ignore[method-assign]
        return_value=SearchPage(items=(passage,), next_cursor=None, index_version="mvp-v1")
    )
    service._search_tables_with_connection = MagicMock(return_value=[table])  # type: ignore[method-assign]

    with pytest.raises(MixedIndexVersionError):
        service.build_context_pack(query="mixed")


def test_search_passages_allows_mixed_index_versions_when_searching_active_runs() -> None:
    service = _service()
    service._resolve_filtered_document_ids = MagicMock(return_value=None)  # type: ignore[method-assign]
    service._resolve_active_run_selection = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            run_ids=(uuid4(), uuid4()),
            index_versions=("mvp-v2", "mvp-v1"),
        )
    )

    first = _passage_candidate(
        entity_id=uuid4(),
        retrieval_index_run_id=uuid4(),
        index_version="mvp-v2",
        text="Newer active result",
        modes={"sparse"},
    )
    second = _passage_candidate(
        entity_id=uuid4(),
        retrieval_index_run_id=uuid4(),
        index_version="mvp-v1",
        text="Older active result",
        modes={"sparse"},
    )
    service._load_sparse_passage_candidates = MagicMock(return_value=[first, second])  # type: ignore[method-assign]
    service._load_dense_passage_candidates = MagicMock(return_value=[])  # type: ignore[method-assign]

    results = service.search_passages(query="rollout")

    assert [result.index_version for result in results] == ["mvp-v2", "mvp-v1"]
