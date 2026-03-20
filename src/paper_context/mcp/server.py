from __future__ import annotations

from fastmcp import FastMCP

from paper_context.config import get_settings
from paper_context.db.engine import get_engine
from paper_context.db.session import connection_scope
from paper_context.ingestion.api import DocumentsApiService
from paper_context.queue.contracts import IngestionQueue
from paper_context.retrieval import (
    DeterministicEmbeddingClient,
    HeuristicRerankerClient,
    RetrievalFilters,
    RetrievalService,
    VoyageEmbeddingClient,
    ZeroEntropyRerankerClient,
)
from paper_context.schemas.mcp import (
    ContextPackResponse,
    DocumentListResponse,
    DocumentOutlineResponse,
    PassageContextResponse,
    PassageSearchResponse,
    RetrievalFiltersInput,
    TableDetailResponse,
    TableSearchResponse,
)
from paper_context.schemas.public import (
    ContextPackProvenanceModel,
    ContextPassageModel,
    DocumentResult,
    ParentSectionResultModel,
    PassageContextTarget,
    PassageResultModel,
    TablePreviewModel,
    TableResultModel,
)
from paper_context.storage.local_fs import LocalFilesystemStorage


def _clamp_limit(limit: int, *, maximum: int) -> int:
    return max(1, min(limit, maximum))


def create_server(
    *,
    documents_service: DocumentsApiService | None = None,
    retrieval_service: RetrievalService | None = None,
) -> FastMCP:
    documents = documents_service or _build_documents_service()
    retrieval = retrieval_service or _build_retrieval_service()
    mcp = FastMCP(name="paper-context")

    @mcp.tool
    def search_documents(
        query: str,
        filters: RetrievalFiltersInput | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> DocumentListResponse:
        return documents.search_documents(
            query=query,
            filters=filters,
            cursor=cursor,
            limit=_clamp_limit(limit, maximum=100),
        )

    @mcp.tool
    def search_passages(
        query: str,
        filters: RetrievalFiltersInput | None = None,
        cursor: str | None = None,
        limit: int = 8,
    ) -> PassageSearchResponse:
        page = retrieval.search_passages_page(
            query=query,
            filters=_to_retrieval_filters(filters),
            cursor=cursor,
            limit=_clamp_limit(limit, maximum=8),
        )
        return PassageSearchResponse(
            query=query,
            passages=[_to_passage_model(item) for item in page.items],
            next_cursor=page.next_cursor,
        )

    @mcp.tool
    def search_tables(
        query: str,
        filters: RetrievalFiltersInput | None = None,
        cursor: str | None = None,
        limit: int = 5,
    ) -> TableSearchResponse:
        page = retrieval.search_tables_page(
            query=query,
            filters=_to_retrieval_filters(filters),
            cursor=cursor,
            limit=_clamp_limit(limit, maximum=5),
        )
        return TableSearchResponse(
            query=query,
            tables=[_to_table_model(item) for item in page.items],
            next_cursor=page.next_cursor,
        )

    @mcp.tool
    def get_document_outline(document_id: str) -> DocumentOutlineResponse:
        outline = documents.get_document_outline(document_id=_uuid(document_id))
        if outline is None:
            raise ValueError("document not found")
        return outline

    @mcp.tool
    def get_table(table_id: str) -> TableDetailResponse:
        table = retrieval.get_table(table_id=_uuid(table_id))
        if table is None:
            raise ValueError("table not found")
        return TableDetailResponse(
            table_id=table.table_id,
            document_id=table.document_id,
            section_id=table.section_id,
            document_title=table.document_title,
            section_path=list(table.section_path),
            caption=table.caption,
            table_type=table.table_type,
            headers=list(table.headers),
            rows=[list(row) for row in table.rows],
            row_count=table.row_count,
            page_start=table.page_start,
            page_end=table.page_end,
            index_version=table.index_version,
            retrieval_index_run_id=table.retrieval_index_run_id,
            parser_source=table.parser_source,
            warnings=list(table.warnings),
        )

    @mcp.tool
    def get_passage_context(
        passage_id: str,
        before: int = 1,
        after: int = 1,
    ) -> PassageContextResponse:
        context = retrieval.get_passage_context(
            passage_id=_uuid(passage_id),
            before=before,
            after=after,
        )
        if context is None:
            raise ValueError("passage not found")
        return PassageContextResponse(
            passage=PassageContextTarget(
                passage_id=context.passage.passage_id,
                document_id=context.passage.document_id,
                section_id=context.passage.section_id,
                document_title=context.passage.document_title,
                section_path=list(context.passage.section_path),
                text=context.passage.text,
                chunk_ordinal=context.passage.chunk_ordinal,
                page_start=context.passage.page_start,
                page_end=context.passage.page_end,
                index_version=context.passage.index_version,
                retrieval_index_run_id=context.passage.retrieval_index_run_id,
                parser_source=context.passage.parser_source,
                warnings=list(context.passage.warnings),
            ),
            context_passages=[
                ContextPassageModel(
                    passage_id=passage.passage_id,
                    text=passage.text,
                    chunk_ordinal=passage.chunk_ordinal,
                    page_start=passage.page_start,
                    page_end=passage.page_end,
                    relationship=passage.relationship,
                )
                for passage in context.context_passages
            ],
            warnings=list(context.warnings),
        )

    @mcp.tool
    def build_context_pack(
        query: str,
        filters: RetrievalFiltersInput | None = None,
        cursor: str | None = None,
        limit: int = 8,
    ) -> ContextPackResponse:
        pack = retrieval.build_context_pack(
            query=query,
            filters=_to_retrieval_filters(filters),
            cursor=cursor,
            limit=_clamp_limit(limit, maximum=8),
        )
        return ContextPackResponse(
            context_pack_id=pack.context_pack_id,
            query=pack.query,
            passages=[_to_passage_model(item) for item in pack.passages],
            tables=[_to_table_model(item) for item in pack.tables],
            parent_sections=[
                ParentSectionResultModel(
                    section_id=section.section_id,
                    document_id=section.document_id,
                    document_title=section.document_title,
                    heading=section.heading,
                    section_path=list(section.section_path),
                    page_start=section.page_start,
                    page_end=section.page_end,
                    supporting_passages=[
                        ContextPassageModel(
                            passage_id=passage.passage_id,
                            text=passage.text,
                            chunk_ordinal=passage.chunk_ordinal,
                            page_start=passage.page_start,
                            page_end=passage.page_end,
                            relationship=passage.relationship,
                        )
                        for passage in section.supporting_passages
                    ],
                    warnings=list(section.warnings),
                )
                for section in pack.parent_sections
            ],
            documents=[
                DocumentResult(
                    document_id=document.document_id,
                    title=document.title,
                    authors=list(document.authors),
                    publication_year=document.publication_year,
                    quant_tags=document.quant_tags,
                    current_status=document.current_status,
                    active_index_version=document.active_index_version,
                )
                for document in pack.documents
            ],
            provenance=ContextPackProvenanceModel(
                active_index_version=pack.provenance.active_index_version,
                retrieval_index_run_ids=list(pack.provenance.retrieval_index_run_ids),
                retrieval_modes=list(pack.provenance.retrieval_modes),
            ),
            warnings=list(pack.warnings),
            next_cursor=pack.next_cursor,
        )

    return mcp


def create_http_app(
    *,
    documents_service: DocumentsApiService | None = None,
    retrieval_service: RetrievalService | None = None,
):
    mcp = create_server(
        documents_service=documents_service,
        retrieval_service=retrieval_service,
    )
    return mcp.http_app(path="/", transport="streamable-http")


def _build_documents_service() -> DocumentsApiService:
    settings = get_settings()
    return DocumentsApiService(
        engine=get_engine(),
        queue=IngestionQueue(settings.queue.name),
        storage=LocalFilesystemStorage(settings.storage.root_path),
        max_upload_bytes=settings.upload.max_bytes,
    )


def _build_retrieval_service() -> RetrievalService:
    settings = get_settings()
    voyage_api_key = settings.providers.voyage_api_key
    zero_entropy_api_key = settings.providers.zero_entropy_api_key
    embedding_client = (
        VoyageEmbeddingClient(
            api_key=voyage_api_key,
            model=settings.providers.voyage_model,
        )
        if voyage_api_key
        else DeterministicEmbeddingClient(model=settings.providers.voyage_model)
    )
    reranker_client = (
        ZeroEntropyRerankerClient(
            api_key=zero_entropy_api_key,
            model=settings.providers.reranker_model,
        )
        if zero_entropy_api_key
        else HeuristicRerankerClient(model=settings.providers.reranker_model)
    )
    return RetrievalService(
        connection_factory=lambda: connection_scope(get_engine()),
        active_index_version=None,
        embedding_client=embedding_client,
        reranker_client=reranker_client,
    )


def _to_retrieval_filters(filters: RetrievalFiltersInput | None) -> RetrievalFilters:
    if filters is None:
        return RetrievalFilters()
    return RetrievalFilters(
        document_ids=tuple(filters.document_ids),
        publication_years=tuple(filters.publication_years),
    )


def _to_passage_model(passage) -> PassageResultModel:
    return PassageResultModel(
        passage_id=passage.passage_id,
        document_id=passage.document_id,
        section_id=passage.section_id,
        document_title=passage.document_title,
        section_path=list(passage.section_path),
        text=passage.text,
        score=passage.score,
        retrieval_modes=list(passage.retrieval_modes),
        page_start=passage.page_start,
        page_end=passage.page_end,
        index_version=passage.index_version,
        retrieval_index_run_id=passage.retrieval_index_run_id,
        parser_source=passage.parser_source,
        warnings=list(passage.warnings),
    )


def _to_table_model(table) -> TableResultModel:
    return TableResultModel(
        table_id=table.table_id,
        document_id=table.document_id,
        section_id=table.section_id,
        document_title=table.document_title,
        section_path=list(table.section_path),
        caption=table.caption,
        table_type=table.table_type,
        preview=TablePreviewModel(
            headers=list(table.preview.headers),
            rows=[list(row) for row in table.preview.rows],
            row_count=table.preview.row_count,
        ),
        score=table.score,
        retrieval_modes=list(table.retrieval_modes),
        page_start=table.page_start,
        page_end=table.page_end,
        index_version=table.index_version,
        retrieval_index_run_id=table.retrieval_index_run_id,
        parser_source=table.parser_source,
        warnings=list(table.warnings),
    )


def _uuid(value: str):
    from uuid import UUID

    return UUID(value)
