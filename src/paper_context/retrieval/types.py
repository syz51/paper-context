from __future__ import annotations

import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar

EmbeddingInputType = Literal["query", "document"]
RetrievalMode = Literal["sparse", "dense"]
ContextRelationship = Literal["selected", "sibling"]


class RetrievalError(RuntimeError):
    """Base exception for retrieval failures."""


class MixedIndexVersionError(RetrievalError):
    """Raised when a result set spans multiple index versions."""


@dataclass(frozen=True)
class EmbeddingBatch:
    provider: str
    model: str
    dimensions: int
    embeddings: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RerankItem:
    index: int
    score: float


class EmbeddingClient(Protocol):
    provider: str
    model: str

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        *,
        input_type: EmbeddingInputType,
    ) -> EmbeddingBatch:  # pragma: no cover - protocol stub
        raise NotImplementedError


class RerankerClient(Protocol):
    provider: str
    model: str

    @abstractmethod
    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankItem]:  # pragma: no cover - protocol stub
        raise NotImplementedError


@dataclass(frozen=True)
class RetrievalFilters:
    document_ids: tuple[uuid.UUID, ...] = ()
    publication_years: tuple[int, ...] = ()


@dataclass(frozen=True)
class TablePreview:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    row_count: int


@dataclass(frozen=True)
class PassageResult:
    passage_id: uuid.UUID
    document_id: uuid.UUID
    section_id: uuid.UUID
    document_title: str
    section_path: tuple[str, ...]
    text: str
    score: float
    retrieval_modes: tuple[RetrievalMode, ...]
    page_start: int | None
    page_end: int | None
    index_version: str
    retrieval_index_run_id: uuid.UUID
    parser_source: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TableResult:
    table_id: uuid.UUID
    document_id: uuid.UUID
    section_id: uuid.UUID
    document_title: str
    section_path: tuple[str, ...]
    caption: str | None
    table_type: str | None
    preview: TablePreview
    score: float
    retrieval_modes: tuple[RetrievalMode, ...]
    page_start: int | None
    page_end: int | None
    index_version: str
    retrieval_index_run_id: uuid.UUID
    parser_source: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TableDetailResult:
    table_id: uuid.UUID
    document_id: uuid.UUID
    section_id: uuid.UUID
    document_title: str
    section_path: tuple[str, ...]
    caption: str | None
    table_type: str | None
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    row_count: int
    page_start: int | None
    page_end: int | None
    index_version: str | None
    retrieval_index_run_id: uuid.UUID | None
    parser_source: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PassageContextTarget:
    passage_id: uuid.UUID
    document_id: uuid.UUID
    section_id: uuid.UUID
    document_title: str
    section_path: tuple[str, ...]
    text: str
    chunk_ordinal: int
    page_start: int | None
    page_end: int | None
    index_version: str | None
    retrieval_index_run_id: uuid.UUID | None
    parser_source: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextPassage:
    passage_id: uuid.UUID
    text: str
    chunk_ordinal: int
    page_start: int | None
    page_end: int | None
    relationship: ContextRelationship


@dataclass(frozen=True)
class ParentSectionResult:
    section_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    heading: str | None
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    supporting_passages: tuple[ContextPassage, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentSummary:
    document_id: uuid.UUID
    title: str
    authors: tuple[str, ...]
    publication_year: int | None
    quant_tags: dict[str, object] = field(default_factory=dict)
    current_status: str = "ready"
    active_index_version: str | None = None


@dataclass(frozen=True)
class ContextPackProvenance:
    active_index_version: str | None
    retrieval_index_run_ids: tuple[uuid.UUID, ...]
    retrieval_modes: tuple[RetrievalMode, ...]


@dataclass(frozen=True)
class ContextPackResult:
    context_pack_id: uuid.UUID
    query: str
    passages: tuple[PassageResult, ...]
    tables: tuple[TableResult, ...]
    parent_sections: tuple[ParentSectionResult, ...]
    documents: tuple[DocumentSummary, ...]
    provenance: ContextPackProvenance
    warnings: tuple[str, ...] = ()
    next_cursor: str | None = None


@dataclass(frozen=True)
class PassageContextResult:
    passage: PassageContextTarget
    context_passages: tuple[ContextPassage, ...]
    warnings: tuple[str, ...] = ()


TResult = TypeVar("TResult")


@dataclass(frozen=True)
class SearchPage[TResult]:
    items: tuple[TResult, ...]
    next_cursor: str | None = None
    index_version: str | None = None
