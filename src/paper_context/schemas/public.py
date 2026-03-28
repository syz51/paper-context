from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalFiltersInput(BaseModel):
    document_ids: list[UUID] = Field(default_factory=list)
    publication_years: list[int] = Field(default_factory=list)


class DocumentResult(BaseModel):
    document_id: UUID
    title: str
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    quant_tags: dict[str, object] = Field(default_factory=dict)
    current_status: str
    active_index_version: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentResult] = Field(default_factory=list)
    next_cursor: str | None = None


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    ingest_job_id: UUID
    status: str


class DocumentReplaceResponse(BaseModel):
    document_id: UUID
    ingest_job_id: UUID
    status: str


class IngestJobResponse(BaseModel):
    id: UUID
    document_id: UUID
    status: str
    failure_code: str | None = None
    failure_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    trigger: str | None = None


class DocumentOutlineNode(BaseModel):
    section_id: UUID
    parent_section_id: UUID | None = None
    heading: str | None = None
    section_path: list[str] = Field(default_factory=list)
    ordinal: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    children: list[DocumentOutlineNode] = Field(default_factory=list)


class DocumentOutlineResponse(BaseModel):
    document_id: UUID
    title: str
    sections: list[DocumentOutlineNode] = Field(default_factory=list)


class TablePreviewModel(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    row_count: int


class DocumentTableRecord(BaseModel):
    table_id: UUID
    document_id: UUID
    section_id: UUID
    document_title: str
    section_path: list[str] = Field(default_factory=list)
    caption: str | None = None
    table_type: str | None = None
    preview: TablePreviewModel
    page_start: int | None = None
    page_end: int | None = None


class DocumentTablesResponse(BaseModel):
    document_id: UUID
    title: str
    tables: list[DocumentTableRecord] = Field(default_factory=list)


class TableDetailResponse(BaseModel):
    table_id: UUID
    document_id: UUID
    section_id: UUID
    document_title: str
    section_path: list[str] = Field(default_factory=list)
    caption: str | None = None
    table_type: str | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    row_count: int
    page_start: int | None = None
    page_end: int | None = None
    index_version: str | None = None
    retrieval_index_run_id: UUID | None = None
    parser_source: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PassageResultModel(BaseModel):
    passage_id: UUID
    document_id: UUID
    section_id: UUID
    document_title: str
    section_path: list[str] = Field(default_factory=list)
    text: str
    score: float
    retrieval_modes: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    index_version: str
    retrieval_index_run_id: UUID
    parser_source: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PassageSearchResponse(BaseModel):
    query: str
    passages: list[PassageResultModel] = Field(default_factory=list)
    next_cursor: str | None = None
    exact: bool = True
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class TableResultModel(BaseModel):
    table_id: UUID
    document_id: UUID
    section_id: UUID
    document_title: str
    section_path: list[str] = Field(default_factory=list)
    caption: str | None = None
    table_type: str | None = None
    preview: TablePreviewModel
    score: float
    retrieval_modes: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    index_version: str
    retrieval_index_run_id: UUID
    parser_source: str | None = None
    warnings: list[str] = Field(default_factory=list)


class TableSearchResponse(BaseModel):
    query: str
    tables: list[TableResultModel] = Field(default_factory=list)
    next_cursor: str | None = None
    exact: bool = True
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class ContextPassageModel(BaseModel):
    passage_id: UUID
    text: str
    chunk_ordinal: int
    page_start: int | None = None
    page_end: int | None = None
    relationship: str


class ParentSectionResultModel(BaseModel):
    section_id: UUID
    document_id: UUID
    document_title: str
    heading: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    supporting_passages: list[ContextPassageModel] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ContextPackProvenanceModel(BaseModel):
    active_index_version: str | None = None
    retrieval_index_run_ids: list[UUID] = Field(default_factory=list)
    retrieval_modes: list[str] = Field(default_factory=list)


class ContextPackResponse(BaseModel):
    context_pack_id: UUID
    query: str
    passages: list[PassageResultModel] = Field(default_factory=list)
    tables: list[TableResultModel] = Field(default_factory=list)
    parent_sections: list[ParentSectionResultModel] = Field(default_factory=list)
    documents: list[DocumentResult] = Field(default_factory=list)
    provenance: ContextPackProvenanceModel
    warnings: list[str] = Field(default_factory=list)
    next_cursor: str | None = None


class PassageContextTarget(BaseModel):
    passage_id: UUID
    document_id: UUID
    section_id: UUID
    document_title: str
    section_path: list[str] = Field(default_factory=list)
    text: str
    chunk_ordinal: int
    page_start: int | None = None
    page_end: int | None = None
    index_version: str | None = None
    retrieval_index_run_id: UUID | None = None
    parser_source: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PassageContextResponse(BaseModel):
    passage: PassageContextTarget
    context_passages: list[ContextPassageModel] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


DocumentOutlineNode.model_rebuild()
