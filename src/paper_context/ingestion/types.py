from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

GateStatus = Literal["pass", "degraded", "fail"]


@dataclass(frozen=True)
class ParsedParagraph:
    text: str
    page_start: int | None
    page_end: int | None
    provenance_offsets: dict[str, Any] | None = None


@dataclass(frozen=True)
class ParsedSection:
    key: str
    heading: str | None
    heading_path: list[str]
    level: int
    page_start: int | None
    page_end: int | None
    parent_key: str | None = None
    paragraphs: list[ParsedParagraph] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedTable:
    section_key: str
    caption: str | None
    headers: list[str]
    rows: list[list[str]]
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class ParsedReference:
    raw_citation: str
    normalized_title: str | None = None
    authors: list[str] | None = None
    publication_year: int | None = None
    doi: str | None = None
    source_confidence: float | None = None


@dataclass(frozen=True)
class ParsedDocument:
    title: str | None
    authors: list[str]
    abstract: str | None
    publication_year: int | None
    metadata_confidence: float | None
    sections: list[ParsedSection]
    tables: list[ParsedTable]
    references: list[ParsedReference]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParserArtifact:
    artifact_type: str
    parser: str
    filename: str
    content: bytes


@dataclass(frozen=True)
class ParserResult:
    gate_status: GateStatus
    parsed_document: ParsedDocument | None
    artifact: ParserArtifact
    warnings: list[str] = field(default_factory=list)
    failure_code: str | None = None
    failure_message: str | None = None


@dataclass(frozen=True)
class EnrichmentResult:
    title: str | None = None
    authors: list[str] | None = None
    abstract: str | None = None
    publication_year: int | None = None
    metadata_confidence: float | None = None
    warnings: list[str] = field(default_factory=list)
