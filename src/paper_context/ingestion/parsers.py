from __future__ import annotations

import re
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Protocol

import orjson
import pdfplumber
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import SectionHeaderItem, TableItem, TextItem, TitleItem
from docling_core.types.io import DocumentStream

from .types import (
    GateStatus,
    ParsedDocument,
    ParsedParagraph,
    ParsedReference,
    ParsedSection,
    ParsedTable,
    ParserArtifact,
    ParserResult,
)

_SUCCESSFUL_CONVERSION_STATUSES = {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}
_SKIPPED_TEXTUAL_LABELS = {"page_header", "page_footer", "footnote"}
_NUMBERED_HEADING_PATTERN = re.compile(r"^\d+(?:\.\d+){0,3}\s+\S+")
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


class PdfParser(Protocol):
    name: str

    def parse(
        self,
        filename: str,
        content: bytes | None = None,
        *,
        source_path: Path | None = None,
    ) -> ParserResult:
        """Parse a PDF document into a normalized intermediate representation."""
        ...


def _page_bounds(item) -> tuple[int | None, int | None]:
    if not getattr(item, "prov", None):
        return None, None
    pages = [prov.page_no for prov in item.prov]
    return min(pages), max(pages)


def _provenance_offsets(item) -> dict[str, object] | None:
    if not getattr(item, "prov", None):
        return None
    return {
        "charspans": [list(prov.charspan) for prov in item.prov],
        "pages": [prov.page_no for prov in item.prov],
    }


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _estimate_metadata_confidence(
    *,
    title: str | None,
    authors: list[str],
    abstract: str | None,
    publication_year: int | None,
    has_headings: bool,
) -> float:
    score = 0.2
    if title:
        score += 0.3
    if authors:
        score += 0.15
    if abstract:
        score += 0.15
    if publication_year:
        score += 0.1
    if has_headings:
        score += 0.1
    return round(min(score, 0.95), 2)


def _classify_docling_result(parsed_document: ParsedDocument) -> GateStatus:
    paragraph_count = sum(len(section.paragraphs) for section in parsed_document.sections)
    structured_sections = sum(1 for section in parsed_document.sections if section.heading)
    if paragraph_count == 0:
        return "fail"
    if structured_sections == 0:
        return "degraded"
    return "pass"


def _heading_level_from_text(line: str) -> int:
    if _NUMBERED_HEADING_PATTERN.match(line):
        prefix = line.split(maxsplit=1)[0]
        return prefix.count(".") + 1
    return 1


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.endswith("."):
        return False
    if stripped.lower() == "references":
        return True
    if _NUMBERED_HEADING_PATTERN.match(stripped):
        return True
    words = stripped.split()
    if len(words) > 12:
        return False
    if stripped.isupper():
        return True
    titleish = sum(word[:1].isupper() for word in words if word)
    return titleish >= max(1, len(words) - 1)


def _infer_publication_year(text: str) -> int | None:
    match = _YEAR_PATTERN.search(text)
    if not match:
        return None
    return int(match.group(0))


def _build_heading_path(stack: list[tuple[int, str, str]]) -> list[str]:
    return [heading for _, _, heading in stack]


def _build_parser_artifact(
    *,
    artifact_type: str,
    parser: str,
    filename: str,
    content: bytes,
) -> ParserArtifact:
    cleanup_root = Path(tempfile.mkdtemp(prefix=f"{parser}-artifact-"))
    artifact_path = cleanup_root / filename
    artifact_path.write_bytes(content)
    return ParserArtifact(
        artifact_type=artifact_type,
        parser=parser,
        filename=filename,
        content_path=artifact_path,
        cleanup_root=cleanup_root,
    )


def _build_file_backed_parser_artifact(
    *,
    artifact_type: str,
    parser: str,
    filename: str,
    writer: Callable[[Path], None],
) -> ParserArtifact:
    cleanup_root = Path(tempfile.mkdtemp(prefix=f"{parser}-artifact-"))
    artifact_path = cleanup_root / filename
    writer(artifact_path)
    return ParserArtifact(
        artifact_type=artifact_type,
        parser=parser,
        filename=filename,
        content_path=artifact_path,
        cleanup_root=cleanup_root,
    )


class DoclingPdfParser:
    name = "docling"

    def __init__(self) -> None:
        options = PdfPipelineOptions()
        options.do_ocr = False
        options.do_table_structure = True
        options.force_backend_text = True
        # Docling deprecated table-image generation in favor of page images.
        # Enabling page images keeps the standard pipeline off the deprecated field path.
        options.generate_page_images = True
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            }
        )

    def parse(
        self,
        filename: str,
        content: bytes | None = None,
        *,
        source_path: Path | None = None,
    ) -> ParserResult:
        try:
            if source_path is not None:
                source: Path | DocumentStream = source_path
            elif content is not None:
                source = DocumentStream(name=filename, stream=BytesIO(content))
            else:
                raise ValueError("docling parser requires content bytes or a source path")
            conversion = self._converter.convert(
                source,
                raises_on_error=False,
            )
            artifact = _build_file_backed_parser_artifact(
                artifact_type="docling_parse",
                parser=self.name,
                filename="docling-document.json",
                writer=conversion.document.save_as_json,
            )
        except Exception as exc:
            artifact = _build_parser_artifact(
                artifact_type="docling_parse",
                parser=self.name,
                filename="docling-error.json",
                content=orjson.dumps({"error": str(exc)}),
            )
            return ParserResult(
                gate_status="fail",
                parsed_document=None,
                artifact=artifact,
                failure_code="docling_conversion_failed",
                failure_message=str(exc),
            )
        if conversion.status not in _SUCCESSFUL_CONVERSION_STATUSES:
            return ParserResult(
                gate_status="fail",
                parsed_document=None,
                artifact=artifact,
                failure_code="docling_conversion_failed",
                failure_message="Docling could not parse the PDF into a stable document.",
            )

        title: str | None = None
        abstract: str | None = None
        sections_by_key: OrderedDict[str, ParsedSection] = OrderedDict()
        tables: list[ParsedTable] = []
        references: list[ParsedReference] = []
        section_stack: list[tuple[int, str, str]] = []
        root_key = "section-root"
        sections_by_key[root_key] = ParsedSection(
            key=root_key,
            heading=None,
            heading_path=[],
            level=0,
            page_start=None,
            page_end=None,
        )

        def current_section_key() -> str:
            if section_stack:
                return section_stack[-1][1]
            return root_key

        for item, _ in conversion.document.iterate_items():
            if isinstance(item, TitleItem) and not title:
                title = item.text.strip() or None
                continue

            if isinstance(item, SectionHeaderItem):
                heading = item.text.strip()
                if not heading:
                    continue
                while section_stack and section_stack[-1][0] >= item.level:
                    section_stack.pop()
                key = f"section-{len(sections_by_key)}"
                page_start, page_end = _page_bounds(item)
                heading_path = _build_heading_path(section_stack) + [heading]
                sections_by_key[key] = ParsedSection(
                    key=key,
                    heading=heading,
                    heading_path=heading_path,
                    level=item.level,
                    page_start=page_start,
                    page_end=page_end,
                    parent_key=section_stack[-1][1] if section_stack else None,
                )
                section_stack.append((item.level, key, heading))
                continue

            if isinstance(item, TableItem):
                page_start, page_end = _page_bounds(item)
                dataframe = item.export_to_dataframe(conversion.document)
                headers = [_coerce_text(value) for value in list(dataframe.columns)]
                rows = [
                    [_coerce_text(value) for value in row]
                    for row in dataframe.itertuples(index=False, name=None)
                ]
                caption = None
                for caption_ref in getattr(item, "captions", []):
                    caption_item = caption_ref.resolve(conversion.document)
                    caption_text = getattr(caption_item, "text", "").strip()
                    if caption_text:
                        caption = caption_text
                        break
                tables.append(
                    ParsedTable(
                        section_key=current_section_key(),
                        caption=caption,
                        headers=headers,
                        rows=rows,
                        page_start=page_start,
                        page_end=page_end,
                    )
                )
                continue

            if not isinstance(item, TextItem):
                continue

            text = item.text.strip()
            if not text:
                continue
            label = str(item.label.value)
            if label in _SKIPPED_TEXTUAL_LABELS:
                continue

            section_key = current_section_key()
            page_start, page_end = _page_bounds(item)
            if label == "reference" or (
                sections_by_key[section_key].heading_path
                and sections_by_key[section_key].heading_path[-1].lower() == "references"
            ):
                references.append(
                    ParsedReference(
                        raw_citation=text,
                        publication_year=_infer_publication_year(text),
                    )
                )
                continue

            paragraph = ParsedParagraph(
                text=text,
                page_start=page_start,
                page_end=page_end,
                provenance_offsets=_provenance_offsets(item),
            )
            section = sections_by_key[section_key]
            updated_paragraphs = [*section.paragraphs, paragraph]
            if not abstract and len(updated_paragraphs) <= 2 and len(text.split()) > 40:
                abstract = text
            sections_by_key[section_key] = ParsedSection(
                key=section.key,
                heading=section.heading,
                heading_path=section.heading_path,
                level=section.level,
                page_start=section.page_start if section.page_start is not None else page_start,
                page_end=page_end if page_end is not None else section.page_end,
                parent_key=section.parent_key,
                paragraphs=updated_paragraphs,
            )

        sections = [
            section
            for key, section in sections_by_key.items()
            if key != root_key or section.paragraphs
        ]
        flattened_text = " ".join(
            paragraph.text for section in sections for paragraph in section.paragraphs
        )
        parsed_document = ParsedDocument(
            title=title,
            authors=[],
            abstract=abstract,
            publication_year=_infer_publication_year(flattened_text),
            metadata_confidence=_estimate_metadata_confidence(
                title=title,
                authors=[],
                abstract=abstract,
                publication_year=_infer_publication_year(flattened_text),
                has_headings=any(section.heading for section in sections),
            ),
            sections=sections,
            tables=tables,
            references=references,
        )
        gate_status: GateStatus = _classify_docling_result(parsed_document)
        warnings = ["reduced_structure_confidence"] if gate_status == "degraded" else []
        return ParserResult(
            gate_status=gate_status,
            parsed_document=parsed_document if gate_status != "fail" else None,
            artifact=artifact,
            warnings=warnings,
            failure_code="docling_structure_failed" if gate_status == "fail" else None,
            failure_message=(
                "Docling could not recover stable passages with provenance."
                if gate_status == "fail"
                else None
            ),
        )


class PdfPlumberPdfParser:
    name = "pdfplumber"

    def parse(
        self,
        filename: str,
        content: bytes | None = None,
        *,
        source_path: Path | None = None,
    ) -> ParserResult:
        sections_by_key: OrderedDict[str, ParsedSection] = OrderedDict()
        tables: list[ParsedTable] = []
        references: list[ParsedReference] = []
        section_stack: list[tuple[int, str, str]] = []
        title: str | None = None
        abstract: str | None = None
        root_key = "section-root"
        sections_by_key[root_key] = ParsedSection(
            key=root_key,
            heading=None,
            heading_path=[],
            level=0,
            page_start=None,
            page_end=None,
        )

        def current_section_key() -> str:
            if section_stack:
                return section_stack[-1][1]
            return root_key

        def ensure_section(line: str, page_no: int) -> None:
            level = _heading_level_from_text(line)
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            key = f"section-{len(sections_by_key)}"
            heading_path = _build_heading_path(section_stack) + [line]
            sections_by_key[key] = ParsedSection(
                key=key,
                heading=line,
                heading_path=heading_path,
                level=level,
                page_start=page_no,
                page_end=page_no,
                parent_key=section_stack[-1][1] if section_stack else None,
            )
            section_stack.append((level, key, line))

        try:
            if source_path is not None:
                pdf_source: Path | BytesIO = source_path
            elif content is not None:
                pdf_source = BytesIO(content)
            else:
                raise ValueError("pdfplumber parser requires content bytes or a source path")
            with pdfplumber.open(pdf_source) as pdf:
                for page_no, page in enumerate(pdf.pages, start=1):
                    try:
                        page_text = page.extract_text() or ""
                        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
                        if page_no == 1 and lines:
                            title = lines[0]
                        for line in lines:
                            if _looks_like_heading(line):
                                ensure_section(line, page_no)
                                continue
                            if sections_by_key[current_section_key()].heading_path and (
                                sections_by_key[current_section_key()].heading_path[-1].lower()
                                == "references"
                            ):
                                references.append(
                                    ParsedReference(
                                        raw_citation=line,
                                        publication_year=_infer_publication_year(line),
                                    )
                                )
                                continue
                            section = sections_by_key[current_section_key()]
                            paragraph = ParsedParagraph(
                                text=line,
                                page_start=page_no,
                                page_end=page_no,
                                provenance_offsets={"pages": [page_no]},
                            )
                            sections_by_key[section.key] = ParsedSection(
                                key=section.key,
                                heading=section.heading,
                                heading_path=section.heading_path,
                                level=section.level,
                                page_start=(
                                    section.page_start
                                    if section.page_start is not None
                                    else page_no
                                ),
                                page_end=page_no,
                                parent_key=section.parent_key,
                                paragraphs=[*section.paragraphs, paragraph],
                            )
                            if not abstract and page_no == 1 and len(line.split()) > 40:
                                abstract = line

                        for table_index, table in enumerate(page.extract_tables() or [], start=1):
                            cleaned_rows = [
                                [_coerce_text(value) for value in row] for row in table if row
                            ]
                            if not cleaned_rows:
                                continue
                            headers = cleaned_rows[0]
                            body_rows = cleaned_rows[1:] if len(cleaned_rows) > 1 else []
                            tables.append(
                                ParsedTable(
                                    section_key=current_section_key(),
                                    caption=f"Table {table_index}",
                                    headers=headers,
                                    rows=body_rows,
                                    page_start=page_no,
                                    page_end=page_no,
                                )
                            )
                    finally:
                        close_page = getattr(page, "close", None)
                        if callable(close_page):
                            close_page()
        except Exception as exc:
            error_artifact = _build_parser_artifact(
                artifact_type="pdfplumber_parse",
                parser=self.name,
                filename="pdfplumber-error.json",
                content=orjson.dumps({"error": str(exc)}),
            )
            return ParserResult(
                gate_status="fail",
                parsed_document=None,
                artifact=error_artifact,
                failure_code="pdfplumber_conversion_failed",
                failure_message=str(exc),
            )

        sections = [
            section
            for key, section in sections_by_key.items()
            if key != root_key or section.paragraphs
        ]
        flattened_text = " ".join(
            paragraph.text for section in sections for paragraph in section.paragraphs
        )
        parsed_document = ParsedDocument(
            title=title,
            authors=[],
            abstract=abstract,
            publication_year=_infer_publication_year(flattened_text),
            metadata_confidence=_estimate_metadata_confidence(
                title=title,
                authors=[],
                abstract=abstract,
                publication_year=_infer_publication_year(flattened_text),
                has_headings=any(section.heading for section in sections),
            ),
            sections=sections,
            tables=tables,
            references=references,
        )
        gate_status = "pass"
        if sum(len(section.paragraphs) for section in sections) == 0:
            gate_status = "fail"
        warnings = []
        if not any(section.heading for section in sections):
            warnings.append("reduced_structure_confidence")
        document_artifact = _build_parser_artifact(
            artifact_type="pdfplumber_parse",
            parser=self.name,
            filename="pdfplumber-document.json",
            content=orjson.dumps(parsed_document),
        )
        return ParserResult(
            gate_status=gate_status,
            parsed_document=parsed_document if gate_status == "pass" else None,
            artifact=document_artifact,
            warnings=warnings,
            failure_code="pdfplumber_structure_failed" if gate_status == "fail" else None,
            failure_message=(
                "pdfplumber could not recover stable passages with provenance."
                if gate_status == "fail"
                else None
            ),
        )
