"""Shared ingestion service contracts used by the worker."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, TypedDict, cast

from sqlalchemy import insert, text
from sqlalchemy.engine import Connection

from paper_context.models import RetrievalIndexRun
from paper_context.queue.contracts import IngestionQueue, IngestQueuePayload
from paper_context.queue.pgmq import PgmqMessage
from paper_context.storage.base import StorageInterface

from .enrichment import MetadataEnricher
from .parsers import PdfParser
from .types import EnrichmentResult, ParsedDocument, ParsedParagraph, ParsedSection, ParserResult


class IngestJobRow(TypedDict):
    status: str
    warnings: list[str]


class SourceArtifactRow(TypedDict):
    storage_ref: str


@dataclass(frozen=True)
class IngestJobContext:
    message: PgmqMessage
    payload: IngestQueuePayload


class LeaseExtender:
    def __init__(
        self,
        connection: Connection,
        queue_adapter: IngestionQueue,
        message: PgmqMessage,
        default_vt_seconds: int,
    ) -> None:
        self._connection = connection
        self._queue_adapter = queue_adapter
        self._message = message
        self._default_vt_seconds = default_vt_seconds

    def extend(self, vt_seconds: int | None = None) -> None:
        vt_seconds = vt_seconds or self._default_vt_seconds
        self._queue_adapter.extend_lease(self._connection, self._message.msg_id, vt_seconds)


class IngestProcessor(Protocol):
    def process(
        self,
        connection: Connection,
        context: IngestJobContext,
        lease: LeaseExtender,
    ) -> None:
        """Process an ingest job context while the lease remains granted."""
        pass


class DeterministicIngestProcessor:
    TERMINAL_STATUSES = {"ready", "failed"}

    def __init__(
        self,
        *,
        storage: StorageInterface,
        primary_parser: PdfParser,
        fallback_parser: PdfParser,
        metadata_enricher: MetadataEnricher,
        index_version: str,
        chunking_version: str,
        embedding_model: str,
        reranker_model: str,
        min_tokens: int,
        max_tokens: int,
        overlap_fraction: float,
    ) -> None:
        self._storage = storage
        self._primary_parser = primary_parser
        self._fallback_parser = fallback_parser
        self._metadata_enricher = metadata_enricher
        self._index_version = index_version
        self._chunking_version = chunking_version
        self._embedding_model = embedding_model
        self._reranker_model = reranker_model
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._overlap_fraction = overlap_fraction

    def process(
        self,
        connection: Connection,
        context: IngestJobContext,
        lease: LeaseExtender,
    ) -> None:
        job = self._lock_ingest_job(connection, context.payload.ingest_job_id)
        if job is None:
            raise LookupError(f"missing ingest job {context.payload.ingest_job_id}")
        if job["status"] in self.TERMINAL_STATUSES:
            return

        source_artifact = self._load_source_artifact(connection, context.payload.document_id)
        if source_artifact is None:
            self._mark_failed(
                connection,
                ingest_job_id=context.payload.ingest_job_id,
                document_id=context.payload.document_id,
                failure_code="missing_source_artifact",
                failure_message="No source PDF artifact exists for this ingest job.",
                warnings=[],
            )
            return

        self._reset_document_state(connection, context.payload.document_id)
        storage_ref = source_artifact["storage_ref"]
        pdf_bytes = self._storage.resolve(storage_ref).read_bytes()
        warnings = list(job["warnings"])

        self._mark_stage(
            connection,
            ingest_job_id=context.payload.ingest_job_id,
            document_id=context.payload.document_id,
            status="parsing",
            warnings=warnings,
        )

        primary_result = self._primary_parser.parse(
            filename=storage_ref,
            content=pdf_bytes,
        )
        primary_artifact_id = self._persist_parser_artifact(
            connection,
            document_id=context.payload.document_id,
            ingest_job_id=context.payload.ingest_job_id,
            artifact=primary_result,
            is_primary=primary_result.gate_status == "pass",
        )
        lease.extend()

        active_result = primary_result
        active_artifact_id = primary_artifact_id
        parser_source = primary_result.artifact.parser
        warnings = _dedupe_warnings([*warnings, *primary_result.warnings])
        if primary_result.gate_status == "degraded":
            warnings = _dedupe_warnings([*warnings, "parser_fallback_used"])
            fallback_result = self._fallback_parser.parse(
                filename=storage_ref,
                content=pdf_bytes,
            )
            fallback_artifact_id = self._persist_parser_artifact(
                connection,
                document_id=context.payload.document_id,
                ingest_job_id=context.payload.ingest_job_id,
                artifact=fallback_result,
                is_primary=fallback_result.gate_status == "pass",
            )
            lease.extend()
            warnings = _dedupe_warnings([*warnings, *fallback_result.warnings])
            if fallback_result.gate_status != "pass" or fallback_result.parsed_document is None:
                self._mark_failed(
                    connection,
                    ingest_job_id=context.payload.ingest_job_id,
                    document_id=context.payload.document_id,
                    failure_code=fallback_result.failure_code or "pdfplumber_structure_failed",
                    failure_message=(
                        fallback_result.failure_message
                        or "Fallback parsing could not recover stable passages with provenance."
                    ),
                    warnings=warnings,
                )
                return
            active_result = fallback_result
            active_artifact_id = fallback_artifact_id
            parser_source = fallback_result.artifact.parser
        elif primary_result.gate_status == "fail" or primary_result.parsed_document is None:
            self._mark_failed(
                connection,
                ingest_job_id=context.payload.ingest_job_id,
                document_id=context.payload.document_id,
                failure_code=primary_result.failure_code or "docling_structure_failed",
                failure_message=(
                    primary_result.failure_message
                    or "Docling could not recover stable passages with provenance."
                ),
                warnings=warnings,
            )
            return

        parsed_document = active_result.parsed_document
        if parsed_document is None:
            raise RuntimeError("parsed_document is unexpectedly missing after parser validation")

        if (parsed_document.metadata_confidence or 0.0) < 0.5:
            warnings = _dedupe_warnings([*warnings, "metadata_low_confidence"])

        self._mark_stage(
            connection,
            ingest_job_id=context.payload.ingest_job_id,
            document_id=context.payload.document_id,
            status="normalizing",
            warnings=warnings,
        )
        section_ids = self._normalize_document(
            connection,
            document_id=context.payload.document_id,
            parsed_document=parsed_document,
            artifact_id=active_artifact_id,
        )
        self._apply_document_metadata(
            connection,
            document_id=context.payload.document_id,
            title=parsed_document.title,
            authors=parsed_document.authors,
            abstract=parsed_document.abstract,
            publication_year=parsed_document.publication_year,
            metadata_confidence=parsed_document.metadata_confidence,
        )
        lease.extend()

        self._mark_stage(
            connection,
            ingest_job_id=context.payload.ingest_job_id,
            document_id=context.payload.document_id,
            status="enriching_metadata",
            warnings=warnings,
        )
        enrichment = self._metadata_enricher.enrich(parsed_document)
        warnings = _dedupe_warnings([*warnings, *enrichment.warnings])
        self._apply_enriched_document_metadata(
            connection,
            document_id=context.payload.document_id,
            parsed_document=parsed_document,
            enrichment=enrichment,
            warnings=warnings,
        )
        lease.extend()

        self._mark_stage(
            connection,
            ingest_job_id=context.payload.ingest_job_id,
            document_id=context.payload.document_id,
            status="chunking",
            warnings=warnings,
        )
        self._insert_passages(
            connection,
            document_id=context.payload.document_id,
            parsed_document=parsed_document,
            section_ids=section_ids,
            artifact_id=active_artifact_id,
        )
        lease.extend()

        self._mark_stage(
            connection,
            ingest_job_id=context.payload.ingest_job_id,
            document_id=context.payload.document_id,
            status="indexing",
            warnings=warnings,
        )
        self._replace_index_run(
            connection,
            document_id=context.payload.document_id,
            ingest_job_id=context.payload.ingest_job_id,
            parser_source=parser_source,
        )
        self._mark_ready(
            connection,
            ingest_job_id=context.payload.ingest_job_id,
            document_id=context.payload.document_id,
            warnings=warnings,
        )

    def _lock_ingest_job(
        self,
        connection: Connection,
        ingest_job_id: uuid.UUID,
    ) -> IngestJobRow | None:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, document_id, status, COALESCE(warnings, '[]'::jsonb) AS warnings
                    FROM ingest_jobs
                    WHERE id = :ingest_job_id
                    FOR UPDATE
                    """
                ),
                {"ingest_job_id": ingest_job_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return cast(IngestJobRow, dict(row))

    def _load_source_artifact(
        self,
        connection: Connection,
        document_id: uuid.UUID,
    ) -> SourceArtifactRow | None:
        row = (
            connection.execute(
                text(
                    """
                    SELECT id, storage_ref, checksum
                    FROM document_artifacts
                    WHERE document_id = :document_id
                      AND artifact_type = 'source_pdf'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ),
                {"document_id": document_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return cast(SourceArtifactRow, dict(row))

    def _reset_document_state(self, connection: Connection, document_id: uuid.UUID) -> None:
        for statement in [
            "DELETE FROM document_passages WHERE document_id = :document_id",
            "DELETE FROM document_tables WHERE document_id = :document_id",
            "DELETE FROM document_references WHERE document_id = :document_id",
            "DELETE FROM document_sections WHERE document_id = :document_id",
            "DELETE FROM retrieval_index_runs WHERE document_id = :document_id",
            """
            DELETE FROM document_artifacts
            WHERE document_id = :document_id
              AND artifact_type <> 'source_pdf'
            """,
            """
            UPDATE document_artifacts
            SET is_primary = false
            WHERE document_id = :document_id
            """,
        ]:
            connection.execute(text(statement), {"document_id": document_id})

    def _persist_parser_artifact(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        ingest_job_id: uuid.UUID,
        artifact: ParserResult,
        is_primary: bool,
    ) -> uuid.UUID:
        stored_artifact = self._storage.store_bytes(
            f"documents/{document_id}/{ingest_job_id}/{artifact.artifact.filename}",
            artifact.artifact.content,
        )
        artifact_id = uuid.uuid4()
        connection.execute(
            text(
                """
                INSERT INTO document_artifacts (
                    id,
                    document_id,
                    artifact_type,
                    parser,
                    storage_ref,
                    checksum,
                    is_primary
                )
                VALUES (
                    :id,
                    :document_id,
                    :artifact_type,
                    :parser,
                    :storage_ref,
                    :checksum,
                    :is_primary
                )
                """
            ),
            {
                "id": artifact_id,
                "document_id": document_id,
                "artifact_type": artifact.artifact.artifact_type,
                "parser": artifact.artifact.parser,
                "storage_ref": stored_artifact.storage_ref,
                "checksum": stored_artifact.checksum,
                "is_primary": is_primary,
            },
        )
        return artifact_id

    def _normalize_document(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        parsed_document: ParsedDocument,
        artifact_id: uuid.UUID,
    ) -> dict[str, uuid.UUID]:
        section_ids: dict[str, uuid.UUID] = {}
        sections = parsed_document.sections or [
            ParsedSection(
                key="section-root",
                heading=None,
                heading_path=[],
                level=0,
                page_start=None,
                page_end=None,
            )
        ]
        for section in sections:
            section_id = uuid.uuid4()
            section_ids[section.key] = section_id
            parent_section_id = (
                section_ids[section.parent_key] if section.parent_key is not None else None
            )
            connection.execute(
                text(
                    """
                    INSERT INTO document_sections (
                        id,
                        document_id,
                        parent_section_id,
                        heading,
                        heading_path,
                        ordinal,
                        page_start,
                        page_end,
                        artifact_id
                    )
                    VALUES (
                        :id,
                        :document_id,
                        :parent_section_id,
                        :heading,
                        CAST(:heading_path AS jsonb),
                        :ordinal,
                        :page_start,
                        :page_end,
                        :artifact_id
                    )
                    """
                ),
                {
                    "id": section_id,
                    "document_id": document_id,
                    "parent_section_id": parent_section_id,
                    "heading": section.heading,
                    "heading_path": _json_array(section.heading_path),
                    "ordinal": len(section_ids),
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "artifact_id": artifact_id,
                },
            )

        for table in parsed_document.tables:
            connection.execute(
                text(
                    """
                    INSERT INTO document_tables (
                        id,
                        document_id,
                        section_id,
                        caption,
                        table_type,
                        headers_json,
                        rows_json,
                        page_start,
                        page_end,
                        artifact_id
                    )
                    VALUES (
                        :id,
                        :document_id,
                        :section_id,
                        :caption,
                        'lexical',
                        CAST(:headers_json AS jsonb),
                        CAST(:rows_json AS jsonb),
                        :page_start,
                        :page_end,
                        :artifact_id
                    )
                    """
                ),
                {
                    "id": uuid.uuid4(),
                    "document_id": document_id,
                    "section_id": section_ids[table.section_key],
                    "caption": table.caption,
                    "headers_json": _json_array(table.headers),
                    "rows_json": _json_array(table.rows),
                    "page_start": table.page_start,
                    "page_end": table.page_end,
                    "artifact_id": artifact_id,
                },
            )

        for reference in parsed_document.references:
            connection.execute(
                text(
                    """
                    INSERT INTO document_references (
                        id,
                        document_id,
                        raw_citation,
                        normalized_title,
                        authors,
                        publication_year,
                        doi,
                        source_confidence,
                        artifact_id
                    )
                    VALUES (
                        :id,
                        :document_id,
                        :raw_citation,
                        :normalized_title,
                        CAST(:authors AS jsonb),
                        :publication_year,
                        :doi,
                        :source_confidence,
                        :artifact_id
                    )
                    """
                ),
                {
                    "id": uuid.uuid4(),
                    "document_id": document_id,
                    "raw_citation": reference.raw_citation,
                    "normalized_title": reference.normalized_title,
                    "authors": _json_array(reference.authors or []),
                    "publication_year": reference.publication_year,
                    "doi": reference.doi,
                    "source_confidence": reference.source_confidence,
                    "artifact_id": artifact_id,
                },
            )

        return section_ids

    def _apply_document_metadata(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        title: str | None,
        authors: list[str],
        abstract: str | None,
        publication_year: int | None,
        metadata_confidence: float | None,
    ) -> None:
        connection.execute(
            text(
                """
                UPDATE documents
                SET title = COALESCE(:title, title),
                    authors = CAST(:authors AS jsonb),
                    abstract = COALESCE(:abstract, abstract),
                    publication_year = COALESCE(:publication_year, publication_year),
                    metadata_confidence = :metadata_confidence,
                    updated_at = :now
                WHERE id = :document_id
                """
            ),
            {
                "document_id": document_id,
                "title": title,
                "authors": _json_array(authors),
                "abstract": abstract,
                "publication_year": publication_year,
                "metadata_confidence": metadata_confidence,
                "now": datetime.now(UTC),
            },
        )

    def _apply_enriched_document_metadata(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        parsed_document: ParsedDocument,
        enrichment: EnrichmentResult,
        warnings: list[str],
    ) -> None:
        base_confidence = parsed_document.metadata_confidence or 0.0
        enriched_confidence = enrichment.metadata_confidence

        title = parsed_document.title
        authors = parsed_document.authors
        abstract = parsed_document.abstract
        publication_year = parsed_document.publication_year
        confidence = parsed_document.metadata_confidence
        if enriched_confidence is None or enriched_confidence >= base_confidence:
            title = enrichment.title or title
            authors = enrichment.authors or authors
            abstract = enrichment.abstract or abstract
            publication_year = enrichment.publication_year or publication_year
            confidence = enriched_confidence or confidence
        elif any(
            value is not None
            for value in [
                enrichment.title,
                enrichment.authors,
                enrichment.abstract,
                enrichment.publication_year,
            ]
        ):
            warnings[:] = _dedupe_warnings([*warnings, "metadata_conflict_ignored"])

        self._apply_document_metadata(
            connection,
            document_id=document_id,
            title=title,
            authors=authors,
            abstract=abstract,
            publication_year=publication_year,
            metadata_confidence=confidence,
        )

    def _insert_passages(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        parsed_document: ParsedDocument,
        section_ids: dict[str, uuid.UUID],
        artifact_id: uuid.UUID,
    ) -> None:
        title = parsed_document.title or "Untitled document"
        for section in parsed_document.sections:
            for chunk_ordinal, chunk in enumerate(
                _chunk_paragraphs(
                    section.paragraphs,
                    min_tokens=self._min_tokens,
                    max_tokens=self._max_tokens,
                    overlap_fraction=self._overlap_fraction,
                ),
                start=1,
            ):
                body_text = "\n\n".join(paragraph.text for paragraph in chunk)
                token_count = _token_count(body_text)
                contextualized_text = _contextualize_text(title, section.heading_path, body_text)
                connection.execute(
                    text(
                        """
                        INSERT INTO document_passages (
                            id,
                            document_id,
                            section_id,
                            chunk_ordinal,
                            body_text,
                            contextualized_text,
                            token_count,
                            page_start,
                            page_end,
                            provenance_offsets,
                            artifact_id
                        )
                        VALUES (
                            :id,
                            :document_id,
                            :section_id,
                            :chunk_ordinal,
                            :body_text,
                            :contextualized_text,
                            :token_count,
                            :page_start,
                            :page_end,
                            CAST(:provenance_offsets AS jsonb),
                            :artifact_id
                        )
                        """
                    ),
                    {
                        "id": uuid.uuid4(),
                        "document_id": document_id,
                        "section_id": section_ids[section.key],
                        "chunk_ordinal": chunk_ordinal,
                        "body_text": body_text,
                        "contextualized_text": contextualized_text,
                        "token_count": token_count,
                        "page_start": min(
                            (
                                paragraph.page_start
                                for paragraph in chunk
                                if paragraph.page_start is not None
                            ),
                            default=None,
                        ),
                        "page_end": max(
                            (
                                paragraph.page_end
                                for paragraph in chunk
                                if paragraph.page_end is not None
                            ),
                            default=None,
                        ),
                        "provenance_offsets": _json_array(
                            {
                                "pages": [
                                    page
                                    for paragraph in chunk
                                    for page in (paragraph.provenance_offsets or {}).get(
                                        "pages", []
                                    )
                                ],
                                "charspans": [
                                    charspan
                                    for paragraph in chunk
                                    for charspan in (paragraph.provenance_offsets or {}).get(
                                        "charspans", []
                                    )
                                ],
                            }
                        ),
                        "artifact_id": artifact_id,
                    },
                )

    def _replace_index_run(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        ingest_job_id: uuid.UUID,
        parser_source: str,
    ) -> None:
        connection.execute(
            text(
                """
                DELETE FROM retrieval_index_runs
                WHERE document_id = :document_id
                """
            ),
            {"document_id": document_id},
        )
        connection.execute(
            insert(RetrievalIndexRun).values(
                id=uuid.uuid4(),
                document_id=document_id,
                ingest_job_id=ingest_job_id,
                index_version=self._index_version,
                embedding_provider="voyage",
                embedding_model=self._embedding_model,
                reranker_provider="zero_entropy",
                reranker_model=self._reranker_model,
                chunking_version=self._chunking_version,
                parser_source=parser_source,
                status="ready",
            )
        )

    def _mark_stage(
        self,
        connection: Connection,
        *,
        ingest_job_id: uuid.UUID,
        document_id: uuid.UUID,
        status: str,
        warnings: list[str],
    ) -> None:
        now = datetime.now(UTC)
        connection.execute(
            text(
                """
                UPDATE ingest_jobs
                SET status = :status,
                    failure_code = NULL,
                    failure_message = NULL,
                    warnings = CAST(:warnings AS jsonb),
                    started_at = COALESCE(started_at, :now),
                    finished_at = NULL
                WHERE id = :ingest_job_id
                """
            ),
            {
                "ingest_job_id": ingest_job_id,
                "status": status,
                "warnings": _json_array(warnings),
                "now": now,
            },
        )
        connection.execute(
            text(
                """
                UPDATE documents
                SET current_status = :status,
                    updated_at = :now
                WHERE id = :document_id
                """
            ),
            {"document_id": document_id, "status": status, "now": now},
        )

    def _mark_failed(
        self,
        connection: Connection,
        *,
        ingest_job_id: uuid.UUID,
        document_id: uuid.UUID,
        failure_code: str,
        failure_message: str,
        warnings: list[str],
    ) -> None:
        now = datetime.now(UTC)
        connection.execute(
            text(
                """
                UPDATE ingest_jobs
                SET status = 'failed',
                    failure_code = :failure_code,
                    failure_message = :failure_message,
                    warnings = CAST(:warnings AS jsonb),
                    started_at = COALESCE(started_at, :now),
                    finished_at = :now
                WHERE id = :ingest_job_id
                """
            ),
            {
                "ingest_job_id": ingest_job_id,
                "failure_code": failure_code,
                "failure_message": failure_message,
                "warnings": _json_array(_dedupe_warnings(warnings)),
                "now": now,
            },
        )
        connection.execute(
            text(
                """
                UPDATE documents
                SET current_status = 'failed',
                    updated_at = :now
                WHERE id = :document_id
                """
            ),
            {"document_id": document_id, "now": now},
        )

    def _mark_ready(
        self,
        connection: Connection,
        *,
        ingest_job_id: uuid.UUID,
        document_id: uuid.UUID,
        warnings: list[str],
    ) -> None:
        now = datetime.now(UTC)
        connection.execute(
            text(
                """
                UPDATE ingest_jobs
                SET status = 'ready',
                    warnings = CAST(:warnings AS jsonb),
                    finished_at = :now
                WHERE id = :ingest_job_id
                """
            ),
            {
                "ingest_job_id": ingest_job_id,
                "warnings": _json_array(_dedupe_warnings(warnings)),
                "now": now,
            },
        )
        connection.execute(
            text(
                """
                UPDATE documents
                SET current_status = 'ready',
                    updated_at = :now
                WHERE id = :document_id
                """
            ),
            {"document_id": document_id, "now": now},
        )


class SyntheticIngestProcessor:
    """Phase 0 worker processor that simulates terminal completion for smoke testing."""

    TERMINAL_STATUSES = {"ready", "failed"}

    def process(
        self,
        connection: Connection,
        context: IngestJobContext,
        lease: LeaseExtender,
    ) -> None:
        row = (
            connection.execute(
                text(
                    """
                SELECT status
                FROM ingest_jobs
                WHERE id = :ingest_job_id
                FOR UPDATE
                """
                ),
                {"ingest_job_id": context.payload.ingest_job_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise LookupError(f"missing ingest job {context.payload.ingest_job_id}")

        if row["status"] in self.TERMINAL_STATUSES:
            return

        now = datetime.now(UTC)
        connection.execute(
            text(
                """
                UPDATE ingest_jobs
                SET status = 'parsing',
                    started_at = COALESCE(started_at, :now)
                WHERE id = :ingest_job_id
                """
            ),
            {"ingest_job_id": context.payload.ingest_job_id, "now": now},
        )
        lease.extend()
        connection.execute(
            text(
                """
                UPDATE ingest_jobs
                SET status = 'ready',
                    finished_at = :now,
                    warnings = COALESCE(warnings, '[]'::jsonb)
                WHERE id = :ingest_job_id
                """
            ),
            {"ingest_job_id": context.payload.ingest_job_id, "now": now},
        )
        connection.execute(
            text(
                """
                UPDATE documents
                SET current_status = 'ready',
                    updated_at = :now
                WHERE id = :document_id
                """
            ),
            {"document_id": context.payload.document_id, "now": now},
        )


def _json_array(value: object) -> str:
    import json

    return json.dumps(value)


def _dedupe_warnings(warnings: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for warning in warnings:
        if warning not in seen:
            seen.add(warning)
            deduped.append(warning)
    return deduped


def _token_count(text: str) -> int:
    return len(text.split())


def _chunk_paragraphs(
    paragraphs: list[ParsedParagraph],
    *,
    min_tokens: int,
    max_tokens: int,
    overlap_fraction: float,
) -> list[list[ParsedParagraph]]:
    if not paragraphs:
        return []
    chunks: list[list[ParsedParagraph]] = []
    start = 0
    while start < len(paragraphs):
        token_total = 0
        end = start
        while end < len(paragraphs):
            paragraph_tokens = _token_count(paragraphs[end].text)
            if token_total and token_total + paragraph_tokens > max_tokens:
                break
            token_total += paragraph_tokens
            end += 1
            if token_total >= min_tokens:
                next_tokens = _token_count(paragraphs[end].text) if end < len(paragraphs) else 0
                if token_total + next_tokens > max_tokens:
                    break
        if end == start:
            end = start + 1
        chunk = paragraphs[start:end]
        chunks.append(chunk)
        if end >= len(paragraphs):
            break
        overlap_target = max(1, int(token_total * overlap_fraction))
        overlap_tokens = 0
        next_start = end
        while next_start > start and overlap_tokens < overlap_target:
            next_start -= 1
            overlap_tokens += _token_count(paragraphs[next_start].text)
        if next_start == start:
            start = end
        else:
            start = next_start
    return chunks


def _contextualize_text(title: str, heading_path: list[str], body_text: str) -> str:
    section_path = " > ".join(heading_path) if heading_path else "Body"
    local_context = heading_path[-2] if len(heading_path) > 1 else section_path
    return (
        f"Document title: {title}\n"
        f"Section path: {section_path}\n"
        f"Local heading context: {local_context}\n\n"
        f"{body_text}"
    )
