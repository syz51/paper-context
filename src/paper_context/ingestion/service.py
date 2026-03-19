"""Shared ingestion service contracts used by the worker."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, TypedDict, cast

from sqlalchemy import (
    String,
    and_,
    bindparam,
    delete,
    exists,
    func,
    insert,
    or_,
    select,
    text,
    update,
)
from sqlalchemy import cast as sa_cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection
from sqlalchemy.orm import aliased

from paper_context.models import (
    Document,
    DocumentArtifact,
    DocumentPassage,
    DocumentReference,
    DocumentRevision,
    DocumentSection,
    DocumentTable,
    IngestJob,
    RetrievalIndexRun,
)
from paper_context.queue.contracts import IngestionQueue, IngestQueuePayload
from paper_context.queue.pgmq import PgmqMessage
from paper_context.retrieval import DocumentRetrievalIndexer
from paper_context.storage.base import StorageArtifact, StorageInterface

from .enrichment import MetadataEnricher
from .identifiers import artifact_id
from .parsers import PdfParser
from .types import EnrichmentResult, ParsedDocument, ParsedParagraph, ParsedSection, ParserResult


class IngestJobRow(TypedDict):
    id: uuid.UUID
    document_id: uuid.UUID
    revision_id: uuid.UUID
    created_at: datetime
    status: str
    warnings: list[str]
    source_artifact_id: uuid.UUID | None


class SourceArtifactRow(TypedDict):
    id: uuid.UUID
    storage_ref: str
    checksum: str | None


@dataclass(frozen=True)
class _PreparedIngestState:
    document_id: uuid.UUID
    revision_id: uuid.UUID
    created_at: datetime
    storage_ref: str
    warnings: list[str]


@dataclass(frozen=True)
class _StagedParserArtifact:
    parser_result: ParserResult
    stored_artifact: StorageArtifact
    is_primary: bool


@dataclass(frozen=True)
class IngestJobContext:
    message: PgmqMessage
    payload: IngestQueuePayload


ConnectionFactory = Callable[..., AbstractContextManager[Connection]]


class LeaseExtender:
    def __init__(
        self,
        connection_factory: ConnectionFactory,
        queue_adapter: IngestionQueue,
        message: PgmqMessage,
        default_vt_seconds: int,
    ) -> None:
        self._connection_factory = connection_factory
        self._queue_adapter = queue_adapter
        self._message = message
        self._default_vt_seconds = default_vt_seconds

    def extend(self, vt_seconds: int | None = None) -> None:
        vt_seconds = vt_seconds or self._default_vt_seconds
        try:
            context = self._connection_factory(transactional=True)
        except TypeError:
            context = self._connection_factory()
        with context as connection:
            self._queue_adapter.extend_lease(connection, self._message.msg_id, vt_seconds)


class IngestExecutionDeferred(RuntimeError):
    """Raised when a claimed job should be retried later without archiving the queue message."""


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
        retrieval_indexer: DocumentRetrievalIndexer | None = None,
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
        self._retrieval_indexer = retrieval_indexer or DocumentRetrievalIndexer(
            index_version=index_version,
            chunking_version=chunking_version,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )

    def process(
        self,
        connection: Connection,
        context: IngestJobContext,
        lease: LeaseExtender,
    ) -> None:
        if not self._try_claim_processing_lock(
            connection,
            ingest_job_id=context.payload.ingest_job_id,
        ):
            raise IngestExecutionDeferred(
                f"ingest job {context.payload.ingest_job_id} is already being processed"
            )
        self._clear_implicit_transaction(connection)
        created_storage_refs: list[str] = []
        try:
            with connection.begin():
                prepared = self._prepare_for_processing(
                    connection,
                    ingest_job_id=context.payload.ingest_job_id,
                )
            if prepared is None:
                return

            document_id = prepared.document_id
            revision_id = prepared.revision_id
            warnings = list(prepared.warnings)
            source_path = self._storage.resolve(prepared.storage_ref)
            primary_result = self._primary_parser.parse(
                filename=prepared.storage_ref,
                source_path=source_path,
            )
            primary_artifact = self._store_parser_artifact(
                document_id=document_id,
                revision_id=revision_id,
                ingest_job_id=context.payload.ingest_job_id,
                artifact=primary_result,
                is_primary=primary_result.gate_status == "pass",
                created_storage_refs=created_storage_refs,
            )
            lease.extend()

            active_result = primary_result
            active_artifact = primary_artifact
            parser_source = primary_result.artifact.parser
            warnings = _dedupe_warnings([*warnings, *primary_result.warnings])
            if primary_result.gate_status == "degraded":
                warnings = _dedupe_warnings([*warnings, "parser_fallback_used"])
                fallback_result = self._fallback_parser.parse(
                    filename=prepared.storage_ref,
                    source_path=source_path,
                )
                fallback_artifact = self._store_parser_artifact(
                    document_id=document_id,
                    revision_id=revision_id,
                    ingest_job_id=context.payload.ingest_job_id,
                    artifact=fallback_result,
                    is_primary=fallback_result.gate_status == "pass",
                    created_storage_refs=created_storage_refs,
                )
                lease.extend()
                warnings = _dedupe_warnings([*warnings, *fallback_result.warnings])
                if fallback_result.gate_status != "pass" or fallback_result.parsed_document is None:
                    with connection.begin():
                        warnings = self._sync_processing_warnings(
                            connection,
                            ingest_job_id=context.payload.ingest_job_id,
                            document_id=document_id,
                            created_at=prepared.created_at,
                            warnings=warnings,
                        )
                        if warnings is None:
                            return
                        self._record_parser_artifact(
                            connection,
                            document_id=document_id,
                            revision_id=revision_id,
                            ingest_job_id=context.payload.ingest_job_id,
                            staged_artifact=primary_artifact,
                        )
                        self._record_parser_artifact(
                            connection,
                            document_id=document_id,
                            revision_id=revision_id,
                            ingest_job_id=context.payload.ingest_job_id,
                            staged_artifact=fallback_artifact,
                        )
                        self._mark_failed(
                            connection,
                            ingest_job_id=context.payload.ingest_job_id,
                            document_id=document_id,
                            revision_id=revision_id,
                            failure_code=(
                                fallback_result.failure_code or "pdfplumber_structure_failed"
                            ),
                            failure_message=(
                                fallback_result.failure_message
                                or (
                                    "Fallback parsing could not recover stable passages "
                                    "with provenance."
                                )
                            ),
                            warnings=warnings,
                        )
                    return
                active_result = fallback_result
                active_artifact = fallback_artifact
                parser_source = fallback_result.artifact.parser
            elif primary_result.gate_status == "fail":
                with connection.begin():
                    warnings = self._sync_processing_warnings(
                        connection,
                        ingest_job_id=context.payload.ingest_job_id,
                        document_id=document_id,
                        created_at=prepared.created_at,
                        warnings=warnings,
                    )
                    if warnings is None:
                        return
                    self._record_parser_artifact(
                        connection,
                        document_id=document_id,
                        revision_id=revision_id,
                        ingest_job_id=context.payload.ingest_job_id,
                        staged_artifact=primary_artifact,
                    )
                    self._mark_failed(
                        connection,
                        ingest_job_id=context.payload.ingest_job_id,
                        document_id=document_id,
                        revision_id=revision_id,
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
                parsed_document = active_result.load_parsed_document()
            if parsed_document is None:
                raise RuntimeError(
                    "parsed_document is unexpectedly missing after parser validation"
                )

            if (parsed_document.metadata_confidence or 0.0) < 0.5:
                warnings = _dedupe_warnings([*warnings, "metadata_low_confidence"])

            with connection.begin():
                warnings = self._sync_processing_warnings(
                    connection,
                    ingest_job_id=context.payload.ingest_job_id,
                    document_id=document_id,
                    created_at=prepared.created_at,
                    warnings=warnings,
                )
                if warnings is None:
                    return
                self._record_parser_artifact(
                    connection,
                    document_id=document_id,
                    revision_id=revision_id,
                    ingest_job_id=context.payload.ingest_job_id,
                    staged_artifact=primary_artifact,
                )
                if active_result is not primary_result:
                    self._record_parser_artifact(
                        connection,
                        document_id=document_id,
                        revision_id=revision_id,
                        ingest_job_id=context.payload.ingest_job_id,
                        staged_artifact=active_artifact,
                    )
                active_artifact_id = self._parser_artifact_id(
                    ingest_job_id=context.payload.ingest_job_id,
                    artifact=active_result,
                )
                self._mark_stage(
                    connection,
                    ingest_job_id=context.payload.ingest_job_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    status="normalizing",
                    warnings=warnings,
                )
                section_ids = self._normalize_document(
                    connection,
                    document_id=document_id,
                    revision_id=revision_id,
                    parsed_document=parsed_document,
                    artifact_id=active_artifact_id,
                )
                self._apply_document_metadata(
                    connection,
                    document_id=document_id,
                    revision_id=revision_id,
                    title=parsed_document.title,
                    authors=parsed_document.authors,
                    abstract=parsed_document.abstract,
                    publication_year=parsed_document.publication_year,
                    metadata_confidence=parsed_document.metadata_confidence,
                )
                self._mark_stage(
                    connection,
                    ingest_job_id=context.payload.ingest_job_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    status="enriching_metadata",
                    warnings=warnings,
                )
            lease.extend()
            enrichment = self._metadata_enricher.enrich(parsed_document)
            warnings = _dedupe_warnings([*warnings, *enrichment.warnings])
            lease.extend()
            with connection.begin():
                warnings = self._sync_processing_warnings(
                    connection,
                    ingest_job_id=context.payload.ingest_job_id,
                    document_id=document_id,
                    created_at=prepared.created_at,
                    warnings=warnings,
                )
                if warnings is None:
                    return
                self._apply_enriched_document_metadata(
                    connection,
                    document_id=document_id,
                    revision_id=revision_id,
                    parsed_document=parsed_document,
                    enrichment=enrichment,
                    warnings=warnings,
                )
                self._mark_stage(
                    connection,
                    ingest_job_id=context.payload.ingest_job_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    status="chunking",
                    warnings=warnings,
                )
                self._insert_passages(
                    connection,
                    document_id=document_id,
                    revision_id=revision_id,
                    parsed_document=parsed_document,
                    section_ids=section_ids,
                    artifact_id=active_artifact_id,
                )
                self._mark_stage(
                    connection,
                    ingest_job_id=context.payload.ingest_job_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    status="indexing",
                    warnings=warnings,
                )
            lease.extend()
            with connection.begin():
                warnings = self._sync_processing_warnings(
                    connection,
                    ingest_job_id=context.payload.ingest_job_id,
                    document_id=document_id,
                    created_at=prepared.created_at,
                    warnings=warnings,
                )
                if warnings is None:
                    return
                self._retrieval_indexer.rebuild(
                    connection,
                    document_id=document_id,
                    revision_id=revision_id,
                    ingest_job_id=context.payload.ingest_job_id,
                    parser_source=parser_source,
                )
                self._mark_ready(
                    connection,
                    ingest_job_id=context.payload.ingest_job_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    warnings=warnings,
                )
        except Exception:
            for storage_ref in reversed(created_storage_refs):
                self._storage.delete(storage_ref)
            raise
        finally:
            self._release_processing_lock(
                connection,
                ingest_job_id=context.payload.ingest_job_id,
            )

    def _prepare_for_processing(
        self,
        connection: Connection,
        *,
        ingest_job_id: uuid.UUID,
    ) -> _PreparedIngestState | None:
        job = self._lock_ingest_job(connection, ingest_job_id)
        if job is None:
            raise LookupError(f"missing ingest job {ingest_job_id}")
        if job["status"] in self.TERMINAL_STATUSES:
            return None

        document_id = job["document_id"]
        revision_id = job["revision_id"]
        warnings = list(job["warnings"])
        self._lock_document(connection, document_id)
        self._lock_revision(connection, revision_id)
        if self._is_superseded(
            connection,
            ingest_job_id=ingest_job_id,
            document_id=document_id,
            created_at=job["created_at"],
        ):
            self._mark_failed(
                connection,
                ingest_job_id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                failure_code="superseded_by_newer_ingest_job",
                failure_message="A newer ingest job superseded this run before processing began.",
                warnings=warnings,
            )
            return None

        source_artifact = self._load_source_artifact(
            connection,
            ingest_job_id=ingest_job_id,
            source_artifact_id=job["source_artifact_id"],
        )
        if source_artifact is None:
            self._mark_failed(
                connection,
                ingest_job_id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                failure_code="missing_source_artifact",
                failure_message="No source PDF artifact exists for this ingest job.",
                warnings=warnings,
            )
            return None

        self._reset_revision_state(
            connection,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
        )
        self._mark_stage(
            connection,
            ingest_job_id=ingest_job_id,
            document_id=document_id,
            revision_id=revision_id,
            status="parsing",
            warnings=warnings,
        )
        return _PreparedIngestState(
            document_id=document_id,
            revision_id=revision_id,
            created_at=job["created_at"],
            storage_ref=source_artifact["storage_ref"],
            warnings=warnings,
        )

    def _try_claim_processing_lock(
        self,
        connection: Connection,
        *,
        ingest_job_id: uuid.UUID,
    ) -> bool:
        key_hi, key_lo = _advisory_lock_keys(ingest_job_id)
        return bool(
            connection.execute(
                text("SELECT pg_try_advisory_lock(:key_hi, :key_lo)"),
                {"key_hi": key_hi, "key_lo": key_lo},
            ).scalar_one()
        )

    def _release_processing_lock(
        self,
        connection: Connection,
        *,
        ingest_job_id: uuid.UUID,
    ) -> None:
        key_hi, key_lo = _advisory_lock_keys(ingest_job_id)
        connection.execute(
            text("SELECT pg_advisory_unlock(:key_hi, :key_lo)"),
            {"key_hi": key_hi, "key_lo": key_lo},
        )

    def _clear_implicit_transaction(self, connection: Connection) -> None:
        in_transaction = connection.in_transaction()
        if in_transaction is True:
            connection.commit()

    def _sync_processing_warnings(
        self,
        connection: Connection,
        *,
        ingest_job_id: uuid.UUID,
        document_id: uuid.UUID,
        created_at: datetime,
        warnings: list[str],
    ) -> list[str] | None:
        job = self._load_ingest_job(connection, ingest_job_id=ingest_job_id)
        if job is None:
            raise LookupError(f"missing ingest job {ingest_job_id}")
        job_status = str(job.get("status", "queued"))
        if job_status in self.TERMINAL_STATUSES:
            return None
        job_warnings = cast(list[str], job.get("warnings", []))
        merged_warnings = _dedupe_warnings([*job_warnings, *warnings])
        if self._is_superseded(
            connection,
            ingest_job_id=ingest_job_id,
            document_id=document_id,
            created_at=created_at,
        ):
            self._mark_failed(
                connection,
                ingest_job_id=ingest_job_id,
                document_id=document_id,
                revision_id=job["revision_id"],
                failure_code="superseded_by_newer_ingest_job",
                failure_message=(
                    "A newer ingest job superseded this run before processing completed."
                ),
                warnings=merged_warnings,
            )
            return None
        return merged_warnings

    def _load_ingest_job(
        self,
        connection: Connection,
        ingest_job_id: uuid.UUID,
        *,
        for_update: bool = False,
    ) -> IngestJobRow | None:
        statement = select(
            IngestJob.id,
            IngestJob.document_id,
            IngestJob.revision_id,
            IngestJob.source_artifact_id,
            IngestJob.created_at,
            IngestJob.status,
            func.coalesce(IngestJob.warnings, sa_cast([], JSONB)).label("warnings"),
        ).where(IngestJob.id == ingest_job_id)
        if for_update:
            statement = statement.with_for_update()
        row = connection.execute(statement).mappings().one_or_none()
        if row is None:
            return None
        return cast(IngestJobRow, dict(row))

    def _lock_ingest_job(
        self,
        connection: Connection,
        ingest_job_id: uuid.UUID,
    ) -> IngestJobRow | None:
        return self._load_ingest_job(
            connection,
            ingest_job_id=ingest_job_id,
            for_update=True,
        )

    def _lock_document(self, connection: Connection, document_id: uuid.UUID) -> None:
        connection.execute(
            select(Document.id).where(Document.id == document_id).with_for_update()
        ).scalar_one()

    def _lock_revision(self, connection: Connection, revision_id: uuid.UUID) -> None:
        connection.execute(
            select(DocumentRevision.id).where(DocumentRevision.id == revision_id).with_for_update()
        ).scalar_one()

    def _load_source_artifact(
        self,
        connection: Connection,
        *,
        ingest_job_id: uuid.UUID,
        source_artifact_id: uuid.UUID | None,
    ) -> SourceArtifactRow | None:
        if source_artifact_id is None:
            return None
        row = (
            connection.execute(
                select(
                    DocumentArtifact.id,
                    DocumentArtifact.storage_ref,
                    DocumentArtifact.checksum,
                ).where(
                    DocumentArtifact.id == source_artifact_id,
                    DocumentArtifact.ingest_job_id == ingest_job_id,
                    DocumentArtifact.artifact_type == "source_pdf",
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            return None
        return cast(SourceArtifactRow, dict(row))

    def _reset_revision_state(
        self,
        connection: Connection,
        *,
        revision_id: uuid.UUID,
        ingest_job_id: uuid.UUID,
    ) -> None:
        params = {"revision_id": revision_id, "ingest_job_id": ingest_job_id}
        connection.execute(
            delete(DocumentPassage).where(
                DocumentPassage.revision_id == bindparam("b_revision_id")
            ),
            {**params, "b_revision_id": revision_id},
        )
        connection.execute(
            delete(DocumentTable).where(DocumentTable.revision_id == bindparam("b_revision_id")),
            {**params, "b_revision_id": revision_id},
        )
        connection.execute(
            delete(DocumentReference).where(
                DocumentReference.revision_id == bindparam("b_revision_id")
            ),
            {**params, "b_revision_id": revision_id},
        )
        connection.execute(
            delete(DocumentSection).where(
                DocumentSection.revision_id == bindparam("b_revision_id")
            ),
            {**params, "b_revision_id": revision_id},
        )
        connection.execute(
            delete(RetrievalIndexRun).where(
                RetrievalIndexRun.revision_id == bindparam("b_revision_id")
            ),
            {**params, "b_revision_id": revision_id},
        )
        connection.execute(
            delete(DocumentArtifact).where(
                DocumentArtifact.revision_id == bindparam("b_revision_id"),
                DocumentArtifact.artifact_type != "source_pdf",
            ),
            {**params, "b_revision_id": revision_id},
        )
        connection.execute(
            update(DocumentArtifact)
            .where(DocumentArtifact.revision_id == bindparam("b_revision_id"))
            .values(is_primary=False),
            {**params, "b_revision_id": revision_id},
        )

    def _parser_artifact_id(
        self,
        *,
        ingest_job_id: uuid.UUID,
        artifact: ParserResult,
    ) -> uuid.UUID:
        return artifact_id(
            ingest_job_id=ingest_job_id,
            artifact_type=artifact.artifact.artifact_type,
            parser=artifact.artifact.parser,
        )

    def _store_parser_artifact(
        self,
        *,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        ingest_job_id: uuid.UUID,
        artifact: ParserResult,
        is_primary: bool,
        created_storage_refs: list[str] | None = None,
    ) -> _StagedParserArtifact:
        artifact_path = f"documents/{document_id}/{ingest_job_id}/{artifact.artifact.filename}"
        try:
            if artifact.parsed_document is None and artifact.parsed_document_loader is not None:
                artifact.load_parsed_document()
            if artifact.artifact.content_path is not None:
                with artifact.artifact.content_path.open("rb") as handle:
                    stored_artifact = self._storage.store_file(artifact_path, handle)
            elif artifact.artifact.content is not None:
                stored_artifact = self._storage.store_bytes(
                    artifact_path, artifact.artifact.content
                )
            else:
                raise ValueError("parser artifact is missing both content and content_path")
        finally:
            artifact.artifact.cleanup_local_copy()
        if created_storage_refs is not None:
            created_storage_refs.append(stored_artifact.storage_ref)
        return _StagedParserArtifact(
            parser_result=artifact,
            stored_artifact=stored_artifact,
            is_primary=is_primary,
        )

    def _record_parser_artifact(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        ingest_job_id: uuid.UUID,
        staged_artifact: _StagedParserArtifact,
    ) -> uuid.UUID:
        parser_artifact_id = self._parser_artifact_id(
            ingest_job_id=ingest_job_id,
            artifact=staged_artifact.parser_result,
        )
        statement = pg_insert(DocumentArtifact).values(
            id=bindparam("b_id"),
            document_id=bindparam("b_document_id"),
            revision_id=bindparam("b_revision_id"),
            ingest_job_id=bindparam("b_ingest_job_id"),
            artifact_type=bindparam("b_artifact_type"),
            parser=bindparam("b_parser"),
            storage_ref=bindparam("b_storage_ref"),
            checksum=bindparam("b_checksum"),
            is_primary=bindparam("b_is_primary"),
        )
        connection.execute(
            statement.on_conflict_do_update(
                index_elements=[DocumentArtifact.id],
                set_={
                    "storage_ref": statement.excluded.storage_ref,
                    "checksum": statement.excluded.checksum,
                    "is_primary": statement.excluded.is_primary,
                },
            ),
            {
                "id": parser_artifact_id,
                "b_id": parser_artifact_id,
                "document_id": document_id,
                "b_document_id": document_id,
                "revision_id": revision_id,
                "b_revision_id": revision_id,
                "ingest_job_id": ingest_job_id,
                "b_ingest_job_id": ingest_job_id,
                "artifact_type": staged_artifact.parser_result.artifact.artifact_type,
                "b_artifact_type": staged_artifact.parser_result.artifact.artifact_type,
                "parser": staged_artifact.parser_result.artifact.parser,
                "b_parser": staged_artifact.parser_result.artifact.parser,
                "storage_ref": staged_artifact.stored_artifact.storage_ref,
                "b_storage_ref": staged_artifact.stored_artifact.storage_ref,
                "checksum": staged_artifact.stored_artifact.checksum,
                "b_checksum": staged_artifact.stored_artifact.checksum,
                "is_primary": staged_artifact.is_primary,
                "b_is_primary": staged_artifact.is_primary,
            },
        )
        return parser_artifact_id

    def _persist_parser_artifact(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        ingest_job_id: uuid.UUID,
        artifact: ParserResult,
        is_primary: bool,
        created_storage_refs: list[str] | None = None,
    ) -> uuid.UUID:
        staged_artifact = self._store_parser_artifact(
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            artifact=artifact,
            is_primary=is_primary,
            created_storage_refs=created_storage_refs,
        )
        return self._record_parser_artifact(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            staged_artifact=staged_artifact,
        )

    def _normalize_document(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
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
                insert(DocumentSection).values(
                    id=bindparam("b_id"),
                    document_id=bindparam("b_document_id"),
                    revision_id=bindparam("b_revision_id"),
                    parent_section_id=bindparam("b_parent_section_id"),
                    heading=bindparam("b_heading"),
                    heading_path=bindparam("b_heading_path", type_=JSONB),
                    ordinal=bindparam("b_ordinal"),
                    page_start=bindparam("b_page_start"),
                    page_end=bindparam("b_page_end"),
                    artifact_id=bindparam("b_artifact_id"),
                ),
                {
                    "id": section_id,
                    "b_id": section_id,
                    "document_id": document_id,
                    "b_document_id": document_id,
                    "revision_id": revision_id,
                    "b_revision_id": revision_id,
                    "parent_section_id": parent_section_id,
                    "b_parent_section_id": parent_section_id,
                    "heading": section.heading,
                    "b_heading": section.heading,
                    "heading_path": section.heading_path,
                    "b_heading_path": section.heading_path,
                    "ordinal": len(section_ids),
                    "b_ordinal": len(section_ids),
                    "page_start": section.page_start,
                    "b_page_start": section.page_start,
                    "page_end": section.page_end,
                    "b_page_end": section.page_end,
                    "artifact_id": artifact_id,
                    "b_artifact_id": artifact_id,
                },
            )

        for table in parsed_document.tables:
            table_row_id = uuid.uuid4()
            connection.execute(
                insert(DocumentTable).values(
                    id=bindparam("b_id"),
                    document_id=bindparam("b_document_id"),
                    revision_id=bindparam("b_revision_id"),
                    section_id=bindparam("b_section_id"),
                    caption=bindparam("b_caption"),
                    table_type=bindparam("b_table_type"),
                    headers_json=bindparam("b_headers_json", type_=JSONB),
                    rows_json=bindparam("b_rows_json", type_=JSONB),
                    page_start=bindparam("b_page_start"),
                    page_end=bindparam("b_page_end"),
                    artifact_id=bindparam("b_artifact_id"),
                ),
                {
                    "id": table_row_id,
                    "b_id": table_row_id,
                    "document_id": document_id,
                    "b_document_id": document_id,
                    "revision_id": revision_id,
                    "b_revision_id": revision_id,
                    "section_id": section_ids[table.section_key],
                    "b_section_id": section_ids[table.section_key],
                    "caption": table.caption,
                    "b_caption": table.caption,
                    "table_type": "lexical",
                    "b_table_type": "lexical",
                    "headers_json": table.headers,
                    "b_headers_json": table.headers,
                    "rows_json": table.rows,
                    "b_rows_json": table.rows,
                    "page_start": table.page_start,
                    "b_page_start": table.page_start,
                    "page_end": table.page_end,
                    "b_page_end": table.page_end,
                    "artifact_id": artifact_id,
                    "b_artifact_id": artifact_id,
                },
            )

        for reference in parsed_document.references:
            reference_row_id = uuid.uuid4()
            connection.execute(
                insert(DocumentReference).values(
                    id=bindparam("b_id"),
                    document_id=bindparam("b_document_id"),
                    revision_id=bindparam("b_revision_id"),
                    raw_citation=bindparam("b_raw_citation"),
                    normalized_title=bindparam("b_normalized_title"),
                    authors=bindparam("b_authors", type_=JSONB),
                    publication_year=bindparam("b_publication_year"),
                    doi=bindparam("b_doi"),
                    source_confidence=bindparam("b_source_confidence"),
                    artifact_id=bindparam("b_artifact_id"),
                ),
                {
                    "id": reference_row_id,
                    "b_id": reference_row_id,
                    "document_id": document_id,
                    "b_document_id": document_id,
                    "revision_id": revision_id,
                    "b_revision_id": revision_id,
                    "raw_citation": reference.raw_citation,
                    "b_raw_citation": reference.raw_citation,
                    "normalized_title": reference.normalized_title,
                    "b_normalized_title": reference.normalized_title,
                    "authors": reference.authors or [],
                    "b_authors": reference.authors or [],
                    "publication_year": reference.publication_year,
                    "b_publication_year": reference.publication_year,
                    "doi": reference.doi,
                    "b_doi": reference.doi,
                    "source_confidence": reference.source_confidence,
                    "b_source_confidence": reference.source_confidence,
                    "artifact_id": artifact_id,
                    "b_artifact_id": artifact_id,
                },
            )

        return section_ids

    def _apply_document_metadata(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        title: str | None,
        authors: list[str],
        abstract: str | None,
        publication_year: int | None,
        metadata_confidence: float | None,
    ) -> None:
        now = datetime.now(UTC)
        connection.execute(
            update(DocumentRevision)
            .where(DocumentRevision.id == bindparam("b_revision_id"))
            .values(
                title=func.coalesce(bindparam("b_title"), DocumentRevision.title),
                authors=bindparam("b_authors", type_=JSONB),
                abstract=bindparam("b_abstract"),
                publication_year=bindparam("b_publication_year"),
                metadata_confidence=bindparam("b_metadata_confidence"),
                status=bindparam("b_status"),
                updated_at=bindparam("b_now"),
            ),
            {
                "document_id": document_id,
                "b_document_id": document_id,
                "revision_id": revision_id,
                "b_revision_id": revision_id,
                "title": title,
                "b_title": title,
                "authors": authors,
                "b_authors": authors,
                "abstract": abstract,
                "b_abstract": abstract,
                "publication_year": publication_year,
                "b_publication_year": publication_year,
                "metadata_confidence": metadata_confidence,
                "b_metadata_confidence": metadata_confidence,
                "status": "normalizing",
                "b_status": "normalizing",
                "now": now,
                "b_now": now,
            },
        )

    def _apply_enriched_document_metadata(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
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
            revision_id=revision_id,
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
        revision_id: uuid.UUID,
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
                provenance_offsets = {
                    "pages": [
                        page
                        for paragraph in chunk
                        for page in (paragraph.provenance_offsets or {}).get("pages", [])
                    ],
                    "charspans": [
                        charspan
                        for paragraph in chunk
                        for charspan in (paragraph.provenance_offsets or {}).get("charspans", [])
                    ],
                }
                passage_row_id = uuid.uuid4()
                connection.execute(
                    insert(DocumentPassage).values(
                        id=bindparam("b_id"),
                        document_id=bindparam("b_document_id"),
                        revision_id=bindparam("b_revision_id"),
                        section_id=bindparam("b_section_id"),
                        chunk_ordinal=bindparam("b_chunk_ordinal"),
                        body_text=bindparam("b_body_text"),
                        contextualized_text=bindparam("b_contextualized_text"),
                        token_count=bindparam("b_token_count"),
                        page_start=bindparam("b_page_start"),
                        page_end=bindparam("b_page_end"),
                        provenance_offsets=bindparam("b_provenance_offsets", type_=JSONB),
                        artifact_id=bindparam("b_artifact_id"),
                    ),
                    {
                        "id": passage_row_id,
                        "b_id": passage_row_id,
                        "document_id": document_id,
                        "b_document_id": document_id,
                        "revision_id": revision_id,
                        "b_revision_id": revision_id,
                        "section_id": section_ids[section.key],
                        "b_section_id": section_ids[section.key],
                        "chunk_ordinal": chunk_ordinal,
                        "b_chunk_ordinal": chunk_ordinal,
                        "body_text": body_text,
                        "b_body_text": body_text,
                        "contextualized_text": contextualized_text,
                        "b_contextualized_text": contextualized_text,
                        "token_count": token_count,
                        "b_token_count": token_count,
                        "page_start": min(
                            (
                                paragraph.page_start
                                for paragraph in chunk
                                if paragraph.page_start is not None
                            ),
                            default=None,
                        ),
                        "b_page_start": min(
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
                        "b_page_end": max(
                            (
                                paragraph.page_end
                                for paragraph in chunk
                                if paragraph.page_end is not None
                            ),
                            default=None,
                        ),
                        "provenance_offsets": provenance_offsets,
                        "b_provenance_offsets": provenance_offsets,
                        "artifact_id": artifact_id,
                        "b_artifact_id": artifact_id,
                    },
                )

    def _replace_index_run(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        ingest_job_id: uuid.UUID,
        parser_source: str,
    ) -> None:
        self._retrieval_indexer.rebuild(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            parser_source=parser_source,
        )

    def _is_superseded(
        self,
        connection: Connection,
        *,
        ingest_job_id: uuid.UUID,
        document_id: uuid.UUID,
        created_at: datetime,
    ) -> bool:
        newer_job = aliased(IngestJob)
        result = connection.execute(
            select(
                exists(
                    select(1).where(
                        newer_job.document_id == document_id,
                        newer_job.id != ingest_job_id,
                        or_(
                            newer_job.created_at > created_at,
                            and_(
                                newer_job.created_at == created_at,
                                sa_cast(newer_job.id, String) > str(ingest_job_id),
                            ),
                        ),
                    )
                )
            )
        )
        return bool(result.scalar_one())

    def _update_document_status_if_current(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        ingest_job_id: uuid.UUID,
        status: str,
        now: datetime,
    ) -> None:
        current_job = aliased(IngestJob)
        newer_job = aliased(IngestJob)
        newer_job_exists = exists(
            select(1)
            .select_from(current_job)
            .join(
                newer_job,
                and_(
                    newer_job.document_id == current_job.document_id,
                    newer_job.id != current_job.id,
                    or_(
                        newer_job.created_at > current_job.created_at,
                        and_(
                            newer_job.created_at == current_job.created_at,
                            sa_cast(newer_job.id, String) > sa_cast(current_job.id, String),
                        ),
                    ),
                ),
            )
            .where(current_job.id == ingest_job_id)
        )
        connection.execute(
            update(Document)
            .where(
                Document.id == document_id,
                ~newer_job_exists,
            )
            .values(
                current_status=status,
                updated_at=now,
            )
        )

    def _mark_stage(
        self,
        connection: Connection,
        *,
        ingest_job_id: uuid.UUID,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        status: str,
        warnings: list[str],
    ) -> None:
        now = datetime.now(UTC)
        connection.execute(
            update(IngestJob)
            .where(IngestJob.id == bindparam("b_ingest_job_id"))
            .values(
                status=bindparam("b_status"),
                failure_code=None,
                failure_message=None,
                warnings=bindparam("b_warnings", type_=JSONB),
                started_at=func.coalesce(IngestJob.started_at, bindparam("b_now")),
                finished_at=None,
            ),
            {
                "ingest_job_id": ingest_job_id,
                "b_ingest_job_id": ingest_job_id,
                "status": status,
                "b_status": status,
                "warnings": warnings,
                "b_warnings": warnings,
                "now": now,
                "b_now": now,
            },
        )
        connection.execute(
            update(DocumentRevision)
            .where(DocumentRevision.id == bindparam("b_revision_id"))
            .values(
                status=bindparam("b_status"),
                updated_at=bindparam("b_now"),
            ),
            {
                "revision_id": revision_id,
                "b_revision_id": revision_id,
                "status": status,
                "b_status": status,
                "now": now,
                "b_now": now,
            },
        )
        self._update_document_status_if_current(
            connection,
            document_id=document_id,
            ingest_job_id=ingest_job_id,
            status=status,
            now=now,
        )

    def _mark_failed(
        self,
        connection: Connection,
        *,
        ingest_job_id: uuid.UUID,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        failure_code: str,
        failure_message: str,
        warnings: list[str],
    ) -> None:
        now = datetime.now(UTC)
        connection.execute(
            update(IngestJob)
            .where(IngestJob.id == bindparam("b_ingest_job_id"))
            .values(
                status="failed",
                failure_code=bindparam("b_failure_code"),
                failure_message=bindparam("b_failure_message"),
                warnings=bindparam("b_warnings", type_=JSONB),
                started_at=func.coalesce(IngestJob.started_at, bindparam("b_now")),
                finished_at=bindparam("b_now"),
            ),
            {
                "ingest_job_id": ingest_job_id,
                "b_ingest_job_id": ingest_job_id,
                "failure_code": failure_code,
                "b_failure_code": failure_code,
                "failure_message": failure_message,
                "b_failure_message": failure_message,
                "warnings": _dedupe_warnings(warnings),
                "b_warnings": _dedupe_warnings(warnings),
                "now": now,
                "b_now": now,
            },
        )
        connection.execute(
            update(DocumentRevision)
            .where(DocumentRevision.id == bindparam("b_revision_id"))
            .values(
                status="failed",
                updated_at=bindparam("b_now"),
            ),
            {
                "revision_id": revision_id,
                "b_revision_id": revision_id,
                "now": now,
                "b_now": now,
            },
        )
        self._update_document_status_if_current(
            connection,
            document_id=document_id,
            ingest_job_id=ingest_job_id,
            status="failed",
            now=now,
        )

    def _mark_ready(
        self,
        connection: Connection,
        *,
        ingest_job_id: uuid.UUID,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        warnings: list[str],
    ) -> None:
        now = datetime.now(UTC)
        connection.execute(
            update(IngestJob)
            .where(IngestJob.id == bindparam("b_ingest_job_id"))
            .values(
                status="ready",
                warnings=bindparam("b_warnings", type_=JSONB),
                finished_at=bindparam("b_now"),
            ),
            {
                "ingest_job_id": ingest_job_id,
                "b_ingest_job_id": ingest_job_id,
                "warnings": _dedupe_warnings(warnings),
                "b_warnings": _dedupe_warnings(warnings),
                "now": now,
                "b_now": now,
            },
        )
        self._activate_revision_if_current(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            ingest_job_id=ingest_job_id,
            now=now,
        )
        self._update_document_status_if_current(
            connection,
            document_id=document_id,
            ingest_job_id=ingest_job_id,
            status="ready",
            now=now,
        )

    def _activate_revision_if_current(
        self,
        connection: Connection,
        *,
        document_id: uuid.UUID,
        revision_id: uuid.UUID,
        ingest_job_id: uuid.UUID,
        now: datetime,
    ) -> None:
        current_job = aliased(IngestJob)
        newer_job = aliased(IngestJob)
        newer_job_exists = exists(
            select(1)
            .select_from(current_job)
            .join(
                newer_job,
                and_(
                    newer_job.document_id == current_job.document_id,
                    newer_job.id != current_job.id,
                    or_(
                        newer_job.created_at > current_job.created_at,
                        and_(
                            newer_job.created_at == current_job.created_at,
                            sa_cast(newer_job.id, String) > sa_cast(current_job.id, String),
                        ),
                    ),
                ),
            )
            .where(current_job.id == ingest_job_id)
        )
        previous_active_revision_id = connection.execute(
            select(Document.active_revision_id).where(Document.id == document_id)
        ).scalar_one()
        revision_row = (
            connection.execute(
                select(
                    DocumentRevision.title,
                    DocumentRevision.authors,
                    DocumentRevision.abstract,
                    DocumentRevision.publication_year,
                    DocumentRevision.source_type,
                    DocumentRevision.metadata_confidence,
                    DocumentRevision.quant_tags,
                ).where(DocumentRevision.id == revision_id)
            )
            .mappings()
            .one()
        )
        promoted_rows = connection.execute(
            update(Document)
            .where(
                Document.id == document_id,
                ~newer_job_exists,
            )
            .values(
                active_revision_id=revision_id,
                title=revision_row["title"],
                authors=revision_row["authors"],
                abstract=revision_row["abstract"],
                publication_year=revision_row["publication_year"],
                source_type=revision_row["source_type"],
                metadata_confidence=revision_row["metadata_confidence"],
                quant_tags=revision_row["quant_tags"],
                updated_at=now,
            )
        ).rowcount
        if promoted_rows == 0:
            connection.execute(
                update(DocumentRevision)
                .where(DocumentRevision.id == bindparam("b_revision_id"))
                .values(
                    status="ready",
                    updated_at=bindparam("b_now"),
                ),
                {
                    "revision_id": revision_id,
                    "b_revision_id": revision_id,
                    "now": now,
                    "b_now": now,
                },
            )
            return
        connection.execute(
            update(DocumentRevision)
            .where(DocumentRevision.id == bindparam("b_revision_id"))
            .values(
                status="ready",
                activated_at=bindparam("b_now"),
                superseded_at=None,
                updated_at=bindparam("b_now"),
            ),
            {
                "revision_id": revision_id,
                "b_revision_id": revision_id,
                "now": now,
                "b_now": now,
            },
        )
        if previous_active_revision_id is not None and previous_active_revision_id != revision_id:
            connection.execute(
                update(DocumentRevision)
                .where(DocumentRevision.id == bindparam("b_previous_revision_id"))
                .values(
                    superseded_at=bindparam("b_now"),
                    updated_at=bindparam("b_now"),
                ),
                {
                    "previous_revision_id": previous_active_revision_id,
                    "b_previous_revision_id": previous_active_revision_id,
                    "now": now,
                    "b_now": now,
                },
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
                select(IngestJob.status, IngestJob.revision_id)
                .where(IngestJob.id == context.payload.ingest_job_id)
                .with_for_update()
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise LookupError(f"missing ingest job {context.payload.ingest_job_id}")

        if row["status"] in self.TERMINAL_STATUSES:
            return

        revision_id = row["revision_id"]
        now = datetime.now(UTC)
        connection.execute(
            update(IngestJob)
            .where(IngestJob.id == bindparam("b_ingest_job_id"))
            .values(
                status="parsing",
                started_at=func.coalesce(IngestJob.started_at, bindparam("b_now")),
            ),
            {
                "ingest_job_id": context.payload.ingest_job_id,
                "b_ingest_job_id": context.payload.ingest_job_id,
                "now": now,
                "b_now": now,
            },
        )
        connection.execute(
            update(DocumentRevision)
            .where(DocumentRevision.id == bindparam("b_revision_id"))
            .values(
                status="parsing",
                updated_at=bindparam("b_now"),
            ),
            {
                "revision_id": revision_id,
                "b_revision_id": revision_id,
                "now": now,
                "b_now": now,
            },
        )
        lease.extend()
        connection.execute(
            update(IngestJob)
            .where(IngestJob.id == bindparam("b_ingest_job_id"))
            .values(
                status="ready",
                finished_at=bindparam("b_now"),
                warnings=func.coalesce(IngestJob.warnings, sa_cast([], JSONB)),
            ),
            {
                "ingest_job_id": context.payload.ingest_job_id,
                "b_ingest_job_id": context.payload.ingest_job_id,
                "now": now,
                "b_now": now,
            },
        )
        DeterministicIngestProcessor._activate_revision_if_current(
            self,  # type: ignore[arg-type]
            connection,
            document_id=context.payload.document_id,
            revision_id=revision_id,
            ingest_job_id=context.payload.ingest_job_id,
            now=now,
        )
        connection.execute(
            update(Document)
            .where(Document.id == bindparam("b_document_id"))
            .values(
                current_status="ready",
                updated_at=bindparam("b_now"),
            ),
            {
                "document_id": context.payload.document_id,
                "b_document_id": context.payload.document_id,
                "now": now,
                "b_now": now,
            },
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


def _advisory_lock_keys(value: uuid.UUID) -> tuple[int, int]:
    uuid_int = value.int
    return _to_signed_int32((uuid_int >> 96) & 0xFFFFFFFF), _to_signed_int32(uuid_int & 0xFFFFFFFF)


def _to_signed_int32(value: int) -> int:
    return value - 0x100000000 if value >= 0x80000000 else value


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
