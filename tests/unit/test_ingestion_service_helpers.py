from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from paper_context.ingestion.service import (
    DeterministicIngestProcessor,
    IngestJobContext,
    SyntheticIngestProcessor,
    _chunk_paragraphs,
    _contextualize_text,
    _dedupe_warnings,
    _json_array,
    _token_count,
)
from paper_context.ingestion.types import (
    EnrichmentResult,
    GateStatus,
    ParsedDocument,
    ParsedParagraph,
    ParsedReference,
    ParsedSection,
    ParsedTable,
    ParserArtifact,
    ParserResult,
)
from paper_context.queue.contracts import IngestQueuePayload
from paper_context.queue.pgmq import PgmqMessage

pytestmark = pytest.mark.unit


def _make_message(payload: Mapping[str, object]) -> PgmqMessage:
    return PgmqMessage(
        msg_id=1,
        read_ct=0,
        enqueued_at=datetime.now(UTC),
        vt=datetime.now(UTC),
        message=dict(payload),
    )


def _make_context() -> IngestJobContext:
    ingest_job_id = uuid4()
    document_id = uuid4()
    return IngestJobContext(
        message=_make_message(
            {"ingest_job_id": str(ingest_job_id), "document_id": str(document_id)}
        ),
        payload=IngestQueuePayload(ingest_job_id=ingest_job_id, document_id=document_id),
    )


def _make_paragraph(
    text: str,
    *,
    page_start: int,
    page_end: int,
    charspan_start: int,
    charspan_end: int,
) -> ParsedParagraph:
    return ParsedParagraph(
        text=text,
        page_start=page_start,
        page_end=page_end,
        provenance_offsets={"pages": [page_start], "charspans": [[charspan_start, charspan_end]]},
    )


def _make_document(
    *,
    metadata_confidence: float = 0.4,
    sections: list[ParsedSection] | None = None,
    tables: list[ParsedTable] | None = None,
    references: list[ParsedReference] | None = None,
) -> ParsedDocument:
    return ParsedDocument(
        title="Phase 1 paper",
        authors=["Ada Lovelace"],
        abstract="A deterministic ingestion test document.",
        publication_year=2024,
        metadata_confidence=metadata_confidence,
        sections=sections
        if sections is not None
        else [
            ParsedSection(
                key="s1",
                heading="Introduction",
                heading_path=["Introduction"],
                level=1,
                page_start=1,
                page_end=1,
                paragraphs=[
                    _make_paragraph(
                        "This is a long enough paragraph to create at least one chunk of content.",
                        page_start=1,
                        page_end=1,
                        charspan_start=0,
                        charspan_end=72,
                    )
                ],
            )
        ],
        tables=tables or [],
        references=references or [],
    )


def _make_parser_result(
    *,
    gate_status: GateStatus = "pass",
    parser_name: str = "docling",
    parsed_document: ParsedDocument | None = None,
) -> ParserResult:
    return ParserResult(
        gate_status=gate_status,
        parsed_document=parsed_document if parsed_document is not None else _make_document(),
        artifact=ParserArtifact(
            artifact_type=f"{parser_name}_parse",
            parser=parser_name,
            filename=f"{parser_name}.json",
            content=b"{}",
        ),
        warnings=["reduced_structure_confidence"] if gate_status == "degraded" else [],
        failure_code=f"{parser_name}_failed" if gate_status == "fail" else None,
        failure_message=f"{parser_name} failed" if gate_status == "fail" else None,
    )


def _make_processor() -> tuple[
    DeterministicIngestProcessor,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    storage = MagicMock()
    resolved_path = MagicMock(spec=Path)
    resolved_path.read_bytes.return_value = b"%PDF-1.4"
    storage.resolve.return_value = resolved_path
    storage.store_bytes.return_value = SimpleNamespace(
        storage_ref="stored://artifact",
        checksum="abc123",
    )

    primary_parser = MagicMock()
    fallback_parser = MagicMock()
    metadata_enricher = MagicMock()
    metadata_enricher.enrich.return_value = EnrichmentResult()

    processor = DeterministicIngestProcessor(
        storage=storage,
        primary_parser=primary_parser,
        fallback_parser=fallback_parser,
        metadata_enricher=metadata_enricher,
        index_version="index-v1",
        chunking_version="chunk-v1",
        embedding_model="emb-v1",
        reranker_model="rank-v1",
        min_tokens=3,
        max_tokens=6,
        overlap_fraction=0.5,
    )
    return processor, storage, primary_parser, fallback_parser, metadata_enricher


def _result_with_row(row: dict[str, object] | None) -> MagicMock:
    result = MagicMock()
    result.mappings.return_value.one_or_none.return_value = row
    return result


def _row_sequence(rows: list[dict[str, object] | None]) -> Callable[..., MagicMock]:
    iterator = iter(rows)

    def _execute(*args, **kwargs):
        try:
            row = next(iterator)
        except StopIteration:
            return MagicMock()
        return _result_with_row(row)

    return _execute


def test_small_helpers_cover_json_array_token_count_dedupe_and_context() -> None:
    assert _json_array({"status": "queued", "warnings": ["a", "b"]}) == (
        '{"status": "queued", "warnings": ["a", "b"]}'
    )
    assert _dedupe_warnings(["a", "a", "b", "a"]) == ["a", "b"]
    assert _token_count("one two three") == 3
    assert _contextualize_text("Paper", ["Intro", "Methods"], "body") == (
        "Document title: Paper\nSection path: Intro > Methods\nLocal heading context: Intro\n\nbody"
    )


def test_chunk_paragraphs_handles_empty_input_and_overlap() -> None:
    assert _chunk_paragraphs([], min_tokens=1, max_tokens=3, overlap_fraction=0.5) == []

    paragraphs = [
        _make_paragraph(
            "one two three",
            page_start=1,
            page_end=1,
            charspan_start=0,
            charspan_end=13,
        ),
        _make_paragraph(
            "four five six",
            page_start=1,
            page_end=1,
            charspan_start=14,
            charspan_end=27,
        ),
        _make_paragraph(
            "seven eight nine",
            page_start=2,
            page_end=2,
            charspan_start=28,
            charspan_end=44,
        ),
    ]

    chunks = _chunk_paragraphs(paragraphs, min_tokens=3, max_tokens=6, overlap_fraction=0.5)

    assert [[paragraph.text for paragraph in chunk] for chunk in chunks] == [
        ["one two three", "four five six"],
        ["four five six", "seven eight nine"],
    ]


def test_lock_and_load_helpers_return_rows_and_none() -> None:
    processor, _, _, _, _ = _make_processor()
    connection = MagicMock()
    connection.execute.side_effect = _row_sequence(
        [
            {"status": "queued", "warnings": ["parser_fallback_used"]},
            {"storage_ref": "documents/source.pdf"},
            None,
            None,
        ]
    )

    ingest_job_id = uuid4()
    document_id = uuid4()

    assert processor._lock_ingest_job(connection, ingest_job_id) == {
        "status": "queued",
        "warnings": ["parser_fallback_used"],
    }
    assert processor._load_source_artifact(connection, document_id) == {
        "storage_ref": "documents/source.pdf",
    }
    assert processor._lock_ingest_job(connection, ingest_job_id) is None
    assert processor._load_source_artifact(connection, document_id) is None


def test_reset_document_state_executes_all_cleanup_statements() -> None:
    processor, _, _, _, _ = _make_processor()
    connection = MagicMock()
    document_id = uuid4()

    processor._reset_document_state(connection, document_id)

    assert connection.execute.call_count == 7
    for call in connection.execute.call_args_list:
        assert call.args[1] == {"document_id": document_id}


def test_persist_parser_artifact_stores_bytes_and_records_row() -> None:
    processor, storage, _, _, _ = _make_processor()
    connection = MagicMock()
    document_id = uuid4()
    ingest_job_id = uuid4()
    artifact = _make_parser_result()

    artifact_id = processor._persist_parser_artifact(
        connection,
        document_id=document_id,
        ingest_job_id=ingest_job_id,
        artifact=artifact,
        is_primary=True,
    )

    assert isinstance(artifact_id, UUID)
    storage.store_bytes.assert_called_once_with(
        f"documents/{document_id}/{ingest_job_id}/docling.json",
        b"{}",
    )
    params = connection.execute.call_args.args[1]
    assert params["document_id"] == document_id
    assert params["artifact_type"] == "docling_parse"
    assert params["parser"] == "docling"
    assert params["storage_ref"] == "stored://artifact"
    assert params["checksum"] == "abc123"
    assert params["is_primary"] is True


def test_normalize_document_persists_hierarchy_tables_and_references() -> None:
    processor, _, _, _, _ = _make_processor()
    connection = MagicMock()
    document_id = uuid4()
    artifact_id = uuid4()
    root_section_id = uuid4()
    child_section_id = uuid4()
    table_row_id = uuid4()
    reference_row_id = uuid4()

    document = ParsedDocument(
        title="Phase 1 paper",
        authors=[],
        abstract=None,
        publication_year=2024,
        metadata_confidence=0.9,
        sections=[
            ParsedSection(
                key="root",
                heading="Introduction",
                heading_path=["Introduction"],
                level=1,
                page_start=1,
                page_end=1,
            ),
            ParsedSection(
                key="child",
                heading="Method",
                heading_path=["Introduction", "Method"],
                level=2,
                page_start=2,
                page_end=2,
                parent_key="root",
            ),
        ],
        tables=[
            ParsedTable(
                section_key="child",
                caption="Table 1",
                headers=["a", "b"],
                rows=[["1", "2"]],
                page_start=2,
                page_end=2,
            )
        ],
        references=[
            ParsedReference(
                raw_citation="Smith 2024",
                publication_year=2024,
            )
        ],
    )

    with patch(
        "paper_context.ingestion.service.uuid.uuid4",
        side_effect=[root_section_id, child_section_id, table_row_id, reference_row_id],
    ):
        section_ids = processor._normalize_document(
            connection,
            document_id=document_id,
            parsed_document=document,
            artifact_id=artifact_id,
        )

    assert section_ids == {"root": root_section_id, "child": child_section_id}
    assert connection.execute.call_count == 4
    first_params = connection.execute.call_args_list[0].args[1]
    second_params = connection.execute.call_args_list[1].args[1]
    table_params = connection.execute.call_args_list[2].args[1]
    reference_params = connection.execute.call_args_list[3].args[1]

    assert first_params["parent_section_id"] is None
    assert second_params["parent_section_id"] == root_section_id
    assert table_params["section_id"] == child_section_id
    assert table_params["caption"] == "Table 1"
    assert reference_params["raw_citation"] == "Smith 2024"
    assert reference_params["publication_year"] == 2024


def test_normalize_document_inserts_root_section_when_sections_missing() -> None:
    processor, _, _, _, _ = _make_processor()
    connection = MagicMock()
    document_id = uuid4()
    artifact_id = uuid4()
    root_section_id = uuid4()

    with patch("paper_context.ingestion.service.uuid.uuid4", side_effect=[root_section_id]):
        section_ids = processor._normalize_document(
            connection,
            document_id=document_id,
            parsed_document=_make_document(sections=[], tables=[], references=[]),
            artifact_id=artifact_id,
        )

    assert section_ids == {"section-root": root_section_id}
    assert connection.execute.call_count == 1
    assert connection.execute.call_args.args[1]["heading"] is None


@pytest.mark.parametrize(
    (
        "enrichment",
        "expected_title",
        "expected_authors",
        "expected_abstract",
        "expected_year",
        "expected_warnings",
    ),
    [
        pytest.param(
            EnrichmentResult(
                title="Enriched title",
                authors=["Grace Hopper"],
                abstract="Better abstract",
                publication_year=2025,
                metadata_confidence=0.8,
            ),
            "Enriched title",
            ["Grace Hopper"],
            "Better abstract",
            2025,
            ["existing"],
            id="preferred",
        ),
        pytest.param(
            EnrichmentResult(
                title="Conflicting title",
                authors=["Grace Hopper"],
                abstract="Conflicting abstract",
                publication_year=2023,
                metadata_confidence=0.1,
            ),
            "Parsed title",
            ["Ada Lovelace"],
            "Parsed abstract",
            2024,
            ["existing", "metadata_conflict_ignored"],
            id="conflict",
        ),
    ],
)
def test_apply_enriched_document_metadata_prefers_higher_confidence_and_flags_lower_conflicts(
    enrichment: EnrichmentResult,
    expected_title: str | None,
    expected_authors: list[str],
    expected_abstract: str | None,
    expected_year: int | None,
    expected_warnings: list[str],
) -> None:
    processor, _, _, _, _ = _make_processor()
    connection = MagicMock()
    parsed_document = ParsedDocument(
        title="Parsed title",
        authors=["Ada Lovelace"],
        abstract="Parsed abstract",
        publication_year=2024,
        metadata_confidence=0.4,
        sections=[],
        tables=[],
        references=[],
    )
    warnings = ["existing"]

    with patch.object(processor, "_apply_document_metadata") as apply_document_metadata:
        document_id = uuid4()
        processor._apply_enriched_document_metadata(
            connection,
            document_id=document_id,
            parsed_document=parsed_document,
            enrichment=enrichment,
            warnings=warnings,
        )

    apply_document_metadata.assert_called_once_with(
        connection,
        document_id=document_id,
        title=expected_title,
        authors=expected_authors,
        abstract=expected_abstract,
        publication_year=expected_year,
        metadata_confidence=enrichment.metadata_confidence
        if enrichment.metadata_confidence is not None
        and enrichment.metadata_confidence >= (parsed_document.metadata_confidence or 0.0)
        else parsed_document.metadata_confidence,
    )
    assert warnings == expected_warnings


def test_process_success_path_uses_real_helpers_and_marks_low_confidence_warning() -> None:
    processor, storage, primary_parser, fallback_parser, metadata_enricher = _make_processor()
    connection = MagicMock()
    context = _make_context()
    parsed_document = _make_document(
        metadata_confidence=0.4,
        sections=[
            ParsedSection(
                key="s1",
                heading="Introduction",
                heading_path=["Introduction", "Methods"],
                level=1,
                page_start=1,
                page_end=2,
                paragraphs=[
                    _make_paragraph(
                        "one two three",
                        page_start=1,
                        page_end=1,
                        charspan_start=0,
                        charspan_end=13,
                    ),
                    _make_paragraph(
                        "four five six",
                        page_start=1,
                        page_end=1,
                        charspan_start=14,
                        charspan_end=27,
                    ),
                ],
            )
        ],
    )
    primary_parser.parse.return_value = _make_parser_result(parsed_document=parsed_document)

    connection.execute.side_effect = _row_sequence(
        [
            {"status": "queued", "warnings": ["parser_fallback_used"]},
            {"storage_ref": "documents/test/source.pdf"},
        ]
    )

    lease = MagicMock()
    processor.process(connection, context, lease)

    primary_parser.parse.assert_called_once_with(
        filename="documents/test/source.pdf",
        content=b"%PDF-1.4",
    )
    fallback_parser.parse.assert_not_called()
    metadata_enricher.enrich.assert_called_once_with(parsed_document)
    storage.store_bytes.assert_called_once()
    assert lease.extend.call_count == 4
    assert connection.execute.call_count >= 16


def test_process_missing_job_raises_lookup_error() -> None:
    processor, _, _, _, _ = _make_processor()
    connection = MagicMock()
    connection.execute.side_effect = _row_sequence([None])
    lease = MagicMock()

    with pytest.raises(LookupError, match="missing ingest job"):
        processor.process(connection, _make_context(), lease)


def test_process_missing_source_artifact_marks_failure() -> None:
    processor, _, primary_parser, fallback_parser, _ = _make_processor()
    connection = MagicMock()
    connection.execute.side_effect = _row_sequence(
        [
            {"status": "queued", "warnings": []},
            None,
        ]
    )
    lease = MagicMock()
    processor._mark_failed = MagicMock()  # type: ignore[method-assign]

    context = _make_context()
    processor.process(connection, context, lease)

    processor._mark_failed.assert_called_once_with(
        connection,
        ingest_job_id=context.payload.ingest_job_id,
        document_id=context.payload.document_id,
        failure_code="missing_source_artifact",
        failure_message="No source PDF artifact exists for this ingest job.",
        warnings=[],
    )
    primary_parser.parse.assert_not_called()
    fallback_parser.parse.assert_not_called()


def test_process_degraded_primary_with_failing_fallback_marks_failure() -> None:
    processor, _, primary_parser, fallback_parser, _ = _make_processor()
    connection = MagicMock()
    connection.execute.side_effect = _row_sequence(
        [
            {"status": "queued", "warnings": []},
            {"storage_ref": "documents/test/source.pdf"},
        ]
    )
    lease = MagicMock()
    processor._mark_failed = MagicMock()  # type: ignore[method-assign]

    primary_result = _make_parser_result(gate_status="degraded", parser_name="docling")
    fallback_result = _make_parser_result(gate_status="fail", parser_name="pdfplumber")
    primary_parser.parse.return_value = primary_result
    fallback_parser.parse.return_value = fallback_result

    context = _make_context()
    processor.process(connection, context, lease)

    processor._mark_failed.assert_called_once_with(
        connection,
        ingest_job_id=context.payload.ingest_job_id,
        document_id=context.payload.document_id,
        failure_code="pdfplumber_failed",
        failure_message="pdfplumber failed",
        warnings=["reduced_structure_confidence", "parser_fallback_used"],
    )
    primary_parser.parse.assert_called_once()
    fallback_parser.parse.assert_called_once()
    assert lease.extend.call_count == 2


def test_insert_passages_chunks_contextualizes_and_merges_provenance() -> None:
    processor, _, _, _, _ = _make_processor()
    connection = MagicMock()
    document_id = uuid4()
    artifact_id = uuid4()
    section_id = uuid4()
    parsed_document = ParsedDocument(
        title="Phase 1 paper",
        authors=[],
        abstract=None,
        publication_year=2024,
        metadata_confidence=0.8,
        sections=[
            ParsedSection(
                key="s1",
                heading="Introduction",
                heading_path=["Introduction", "Methods"],
                level=1,
                page_start=1,
                page_end=2,
                paragraphs=[
                    _make_paragraph(
                        "one two three",
                        page_start=1,
                        page_end=1,
                        charspan_start=0,
                        charspan_end=13,
                    ),
                    _make_paragraph(
                        "four five six",
                        page_start=1,
                        page_end=1,
                        charspan_start=14,
                        charspan_end=27,
                    ),
                    _make_paragraph(
                        "seven eight nine",
                        page_start=2,
                        page_end=2,
                        charspan_start=28,
                        charspan_end=44,
                    ),
                ],
            )
        ],
        tables=[],
        references=[],
    )

    processor._insert_passages(
        connection,
        document_id=document_id,
        parsed_document=parsed_document,
        section_ids={"s1": section_id},
        artifact_id=artifact_id,
    )

    assert connection.execute.call_count == 2
    first_params = connection.execute.call_args_list[0].args[1]
    second_params = connection.execute.call_args_list[1].args[1]
    assert first_params["section_id"] == section_id
    assert first_params["chunk_ordinal"] == 1
    assert first_params["body_text"] == "one two three\n\nfour five six"
    assert json.loads(first_params["provenance_offsets"]) == {
        "pages": [1, 1],
        "charspans": [[0, 13], [14, 27]],
    }
    assert "Document title: Phase 1 paper" in first_params["contextualized_text"]
    assert "Section path: Introduction > Methods" in first_params["contextualized_text"]
    assert "Local heading context: Introduction" in first_params["contextualized_text"]
    assert second_params["chunk_ordinal"] == 2
    assert second_params["page_start"] == 1
    assert second_params["page_end"] == 2


def test_mark_helpers_issue_expected_updates() -> None:
    processor, _, _, _, _ = _make_processor()
    connection = MagicMock()
    ingest_job_id = uuid4()
    document_id = uuid4()

    processor._mark_stage(
        connection,
        ingest_job_id=ingest_job_id,
        document_id=document_id,
        status="parsing",
        warnings=["a", "a"],
    )
    processor._mark_failed(
        connection,
        ingest_job_id=ingest_job_id,
        document_id=document_id,
        failure_code="parse_failed",
        failure_message="Parser failed",
        warnings=["existing", "existing", "new"],
    )
    processor._mark_ready(
        connection,
        ingest_job_id=ingest_job_id,
        document_id=document_id,
        warnings=["ready", "ready"],
    )

    assert connection.execute.call_count == 6
    stage_params = connection.execute.call_args_list[0].args[1]
    failed_params = connection.execute.call_args_list[2].args[1]
    ready_params = connection.execute.call_args_list[4].args[1]

    assert stage_params["status"] == "parsing"
    assert json.loads(stage_params["warnings"]) == ["a", "a"]
    assert failed_params["failure_code"] == "parse_failed"
    assert failed_params["failure_message"] == "Parser failed"
    assert json.loads(failed_params["warnings"]) == ["existing", "new"]
    assert json.loads(ready_params["warnings"]) == ["ready"]


def test_synthetic_processor_updates_non_terminal_job() -> None:
    processor = SyntheticIngestProcessor()
    connection = MagicMock()
    select_result = MagicMock()
    select_result.mappings.return_value.one_or_none.return_value = {"status": "queued"}
    connection.execute.side_effect = [
        select_result,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]
    lease = MagicMock()
    context = _make_context()

    processor.process(connection, context, lease)

    assert connection.execute.call_count == 4
    lease.extend.assert_called_once()
