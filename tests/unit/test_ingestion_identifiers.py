from __future__ import annotations

from uuid import uuid4

import pytest

from paper_context.ingestion.identifiers import artifact_id, retrieval_index_run_id

pytestmark = pytest.mark.unit


def test_artifact_id_is_deterministic_for_identical_inputs() -> None:
    ingest_job_id = uuid4()

    assert artifact_id(
        ingest_job_id=ingest_job_id,
        artifact_type="parsed_document",
        parser="docling",
    ) == artifact_id(
        ingest_job_id=ingest_job_id,
        artifact_type="parsed_document",
        parser="docling",
    )


def test_artifact_id_changes_when_identity_inputs_change() -> None:
    ingest_job_id = uuid4()

    assert artifact_id(
        ingest_job_id=ingest_job_id,
        artifact_type="parsed_document",
        parser="docling",
    ) != artifact_id(
        ingest_job_id=ingest_job_id,
        artifact_type="source_pdf",
        parser="docling",
    )
    assert artifact_id(
        ingest_job_id=ingest_job_id,
        artifact_type="parsed_document",
        parser="docling",
    ) != artifact_id(
        ingest_job_id=ingest_job_id,
        artifact_type="parsed_document",
        parser="pdfplumber",
    )


def test_retrieval_index_run_id_is_deterministic_and_varies_by_ingest_job() -> None:
    ingest_job_id = uuid4()

    assert retrieval_index_run_id(ingest_job_id=ingest_job_id) == retrieval_index_run_id(
        ingest_job_id=ingest_job_id
    )
    assert retrieval_index_run_id(ingest_job_id=ingest_job_id) != retrieval_index_run_id(
        ingest_job_id=uuid4()
    )
