from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert, text

from paper_context.mcp import server as mcp_module
from paper_context.models import (
    Document,
    DocumentPassage,
    DocumentSection,
    IngestJob,
    RetrievalIndexRun,
)
from paper_context.retrieval.clients import EmbeddingBatch
from paper_context.retrieval.types import RerankItem

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_postgres,
    pytest.mark.slow,
]

VECTOR_DIMENSIONS = 1024


def _vector_values(index: int) -> list[float]:
    vector = [0.0] * VECTOR_DIMENSIONS
    vector[index] = 1.0
    return vector


def _vector_string(index: int) -> str:
    return (
        "["
        + ",".join("1.0" if position == index else "0.0" for position in range(VECTOR_DIMENSIONS))
        + "]"
    )


class _FixedEmbeddingClient:
    provider = "fake"

    def __init__(self, *, model: str) -> None:
        self.model = model

    def embed(self, texts: list[str], *, input_type: str) -> EmbeddingBatch:
        del input_type
        embeddings = tuple(tuple(_vector_values(0)) for _ in texts)
        return EmbeddingBatch(
            provider=self.provider,
            model=self.model,
            dimensions=VECTOR_DIMENSIONS,
            embeddings=embeddings,
        )


class _IdentityRerankerClient:
    provider = "fake"

    def __init__(self, *, model: str) -> None:
        self.model = model

    def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankItem]:
        del query
        indexes = list(range(len(documents)))
        if top_n is not None:
            indexes = indexes[:top_n]
        return [
            RerankItem(index=index, score=float(len(indexes) - position))
            for position, index in enumerate(indexes)
        ]


class _UnusedDocumentsService:
    def search_documents(self, **kwargs):
        del kwargs
        raise AssertionError("search_documents should not be called in this test")

    def get_document_outline(self, **kwargs):
        del kwargs
        raise AssertionError("get_document_outline should not be called in this test")


def _sse_json(response_text: str) -> dict[str, Any]:
    for line in response_text.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"missing SSE data line in response: {response_text!r}")


def _insert_ready_document(connection, *, document_id, revision_id, now: datetime) -> None:
    connection.execute(
        insert(Document).values(
            id=document_id,
            title="MCP retrieval paper",
            source_type="upload",
            current_status="ready",
            active_revision_id=None,
            created_at=now,
            updated_at=now,
        )
    )
    connection.execute(
        text(
            """
            INSERT INTO document_revisions (
                id,
                document_id,
                revision_number,
                status,
                title,
                authors,
                abstract,
                publication_year,
                source_type,
                metadata_confidence,
                quant_tags,
                source_artifact_id,
                ingest_job_id,
                activated_at,
                superseded_at,
                created_at,
                updated_at
            ) VALUES (
                :id,
                :document_id,
                1,
                'ready',
                :title,
                '[]'::jsonb,
                NULL,
                NULL,
                'upload',
                NULL,
                '{}'::jsonb,
                NULL,
                NULL,
                :now,
                NULL,
                :now,
                :now
            )
            """
        ),
        {
            "id": revision_id,
            "document_id": document_id,
            "title": "MCP retrieval paper",
            "now": now,
        },
    )
    connection.execute(
        text(
            """
            UPDATE documents
            SET active_revision_id = :revision_id
            WHERE id = :document_id
            """
        ),
        {"revision_id": revision_id, "document_id": document_id},
    )


def test_mcp_search_passages_uses_db_active_index_version_when_app_config_is_stale(
    migrated_postgres_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    document_id = uuid4()
    revision_id = uuid4()
    section_id = uuid4()
    passage_id = uuid4()
    ingest_job_id = uuid4()
    run_id = uuid4()

    with migrated_postgres_engine.begin() as connection:
        _insert_ready_document(
            connection,
            document_id=document_id,
            revision_id=revision_id,
            now=now,
        )
        connection.execute(
            insert(IngestJob).values(
                id=ingest_job_id,
                document_id=document_id,
                revision_id=revision_id,
                status="ready",
                trigger="upload",
                warnings=[],
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            insert(DocumentSection).values(
                id=section_id,
                document_id=document_id,
                revision_id=revision_id,
                heading="Methods",
                heading_path=["Methods"],
                ordinal=1,
                page_start=1,
                page_end=1,
            )
        )
        connection.execute(
            insert(DocumentPassage).values(
                id=passage_id,
                document_id=document_id,
                revision_id=revision_id,
                section_id=section_id,
                chunk_ordinal=1,
                body_text="Fresh DB-active MCP passage.",
                contextualized_text="Fresh DB-active MCP passage.",
                token_count=5,
                page_start=1,
                page_end=1,
                provenance_offsets={"pages": [1], "charspans": [[0, 27]]},
                artifact_id=None,
            )
        )
        connection.execute(
            insert(RetrievalIndexRun).values(
                id=run_id,
                document_id=document_id,
                revision_id=revision_id,
                ingest_job_id=ingest_job_id,
                index_version="mvp-v2",
                embedding_provider="fake",
                embedding_model="fixed-embedding",
                embedding_dimensions=VECTOR_DIMENSIONS,
                reranker_provider="fake",
                reranker_model="identity",
                chunking_version="phase2",
                parser_source="docling",
                status="ready",
                is_active=True,
                activated_at=now,
                deactivated_at=None,
                created_at=now,
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO retrieval_passage_assets (
                    id,
                    retrieval_index_run_id,
                    revision_id,
                    passage_id,
                    document_id,
                    section_id,
                    publication_year,
                    search_text,
                    search_tsvector,
                    embedding
                )
                VALUES (
                    :id,
                    :retrieval_index_run_id,
                    :revision_id,
                    :passage_id,
                    :document_id,
                    :section_id,
                    NULL,
                    :search_text,
                    to_tsvector('english', :search_text),
                    CAST(:embedding AS vector)
                )
                """
            ),
            {
                "id": uuid4(),
                "retrieval_index_run_id": run_id,
                "revision_id": revision_id,
                "passage_id": passage_id,
                "document_id": document_id,
                "section_id": section_id,
                "search_text": "fresh db active keyword",
                "embedding": _vector_string(0),
            },
        )

    settings = SimpleNamespace(
        providers=SimpleNamespace(
            voyage_api_key=None,
            zero_entropy_api_key=None,
            voyage_model="voyage-4-large",
            reranker_model="zerank-2",
            index_version="stale-v1",
        )
    )
    monkeypatch.setattr(mcp_module, "get_settings", lambda: settings)
    monkeypatch.setattr(mcp_module, "get_engine", lambda: migrated_postgres_engine)
    monkeypatch.setattr(mcp_module, "DeterministicEmbeddingClient", _FixedEmbeddingClient)
    monkeypatch.setattr(mcp_module, "HeuristicRerankerClient", _IdentityRerankerClient)

    app = mcp_module.create_http_app(documents_service=cast(Any, _UnusedDocumentsService()))
    headers = {
        "accept": "application/json, text/event-stream",
        "content-type": "application/json",
    }

    with TestClient(app) as client:
        initialize = client.post(
            "/",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "integration-test", "version": "0"},
                },
            },
        )
        session_id = initialize.headers.get("mcp-session-id")
        response = client.post(
            "/",
            headers={**headers, "mcp-session-id": session_id or ""},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "search_passages",
                    "arguments": {"query": "fresh db active keyword"},
                },
            },
        )

    payload = _sse_json(response.text)

    assert initialize.status_code == 200
    assert response.status_code == 200
    assert payload["result"]["isError"] is False
    assert payload["result"]["structuredContent"]["passages"][0]["index_version"] == "mvp-v2"
    assert payload["result"]["structuredContent"]["passages"][0]["retrieval_index_run_id"] == str(
        run_id
    )
