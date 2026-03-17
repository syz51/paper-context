# Architecture

> **Decision:** The MVP is a deterministic retrieval system built from FastAPI, FastMCP, Postgres + pgvector, Docling, Voyage `voyage-4-large`, and Zero Entropy `zerank-2`. LlamaIndex and PydanticAI are explicitly outside the MVP core.

This document describes the runtime surfaces, the boundary between ingestion and retrieval responsibilities, and the point where any future agent layer begins. For pipeline details, see [Ingestion and Indexing](./ingestion-and-indexing.md), [Retrieval](./retrieval.md), [Data Model](./data-model.md), and [APIs and Tools](./apis-and-tools.md).

## Runtime surfaces

### `api`

FastAPI service for uploads, document inspection, ingest job status, and operational reads. This surface owns request validation, job creation, and stable HTTP contracts.

### `worker`

Background ingestion and indexing process. It owns parsing, structure gating, metadata enrichment, normalization, contextualized chunk generation, embedding, and index-version writes.

### `mcp`

FastMCP server exposing curated retrieval tools. It reuses the retrieval service layer directly rather than auto-wrapping FastAPI endpoints.

### `skill` (optional)

A thin downstream companion layer that teaches agents how to use the MCP tools well. It is not part of the retrieval core and should not contain canonical business logic.

## Component responsibilities

- `api`
  - accept `POST /documents`
  - create and expose `ingest_jobs`
  - expose document metadata, outline, and table reads
- `worker`
  - parse with Docling, then `pdfplumber` only when needed
  - normalize document structure into canonical entities
  - generate contextualized passage chunks
  - write embeddings and retrieval index metadata
- `mcp`
  - expose search and context-pack tools
  - return provenance, warnings, and stable identifiers
  - keep agent-facing tool contracts small and explicit
- `skill`
  - prefer `build_context_pack` over raw primitive calls
  - encode downstream usage guidance, not retrieval logic

## Canonical request and data flow

1. A client uploads a PDF through `POST /documents`.
2. The API creates a `document` shell and an `ingest_job`, then hands work to the worker.
3. The worker stores the original file as an artifact, runs Docling, evaluates structure quality, and falls back to `pdfplumber` only if the Docling output is not usable.
4. The worker recovers metadata, enriches it synchronously from OpenAlex and Semantic Scholar, and normalizes sections, passages, tables, references, and artifacts into Postgres.
5. The worker creates contextualized chunks, embeds them with Voyage `voyage-4-large`, and records the run under an `index version`.
6. Retrieval queries run through metadata filtering, sparse search, dense search, fusion, reranking with `zerank-2`, parent expansion, and context-pack assembly.
7. FastAPI and FastMCP return the same underlying document, passage, table, and context-pack semantics with provenance and warnings attached.

## Deterministic retrieval boundary

The deterministic core ends at retrieval outputs and context packs. That means:

- ingestion, normalization, indexing, search, reranking, and provenance live in this codebase
- MCP tools expose those deterministic capabilities directly
- any future agent layer consumes retrieval outputs but does not replace retrieval logic

This boundary is deliberate. It keeps the MVP debuggable, testable, and implementation-first.

## Major stack choices

- **Docling + `pdfplumber` fallback**
  - Docling is the primary parser because it preserves paper structure well enough to support sections, passages, and tables.
  - `pdfplumber` is the fallback because it gives a deterministic lower-level recovery path when Docling structure is weak.
- **Postgres + pgvector**
  - One authoritative store for canonical records, filters, provenance, and vector retrieval.
  - Avoids splitting the MVP across multiple persistence systems.
- **Voyage `voyage-4-large`**
  - Default dense retrieval model for contextualized passage and table embeddings.
- **Zero Entropy `zerank-2`**
  - Default reranker for fused sparse+dense candidates before final result selection.
- **FastAPI + FastMCP**
  - FastAPI is the operational surface.
  - FastMCP is the tool surface.
  - Keeping them separate avoids turning the HTTP API into an accidental tool contract.
- **Contextualized chunking + parent-child retrieval**
  - The embedding unit is a section-bounded passage with deterministic context added.
  - Retrieval returns the best child results plus the parent structure needed to interpret them.

## Out of scope

The MVP core does not include LlamaIndex, PydanticAI, autonomous answer generation, citation-graph traversal, or a memory layer. Those can be layered on later, but they must consume the deterministic retrieval contracts defined here rather than replace them.
