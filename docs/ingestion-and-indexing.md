# Ingestion and Indexing

> **Decision:** Ingestion is Docling-first, `pdfplumber`-fallback, and index-versioned. The worker produces canonical document structure first, then derived retrieval data. It does not depend on LlamaIndex or PydanticAI.

This document defines the upload path, parser flow, structure gate, metadata enrichment, normalization steps, contextualized chunk generation, and index-version rules.

## Upload flow and ingest job lifecycle

`POST /documents` accepts a multipart PDF upload and creates:

- a `documents` row with initial status
- an `ingest_jobs` row in `queued`
- a `document_artifacts` entry for the original PDF

The worker advances the job through these states:

1. `queued`
2. `parsing`
3. `normalizing`
4. `enriching_metadata`
5. `chunking`
6. `indexing`
7. `ready` or `failed`

`failed` is terminal for the current job run. Retries create a new job row linked to the same document.

## Docling-first parse path

The worker runs Docling in born-digital mode with OCR disabled.

Expected outputs from the primary path:

- page-level text coverage
- heading and section structure
- table detections
- page spans for sections, passages, and tables

The raw Docling output is stored in `document_artifacts` and marked as the primary parse artifact when it passes the structure gate.

## Structure-quality gate

The gate runs after Docling parse and before chunk generation. It classifies the parse as `pass`, `degraded`, or `fail`.

### `pass`

All of the following are true:

- most text-bearing pages are mapped into ordered sections
- page ordering is stable
- passage spans are recoverable
- any detected tables have page provenance

### `degraded`

The text layer is usable, but at least one structural signal is weak:

- headings are incomplete or noisy
- section boundaries are unreliable on some pages
- tables are partially recoverable but not well structured

`degraded` triggers the `pdfplumber` fallback path instead of indexing the Docling result directly.

### `fail`

The PDF is not usable for MVP ingestion:

- the text layer is missing or too sparse
- page order or spans are unreliable after fallback
- the parser cannot produce stable passages with provenance

`fail` stops the job with an explicit reason. The system does not degrade into blob-text indexing.

## `pdfplumber` fallback path

When the Docling output is `degraded`, the worker retries extraction with `pdfplumber`.

Fallback behavior:

- create a new `document_artifacts` row with `parser = pdfplumber`
- preserve the original Docling artifact for debugging
- recover text spans, page spans, and simple table candidates where possible
- continue normalization only if the fallback result reaches `pass`

If fallback succeeds, the document is indexed with warnings:

- `parser_fallback_used`
- `reduced_structure_confidence`
- optional `table_structure_partial`

Those warnings must propagate into retrieval outputs and context packs.

## Metadata recovery and synchronous enrichment

Base metadata is recovered from parser output first:

- title
- authors
- abstract
- publication year
- reference-section presence

Each field stores a confidence level and a source.

The worker then performs synchronous enrichment against OpenAlex and Semantic Scholar to fill or validate:

- author identities
- venue
- DOI
- citation metadata

Enrichment does not block the document from becoming searchable unless it causes a hard data-shape error. If enrichment fails, the job continues with warnings and partial metadata.

## Normalization

The worker writes canonical record data in this order:

1. `documents`
2. `document_sections`
3. `document_tables`
4. `document_passages`
5. `document_references`
6. `document_artifacts`

Normalization rules:

- a **document** is the top-level paper record
- a **section** is a heading-bounded structural unit
- a **table** is a first-class extracted object, not serialized into passage text
- a **passage** is a section-bounded chunkable prose span
- a **reference** is a cited-work record extracted from the paper

All normalized rows must keep page provenance and a link back to the parse artifact that produced them.

## Contextualized chunk generation

Chunking runs after normalization and only on passage text.

Defaults:

- stay within section boundaries
- target 300-700 tokens
- use 10-15% overlap
- store both the raw passage text and the contextualized embedding text

The contextualized text is deterministic. It prepends a short context string derived from:

- document title
- section heading path
- immediate local heading context when available

The raw passage text remains the canonical record. The contextualized text is derived index input used for embeddings and retrieval.

## Index-version creation and activation rules

Every indexing run writes a `retrieval_index_runs` row that records:

- `index_version`
- embedding provider and model
- reranker provider and model
- chunking policy version
- parser provenance
- status

Rules:

- the active index version is corpus-scoped for the workspace
- new documents are indexed under the current active version by default
- any model or chunking-policy change creates a new index version
- a new version becomes active only after its build completes successfully
- retrieval must never mix embeddings from two index versions in one result set

## Failure modes

- **parse failure**
  - Docling and fallback cannot produce stable passages with provenance
- **metadata enrichment failure**
  - OpenAlex or Semantic Scholar is unavailable or returns incompatible matches
- **normalization failure**
  - parser output cannot be mapped into canonical entities cleanly
- **indexing failure**
  - embedding, vector write, or index-run bookkeeping fails

Each failure must leave the job in `failed`, preserve the artifacts that were produced, and expose a machine-readable failure code plus a human-readable message.

## Non-goals

- OCR and scanned PDFs
- multimodal parsing
- free-text fallback indexing with no structure
- LlamaIndex or PydanticAI abstractions in the ingestion path
