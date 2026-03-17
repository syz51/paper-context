# Data Model

> **Default:** Postgres + pgvector is the authoritative store for canonical paper records, provenance, filters, and retrieval index metadata. Canonical record data and derived index data are kept separate.

This document describes the canonical entities, key fields, relationships, provenance requirements, quant tags, and index-version rules used by the MVP.

## Canonical entities

### `documents`

Top-level paper record.

Key fields:

- `id`
- `title`
- `authors`
- `abstract`
- `publication_year`
- `source_type`
- `metadata_confidence`
- `quant_tags`
- `current_status`
- `created_at`
- `updated_at`

### `document_sections`

Normalized heading-bounded structure within a document.

Key fields:

- `id`
- `document_id`
- `parent_section_id`
- `heading`
- `heading_path`
- `ordinal`
- `page_start`
- `page_end`
- `artifact_id`

### `document_passages`

Canonical prose spans and derived retrieval inputs.

Key fields:

- `id`
- `document_id`
- `section_id`
- `chunk_ordinal`
- `body_text`
- `contextualized_text`
- `token_count`
- `page_start`
- `page_end`
- `provenance_offsets`
- `quant_tags`
- `artifact_id`

`body_text` is canonical record data. `contextualized_text` is derived retrieval input.

### `document_tables`

First-class extracted tables.

Key fields:

- `id`
- `document_id`
- `section_id`
- `caption`
- `table_type`
- `headers_json`
- `rows_json`
- `page_start`
- `page_end`
- `quant_tags`
- `artifact_id`

### `document_references`

Extracted cited-work records from the document.

Key fields:

- `id`
- `document_id`
- `raw_citation`
- `normalized_title`
- `authors`
- `publication_year`
- `doi`
- `source_confidence`
- `artifact_id`

### `document_artifacts`

Stored source and parse artifacts.

Key fields:

- `id`
- `document_id`
- `artifact_type`
- `parser`
- `storage_ref`
- `checksum`
- `is_primary`
- `created_at`

Artifact types include the original PDF, Docling output, and fallback parser output.

### `ingest_jobs`

Worker-tracked ingestion state.

Key fields:

- `id`
- `document_id`
- `status`
- `failure_code`
- `failure_message`
- `warnings`
- `started_at`
- `finished_at`
- `trigger`

### `retrieval_index_runs`

Derived indexing metadata for a document under a specific index version.

Key fields:

- `id`
- `document_id`
- `ingest_job_id`
- `index_version`
- `embedding_provider`
- `embedding_model`
- `embedding_dimensions`
- `reranker_provider`
- `reranker_model`
- `chunking_version`
- `parser_source`
- `status`
- `created_at`

## Relationships and provenance links

Required relationships:

- `documents.id` -> `document_sections.document_id`
- `documents.id` -> `document_passages.document_id`
- `documents.id` -> `document_tables.document_id`
- `documents.id` -> `document_references.document_id`
- `documents.id` -> `document_artifacts.document_id`
- `documents.id` -> `ingest_jobs.document_id`
- `documents.id` -> `retrieval_index_runs.document_id`
- `document_sections.id` -> `document_passages.section_id`
- `document_sections.id` -> `document_tables.section_id`
- `document_artifacts.id` -> normalized rows via `artifact_id`
- `ingest_jobs.id` -> `retrieval_index_runs.ingest_job_id`

Required provenance rule:

Every section, passage, table, and reference must link back to the `document_artifacts` row that produced it. Every retrieval result must link back to the `retrieval_index_runs` row that indexed it.

## Index-versioning rules

`index_version` identifies a retrieval-compatible combination of:

- parser policy
- chunking policy
- embedding provider and model
- reranker provider and model

Rules:

- a document can have many `retrieval_index_runs`
- only one index version is active for live retrieval at a time
- all results in one response must come from the same active index version
- model changes or chunking changes require a new index version
- canonical record rows remain stable across reindexing where possible; derived index rows change freely

## Quant-specific tags

Quant-specific tags live on:

- `documents.quant_tags`
- `document_passages.quant_tags`
- `document_tables.quant_tags`

Standard keys:

- `asset_universe`
- `exchange_or_market`
- `market_type`
- `sampling_frequency`
- `holding_period`
- `sample_period`
- `transaction_cost_model`
- `baseline_type`
- `metric_type`
- `data_source_mentions`
- `implementation_cues`

Document-level tags capture paper-wide signals. Passage and table tags capture local signals used in filtering and ranking.

## Canonical record data vs derived index data

Canonical record data:

- `documents`
- `document_sections`
- `document_passages.body_text`
- `document_tables`
- `document_references`
- `document_artifacts`
- `ingest_jobs`

Derived or index data:

- `document_passages.contextualized_text`
- vector embeddings stored through pgvector
- sparse-search tsvector materialization
- rerank scores
- `retrieval_index_runs`

The rule is simple: canonical data represents the paper as parsed; derived data represents how the system currently retrieves it.

## Out of scope

The MVP data model does not include LlamaIndex node graphs, PydanticAI state, user memory, or multi-tenant personalization tables. Those layers can read from this schema later, but they are not canonical data.
