# Data Model

Postgres is the authoritative store for canonical paper data, revision state, queue-linked ingest state, and derived retrieval assets. The important modeling decision is that a paper has a stable document identity and one or more retained revisions.

## Active-Revision Model

### `documents`

`documents` is the stable identity for a logical paper.

Important fields:

- `id`
- `title`
- `authors`
- `abstract`
- `publication_year`
- `source_type`
- `metadata_confidence`
- `quant_tags`
- `current_status`
- `active_revision_id`
- `created_at`
- `updated_at`

The top-level document row mirrors the currently active revision for convenience, but the canonical versioned state lives in `document_revisions` and revision-scoped child rows.

### `document_revisions`

Each upload or replacement creates a new document revision.

Important fields:

- `id`
- `document_id`
- `revision_number`
- `status`
- `title`
- `authors`
- `abstract`
- `publication_year`
- `source_type`
- `metadata_confidence`
- `quant_tags`
- `source_artifact_id`
- `ingest_job_id`
- `activated_at`
- `superseded_at`
- `created_at`
- `updated_at`

This table is what makes replacement non-destructive.

## Revision-Scoped Canonical Entities

Every canonical child entity is keyed to both `document_id` and `revision_id`.

### `document_artifacts`

Stored source and parser artifacts.

Important fields:

- `id`
- `document_id`
- `revision_id`
- `ingest_job_id`
- `artifact_type`
- `parser`
- `storage_ref`
- `checksum`
- `is_primary`
- `created_at`

Artifact types include the source PDF plus parser outputs.

### `document_sections`

Normalized section tree for one revision.

Important fields:

- `id`
- `document_id`
- `revision_id`
- `parent_section_id`
- `heading`
- `heading_path`
- `ordinal`
- `page_start`
- `page_end`
- `artifact_id`

### `document_passages`

Canonical prose chunks for one revision.

Important fields:

- `id`
- `document_id`
- `revision_id`
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

`body_text` is canonical display text. `contextualized_text` is derived retrieval input.

### `document_tables`

First-class extracted tables for one revision.

Important fields:

- `id`
- `document_id`
- `revision_id`
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

Extracted references for one revision.

Important fields:

- `id`
- `document_id`
- `revision_id`
- `raw_citation`
- `normalized_title`
- `authors`
- `publication_year`
- `doi`
- `source_confidence`
- `artifact_id`

## Ingest State

### `ingest_jobs`

One ingest job belongs to one document revision.

Important fields:

- `id`
- `document_id`
- `revision_id`
- `source_artifact_id`
- `status`
- `failure_code`
- `failure_message`
- `warnings`
- `stage_timings`
- `started_at`
- `finished_at`
- `trigger`
- `created_at`

This is the canonical lifecycle record for upload and replacement processing.

## Retrieval State

### `retrieval_index_runs`

One indexing build for one revision under one index version.

Important fields:

- `id`
- `document_id`
- `revision_id`
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
- `is_active`
- `activated_at`
- `deactivated_at`
- `created_at`

The schema enforces one active run per revision, not one single active run globally for the whole corpus.

### `retrieval_passage_assets`

Derived passage search rows for one run.

Important fields:

- `id`
- `retrieval_index_run_id`
- `document_id`
- `revision_id`
- `passage_id`
- `section_id`
- `publication_year`
- `search_text`
- `search_tsvector`
- `embedding`
- `created_at`

### `retrieval_table_assets`

Derived table search rows for one run.

Important fields:

- `id`
- `retrieval_index_run_id`
- `document_id`
- `revision_id`
- `table_id`
- `section_id`
- `publication_year`
- `search_text`
- `semantic_text`
- `search_tsvector`
- `embedding`
- `created_at`

Table embeddings are already present in the schema and retrieval build path.

## Provenance Rules

Required relationships:

- every canonical child row references `document_id`
- every canonical child row references `revision_id`
- passages and tables reference `section_id`
- canonical rows reference the `document_artifacts` row that produced them through `artifact_id`
- retrieval assets reference the `retrieval_index_runs` row that produced them

Required retrieval provenance:

- `document_id`
- `section_id`
- `index_version`
- `retrieval_index_run_id`
- page range
- `parser_source`

## Read Semantics

User-facing document reads are active-revision reads:

- `GET /documents/{id}`
- `GET /documents/{id}/outline`
- `GET /documents/{id}/tables`
- retrieval responses returned through MCP

Those reads join through `documents.active_revision_id` so they reflect the current live revision, while older revisions remain stored.

## Index-Version Rules

`index_version` identifies a retrieval-compatible combination of:

- parser policy
- chunking policy
- embedding provider and model
- reranker provider and model

Rules:

- a revision can have multiple historical runs
- only one run is active per revision
- one response must not mix index versions
- changing embedding or reranking defaults requires a new versioned build

## Quant Tags

Quant-specific tags may appear on:

- `documents.quant_tags`
- `document_revisions.quant_tags`
- `document_passages.quant_tags`
- `document_tables.quant_tags`

These fields are available for future filtering and ranking, but the currently exposed public filters are still narrow.
