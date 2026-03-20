# APIs And Tools

FastAPI owns operational HTTP routes. FastMCP owns the agent-facing retrieval surface. The codebase does not expose retrieval over plain REST today; retrieval is intentionally concentrated in MCP.

## FastAPI

### `POST /documents`

Creates a new logical document and its first revision.

Input:

- multipart `file`
- optional form `title`

Returns:

- `document_id`
- `ingest_job_id`
- `status`

### `POST /documents/{document_id}/replace`

Creates a new revision for an existing document and enqueues a replacement ingest job.

Input:

- multipart `file`
- optional form `title`

Returns:

- `document_id`
- `ingest_job_id`
- `status`

Response status is `202`.

### `GET /documents`

Lists current document summaries with cursor pagination.

Fields:

- `documents`
- `next_cursor`

Each document summary includes:

- `document_id`
- `title`
- `authors`
- `publication_year`
- `quant_tags`
- `current_status`
- `active_index_version`

### `GET /documents/{document_id}`

Returns the active document summary for one logical document.

### `GET /documents/{document_id}/outline`

Returns the active revision’s section tree.

Fields:

- `document_id`
- `title`
- `sections`

Each section node includes:

- `section_id`
- `parent_section_id`
- `heading`
- `section_path`
- `ordinal`
- `page_start`
- `page_end`
- `children`

### `GET /documents/{document_id}/tables`

Returns active-revision tables for a document.

Fields:

- `document_id`
- `title`
- `tables`

Each table record includes:

- `table_id`
- `document_id`
- `section_id`
- `document_title`
- `section_path`
- `caption`
- `table_type`
- `preview`
- `page_start`
- `page_end`

### `GET /ingest-jobs/{ingest_job_id}`

Returns ingest status for one job.

Fields:

- `id`
- `document_id`
- `status`
- `failure_code`
- `failure_message`
- `warnings`
- `started_at`
- `finished_at`
- `trigger`

### `GET /healthz`

Simple liveness probe.

Fields:

- `service`
- `status`
- `version`

### `GET /readyz`

Operational readiness probe.

Fields:

- `service`
- `status`
- `version`
- `database_ready`
- `storage_root`
- `storage_ready`
- `queue_name`
- `queue_ready`
- `queue_metrics`
- `operation_timings`

`queue_metrics` includes:

- `queue_name`
- `queue_length`
- `queue_visible_length`
- `newest_msg_age_sec`
- `oldest_msg_age_sec`
- `total_messages`
- `scrape_time`

`operation_timings` reports recent observed timings for app and worker operations.

## MCP Tools

The app exposes FastMCP over Streamable HTTP at `/mcp`.

Current tools:

### `search_documents`

Arguments:

- `query`
- `filters`
- `cursor`
- `limit`

`filters` currently supports:

- `document_ids`
- `publication_years`

Use this for document-level narrowing before passage or table retrieval.

### `search_passages`

Arguments:

- `query`
- `filters`
- `cursor`
- `limit`

Returns:

- `query`
- `passages`
- `next_cursor`

Each passage result includes:

- `passage_id`
- `document_id`
- `section_id`
- `document_title`
- `section_path`
- `text`
- `score`
- `retrieval_modes`
- `page_start`
- `page_end`
- `index_version`
- `retrieval_index_run_id`
- `parser_source`
- `warnings`

### `search_tables`

Arguments:

- `query`
- `filters`
- `cursor`
- `limit`

Returns:

- `query`
- `tables`
- `next_cursor`

Each table result includes:

- `table_id`
- `document_id`
- `section_id`
- `document_title`
- `section_path`
- `caption`
- `table_type`
- `preview`
- `score`
- `retrieval_modes`
- `page_start`
- `page_end`
- `index_version`
- `retrieval_index_run_id`
- `parser_source`
- `warnings`

### `get_document_outline`

Arguments:

- `document_id`

Returns the same outline structure as the HTTP route.

### `get_table`

Arguments:

- `table_id`

Returns:

- table identity and document metadata
- `headers`
- `rows`
- `row_count`
- `index_version`
- `retrieval_index_run_id`
- `parser_source`
- `warnings`

### `get_passage_context`

Arguments:

- `passage_id`
- `before`
- `after`

Returns:

- selected passage metadata
- bounded neighboring passages
- warning propagation

### `build_context_pack`

Arguments:

- `query`
- `filters`
- `cursor`
- `limit`

Returns:

- `context_pack_id`
- `query`
- `passages`
- `tables`
- `parent_sections`
- `documents`
- `provenance`
- `warnings`
- `next_cursor`

This is the preferred default tool for downstream consumers.

## Current Limits

MCP route-level limit clamps:

- `search_documents`: max `100`
- `search_passages`: max `8`
- `search_tables`: max `5`
- `build_context_pack`: max `8`

## Usage Guidance

Prefer `build_context_pack` when the caller wants grounded evidence for downstream synthesis or reasoning.

Use raw search tools directly only when the workflow needs:

- independent passage and table pagination
- explicit structure inspection
- a follow-up call like `get_table` or `get_passage_context`

Warnings are contract data. Consumers should surface them rather than hiding them.

## Out Of Scope

The current public surface does not expose:

- retrieval over ordinary REST routes
- auto-generated tools from FastAPI
- framework-specific abstractions from LlamaIndex or PydanticAI
