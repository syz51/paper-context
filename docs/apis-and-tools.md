# APIs and Tools

> **Default:** FastAPI owns operational HTTP endpoints and FastMCP owns the agent-facing tool surface. Downstream workflows should prefer `build_context_pack` over raw retrieval primitives.

This document defines the external interfaces for the MVP. For retrieval semantics and output provenance rules, see [Retrieval](./retrieval.md).

## FastAPI endpoints

### `POST /documents`

Upload a PDF and create an ingest job.

Returns:

- `document_id`
- `ingest_job_id`
- initial status

### `GET /documents`

List documents with pagination and lightweight metadata.

### `GET /documents/{document_id}`

Return one document record, including current ingest status and active index-version metadata.

### `GET /documents/{document_id}/outline`

Return the section tree for the document.

### `GET /documents/{document_id}/tables`

Return tables for the document with captions, page spans, and preview data.

### `GET /ingest-jobs/{ingest_job_id}`

Return job status, warnings, and failure information.

### `POST /documents/{document_id}/replace`

Replace the source PDF and trigger a new ingest job.

## FastMCP tools

### `search_documents(query, filters, cursor)`

Use when the caller needs document-level narrowing before passage or table retrieval.

### `search_passages(query, filters, cursor)`

Use for direct passage retrieval through the standard pipeline.

### `search_tables(query, filters, cursor)`

Use for table-first retrieval when the question is metric-, result-, or parameter-table oriented.

### `get_document_outline(document_id)`

Use to inspect structure before or after retrieval.

### `get_table(table_id)`

Use to fetch the full structured table payload for a specific result.

### `get_passage_context(passage_id, before, after)`

Use when a caller already has a passage and needs bounded neighboring context.

### `build_context_pack(query, filters, cursor)`

Preferred default tool for downstream workflows. It returns the best passages, linked tables, parent section context, provenance, and warnings in one call.

## Output shapes

### Document result

- `document_id`
- `title`
- `authors`
- `publication_year`
- `quant_tags`
- `current_status`
- `active_index_version`

### Passage result

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
- `warnings`

### Table result

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
- `warnings`

### Context-pack result

- `context_pack_id`
- `query`
- `documents`
- `parent_sections`
- `passages`
- `tables`
- `provenance`
- `warnings`
- `next_cursor`

Output semantics for ranking, provenance, citations, and warnings are defined in [Retrieval](./retrieval.md).

## Agent usage guidance

Use `build_context_pack` by default in downstream agent workflows.

Only use raw search tools directly when the workflow specifically needs:

- custom pagination
- separate passage and table handling
- structure inspection before pack assembly

Agents should surface warnings, not suppress them. Fallback parser use and low-confidence metadata are part of the contract.

## Transport

FastMCP transport defaults:

- production: Streamable HTTP
- local development: stdio

FastAPI remains plain HTTP. The system should not auto-generate MCP tools from FastAPI routes.

## Out of scope

The MVP tool surface does not expose LlamaIndex or PydanticAI primitives. Those belong in later downstream layers if they are introduced at all.
