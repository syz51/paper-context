# Retrieval

Retrieval is implemented as one shared deterministic service used by MCP. It operates over the active revision for each document and returns provenance-rich results tied to a single index version.

## Pipeline

Every retrieval request follows the same broad sequence:

1. apply document-level filters
2. select active retrieval runs for active document revisions
3. run sparse retrieval
4. run dense retrieval
5. fuse sparse and dense candidates
6. rerank the fused set
7. assemble bounded result pages or context packs

Current public filters are intentionally narrow:

- `document_ids`
- `publication_years`

## Passage Retrieval

Passage retrieval is the default path for narrative content.

Current budgets:

- sparse candidates: `30`
- dense candidates: `30`
- fused candidates: `40`
- page limit: `8`

Behavior:

- sparse search runs over contextualized passage search text
- dense search runs over passage embeddings in `retrieval_passage_assets`
- final results return canonical `body_text`, not contextualized text
- pagination is cursor-based and tied to query, filters, and index version
- exact page retrieval certifies the fused shortlist needed for the requested cursor window before reranking
- exact page retrieval advances sparse and dense candidate streams incrementally instead of refetching widened prefixes

## Table Retrieval

Tables are a first-class retrieval target, not an attachment-only feature.

Current budgets:

- sparse candidates: `20`
- dense candidates: `20`
- fused candidates: `24`
- page limit: `5`

Behavior:

- sparse search emphasizes caption, headers, and serialized cell text
- dense search uses table embeddings in `retrieval_table_assets`
- result previews are bounded row samples, not the full table body
- `get_table` is the follow-up call for full structured rows
- exact page retrieval uses the same shortlist certification and incremental candidate expansion rules as passages

## Fusion And Reranking

Sparse and dense candidates are fused and deduplicated before reranking. Result provenance preserves whether an item matched through sparse, dense, or both modes.

For exact paginated search, the service first certifies the fused shortlist required for the current cursor offset and page size, then reranks that shortlist once. This keeps the default path exact without reranking every widened prefix.

Default model settings:

- embedding model: `voyage-4-large`
- reranker model: `zerank-2`

If provider keys are missing, the runtime uses deterministic embedding and heuristic reranking implementations so retrieval remains available in local development.

## Active Revision And Index-Version Rules

Retrieval always resolves through the active document revision.

Rules:

- results in one response must not mix index versions
- result cursors are bound to the index version that produced them
- paginated cursors are opaque and carry the next absolute offset for exact replay of the requested window
- a revision can have multiple historical runs, but one active run
- changing model or chunking policy requires a new run and versioned activation

If an attempted page or pack would mix index versions, retrieval treats that as an error rather than returning ambiguous provenance.

## Parent Expansion And Context Packs

The primary retrieval units are passages and tables. Parent expansion adds only the minimum structure needed to interpret them.

Context-pack behavior:

- starts from reranked passages
- attaches relevant tables
- attaches parent section metadata
- includes supporting sibling passages when needed
- remains bounded rather than dumping full documents

When the parent expansion must be trimmed, the pack includes `parent_context_truncated`.

## Warning Semantics

Warnings are contract-level data and may appear on job records, retrieval items, and whole packs.

Common warnings:

- `parser_fallback_used`
- `reduced_structure_confidence`
- `metadata_low_confidence`
- `parent_context_truncated`

Consumers should propagate these warnings instead of hiding them.

## Output Contract

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
- `parser_source`
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
- `parser_source`
- `warnings`

### Table detail result

- all table identity and location fields
- `headers`
- `rows`
- `row_count`
- `index_version`
- `retrieval_index_run_id`
- `parser_source`
- `warnings`

### Passage context result

- selected passage metadata
- neighboring passages with `relationship`
- pack-level warnings

### Context-pack result

- `context_pack_id`
- `query`
- `passages`
- `tables`
- `parent_sections`
- `documents`
- `provenance`
- `warnings`
- `next_cursor`

## Out Of Scope

The retrieval layer still does not implement:

- query rewriting by default
- multi-query retrieval
- citation-graph traversal
- answer synthesis
- framework-managed retrievers
