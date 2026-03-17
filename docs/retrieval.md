# Retrieval

> **Default:** Retrieval uses metadata filter -> sparse -> dense -> fusion -> rerank -> parent expansion, with contextualized passage text feeding both sparse and dense passage retrieval, Voyage `voyage-4-large` for embeddings, and Zero Entropy `zerank-2` for reranking. Results are provenance-rich and index-version aware.

This document defines the query pipeline, top-k defaults, table handling, parent-child retrieval behavior, context-pack assembly, and the retrieval output contract. For provenance and index fields, see [Data Model](./data-model.md).

## Query pipeline

Every retrieval request follows the same ordered stages:

1. **Metadata filter**
   - apply document-level and passage-level filters first
   - common filters include year, tags, document IDs, and result type
2. **Sparse retrieval**
   - run Postgres full-text search over contextualized passage text, section headings, and selected metadata
   - for tables, keep a separate lexical path over captions, headers, and serialized cell text
3. **Dense retrieval**
   - embed the query once with Voyage `voyage-4-large`
   - search pgvector against the active index version
4. **Fusion**
   - merge sparse and dense candidates
   - deduplicate by entity ID
   - preserve source-stage provenance for debugging
5. **Rerank**
   - rerank the fused candidate set with Zero Entropy `zerank-2`
6. **Parent expansion**
   - attach section and document context for the selected child items

The pipeline is deterministic and service-owned. Future agents consume these outputs; they do not change the retrieval algorithm.

Query rewriting, query expansion, and multi-query retrieval are not part of the MVP default pipeline. If they are revisited later, they should be introduced as optional pre-retrieval augmentation modes that still fuse into the same rerank and provenance-aware output contract.

## Passage retrieval policy

Passage search is the default retrieval path for narrative content.

Default budgets:

- sparse candidate set: top 30
- dense candidate set: top 30
- fused rerank set: top 40
- final returned passages: top 8

Passage behavior:

- search over the contextualized passage text for sparse retrieval
- search over the same contextualized passage text for dense retrieval
- return the raw passage text for display and citation
- preserve section path, page range, chunk ordinal, and index-version provenance
- allow pagination by cursor over the reranked result set

## Table retrieval path

Table retrieval is a separate first-class path.

Default budgets:

- sparse candidate set: top 20
- dense candidate set: top 10
- fused rerank set: top 20
- final returned tables: top 5

Table behavior:

- lexical search emphasizes caption, headers, and serialized cell text
- dense search is allowed when table embeddings exist for the active version
- results return structured previews, not flattened prose
- tables may also be attached during context-pack assembly when they are linked to selected passages or sections

## Parent-child retrieval behavior

The child retrieval unit is the passage or table. The parent retrieval unit is the section, with document metadata attached above it.

Rules:

- child results always include `document_id` and `section_id`
- parent expansion returns the minimal section context needed to interpret the child
- section expansion must not silently pull the whole document body
- context packs may include sibling passages when needed for coherence, but the triggering child items stay explicit

This keeps retrieval precise while still returning enough structure for downstream use.

## Context-pack assembly

A **context pack** is the preferred retrieval product for downstream agents and tools.

A context pack contains:

- selected passage results
- related table results when relevant
- parent section metadata
- document metadata
- pack-level warnings
- pack-level provenance

Assembly rules:

- start from reranked child results
- attach only the parent structure needed to interpret those results
- include citations at the passage or table level, not only at the pack level
- keep the pack bounded; do not use it as a full-document export

## Citation and provenance rules

Every returned passage or table must include enough provenance to be cited directly.

Required provenance fields:

- `document_id`
- `section_id`
- `index_version`
- `retrieval_index_run_id`
- `page_start`
- `page_end`
- `parser_source`

Recommended citation fields:

- document title
- section path
- passage or table ID
- table caption when applicable

Warnings must be explicit. Common warnings include:

- `parser_fallback_used`
- `reduced_structure_confidence`
- `metadata_low_confidence`
- `parent_context_truncated`

## Embedding and reranker defaults

- embedding model: Voyage `voyage-4-large`
- reranker model: Zero Entropy `zerank-2`
- sparse passage retrieval reads `contextualized_text` through the active sparse index materialization
- dense retrieval always reads from the active `index version`
- any model swap requires a new index version before activation

## Retrieval output contract

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
- `passages[]`
- `tables[]`
- `parent_sections[]`
- `documents[]`
- `provenance`
- `warnings`

## Out of scope

The MVP retrieval layer does not expose LlamaIndex retrievers, PydanticAI plans, citation-graph traversal, or answer synthesis. Those are downstream consumers of this contract, not part of it.
