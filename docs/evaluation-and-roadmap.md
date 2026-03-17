# Evaluation and Roadmap

> **Default:** Validate the deterministic retrieval stack first: parser quality, normalization, indexing, retrieval ranking, provenance, and interface stability. Deferred work stays separate from MVP requirements.

This document defines how to validate the MVP and what is intentionally deferred. It links back to the subsystem docs that own the design details.

## Retrieval and end-to-end evaluation

Reference docs:

- [Retrieval](./retrieval.md)
- [APIs and Tools](./apis-and-tools.md)
- [Data Model](./data-model.md)

Required checks:

- labeled retrieval queries return the expected passages for methods, results, limitations, and implementation details
- passage sparse and dense retrieval both use contextualized passage representations without regressing exact-match lookup or citation fidelity
- table-oriented queries return the expected result or parameter tables
- context packs contain the expected child results, parent context, provenance, and warnings
- one response never mixes index versions
- FastAPI and FastMCP return semantically aligned results for the same underlying record

## Fallback parser quality checks

Reference doc:

- [Ingestion and Indexing](./ingestion-and-indexing.md)

Required checks:

- Docling-primary parses pass on representative born-digital papers
- `pdfplumber` fallback only runs when the Docling structure gate classifies a parse as degraded
- fallback results emit the required warnings
- failed parses do not silently enter the retrieval index

## Metadata enrichment validation

Reference docs:

- [Ingestion and Indexing](./ingestion-and-indexing.md)
- [Data Model](./data-model.md)

Required checks:

- OpenAlex and Semantic Scholar matches do not overwrite higher-confidence parser metadata without recording source and confidence
- DOI, venue, year, and author enrichment remain traceable to their source
- enrichment failure produces warnings rather than corrupting canonical records

## Versioning and regression tests

Reference docs:

- [Data Model](./data-model.md)
- [Retrieval](./retrieval.md)
- [APIs and Tools](./apis-and-tools.md)

Required checks:

- changing the embedding or reranker model creates a new index version
- retrieval results remain stable for a fixed index version and corpus
- active-version flips happen only after a successful build
- passage, table, and context-pack contracts do not drift without an intentional versioned change

## Deferred items

These are not MVP requirements:

- knowledge-graph-style retrieval
- query expansion and multi-query retrieval
- self-reflective or corrective retrieval loops
- Recursive Language Models (RLM)-style downstream orchestration
- citation graph traversal
- notes and memory
- multimodal retrieval
- OCR and scanned PDFs
- LlamaIndex experiments
- PydanticAI downstream agent layer

## Future reference: query expansion and multi-query retrieval

Query expansion and multi-query retrieval are intentionally deferred for the MVP.

Rationale:

- the current retrieval stack is optimized for a deterministic single-query pipeline that is easy to debug and evaluate
- generated rewrites, expansions, or sub-queries add latency, model cost, and another failure surface before retrieval even begins
- the most plausible value for this corpus is higher recall on terminology-mismatched, underspecified, or multi-hop questions, not routine single-passage lookup

Expected value if revisited later:

- better recall when user wording does not match the paper's terminology exactly
- better coverage for short, ambiguous, or under-specified research questions
- better support for multi-hop retrieval across sections or papers when one query is not enough
- limited benefit for direct single-passage or single-table lookup when sparse+dense+rerank is already returning the right evidence

If query augmentation is revisited, prefer this order:

1. lightweight query rewriting or expansion as an optional pre-retrieval mode
2. result fusion back into the same rerank stage and output contract
3. only then consider more expensive multi-query generation for clearly complex or failing query classes

## Future reference: self-reflective and corrective retrieval loops

Self-reflective and corrective retrieval loops are intentionally deferred for the MVP.

Rationale:

- the current system is a deterministic retrieval substrate, not an answer-generation product, so most of the value from reflective retrieval would land in a later downstream agent layer rather than the retrieval core
- reflective retrieval adds latency, model cost, and another non-deterministic decision layer on top of a stack that is currently optimized for debuggability and stable retrieval contracts
- the most plausible value for this corpus is recovering from hard retrieval misses, weak initial context packs, or ambiguous research questions, not replacing the default single-pass retrieval path

Expected value if revisited later:

- better recovery when the first retrieval pass returns weak or conflicting evidence
- better handling of underspecified, terminology-mismatched, or multi-hop research questions that need one more bounded retrieval attempt
- better routing between cheap single-pass retrieval and more expensive retry behavior for clearly difficult queries
- limited benefit for direct passage or table lookup when the default sparse+dense+rerank pipeline is already returning the right evidence

If reflective retrieval is revisited, prefer this order:

1. a bounded retrieval-confidence or retrieval-quality check over the first-pass result set
2. one optional in-corpus retry using query rewriting, alternate filters, or table-first versus passage-first routing
3. only then consider broader self-critique loops in a downstream agent layer, not in the deterministic retrieval core

## Future reference: Recursive Language Models (RLM)

Recursive Language Models (RLM) are intentionally deferred for the MVP.

Rationale:

- the current system is a deterministic retrieval substrate, while most RLM value would land in a later downstream agent layer that consumes retrieval outputs rather than replacing retrieval logic
- RLM adds latency, cost, operational complexity, and a more non-deterministic execution surface than the MVP currently targets
- the most plausible value for this corpus is handling hard multi-hop, cross-paper, or terminology-mismatched research questions that are not solved by the default single-pass retrieval pipeline

Expected value if revisited later:

- better orchestration over multiple context packs when a downstream workflow needs iterative evidence gathering or synthesis across papers
- better handling of difficult research questions that need more than one bounded retrieval pass or one context pack
- limited benefit for direct passage or table lookup when the default sparse+dense+rerank pipeline is already returning the right evidence

If RLM is revisited, prefer this order:

1. use it as a downstream consumer of `build_context_pack` and the existing MCP retrieval tools
2. keep canonical ingestion, indexing, and provenance-aware retrieval deterministic and service-owned
3. only then consider deeper agentic orchestration if it shows a clear gain on hard-query evaluation without undermining debuggability or interface stability

## Future reference: graph-based retrieval options

Knowledge-graph-style retrieval is intentionally deferred for the MVP.

Rationale:

- the current retrieval stack is optimized for deterministic passage and table retrieval, not graph construction
- a full entity-relation graph adds extraction, linking, storage, and evaluation complexity that is not yet justified for the MVP
- the most plausible graph-shaped extension for this corpus is citation-linked retrieval built from extracted references, not a broad ontology or general-purpose knowledge graph

Relevant references for later evaluation:

- **GraphRAG**: Microsoft Research, "From Local to Global: A GraphRAG Approach to Query-Focused Summarization"
- **KG²RAG**: "Knowledge Graph-Guided Retrieval Augmented Generation"
- **CG-RAG**: citation-graph retrieval for research-question answering

Expected value if revisited later:

- better multi-hop and cross-document retrieval when answers depend on connecting evidence across papers
- stronger support for literature-mapping workflows such as tracing influence, support, contradiction, or follow-on methods
- limited benefit for direct single-passage or single-table lookup, where the default sparse+dense+rerank pipeline should remain the primary path

If graph-based retrieval is revisited, prefer this order:

1. citation-linked retrieval using `document_references`
2. section-and-passage graph expansion using existing document structure
3. only then consider a broader entity-relation knowledge graph if there is a demonstrated gap that citation and structural links cannot cover

## When to revisit deferred items

Revisit a deferred item only when there is a concrete trigger:

- **knowledge-graph-style retrieval**
  - revisit when evaluation shows repeated multi-hop or cross-document retrieval failures that are not fixed by chunking, reranking, or metadata filters
  - prefer citation-linked retrieval first, using `document_references`, before introducing a broader entity-relation graph
- **query expansion and multi-query retrieval**
  - revisit when evaluation shows repeated retrieval misses caused by terminology mismatch, ambiguous wording, or multi-hop query structure that are not fixed by chunking, reranking, or metadata filters
  - keep the default single-query path intact unless an optional augmentation mode shows a clear gain on labeled retrieval queries
- **self-reflective or corrective retrieval loops**
  - revisit when evaluation shows repeated first-pass retrieval failures that are improved by one bounded retry, confidence gate, or route change rather than by better chunking, reranking, or metadata filters
  - keep reflective behavior optional and bounded unless it shows a clear gain on labeled retrieval queries without undermining provenance, latency, or debuggability
- **Recursive Language Models (RLM)-style downstream orchestration**
  - revisit when evaluation shows repeated hard-query failures on multi-hop, cross-paper, or terminology-mismatched questions that are not fixed by chunking, reranking, metadata filters, or one bounded retry
  - keep RLM-style behavior downstream of the MCP retrieval contracts unless it shows a clear gain without undermining provenance, debuggability, or interface stability
- **citation graph traversal**
  - revisit when `document_references` extraction is reliable enough to support citation-linked retrieval
- **notes and memory**
  - revisit when downstream workflows need persistent user or agent state beyond one retrieval call
- **multimodal retrieval**
  - revisit when figures or non-text tables become a frequent retrieval target
- **OCR and scanned PDFs**
  - revisit when scanned PDFs are a meaningful share of the input corpus
- **LlamaIndex experiments**
  - revisit when there is a demonstrated gap in the deterministic retrieval layer that a higher-level framework would solve
- **PydanticAI downstream agent layer**
  - revisit when downstream agent orchestration needs typed state on top of stable MCP contracts

## Out of scope

The MVP evaluation plan does not score answer-generation quality because answer generation is not part of the core system defined in this doc set.
