# Paper Context MVP Docs

This `docs/` set is the implementation-facing design for the revised Paper Context MVP. It defines a deterministic ingestion and retrieval system for born-digital research PDFs, optimized for quant and trading papers, with enough rationale to preserve decisions without turning the docs into long essays.

The system is a retrieval substrate for downstream tools and agents. It ingests PDFs, preserves sections, passages, tables, and provenance, and returns grounded context packs. It is not an answer-generation product, not a generic agent framework, and not a place to hide retrieval decisions behind LlamaIndex or PydanticAI abstractions.

Status note: this documentation set supersedes `~/Downloads/rag-plan.md` and is the current source of truth for the MVP design. Phase 4 is now focused on self-hosted hardening; revision-safe retention and stable document keyset pagination are already implemented in the runtime.

> **Current defaults**
>
> - Parser: Docling first, `pdfplumber` fallback when Docling structure is not usable.
> - Embeddings: Voyage `voyage-4-large`.
> - Reranker: Zero Entropy `zerank-2`.
> - Storage: Postgres + pgvector.
> - API stack: FastAPI for operational HTTP, FastMCP for MCP tools.
> - Retrieval shape: contextual retrieval over contextualized passage text plus parent-child retrieval.
> - Framework boundary: no LlamaIndex or PydanticAI in the MVP core.

## What the system is and is not

**The system is:**

- a single-tenant ingestion and retrieval service for born-digital paper PDFs
- a normalized store for documents, sections, passages, tables, references, and artifacts
- a deterministic retrieval layer that returns provenance-rich results and context packs

**The system is not:**

- a chat application or answer synthesizer
- an OCR-first or scanned-PDF pipeline
- a multi-tenant search platform
- a LlamaIndex- or PydanticAI-based core architecture

## Quick links

- [Architecture](./architecture.md)
- [Ingestion and Indexing](./ingestion-and-indexing.md)
- [Retrieval](./retrieval.md)
- [Data Model](./data-model.md)
- [APIs and Tools](./apis-and-tools.md)
- [Evaluation and Roadmap](./evaluation-and-roadmap.md)
