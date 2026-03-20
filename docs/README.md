# Paper Context Docs

This `docs/` directory documents the implemented MVP runtime, not just the original phase plan. The codebase is the final authority, and these files explain the current architecture, data model, contracts, and operational expectations in a form that is easier to navigate than the source alone.

Use this doc set when you need to understand how Paper Context works today:

- revision-aware ingestion and replacement
- active document reads and retrieval semantics
- FastAPI and MCP contracts
- retrieval indexing and provenance guarantees
- test coverage and next-step hardening work

Current defaults:

- parser: Docling first, `pdfplumber` fallback
- queue: PGMQ
- storage: Postgres + pgvector plus local filesystem artifacts
- API surface: FastAPI
- MCP transport: FastMCP Streamable HTTP mounted at `/mcp`
- dense model: `voyage-4-large`
- reranker: `zerank-2`
- retrieval shape: passage and table retrieval plus bounded context packs

Important current caveats:

- metadata enrichment is still intentionally minimal and defaults to a no-op enricher
- provider-backed retrieval is optional; deterministic fallbacks are used when API keys are absent
- older document revisions are retained, while all user-facing reads resolve through `documents.active_revision_id`

## Quick Links

- [Architecture](./architecture.md)
- [Ingestion and Indexing](./ingestion-and-indexing.md)
- [Data Model](./data-model.md)
- [APIs and Tools](./apis-and-tools.md)
- [Retrieval](./retrieval.md)
- [Evaluation and Roadmap](./evaluation-and-roadmap.md)
- [Test Strategy](./test-strategy.md)
