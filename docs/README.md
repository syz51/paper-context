# Paper Context Docs

This `docs/` directory documents the implemented MVP runtime, not just the original phase plan. The codebase is the final authority, and these files explain the current architecture, data model, contracts, and operational expectations in a form that is easier to navigate than the source alone.

Use this doc set when you need to understand how Paper Context works today:

- revision-aware ingestion and replacement
- active document reads and retrieval semantics
- FastAPI and MCP contracts
- retrieval indexing and provenance guarantees
- test coverage and next-step hardening work

Environment assumptions:

- Python `3.14`
- `uv` for local commands
- Docker only for the repo-managed Postgres service, Compose bring-up, or Postgres-backed test lanes

Important current caveats:

- metadata enrichment is still intentionally minimal and defaults to a no-op enricher
- provider-backed retrieval is optional; deterministic fallbacks are used when API keys are absent
- older document revisions are retained, while all user-facing reads resolve through `documents.active_revision_id`
- production Compose assumes an external Postgres instead of starting `db` itself

Compatibility backstop:

- the code remains the final authority
- the HTTP and MCP payloads documented here are also pinned by contract goldens under `tests/contract/golden/`

## Quick Links

- [Architecture](./architecture.md)
- [Ingestion and Indexing](./ingestion-and-indexing.md)
- [Data Model](./data-model.md)
- [APIs and Tools](./apis-and-tools.md)
- [Retrieval](./retrieval.md)
- [Evaluation and Roadmap](./evaluation-and-roadmap.md)
- [Test Strategy](./test-strategy.md)
