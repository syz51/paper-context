# Paper Context

Paper Context is my personal, design-first experiment for building a retrieval system around born-digital research PDFs, especially quant and trading-relevant papers. The current direction is a deterministic ingestion and retrieval stack built around Docling-first parsing with `pdfplumber` fallback, Postgres + pgvector storage, FastAPI + FastMCP interfaces, Voyage `voyage-4-large` embeddings, Zero Entropy `zerank-2` reranking, and contextual retrieval with parent-child expansion.

> **Status**
>
> The repository now includes a phase-0 runtime skeleton alongside the design docs:
>
> Docker Compose for Postgres + pgvector + PGMQ, a FastAPI health surface that mounts FastMCP at `/mcp`, Alembic migrations, a direct-SQL PGMQ adapter, and a synthetic worker smoke path.
> The ingestion and retrieval logic beyond the queue/bootstrap flow are still intentionally incomplete.

## Why This Exists

Most RAG examples collapse important retrieval decisions into high-level frameworks, generic chat demos, or vague architecture diagrams. I want a tighter design for research-paper retrieval that stays explicit about parsing quality, provenance, indexing policy, and what downstream agents should consume.

This repo is where I am working through that design in public. It is useful both as a personal reference and as a way to pressure-test the system shape before committing to a full implementation.

## What The System Is

- A personal experimental retrieval system for ingesting and retrieving born-digital paper PDFs
- A deterministic retrieval substrate for downstream tools and agents
- A normalized store for documents, sections, passages, tables, references, and artifacts
- A provenance-aware retrieval design that favors debuggability over abstraction-heavy convenience
- Currently centered on quant and trading-relevant research papers

## What The System Is Not

- A production-ready OSS package
- A full end-to-end RAG application or answer-generation product
- An OCR-first pipeline for scanned PDFs
- A generic agent framework
- A place to hide retrieval behavior behind LlamaIndex or PydanticAI abstractions

## Current Architecture Snapshot

The current MVP design is organized around three planned runtime surfaces plus one optional downstream layer:

- `app`: a FastAPI surface that hosts `/healthz` and `/readyz` and mounts the FastMCP transport at `/mcp`
- `worker`: a background process for parsing, normalization, enrichment, chunking, embedding, and index-version writes
- `skill` (optional): downstream agent guidance, not canonical retrieval logic

The retrieval path is intentionally deterministic: metadata filtering, sparse retrieval, dense retrieval, fusion, reranking, and parent expansion. The goal is to return grounded context packs with stable provenance for downstream agents rather than hide the system behind answer synthesis.

## Repo Layout

```text
.
├── alembic/
├── docs/
├── src/paper_context/
├── tests/
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.db
├── pyproject.toml
└── AGENTS.md
```

## Documentation Guide

If you want the current source of truth, start here:

- [`docs/README.md`](./docs/README.md): overview of the design set and current defaults
- [`docs/architecture.md`](./docs/architecture.md): runtime surfaces, component boundaries, and system shape
- [`docs/retrieval.md`](./docs/retrieval.md): query pipeline, context-pack behavior, and provenance rules
- [`docs/ingestion-and-indexing.md`](./docs/ingestion-and-indexing.md): parsing, normalization, chunking, enrichment, and indexing flow
- [`docs/data-model.md`](./docs/data-model.md): canonical entities, provenance links, and index-version rules
- [`docs/apis-and-tools.md`](./docs/apis-and-tools.md): FastAPI and FastMCP contracts
- [`docs/evaluation-and-roadmap.md`](./docs/evaluation-and-roadmap.md): validation criteria, deferred items, and near-term roadmap

## Current Defaults / Tech Choices

- Parser: Docling first, `pdfplumber` fallback when structure is degraded
- Storage: Postgres + pgvector
- API surface: FastAPI
- Tool surface: FastMCP
- Embeddings: Voyage `voyage-4-large`
- Reranker: Zero Entropy `zerank-2`
- Retrieval shape: contextualized passage retrieval plus parent-child expansion
- Framework boundary: no LlamaIndex or PydanticAI in the MVP core

## Near-Term Roadmap

- Refine the ingestion pipeline for born-digital paper PDFs
- Validate structure quality gates and fallback behavior
- Lock down the normalized data model and provenance requirements
- Test retrieval quality for passages, tables, and context packs
- Keep the hosted app and MCP contracts narrow, explicit, and version-aware

## Phase 0 Bring-Up

Copy `.env.example` to `.env`, then:

```bash
uv sync --extra dev
docker compose up --build -d db
docker compose run --rm migrate
python -m paper_context.cli serve
python -m paper_context.cli verify-synthetic-job
```

Or run the full stack in Compose:

```bash
docker compose --profile migrate up --build
```

Phase 0 exit checks:

- App health: `GET /healthz`, readiness: `GET /readyz`
- Mounted MCP transport: `GET /mcp`
- Synthetic job verification: `python -m paper_context.cli verify-synthetic-job`

`Dockerfile.db` is the local Postgres image used by Compose today. It keeps the PGMQ + `pgvector`
setup self-contained for local bring-up and leaves room for future integration or regression tests
to reuse the same database image.

## Developer Quality Gate

The repo targets Python 3.14 and separates static checks from test execution:

- Local hooks: `uv run pre-commit install --hook-type pre-commit`
- Manual static run: `uv run pre-commit run --all-files`
- Type checking only: `uv run pyright`
- Fast test lane: `uv run pytest -m "unit or slice"`
- Integration lane: `uv run pytest -m "integration or migration"`
- Contract lane: `uv run pytest -m contract`
- Regression lane: `uv run pytest -m "regression and not staging_only"`

Pre-push hooks are intentionally not installed by default here because they would mostly duplicate the dedicated GitHub Actions test lanes and add local latency without new signal.

GitHub Actions mirrors that baseline with a PR/push quality workflow and a separate security workflow for dependency review and CodeQL scanning.

## Contributing

Issues and discussion are welcome, especially if you see weak assumptions, missing edge cases, or better ways to structure the retrieval core. That said, this is still a personal experimental project, so interfaces, scope, and implementation choices may change quickly.

I have not added a final open source license yet. That will need to be clarified before any broader public release.
