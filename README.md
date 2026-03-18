# Paper Context

Paper Context is a personal, retrieval-first project for born-digital research PDFs, especially quant and trading-relevant papers. The codebase now implements an ingestion-focused MVP slice: PDF upload, queued ingest jobs, deterministic worker processing, canonical Postgres normalization, local artifact storage, and a small operational API surface. The broader retrieval and MCP tool design still exists in `docs/`, but parts of that design are ahead of the current runtime.

> **Status**
>
> What is implemented today:
>
> - FastAPI endpoints for `POST /documents`, `GET /ingest-jobs/{ingest_job_id}`, `GET /healthz`, and `GET /readyz`
> - A queue-backed worker that runs deterministic ingest stages from upload through `ready` or `failed`
> - Docling-first parsing with `pdfplumber` fallback when structure is degraded
> - Normalization into documents, sections, passages, tables, references, artifacts, and retrieval-index metadata
> - Local filesystem artifact storage and Docker Compose bring-up for `db`, `app`, `worker`, and `migrate`
>
> What is still target-state / not fully implemented yet:
>
> - Retrieval query APIs and the shared retrieval service are still placeholders
> - The mounted MCP transport exists at `/mcp`, but the curated retrieval tools described in `docs/apis-and-tools.md` are not wired up yet
> - Provider-backed enrichment, embedding writes, and reranking are represented in the design and index metadata, but not yet exposed as a live end-to-end retrieval path

## Why This Exists

Most RAG examples blur important retrieval decisions behind high-level frameworks, generic chat demos, or vague diagrams. I want a narrower system for research-paper retrieval that stays explicit about parsing quality, provenance, chunking policy, and what downstream tools should consume.

This repo is where I pressure-test that design in code. The current implementation is centered on ingestion correctness first, with retrieval remaining intentionally explicit rather than abstracted away behind framework magic.

## What The System Is

- A personal experimental ingestion and retrieval substrate for born-digital paper PDFs
- A deterministic worker pipeline that preserves structure and provenance
- A normalized store for documents, sections, passages, tables, references, and artifacts
- A local API surface for uploads, ingest job inspection, and operational health
- A project that prefers explicit system boundaries over LlamaIndex- or PydanticAI-led abstractions

## What The System Is Not

- A production-ready OSS package
- A chat product or answer-generation application
- An OCR-first pipeline for scanned PDFs
- A fully implemented retrieval service today
- A generic agent framework

## Current Runtime Snapshot

The implemented runtime is organized around three active surfaces:

- `app`: FastAPI service exposing upload, ingest-status, health, readiness, and a mounted MCP HTTP transport
- `worker`: queue-backed ingestion process that parses, normalizes, writes artifacts, creates passages, and records retrieval-index runs
- `db`: Postgres + pgvector + PGMQ, started locally through Docker Compose

The top-level README tracks the current runtime. The `docs/` directory describes the broader MVP design, including retrieval and MCP contracts that are only partially implemented today.

## Implemented Ingestion Flow

Current happy-path behavior:

1. `POST /documents` accepts a multipart PDF upload.
2. The app validates that the upload is a non-empty PDF and enforces the configured upload limit.
3. The source PDF is stored under the local artifact root.
4. A `documents` row and an `ingest_jobs` row are created, and work is enqueued through PGMQ.
5. The worker claims the job and advances it through `queued`, `parsing`, `normalizing`, `enriching_metadata`, `chunking`, `indexing`, and finally `ready` or `failed`.
6. The worker parses with Docling first and falls back to `pdfplumber` when Docling is structurally degraded.
7. Parsed content is normalized into sections, tables, references, passages, artifacts, and a `retrieval_index_runs` record.

Some operational details worth knowing:

- Default upload limit: `25 MiB`
- Default artifact root: `./var/artifacts`
- Trace headers such as `traceparent`, `tracestate`, `baggage`, `x-request-id`, and `x-trace-id` are forwarded into queue payload metadata
- Failed jobs expose both `failure_code` and `failure_message`
- Warnings such as `parser_fallback_used`, `reduced_structure_confidence`, and `metadata_low_confidence` are preserved on the ingest job
- If a newer ingest job supersedes an older queued/running one for the same document, the older job is failed explicitly as superseded

## Current API Surface

Implemented endpoints:

- `POST /documents`
  - accepts multipart upload with `file` and optional `title`
  - returns `document_id`, `ingest_job_id`, and initial `status`
- `GET /ingest-jobs/{ingest_job_id}`
  - returns current job status, warnings, timestamps, trigger, and any failure details
- `GET /healthz`
  - lightweight liveness check
- `GET /readyz`
  - readiness check that includes database status, storage root, and queue name
- `GET /mcp`
  - mounted Streamable HTTP MCP transport shell

Not yet implemented in the runtime:

- document listing and detail reads
- retrieval queries
- table/document/passage search endpoints
- curated MCP retrieval tools

## CLI Commands

The project exposes a `paper-context` CLI:

- `uv run paper-context serve`
- `uv run paper-context worker`
- `uv run paper-context worker --once`
- `uv run paper-context verify-synthetic-job`

`verify-synthetic-job` is still useful as a fast queue/worker smoke test, but the main user-visible workflow is now real PDF upload plus job polling.

## Repo Layout

```text
.
├── alembic/
├── docs/
├── src/paper_context/
├── tests/
├── docker-compose.yml
├── docker-compose.prod.yml
├── Dockerfile
├── Dockerfile.db
├── pyproject.toml
└── AGENTS.md
```

## Quickstart

Copy `.env.example` to `.env`, then install dependencies and start Postgres:

```bash
uv sync --extra dev
docker compose up --build -d db
docker compose run --rm migrate
```

Run the app and worker locally against the Compose database:

```bash
uv run paper-context serve
uv run paper-context worker
```

Or run the full local stack in Compose:

```bash
docker compose up --build -d
```

Notes:

- Host-run commands use the default database URL at `localhost:5433`
- Compose services override the database hostname to `db:5432`
- App and worker share the local artifact volume at `./var/artifacts`
- Host-run `uv run ...` commands still expect you to run `docker compose run --rm migrate` first

## Try The Upload Flow

With the app running on `localhost:8000`:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@/absolute/path/to/paper.pdf" \
  -F "title=Optional Paper Title"
```

That returns a `document_id`, an `ingest_job_id`, and an initial `queued` status. Then poll the job:

```bash
curl http://127.0.0.1:8000/ingest-jobs/<ingest_job_id>
```

Readiness and liveness checks:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

## Documentation Guide

Use the root README for implemented runtime behavior. Use `docs/` for the broader MVP design and planned retrieval direction:

- [`docs/README.md`](./docs/README.md): documentation overview
- [`docs/architecture.md`](./docs/architecture.md): runtime surfaces and system boundaries
- [`docs/ingestion-and-indexing.md`](./docs/ingestion-and-indexing.md): ingestion pipeline and index-version rules
- [`docs/data-model.md`](./docs/data-model.md): canonical entities and provenance links
- [`docs/apis-and-tools.md`](./docs/apis-and-tools.md): target API and MCP contracts
- [`docs/retrieval.md`](./docs/retrieval.md): target retrieval pipeline and context-pack behavior
- [`docs/evaluation-and-roadmap.md`](./docs/evaluation-and-roadmap.md): roadmap and deferred items
- [`docs/test-strategy.md`](./docs/test-strategy.md): test layers and coverage expectations

If the README and the design docs appear to disagree, treat that as:

- README: what is implemented in the repo now
- `docs/`: where the project is intended to go next

## Current Defaults / Tech Choices

- Python: 3.14
- Package/tooling: `uv`
- API surface: FastAPI
- MCP transport: FastMCP Streamable HTTP mount
- Database: Postgres + pgvector
- Queue: PGMQ
- Primary parser: Docling
- Fallback parser: `pdfplumber`
- Storage: local filesystem artifacts
- Retrieval design target: contextualized passage retrieval plus parent-child expansion

## Developer Quality Gate

The repo separates static checks from test execution:

- `uv run pre-commit install --hook-type pre-commit`
- `uv run pre-commit run --all-files`
- `uv run pyright`
- `uv run pytest -m "unit or slice"`
- `uv run pytest -m "integration or migration"`
- `uv run pytest -m contract`
- `uv run pytest -m "regression and not staging_only"`

GitHub Actions mirrors that split with dedicated quality and security workflows.

## Contributing

Issues and discussion are welcome, especially around weak assumptions, edge cases, ingestion correctness, and retrieval-boundary decisions. Interfaces and implementation details may still change quickly because this is an active personal project rather than a stabilized public package.

I have not added a final open-source license yet. That still needs to be clarified before any broader release.
