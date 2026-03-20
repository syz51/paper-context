# Paper Context

Paper Context is a retrieval-first system for born-digital research PDFs. The MVP runtime is implemented end to end: upload and replacement flows, queued ingestion, revision-aware normalization, retrieval indexing in Postgres + pgvector, document inspection endpoints, and MCP retrieval tools.

The codebase is optimized for explicit retrieval behavior and provenance, not for generic chat demos or framework abstraction. The key boundary is simple: ingestion, normalization, indexing, and retrieval are deterministic service-owned logic; downstream agents consume that surface through HTTP and MCP.

## Status

Implemented now:

- FastAPI endpoints for upload, replacement, document reads, ingest status, health, and readiness
- A PGMQ-backed worker that advances jobs through `queued`, `parsing`, `normalizing`, `enriching_metadata`, `chunking`, `indexing`, and `ready` or `failed`
- Docling-first parsing with `pdfplumber` fallback when structure is degraded
- Revision-aware storage of documents, artifacts, sections, passages, tables, references, ingest jobs, and retrieval assets
- Retrieval over passages and tables with sparse search, dense search, fusion, reranking, bounded parent expansion, and context-pack assembly
- Mounted FastMCP tools for `search_documents`, `search_passages`, `search_tables`, `get_document_outline`, `get_table`, `get_passage_context`, and `build_context_pack`
- Compose bring-up for `db`, `migrate`, `app`, and `worker`
- Readiness reporting with database state, storage checks, queue metrics, and recent operation timings

Still intentionally limited:

- Metadata enrichment is wired as a no-op `NullMetadataEnricher` by default
- Provider-backed retrieval is optional; without API keys the runtime uses deterministic embeddings and a heuristic reranker
- The project is still a self-hosted MVP, not a polished multi-tenant product or answer-generation app

## Runtime Model

The deployed runtime has four processes:

- `db`: Postgres with `pgvector` and `pgmq`
- `migrate`: one-shot Alembic runner
- `app`: FastAPI service, which mounts the MCP Streamable HTTP app at `/mcp`
- `worker`: background ingestion and indexing loop

The core data model is revision-aware:

- `documents` is the stable paper identity
- each upload or replacement creates a new `document_revisions` row
- `documents.active_revision_id` points reads and retrieval at the current live revision
- older revisions are retained instead of being destructively overwritten

That matters for replacement behavior. `POST /documents/{document_id}/replace` does not mutate canonical rows in place; it stages a new revision, enqueues a new ingest job, and only promotes that revision when indexing completes successfully.

## What The System Does

- Ingest born-digital PDFs and preserve page provenance
- Normalize sections, passages, tables, references, and artifacts into Postgres
- Build passage and table retrieval assets tied to `retrieval_index_runs`
- Serve operational HTTP routes for upload, status, and document inspection
- Serve agent-facing MCP tools for document search and retrieval

## What It Does Not Do

- OCR or scanned-PDF ingestion
- answer synthesis or chat orchestration
- multi-tenant SaaS concerns
- LlamaIndex or PydanticAI as part of the retrieval core

## Quickstart

Install dependencies and prepare local config:

```bash
cp .env.example .env
uv sync --extra dev
```

Start Postgres and run migrations:

```bash
docker compose up --build -d db
docker compose run --rm migrate
```

Run the app and worker from the host:

```bash
uv run paper-context serve
uv run paper-context worker
```

Or bring up the full stack in Compose:

```bash
docker compose up --build -d
```

Useful defaults:

- host database URL: `postgresql+psycopg://paper_context:paper_context@localhost:5433/paper_context`
- default artifact root on host: `./var/artifacts`
- Compose artifact mount: `/var/lib/paper-context/artifacts`
- default upload limit: `25 MiB`

## Main Workflow

Upload a PDF:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@/absolute/path/to/paper.pdf" \
  -F "title=Optional Paper Title"
```

Poll the job:

```bash
curl http://127.0.0.1:8000/ingest-jobs/<ingest_job_id>
```

Inspect the current active document state:

```bash
curl http://127.0.0.1:8000/documents/<document_id>
curl http://127.0.0.1:8000/documents/<document_id>/outline
curl http://127.0.0.1:8000/documents/<document_id>/tables
```

Replace the source PDF for the same logical document:

```bash
curl -X POST http://127.0.0.1:8000/documents/<document_id>/replace \
  -F "file=@/absolute/path/to/replacement.pdf" \
  -F "title=Optional Replacement Title"
```

Operational checks:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

## MCP Surface

The app mounts FastMCP at `http://127.0.0.1:8000/mcp` using Streamable HTTP.

Current tools:

- `search_documents`
- `search_passages`
- `search_tables`
- `get_document_outline`
- `get_table`
- `get_passage_context`
- `build_context_pack`

Minimal raw MCP example:

```bash
SESSION_ID=$(
  curl -si http://127.0.0.1:8000/mcp \
    -H 'accept: application/json, text/event-stream' \
    -H 'content-type: application/json' \
    -d '{
      "jsonrpc": "2.0",
      "id": 1,
      "method": "initialize",
      "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "example", "version": "0"}
      }
    }' | awk '/mcp-session-id:/ {print $2}' | tr -d '\r'
)

curl http://127.0.0.1:8000/mcp \
  -H 'accept: application/json, text/event-stream' \
  -H 'content-type: application/json' \
  -H "mcp-session-id: ${SESSION_ID}" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "build_context_pack",
      "arguments": {"query": "attention mechanism"}
    }
  }'
```

For most downstream consumers, `build_context_pack` is the best default entrypoint.

## Retrieval And Ingestion Notes

Important implemented behavior:

- warnings such as `parser_fallback_used`, `reduced_structure_confidence`, `metadata_low_confidence`, and `parent_context_truncated` are part of the public retrieval contract
- document reads always resolve against the active revision
- retrieval results include `index_version`, `retrieval_index_run_id`, and `parser_source`
- if a newer job supersedes an older queued or in-flight job for the same document, the older job is failed explicitly as superseded
- replacement retains prior revisions; a failed replacement can leave the previous ready revision active

## CLI Commands

The repo exposes a single CLI entrypoint:

- `uv run paper-context serve`
- `uv run paper-context worker`
- `uv run paper-context worker --once`
- `uv run paper-context verify-synthetic-job`

`verify-synthetic-job` is a fast queue and worker smoke check. The main product workflow is still real PDF upload plus status polling.

## Verification

Repo-standard checks:

- `uv run pre-commit run --all-files`
- `uv run pyright`
- `uv run pytest -m "unit or slice"`
- `uv run pytest -m "integration or migration" -n 2 --dist=loadfile`
- `uv run pytest -m contract`
- `uv run pytest -m "regression and not staging_only"`

## Documentation Map

- [`docs/README.md`](./docs/README.md): documentation overview
- [`docs/architecture.md`](./docs/architecture.md): runtime topology and service boundaries
- [`docs/ingestion-and-indexing.md`](./docs/ingestion-and-indexing.md): ingest lifecycle, parser fallback, revision activation, and indexing
- [`docs/data-model.md`](./docs/data-model.md): canonical schema, revisions, provenance, and retrieval assets
- [`docs/apis-and-tools.md`](./docs/apis-and-tools.md): HTTP and MCP contracts
- [`docs/retrieval.md`](./docs/retrieval.md): retrieval pipeline, budgets, warnings, and output semantics
- [`docs/evaluation-and-roadmap.md`](./docs/evaluation-and-roadmap.md): what is validated now and what comes next
- [`docs/test-strategy.md`](./docs/test-strategy.md): suite taxonomy and CI lanes

## Contributing

This is still an active project, so interfaces may move as the self-hosted runtime is hardened. Useful feedback is on ingestion correctness, provenance guarantees, retrieval quality, replacement semantics, and contract stability.
