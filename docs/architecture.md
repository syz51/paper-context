# Architecture

Paper Context is implemented as a self-hosted, revision-aware ingestion and retrieval system. The architecture is intentionally narrow: one FastAPI app, one worker, one Postgres instance with `pgvector` and `pgmq`, one local artifact store, and one MCP surface mounted inside the app.

## Runtime Topology

Local development uses four processes:

- `db`: Postgres with the `vector` and `pgmq` extensions
- `migrate`: one-shot Alembic runner
- `app`: FastAPI service for HTTP endpoints and MCP mounting
- `worker`: background ingestion and indexing loop

Production Compose is narrower:

- `migrate`, `app`, and `worker` run in `docker-compose.prod.yml`
- Postgres is expected to exist outside this Compose file on `dokploy-network`
- MCP is still not a separate process; FastMCP remains mounted by the FastAPI service at `/mcp`

## Service Responsibilities

### `app`

Owns:

- upload and replacement request validation
- source PDF staging
- `documents` and `ingest_jobs` creation
- document inspection routes
- `healthz` and `readyz`
- mounted MCP transport

It does not perform parsing or indexing inline.

### `worker`

Owns:

- queue claim, lease extension, archive, and redelivery behavior
- advisory-lock based single-job processing
- parser execution and structure gating
- parser subprocess isolation when `PAPER_CONTEXT_PARSER__EXECUTION_MODE=subprocess`
- canonical normalization
- chunking and contextualized retrieval text generation
- retrieval asset materialization
- revision activation on success and rollback-on-failure behavior

### `db`

Owns:

- canonical records
- revision state
- ingest-job lifecycle state
- retrieval index runs and retrieval assets
- queue dispatch through PGMQ

### local artifact storage

Owns:

- source PDFs
- parser-produced artifacts

Artifacts are referenced from Postgres through `document_artifacts.storage_ref`.

## Core Architectural Decision

The stable paper identity is `documents.id`. Uploads and replacements create new `document_revisions` rows instead of overwriting canonical state in place.

That gives the runtime these properties:

- current reads resolve through `documents.active_revision_id`
- replacement can be staged and indexed before activation
- failed replacements do not need to destroy the previous ready state
- retrieval results stay tied to the specific revision and `retrieval_index_run_id` that produced them

## End-To-End Flow

1. A client uploads a PDF through `POST /documents` or replaces an existing one through `POST /documents/{document_id}/replace`.
2. The app stages the source PDF, creates or reuses the stable document row, creates a new revision, creates an ingest job, and enqueues minimal queue payload metadata.
3. The worker claims the queue message, locks the ingest job and revision, and skips work if the job is already terminal or superseded.
4. The worker parses with Docling first and falls back to `pdfplumber` only when the primary parse is structurally degraded.
5. The worker writes canonical sections, tables, references, passages, and parser artifacts for the revision.
6. The worker creates contextualized retrieval text, builds passage and table retrieval assets, records a `retrieval_index_runs` row, and marks that run active for the revision.
7. On success, the worker updates `documents.active_revision_id` to the new revision. On failure, the previous ready revision can remain active.
8. FastAPI and MCP reads return only the active revision for a document, with provenance and warnings preserved.

## Retrieval Boundary

The deterministic core includes:

- parsing and fallback policy
- normalization
- chunking policy
- sparse and dense retrieval
- fusion and reranking
- parent expansion
- context-pack assembly
- warnings and provenance fields

Future agent logic is downstream of this boundary. Agents can use MCP tools, but they should not replace retrieval logic implemented in the service layer.

## Current Stack Choices

- **FastAPI**
  - operational HTTP surface
  - app lifecycle, storage root creation, and MCP mounting
- **FastMCP**
  - curated tool surface mounted at `/mcp`
  - no auto-generated tool surface from FastAPI routes
- **Postgres + `pgvector` + `pgmq`**
  - canonical records, queue dispatch, filters, vectors, and search assets in one store
- **Docling + `pdfplumber`**
  - primary parser plus deterministic fallback
  - subprocess isolation is the default runtime mode, with timeout, memory, and output limits
- **Voyage `voyage-4-large`**
  - default dense embedding model
- **Zero Entropy `zerank-2`**
  - default reranker model

When provider API keys are missing, the runtime uses deterministic embedding and heuristic reranking implementations so the system remains operable locally.

## Production Runtime Guards

When `PAPER_CONTEXT_ENVIRONMENT=production`, runtime settings enforce stricter database configuration:

- secure SSL mode
- explicit application name
- non-null connection timeout, statement timeout, lock timeout, idle transaction timeout, and pool settings

## Out Of Scope

The current architecture does not include:

- OCR or scanned-PDF support
- answer synthesis
- a separate orchestration or agent runtime
- LlamaIndex or PydanticAI in the core ingestion or retrieval path
