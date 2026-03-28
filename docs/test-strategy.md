# Paper Context Test Strategy

The test suite is organized around behavior boundaries: pure logic, app slices, real Postgres seams, contract stability, migrations, and deployment-shape regression checks.

## Suite Taxonomy

- `unit`: isolated logic for parsing, queue helpers, ingestion helpers, retrieval helpers, CLI behavior, config, and observability
- `slice`: in-process FastAPI and MCP app coverage, including lifespan, routing, health, readiness, and mounted MCP behavior
- `integration`: real Postgres and PGMQ tests for queue flow, ingestion, retrieval, readiness, and MCP integration
- `contract`: OpenAPI checks plus golden payload compatibility for HTTP and MCP contracts
- `migration`: fresh-database Alembic upgrade smoke
- `regression`: deployment-shape and smoke checks, including optional staging-only coverage

## Standard Commands

Local commands:

- `uv run pre-commit run --all-files`
- `uv run pyright`
- `uv run pytest -m "unit or slice"`
- `uv run pytest -m "integration or migration" -n 2 --dist=loadfile`
- `uv run pytest -m contract`
- `uv run pytest -m "regression and not staging_only"`

These are the narrowest repo-standard commands for local verification. CI uses the same lane names, but some lanes add artifact capture or coverage flags.

## What Contract Tests Cover

Contract tests currently check more than health endpoints. They include:

- `/openapi.json` compatibility
- document upload, list, detail, outline, tables, replace, and ingest-job payloads
- `healthz` and `readyz`
- MCP tool payloads for:
  - `search_documents`
  - `search_passages`
  - `search_tables`
  - `get_document_outline`
  - `get_table`
  - `get_passage_context`
  - `build_context_pack`

Golden files live under `tests/contract/golden/`.

Those goldens are the compatibility backstop for the documented HTTP and MCP payloads.

## Postgres-Backed Integration Coverage

Integration tests are the main protection for behavior that unit tests cannot prove alone:

- PGMQ enqueue, claim, lease extension, archive, and redelivery
- end-to-end ingestion flow
- parser fallback behavior
- revision retention and replacement activation
- retrieval against real schema objects and vector/search assets
- MCP behavior against the integrated app stack

`requires_postgres` tests run against a real Postgres instance with `pgmq` and `vector`.

By default the suite can use the repo’s Compose `db` service. To point tests at an existing database, set `PAPER_CONTEXT_TEST_DATABASE_URL`.

## Current High-Value Regression Areas

Tests should remain especially strong around:

- active-revision reads after replacement
- failure rollback that preserves the previous ready revision
- queue redelivery without duplicate canonical state
- index-version isolation in result pages and context packs
- warning propagation from ingest into retrieval

## CI Lanes

- `static-checks`: pre-commit and type checking
- `unit-slice`: fast PR-blocking logic and app tests with coverage reporting in CI
- `contract`: schema and golden-response compatibility
- `integration-postgres`: real Postgres and migration coverage
- `regression-smoke`: deployment and non-staging regression checks

The Postgres lane uses `pytest-xdist` with `-n 2 --dist=loadfile` to keep runtime reasonable without over-parallelizing shared infrastructure.

## Staging Regression

`staging_only` tests stay disabled unless `PAPER_CONTEXT_RUN_STAGING_TESTS=1`.

When enabled, provide `PAPER_CONTEXT_STAGING_BASE_URL` and keep the target isolated from production:

- separate Compose project
- separate domain
- separate database volume
- separate env vars
- separate artifacts path
