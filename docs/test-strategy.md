# Paper Context Test Strategy

## Suite taxonomy

- `unit`: pure logic, parsing, configuration, storage helpers, worker control flow, and adapter translation tests.
- `slice`: in-process `FastAPI` `TestClient` coverage for lifespan, `/healthz`, `/readyz`, and MCP mounting behavior.
- `integration`: real Postgres/PGMQ seams, including queue round-trips, ingestion flow, and readiness against a migrated database.
- `contract`: `/openapi.json` stability, response-model compatibility, and golden JSON payloads for the current health endpoints.
- `regression`: deployment-shape assertions plus optional staging-only smoke coverage.
- `migration`: fresh-database Alembic upgrade smoke, layered on top of the integration lane.

## Supported commands

- `uv run pytest -m "unit or slice"`
- `uv run pytest -m "integration or migration" -n 2 --dist=loadfile`
- `uv run pytest -m contract`
- `uv run pytest -m regression`

## Postgres-backed integration tests

- `requires_postgres` tests run against a real Postgres instance with `pgmq` and `vector`.
- By default the suite starts the repo's `db` service from [docker-compose.yml](/Users/roy/Documents/rag/docker-compose.yml) if Docker is available.
- To target an already-running database, set `PAPER_CONTEXT_TEST_DATABASE_URL`.
- Each integration test gets a fresh database, and Alembic is applied programmatically before the test runs.

## CI lanes

- `static-checks`: `pre-commit` static checks, including `pyright`
- `unit-slice`: PR-blocking fast tests with branch coverage, JUnit XML, and coverage XML artifacts
- `contract`: schema and golden-response compatibility checks with JUnit XML
- `integration-postgres`: PR-blocking real Postgres/PGMQ tests, including migration smoke, with retained pytest and Docker logs
- `integration-postgres` uses `pytest-xdist` with `-n 2` and `--dist=loadfile` to reduce wall-clock time without fanning out the Postgres-backed lane too aggressively.
- `regression-smoke`: lightweight regression checks that do not require a deployed staging stack

## Staging regression

- `staging_only` tests are disabled unless `PAPER_CONTEXT_RUN_STAGING_TESTS=1`.
- Provide `PAPER_CONTEXT_STAGING_BASE_URL` to target the same-VPS non-production Dokploy stack.
- Keep that environment isolated from production:
  separate Compose project, separate domain, separate database volume and env vars, and separate artifacts path.
