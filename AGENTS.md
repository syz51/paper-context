# Important

Always use a pool of sub agents for all the tasks that could be safely delegated
Use uv for python
When receiving user input or making code changes, take notes on what choices were made, and whats working whats not working, etc, Anything you deem worth noting down. In a separate folder called memory.
When you need to look for past memory, look at the memory/ folder for past experiences.

## Verification

- Run the narrowest meaningful checks for the files you changed, then widen if the change crosses boundaries.
- Static checks: `uv run pre-commit run --all-files`
- Types: `uv run pyright`
- Unit and slice tests: `uv run pytest -m "unit or slice"`
- Contract tests: `uv run pytest -m contract`
- Integration and migration tests: `uv run pytest -m "integration or migration" -n 2 --dist=loadfile`
- Regression tests: `uv run pytest -m "regression and not staging_only"`
- If you touch `alembic/`, `src/paper_context/db/`, `src/paper_context/queue/`, `src/paper_context/worker/`, or ingest lifecycle and status behavior, validate with the Postgres-backed integration lane, not only unit tests.

## Change Triggers

- Keep schema-shape changes synchronized across SQLAlchemy models, Alembic migrations, API or Pydantic schemas, and contract golden files.
- If you change API responses or status payloads, update the golden files under `tests/contract/golden/`.
- If you change ingest stages, warnings, parser fallback behavior, or job state transitions, update both focused unit tests and Postgres-backed integration tests.

## Dependency References

Prefer these pinned upstream paths over generic homepages. Where upstream docs are not versioned cleanly, the pinned path is the exact release or tag page for the version used here.

### Runtime Stack

| Dependency | Version used here | Pinned upstream doc path | Consult when |
| --- | --- | --- | --- |
| `alembic` | `1.18.4` | <https://github.com/sqlalchemy/alembic/releases/tag/rel_1_18_4> | migrations, `env.py`, revision scripts, downgrade/upgrade behavior |
| `docling` | `2.80.0` | <https://github.com/docling-project/docling/releases/tag/v2.80.0> | parser capabilities, conversion behavior, extraction regressions |
| `fastapi` | `0.135.1` | <https://github.com/fastapi/fastapi/releases/tag/0.135.1> | route behavior, dependency injection, request/response model handling |
| `fastmcp` | `3.1.1` | <https://github.com/PrefectHQ/fastmcp/releases/tag/v3.1.1> | MCP server wiring, mounted transport behavior, tool registration |
| `orjson` | `3.11.7` | <https://github.com/ijl/orjson/releases/tag/3.11.7> | serialization edge cases, bytes/datetime behavior, performance-sensitive JSON paths |
| `pdfplumber` | `0.11.9` | <https://github.com/jsvine/pdfplumber/releases/tag/v0.11.9> | parser fallback behavior, layout extraction quirks, PDF text/table issues |
| `pgmq` | `1.7.0` | <https://github.com/pgmq/pgmq/releases/tag/v1.7.0> | queue SQL functions, visibility timeout semantics, Postgres queue behavior |
| `pgvector` | `0.8.1` | <https://github.com/pgvector/pgvector/tree/v0.8.1> | extension DDL, vector column/index behavior, similarity operator semantics |
| `psycopg` | `3.3.3` | <https://github.com/psycopg/psycopg/releases/tag/3.3.3> | connection/session behavior, SQL execution, transaction handling |
| `pydantic` | `2.12.5` | <https://docs.pydantic.dev/2.12/> | model validation, serialization, settings/model config behavior |
| `pydantic-settings` | `2.13.1` | <https://github.com/pydantic/pydantic-settings/releases/tag/v2.13.1> | settings loading, env parsing, config source precedence |
| `python-multipart` | `0.0.22` | <https://github.com/Kludex/python-multipart/releases/tag/0.0.22> | multipart upload parsing, form/file edge cases |
| `sqlalchemy` | `2.0.48` | <https://docs.sqlalchemy.org/en/20/> | ORM mappings, sessions, Core queries, migration-adjacent SQLAlchemy behavior |
| `uvicorn` | `0.42.0` | <https://github.com/Kludex/uvicorn/releases/tag/0.42.0> | ASGI serving behavior, local dev server issues, worker/process runtime quirks |

### Dev And Test

| Dependency | Version used here | Pinned upstream doc path | Consult when |
| --- | --- | --- | --- |
| `bandit` | `1.9.4` | <https://bandit.readthedocs.io/en/1.9.4/> | security lint findings, rule behavior, ignore annotations |
| `httpx` | `0.28.1` | <https://github.com/encode/httpx/releases/tag/0.28.1> | test clients, request/response API behavior, timeout/transport questions |
| `pre-commit` | `4.5.1` | <https://github.com/pre-commit/pre-commit/releases/tag/v4.5.1> | hook execution behavior, local hook config, CI/local hook mismatches |
| `pyright` | `1.1.408` | <https://github.com/RobertCraigie/pyright-python/releases/tag/v1.1.408> | type-check diagnostics, config behavior, Python-version typing issues |
| `pytest` | `9.0.2` | <https://docs.pytest.org/en/9.0.x/> | fixture behavior, markers, parametrization, assertion semantics |
| `pytest-cov` | `7.0.0` | <https://github.com/pytest-dev/pytest-cov/releases/tag/v7.0.0> | coverage flags, reporting behavior, coverage integration quirks |
| `pytest-xdist` | `3.8.0` | <https://github.com/pytest-dev/pytest-xdist/releases/tag/v3.8.0> | parallel test execution, worker isolation, `-n` and `--dist` behavior |
| `ruff` | `0.15.6` | <https://github.com/astral-sh/ruff/releases/tag/0.15.6> | lint rule behavior, autofix expectations, formatter/linter conflicts |
