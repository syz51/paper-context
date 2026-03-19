# Important

Use uv for python

## Verification

- Run the narrowest meaningful checks for the files you changed, then widen if the change crosses boundaries.
- Static checks: `uv run pre-commit run --all-files`
- Types: `uv run pyright`
- Unit and slice tests: `uv run pytest -m "unit or slice"`. Important: Don't lower coverage threshold. Implement more tests to increase coverage.
- Contract tests: `uv run pytest -m contract`
- Integration and migration tests: `uv run pytest -m "integration or migration" -n 2 --dist=loadfile`
- Regression tests: `uv run pytest -m "regression and not staging_only"`
- If you touch `alembic/`, `src/paper_context/db/`, `src/paper_context/queue/`, `src/paper_context/worker/`, or ingest lifecycle and status behavior, validate with the Postgres-backed integration lane, not only unit tests.

## Change Triggers

- Whenever you write code, make sure it's covered by tests.
- Keep schema-shape changes synchronized across SQLAlchemy models, Alembic migrations, API or Pydantic schemas, and contract golden files.
- If you change API responses or status payloads, update the golden files under `tests/contract/golden/`.
- If you change ingest stages, warnings, parser fallback behavior, or job state transitions, update both focused unit tests and Postgres-backed integration tests.

## Dependency References

Prefer official `llms.txt` endpoints when they exist. Otherwise prefer the simplest LLM-readable upstream docs entrypoint available, using versioned raw docs or README paths where practical.

### Runtime Stack

[dependency]
name: alembic
version: 1.18.4
docs: <https://alembic.sqlalchemy.org/llms.txt>
consult_when: migrations, alembic env.py, revision scripts, upgrade or downgrade behavior

[dependency]
name: docling
version: 2.80.0
docs: <https://raw.githubusercontent.com/docling-project/docling/v2.80.0/README.md>
consult_when: parser capabilities, conversion behavior, extraction regressions

[dependency]
name: fastapi
version: 0.135.1
docs: <https://raw.githubusercontent.com/fastapi/fastapi/0.135.1/docs/en/docs/index.md>
consult_when: route behavior, dependency injection, request parsing, response model handling

[dependency]
name: fastmcp
version: 3.1.1
docs: <https://gofastmcp.com/llms.txt>
consult_when: MCP server wiring, mounted transport behavior, tool registration

[dependency]
name: orjson
version: 3.11.7
docs: <https://raw.githubusercontent.com/ijl/orjson/3.11.7/README.md>
consult_when: serialization edge cases, bytes or datetime behavior, performance-sensitive JSON paths

[dependency]
name: pdfplumber
version: 0.11.9
docs: <https://raw.githubusercontent.com/jsvine/pdfplumber/v0.11.9/README.md>
consult_when: parser fallback behavior, layout extraction quirks, PDF text or table issues

[dependency]
name: pgmq
version: 1.7.0
docs: <https://raw.githubusercontent.com/pgmq/pgmq/v1.7.0/README.md>
consult_when: queue SQL functions, visibility timeout semantics, Postgres queue behavior

[dependency]
name: pgvector
version: 0.8.1
docs: <https://raw.githubusercontent.com/pgvector/pgvector/v0.8.1/README.md>
consult_when: extension DDL, vector column or index behavior, similarity operator semantics

[dependency]
name: psycopg
version: 3.3.3
docs: <https://www.psycopg.org/psycopg3/docs/>
consult_when: connection or session behavior, SQL execution, transaction handling

[dependency]
name: pydantic
version: 2.12.5
docs: <https://docs.pydantic.dev/latest/llms.txt>
consult_when: model validation, serialization, field config, model config behavior

[dependency]
name: pydantic-settings
version: 2.13.1
docs: <https://docs.pydantic.dev/latest/concepts/pydantic_settings/>
consult_when: settings loading, env parsing, config source precedence

[dependency]
name: python-multipart
version: 0.0.22
docs: <https://raw.githubusercontent.com/Kludex/python-multipart/0.0.22/docs/index.md>
consult_when: multipart upload parsing, form or file edge cases

[dependency]
name: sqlalchemy
version: 2.0.48
docs: <https://docs.sqlalchemy.org/llms.txt>
consult_when: ORM mappings, sessions, Core queries, migration-adjacent SQLAlchemy behavior

[dependency]
name: uvicorn
version: 0.42.0
docs: <https://www.uvicorn.org/llms.txt>
consult_when: ASGI serving behavior, local dev server issues, worker or process runtime quirks

### Dev And Test

[dependency]
name: bandit
version: 1.9.4
docs: <https://bandit.readthedocs.io/en/1.9.4/>
consult_when: security lint findings, rule behavior, ignore annotations

[dependency]
name: httpx
version: 0.28.1
docs: <https://www.python-httpx.org/>
consult_when: test clients, request or response API behavior, timeout or transport questions

[dependency]
name: pre-commit
version: 4.5.1
docs: <https://pre-commit.com/>
consult_when: hook execution behavior, local hook config, CI or local hook mismatches

[dependency]
name: pyright
version: 1.1.408
docs: <https://microsoft.github.io/pyright/>
consult_when: type-check diagnostics, config behavior, Python-version typing issues

[dependency]
name: pytest
version: 9.0.2
docs: <https://docs.pytest.org/en/9.0.x/>
consult_when: fixture behavior, markers, parametrization, assertion semantics

[dependency]
name: pytest-cov
version: 7.0.0
docs: <https://pytest-cov.readthedocs.io/en/latest/>
consult_when: coverage flags, reporting behavior, coverage integration quirks

[dependency]
name: pytest-xdist
version: 3.8.0
docs: <https://pytest-xdist.readthedocs.io/en/latest/>
consult_when: parallel test execution, worker isolation, -n and --dist behavior

[dependency]
name: ruff
version: 0.15.6
docs: <https://docs.astral.sh/ruff/llms.txt>
consult_when: lint rule behavior, autofix expectations, formatter or linter conflicts
