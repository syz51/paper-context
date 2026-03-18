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

## Reference URLs

### Runtime Stack

- `alembic`: <https://alembic.sqlalchemy.org/>
- `docling`: <https://docling-project.github.io/docling/>
- `fastapi`: <https://fastapi.tiangolo.com/>
- `fastmcp`: <https://gofastmcp.com/llms.txt>
- `orjson`: <https://github.com/ijl/orjson>
- `pdfplumber`: <https://github.com/jsvine/pdfplumber>
- `pgmq`: <https://pgmq.github.io/pgmq/>
- `pgvector`: <https://github.com/pgvector/pgvector>
- `psycopg`: <https://www.psycopg.org/psycopg3/docs/>
- `pydantic`: <https://docs.pydantic.dev/latest/llms.txt>
- `pydantic-settings`: <https://docs.pydantic.dev/latest/llms.txt>
- `python-multipart`: <https://multipart.fastapiexpert.com/>
- `sqlalchemy`: <https://docs.sqlalchemy.org/>
- `uvicorn`: <https://www.uvicorn.org/llms.txt>

### Dev And Test

- `bandit`: <https://bandit.readthedocs.io/>
- `httpx`: <https://www.python-httpx.org/>
- `pre-commit`: <https://github.com/pre-commit/pre-commit>
- `pyright`: <https://github.com/microsoft/pyright>
- `pytest`: <https://docs.pytest.org/>
- `pytest-cov`: <https://pytest-cov.readthedocs.io/>
- `pytest-xdist`: <https://pytest-xdist.readthedocs.io/>
- `ruff`: <https://docs.astral.sh/ruff/>
