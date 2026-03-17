FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv>=0.9.17,<0.10"

COPY pyproject.toml README.md alembic.ini ./
COPY alembic ./alembic
COPY src ./src

RUN uv pip install --system -e .

EXPOSE 8000 8001
