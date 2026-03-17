FROM ghcr.io/pgmq/pg18-pgmq:v1.7.0

ARG PGVECTOR_VERSION=v0.8.1

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      git \
      postgresql-server-dev-18 \
    && git clone --branch "${PGVECTOR_VERSION}" https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector \
    && apt-get purge -y build-essential git postgresql-server-dev-18 \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

USER postgres
