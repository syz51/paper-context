# Ingestion And Indexing

The ingestion pipeline is implemented and revision-aware. The worker always builds canonical document structure first, then derived retrieval assets. Upload and replacement share the same pipeline; the main difference is whether a new stable document row is created.

## Upload And Replace Flow

`POST /documents`:

- validates the upload as a non-empty PDF
- enforces the configured upload limit
- stages the source file into local artifact storage
- creates `documents`, `document_revisions`, `ingest_jobs`, and `document_artifacts` rows
- enqueues a minimal PGMQ payload containing document and job identifiers

`POST /documents/{document_id}/replace`:

- reuses the stable `documents.id`
- creates a new revision and ingest job
- stages a new source artifact for that revision
- supersedes older queued jobs for the same document when appropriate

The document stays pointed at its previous active revision until the replacement finishes successfully.

## Ingest Job Lifecycle

The worker advances jobs through these statuses:

1. `queued`
2. `parsing`
3. `normalizing`
4. `enriching_metadata`
5. `chunking`
6. `indexing`
7. `ready` or `failed`

Each job also records:

- `failure_code`
- `failure_message`
- `warnings`
- `stage_timings`
- `trigger`
- `started_at`
- `finished_at`

Warnings are part of the durable job state, not ephemeral logs.

## Queue And Claim Semantics

The queue payload is intentionally small. `ingest_job_id` is the authoritative lookup key.

Worker behavior:

- claims one message through PGMQ
- locks the matching ingest job and revision
- extends the lease during long-running stages
- archives completed messages
- skips duplicate work for terminal jobs
- fails superseded jobs explicitly

If a newer job exists for the same document, an older queued or in-flight job can be marked failed with `superseded_by_newer_ingest_job`.

## Parser Flow

### Primary path

The worker runs Docling first. If the parse is structurally usable, the Docling artifact becomes the primary parser artifact for the revision.

### Structure gate

The parser output is classified as:

- `pass`
- `degraded`
- `fail`

`pass` continues directly into normalization. `degraded` triggers fallback. `fail` ends the job with explicit failure metadata.

### Fallback path

When Docling returns `degraded`, the worker retries with `pdfplumber`.

If fallback succeeds:

- the fallback artifact is stored
- warnings are merged into the job
- `parser_fallback_used` is added
- downstream retrieval results preserve the fallback warning state

If fallback cannot recover stable structure and provenance, the job is failed rather than downgraded into blob-text indexing.

## Metadata Handling

Base metadata comes from the parsed document:

- title
- authors
- abstract
- publication year
- metadata confidence

The current runtime wires `NullMetadataEnricher`, so enrichment is effectively a no-op by default. The lifecycle still includes `enriching_metadata` because the pipeline is already structured for later provider-backed enrichment without needing to redesign job flow or contracts.

If parsed metadata confidence is low, the worker adds `metadata_low_confidence`.

## Normalization

Canonical rows are written per revision:

1. document metadata
2. `document_sections`
3. `document_tables`
4. `document_references`
5. `document_passages`
6. parser artifacts and retrieval index metadata

Normalization rules:

- sections preserve hierarchy and page spans
- tables are first-class objects, not flattened into prose
- passages are section-bounded chunks with canonical `body_text`
- every derived row links back to the parser artifact that produced it

On reprocessing the same revision, the worker resets prior non-source rows for that revision before rebuilding them.

## Chunking

Chunking happens after normalization and before retrieval indexing.

Current defaults:

- section-bounded chunks
- `min_tokens = 300`
- `max_tokens = 700`
- `overlap_fraction = 0.15`

Each passage stores:

- canonical `body_text`
- derived `contextualized_text`
- page provenance
- chunk ordinal

The contextualized text is deterministic and includes document and section context so sparse and dense retrieval operate over the same enriched representation.

## Retrieval Indexing

The worker writes one `retrieval_index_runs` row for each indexing build. That row records:

- `index_version`
- embedding provider and model
- embedding dimensions
- reranker provider and model
- `chunking_version`
- `parser_source`
- build `status`
- activation timestamps

The indexing step then materializes:

- `retrieval_passage_assets`
- `retrieval_table_assets`

Both asset sets are keyed to the revision and run that produced them.

## Revision Activation

Successful indexing does not just mark the ingest job `ready`. It also:

- marks the run active for that revision
- updates the revision status to `ready`
- updates `documents.active_revision_id`
- copies active revision metadata back onto the top-level document row

If a replacement fails and there is an older ready revision, the document can remain pointed at that earlier revision. This is the core reason replacement is documented as revision-safe rather than destructive.

## Failure Behavior

The worker fails the job with explicit machine-readable codes for cases such as:

- missing source artifact
- invalid artifact reference
- Docling structure failure
- fallback structure failure
- normalization failure
- indexing failure
- superseded ingest job

Failure preserves already-written artifacts when useful for debugging and keeps warnings attached to the job.

## Non-Goals

The current ingestion pipeline does not attempt:

- OCR or scanned-PDF recovery
- free-text indexing without provenance
- multimodal ingestion
- framework-managed ingestion abstractions
