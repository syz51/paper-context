# Evaluation And Roadmap

The MVP implementation is complete enough to shift the roadmap from phase delivery to hardening, measurement, and extension. This document separates what should already be true in the current runtime from what is still next.

## What Must Be True Now

### Ingestion correctness

- born-digital PDFs can move from upload to `ready`
- degraded Docling parses trigger `pdfplumber` fallback rather than silent low-quality indexing
- failed parses do not produce active retrieval runs
- warnings survive from ingest job state into retrieval outputs

### Revision behavior

- replacement creates a new revision rather than overwriting canonical rows in place
- current reads resolve through `documents.active_revision_id`
- a failed replacement does not have to destroy the previous ready revision
- superseded jobs are marked explicitly instead of racing into ambiguous state

### Retrieval behavior

- passage queries return expected narrative evidence
- table queries return structured table hits when relevant
- context packs remain bounded and provenance-rich
- one result page or pack never mixes index versions
- exact paginated retrieval uses versioned offset cursors, search-after candidate streaming, and ranked snapshot reuse for later pages
- bounded pagination is explicit and discloses truncation when its cost ceiling binds

### Interface stability

- HTTP contracts match the response models and contract goldens
- MCP contracts match the current tool schemas and goldens
- readiness and operational probes reflect actual runtime state

## Current Validation Focus

The highest-value validation areas are:

- parser fallback behavior
- queue redelivery and lease handling
- replacement retention and active-revision promotion
- retrieval correctness for passages and tables
- warning propagation
- contract drift across HTTP and MCP

## Near-Term Hardening Work

The next tranche of work is not “build the MVP.” It is tightening the implemented runtime:

- broaden retrieval evaluation against a representative paper corpus
- improve observability around queue lag, ingest timings, and retrieval timings
- harden self-hosted deployment defaults and runbook guidance
- expand regression coverage around revision retention and replacement edge cases
- keep deep-pagination and shortlist-certification regressions covered as retrieval evolves
- tighten documentation and operator workflows for recovery and troubleshooting

## Still Intentionally Minimal

These areas are present as extension points but not fully developed product features yet:

- provider-backed metadata enrichment
- richer retrieval filtering
- more rigorous retrieval benchmarking and relevance datasets
- broader operator tooling beyond readiness and logs

## Deferred Features

These remain outside the current core:

- OCR and scanned PDFs
- answer synthesis
- query rewriting as a default path
- multi-query retrieval
- knowledge-graph or citation-graph retrieval
- agent orchestration inside the retrieval core
- LlamaIndex or PydanticAI in the core runtime

## Revisit Triggers

Only revisit deferred retrieval complexity when evaluation shows a real gap:

- add query rewriting only if terminology mismatch is a persistent retrieval failure mode
- add citation or graph retrieval only if cross-paper questions are common and not solved by current passage and table retrieval
- add downstream agent orchestration only if stable MCP contracts are already insufficient for the real workflow
- add OCR only if scanned PDFs become a meaningful part of the corpus

## Success Criteria For The Next Stage

The next stage is successful when:

- self-hosted bring-up is routine and well documented
- replacement and active-revision behavior stay regression-covered
- retrieval quality is measured against labeled or at least repeatable queries
- API and MCP contracts stay stable under continued implementation work
