# Issue 1 Resolution Plan

This plan treats commit `fbb4674` as the current baseline.

## Goal

Fully resolve issue `#1` so that:

1. exact paginated retrieval always certifies a fused shortlist before rerank
2. exact paginated retrieval no longer relies on rerank-until-stable loops on any supported cursor path
3. candidate expansion advances incrementally without reloading wider prefixes from scratch
4. repeated page traversal for the same request fingerprint can reuse a ranked snapshot
5. callers that need a hard cost ceiling use an explicit bounded mode with visible truncation semantics

## Current Status

Implemented already:

1. exact page retrieval uses shortlist certification and single-pass rerank for new offset cursors in `src/paper_context/retrieval/service.py`
2. candidate expansion state is maintained incrementally across sparse and dense rounds
3. page cursors now carry absolute offsets

Still missing or incomplete:

1. legacy score/entity cursors still route through the old rerank-until-stable widening loop
2. there is no ranked snapshot cache for later pages of the same request chain
3. there is no explicit bounded or approximate pagination mode
4. current incremental expansion uses SQL `OFFSET`; the service no longer refetches wider prefixes into Python, but it still re-executes ranked queries rather than true search-after streaming
5. end-to-end validation does not yet cover all acceptance criteria from issue `#1`

## Required End State

Issue `#1` is considered fully resolved only when all of the following are true:

1. the default exact path never uses repeated rerank-until-stable logic
2. the default exact path certifies the shortlist before rerank
3. the default exact path reranks only the certified shortlist
4. candidate expansion reuses prior state and advances by stream anchor rather than restart-from-zero widening
5. repeated pages for the same request fingerprint can reuse a valid ranked snapshot
6. bounded mode is explicit in the API and MCP contracts and exposes truncation clearly
7. docs, unit tests, contract tests, and Postgres-backed integration tests all reflect the final behavior

## Work Plan

### 1. Remove the legacy exact-path fallback

Objective:
Eliminate the old rerank-until-stable path from supported exact pagination behavior.

Changes:

1. add an explicit cursor version field for exact offset cursors
2. stop routing legacy cursors through `_search_passages_page_legacy_with_connection()` and `_search_tables_page_legacy_with_connection()`
3. choose one migration policy and document it:
4. reject legacy cursors with `RetrievalError("cursor is no longer supported")`
5. or translate legacy cursors only when a valid ranked snapshot exists
6. remove `_search_passages_page_legacy_with_connection()` and `_search_tables_page_legacy_with_connection()` after migration

Files:

1. `src/paper_context/retrieval/service.py`
2. `tests/unit/test_retrieval_service_phase3.py`
3. `tests/integration/test_retrieval_service_integration.py`
4. `docs/retrieval.md`

Done when:

1. no exact request path can reach the rerank-until-stable loop
2. legacy cursor behavior is either unsupported or losslessly translated by an explicit documented rule

### 2. Finish incremental candidate streaming

Objective:
Advance sparse and dense streams without restart-from-zero query widening.

Changes:

1. replace `OFFSET`-based expansion with stream anchors or search-after helpers
2. keep a per-stream anchor that is stable under the stream’s ordering:
3. sparse passages: `(rank_score, passage_id)`
4. dense passages: `(distance or dense_score, passage_id)`
5. sparse tables: `(rank_score, table_id)`
6. dense tables: `(distance or dense_score, table_id)`
7. add loader variants that accept `after_*` anchors instead of `offset`
8. keep `_CandidateExpansionState` but extend it with stream anchors and per-round additions
9. preserve deterministic ordering and mixed-index safeguards
10. add metrics for added candidates per round, sparse rounds, dense rounds, and certification stop reason

Files:

1. `src/paper_context/retrieval/service.py`
2. `docs/retrieval.md`
3. `tests/unit/test_retrieval_service_helpers_extra.py`
4. `tests/integration/test_retrieval_service_integration.py`

Done when:

1. the service no longer depends on SQL `OFFSET` for exact candidate expansion
2. expansion only requests unseen candidates per stream
3. certification math still proves exact semantics

### 3. Add ranked snapshot caching for repeated pages

Objective:
Avoid recomputing retrieval and rerank on every later page request for the same fingerprint.

Changes:

1. define a snapshot key with:
2. request fingerprint
3. active index version
4. entity kind
5. exact or bounded mode
6. page-size-independent retrieval parameters
7. define snapshot payload with:
8. ordered entity ids
9. reranked scores
10. retrieval modes
11. retrieval index run ids and index version
12. created-at timestamp and expiry
13. add an in-memory TTL cache first; keep the cache behind a narrow interface so it can be swapped later
14. populate the snapshot after the first exact page request completes
15. reuse the snapshot for later exact pages while the key matches and the snapshot is still valid
16. invalidate on active index version change or TTL expiry
17. expose cache hit and miss metrics

Files:

1. `src/paper_context/retrieval/service.py`
2. `src/paper_context/retrieval/types.py` if new service-owned response metadata is needed
3. `docs/retrieval.md`
4. `docs/evaluation-and-roadmap.md`
5. `tests/unit/test_retrieval_service_phase3.py`
6. `tests/integration/test_retrieval_service_integration.py`

Done when:

1. page 2+ of the same exact request chain can be served from cached ranked state
2. snapshot reuse does not weaken exactness or mix index versions

### 4. Add explicit bounded mode

Objective:
Provide a hard-cost option without weakening the exact default path silently.

Changes:

1. define one bounded-mode contract and use it consistently across HTTP and MCP
2. preferred shape:
3. `pagination_mode="exact" | "bounded"`
4. bounded-mode controls:
5. `max_rerank_candidates`
6. `max_expansion_rounds`
7. response metadata:
8. `exact`
9. `truncated`
10. warnings such as `bounded_pagination_truncated`
11. make the default remain `exact`
12. ensure bounded mode never reuses an exact snapshot and vice versa
13. update contract goldens if any API or MCP payload changes

Files:

1. `src/paper_context/retrieval/service.py`
2. `src/paper_context/retrieval/types.py`
3. API and MCP route/schema files if request or response contracts change
4. `tests/contract/golden/`
5. `tests/contract/`
6. `docs/retrieval.md`
7. `docs/apis-and-tools.md`

Done when:

1. callers can opt into bounded mode explicitly
2. bounded responses always disclose truncation
3. exact mode remains unchanged and exact by default

### 5. Expand validation to cover the real failure modes

Objective:
Prove the new implementation matches the issue acceptance criteria.

Add unit coverage for:

1. shortlist certification when a later sparse candidate can still enter
2. shortlist certification when a later dense candidate can still enter
3. anchor advancement and dedupe across sparse and dense streams
4. snapshot cache hit, miss, expiry, and invalidation
5. bounded-mode truncation and warning behavior
6. legacy cursor rejection or translation, whichever migration policy is chosen

Add Postgres-backed integration coverage for:

1. passages beyond the initial sparse candidate cap
2. tables beyond the initial sparse candidate cap
3. a late sparse candidate entering the certified shortlist and changing page membership
4. a late dense candidate entering the certified shortlist and changing page membership
5. certification stopping before stream exhaustion
6. repeated page traversal that reuses a valid snapshot
7. bounded mode returning explicit truncation metadata
8. no exact path using the old rerank-until-stable loop

Run and keep green:

1. `uv run pre-commit run --all-files`
2. `uv run pyright`
3. `uv run pytest -m "unit or slice"`
4. `uv run pytest -m contract`
5. `uv run pytest tests/integration/test_retrieval_service_integration.py -q`

### 6. Update documentation and rollout notes

Objective:
Make the final behavior explicit for operators and callers.

Update:

1. `docs/retrieval.md`
2. `docs/evaluation-and-roadmap.md`
3. `docs/apis-and-tools.md`
4. any API or MCP docs that mention cursor or pagination behavior

Document clearly:

1. exact-mode semantics
2. bounded-mode semantics
3. cursor versioning and migration behavior
4. snapshot cache behavior and invalidation rules
5. why exact mode remains the default

## Recommended Delivery Order

1. remove legacy fallback and define cursor migration policy
2. replace `OFFSET` streaming with anchor-based search-after expansion
3. add snapshot cache for exact repeated paging
4. add explicit bounded mode
5. widen validation and update docs

## Acceptance Checklist

1. no supported exact cursor path uses rerank-until-stable
2. exact path certifies before rerank
3. exact path reranks only the certified shortlist
4. stream expansion advances from anchors, not `OFFSET`
5. snapshot cache serves later exact pages when valid
6. bounded mode is explicit and warns on truncation
7. docs and contract goldens are updated
8. unit, contract, and Postgres-backed integration tests are green

## Unresolved Questions

1. reject legacy cursors or translate via snapshot
2. in-memory snapshot cache only, or pluggable backend now
3. bounded mode fields: only `pagination_mode`, or also candidate caps in public API
