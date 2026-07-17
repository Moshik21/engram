# Silent-Inert Hardening — retire the bug class

**Status:** EXECUTED 2026-07-17 (founder approved early execution over the
interlock; changes surface failures, they do not alter success-path behavior)
**Created:** 2026-07-16 (follow-up from NEXT_LEVEL_OBJECTIVE execution)
**Owner:** founder + coding agents
**Landed:** commits bba5228, 4e48798, 897472f (+ 6d3799f PyPI-fallback defusal)

---

## Why this doc exists

Engram's most persistent bug class is **computed-but-silently-inert**: a
native-layer failure is swallowed into an empty result (`[]`, `None`, no-op)
that upstream code meters as success. Confirmed instances to date — **eight**:

1. Embeddings computed but never written (the original 23%→56% eval jump).
2. Graph write-path phantom (writes reported ok, nothing persisted).
3. Seed discard in retrieval.
4. Dead structural weight (graph embeddings scored but zeroed).
5. Live Gemini 429s metered as empty results.
6. `update_episode` dropping ENTIRE writes when `error=None` (null String
   param rejected without raising) — broke cue promotion silently for weeks.
7. Storage count snapshots pinned stale forever (timed-out scan cancelled,
   background refresh machinery had zero callers).
8. Type-only entity listings (`find_entities_by_type`) returning `[]` on the
   17GB brain (label-scan timeout swallowed) — broke the organic gate's
   Decision listing on first contact.

Every instance had the same anatomy: **an `except`/timeout path in the native
storage layer that returns an empty value instead of surfacing the failure.**
Fixing instances one at a time has cost multiple audits; this doc retires the
species.

## Objective (verifiable)

> No storage-layer failure can masquerade as an empty success. Every
> swallowed exception either surfaces to the caller, degrades with an
> explicit marker the caller must acknowledge, or is individually
> allowlisted with a written justification.

**Definition of done:**

- [x] `NativeTransport._query` distinguishes empty-result from
      failed/timed-out queries: failures raise (or return a typed
      `QueryFailure`), never bare `[]`. Callers updated to handle it.
      — `NativeQueryError` (endpoint, cause, elapsed, timeout flag); engine
      error-JSON strings raise instead of parsing as empty (instance #6);
      NoValue/NotFound/missing-hnsw stay `[]`; dim-mismatch tolerated but
      counted. New `helix.query_timeout_seconds` (20s) bounds every call.
      9 new tests in `test_native_transport_hardening.py`.
- [x] Every `except Exception: return []` / `return None` / `pass` in
      `server/engram/storage/` is either removed, converted to
      raise/degrade-with-marker, or tagged `# silent-ok: <reason>` with a
      one-line justification.
      — 108 sites across 20 files: 6 upgraded to raise (conversation delete,
      cue indexing, entity-tag resolution, HTTP connect, legacy query path),
      the rest individually justified (probe ladders, stored-field parse
      tolerance, best-effort cascade cleanup, batch item tolerance).
- [x] A static contract test (like the group-scope one) fails CI on any NEW
      untagged silent-swallow pattern in `storage/`.
      — `test_storage_silent_swallow_contract.py` (AST walk; marker on the
      except line, the line above, or in the body).
- [x] Query timeouts are observable: per-query timeout counters exposed on
      `/api/storage` diagnostics (`queryTimeouts` by query name), so the next
      "returns nothing on the big brain" is visible in the dashboard instead
      of discovered by a failing product feature.
      — Shipped as `diagnostics.queryFailures` {endpoint: {errors, timeouts,
      dim_mismatch, batch_item_errors}} + StatsPanel tile (renders only when
      non-zero). Verified live on the restarted shell 2026-07-17.
- [~] `find_entities_by_type` works on the live brain (either via native
      query timeout raise + caller fallback, or a typed secondary index) —
      the organic gate's probe fallback becomes a fallback, not the path.
      — First variant landed: the timeout now raises and the API listing
      returns explicit `status: timeout` instead of an empty-200; the organic
      gate's indexed name-probe remains the working path on the 17GB brain.
      The typed secondary index (label pushdown / entity_type index) is the
      remaining half — tracked in the refactor list, native-lane effort.
- [x] Full lite suite green; one mop window on the live brain green.
      — 4480 passed; remaining 61 fails/errors are all the pre-existing
      Docker-HTTP-lane (:6969 hardcoded fixtures) + native_surface_parity
      environment set, individually verified untouched by this change.
      Live mop 2026-07-17 (budget 100, 600s deadline, --force over the
      battery gate): all drains green, errors 0, evidence adjudication 138
      processed / edge 19/19 / replay 50 / prune 100; shell paused and
      resumed healthy in 55s. Bonus catch: the loud failure path exposed
      that the local FastEmbed model had been a broken partial download
      since Jul 13 (110-byte LFS pointer instead of weights) — repaired by
      cache reset + re-download the same day.

## Approach (suggested, 2–4 focused sessions)

1. **Inventory** (mechanical): grep `storage/` for swallow patterns; classify
   each as surface / degrade / allowlist. Expect ~30–60 sites.
2. **Transport first**: make `_query` failure-aware (typed result or raise
   with query name + elapsed). This single change covers most sites.
3. **Callers**: for read paths, degrade with explicit markers (the
   diagnostics `countsStatus` pattern is the house style); for write paths,
   failures must raise — a dropped write is never acceptable (instance #6).
4. **Static contract test** so the class cannot regrow.
5. **Timeout observability** in diagnostics + dashboard.

## Non-goals

- Native query performance work (label-scan indexes) beyond what the
  `find_entities_by_type` row requires — that is its own effort.
- Touching the public MCP surface or recall semantics.

## Interlock

Run AFTER the current measurement windows have a few clean days — this
touches the storage layer under the live brain, so land it behind the same
per-stack verify discipline (targeted tests green per commit, mop window
observed after deploy).
