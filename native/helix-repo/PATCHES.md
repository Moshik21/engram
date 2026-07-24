# Local patches to the vendored helix-db source

This checkout is the BUNDLED source used by native builds
(`helix-python` → `maturin develop --release` into `server/.venv`).
Deviations from upstream that are candidates for upstreaming are logged here.

## 1. BM25 doc insert is an idempotent upsert (2026-07-21)

Files: `helix-db/src/helix_engine/bm25/bm25.rs`
(`insert_doc_for_node`, `insert_doc`, new `purge_stale_doc`),
tests in `helix-db/src/helix_engine/bm25/bm25_tests.rs`.

BM25 documents are keyed by the internal node id
(`traversal_core/ops/source/add_n.rs`: `id = stable_node_id(...)` →
`insert_doc_for_node(txn, node.id, ...)`). Node ids are deterministic
functions of the business key (see the earlier stable-id patch in
`utils/id.rs` / `add_n.rs`), so a delete + re-create of the same key lands on
the SAME id. Historically `drop_traversal` swallowed `bm25.delete_doc`
failures (patch #2 below), leaving orphan docs behind — after which EVERY
re-create of that key failed with `BM25 document {id} already exists`,
forever (the Engram "bootstrap-500" defect class).

Fix: `insert_doc_for_node` / `insert_doc` no longer error on a pre-existing
doc. A new `purge_stale_doc` best-effort-purges whatever state exists at the
id (tolerating partially-deleted shapes: missing postings / df rows /
metadata are skipped, not errors), logs one loud `eprintln!` line, then the
insert proceeds fresh. Rationale: the inserting node legitimately owns the
id; any prior doc content at that id is garbage from a deleted node. This
also completes the intended idempotent re-ingest semantics already noted in
`add_n.rs` (nodes_db uses a plain overwriting `put`; BM25 was the only layer
still erroring).

Not covered: dangling inverted-index postings under terms absent from the
doc's reverse entries can't be purged without a full index scan — they
re-bind to the new doc content (harmless tf skew) and are the job of the
Engram index-consistency drain (M0.2).

## 2. drop_traversal propagates BM25 delete failures (2026-07-21)

File: `helix-db/src/helix_engine/traversal_core/ops/util/drop.rs`.

Was: `println!("failed to delete doc from bm25: {e}")` and continue — the
silent swallow that created the orphan docs behind patch #1. Now the error
propagates as `GraphError`, aborting the transaction so graph and index
never diverge. The per-node `println!("Dropped node: ...")` stdout spam was
demoted to `debug_println!`.

Known remaining swallow (unchanged, noting for upstream): line ~24
`.filter_map(|item| item.ok())` silently discards erroring traversal items
before the drop loop.

## Earlier undocumented local patches (context, pre-dating this file)

- Deterministic node ids from business keys (`utils/id.rs::stable_node_id`,
  used in `add_n.rs` / `upsert.rs`), replacing pure `v6_uuid`.
- Per-label BM25 field filters (`HBM25Config::bm25_field_filters` /
  `register_bm25_fields`; `Config` gained a `bm25_field_filters` field the
  helix CLI's generated config must be patched to include — see
  REGEN_PROCEDURE.md step 3).
- `add_n.rs` / `upsert.rs`: plain LMDB `put` instead of `APPEND`/`APPEND_DUP`
  (deterministic ids are non-monotonic).
