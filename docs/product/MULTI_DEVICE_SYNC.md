# Multi-device / multi-brain sync (design only)

**Status:** design — **do not implement** until Phase A–C are boring for ≥14 days
(continuity CI green on main, native path dogfooded, public surface frozen).

**Related vision:** `docs/vision/one-brain-per-person.md`,
`docs/vision/federated-policy-intelligence.md` (Phase E item 16 — vision only).

---

## Problem

Users run Engram on laptop + desktop (and later phone). Today each install owns
a local native data-dir (or lite SQLite). Continuity breaks when Decision
promotions live on only one device.

## Non-goals (now)

- CRDT multi-writer production sync
- Federated multi-tenant intelligence
- Always-on cloud brain as default product
- Replacing native local-first with a remote-only store

## Goals (later)

1. **One person, multiple devices** share the same identity_core + Decisions
2. Conflict policy that prefers identity_core and newer valid_from
3. Offline-first: local native remains authoritative while offline
4. Explicit opt-in cloud/relay — never silent upload

## Proposed layers

| Layer | Role |
|-------|------|
| Local native | Primary store (PyO3 data-dir) |
| Export pack | Encrypted snapshot of entities/edges/episodes since watermark |
| Sync relay | Optional user-owned object store or Engram-hosted vault |
| Import merge | Idempotent apply with entity resolution + identity_core protect |

## Conflict rules (draft)

1. Soft-deleted + identity_core never auto-hard-delete via sync
2. Same entity_id: higher `updated_at` / newer evidence wins fields
3. Name collisions: run existing merge Tier-0 signals; never auto-merge without score ≥ threshold
4. Episodes: append-only by id; no rewrite of remote history
5. Promotion windows stay **device-local** (compaction caps are not global)

## MVP sequence (when unblocked)

1. `engram export --since <watermark> --encrypt` → pack file
2. Manual/airdrop import on second device
3. Background optional sync daemon (relay) with the same pack format
4. Dashboard continuity scorecard shows **last sync age**

## Metrics before build

- Cold Decision hit rate stable on **single** native brain for 2 weeks
- Public MCP surface unchanged (no new public tools for sync)
- Doctor golden-loop still passes post-import

## Open questions

- Group_id multi-brain: sync all groups or selected?
- Attachment/blob (observe_file) size limits for packs?
- Cross-OS path rewriting for project_path artifacts?

---

## Policy / federated intelligence

Parked under vision docs. No code until multi-device sync MVP is shipped and
continuity remains the north star metric.
