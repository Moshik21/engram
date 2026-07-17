# I3 — MCP concurrent-open of the native graph (2026-07-17)

The child provably opens the real native engine (189 routes, `data.mdb` + `lock.mdb` created); init is just genuinely fast on an empty dir. Final verification is complete.

## I3 Report: MCP concurrent-open of the native graph

**Files changed**
- `server/tests/test_native_concurrent_open.py` (NEW, only file touched) — 2 tests, both use `tmp_path` data dirs only; second opener is a subprocess with `HOME` pointed at a scratch dir. Never touches `~/.helix` or `~/.engram`.

**Observed behavior (with evidence)**
1. **Second PROCESS open SUCCEEDS — no exclusion, no corruption.** helix-db opens LMDB with default flags (no `NOLOCK`): `native/helix-repo/helix-db/src/helix_engine/storage_core/mod.rs:84-90` (`EnvOpenOptions::new().map_size(..).max_dbs(200).max_readers(200).open(path)`). LMDB's `lock.mdb` multi-process protocol applies: test shows child (while parent holds the env open) reads the parent's committed evidence, writes its own, and the parent then sees both. Vectors (HNSW) and BM25 live in the same env (`mod.rs:178-190`), so all data is transactionally shared.
2. **Same-process second open fails CLEANLY.** heed3 0.22 keeps a global `OPENED_ENV` path registry → `Error::EnvAlreadyOpened` ("environment already open in this program") → `PyRuntimeError: Engine init failed` (`heed3-0.22.0/src/envs/env_open_options.rs:405,420-421`). This is why `NativeTransport` caches engines per data dir until process exit (`native_transport.py:118-133,165-171`).
3. **Latent hazard the test cannot demonstrate but the source proves:** on macOS LMDB compiles with `MDB_USE_POSIX_SEM` and **without** `MDB_USE_ROBUST` (`lmdb-master3-sys-0.2.5/.../mdb.c:166-169`). The cross-process writer lock is a non-robust POSIX semaphore: a session killed mid-write (SIGKILL of a Claude Code process) leaves it held and every other process's next write blocks forever.
4. **Application-level races even when LMDB behaves:** each MCP session keeps derived state in process memory (activation store + owner-checked snapshot save-back `mcp/server.py:316-325`, cue-index outbox, episode worker, engine cache). Read-modify-write flows (access_count bumps, reconsolidation summary updates) span multiple txns with no cross-process coordination → silent lost updates between two sessions.

**Recommendation**
Advisory flock in the MCP native init path, mirroring `brain_runtime.exclusive_brain_lock` (`brain_runtime.py:68-88`):
- Per-data-dir sentinel file (e.g. `<data_dir>/engram-shell.lock` — a separate file, NOT `data.mdb`/`lock.mdb`, so it can't interfere with LMDB's own fcntl locks), `fcntl.flock(LOCK_EX | LOCK_NB)` taken in `_init` when the resolved backend is helix-native, held for process lifetime.
- On conflict: fail startup fast with a message naming the holder PID (write PID into the lock file like `brain_runtime` does) — a clean refusal beats the actual failure modes (delayed semaphore deadlock, divergent activation state, lost updates), which are silent.
- flock auto-releases on process death, so a killed session never wedges the next one — exactly the property the LMDB POSIX-sem writer lock lacks on macOS.
- Keep brain-vs-shell coordination unchanged (`--pause-shell` + `~/.engram/brain.lock`); this new lock is shell-vs-shell.
- A documented single-client rule alone is insufficient: the second open succeeds silently today, so nothing tells the user they are in the unsafe configuration.

No fix landed (per goal doc: "No fix landed without the test first") — test is in place to gate the follow-up.

**Test evidence**
```
uv run ruff check tests/test_native_concurrent_open.py  → All checks passed!
env HOME=<scratch>/fakehome ENGRAM_MODE=lite uv run pytest tests/test_native_concurrent_open.py -v
  test_second_process_open_succeeds_and_shares_committed_state PASSED
  test_same_process_second_open_fails_cleanly PASSED
  2 passed in 0.33s
```
Standalone child-run confirmation: rc 0, result `{"open_ok": true, "seen_before": [], "seen_after": ["ev_child"]}`, data dir contains `data.mdb` + `lock.mdb`, stderr shows `helix_native: engine initialized (routes=189)`.

**Skipped**: nothing. Doc checkbox updates in `COGNITIVE_CORE_GOAL.md` left to `fix-docs-truth` (not my file).