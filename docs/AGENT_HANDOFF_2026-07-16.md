# Engram agent handoff — 2026-07-16

**Audience:** any future coding agent continuing Engram polish / dogfood / product work.  
**Repo:** `/Users/konnermoshier/Engram` (branch `main`, **large uncommitted working tree** — not pushed).  
**Live as of:** 2026-07-16 ~09:00 MST.

---

## 1. Product thesis (do not drift)

Engram’s public job is **not** LongMemEval. It is:

> A fresh agent on a different session surfaces **≥1 high-signal prior Decision** without a handoff doc.

**Public MCP freeze:** `ENGRAM_MCP_SURFACE=public` → golden-loop tools only  
(`get_context`, `recall`, `observe`, `remember`, `intend`, `forget`, `claim_authority`, `bootstrap_project`, `get_runtime_state`).  
See `docs/GOLDEN_LOOP.md`, `server/engram/mcp/surface.py`.

Operator / Loop Steward / mop / brain stay on **CLI, AXI, operator MCP** — not public.

**North-star check:** `engram continuity --against-live` (must PASS when shell is up).

---

## 2. Live dogfood machine (this host)

| Item | Value |
|------|--------|
| Shell LaunchAgent | `dev.engram.local` (running) |
| Brain LaunchAgent | `dev.engram.brain` (loaded; StartInterval 7200s) |
| API | `http://127.0.0.1:8100` |
| Mode | Helix native PyO3 |
| Data | `~/.helix/engram-native-dogfood-axi` (~**17G**) |
| Config | `~/.engram/.env` |
| Logs | `~/.engram/logs/engram.log`, `engram-brain.log` |
| Brain status | `~/.engram/brain-status.json` |
| Embed cache | `~/.engram/models/fastembed/hf` |

### Active env (consumer quiet path)

```bash
ENGRAM_MODE=helix
ENGRAM_RUNTIME_ROLE=shell
ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE=quiet
ENGRAM_ACTIVATION__RECALL_PROFILE=wave2
ENGRAM_ACTIVATION__INTEGRATION_PROFILE=off
ENGRAM_ACTIVATION__WORKER_ENABLED=false
ENGRAM_MCP_SURFACE=public
ENGRAM_EMBEDDING__PROVIDER=local
ENGRAM_EMBEDDING__LOCAL_MODEL=nomic-ai/nomic-embed-text-v1.5-Q
FASTEMBED_CACHE_PATH=~/.engram/models/fastembed/hf
ENGRAM_HELIX__TRANSPORT=native
ENGRAM_HELIX__DATA_DIR=~/.helix/engram-native-dogfood-axi
```

**Do not** casually switch dogfood back to `standard` + `rework` + `recall=all` + `monolith` — that co-located merge/CE and caused **macOS Jetsam** every ~10 minutes.

### Live metrics (2026-07-16 morning)

| Metric | Value |
|--------|--------|
| Health | healthy |
| Role | shell (scheduler + EpisodeWorker **off** in-process) |
| Continuity | **PASS** (recall ~1.8s, context ~75ms) |
| Graph | ~749 entities, ~8632 episodes |
| Deferred evidence | **54** (was ~3200 mid-session Jul 13) |
| Pending evidence | 446 |
| cue_only | ~3039 |
| open_adjudication | 419 |
| open_work | ~919 |
| pressure | ~57 (threshold 100); `should_mop` still True (open_work formula) |
| Embeds | Quantized nomic **working** (768-d); log: `FastEmbedProvider ready: …-Q` |
| Shell RSS | ~1.6–1.8 GB (large Helix open — expected) |

**Debt arc (important):**

```
~3223 deferred (Jul 10–13 hangover)
  → ~1793 (pre-mop recount)
  → ~1054 (first hygiene mop, stale/already_exists)
  → 54 (recovery mop: age floor 3d when deferred≥200)
  → 54 stable (scheduled mops reject 0 — remainder not matching selectors)
```

---

## 3. Architecture now: hot shell / cold brain

**Design:** `docs/design/hot-cold-process-split.md`  
**Agreed leans:**

1. **Worker cold-only** in shell (observe queues; `remember` can still promote).  
2. **Exclusive graph via `--pause-shell`** (not concurrent Helix multi-open).  
3. **New `quiet` profile** (not overload `conservative`).

| Process | Role |
|---------|------|
| `engram serve` + `ENGRAM_RUNTIME_ROLE=shell` | Golden loop API + MCP; **no** in-process consolidation scheduler / EpisodeWorker |
| `engram brain run` | Cold process; exclusive flock `~/.engram/brain.lock` |
| `engram brain run --tier mop` | **Not** full consolidation phases — calls shared `hygiene_ops.execute_hygiene_mop` |
| `engram hygiene report\|mop` | Operator scoreboard + same drains |

Key code:

- `config.EngramConfig.runtime_role` + `shell_runs_in_process_brain()`
- `ActivationConfig` profile `quiet` (worker off, CE off, reranker off, wave2)
- `main._startup` gates scheduler/worker on role
- `brain_cli.py` / `brain_runtime.py` / `hygiene_ops.py` / `hygiene_cli.py`
- Loop Steward (separate): `loop_adjustment.py`, `loop_cli.py`, operator MCP `loop_*`

### Loop Steward (harness subconscious)

- TTL `LoopAdjustment` biases shell budgets / phase boost-defer when **monolith** or when brain honors overlay on mop.
- CLI: `engram loop status|apply|clear|steward-once|propose…`
- Public MCP does **not** expose loop tools.
- Design: `docs/design/loop-steward-protocol.md` and related.

---

## 4. What shipped in this workstream (uncommitted)

Much of the following is **on disk but not committed** (`git status` dirty + many `??` files). **Commit/PR before assuming origin/main has it.**

**Hot/cold + quiet**

- Runtime role, quiet profile, installer defaults → shell+quiet+wave2
- Brain CLI + LaunchAgent example + dogfood `dev.engram.brain` plist
- Mop path shared; recovery stale thresholds when deferred ≥ 200
- FastEmbed stable cache + soft-fail on broken model; quantized nomic dogfood

**Hygiene / debt**

- Scoreboard, junk/stale/already_exists drains, pressure integration
- Evidence drain prioritization (junk-first)

**Loop Steward**

- File dual-write + Helix consolidation sidecar
- steward-once, propose-from-report, dashboard card (untracked), dogfood scripts

---

## 5. Known issues / landmines

### P0 awareness for next agent

1. **Brain `--pause-shell` every 2h**  
   Shell logs show restart ~every 2h when brain agent fires (stop → mop → start).  
   Last `brain-status.json` showed `duration_s ≈ 38690` (~10.7h) for a mop that **rejected 0** — investigate hang in pause/resume or lock; shell availability is not “always on” during brain windows.

2. **Mop plateau at deferred=54**  
   Remaining rows don’t match junk/stale/already_exists (many are “fresh” narrow sludge or high-signal-typed).  
   Next drain needs better classifiers or adjudication commit/reject — not bigger budget alone.

3. **Git debt**  
   Huge uncommitted set. Do **not** `git reset --hard`. Prefer stacking commits: loop-steward / hygiene / hot-cold / embeds.

4. **Full nomic ONNX never completed**  
   Partial `blobs/model.onnx` under non-`hf` cache is junk (~110B error page). **Use quantized Q + `…/fastembed/hf`.**

5. **Ollama extraction**  
   Config may point at unreachable Ollama → **narrow** fallback. High-signal path = harness `remember` with proposals.

6. **Helix multi-open**  
   Do not open graph from two writers without pause-shell/lease. Capture outbox **not built**.

7. **`should_mop=True` with pressure &lt; 100**  
   Trigger uses open_work/debt heuristics, not pressure alone — expected.

### Product stage

**Private beta / founder dogfood:** golden loop + continuity real; quiet shell + mop recovery real; not yet “install and forget” consumer polish.

---

## 6. Commands cheatsheet

```bash
# Status
engramctl status
curl -s http://127.0.0.1:8100/health
engram brain status
engram hygiene report

# Continuity (product metric)
cd server && uv run engram continuity --against-live

# Cold mop (pauses shell by default)
engram brain run --tier mop --budget 1000 --profile quiet --pause-shell

# Hygiene only (also opens graph — prefer shell stopped or exclusive)
engram hygiene mop --budget 1000

# Loop steward
engram loop status
engram loop steward-once --dry-run

# Tests (lite-ish)
cd server && uv run pytest tests/test_runtime_role_and_quiet.py tests/test_evidence_drain.py -q

# Restart shell after .env change
engramctl stop && engramctl start
```

---

## 7. Recommended next work (priority order)

1. **Investigate multi-hour brain mop / pause-shell** — shell must not stay down for hours; cap brain runtime; fix duration accounting; consider shorter mop or no pause if Helix allows safer exclusive pattern.  
2. **Commit the working tree** in logical PRs (hot-cold, hygiene, loop steward, dashboard).  
3. **Drain plateau** — classify remaining 54 deferred + pending/cue_only/adjudication backlog.  
4. **Steward-once on brain start** when `should_mop` (TTL overlay auto-renew).  
5. **engramctl first-class brain agent** install (plist example exists).  
6. **Capture outbox** only if pause-shell UX remains painful after (1).  
7. **Avoid** full Rust rewrite; Helix is already native. Thin native shell only if RSS still hurts after brain split is solid.

---

## 8. Philosophy reminders (from founder + Agents.md)

- **Harness = subconscious**, shell = body; Loop Steward biases via TTL knobs, not in-process auto-AI.  
- **observe** default; **remember** sparse high-signal.  
- **Deferred = temporary queue**, not museum.  
- Surgical diffs; no speculative features.  
- Public surface stays frozen at golden loop.

---

## 9. Quick “are we healthy?” checklist

```
[ ] curl :8100/health → healthy
[ ] engramctl status → Role: shell
[ ] FastEmbedProvider ready …-Q in logs (not Vector search OFF)
[ ] continuity --against-live → PASS
[ ] deferred_evidence low (order ~50s, not thousands)
[ ] brain status ok=True, duration not multi-hour
[ ] no Jetsam thrash (restart storm every ~10m)
```

If all checked: **continue product polish**.  
If Jetsam returns: confirm `RUNTIME_ROLE=shell` and brain not running full consolidation in-process.  
If vectors OFF: check `FASTEMBED_CACHE_PATH` and `LOCAL_MODEL=…-Q`.

---

## 10. Related docs

| Doc | Use |
|-----|-----|
| `docs/GOLDEN_LOOP.md` | Public product contract |
| `docs/design/hot-cold-process-split.md` | Shell/brain design + implement notes |
| `docs/design/loop-steward-protocol.md` | Steward knobs |
| `docs/design/memory-loop-self-regulation.md` | Debt / self-drain |
| `docs/CURRENT_HANDOFF.md` | Older large handoff (may be stale vs this file) |
| `Claude.md` / `Agents.md` | Project conventions |

**Prefer this file (`docs/AGENT_HANDOFF_2026-07-16.md`) as the entrypoint for agents resuming after Jul 16.**
