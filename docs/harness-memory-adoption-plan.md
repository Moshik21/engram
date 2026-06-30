# Harness-Mediated Memory Adoption Plan

Date: 2026-06-30

Status: proposed

Implementation status: phases 1–5 shipped (commit `6c8d3b7`)

Owner surface: harness adoption, AXI hooks, MCP prompts, evaluation

Related docs:

- [axi-interface-plan.md](./axi-interface-plan.md) — session-start injection, read-only default
- [memory-value-latency-plan.md](./memory-value-latency-plan.md) — budgets, auto-recall, packet cache
- [CHANNEL_SEPARATION_DESIGN.md](./CHANNEL_SEPARATION_DESIGN.md) — recall channel tiers

## Problem Statement

Engram's adoption story over-indexes on **agent-initiated** `observe` every turn. That conflicts
with how the runtime actually works and with the user's expectation that memory should compound
**passively** when a harness is connected.

The product thesis (validated in dogfood and cross-project testing):

| Layer | Responsibility | Token cost |
|-------|----------------|------------|
| **Harness** | Inject context at session start; capture turns via hooks; queue offline on failure | Zero/low |
| **Agent** | `get_context` once; `recall` when cues fire; `remember` for high-signal | Judgment only |
| **Brain** | Triage, project, consolidate in background | Background |

[firstmate](https://github.com/kunchenguid/firstmate) is useful inspiration here, not a clone
target. Its core idea is **event-driven infrastructure**: bash classifiers and filesystem state
wake the LLM only when something is actionable. Engram should apply that pattern to memory
adoption — harness wakes and capture, agent recalls when it matters.

## Current Baseline (Already Built)

Do not rebuild what exists. Wire and message it correctly.

### Injection (read path)

| Surface | Location | Behavior |
|---------|----------|----------|
| AXI home packet | `server/engram/axi/surfaces.py` (`build_home_payload`) | Briefing + artifact hits + next commands |
| Session prime | `server/engram/mcp/server.py` (`_session_prime`) | First MCP tool call primes context |
| AXI session-start hook | `server/engram/axi/hooks.py` | Read-only `engram axi hook-run` (Codex, Claude Code) |
| Priming rules | `server/engram/harness_adoption.py` | Cursor, Windsurf, Grok Build project rules |
| Auto-recall piggyback | `server/engram/retrieval/auto_recall.py` | Lite/medium recall on read tools |
| Adoption debt | `server/engram/retrieval/adoption_debt.py` | Capture vs recall imbalance signal |

### Capture (write path)

| Surface | Location | Behavior |
|---------|----------|----------|
| REST auto-observe | `server/engram/api/knowledge.py` (`POST /auto-observe`) | Deduped fast store; hook entrypoint |
| AutoCapture hooks | `server/engram/setup.py` | `UserPromptSubmit`, `Stop`, `SessionStart`, `SessionEnd` → `auto:prompt`, `auto:response`, `auto:session` |
| Offline queue | `~/.engram/capture-queue.jsonl` | Replay on next session start |
| Worker batching | `server/engram/ingestion/worker_batching.py` | Adjacent auto-capture turns merged before triage |
| MCP observe/remember | `server/engram/mcp/server.py` | Agent-initiated; bounded capture wait (~100ms) |
| Live-turn fingerprint | `server/engram/mcp/server.py` (`_ingest_live_turn`) | Session conv context on read tools, not full durable capture |

### Installer gap

`engramctl connect` installs **read-only AXI hooks** for Codex/Claude Code and **MCP + priming**
for Cursor/Windsurf/Grok. It does **not** install the full Claude AutoCapture hook set from
`setup.py` unless the user runs `engram hooks install` separately. Harness capture and agent
protocol are therefore disconnected in the default connect path.

## Design Principles

1. **Harness injects and captures; agent judges.** Routine chat capture is infrastructure, not a
   tool-call habit.
2. **Wake on actionable signals only.** Session start, project switch, identity/prior-context
   queries, consolidation notifications — not every turn.
3. **Two-layer memory ownership** (firstmate-aligned):
   - **Portable captain memory** → Engram global graph (+ optional human-editable export)
   - **Project memory** → `bootstrap_project` artifacts + repo docs; not a substitute for portable recall
4. **Scout vs ship capture shapes:**
   - Scout (explore, audit, compare): harness `auto:*` sources → cues; triage may skip
   - Ship (preferences, corrections, decisions): agent `remember` or high-confidence promotion
5. **Read-only default, explicit capture opt-in** — unchanged from AXI contract.

## Target Architecture

```text
Session start
  ├─ AXI hook-run / home packet  → briefing + artifacts + growth line
  ├─ bootstrap_project (idempotent) → project artifacts
  └─ optional AutoCapture hooks installed

Each turn (harness, zero agent tokens)
  ├─ UserPromptSubmit → POST /auto-observe (auto:prompt)
  ├─ Stop             → POST /auto-observe (auto:response)
  └─ read-tool middleware → auto_recall lite + session fingerprint

Agent (judgment only)
  ├─ get_context once per session (or when adoptionDebt / project switch)
  ├─ recall when people/projects/prior work appear
  └─ remember for high-signal; observe only for explicit "store this"

Background
  ├─ worker + triage (~35% promoted)
  └─ consolidation → maturation, merge, dream
```

## Implementation Plan

Work is split into five PR-sized phases. Each phase has acceptance criteria and evaluation
hooks. Phases 1–3 are the highest leverage; 4–5 are polish and operator UX.

---

### Phase 1 — Protocol Rewrite (Priming + MCP Prompts)

**Goal:** Align agent-facing text with harness-mediated memory. Remove "observe every turn" as the
default story.

**Files:**

- `server/engram/mcp/prompts.py` — `ENGRAM_SYSTEM_PROMPT`, `_ADOPTION_MOTIVATION`
- `server/engram/harness_adoption.py` — `priming_instruction_text()`
- `server/engram/retrieval/memory_authority.py` — `claim_authority` capture guidance if duplicated
- `server/tests/test_agent_adoption_surfaces.py` — prompt assertion updates

**Changes:**

1. Replace Pre-Response step 1 ("Call `observe(user_message)` …") with tiered capture policy:

   ```text
   Capture policy:
   - Harness auto-capture (when installed) handles routine turns — do not duplicate with observe.
   - remember: explicit preferences, corrections, identity facts, durable decisions.
   - observe: only when the user asks to store something or you have high-value context
     the harness cannot see (tool-internal reasoning you must persist).
   ```

2. Session protocol becomes:

   ```text
   Before first substantive answer: get_context(project_path=...) once.
   When prior context could change the answer: recall(query, project_path=...).
   When adoptionDebt is actionable: get_context before answering.
   Do NOT call observe on every turn when harness capture is active.
   ```

3. Add one line acknowledging harness capture: "If `api_auto_observe` or `auto:*` sources appear
   in runtime metrics, capture is already happening — focus on recall, not re-capture."

**Acceptance criteria:**

- [x] `ENGRAM_SYSTEM_PROMPT` and priming rules no longer mandate per-turn `observe`
- [x] `engram adoption` transcript schema still valid (`before_answer`: context/recall; `capture`: observe OR harness auto_observe)
- [x] `test_agent_adoption_surfaces.py` updated; all prompt tests pass

**Evaluation:**

```bash
cd server && uv run pytest tests/test_agent_adoption_surfaces.py -v
cd server && uv run pytest tests/test_mcp_prompts.py -v  # if present
```

---

### Phase 2 — Unify Harness Capture in `engramctl connect`

**Goal:** Default connect path offers a single, clear capture story. Users should not need to
discover `engram hooks install` separately from `engramctl connect`.

**Files:**

- `installer/engramctl` — `connect`, `install_axi_hook_for_client`, new `install_autocapture_hooks`
- `server/engram/setup.py` — export `install_hooks()` for programmatic use; parameterize client label in trace
- `server/engram/axi/hooks.py` — optional: document relationship between AXI read-only and AutoCapture
- `server/engram/harness_adoption.py` — connect summary text
- `server/tests/test_auto_observe.py` — connect integration test stub if needed
- `docs/install/*.md` — capture opt-in documentation (only if install docs already mention hooks)

**Proposed CLI contract:**

```bash
# Default: MCP + read-only AXI (Codex/Claude) + priming (Cursor/Windsurf/Grok)
engramctl connect cursor --project /path/to/project

# Explicit transcript capture (Claude Code AutoCapture hooks)
engramctl connect claude-code --project /path --capture-transcript

# AXI capture flag remains for AXI-specific opt-in; --capture-transcript is the user-facing name
# for full turn capture. Internally:
#   --capture-transcript → setup.install_hooks() for claude-code
#   --capture            → existing AXI capture metadata (keep for backward compat)
```

**Per-client matrix after Phase 2:**

| Client | MCP | Priming | AXI read-only | AutoCapture |
|--------|-----|---------|---------------|-------------|
| Codex | — | — | default | future: Codex Stop hook (Phase 2b) |
| Claude Code | — | — | default | `--capture-transcript` |
| Cursor | yes | yes | — | future: no hook API yet |
| Windsurf | yes | yes | — | — |
| Grok Build | yes | yes | — | — |

**Phase 2b (stretch, same phase or fast follow):** Codex `Stop` / turn-end hook calling
`POST /auto-observe` with payload shape already accepted by
`build_api_auto_observe_request_surface` (raw hook JSON). Reuse offline queue pattern from
`setup.py`.

**Acceptance criteria:**

- [x] `engramctl connect claude-code --capture-transcript` installs all four AutoCapture hooks
- [x] `engramctl connect --help` documents capture tiers (read-only vs transcript)
- [x] Adoption trace (`~/.engram/adoption-trace.jsonl`) records `rest_hook_prompt` / `rest_hook_response`
- [x] `engram adoption --require-live-evidence` accepts harness capture as `capture` phase evidence
- [x] Existing `--capture` AXI behavior unchanged

**Evaluation:**

```bash
# Install + smoke (Claude Code)
engramctl connect claude-code --project "$PWD" --capture-transcript --verify
# Send one prompt; confirm episodes with source auto:prompt in graph or runtime metrics
cd server && uv run engram adoption --authority /tmp/claim.json --calls /tmp/trace.jsonl
```

---

### Phase 3 — Actionable Adoption Debt (Event-Driven Nudges)

**Goal:** Adoption debt behaves like firstmate's wake queue — surface only when the agent should
act, not on every tool response after context is loaded.

**Files:**

- `server/engram/retrieval/adoption_debt.py` — extend `adoption_debt_is_actionable()`, add `wakeReason`
- `server/engram/retrieval/auto_recall.py` — `apply_mcp_recall_enrichment` gate
- `server/engram/mcp/server.py` — `SessionState`: track `project_path`, `harness_capture_active`
- `server/engram/retrieval/runtime_state.py` — expose harness capture signal in `agentAdoption`
- `server/tests/test_agent_adoption_surfaces.py` — new wake-reason cases

**Proposed actionable triggers:**

| `wakeReason` | Condition | Suggested action |
|--------------|-----------|------------------|
| `session_unprimed` | First session tool call, no `get_context` yet | `get_context` |
| `project_switched` | `project_path` changed since last context load | `get_context` + `search_artifacts` |
| `identity_query` | recall_need analyzer detects person/preference query, personal graph thin | `recall` + optional `remember` seed |
| `capture_without_recall` | harness captured N episodes, agent recall count 0, session > 2 turns | `get_context` once |
| *(none)* | `contextLoadedThisSession` and no other trigger | suppress `adoptionDebt` |

**Non-actionable (absorb silently):**

- Debt cleared after `get_context`
- Harness actively capturing (`api_auto_observe` in recent metrics) and agent already recalled once
- Pure code-navigation turns with rich artifact bootstrap

**Acceptance criteria:**

- [x] `adoptionDebt` omitted from MCP responses when not actionable
- [x] `wakeReason` present when debt is actionable
- [x] `get_runtime_state().agentAdoption.adoptionDebt` uses same logic
- [x] No regression in `test_adoption_debt_cleared_after_context_load`

---

### Phase 4 — Captain Export (Human-Editable Identity Layer)

**Goal:** firstmate's `data/captain.md` pattern — canonical human-editable prefs with Engram graph
as the recall engine.

**Files (new + existing):**

- `server/engram/identity/captain_export.py` *(new)* — export/import between markdown and graph
- `server/engram/cli/` or `server/engram/__main__.py` — `engram captain export|import|sync`
- `server/engram/storage/*` — read `identity_core` entities via existing `get_identity_core_entities`
- `server/tests/test_captain_export.py` *(new)*

**Format sketch (`~/.engram/captain.md`):**

```markdown
# Captain preferences
<!-- engram-captain-version: 1 -->

## Identity
- Name: ...
- Location: ...

## Working style
- Prefers passive harness memory over per-turn observe
- ...

## Protected entities
- Liam (person, identity_core)
```

**Sync rules:**

1. **Export:** graph `identity_core` entities + preference facts → markdown
2. **Import:** parse markdown → `remember()` high-signal stubs or direct entity patch
3. **Conflict:** file mtime vs graph `last_corrected_at` — prefer latest user edit; log reconciliation episode
4. **Bootstrap:** AXI home packet mentions captain file if present ("portable prefs loaded from ~/.engram/captain.md")

**Acceptance criteria:**

- [x] `engram captain export` writes valid markdown from identity_core entities
- [x] Round-trip import does not duplicate entities (parse-only import; no auto-write on session start)
- [x] Export is optional — empty graph still works
- [x] No automatic import on every session (explicit `sync` or hook opt-in only)

---

### Phase 5 — Tiered Loop Documentation + Evaluation Gates

**Goal:** Operator-visible documentation and measurable adoption evidence for the harness-first model.

**Files:**

- `docs/harness-memory-adoption-plan.md` — this doc; mark phases complete as shipped
- `server/engram/evaluation/adoption_evidence.py` — recognize harness capture in live evidence
- `server/engram/lifecycle_cli.py` or `evaluation/cli.py` — `engram adoption --harness-first` checklist
- `README.md` — short "How memory compounds" section pointing here (only if README already discusses adoption)

**Tiered loop reference (for docs and dashboard):**

| Tier | Trigger | Engram action | Latency budget |
|------|---------|---------------|----------------|
| Hot | Session start, project open | AXI home, bootstrap, session prime | < 3s hook / < 2s MCP prime |
| Warm | Read tools, proper nouns | auto_recall lite, packet cache | < 350ms |
| Cold | Identity query, deep prior work | `recall`, `search_artifacts` | < 2s |
| Background | Session end, consolidation schedule | triage, merge, dream | async |

**Adoption scorecard (dogfood gate):**

```bash
cd server
uv run engram adoption \
  --authority claim-authority.json \
  --calls live-harness-transcript.json \
  --require-live-evidence \
  --expect-harness-capture  # new flag: capture phase satisfied by auto_observe OR observe
```

**Acceptance criteria:**

- [x] Live SessionStart evidence: AXI home or MCP session prime within budget
- [x] Live capture evidence: at least one `auto:prompt` or `auto:response` per session OR explicit `remember`
- [x] Live recall evidence: `get_context` or `recall` before substantive answer on identity/project query
- [x] `engram adoption --expect-harness-capture` gate accepts `auto_observe` capture evidence

---

## PR Dependency Graph

```text
Phase 1 (prompts)     ──┐
                        ├──► Phase 5 (docs/eval)
Phase 2 (connect)     ──┤
Phase 3 (debt)        ──┘
Phase 4 (captain)     ───► independent; after Phase 1 messaging stabilizes
```

Recommended merge order: **1 → 2 → 3 → 5 → 4**

Phase 4 is user-facing polish and can slip without blocking harness adoption.

## Success Metrics

After Phases 1–3 ship, a healthy cross-project session should show:

| Metric | Target |
|--------|--------|
| Agent `observe` calls per session | ↓ 80% vs prompt-driven baseline |
| `api_auto_observe` / `auto:*` episodes per session | ≥ 1 per active session with capture installed |
| `get_context` per session | 1–2 (start + project switch), not per turn |
| `recall` on identity/project queries | ≥ 1 when query warrants it |
| Personal graph growth over 2 weeks | Non-zero entity promotion from triage without manual `remember` spam |
| Adoption debt noise | Actionable debt on < 20% of tool responses |

## Non-Goals

- Cloning firstmate orchestration (tmux crew, worktrees, backlog.md task queue)
- Replacing project `AGENTS.md` / artifacts with graph-only memory
- Mandatory per-turn `observe` in MCP or priming rules
- Silent capture without opt-in (AXI read-only default preserved)
- Grok/Cursor turn-end hooks before stable hook APIs exist (primacy + MCP injection only)

## Open Questions

1. **Should `--capture-transcript` become default for Claude Code connect?** Recommendation: no —
   explicit opt-in first; flip default after one dogfood week with metrics.
2. **Cursor hook path?** Cursor lacks documented turn-end hooks; rely on priming + session injection
   until a supported mechanism exists.
3. **Captain file location:** `~/.engram/captain.md` (global) vs project-local — recommend global
   only for portable prefs; project prefs stay in artifacts.

## Immediate Next Action

Dogfood the shipped phases and measure success metrics (↓80% agent `observe` when
`--capture-transcript` is active):

```bash
engramctl connect claude-code --project "$PWD" --capture-transcript
cd server && uv run engram adoption --authority claim.json --calls trace.jsonl --expect-harness-capture
```