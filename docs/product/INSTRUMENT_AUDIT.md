# Instrument Audit

**Status:** active ledger · opened 2026-07-24 (task #14 / P9)
**Enforced by:** `server/tests/test_metric_honesty_contract.py`

---

## Why this document exists

Engram has repeatedly made confident, wrong decisions. Not because the reasoning
was bad — because the **gauges lied**. In the 48 hours before this audit was
opened, seven separate instruments were caught reporting numbers that were not
measurements. At least one of them changed an engineering decision: an argument
was built on "the graph has fewer relationships than entities," which was false,
and which came from a number that had been extrapolated from a 10-entity sample.

This is the same disease as the project's dominant bug class
(*computed-but-silently-inert*), viewed from the other end. Silent inertness is
work that happens and is discarded. Instrument dishonesty is a number that is
reported without the work happening. Both present as a plausible-looking system
that is not doing what its output claims.

---

## THE RULE

> **A metric must be either accurate or absent. Never plausible-but-wrong.**

The argument is asymmetric, and the asymmetry is the whole point:

- An **absent** metric prompts investigation. Someone sees `null`, asks why, and
  either builds the measurement or writes down that it is unavailable. The
  failure mode is *delay*.
- A **wrong** metric **ends** investigation with a false answer. Nobody
  investigates `0.0 relationships per episode` — they conclude extraction is
  broken and go fix extraction. The failure mode is *acting on a fiction*, and
  it costs more the more competent and decisive the reader is.

A missing number costs you an hour. A wrong number costs you a week and a
retraction. Absence is therefore not the lesser evil, it is *strictly better*
than a fabricated value, and code should prefer it actively — drop the key,
return `None`, or attach an explicit status — rather than zero-filling to keep
a schema tidy.

Schema tidiness is the usual excuse. It is not worth it. A consumer that must
handle a missing field is a consumer that has been forced to think about
whether the number exists.

### The honest counter-case

Cheap approximate metrics have real value. Exact counts on a 17GB brain can hang
the shell; that is a genuine reason to sample. The rule does not forbid
approximation — it forbids **unlabelled** approximation. An approximate metric
is legitimate when, and only when:

1. its **name or its payload says it is approximate** (`relationships_approx`,
   or a sibling `"method": "sampled"` field), and
2. its **error bound is stated** — sample size, staleness window, or a
   confidence interval — so a reader knows whether a 2x difference is signal.

`AUDIT-1` fails both tests. It is not "an approximation we accepted"; it is an
approximation that presented itself as a count. Had it been called
`relationship_count_estimate` with `sample_size: 10` beside it, nobody would
have built an argument on it.

A second-order caution, honestly stated: a rule that is enforced too eagerly
produces payloads full of `null`, and consumers that treat `null` as `0`
anyway — see `AUDIT-12`, where exactly that happens at the last mile. Absence
only works if the **whole chain** preserves it. Deleting a metric without
auditing its readers converts a wrong number into a differently-wrong number.

---

## Ledger

Severity: **HIGH** = has changed or can change a decision · **MED** = misleads a
reader · **LOW** = wrong but currently inert.

Line numbers drift (this file was written while two other lanes were editing
`graph.py`); the **symbol names are authoritative**, and the contract test keys
on names, not lines.

### AUDIT-1 — relationship count extrapolated from a 10-entity sample · HIGH

`server/engram/storage/helix/graph.py`, `get_stats()` ~L2398-2416:

```python
# Relationship count is expensive (per-entity edge scan); approximate from first few
sample_ents = entities[:10]
...
if entity_count > 10 and sample_hids:
    relationship_count = int(relationship_count * entity_count / 10)
```

Reported **750**; the true count was **1637**. Presented as `"relationships"` in
`/api/stats` with no marker of any kind. This produced a real wrong decision.

**Aggravating detail found during this audit:** the "expensive" justification is
stale. An O(1) exact route already exists and is already used by the sibling
fast path — `count_relationships_by_group` at
`server/engram/storage/helix/schema.hx:451`. The extrapolation is not a
performance tradeoff; it is dead code that outlived its reason.

### AUDIT-2 — projection yield is a hardcoded literal · HIGH

`server/engram/storage/helix/graph.py`, `get_stats()` ~L2534, ~L2540:

```python
"yield": {
    "linked_entity_count": linked_entity_count,   # real
    "relationship_count": 0,                      # literal
    "avg_linked_entities_per_projected_episode": ...,  # real
    "avg_relationships_per_projected_episode": 0.0,    # literal
},
```

The "0.0 rels/episode" that appears in `/api/stats` and on the dashboard is
typed in, not counted. It sits directly beside two genuinely computed fields,
which is what makes it dangerous: the payload's overall credibility launders it.

### AUDIT-3 — `/api/episodes` reports every episode as having zero facts · MED

`server/engram/retrieval/graph_state.py`, `_build_episode_summary_item()`
~L1050-1051: `"entities": []` and `"factsCount": 0`, hardcoded, in a payload
that otherwise carries thirteen real fields.

Any consumer asking "did projection actually extract anything from this
episode?" gets a confident **no** for all of them.

### AUDIT-4 — the RF flip gate reads a field nothing writes · HIGH

*(Owned by lane 2 of this workflow; cited here, not fixed here.)*

The cue-usage metric aggregates the **legacy int** `used_count`
(`storage/helix/graph.py` `cue_used_count = sum(_as_int(cue.get("used_count")) ...)`,
surfaced as `usedCount` at `lifecycle_summary.py:353`), while the retrieval
write path increments the **tier-weighted float** `usage_used_count`
(`server/engram/retrieval/feedback.py:565`). Nothing in the recall path writes
the int any more — `ingestion/worker_batching.py:119` only carries the previous
value forward.

The gate therefore reads a permanent zero and can never open. This is why the RF
flip has been parked for days behind an "organic yield" condition that is not
merely unmet but **unmeetable**. Writer writes Y, gate reads X.

### AUDIT-5 — the same count, two methods, 12x apart, no reconciliation · HIGH

`/api/storage` reported `cues=9055`; `/api/lifecycle/summary` reported
`cueCount=756`. Root cause found during this audit — it is not staleness, it is
**two different definitions sharing one field name**:

| Path | Call | Cue definition |
|---|---|---|
| `/api/storage` → `storage/diagnostics.py:531` | `get_stats(group_id, exact=False)` | `count_cues_by_group` — **every** `EpisodeCue` row |
| `/api/stats`, lifecycle → `retrieval/graph_state.py:642` | `get_stats(group_id)` (default `exact=True`) | `sum(1 for cue in cues if cue.get("cue_text"))` — only cues **with text** |

The exact one (756) is right for "cues that can actually fire". 9055 is the row
count including empty shells.

**The generalisation matters more than the instance:** `get_stats(exact=...)` is
**two different instruments behind one name and one output schema**, selected by
a keyword argument. `relationships` diverges across the same boundary too, in
the *opposite* direction — the fast path returns the exact count, the "exact"
path returns AUDIT-1's extrapolation. Two endpoints report the same quantity by
different methods, and which lie you get depends on which caller you are.

### AUDIT-6 — effective config is launcher-dependent · HIGH

Every CLI-run measurement has been reading a different machine than the service
under test.

Reproduced during this audit:

```
$ cd server && .venv/bin/python -c "from engram.config import EngramConfig; \
    print(EngramConfig().activation.consolidation_profile)"
standard
```

but `~/Library/LaunchAgents/dev.engram.local.plist` runs:

```
set -a; source /Users/konnermoshier/.engram/.env; set +a; exec .../python -m engram serve
```

and `~/.engram/.env:10` sets `ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE=quiet`.

The mechanism is precise: `source`+`set -a` exports the value as a **real
process environment variable**, and in pydantic-settings env vars outrank *all*
dotenv files. A plain CLI run exports nothing, so the dotenv chain applies —
and `DEFAULT_ENV_FILES` (`server/engram/config.py:15-19`) puts the **cwd-relative
`.env` last**, i.e. highest precedence among files. Running from `server/` picks
up `server/.env:17`, which says `standard`.

Same working directory. Same venv. Opposite answers. `engram doctor` shares the
flaw (`server/engram/doctor.py:365` constructs a bare `EngramConfig()`).

**Consequence for every future measurement:** verify live config via
`GET /api/knowledge/runtime`. Never by instantiating `EngramConfig()`.

### AUDIT-7 — META: the harness lied, not the instrument · HIGH

An agent ran a measurement subprocess under `env HOME=<sandbox>` and reported a
FastEmbed model outage that did not exist. `DEFAULT_ENV_FILES[0]` is
`Path.home() / ".engram" / ".env"` — the file that sets `FASTEMBED_CACHE_PATH`.
With `HOME` rewritten it was never read, so the models "weren't there."

This belongs in the same ledger as the code-level fabrications because it is the
same failure: a number that described something other than the system under
test. Measurement-environment drift is instrument dishonesty with the bug
located in the harness. **Never sandbox `HOME` for a measurement.** If you do,
you are measuring a different machine, and the numbers you get will be
internally consistent and completely wrong.

Note that AUDIT-6 and AUDIT-7 are the *same three lines of code* —
`DEFAULT_ENV_FILES` — read once as "cwd matters" and once as "HOME matters".
Any config-resolution path that depends on ambient process state will keep
generating instances of this until the resolution is made explicit and
reported.

---

## New findings from this audit's sweep

### AUDIT-8 — the fast count route zero-fills an entire metric block · HIGH

`server/engram/storage/helix/graph.py`, `_get_fast_count_stats()` ~L2589-2601.
This is the path `/api/storage` always takes. It returns **exact** entity,
episode, relationship and cue counts — and then hardcodes, in the same payload:

```
state_counts = {}          attempted_episode_count = 0    total_attempts = 0
failure_count = 0          dead_letter_count = 0          failure_rate = 0.0
avg_processing_duration_ms = 0.0    avg_time_to_projection_ms = 0.0
yield.linked_entity_count = 0       yield.avg_linked_entities_per_... = 0.0
```

Ten fabricated fields sitting next to four real ones, with **no status marker**
distinguishing them. `failure_rate: 0.0` reads as "projection never fails."
`avg_time_to_projection_ms: 0.0` reads as "projection is instant." Both are
literals. This is AUDIT-2 at ten times the scale, on the endpoint that is polled
most often.

Confirmed by: `server/tests/test_metric_honesty_contract.py` (ledgered), and by
reading the function top-to-bottom — the query list it issues contains only the
four count routes, so it has no data from which those ten fields *could* be
computed.

`total_attempts` in that list was **not** found by manual reading. The contract
test found it on its first run.

### AUDIT-9 — AUDIT-3 duplicated on a second surface · MED

`server/engram/lifecycle_summary.py`, `_serialize_episode()` ~L144-145 carries
the identical `"entities": []` / `"factsCount": 0` pair as AUDIT-3, feeding
`/api/lifecycle/summary`. Fixing `graph_state.py` alone would leave this one
lying. Recorded so the fix is scoped to both.

### AUDIT-10 — temporal graph nodes report zero access for every node · MED

`server/engram/retrieval/graph_state.py`, `_build_temporal_graph_node()` ~L1029:
`"activationCurrent": 0.0`, `"accessCount": 0`.

The docstring says "without applying current activation," so the *intent* is
documented — but the intent is documented **in the source**, and the number
travels to the dashboard without it. A reader of the temporal graph sees a graph
in which nothing has ever been accessed. The honest encodings are `null`, or
omitting the keys from this node shape entirely. This is the clearest example in
the ledger of *knowing* a value is not a measurement and shipping it in a
measurement-shaped field anyway.

### AUDIT-11 — "we did not look" is reported as "there is nothing" · HIGH

`server/engram/retrieval/runtime_state.py`, `build_fast_runtime_packet()`
~L78-82:

```python
"artifactBootstrap": {
    "artifactCount": 0, "freshArtifactCount": 0, "staleArtifactCount": 0,
    "lastObservedAt": None,
    "status": "not_inspected",
},
```

The `status: not_inspected` guard is correct and is exactly what this document
advocates. **No consumer reads it.** All three readers branch on the count:

- `server/engram/retrieval/runtime_state.py:361` — `artifact_count = int(artifact_bootstrap.get("artifactCount") or 0)`
- `server/engram/harness_adoption.py:141` — same; it checks `runtime.status == "degraded"` and `adoption.status`, never `artifactBootstrap.status`
- `server/engram/retrieval/memory_authority.py:99` — same; `artifact_gap = artifact_count == 0 or last_observed is None`

And `server/engram/mcp/prompts.py:110` plus `CLAUDE.md` instruct the *agent* to
read it the same broken way: "if the runtime appears fresh or empty
(`artifactCount` is 0, `lastObservedAt` null) ... call `bootstrap_project`."

Confirmed executably:

```
artifactBootstrap.status = "not_inspected"
runtime_needs_bootstrap(fast_packet) -> True
```

A brain with thousands of episodes, probed via the fast packet, is indistinguishable
from a fresh install — and the documented agent response is to bootstrap it.
The guard field was built, shipped, and ignored: an instrument-side instance of
the project's own silent-inert bug class.

### AUDIT-12 — the dashboard launders absence back into zeros · MED

`dashboard/src/components/StatsPanel.tsx:521-522` and
`dashboard/src/components/LifecyclePanel.tsx:568-569`:

```tsx
const cueMetrics = stats?.cueMetrics ?? EMPTY_CUE_METRICS;
const projectionMetrics = stats?.projectionMetrics ?? EMPTY_PROJECTION_METRICS;
```

`EMPTY_CUE_METRICS` / `EMPTY_PROJECTION_METRICS` are objects of ~14 and ~16
hardcoded zeros (`StatsPanel.tsx:16-56`). When the API correctly **omits** a
metric block, the UI substitutes zeros and renders them in the same typography
as measured values.

This is the counter-case from the rule section made concrete: the one place in
the stack that propagates absence honestly has its honesty destroyed at the last
mile. Any fix that converts a fabricated metric to `null` **must** land with the
corresponding UI change, or it will make things worse, not better — a `null`
that renders as `0` is a wrong number with no server-side trace.

### AUDIT-13 — the swallow contract stops at `storage/` · GAP (not a confirmed lie)

`server/tests/test_storage_silent_swallow_contract.py` walks
`server/engram/storage/` only. An AST sweep run for this audit found **76**
except-handlers outside that tree whose only outcome is an empty return
(`return {}`, `return []`, `return 0`) — including in `brain_runtime.py`,
`consolidation/scheduler.py`, `retrieval/…`, and `evaluation/…`.

Most are probably legitimate. None have been audited. Recorded as a scoped
coverage gap, deliberately **not** claimed as 76 bugs — see "What this audit did
not resolve."

The good pattern to copy already exists in-tree:
`server/engram/retrieval/lookup.py:95-106` catches a native query timeout and
returns `{"items": [], "total": 0, "status": "timeout", "detail": ...}` —
an empty result that **says it is empty because it failed**. That is the shape
every degradation should take.

### AUDIT-14 — the packet cache makes any repeated-measures rig lie · HIGH

Found 2026-07-24 while building the recall meter (task #18), which is the only
reason it was found at all: the meter records which lane served each probe.

A 12-pass, 168-probe capture over the live shell reported **σ = 0.0** and
"resolving a 1-answer difference needs N ≥ 2 runs/arm". Both numbers are false.
`cache_satisfied` served **24/168** probes — two questions, on all twelve
passes — and the same rig run once ten minutes earlier scored one answer higher.
The cache had frozen a value and held it for the whole block.

Mechanism, all verified in source:

- `config.py`: `recall_packet_cache_ttl_seconds = 300.0`, enabled by default,
  **persistence enabled** (SQLite sidecar, survives restart).
- `retrieval/packet_cache.py` `build_key`:
  `f"{group_id}:{scope}:{digest(topic_hint)}:{digest(project_path)}"`.

Two consequences:

1. **Repeated-measures noise is understated.** Any rig that loops the same
   queries faster than 300 s measures the cache. Within-block variance is a
   *floor* on the real noise, not an estimate of it. This is very likely part of
   why `engram battery` produced 0-4 spreads: back-to-back runs were partly
   cache replays, minutes-apart runs were not.
2. **The key has no build/config/arm component.** In an A/B where the arms
   differ only by server configuration — the shape of the graph experiment's
   arms A and B — **arm B can be served arm A's cached packets for the same
   query**. The A/B would report "no difference" for a change of any size, with
   a clean-looking low-variance number. This is the most expensive failure mode
   in this ledger, because it makes a *null* look rigorous.

This is pattern 5 (environment drift) with the drift inside the server rather
than the harness: the measurement describes cache state, not retrieval.

**Mitigation shipped:** `engram meter` excludes cache-served probes, refuses
captures whose per-question probe spacing is inside `--cache-ttl-s`, and refuses
captures with no timing/cache provenance at all. **Mitigation NOT shipped:** the
TTL is not exposed on `/api/knowledge/runtime`, so the guard uses the
compile-time default; and nothing prevents an A/B written outside the meter from
walking into consequence (2). Any future A/B must space probes beyond the TTL,
clear the cache between arms, vary `group_id`/`project_path` per arm, or run
with `recall_packet_cache_enabled=False`.

Detail: `docs/product/experiments/RECALL_METER.md` §4.

### AUDIT-15 — the declared engine config is not the effective one · HIGH · **FIXED**

Found by lane 2 of the code census, fixed 2026-07-24 (ticket #27).

Three git-tracked copies of `config.hx.json` declared `ef_search 512`,
`ef_construction 200`, `db_max_size_gb 50`, `mcp false`, plus a
`vector_config.db_max_size: 50` that is **not a field of `VectorConfig`** and
was dropped by serde on every parse. The values the engine actually runs on are
the generated Rust literals in `fn config()` at
`native/helix-repo/helix-python/src/queries.rs:98-107` — `768 / 128 / 20 / true`
— read directly by `HelixEngine::new` (`helix-python/src/lib.rs:53`).
`NativeTransport.initialize()` (`storage/helix/native_transport.py:106-160`)
passes no config path at all, so **no runtime has ever read the JSON**.

Two things make this instrument dishonesty rather than mere duplication:

1. **It reports a false headroom.** The live brain was assessed at 84.8% of
   map_size before compaction. That is 84.8% of the *real* 20 GB; the file an
   operator would read says 50 GB. Anyone sizing future growth off it is wrong
   by **2.5× on the exact axis that causes `MDB_MAP_FULL`**.
2. **It absorbs edits silently.** `RECALL_PERFORMANCE_PLAN.md` M1 and ledger
   ticket 4 both aimed an `ef_search` change at `config.hx.json:5`. That edit
   compiles, commits, deploys, and does nothing.

The effective numbers are also not choices: `768 / 128 / 16` are the helix-db
library defaults (`helix-db/.../traversal_core/config.rs:17-21`). The 512/200
in the JSON tracked `EmbeddingConfig.hnsw_ef_construction = 200` in
`server/engram/config.py:118` — two *declared* surfaces kept in sync with each
other and never with the runtime.

**Fix shipped:** all three copies now declare the effective values, carry a
leading `_authority` key naming `queries.rs` and the required
`make build-native`, and have lost the phantom `db_max_size` key.
`server/tests/test_native_config_authority.py` fails when declared and effective
diverge in **either** direction, when the copies differ from each other, when a
key appears that the Rust `Config`/`VectorConfig` structs cannot honour, or when
a fourth copy is added. Not fixed (needs a rebuild, deliberately deferred): the
Rust still does not read the JSON, so this is a *pinned* duplication, not a
single source of truth.

### AUDIT-16 — half of the Helix schema contract had been skipping for months · HIGH · **FIXED**

`server/tests/test_helix_schema_contract.py:13` pointed `NATIVE_GENERATED_QUERIES`
at `helixdb-cfg/.helix/dev/helix-repo-copy/helix-container/src/queries.rs`. That
path does not exist — the staged artifact is one directory up, at
`.helix/dev/helix-container/src/queries.rs` — and `.helix/` is gitignored, so it
could never exist on a fresh clone or in CI either. Both helper functions
responded with `pytest.skip`, so **8 of the file's 16 tests never ran**:

```
$ uv run pytest tests/test_helix_schema_contract.py -q -rs
.s.s.s.s..ss.s.s
SKIPPED [6] ... Generated Helix Rust queries are unavailable
SKIPPED [2] ... /Users/.../helix-repo-copy/helix-container/src/queries.rs
8 passed, 8 skipped
```

Every assertion about the generated PyO3 bindings — entity provenance fields,
cue feedback fields, the graph-embed delete route, the candidate/vector routes,
the bounded stats routes — was green and vacuous. This is the AUDIT-11 shape
(a guard nobody reads) applied to a test: a skip is an absence, and the CI
summary laundered it into a pass.

**Fix shipped:** repointed at the git-tracked 620 KB
`native/helix-repo/helix-python/src/queries.rs` — the copy maturin actually
compiles — and replaced both `pytest.skip` branches with hard assertions, since
a tracked file's absence is a broken checkout, not a missing optional artifact.
16/16 now run. Proved non-vacuous by renaming one field in the generated Rust
(`Entity.name` → `Entity.nAme`), which turns them red.

**Generalisation worth acting on:** `pytest.skip` on a path is an unlabelled
absence with the same failure mode as a zero-filled metric. Any skip guarding a
**git-tracked** path is a latent AUDIT-16.

---

## The five patterns

Everything above is one of five shapes. Grep for these when adding a metric:

1. **Literal-as-measurement** — a metric field assigned `0` / `[]` / `{}` / `""`
   and returned alongside genuinely computed fields.
   *(AUDIT-2, 3, 8, 9, 10, 11, 12)*
2. **Unlabelled extrapolation** — sample, scale, report as a count.
   *(AUDIT-1)*
3. **Reader/writer field drift** — a gate reads field X while the writer moved
   to field Y; the gate reads a permanent zero and can never fire.
   *(AUDIT-4, AUDIT-11)*
4. **Dual-method divergence** — two paths report the same named quantity by
   different methods, with no reconciliation and no way for a caller to know
   which it got. *(AUDIT-5, and AUDIT-1 vs AUDIT-8 on `relationships`)*
5. **Environment drift** — the measurement describes ambient state rather than
   the system under test: config resolved from process env (AUDIT-6, AUDIT-7),
   or a result replayed from a cache the rig cannot see (AUDIT-14).

---

## Enforcement

`server/tests/test_metric_honesty_contract.py` — an AST contract modelled on the
existing `test_storage_silent_swallow_contract.py`.

It walks the dict literals built by six metric surfaces
(`storage/helix/graph.py`, `storage/sqlite/graph.py`, `storage/diagnostics.py`,
`retrieval/graph_state.py`, `retrieval/runtime_state.py`,
`lifecycle_summary.py`) and fails when a **metric-named key** (`*_count`,
`*_rate`, `avg_*`, `totalMs`, …) is bound to a **zeroish literal** inside a dict
that **also carries computed values**.

Three deliberate design choices:

- **`None` is legal.** It is the encoding the rule asks for. Only `0`, `0.0`,
  `False`, `[]`, `{}` — values that read as a real measurement — fail.
- **All-literal dicts pass.** `{"count": 0, "avg": 0.0}` returned for empty
  input is an honest constant, not a fabrication. The dangerous shape is the
  *mixture*: a function that had data and faked one field.
- **The ledger is keyed by `(file, metric_key)`, not line number.** Line numbers
  drifted 11 lines during the writing of this document.

Two escape hatches, both explicit and both reviewed decisions:

- `# metric-ok: <reason>` beside the line — for a literal that is genuinely
  correct there.
- `KNOWN_FABRICATIONS` — the ledger of the confirmed-but-unfixed sites above.
  **It may only shrink.** `test_ledger_has_no_stale_entries` fails when a
  ledgered site stops fabricating, which forces both the ledger and this
  document to be updated at the moment of the fix. Without that test the
  allowlist would rot into a permanent amnesty.

`test_scanner_is_not_inert` is the canary, and it is the most important test in
the file. This project's dominant bug class is code that runs and whose result
is discarded; a contract test that silently stops matching anything is the same
disease wearing a green checkmark. The canary asserts the scanner still fires on
a synthetic fabrication and still spares `None`.

That it is load-bearing was proved, not assumed. Neutering the matcher:

```
NEUTER (is_fabricated_measurement never matches constants):
  test_metric_surfaces_do_not_fabricate_measurements  PASSED   <-- silently vacuous
  test_scanner_is_not_inert                           FAILED   <-- caught it
  test_ledger_has_no_stale_entries                    FAILED   <-- caught it
RESTORED: 3 passed
```

The contract test alone would have gone green on a dead scanner. The canary is
what makes the suite honest about itself.

---

## What this audit did not resolve

- **AUDIT-13's 76 handlers are unaudited.** Counting them is not triaging them.
  The claim made here is only "the existing contract does not cover this tree,"
  not "there are 76 bugs."
- **No live numbers were taken.** The shell was owned by a concurrent latency
  measurement for the duration of this audit; every claim above is from source,
  from `~/.engram/.env` and the LaunchAgent plist on disk, or from an in-process
  reproduction. The 750-vs-1637 and 9055-vs-756 figures are carried over from
  the reports that opened this task and were **not** independently re-measured —
  their *mechanisms* were confirmed in source, their *values* were not.
- **AUDIT-4 was diagnosed, not fixed** — lane 2 owns those files.
- **The contract covers six modules.** `evaluation/dogfood.py`, `axi/hooks.py`,
  `retrieval/recall_surface.py` and `mcp/adoption_cli.py` all contain
  literal-metric shapes; most carry a `status` field and are probably legitimate,
  but they were reviewed by eye, not enforced. Widening the surface list is the
  obvious next increment.
- **The dashboard is not covered by any contract test.** AUDIT-12 was found by
  grep. A TypeScript equivalent of this contract does not exist.
- **The rule is not yet applied to the things it names as fixes.** No metric in
  the ledger has been converted to `null` — and per AUDIT-12, doing so before the
  UI is fixed would make the reporting worse. Fix order: readers first, then
  writers.
