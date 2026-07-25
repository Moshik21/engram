# The Recall Meter — a ruler that can resolve a ±1-answer change

**Task #18.** `engram battery` cannot measure a retrieval change. This document
describes the replacement instrument, the design decisions behind each rule, the
derivation of its minimum sample size, and what it found about the *server* while
being built.

`engram battery` is **not** deleted. It and its callers are untouched; the meter
is a second instrument, and the meter's rig carries the battery's ten questions
verbatim so both scoring rules can be applied to one capture.

- Code: `server/engram/evaluation/meter.py`
- Rig: `server/tests/rigs/recall_meter_rig.json`
- Tests: `server/tests/test_recall_meter.py`
- CLI: `engram meter --against-live --runs N --run-gap-s 310 --capture out.json`

---

## 1. What was wrong with the battery

Two independent failures, both established before this work started:

| | Failure | Evidence |
|---|---|---|
| **Variance** | Four rescue lanes race the wall clock and a different one wins each run. Identical code at one HEAD scored `0,0,0` then `2,1,1`; and `3,4,4,4,4 / 4,4,4,4,4 / 1,4,4`. Medians 5 and 3 twenty minutes apart. | Session record, task #18 |
| **Structure** | A question is a HIT only when *every* token of one answer group lands inside *one* top-3 row (`battery.py:98-100` + `:111-114`). An answer assembled from two rows is a MISS **by construction** — the exact multi-hop case the graph experiment exists to detect. | GRAPH_THESIS.md M16 |

The reranker knob has been mis-measured three times and the graph tier six.
A single aggregate from a single run cannot separate a +1 effect from lane noise.

---

## 2. What the meter does differently

### 2.1 Capture, then score

A run issues read-only `GET /api/knowledge/recall` probes and writes the raw
rows to a JSON capture. Scoring is a **pure function** of that capture. One
capture can therefore be rescored under any rule — different `k`, different
`max_sources`, or the battery's own one-row rule — with no new load on the
server and no risk that two rules saw two different retrievals.

This is what makes the head-to-head against the battery exact rather than
approximate: `hitSingle` at `k=3` over a `limit=3` capture *is* the battery's
score, computed from the same bytes. `result_texts` imports the battery's own
`_flatten_text`, and a contract test asserts the two agree.

### 2.2 Multi-source scoring — the most important rule

An answer group is satisfied when the smallest set of returned rows jointly
containing all its tokens has size ≤ `max_sources` (default 2). The cover is
**exact, not greedy** (groups are ≤4 tokens, rows ≤10, so exhaustive search is
free), and ties break toward lower row indexes so a reported cover of `(0,1)` is
genuinely the highest-ranked way to assemble the answer.

**The objection this rule has to survive.** `battery.py` carries a verify-pass
comment explaining why it does *not* join texts: "joining texts would let
multi-token groups hit on tokens split across unrelated results." That is a real
false-positive channel, and loosening the rule without a defence would trade a
known blindness for an unknown inflation. Three defences, in order of strength:

1. **Both rules are always reported.** `hitUnion` never appears without
   `hitSingle` beside it. The looser rule cannot silently inflate a headline,
   because the stricter number is printed on the same line.
2. **The cover is bounded.** Two rows is an assembled answer; five rows is
   coincidence. `max_sources` is the knob and it is in the report.
3. **The false-positive rate is measured, not assumed.** A shuffled control
   scores every question's tokens against a *different* question's rows
   (rotation by one within each run — a derangement for any rig with ≥2
   questions) and reports the union rule's false-positive rate alongside the
   one-row rule's. Measured on two live captures: `unionRate 0.143` vs
   `singleRate 0.071` (12-run block, 168 trials) and `0.143` vs `0.125`
   (8-run spaced, 112 trials). The union rule costs between 2 and 7 percentage
   points of extra coincidence on this rig — real, small, and *reported* rather
   than assumed away.

The control is the part that makes the rule defensible. Without it this would
be one more loosened threshold justified by an argument.

### 2.3 Lane attribution — making the race visible

Each probe records `lifecycle.fallbackStatus`, which names the lane that
produced the answer: `not_run` (deep pipeline), `filtered`, `fast_preflight_hit`,
`durable_entity_first`, `durable_entity_rescue`,
`durable_entity_rescue_after_timeout`, `partial_on_timeout`,
`project_file_recall_fallback`, `context_packet_fallback`, `cache_satisfied`.

The report gives a per-question lane histogram and a `laneStable` flag, so a
score flip is attributable rather than mysterious.

**Why visibility rather than pinning.** The brief allowed either. Pinning the
deep lane requires a server config change and a restart — both forbidden here,
and both would measure a configuration nobody ships. Visibility is also the
better instrument: on the live brain the lane determines not just the *ranking*
but the *kind of row*. `fast_preflight_hit` returns compact `cue_episode` stubs
(a `mentions:`/`quotes:` digest); the deep lane returns full episode text. A
question can hit under one lane and miss under the other for reasons that have
nothing to do with the retriever under test. Pinning would have hidden that.

### 2.4 Honest refusal

The report carries `status` ∈ `{resolved, unresolved, degraded, empty}` and a
list of `refusals`. When it is not `resolved`, the formatter prints
`NO HEADLINE SCORE` and the per-question table, and withholds the aggregate
verdict. Refusal conditions:

- fewer than `MIN_RUNS` (3) runs, or fewer than 2 *complete* runs;
- error or degraded probe fraction above 10%;
- any question with no usable rows — reported as `excluded`, **not** scored 0
  (AUDIT-11's shape: "we did not look" must never render as "there is nothing");
- any cache-served probe, or repeats spaced inside the packet-cache TTL (§4);
- a capture lacking timing/cache provenance;
- the observed variance requiring more runs than were taken (§3).

Per-question means, SDs, lane histograms and per-run flags are always emitted:
those are *data*, and they are accurate. What is withheld is the comparison
verdict, which is what the audit rule is about.

---

## 3. The minimum-N derivation

Let `S_r` be the per-run total over the scored questions, and `σ̂` its
**empirical** standard deviation across runs. To detect a difference of `Δ`
answers between two independent arms at two-sided α=0.05 and power 0.80, the
difference of arm means has standard error `σ√(2/N)`, so

```
N ≥ 2 σ̂² (z_{1-α/2} + z_{1-β})² / Δ²  ≈  15.70 σ̂² / Δ²      (Δ in answers)
```

Three deliberate choices:

- **σ̂ is empirical, not a binomial sum.** `Var(S) = Σ_q p_q(1-p_q)` would assume
  the questions are independent within a run. They are not: the dominant noise
  source is the lane race, which is shared across the whole run. The
  independence formula would understate the variance and therefore understate N.
- **A conservative variant is reported.** σ̂ estimated from 8 runs is itself
  noisy, so the report also gives `σ̂ · sqrt((n-1)/χ²_{0.05,n-1})`, the 95% upper
  confidence bound, and derives `minRunsPerArmConservative` from it. The
  comparability verdict uses the conservative number. Reporting only the point
  estimate would be an unlabelled approximation — the exact shape of AUDIT-1.
- **The χ² quantile is computed, not looked up.** `chi2_quantile` inverts the
  regularized lower incomplete gamma by bisection; the published table is used
  as the *test*, not the implementation, so the derivation works for any df.
- **`σ̂ = 0` is handled separately, and it is the case that bites.** Normal
  theory multiplies zero by a constant and returns zero, so an instrument that
  simply applied the formula would certify "N ≥ 2 runs/arm" off a sample that
  merely never happened to flip — the exact over-claim this lane exists to
  prevent. Observing no variation in n runs does not mean there is none. The
  **rule of three** bounds the rate of an event never seen in n trials at `3/n`
  with 95% confidence; treating a flip as a ±1 change in the total bounds the
  variance by that rate, so `σ ≤ √(3/n)`. Twelve identical runs therefore
  license "N ≥ 4 per arm", and four identical runs license nothing
  (`√(3/4) = 0.87` → N ≥ 12).

The unpaired formula is conservative for an interleaved A/B. If both arms are
run within the same pass against the same server state, the paired SD `σ_d` is
smaller and N drops accordingly — but `σ_d` cannot be estimated from a
single-arm capture, so the meter reports the unpaired number.

---

## 4. What the meter found while being built: the packet cache

This was not in the brief and it is the most consequential result of the lane.

The first live capture — 12 back-to-back passes over 14 questions, 168 probes —
reported:

```
# Recall meter: RESOLVED — 3.0 +/- 0.0 of 14 over 12 runs
- resolving a 1.0-answer difference needs N >= 2 runs/arm; have 12
- per-run totals (union rule): [3,3,3,3,3,3,3,3,3,3,3,3]
- lanes that won: {filtered: 82, fast_preflight_hit: 49, cache_satisfied: 24,
                   durable_entity_first: 12, partial_on_timeout: 1}
```

σ = 0.0 over 12 runs. **That number is wrong, and it is wrong in the most
dangerous direction** — it certifies a resolving power the instrument does not
have. Two facts contradict it:

1. `cache_satisfied` served **24/168 probes** — two questions, on all 12 runs.
2. The same rig, run once about ten minutes earlier at the same HEAD, scored
   **4**, with `deleted-phases` a HIT. In the 12-run block `deleted-phases`
   scored 0 on all twelve, served by `cache_satisfied` every time. The cache had
   frozen one value and held it for the whole block.

Root cause, verified in source:

- `config.py`: `recall_packet_cache_ttl_seconds = 300.0`,
  `recall_packet_cache_enabled = True`,
  `recall_packet_cache_persistence_enabled = True`.
- `retrieval/packet_cache.py` `build_key`:
  `f"{group_id}:{scope}:{digest(topic_hint)}:{digest(project_path)}"`.
- Live `/api/knowledge/runtime` at the time: `packetCache.hit_count = 117`,
  `entry_count = 21`, `persistent: true`, sidecar at
  `~/.helix/engram-native-dogfood-axi/packet-cache.sqlite3`.

Two consequences, and the second is worse than the first:

> **(a)** Any repeated-measures instrument that loops the same queries faster
> than 300 s measures the cache. Within-block variance is a **floor** on the
> real measurement noise, not an estimate of it.
>
> **(b) The cache key contains no build, config or arm component, and it is
> persisted across restarts.** In an A/B where the arms differ only by server
> configuration — which is exactly the shape of the graph experiment's arms A
> and B — **arm B can be served arm A's cached packets for the same query.** The
> A/B would then report "no difference" for a change of any size, and would do
> so with a clean-looking, low-variance number.

This is the same disease as every other entry in `INSTRUMENT_AUDIT.md`: a
plausible number that ends investigation. It would have been invisible without
lane attribution, and it explains part of the battery's own mystery — its 0-4
spread came from runs separated by minutes (cache expired) versus back-to-back
runs (cache warm).

**What the meter does about it.** Cache-served probes are excluded from the hit
counts (a replay is not a sample); a question served only from cache is
`excluded: "cache_served"`; per-question probe spacing is measured and a median
gap below `--cache-ttl-s` is a refusal; and a capture without timing/cache
provenance is refused outright rather than silently scored. That last rule was
added after noticing that the *first* capture — taken before the guard existed —
would otherwise have sailed through the new scorer reporting σ=0. A guard that
is present and inert is this project's dominant bug class.

**CORRECTED 2026-07-24 (ticket #29).** Consequence (b) is fixed at the source:
the key now carries an identity fingerprint,
`pc2:<fingerprint>:<group>:<scope>:<topic>:<project>`, over the whole activation
config (minus five cache-plumbing fields), the runtime mode, the package
version and a digest of `engram/retrieval/**` + `engram/pipeline.py`. Two arms
that differ by any of those cannot share an entry, in memory or through the
SQLite sidecar, and pre-`pc2` rows are purged on load rather than left where the
degraded-fallback lane could serve them. Consequence (a) is unchanged — a repeat
inside the TTL is still a replay.

The meter no longer *assumes* any of this. `capture_runs` reads
`/api/knowledge/runtime/fast` before probing and records `serverCache`
(`fingerprint`, `ttlSeconds`, `enabled`, `keySchema`) in the capture;
`score_capture` enforces the **server-reported** TTL rather than
`--cache-ttl-s`, and prints the fingerprint on every report. **Two arm reports
showing the same fingerprint were not isolated by the key.**

**What any future A/B must do**, at minimum one of:
let the fingerprint separate the arms and *verify* it by comparing the two
reports' `fingerprint` lines; set `ENGRAM_PACKET_CACHE_NAMESPACE` per arm when
the arms differ by neither config nor code (planted corpora); space probes
beyond the TTL; or run with `recall_packet_cache_enabled=False`, which the
report confirms as `bypassed=yes` (and which then makes fast repeats
legitimate — the spacing refusal is skipped, because a cache that is off cannot
replay). This belongs in the graph experiment's VOID pre-flight.

---

## 5. The live result — and it is not the result that was expected

**8 passes over 14 questions, 112 read-only probes, spaced 310 s apart (above the
300 s TTL), 2026-07-24 17:31–18:10 local, against the live dogfood shell.**

```
# Recall meter: UNRESOLVED — NO HEADLINE SCORE
- resolving a 1.0-answer difference needs N >= 105 runs/arm
  (point estimate from the observed sd: 29); have 7
- probes: 112 (0 error, 2 degraded, 3 cache-replayed)
- per-run totals (union rule):            [4, 4, 4, 5, 5, 7, 7]
- per-run totals (battery one-row rule):  [4, 4, 4, 5, 5, 7, 7]  mean 5.14  sd 1.345
- order effect: first half 4.0 -> second half 6.0
```

### 5.1 The brief expected a stable signal. There isn't one — and that is the finding

The hypothesis handed to this lane was that the meter would report a stable
per-question signal where `engram battery` reports 0-4. **Partly true, and the
false part is more important.**

- **9 of 14 questions were perfectly stable** across all 8 spaced runs
  (p = 0.0 or 1.0): `ts-kill`, `north-star`, `fastembed-outage`,
  `vector-write-path`, `founder-identity`, and all four multi-source questions.
- **5 flipped**, and the aggregate moved **4 → 7** over the window.

The battery would have reported "4" one hour and "7" the next with no
explanation, and the difference is 3 answers — three times the effect size the
graph experiment is trying to detect.

### 5.2 Every flip is lane-attributable

This is what the lane column buys, and it is unambiguous:

| question | per-run (lane → hit) |
|---|---|
| `flip-condition` | filtered ✗ ✗ · fast_preflight_hit ✗ ✗ · durable_entity_rescue_after_timeout ✓ ✓ · durable_entity_first ✓ ✓ |
| `durable-lane` | filtered ✗ ×6 · fast_preflight_hit ✓ ✓ |
| `bm25-breaker` | filtered ✗ ×5, timeout ✗ · fast_preflight_hit ✓ ✓ |
| `recall-outage` | filtered ✗ ×5 · fast_preflight_hit ✓ ✓ |
| `deleted-phases` | filtered ✓ ×5 · fast_preflight_hit ✗ ✗ |

Not one flip is unexplained. The lane predicts the outcome in every case, and
`deleted-phases` flips the *other* way, which rules out "the brain simply got
better." Lane hit rates over the whole capture:

| lane | hit rate | mean rows returned |
|---|---|---|
| `durable_entity_rescue_after_timeout` | 2/2 | 1.00 |
| `durable_entity_first` | 10/14 = 0.71 | 2.14 |
| `fast_preflight_hit` | 17/52 = 0.33 | 2.02 |
| `filtered` | 10/37 = 0.27 | 3.00 |
| `timeout` | 0/4 = 0.00 | 3.00 |

The lanes do not merely reorder — they return **different numbers of rows of a
different kind**. `fast_preflight_hit` returns ~2 compact `cue_episode` stubs
(a `mentions:`/`quotes:` digest); `filtered` runs the deep pipeline and returns
3 full episode texts.

### 5.3 The variance is a step change, not noise — and no N fixes that

Per-run lane mixes:

```
run 0 17:31  filtered 7, fast_preflight_hit 6, durable_entity_first 1
run 1 17:36  filtered 8, fast_preflight_hit 5, durable_entity_first 1
run 2 17:42  fast_preflight_hit 5, cache_satisfied 3, timeout 3, filtered 2, durable 1
run 3 17:47  fast_preflight_hit 6, filtered 6, timeout 1, durable 1
run 4 17:53  filtered 7, fast_preflight_hit 5, durable 1, durable_rescue_after_timeout 1
run 5 17:58  filtered 7, fast_preflight_hit 5, durable 1, durable_rescue_after_timeout 1
run 6 18:04  fast_preflight_hit 10, durable_entity_first 4        <-- filtered GONE
run 7 18:09  fast_preflight_hit 10, durable_entity_first 4
```

Between 17:58 and 18:04 the `filtered` lane **disappears entirely** and the mix
collapses from four lanes to two. The score steps 5 → 7 in the same interval.
The per-run sequence `[4,4,4,5,5,7,7]` is monotone non-decreasing — a regime
change, not dispersion around a mean.

**Inference, labelled as such:** a concurrent workflow owned `pipeline.py` and
`config.py` during this window and restarts the shell. A step change in lane
mix at a single boundary is what a new build looks like. I did not verify the
shell PID across the boundary — `/api/knowledge/runtime` does not expose
`startedAt` at the path I probed — so this is a strong inference, not a
measurement.

**The consequence for the experiment is the important part.** The reported
"N ≥ 105 runs/arm" is computed from an SD that conflates a trend with noise. It
is therefore the *wrong* remedy in both directions: it overstates the runs
needed for i.i.d. lane jitter, and it understates the problem, because **no N
resolves a regime change** — running arm A before and arm B after a server
change measures the change, not the arm. The fix is arm isolation: a frozen
build, a clone, and interleaved arms within a run. The meter's refusal is what
makes that unavoidable instead of optional.

### 5.4 The union rule never fired on live data

`hitUnion == hitSingle` on all 112 probes; `multiSourceRuns = 0` for every
question, including the four conjunctive ones (all four scored 0.0). Stated
plainly: **on this corpus the multi-source rule is proven by unit test and inert
in production.** That is not a defect of the rule — it corroborates
GRAPH_THESIS M2/M4 from a different direction. No answer on this brain is
currently assembled from two rows, because nothing puts the two halves in the
top 3 together. If the graph experiment ever produces a positive result, this is
the rule that will be able to see it; today there is nothing to see.

The shuffled control on the same capture: union false-positive rate 0.143,
one-row 0.125. The union rule adds ~1.8 percentage points of coincidence here —
a real but small cost, measured rather than assumed.

### 5.5 Verdict on the live brain as an A/B substrate

The dogfood shell is **not** a usable substrate for the graph experiment:
four rescue lanes with hit rates from 0.00 to 1.00, a lane mix that changed
mid-capture, a packet cache that fakes determinism, and probes that mutate the
state they measure. The meter says so instead of producing a number. Use the
lite planted-corpus rig on a clone, as GRAPH_THESIS §5 already specifies.

---

## 6. Proof that the instrument is not inert

The canary is `test_multi_source_answer_scores_hit`. It was proved capable of
failing before it was committed:

```
NEUTER (minimal_cover ceiling forced to 1 — the battery's one-row rule):
  test_multi_source_answer_scores_hit   FAILED   <-- "multi-source rule is DEAD"
  test_cover_is_bounded                 FAILED
  test_k_truncates_before_scoring       FAILED
  3 failed, 33 passed

RESTORED: 36 passed
```

The same file asserts, positively, that the battery scores the identical fixture
a MISS (`test_battery_scores_the_same_case_miss`) — so the difference between
the two instruments is a test, not a claim.

The cache guard was likewise proved live rather than by fixture: rescoring the
real 12-run capture under the new scorer produces
`REFUSAL: 168/168 probes lack timing/cache provenance`.

---

## 7. The rig

Fourteen questions.

- **Ten `single_source`** — copied verbatim from
  `tests/rigs/agent_experience_battery.json`. A test asserts the copy stays
  byte-identical, so the head-to-head cannot silently drift.
- **Four `multi_source`** — each is a conjunction of two battery questions, and
  each answer group draws one token from each parent. A test machine-verifies
  that every token traces to a named parent and that no group draws both halves
  from the same one. This is provenance by construction rather than by prose.

**The honest weakness of the multi-source set**, stated because it bounds what
the instrument can prove: a conjunctive question ("what is X and what is Y") is a
*weaker* multi-hop probe than a natural bridge question ("query names A; the
answer is about B; only an edge connects them"). It requires two facts rather
than one hop to reach the second. It is what can be built from validated ground
truth on the live brain today, where the extraction outage means natural bridges
barely exist. A natural bridge set needs the planted-corpus rig (M3.1), not the
dogfood brain.

Deriving the multi-source questions was done by probing the live brain
read-only and rejecting candidates whose tokens did not appear at all — no
fixture entry was assembled from imagination. The probe transcripts are in the
session scratchpad.

---

## 8. Known limits

- **Recall is not side-effect free.** `GET /api/knowledge/recall` records access,
  priming and fingerprints. Repeated measurement against a live brain is not
  idempotent. The report includes the per-run score sequence and a
  first-half/second-half split so order effects are visible rather than averaged
  away, but it cannot remove them. A clone is the right substrate for an A/B.
- **Containment is still the judge.** The meter changes *how many rows* an answer
  may come from, not *what counts as an answer*. Substring containment remains a
  proxy for "the agent could have answered," and it has all the usual failure
  modes (a token present in a hostile context still counts). Fixing that needs a
  judge, which needs a local model, which is a separate lane.
- **`max_sources=2` is a knob, i.e. a frozen question.** The right long-term
  answer is probably a graded score in the number of sources rather than a
  binary, but a graded score needs a validated weighting and none exists. The
  binary is reported *with* the source histogram so the graded version can be
  computed later from existing captures.
- **The shuffled control is a rotation, not a full permutation test.** It gives
  one false-positive estimate per capture, not a distribution. A full permutation
  would be strictly better and is nearly free from an existing capture.
- **`minRunsPerArm` assumes i.i.d. runs and the live data violates that.** The
  8-run capture's SD is dominated by a single step change (§5.3), so the derived
  N is the right formula applied to the wrong noise model. The instrument does
  not currently test for a trend; it reports the per-run sequence and a
  half-split so a human can see one. A changepoint or lag-1 autocorrelation test
  on `perRunUnion` would let it *say* "this is drift, more runs will not help"
  instead of leaving that to the reader. That is the single highest-value next
  increment, and it is why §5.3's conclusion had to be written by hand.
- ~~**The 300 s TTL is a compile-time default, not read from the live server.**~~
  **CLOSED 2026-07-24 (ticket #29).** `stats.packetCache` on
  `/api/knowledge/runtime` and `/runtime/fast` now carries `ttl_seconds`,
  `fingerprint`, `key_schema`, `enabled` and `identity`; the meter reads them at
  capture time and reports `ttlSource: server`. `--cache-ttl-s` survives as the
  fallback for a server that predates the change (`provenanceStatus:
  unreported`), where the conservative default still applies — that can only
  cause a refusal, never a false pass.
