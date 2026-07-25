"""Recall meter — a retrieval instrument that can resolve a +/-1-answer change.

WHY THIS EXISTS (task #18). ``engram battery`` cannot measure a retrieval
change. Two independent demonstrations, 2026-07-24:

* **Variance.** Identical code at one HEAD scored ``0,0,0`` warm and ``2,1,1``
  cold in one session, and ``3,4,4,4,4 / 4,4,4,4,4 / 1,4,4`` in another. Four
  rescue lanes race the wall clock and a different one wins each run, so the
  same build produced medians 5 and 3 twenty minutes apart. A single aggregate
  from a single run cannot distinguish a +1 effect from lane noise.
* **Structure.** ``battery.py`` scores a question HIT only when *every* token of
  one answer group lands inside *one* top-3 row (``group_contained`` under
  ``any(... for text in result_texts)``). An answer assembled from TWO rows is a
  MISS by construction — which is exactly the multi-hop case the graph
  experiment exists to detect. The instrument is blind to the phenomenon under
  test (GRAPH_THESIS.md M16).

``engram battery`` is NOT replaced or deleted; it and its callers stay intact.
This is a second instrument with four properties the battery lacks:

1. **Per-question hit rate over N runs**, with per-question and per-run variance,
   and a *derived* minimum N for resolving a 1-answer difference at the observed
   variance (:func:`min_runs_per_arm`) rather than a guessed one.
2. **Multi-source scoring.** An answer whose tokens span up to
   ``max_sources`` returned rows can score a HIT (:func:`minimal_cover`). The
   battery's one-row rule is retained *alongside* it as ``hitSingle`` so the two
   are always reported side by side and the looser rule can never silently
   inflate a number. A shuffled control (:func:`_control_false_positives`)
   measures the union rule's false-positive rate instead of assuming it away.
3. **Lane attribution.** ``lifecycle.fallbackStatus`` names which of the racing
   rescue lanes produced each answer, recorded per question per run, so a flip
   is attributable rather than mysterious. (Pinning the deep lane would require
   a server config change and a restart; both are out of scope here, so the race
   is made *visible* instead.)
4. **Honest refusal.** When a run set cannot resolve a 1-answer difference — too
   few runs, too many degraded/errored queries, questions returning no rows —
   the report says so and the formatter prints a REFUSAL banner instead of a
   headline score. Per INSTRUMENT_AUDIT.md: a metric must be either accurate or
   absent, never plausible-but-wrong.
5. **A cache guard**, found while building this and arguably the most important
   part. The first live capture (12 back-to-back passes, 2026-07-24) reported
   ``sd = 0.0`` and "N >= 2 runs/arm" — a resolving power the instrument does not
   have. ``cache_satisfied`` served 24/168 probes, and the same rig ten minutes
   earlier had scored one answer higher. Repeating a query inside
   ``recall_packet_cache_ttl_seconds`` measures the cache, so cache-served
   probes are excluded and under-spaced repeats are refused. See
   :data:`DEFAULT_CACHE_TTL_S` for the A/B consequence, which is worse.

**Capture then score.** A run captures raw rows to JSON; scoring is a pure
function of that capture. One live capture can therefore be rescored under
different rules (k, max_sources, the battery's own rule) with no extra load on
the server and no risk that the two rules saw different retrievals.

**Side effects, stated.** ``GET /api/knowledge/recall`` is a read endpoint but
the recall path records access, priming and fingerprints. Repeated measurement
against a live brain is therefore not perfectly idempotent, and the report
includes the per-run score sequence plus a first-half/second-half split so any
drift across runs is visible rather than averaged away.
"""

from __future__ import annotations

import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from engram.evaluation.battery import _flatten_text

RIG_PATH = Path(__file__).resolve().parents[2] / "tests" / "rigs" / "recall_meter_rig.json"

CAPTURE_SCHEMA = "engram.recall_meter.capture/1"
REPORT_SCHEMA = "engram.recall_meter.report/1"

DEFAULT_LIMIT = 3
DEFAULT_K = 3
DEFAULT_MAX_SOURCES = 2
MIN_RUNS = 3
MAX_DEGRADED_FRACTION = 0.10

# Lanes that replay a cached packet instead of retrieving.
CACHE_LANES = frozenset({"cache_satisfied"})

# Fallback for the server's ``recall_packet_cache_ttl_seconds`` (config.py),
# used ONLY when the server does not report its own. Repeating a query inside
# the TTL measures the cache, not recall.
#
# The second consequence used to be worse: the key was
# ``group_id:scope:digest(query):digest(project_path)`` with NO build or config
# component and SQLite persistence across restarts, so in an A/B **arm B could
# be served arm A's cached packets** and the run would report "no difference"
# for a change of any size. Fixed 2026-07-24 (ticket #29): the key now carries
# an identity fingerprint over the activation config, the runtime mode, the
# build and the packet-building source. This instrument no longer has to assume
# any of that — :func:`fetch_cache_provenance` reads the live identity, TTL and
# enabled flag off ``/api/knowledge/runtime/fast`` and records them in the
# capture, so cache independence is VERIFIED rather than hoped for. Two arms
# whose reports print the same ``fingerprint`` were not isolated by the key.
DEFAULT_CACHE_TTL_S = 300.0

# Where the server reports its packet-cache identity. Startup-safe: it does no
# graph or artifact reads.
RUNTIME_FAST_PATH = "/api/knowledge/runtime/fast"

# --------------------------------------------------------------------------
# rig
# --------------------------------------------------------------------------


def load_rig(path: Path | None = None) -> dict[str, Any]:
    with open(path or RIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def rig_questions(rig: dict[str, Any]) -> list[dict[str, Any]]:
    return list(rig.get("questions") or [])


# --------------------------------------------------------------------------
# text extraction (shared with the battery, by import, so the two rules are
# provably scoring the same characters)
# --------------------------------------------------------------------------


def result_texts(payload: dict[str, Any], limit: int) -> list[str]:
    """Flatten the first ``limit`` surfaced rows to text blobs.

    Row selection and flattening are the battery's, imported rather than
    reimplemented; ``tests/test_recall_meter.py`` asserts the two agree.
    """
    rows = payload.get("items") or payload.get("results") or []
    if not rows:
        rows = payload.get("packets") or payload.get("cached_packets") or []
    texts: list[str] = []
    for row in list(rows)[:limit]:
        parts: list[str] = []
        _flatten_text(row, parts)
        texts.append("\n".join(parts))
    return texts


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def token_rows(token: str, row_texts: Sequence[str]) -> set[int]:
    """Indexes of rows whose text contains ``token`` (casefolded substring)."""
    needle = str(token).casefold()
    return {i for i, text in enumerate(row_texts) if needle in text.casefold()}


def minimal_cover(
    group: Sequence[str],
    row_texts: Sequence[str],
    max_sources: int,
) -> tuple[int, ...] | None:
    """Smallest set of row indexes jointly containing every token of ``group``.

    Returns ``None`` when no set of at most ``max_sources`` rows covers the
    group. Exhaustive, not greedy: groups are <=4 tokens and rows <=10, so exact
    minimality is cheap and removes an approximation from the ruler.

    Ties break toward lower row indexes, i.e. toward the top of the ranking, so
    a cover reported as ``(0, 1)`` really is the highest-ranked way to assemble
    the answer.

    The bound matters. "Assembled from two rows" is the multi-hop case; a group
    scattered across five rows is coincidence, and allowing it would import
    exactly the false-positive the battery's one-row rule was defending against
    (see ``battery.py`` "joining texts would let multi-token groups hit on
    tokens split across unrelated results"). The defence here is different in
    kind: bound the cover, report ``hitSingle`` beside ``hitUnion``, and measure
    the residual false-positive rate with a shuffled control.
    """
    if not group:
        return None
    per_token = [token_rows(t, row_texts) for t in group]
    if any(not rows for rows in per_token):
        return None
    candidate_rows = sorted(set().union(*per_token))
    ceiling = min(max_sources, len(candidate_rows))
    for size in range(1, ceiling + 1):
        for combo in combinations(candidate_rows, size):
            chosen = set(combo)
            if all(rows & chosen for rows in per_token):
                return tuple(sorted(combo))
    return None


def score_rows(
    groups: Sequence[Sequence[str]],
    row_texts: Sequence[str],
    *,
    k: int,
    max_sources: int,
) -> dict[str, Any]:
    """Score one question's answer groups against one retrieval's rows."""
    rows = list(row_texts)[:k]
    best: tuple[int, tuple[int, ...]] | None = None
    single: tuple[int, tuple[int, ...]] | None = None
    for index, group in enumerate(groups):
        cover = minimal_cover(group, rows, max_sources)
        if cover is None:
            continue
        if best is None or len(cover) < len(best[1]):
            best = (index, cover)
        if len(cover) == 1 and single is None:
            single = (index, cover)
    return {
        "hitUnion": best is not None,
        "hitSingle": single is not None,
        "groupIndex": best[0] if best else None,
        "coverRows": list(best[1]) if best else None,
        "sources": len(best[1]) if best else None,
        "rowCount": len(rows),
    }


# --------------------------------------------------------------------------
# capture (live, read-only GETs)
# --------------------------------------------------------------------------


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _lane_of(payload: dict[str, Any]) -> str | None:
    """Which racing lane produced this answer.

    ``lifecycle.fallbackStatus`` is the only surface that names the winner:
    ``not_run`` = the deep pipeline, ``durable_entity_first`` /
    ``durable_entity_rescue`` / ``durable_entity_rescue_after_timeout`` /
    ``fast_preflight_hit`` / ``project_file_recall_fallback`` /
    ``context_packet_fallback`` / ``cache_satisfied`` = a rescue lane.
    """
    lifecycle = payload.get("lifecycle") or {}
    lane = lifecycle.get("fallbackStatus") or lifecycle.get("fallback_status")
    if lane is None:
        budget = payload.get("budget") or {}
        lane = budget.get("fallbackStatus") or budget.get("fallback_status")
    return str(lane) if lane is not None else None


def _is_cache_served(lane: str | None, budget: dict[str, Any]) -> bool:
    """True when this probe replayed a cached packet instead of retrieving.

    A cache-served probe is not an independent sample of the retriever: it is a
    verbatim replay of an earlier one, so counting it as a second observation
    drives the measured variance to zero and makes the instrument claim a
    resolving power it does not have.
    """
    if lane in CACHE_LANES:
        return True
    skip = budget.get("skipReason") or budget.get("skip_reason")
    return str(skip) in CACHE_LANES if skip else False


def _first_present(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def read_cache_provenance(packet_cache: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalise a server ``stats.packetCache`` block into cache provenance.

    Pure, so the three states can be tested without a server. They are distinct
    on purpose:

    * ``ok`` — the server reported its cache identity; the guard uses the real
      TTL and can name the fingerprint the arm's keys carried.
    * ``unreported`` — the server answered but predates ticket #29, so it has
      no identity in its key. Falls back to the conservative assumption (cache
      on, default TTL), which can only cause a refusal, never a false pass.
    * ``unreachable`` — no answer at all.
    """
    if not isinstance(packet_cache, Mapping):
        return {
            "status": "unreported",
            "enabled": None,
            "ttlSeconds": None,
            "fingerprint": None,
            "keySchema": None,
            "detail": "server did not report stats.packetCache",
        }
    fingerprint = _first_present(packet_cache, "fingerprint")
    ttl = _first_present(packet_cache, "ttl_seconds", "ttlSeconds")
    enabled = _first_present(packet_cache, "enabled")
    key_schema = _first_present(packet_cache, "key_schema", "keySchema")
    if fingerprint is None and ttl is None:
        return {
            "status": "unreported",
            "enabled": bool(enabled) if enabled is not None else None,
            "ttlSeconds": None,
            "fingerprint": None,
            "keySchema": None,
            "detail": "server reports no packet-cache identity (pre-#29 build)",
        }
    return {
        "status": "ok",
        "enabled": bool(enabled) if enabled is not None else None,
        "ttlSeconds": float(ttl) if isinstance(ttl, int | float) else None,
        "fingerprint": str(fingerprint) if fingerprint is not None else None,
        "keySchema": str(key_schema) if key_schema is not None else None,
        "detail": None,
    }


def fetch_cache_provenance(
    server_url: str,
    *,
    timeout: float = 10.0,
    fetch: Callable[[str, float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read the live packet-cache identity. Read-only GET; never raises."""
    getter = fetch or _get_json
    url = f"{server_url.rstrip('/')}{RUNTIME_FAST_PATH}"
    try:
        payload = getter(url, timeout)
    except Exception as exc:  # network/timeout/decode — recorded, never swallowed
        return {
            "status": "unreachable",
            "enabled": None,
            "ttlSeconds": None,
            "fingerprint": None,
            "keySchema": None,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    stats = payload.get("stats") if isinstance(payload, Mapping) else None
    packet_cache = stats.get("packetCache") if isinstance(stats, Mapping) else None
    return read_cache_provenance(packet_cache)


def capture_runs(
    *,
    server_url: str = "http://127.0.0.1:8100",
    rig: dict[str, Any] | None = None,
    rig_path: Path | None = None,
    runs: int = 10,
    limit: int = DEFAULT_LIMIT,
    timeout: float = 30.0,
    sleep_s: float = 0.0,
    run_gap_s: float = 0.0,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Capture ``runs`` passes of the rig against a live server. Read-only GETs.

    Nothing is scored here. The capture is the raw material; scoring is a pure
    function of it, so a rule change never requires new live load.

    ``run_gap_s`` spaces consecutive passes. Set it above the server's packet
    cache TTL (see :data:`DEFAULT_CACHE_TTL_S`) or the repeats are cache
    replays and the scorer will refuse to certify their variance.
    """
    rig = rig or load_rig(rig_path)
    questions = rig_questions(rig)
    base = server_url.rstrip("/")
    # Read before the probes: the identity is what the arm's keys will carry.
    cache_provenance = fetch_cache_provenance(base, timeout=timeout)
    started_wall = time.time()
    captured_runs: list[dict[str, Any]] = []

    for run_index in range(runs):
        run_rows: list[dict[str, Any]] = []
        for question in questions:
            qid = str(question.get("id"))
            quoted = urllib.parse.quote(str(question["q"]))
            url = f"{base}/api/knowledge/recall?q={quoted}&limit={limit}"
            t0 = time.perf_counter()
            error: str | None = None
            payload: dict[str, Any] = {}
            try:
                payload = _get_json(url, timeout)
            except Exception as exc:  # network/timeout/decode — recorded, never swallowed
                error = f"{type(exc).__name__}: {exc}"
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            budget = payload.get("budget") or {}
            lane = _lane_of(payload) if not error else None
            run_rows.append(
                {
                    "id": qid,
                    "rows": result_texts(payload, limit) if not error else [],
                    "lane": lane,
                    "cacheServed": _is_cache_served(lane, budget) if not error else None,
                    "status": payload.get("status") if not error else None,
                    "degraded": bool(budget.get("degraded")) if not error else None,
                    "timeout": bool(budget.get("timeout")) if not error else None,
                    "latencyMs": elapsed,
                    "atS": round(time.time() - started_wall, 3),
                    "error": error,
                }
            )
            if on_progress:
                on_progress(run_index, runs, qid)
            if sleep_s:
                time.sleep(sleep_s)
        captured_runs.append({"run": run_index, "questions": run_rows})
        if run_gap_s and run_index < runs - 1:
            time.sleep(run_gap_s)

    return {
        "schema": CAPTURE_SCHEMA,
        "serverUrl": base,
        "startedAt": started_wall,
        "durationS": round(time.time() - started_wall, 2),
        "limit": limit,
        "runs": runs,
        "serverCache": cache_provenance,
        "rig": {
            "description": rig.get("description"),
            "questions": [
                {
                    "id": str(q.get("id")),
                    "q": q.get("q"),
                    "expected_tokens": q.get("expected_tokens"),
                    "kind": q.get("kind"),
                    "derivation": q.get("derivation"),
                }
                for q in questions
            ],
        },
        "captures": captured_runs,
    }


# --------------------------------------------------------------------------
# statistics — the minimum-N derivation
# --------------------------------------------------------------------------

_Z_ALPHA_TWO_SIDED_05 = 1.959963985
_Z_POWER_80 = 0.841621234


def _lower_incomplete_gamma_regularized(a: float, x: float) -> float:
    """P(a, x) — regularized lower incomplete gamma (series + continued fraction).

    Implemented rather than table-looked-up so the chi-square quantile below is
    computed for any df; the unit test checks it against published table values.
    """
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        n = a
        for _ in range(1000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def chi2_quantile(p: float, df: int) -> float:
    """Lower-tail chi-square quantile: x such that P(X <= x) = p, X ~ chi2(df)."""
    if df <= 0:
        raise ValueError("df must be positive")
    lo, hi = 0.0, max(10.0, 4.0 * df)
    while _lower_incomplete_gamma_regularized(df / 2.0, hi / 2.0) < p:
        hi *= 2.0
        if hi > 1e9:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _lower_incomplete_gamma_regularized(df / 2.0, mid / 2.0) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sigma_upper_confidence(sd: float, n: int, confidence: float = 0.95) -> float | None:
    """Upper ``confidence`` bound on the population SD given a sample SD.

    ``sd * sqrt((n-1) / chi2_{1-confidence}(n-1))``. Without this, a minimum-N
    derived from an SD estimated on 8 runs is itself badly noisy, and reporting
    it as a hard number would be the ledger's "unlabelled approximation".

    **The sd == 0 case is handled separately and it matters.** The normal-theory
    bound multiplies zero by a constant and returns zero, which would let the
    instrument certify "N >= 2 runs/arm" off a sample that merely never happened
    to flip. Observing no variation in n runs does not mean there is none. The
    rule of three gives a 95% upper bound of ``3/n`` on the rate of an event
    never observed in n trials; treating a flip as a +/-1 change in the total
    bounds the variance by that rate, so ``sd <= sqrt(3/n)``. With n=8 that is
    0.61, i.e. "you would still need ~6 runs per arm", not 2.
    """
    if n < 2 or sd is None:
        return None
    if sd == 0:
        return math.sqrt(3.0 / n)
    df = n - 1
    q = chi2_quantile(1.0 - confidence, df)
    if q <= 0:
        return None
    return sd * math.sqrt(df / q)


def min_runs_per_arm(sd: float, delta_answers: float = 1.0) -> int | None:
    """Runs per arm needed to resolve a ``delta_answers`` difference.

    Two independent arms, two-sided alpha=0.05, power=0.80. The difference of
    two arm means has SE ``sd * sqrt(2/N)``, so detecting ``delta`` requires

        N >= 2 * sd^2 * (z_{1-a/2} + z_{1-b})^2 / delta^2   (~= 15.7 * sd^2 for delta=1)

    ``sd`` is the EMPIRICAL SD of the per-run total score, not a per-question
    binomial sum. That matters: the dominant noise source is the rescue-lane
    race, which is shared across the questions inside one run, so an
    independence assumption would understate the variance.
    """
    if sd is None or delta_answers <= 0:
        return None
    z = _Z_ALPHA_TWO_SIDED_05 + _Z_POWER_80
    return max(2, math.ceil(2.0 * (sd**2) * (z**2) / (delta_answers**2)))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _control_false_positives(
    capture: dict[str, Any],
    questions: Sequence[dict[str, Any]],
    *,
    k: int,
    max_sources: int,
) -> dict[str, Any]:
    """Shuffled control: score each question against ANOTHER question's rows.

    This measures the union rule's false-positive rate instead of asserting it
    away. Rows are rotated by one position within each run (a derangement for
    any rig with >=2 questions), so every scored pair is a genuine mismatch.
    A high ``unionRate`` here means the union rule is not trustworthy on this
    rig and the report says so.
    """
    order = [str(q.get("id")) for q in questions]
    groups_by_id = {str(q.get("id")): (q.get("expected_tokens") or []) for q in questions}
    union_hits = 0
    single_hits = 0
    trials = 0
    for run in capture.get("captures") or []:
        by_id = {row["id"]: row for row in run.get("questions") or []}
        for position, qid in enumerate(order):
            other = order[(position + 1) % len(order)]
            source = by_id.get(other)
            if not source or source.get("error") or not source.get("rows"):
                continue
            scored = score_rows(
                groups_by_id.get(qid) or [],
                source["rows"],
                k=k,
                max_sources=max_sources,
            )
            trials += 1
            union_hits += 1 if scored["hitUnion"] else 0
            single_hits += 1 if scored["hitSingle"] else 0
    if not trials:
        return {"trials": 0, "unionRate": None, "singleRate": None}
    return {
        "trials": trials,
        "unionHits": union_hits,
        "singleHits": single_hits,
        "unionRate": round(union_hits / trials, 4),
        "singleRate": round(single_hits / trials, 4),
    }


def score_capture(
    capture: dict[str, Any],
    *,
    k: int = DEFAULT_K,
    max_sources: int = DEFAULT_MAX_SOURCES,
    delta_answers: float = 1.0,
    min_runs: int = MIN_RUNS,
    max_degraded_fraction: float = MAX_DEGRADED_FRACTION,
    cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
) -> dict[str, Any]:
    """Score a capture. Pure function — no network, no clock, no config."""
    questions = list((capture.get("rig") or {}).get("questions") or [])
    runs = list(capture.get("captures") or [])
    n_runs = len(runs)
    refusals: list[str] = []

    # Cache provenance recorded at capture time beats the caller's assumption:
    # a TTL passed on the command line is a guess about another process, and
    # AUDIT-14's unshipped half was exactly that the server never reported it.
    provenance = capture.get("serverCache")
    if not isinstance(provenance, Mapping):
        provenance = read_cache_provenance(None)
    server_ttl = provenance.get("ttlSeconds")
    if isinstance(server_ttl, int | float) and server_ttl >= 0:
        cache_ttl_s = float(server_ttl)
        ttl_source = "server"
    else:
        ttl_source = "assumed"
    # A process with the cache OFF cannot replay a packet, so its repeats are
    # not cache replays and need no TTL spacing. `recall_packet_cache_enabled`
    # is the measurement-mode bypass; this is the check that it took effect,
    # read from the measured process rather than from the config file a rig
    # believes it edited.
    cache_bypassed = provenance.get("enabled") is False

    per_question: list[dict[str, Any]] = []
    total_probes = 0
    error_probes = 0
    degraded_probes = 0
    cache_probes = 0
    untimed_probes = 0
    unverifiable_probes = 0
    unspaced_questions: list[str] = []

    for question in questions:
        qid = str(question.get("id"))
        groups = question.get("expected_tokens") or []
        union_flags: list[int] = []
        single_flags: list[int] = []
        lanes: dict[str, int] = {}
        source_sizes: dict[str, int] = {}
        errors = 0
        empty = 0
        degraded = 0
        cached = 0
        latencies: list[float] = []
        timestamps: list[float] = []
        per_run_scores: list[dict[str, Any]] = []

        for run in runs:
            row = next((r for r in (run.get("questions") or []) if r.get("id") == qid), None)
            if row is None:
                continue
            total_probes += 1
            if row.get("latencyMs") is not None:
                latencies.append(float(row["latencyMs"]))
            if row.get("atS") is not None:
                timestamps.append(float(row["atS"]))
            else:
                untimed_probes += 1
            if "cacheServed" not in row:
                unverifiable_probes += 1
            if row.get("error"):
                errors += 1
                error_probes += 1
                per_run_scores.append({"run": run.get("run"), "error": row["error"]})
                continue
            if row.get("degraded"):
                degraded += 1
                degraded_probes += 1
            lane = row.get("lane") or "unknown"
            lanes[lane] = lanes.get(lane, 0) + 1
            if row.get("cacheServed"):
                # A replay of an earlier probe, not a new sample. Counting it
                # would drive the measured variance toward zero.
                cached += 1
                cache_probes += 1
                per_run_scores.append({"run": run.get("run"), "lane": lane, "cacheServed": True})
                continue
            rows = row.get("rows") or []
            if not rows:
                empty += 1
                per_run_scores.append({"run": run.get("run"), "lane": lane, "rowCount": 0})
                continue
            scored = score_rows(groups, rows, k=k, max_sources=max_sources)
            union_flags.append(1 if scored["hitUnion"] else 0)
            single_flags.append(1 if scored["hitSingle"] else 0)
            if scored["sources"] is not None:
                key = str(scored["sources"])
                source_sizes[key] = source_sizes.get(key, 0) + 1
            per_run_scores.append({"run": run.get("run"), "lane": lane, **scored})

        usable = len(union_flags)
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:], strict=False)]
        median_gap = statistics.median(gaps) if gaps else None
        if (
            not cache_bypassed
            and median_gap is not None
            and median_gap < cache_ttl_s
            and len(timestamps) > 1
        ):
            unspaced_questions.append(qid)
        excluded = None
        if cached and usable == 0:
            excluded = "cache_served"
        elif usable == 0:
            excluded = "no_usable_runs"
        elif usable < math.ceil(n_runs / 2):
            excluded = "insufficient_usable_runs"

        p_union = (sum(union_flags) / usable) if usable else None
        p_single = (sum(single_flags) / usable) if usable else None
        per_question.append(
            {
                "id": qid,
                "q": question.get("q"),
                "kind": question.get("kind"),
                "usableRuns": usable,
                "errorRuns": errors,
                "emptyRuns": empty,
                "degradedRuns": degraded,
                "cacheServedRuns": cached,
                "medianRunGapS": round(median_gap, 1) if median_gap is not None else None,
                "pHitUnion": round(p_union, 4) if p_union is not None else None,
                "pHitSingle": round(p_single, 4) if p_single is not None else None,
                "hitsUnion": sum(union_flags) if usable else None,
                "hitsSingle": sum(single_flags) if usable else None,
                # A question is STABLE when every usable run agreed. Instability
                # here is the lane race showing up per question.
                "stable": (p_union in (0.0, 1.0)) if p_union is not None else None,
                "sdUnion": (
                    round(statistics.stdev(union_flags), 4) if len(union_flags) > 1 else None
                ),
                "multiSourceRuns": sum(
                    1 for s in per_run_scores if s.get("hitUnion") and not s.get("hitSingle")
                ),
                "sourcesHistogram": source_sizes or None,
                "lanes": lanes or None,
                "laneStable": (len(lanes) == 1) if lanes else None,
                "medianLatencyMs": round(statistics.median(latencies), 2) if latencies else None,
                "excluded": excluded,
                "perRun": per_run_scores,
            }
        )

    included = [row for row in per_question if not row["excluded"]]
    included_ids = {row["id"] for row in included}

    # Per-run totals over the INCLUDED questions only. A run contributes a total
    # only when every included question produced a usable probe in it; otherwise
    # the run's total would silently mix "missed" with "not measured".
    per_run_union: list[int] = []
    per_run_single: list[int] = []
    complete_runs = 0
    for run in runs:
        by_id = {r.get("id"): r for r in (run.get("questions") or [])}
        union_total = 0
        single_total = 0
        complete = True
        for qid in included_ids:
            row = by_id.get(qid)
            if row is None or row.get("error") or row.get("cacheServed") or not row.get("rows"):
                complete = False
                break
            question = next(q for q in questions if str(q.get("id")) == qid)
            scored = score_rows(
                question.get("expected_tokens") or [],
                row["rows"],
                k=k,
                max_sources=max_sources,
            )
            union_total += 1 if scored["hitUnion"] else 0
            single_total += 1 if scored["hitSingle"] else 0
        if complete:
            complete_runs += 1
            per_run_union.append(union_total)
            per_run_single.append(single_total)

    sd_union = statistics.stdev(per_run_union) if len(per_run_union) > 1 else None
    sd_single = statistics.stdev(per_run_single) if len(per_run_single) > 1 else None
    sd_upper = (
        sigma_upper_confidence(sd_union, len(per_run_union)) if sd_union is not None else None
    )
    n_min = min_runs_per_arm(sd_union, delta_answers) if sd_union is not None else None
    n_min_conservative = min_runs_per_arm(sd_upper, delta_answers) if sd_upper is not None else None

    error_fraction = (error_probes / total_probes) if total_probes else None
    degraded_fraction = (degraded_probes / total_probes) if total_probes else None

    if n_runs < min_runs:
        refusals.append(
            f"only {n_runs} run(s) captured; variance is not estimable below {min_runs}"
        )
    if len(per_run_union) < 2:
        refusals.append("fewer than 2 complete runs; per-run variance cannot be computed")
    if error_fraction is not None and error_fraction > max_degraded_fraction:
        refusals.append(
            f"{error_probes}/{total_probes} probes errored "
            f"({error_fraction:.1%} > {max_degraded_fraction:.0%})"
        )
    if degraded_fraction is not None and degraded_fraction > max_degraded_fraction:
        refusals.append(
            f"{degraded_probes}/{total_probes} probes came back degraded "
            f"({degraded_fraction:.1%} > {max_degraded_fraction:.0%})"
        )
    for row in per_question:
        if row["excluded"]:
            refusals.append(f"question {row['id']} excluded from the score: {row['excluded']}")
    if cache_probes:
        refusals.append(
            f"{cache_probes}/{total_probes} probes were served from the packet cache "
            "(lane=cache_satisfied) and are replays, not samples"
        )
    if untimed_probes or unverifiable_probes:
        # A capture taken before the cache guard existed carries no timestamps
        # and no cache flags. Scoring it silently would be exactly the failure
        # this guard was added to prevent, so it refuses instead.
        refusals.append(
            f"{max(untimed_probes, unverifiable_probes)}/{total_probes} probes lack "
            "timing/cache provenance (pre-guard capture); cache independence "
            "cannot be verified, so the variance cannot be certified"
        )
    if unspaced_questions:
        refusals.append(
            f"{len(unspaced_questions)} question(s) were re-probed faster than the "
            f"{cache_ttl_s:g}s packet-cache TTL "
            f"({', '.join(sorted(unspaced_questions)[:4])}...); repeats inside the TTL "
            "are not independent samples, so the measured variance is a floor, not an "
            "estimate. Re-run with --run-gap-s above the TTL."
        )
    if n_min_conservative is not None and len(per_run_union) < n_min_conservative:
        refusals.append(
            f"cannot resolve a {delta_answers:g}-answer difference: "
            f"{len(per_run_union)} complete runs, need >= {n_min_conservative} per arm "
            f"(sd={sd_union:.3f}, 95% upper sd={sd_upper:.3f})"
        )

    comparable = not refusals
    if not runs:
        status = "empty"
    elif error_fraction and error_fraction > max_degraded_fraction:
        status = "degraded"
    elif comparable:
        status = "resolved"
    else:
        status = "unresolved"

    half = len(per_run_union) // 2
    drift = None
    if half >= 1 and len(per_run_union) >= 4:
        drift = {
            "firstHalfMean": round(statistics.fmean(per_run_union[:half]), 3),
            "secondHalfMean": round(statistics.fmean(per_run_union[half:]), 3),
            "note": (
                "recall records access/priming, so runs are not independent of "
                "each other; a monotone trend here is a real order effect"
            ),
        }

    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "comparison": {
            "usable": comparable,
            "deltaAnswers": delta_answers,
            "completeRuns": len(per_run_union),
            "minRunsPerArm": n_min,
            "minRunsPerArmConservative": n_min_conservative,
            "alpha": 0.05,
            "power": 0.80,
            "note": (
                "minRunsPerArm = 2*sd^2*(z_a/2 + z_b)^2/delta^2, sd = empirical SD "
                "of the per-run total (lane noise is run-correlated, so no "
                "per-question independence is assumed). Conservative variant uses "
                "the 95% upper confidence bound on sd."
            ),
        },
        "refusals": refusals,
        "params": {
            "k": k,
            "maxSources": max_sources,
            "limit": capture.get("limit"),
            "runs": n_runs,
        },
        "cache": {
            "ttlSeconds": cache_ttl_s,
            "ttlSource": ttl_source,
            "enabled": provenance.get("enabled"),
            "bypassed": cache_bypassed,
            "fingerprint": provenance.get("fingerprint"),
            "keySchema": provenance.get("keySchema"),
            "provenanceStatus": provenance.get("status"),
            "provenanceDetail": provenance.get("detail"),
            "cacheServedProbes": cache_probes,
            "untimedProbes": untimed_probes,
            "unverifiableProbes": unverifiable_probes,
            "unspacedQuestions": unspaced_questions,
            "note": (
                "The packet cache key carries an identity fingerprint (schema pc2) over "
                "the activation config, runtime mode, build and packet-building source. "
                "TWO ARMS THAT PRINT THE SAME fingerprint WERE NOT ISOLATED BY THE KEY: "
                "isolate them with ENGRAM_PACKET_CACHE_NAMESPACE, or measure with "
                "recall_packet_cache_enabled=False (reported here as bypassed=true). "
                "Within one arm, repeats faster than ttlSeconds are cache replays."
            ),
        },
        "probes": {
            "total": total_probes,
            "errors": error_probes,
            "degraded": degraded_probes,
            "cacheServed": cache_probes,
            "errorFraction": round(error_fraction, 4) if error_fraction is not None else None,
            "degradedFraction": (
                round(degraded_fraction, 4) if degraded_fraction is not None else None
            ),
        },
        "score": {
            "questionsScored": len(included),
            "questionsExcluded": len(per_question) - len(included),
            "perRunUnion": per_run_union,
            "perRunSingle": per_run_single,
            "meanUnion": round(statistics.fmean(per_run_union), 3) if per_run_union else None,
            "meanSingle": round(statistics.fmean(per_run_single), 3) if per_run_single else None,
            "sdUnion": round(sd_union, 4) if sd_union is not None else None,
            "sdSingle": round(sd_single, 4) if sd_single is not None else None,
            "sdUnionUpper95": round(sd_upper, 4) if sd_upper is not None else None,
            "rangeUnion": ([min(per_run_union), max(per_run_union)] if per_run_union else None),
            "rangeSingle": ([min(per_run_single), max(per_run_single)] if per_run_single else None),
        },
        "drift": drift,
        "control": _control_false_positives(
            capture,
            questions,
            k=k,
            max_sources=max_sources,
        ),
        "questions": per_question,
    }


def _lane_summary(report: dict[str, Any]) -> dict[str, int]:
    lanes: dict[str, int] = {}
    for row in report.get("questions") or []:
        for lane, count in (row.get("lanes") or {}).items():
            lanes[lane] = lanes.get(lane, 0) + count
    return lanes


def format_meter_report(report: dict[str, Any]) -> str:
    score = report.get("score") or {}
    comparison = report.get("comparison") or {}
    params = report.get("params") or {}
    lines: list[str] = []

    status = report.get("status")
    if status == "resolved":
        lines.append(
            f"# Recall meter: RESOLVED — {score.get('meanUnion')} "
            f"+/- {score.get('sdUnion')} of {score.get('questionsScored')} "
            f"over {comparison.get('completeRuns')} runs"
        )
    else:
        lines.append(f"# Recall meter: {str(status).upper()} — NO HEADLINE SCORE")
        lines.append("")
        lines.append(
            "This run set cannot be used to compare two arms. Raw per-question "
            "data follows; the aggregate is withheld deliberately (a plausible "
            "number here is what mis-measured the reranker three times)."
        )
    lines.append("")
    lines.append(
        f"- params: k={params.get('k')} maxSources={params.get('maxSources')} "
        f"limit={params.get('limit')} runs={params.get('runs')}"
    )
    if comparison.get("minRunsPerArm") is not None:
        lines.append(
            # The conservative bound leads deliberately: it is what the verdict
            # uses, and a reader who anchors on the point estimate from a small
            # sample is the failure mode this instrument exists to stop.
            f"- resolving a {comparison.get('deltaAnswers')}-answer difference needs "
            f"N >= {comparison.get('minRunsPerArmConservative')} runs/arm "
            f"(point estimate from the observed sd: {comparison.get('minRunsPerArm')}); "
            f"have {comparison.get('completeRuns')}"
        )
    else:
        lines.append("- minimum N: NOT DERIVABLE (need >= 2 complete runs)")

    probes = report.get("probes") or {}
    lines.append(
        f"- probes: {probes.get('total')} "
        f"({probes.get('errors')} error, {probes.get('degraded')} degraded, "
        f"{probes.get('cacheServed')} cache-replayed)"
    )
    cache = report.get("cache") or {}
    # Printed on every report, resolved or not: an A/B is only cache-independent
    # if the two arms show different fingerprints (or bypassed=yes).
    lines.append(
        f"- packet cache: fingerprint={cache.get('fingerprint')} "
        f"({cache.get('provenanceStatus')}) enabled={cache.get('enabled')} "
        f"bypassed={'yes' if cache.get('bypassed') else 'no'} "
        f"ttl={cache.get('ttlSeconds')}s ({cache.get('ttlSource')})"
    )
    if score.get("perRunUnion"):
        lines.append(f"- per-run totals (union rule): {score.get('perRunUnion')}")
        lines.append(
            f"- per-run totals (battery one-row rule): {score.get('perRunSingle')} "
            f"mean {score.get('meanSingle')} sd {score.get('sdSingle')}"
        )
    control = report.get("control") or {}
    if control.get("trials"):
        lines.append(
            f"- shuffled control (union-rule false positives): "
            f"{control.get('unionHits')}/{control.get('trials')} = "
            f"{control.get('unionRate')} (one-row rule: {control.get('singleRate')})"
        )
    lanes = _lane_summary(report)
    if lanes:
        lines.append(f"- lanes that won: {lanes}")
    drift = report.get("drift")
    if drift:
        lines.append(
            f"- order effect: first half {drift['firstHalfMean']} -> "
            f"second half {drift['secondHalfMean']}"
        )
    for refusal in report.get("refusals") or []:
        lines.append(f"- REFUSAL: {refusal}")

    lines.append("")
    lines.append("| question | p(hit) union | p(hit) 1-row | stable | multi-src | lanes |")
    lines.append("|---|---|---|---|---|---|")
    for row in report.get("questions") or []:
        lane_text = ",".join(f"{k}:{v}" for k, v in sorted((row.get("lanes") or {}).items()))
        flag = " (EXCLUDED)" if row.get("excluded") else ""
        stable = row.get("stable")
        lines.append(
            f"| {row.get('id')}{flag} | {row.get('pHitUnion')} | {row.get('pHitSingle')} | "
            f"{'yes' if stable else 'NO' if stable is not None else '-'} | "
            f"{row.get('multiSourceRuns')} | {lane_text} |"
        )
    return "\n".join(lines) + "\n"


def run_meter_against_live(
    *,
    server_url: str = "http://127.0.0.1:8100",
    rig_path: Path | None = None,
    runs: int = 10,
    limit: int = DEFAULT_LIMIT,
    k: int = DEFAULT_K,
    max_sources: int = DEFAULT_MAX_SOURCES,
    delta_answers: float = 1.0,
    sleep_s: float = 0.0,
    run_gap_s: float = 0.0,
    cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
    timeout: float = 30.0,
    capture_path: Path | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture then score. Returns ``(capture, report)``."""
    capture = capture_runs(
        server_url=server_url,
        rig_path=rig_path,
        runs=runs,
        limit=limit,
        timeout=timeout,
        sleep_s=sleep_s,
        run_gap_s=run_gap_s,
        on_progress=on_progress,
    )
    if capture_path:
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        with open(capture_path, "w", encoding="utf-8") as f:
            json.dump(capture, f, indent=1)
    report = score_capture(
        capture,
        k=k,
        max_sources=max_sources,
        delta_answers=delta_answers,
        cache_ttl_s=cache_ttl_s,
    )
    return capture, report


def load_capture(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        capture = json.load(f)
    if capture.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"not a {CAPTURE_SCHEMA} capture: {path}")
    return capture
