"""Forward episode->entity graph signal for the answer-bearing channel (GAP B).

Every episode / cue_episode / chunk-episode ``ScoredResult`` is constructed in
``pipeline.retrieve`` with ``activation=0.0, spreading=0.0,
edge_proximity=0.0`` literals, and ``scorer.score_candidates`` — the only
function that applies ``weight_activation`` / ``weight_spreading`` /
``weight_edge_proximity`` — runs on the ENTITY channel, which
``passage_first_entity_budget=0`` then truncates to ``[]``. The consequence is
that 55% of the ACT-R weight budget cannot reach an agent-visible answer.

This module builds the missing consumer. It reads the FORWARD direction
(``episode -> HasEntity -> entity``), which is bounded by the episode's own
entity degree (measured 1.54 on the live brain) rather than the unbounded
``entity -> episodes`` direction, and it reuses the per-entity signal that
``score_candidates`` already computed, so no extra activation or spreading work
is done.

Three deliberate properties:

* **Additive, never penalising.** 46.7% of live episodes have no ``HasEntity``
  edge at all (they were never projected), and projection is triage-selected —
  a proxy for value, not for answer-relevance. A multiplicative or subtractive
  formulation would systematically demote half the corpus along an axis
  uncorrelated with the query. With the flag off, or on an edgeless brain, the
  output is byte-identical to today.
* **Bounded.** The episode lane's own score range is ``[0, ~0.4]``. The raw
  ACT-R term maxes at ``0.275`` after hop decay, so it is applied through the
  separate outer scale ``episode_graph_signal_weight`` — the difference between
  a reranker and a displacer.
* **Observable.** The three signals are always written onto the ScoredResult
  (and therefore into ``score_breakdown``) even when ``..._source`` excludes
  them from the score, so a signal that is structurally zero shows up as a zero
  in the output instead of being invisible.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from engram.config import ActivationConfig
from engram.retrieval.recall_graph_gate import skip_secondary_graph_after_probe_timeout
from engram.retrieval.scorer import ScoredResult

logger = logging.getLogger(__name__)

# Quantisation for every derived value. Tie-order stability is asserted by
# three tests (``test_recall_silent_inert_fixes.py``); a float-noisy term would
# break ties that used to be exact.
_QUANT = 6

EPISODE_RESULT_TYPES = frozenset({"episode", "cue_episode"})

# Ticket 26. ``episode_graph_signal_max_candidates`` is 60 and the native
# executor is ``max_workers=4``, so an unbounded gather hands 60 graph reads to
# a 4-thread pool at once.
#
# What that costs was MEASURED, not assumed, because the obvious rationale is
# wrong: bounding the fan-out does NOT reduce worker burn. Isolated four-arm
# run (stub transport, 4 workers, 500 ms scans, 40 ms stage timeout) — pre-fix
# vs semaphore-only: executor submissions fell 60 -> 8 but jobs actually run
# stayed 4 and worker time stayed 3043 ms vs 3042 ms. ``run_in_executor``
# cancels its own still-pending future when the gather wrapper is cancelled, so
# the queued surplus was never going to run anyway.
#
# What it DOES cost is head-of-line blocking on the shared executor, which is
# ticket 31's mechanism (an independent stage timing out in lockstep with a
# saturating one). Isolated, warm path, one independent read submitted 5 ms
# into the stage, median of 5: it waited **40.5 ms behind the unbounded
# fan-out and 5.2 ms behind the bounded one** — 7.8x. That is the reason for
# this number, and it is the only claim it supports.
#
# 8 matches ``pipeline._EPISODE_USAGE_READ_CONCURRENCY``, which bounds the
# identical pattern on the neighbouring cue-read lane.
_EPISODE_GRAPH_READ_CONCURRENCY = 8


class _InFlight:
    """Exact peak-concurrency counter for the bounded read fan-out.

    Counted, not inferred from timing: a semaphore that was silently removed
    would still *look* fine on a fast fixture, and a timing-derived bound is
    exactly the kind of plausible-but-wrong number INSTRUMENT_AUDIT.md is
    about. The event loop is single-threaded, so ``+= 1`` needs no lock.
    """

    __slots__ = ("current", "peak")

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0

    def __enter__(self) -> _InFlight:
        self.current += 1
        self.peak = max(self.peak, self.current)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.current -= 1


@dataclass(frozen=True)
class EntityGraphSignal:
    """Per-entity ACT-R signal, snapshotted straight out of score_candidates."""

    activation: float
    spreading: float
    hop_distance: int | None


@dataclass(frozen=True)
class EpisodeGraphSignal:
    """Signal derived for one episode from the entities it links to."""

    activation: float
    spreading: float
    edge_proximity: float
    min_hop: int | None
    linked_matched: int


def snapshot_entity_signal(scored: Sequence[ScoredResult]) -> dict[str, EntityGraphSignal]:
    """Capture the per-entity signal.

    MUST be called immediately after ``score_candidates`` and BEFORE the Step
    5.5 rerank re-sorts ``scored`` and before MMR *replaces* it with a
    ``top_n=10`` truncation. Snapshotting later silently shrinks the source
    from the measured 24-37 scored entities to <= 10 and reorders it by
    diversity, which has nothing to do with which entity should lend an episode
    its signal. It is also the only point at which ``scored`` is guaranteed to
    be pure-entity — Step 5.05 appends episode ScoredResults into it under
    temporal cues.
    """
    return {
        sr.node_id: EntityGraphSignal(
            activation=float(sr.activation),
            spreading=float(sr.spreading),
            hop_distance=sr.hop_distance,
        )
        for sr in scored
        if sr.result_type == "entity"
    }


def derive_episode_signal(
    linked_entity_ids: Iterable[str],
    entity_signal: dict[str, EntityGraphSignal],
    cfg: ActivationConfig,
) -> EpisodeGraphSignal | None:
    """Derive an episode's graph signal from its linked entities.

    Returns ``None`` when the episode links to no entity that the entity
    channel scored — the additive-only contract means such an episode keeps
    exactly today's score.

    MAX (not mean) over the linked set: mean entity degree is 1.54 so for most
    episodes the two agree, and where an episode has 3-5 entities the mean is
    dragged down by repo-index Artifacts and ``:decision_statement:`` squatters.
    MAX asks "does this episode touch anything the graph considers live", which
    is the question.

    ``edge_proximity`` is derived from HOP DISTANCE, never inherited from the
    parent's own ``edge_proximity``. Inheriting would launder a semantic signal
    as a graph one: with ``hop_distances`` empty (spreading times out on 9/10
    live recalls) ``score_candidates`` assigns ``edge_proximity=1.0`` to every
    seed, and seeds are chosen by ``sem_sim >= seed_threshold``. The ``1 +`` is
    the episode->entity membership hop, so ``min_hop == 0`` (a seed entity)
    yields ``hop_decay`` — "one real graph edge away from something the query
    matched" — and ``min_hop >= 1`` means spreading genuinely reached it.
    """
    matched = [entity_signal[eid] for eid in linked_entity_ids if eid in entity_signal]
    if not matched:
        return None

    decay = float(cfg.episode_graph_signal_hop_decay)
    hops = [sig.hop_distance for sig in matched if sig.hop_distance is not None]
    min_hop = min(hops) if hops else None
    edge_proximity = decay ** (1 + min_hop) if min_hop is not None else 0.0

    return EpisodeGraphSignal(
        activation=round(max(sig.activation for sig in matched) * decay, _QUANT),
        spreading=round(max(sig.spreading for sig in matched) * decay, _QUANT),
        edge_proximity=round(edge_proximity, _QUANT),
        min_hop=min_hop,
        linked_matched=len(matched),
    )


def graph_term(signal: EpisodeGraphSignal, cfg: ActivationConfig) -> float:
    """Weighted ACT-R term for one episode, before the outer scale.

    ``score_candidates`` cannot be reused here: it re-derives activation from
    an entity-keyed activation store (episode ids are absent, so the derived
    value would be silently discarded), it applies eleven other entity-keyed
    boost terms on a different scale, and it returns a fresh sorted list that
    drops the ``chunk_context`` / ``source`` / cue bookkeeping the episode lane
    depends on. It is also pinned hex-exactly by a golden test and must not be
    touched.
    """
    source = cfg.episode_graph_signal_source
    term = cfg.weight_activation * signal.activation
    if source in {"activation_spreading", "full"}:
        term += cfg.weight_spreading * signal.spreading
    if source == "full":
        term += cfg.weight_edge_proximity * signal.edge_proximity
    return term


def _order(candidates: Sequence[ScoredResult]) -> list[str]:
    return [sr.node_id for sr in sorted(candidates, key=lambda r: (-r.score, r.node_id))]


async def apply_episode_graph_signal(
    episode_candidates: list[ScoredResult],
    cue_candidates: list[ScoredResult],
    *,
    entity_signal: dict[str, EntityGraphSignal],
    graph_store,
    group_id: str,
    cfg: ActivationConfig,
    stage_timings_ms: dict[str, float] | None,
    set_metric,
    add_timing,
) -> None:
    """Write derived graph signal onto the episode/cue/chunk ScoredResults.

    Mutates the candidates in place. Emits positive-probe metrics so that a
    zero is always attributable: gated, timed out, no entity signal, no linked
    entities, or a genuinely edgeless brain.
    """
    if not cfg.episode_graph_signal_enabled:
        return

    targets = [
        sr
        for sr in (*episode_candidates, *cue_candidates)
        if sr.result_type in EPISODE_RESULT_TYPES
    ]
    if not targets:
        return

    # Gate for consistency with every other secondary graph read — but record a
    # DISTINCT metric, so a gated zero is never confused with an edgeless zero.
    if skip_secondary_graph_after_probe_timeout(cfg, stage_timings_ms):
        set_metric(stage_timings_ms, "recall_episode_graph_signal_skipped_probe_timeout", 0.0)
        return

    if not entity_signal:
        set_metric(stage_timings_ms, "recall_episode_graph_signal_no_entity_signal", 0.0)
        return

    # One graph read per unique episode, most relevant first under the cap.
    ranked_ids: list[str] = []
    seen: set[str] = set()
    for sr in sorted(targets, key=lambda r: (-r.score, r.node_id)):
        if sr.node_id not in seen:
            seen.add(sr.node_id)
            ranked_ids.append(sr.node_id)
    ranked_ids = ranked_ids[: cfg.episode_graph_signal_max_candidates]

    started = time.perf_counter()
    timeout_ms = int(cfg.episode_graph_signal_timeout_ms or 0)
    semaphore = asyncio.Semaphore(_EPISODE_GRAPH_READ_CONCURRENCY)
    inflight = _InFlight()

    async def _read(episode_id: str):
        async with semaphore:
            with inflight:
                return await graph_store.get_episode_entities(episode_id, group_id=group_id)

    try:
        gather = asyncio.gather(
            *(_read(eid) for eid in ranked_ids),
            return_exceptions=True,
        )
        reads = (
            await asyncio.wait_for(gather, timeout=timeout_ms / 1000.0)
            if timeout_ms > 0
            else await gather
        )
    except asyncio.TimeoutError:
        add_timing(stage_timings_ms, "recall_episode_graph_signal_timeout", started)
        set_metric(stage_timings_ms, "recall_episode_graph_signal_inflight_max", inflight.peak)
        return
    except asyncio.CancelledError:
        add_timing(stage_timings_ms, "recall_episode_graph_signal_cancelled", started)
        raise
    except Exception as exc:  # noqa: BLE001 - a graph read must never fail recall
        logger.warning("Episode graph signal failed (non-fatal): %s", exc)
        set_metric(stage_timings_ms, "recall_episode_graph_signal_error", 0.0)
        return

    linked_by_episode: dict[str, list[str]] = {}
    linked_total = 0
    for episode_id, read in zip(ranked_ids, reads, strict=False):
        if isinstance(read, BaseException) or not read:
            continue
        ids = [eid for eid in read if eid]
        linked_by_episode[episode_id] = ids
        linked_total += len(ids)

    pre_order = _order(episode_candidates)

    covered = 0
    applied = 0
    multi_hop = 0
    max_term = 0.0
    scale = float(cfg.episode_graph_signal_weight)
    for sr in targets:
        linked = linked_by_episode.get(sr.node_id)
        if not linked:
            continue
        signal = derive_episode_signal(linked, entity_signal, cfg)
        if signal is None:
            continue
        covered += 1
        if signal.min_hop is not None and signal.min_hop >= 1:
            multi_hop += 1
        sr.activation = signal.activation
        sr.spreading = signal.spreading
        sr.edge_proximity = signal.edge_proximity
        sr.hop_distance = None if signal.min_hop is None else signal.min_hop + 1
        delta = round(scale * graph_term(signal, cfg), _QUANT)
        if delta > 0.0:
            applied += 1
            max_term = max(max_term, delta)
            sr.score = round(sr.score + delta, 10)

    add_timing(stage_timings_ms, "recall_episode_graph_signal", started)
    set_metric(stage_timings_ms, "recall_episode_graph_signal_pool", len(ranked_ids))
    set_metric(stage_timings_ms, "recall_episode_graph_signal_inflight_max", inflight.peak)
    set_metric(stage_timings_ms, "recall_episode_graph_signal_linked_total", linked_total)
    set_metric(stage_timings_ms, "recall_episode_graph_signal_covered", covered)
    set_metric(stage_timings_ms, "recall_episode_graph_signal_applied", applied)
    set_metric(stage_timings_ms, "recall_episode_graph_signal_multi_hop", multi_hop)
    set_metric(stage_timings_ms, "recall_episode_graph_signal_max", max_term)
    # The one metric that cannot be faked by a computed-and-discarded value: it
    # is measured on the ordering Step 6 actually consumes. Non-zero coverage
    # with zero reorders means the term fires and changes nothing — report it,
    # do not tune around it.
    post_order = _order(episode_candidates)
    set_metric(
        stage_timings_ms,
        "recall_episode_graph_signal_reorders",
        sum(1 for before, after in zip(pre_order, post_order, strict=False) if before != after),
    )
