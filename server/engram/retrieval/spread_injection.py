"""The one rule for how many graph-discovered entities may enter the pool.

Spreading is a bonus channel, not the primary result set: it may supplement the
candidate pool, never swamp it. That rule lives here alone because it used to
live in two places and the two diverged.

Production (``retrieval/pipeline.py`` Step 4.5) applied
``spread_candidate_injection_max``; the benchmark harness
(``benchmark/methods.py`` Step 3.7) ran the same merge uncapped. Measured on the
dogfood corpus, production injected exactly 32 on 100% of 51 completions out of
453-487 discovered, while the harness injected all ~450 — a 12x pool divergence
inside the wrapper whose entire purpose is to make an A/B measure the scoring
code production runs. It was harmless only for as long as live spreading always
returned ``{}``.

A duplicated constant drifts; a shared call cannot. Both call sites import this.
"""

from __future__ import annotations

from collections.abc import Container, Mapping

from engram.config import ActivationConfig


def select_spread_injections(
    bonuses: Mapping[str, float],
    existing_ids: Container[str],
    cfg: ActivationConfig,
) -> tuple[list[str], int]:
    """Pick the spreading discoveries that may join the candidate pool.

    Returns ``(ids_to_inject, discovered)`` where ``discovered`` is the PRE-cap
    count. Returning it is not decoration: without it the cap is invisible,
    because ``injected == 32`` on every recall reads as "32 was enough" when it
    actually means "the cap bound and ~450 discoveries were thrown away".

    Keeps the strongest by spreading bonus; ties break on id so the pool stays
    deterministic. ``spread_candidate_injection_max = 0`` means unbounded.
    """
    new_ids = [nid for nid, bonus in bonuses.items() if nid not in existing_ids and bonus > 0.0]
    discovered = len(new_ids)
    injection_cap = cfg.spread_candidate_injection_max
    if injection_cap and discovered > injection_cap:
        new_ids.sort(key=lambda nid: (-bonuses[nid], nid))
        del new_ids[injection_cap:]
    return new_ids, discovered
