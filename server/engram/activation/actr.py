"""ACT-R spreading activation strategy — faithful 1-hop from working memory."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from engram.config import ActivationConfig

if TYPE_CHECKING:
    from engram.activation.context_gate import ContextGate


class ACTRStrategy:
    """True ACT-R spreading activation: 1-hop from small goal buffer.

    W_j = W/n (total attention W divided among n source items)
    S_ji = max(fan_s_min, fan_s_max - ln(fan+1))
    Bonus = W_j * S_ji * predicate_weight * edge_weight

    Bonuses accumulate across sources — an entity connected to N
    working memory items receives Nx the bonus.

    No energy budget, firing threshold, or decay per hop.
    """

    async def spread(
        self,
        seed_nodes: list[tuple[str, float]],
        neighbor_provider,
        cfg: ActivationConfig,
        group_id: str | None = None,
        community_store=None,
        context_gate: ContextGate | None = None,
        seed_entity_types: dict[str, str] | None = None,
    ) -> tuple[dict[str, float], dict[str, int]]:
        """1-hop spreading from seed nodes (working memory items).

        Seed energy values are ignored; W_j is computed internally
        from actr_total_w / n_sources.
        """
        bonuses: dict[str, float] = {}
        hop_distances: dict[str, int] = {}

        if not seed_nodes:
            return bonuses, hop_distances

        n_sources = len(seed_nodes)
        w_j = cfg.actr_total_w / n_sources

        for source_id, _energy in seed_nodes:
            hop_distances.setdefault(source_id, 0)

            neighbors = await neighbor_provider.get_active_neighbors_with_weights(
                source_id, group_id=group_id
            )
            fan = len(neighbors)
            if fan == 0:
                continue

            s_ji = max(cfg.fan_s_min, cfg.fan_s_max - math.log(fan + 1))

            for neighbor_info in neighbors:
                if len(neighbor_info) >= 3:
                    neighbor_id = neighbor_info[0]
                    edge_weight = neighbor_info[1]
                    predicate = neighbor_info[2]
                    predicate_weight = cfg.predicate_weights.get(
                        predicate, cfg.predicate_weight_default
                    )
                else:
                    neighbor_id = neighbor_info[0]
                    edge_weight = neighbor_info[1]
                    predicate_weight = cfg.predicate_weight_default

                bonus = w_j * s_ji * predicate_weight * edge_weight
                bonuses[neighbor_id] = bonuses.get(neighbor_id, 0.0) + bonus

                if neighbor_id not in hop_distances:
                    hop_distances[neighbor_id] = 1

        return bonuses, hop_distances
