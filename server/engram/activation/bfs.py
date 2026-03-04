"""BFS-based spreading activation strategy."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

from engram.config import ActivationConfig

if TYPE_CHECKING:
    from engram.activation.context_gate import ContextGate


def _resolve_domain(entity_type: str, domain_groups: dict[str, list[str]]) -> str | None:
    """Map an entity type to its topic domain. Returns None for mixed/unclassified."""
    for domain, types in domain_groups.items():
        if entity_type in types:
            return domain
    return None


class BFSStrategy:
    """Bounded BFS spreading activation with typed edge weights."""

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
        """Spread activation from seed nodes through the graph via BFS.

        Returns (bonuses, hop_distances):
          - bonuses: {node_id: spreading_bonus}
          - hop_distances: {node_id: min_hops_from_seed}
        """
        bonuses: dict[str, float] = {}
        hop_distances: dict[str, int] = {}
        visited: set[str] = set()
        energy_spent: float = 0.0

        # Track entity types for cross-domain penalty
        node_types: dict[str, str] = dict(seed_entity_types or {})

        queue: deque[tuple[str, float, int]] = deque()

        for node_id, energy in seed_nodes:
            queue.append((node_id, energy, 0))
            visited.add(node_id)
            hop_distances[node_id] = 0

        while queue and energy_spent < cfg.spread_energy_budget:
            node_id, energy, hop = queue.popleft()

            if hop >= cfg.spread_max_hops:
                continue

            neighbors = await neighbor_provider.get_active_neighbors_with_weights(
                node_id, group_id=group_id
            )
            out_degree = len(neighbors)
            if out_degree == 0:
                continue

            fan_factor = max(cfg.fan_s_min, cfg.fan_s_max - math.log(out_degree + 1))

            for neighbor_info in neighbors:
                neighbor_entity_type = None
                if len(neighbor_info) >= 4:
                    neighbor_id = neighbor_info[0]
                    edge_weight = neighbor_info[1]
                    predicate = neighbor_info[2]
                    neighbor_entity_type = neighbor_info[3]
                    predicate_weight = cfg.predicate_weights.get(
                        predicate, cfg.predicate_weight_default
                    )
                elif len(neighbor_info) >= 3:
                    neighbor_id = neighbor_info[0]
                    edge_weight = neighbor_info[1]
                    predicate = neighbor_info[2]
                    predicate_weight = cfg.predicate_weights.get(
                        predicate, cfg.predicate_weight_default
                    )
                else:
                    neighbor_id = neighbor_info[0]
                    edge_weight = neighbor_info[1]
                    predicate = None
                    predicate_weight = cfg.predicate_weight_default

                # Track discovered entity types
                if neighbor_entity_type:
                    node_types[neighbor_id] = neighbor_entity_type

                # Community factor
                community_factor = 1.0
                if (
                    cfg.community_spreading_enabled
                    and community_store is not None
                    and group_id is not None
                ):
                    is_bridge = community_store.is_bridge_edge(
                        node_id,
                        neighbor_id,
                        group_id,
                    )
                    if is_bridge is True:
                        community_factor = cfg.community_bridge_boost
                    elif is_bridge is False:
                        community_factor = cfg.community_intra_dampen

                # Context gate factor
                context_factor = 1.0
                if (
                    cfg.context_gating_enabled
                    and context_gate is not None
                    and predicate is not None
                ):
                    context_factor = context_gate.gate(predicate)

                # Cross-domain penalty (exempt DREAM_ASSOCIATED edges)
                domain_factor = 1.0
                if cfg.cross_domain_penalty_enabled and predicate != "DREAM_ASSOCIATED":
                    source_type = node_types.get(node_id, "")
                    target_type = neighbor_entity_type or node_types.get(neighbor_id, "")
                    if source_type and target_type:
                        source_domain = _resolve_domain(source_type, cfg.domain_groups)
                        target_domain = _resolve_domain(target_type, cfg.domain_groups)
                        if (
                            source_domain is not None
                            and target_domain is not None
                            and source_domain != target_domain
                        ):
                            domain_factor = cfg.cross_domain_penalty_factor

                spread_amount = (
                    energy
                    * edge_weight
                    * predicate_weight
                    * fan_factor
                    * community_factor
                    * context_factor
                    * domain_factor
                    * cfg.spread_decay_per_hop
                )

                if spread_amount < cfg.spread_firing_threshold:
                    continue

                energy_spent += spread_amount
                if energy_spent > cfg.spread_energy_budget:
                    break

                bonuses[neighbor_id] = bonuses.get(neighbor_id, 0.0) + spread_amount

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_hop = hop + 1
                    if neighbor_id not in hop_distances or new_hop < hop_distances[neighbor_id]:
                        hop_distances[neighbor_id] = new_hop
                    queue.append((neighbor_id, spread_amount, new_hop))

        return bonuses, hop_distances
