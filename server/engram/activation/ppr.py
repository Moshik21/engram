"""Personalized PageRank spreading activation strategy."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

from engram.activation.bfs import _resolve_domain
from engram.config import ActivationConfig

if TYPE_CHECKING:
    from engram.activation.context_gate import ContextGate


class PPRStrategy:
    """Personalized PageRank with local subgraph expansion.

    Instead of loading the full graph, BFS-expands `ppr_expansion_hops`
    from seed nodes to build a local adjacency map, then runs power
    iteration on that subgraph.
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
        if not seed_nodes:
            return {}, {}

        # 1. Build seed vector (normalized)
        total_energy = sum(e for _, e in seed_nodes)
        if total_energy <= 0:
            return {}, {}
        seed_vec: dict[str, float] = {nid: e / total_energy for nid, e in seed_nodes}
        seed_ids = set(seed_vec.keys())

        # Track entity types for cross-domain penalty
        node_types: dict[str, str] = dict(seed_entity_types or {})

        # 2. BFS-expand local subgraph
        adjacency, hop_distances = await self._expand_subgraph(
            seed_ids, neighbor_provider, cfg, group_id, node_types
        )

        if not adjacency:
            return {}, hop_distances

        # 3. Build row-stochastic transition matrix with typed edge weights
        all_nodes = set(adjacency.keys())
        for neighbors in adjacency.values():
            for neighbor_tuple in neighbors:
                all_nodes.add(neighbor_tuple[0])

        transition: dict[str, list[tuple[str, float]]] = {}
        for node, neighbors in adjacency.items():
            if not neighbors:
                continue
            fan_factor = max(cfg.fan_s_min, cfg.fan_s_max - math.log(len(neighbors) + 1))
            # Build raw weights with community, context, and domain factors
            raw_weights: list[tuple[str, float]] = []
            for neighbor_tuple in neighbors:
                nid = neighbor_tuple[0]
                ew = neighbor_tuple[1]
                p = neighbor_tuple[2]
                neighbor_type = neighbor_tuple[3] if len(neighbor_tuple) >= 4 else None
                pw = cfg.predicate_weights.get(p, cfg.predicate_weight_default)
                cf = 1.0
                if cfg.community_spreading_enabled and community_store is not None and group_id:
                    is_bridge = community_store.is_bridge_edge(
                        node,
                        nid,
                        group_id,
                    )
                    if is_bridge is True:
                        cf = cfg.community_bridge_boost
                    elif is_bridge is False:
                        cf = cfg.community_intra_dampen
                cg = 1.0
                if cfg.context_gating_enabled and context_gate is not None:
                    cg = context_gate.gate(p)
                # Cross-domain penalty (exempt DREAM_ASSOCIATED edges)
                df = 1.0
                if cfg.cross_domain_penalty_enabled and p != "DREAM_ASSOCIATED":
                    src_type = node_types.get(node, "")
                    tgt_type = neighbor_type or node_types.get(nid, "")
                    if src_type and tgt_type:
                        src_domain = _resolve_domain(src_type, cfg.domain_groups)
                        tgt_domain = _resolve_domain(tgt_type, cfg.domain_groups)
                        if (
                            src_domain is not None
                            and tgt_domain is not None
                            and src_domain != tgt_domain
                        ):
                            df = cfg.cross_domain_penalty_factor
                raw_weights.append((nid, ew * pw * cf * cg * df))
            total_w = sum(w for _, w in raw_weights)
            if total_w <= 0:
                continue
            transition[node] = [(nid, w * fan_factor / total_w) for nid, w in raw_weights]

        # 4. Power iteration: r = alpha * seed_vec + (1-alpha) * M^T * r
        alpha = cfg.ppr_alpha
        r: dict[str, float] = dict(seed_vec)  # initialize to seed
        for nid in all_nodes:
            r.setdefault(nid, 0.0)

        for _ in range(cfg.ppr_max_iterations):
            new_r: dict[str, float] = {nid: alpha * seed_vec.get(nid, 0.0) for nid in all_nodes}
            for node, edges in transition.items():
                r_node = r.get(node, 0.0)
                if r_node <= 0:
                    continue
                for neighbor, weight in edges:
                    new_r[neighbor] = new_r.get(neighbor, 0.0) + (1 - alpha) * r_node * weight

            # Check convergence (L1 norm)
            diff = sum(abs(new_r.get(n, 0) - r.get(n, 0)) for n in all_nodes)
            r = new_r
            if diff < cfg.ppr_epsilon:
                break

        # 5. Convert to bonuses (exclude seeds from bonuses to match BFS)
        bonuses: dict[str, float] = {}
        for nid, score in r.items():
            if nid not in seed_ids and score > 0:
                bonuses[nid] = score

        return bonuses, hop_distances

    async def _expand_subgraph(
        self,
        seed_ids: set[str],
        neighbor_provider,
        cfg: ActivationConfig,
        group_id: str | None,
        node_types: dict[str, str] | None = None,
    ) -> tuple[
        dict[str, list[tuple[str, float, str, str]]],
        dict[str, int],
    ]:
        """BFS-expand from seeds to build local adjacency map.

        Returns (adjacency, hop_distances).
        """
        adjacency: dict[str, list[tuple[str, float, str, str]]] = {}
        hop_distances: dict[str, int] = {}
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()

        for sid in seed_ids:
            queue.append((sid, 0))
            visited.add(sid)
            hop_distances[sid] = 0

        max_hops = cfg.ppr_expansion_hops

        while queue:
            node_id, hop = queue.popleft()

            neighbors_raw = await neighbor_provider.get_active_neighbors_with_weights(
                node_id, group_id=group_id
            )

            neighbors: list[tuple[str, float, str, str]] = []
            for info in neighbors_raw:
                if len(info) >= 4:
                    neighbors.append((info[0], info[1], info[2], info[3]))
                    if node_types is not None and info[3]:
                        node_types[info[0]] = info[3]
                elif len(info) >= 3:
                    neighbors.append((info[0], info[1], info[2], ""))
                else:
                    neighbors.append((info[0], info[1], "__DEFAULT__", ""))

            adjacency[node_id] = neighbors

            if hop < max_hops:
                for nid, _, _, _ in neighbors:
                    if nid not in visited:
                        visited.add(nid)
                        new_hop = hop + 1
                        hop_distances[nid] = new_hop
                        queue.append((nid, new_hop))

        return adjacency, hop_distances
