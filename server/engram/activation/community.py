"""Community detection via label propagation and community-aware spreading."""

from __future__ import annotations

import random
import time


class CommunityStore:
    """In-memory community assignments with lazy label propagation.

    Holds ``{group_id -> {entity_id -> community_label}}`` and recomputes
    via label propagation when assignments become stale (TTL-based).
    """

    def __init__(
        self,
        stale_seconds: float = 300.0,
        max_iterations: int = 10,
        seed: int = 42,
    ) -> None:
        self._stale_seconds = stale_seconds
        self._max_iterations = max_iterations
        self._seed = seed
        # group_id -> {entity_id -> community_label}
        self._assignments: dict[str, dict[str, str]] = {}
        # group_id -> timestamp of last computation
        self._timestamps: dict[str, float] = {}

    def get_community(self, entity_id: str, group_id: str) -> str | None:
        """Return the community label for an entity, or None if unknown."""
        group_data = self._assignments.get(group_id)
        if group_data is None:
            return None
        return group_data.get(entity_id)

    def is_stale(self, group_id: str) -> bool:
        """Return True if assignments for this group are stale or missing."""
        ts = self._timestamps.get(group_id)
        if ts is None:
            return True
        return (time.monotonic() - ts) > self._stale_seconds

    def is_bridge_edge(
        self, src_id: str, dst_id: str, group_id: str,
    ) -> bool | None:
        """Return True if cross-cluster, False if same-cluster, None if unknown."""
        group_data = self._assignments.get(group_id)
        if group_data is None:
            return None
        src_comm = group_data.get(src_id)
        dst_comm = group_data.get(dst_id)
        if src_comm is None or dst_comm is None:
            return None
        return src_comm != dst_comm

    async def ensure_fresh(
        self,
        group_id: str,
        neighbor_provider,
        entity_ids: list[str] | None = None,
    ) -> None:
        """Recompute community assignments if stale."""
        if not self.is_stale(group_id):
            return
        assignments = await self.compute(
            group_id, neighbor_provider, entity_ids,
        )
        self._assignments[group_id] = assignments
        self._timestamps[group_id] = time.monotonic()

    async def compute(
        self,
        group_id: str,
        neighbor_provider,
        entity_ids: list[str] | None = None,
    ) -> dict[str, str]:
        """Run label propagation and return {entity_id: community_label}."""
        return await label_propagation(
            neighbor_provider,
            group_id,
            entity_ids=entity_ids,
            max_iterations=self._max_iterations,
            seed=self._seed,
        )

    def set_assignments(
        self, group_id: str, assignments: dict[str, str],
    ) -> None:
        """Inject known community assignments (e.g. from benchmark corpus)."""
        self._assignments[group_id] = dict(assignments)
        self._timestamps[group_id] = time.monotonic()

    def clear(self, group_id: str | None = None) -> None:
        """Clear cached assignments. If group_id is None, clear all."""
        if group_id is None:
            self._assignments.clear()
            self._timestamps.clear()
        else:
            self._assignments.pop(group_id, None)
            self._timestamps.pop(group_id, None)


async def label_propagation(
    neighbor_provider,
    group_id: str,
    entity_ids: list[str] | None = None,
    max_iterations: int = 10,
    seed: int = 42,
) -> dict[str, str]:
    """Label propagation community detection.

    Each node starts with its own ID as label. Each iteration, nodes adopt
    the most common label among neighbors (ties broken by seeded RNG).
    O(E) per iteration, converges in ~5 iterations typically.
    """
    rng = random.Random(seed)

    if not entity_ids:
        return {}

    # Initialize: each node gets its own label
    labels: dict[str, str] = {eid: eid for eid in entity_ids}

    # Build adjacency by querying neighbor_provider
    adjacency: dict[str, list[str]] = {}
    for eid in entity_ids:
        neighbors_raw = await neighbor_provider.get_active_neighbors_with_weights(
            eid, group_id=group_id,
        )
        neighbor_ids = []
        for info in neighbors_raw:
            nid = info[0]
            if nid in labels:  # Only include nodes in our entity set
                neighbor_ids.append(nid)
        adjacency[eid] = neighbor_ids

    # Iterate
    node_list = list(entity_ids)
    for _ in range(max_iterations):
        rng.shuffle(node_list)
        changed = False

        for node in node_list:
            neighbors = adjacency.get(node, [])
            if not neighbors:
                continue

            # Count neighbor labels
            label_counts: dict[str, int] = {}
            for nid in neighbors:
                lbl = labels[nid]
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

            # Find max count
            max_count = max(label_counts.values())
            candidates = [
                lbl for lbl, cnt in label_counts.items() if cnt == max_count
            ]

            # Break ties deterministically with seeded RNG
            new_label = rng.choice(sorted(candidates))
            if new_label != labels[node]:
                labels[node] = new_label
                changed = True

        if not changed:
            break

    return labels
