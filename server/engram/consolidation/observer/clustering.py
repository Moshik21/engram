"""Deterministic episode clustering by shared graph entity.

Groups episodes that share at least one resolved graph entity into clusters via
union-find. Membership is fully reproducible: episodes are processed in stable
``episode_id`` order and clusters are emitted sorted by their lowest member id,
so identical inputs always yield identical cluster lists (no dict-iteration or
set-ordering dependence). Singletons and clusters below ``min_cluster_size`` are
dropped — an observation only makes sense as a cross-episode synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass

from engram.models.episode import Episode


@dataclass(frozen=True)
class EpisodeCluster:
    """A reproducible group of episodes sharing graph entities."""

    episode_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]


class _UnionFind:
    """Minimal union-find over string ids (deterministic, no path-compression RNG)."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression toward the deterministic root.
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Always attach the lexicographically-larger root under the smaller one
        # so the chosen representative is stable across runs.
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        self._parent[hi] = lo


def cluster_episodes_by_entity(
    episodes: list[Episode],
    episode_entities: dict[str, list[str]],
    *,
    min_cluster_size: int,
) -> list[EpisodeCluster]:
    """Cluster episodes that share at least one entity.

    Parameters
    ----------
    episodes:
        Candidate episodes (any order; sorted internally for determinism).
    episode_entities:
        Map of ``episode_id -> [entity_id, ...]`` (the resolved graph entities
        linked to each episode).
    min_cluster_size:
        Minimum episodes a cluster must contain to be emitted.
    """
    ordered_ids = sorted(ep.id for ep in episodes)
    uf = _UnionFind()
    for eid in ordered_ids:
        uf.find(eid)  # ensure singleton episodes have a root

    # Link episodes that share an entity. Iterate entities and their episodes in
    # stable id order so the union sequence (and thus the chosen roots) is fixed.
    entity_to_eps: dict[str, list[str]] = {}
    for eid in ordered_ids:
        for ent_id in sorted(set(episode_entities.get(eid, []))):
            entity_to_eps.setdefault(ent_id, []).append(eid)

    for ent_id in sorted(entity_to_eps):
        eps = sorted(set(entity_to_eps[ent_id]))
        for other in eps[1:]:
            uf.union(eps[0], other)

    # Collect members per root.
    members: dict[str, list[str]] = {}
    for eid in ordered_ids:
        members.setdefault(uf.find(eid), []).append(eid)

    clusters: list[EpisodeCluster] = []
    for member_ids in members.values():
        member_ids = sorted(member_ids)
        if len(member_ids) < min_cluster_size:
            continue
        ent_ids: set[str] = set()
        for eid in member_ids:
            ent_ids.update(episode_entities.get(eid, []))
        clusters.append(
            EpisodeCluster(
                episode_ids=tuple(member_ids),
                entity_ids=tuple(sorted(ent_ids)),
            )
        )

    # Emit clusters in a stable order (by lowest member id).
    clusters.sort(key=lambda c: c.episode_ids[0])
    return clusters
