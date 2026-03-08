"""Node2Vec random walk + Skip-gram graph embeddings, pure numpy."""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from engram.config import ActivationConfig
from engram.embeddings.graph.base import GraphEmbeddingTrainer
from engram.embeddings.graph.skipgram import NumpySkipGram

logger = logging.getLogger(__name__)


class Node2VecTrainer(GraphEmbeddingTrainer):
    """Random walk + Skip-gram graph embeddings.

    Generates biased random walks on the knowledge graph, then trains
    Skip-gram embeddings so structurally similar nodes cluster together.
    Pure numpy — no external dependencies beyond numpy.
    """

    def __init__(self, cfg: ActivationConfig) -> None:
        self._cfg = cfg

    def method_name(self) -> str:
        return "node2vec"

    async def train(
        self,
        graph_store,
        group_id: str,
        existing_embeddings: dict[str, list[float]] | None = None,
    ) -> dict[str, list[float]]:
        """Build adjacency, generate walks, train Skip-gram."""
        cfg = self._cfg

        # 1. Build adjacency list
        adj, entity_ids = await self._build_adjacency(graph_store, group_id)

        if len(entity_ids) < cfg.graph_embedding_node2vec_min_entities:
            logger.info(
                "Node2Vec: only %d entities (min %d), skipping",
                len(entity_ids), cfg.graph_embedding_node2vec_min_entities,
            )
            return {}

        # 2. Build ID ↔ index mapping
        id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
        adj_idx: dict[int, list[tuple[int, float]]] = {}
        for eid, neighbors in adj.items():
            src_idx = id_to_idx[eid]
            adj_idx[src_idx] = [
                (id_to_idx[nid], w) for nid, w in neighbors if nid in id_to_idx
            ]

        initial_in = _build_initial_matrix(
            entity_ids,
            cfg.graph_embedding_node2vec_dimensions,
            existing_embeddings,
        )

        # 3. Generate walks and train (CPU-bound, run in thread)
        embeddings_matrix = await asyncio.to_thread(
            self._train_sync, adj_idx, len(entity_ids), cfg, initial_in,
        )

        # 4. Map back to entity IDs
        result = {}
        for i, eid in enumerate(entity_ids):
            result[eid] = embeddings_matrix[i].tolist()

        logger.info(
            "Node2Vec: trained %d entity embeddings (dim=%d)",
            len(result), cfg.graph_embedding_node2vec_dimensions,
        )
        return result

    async def train_incremental(
        self,
        graph_store,
        group_id: str,
        entity_ids: set[str],
        existing_embeddings: dict[str, list[float]] | None = None,
    ) -> dict[str, list[float]]:
        """Train on a dirty subgraph and warm-start from prior embeddings."""
        cfg = self._cfg
        if len(entity_ids) < 2:
            logger.info("Node2Vec incremental: only %d scoped entities, skipping", len(entity_ids))
            return {}

        adj, ordered_entity_ids = await self._build_adjacency_subset(
            graph_store,
            group_id,
            entity_ids,
        )
        if len(ordered_entity_ids) < 2:
            return {}

        id_to_idx = {eid: i for i, eid in enumerate(ordered_entity_ids)}
        adj_idx: dict[int, list[tuple[int, float]]] = {}
        for eid, neighbors in adj.items():
            src_idx = id_to_idx[eid]
            adj_idx[src_idx] = [
                (id_to_idx[nid], w) for nid, w in neighbors if nid in id_to_idx
            ]

        initial_in = _build_initial_matrix(
            ordered_entity_ids,
            cfg.graph_embedding_node2vec_dimensions,
            existing_embeddings,
        )
        embeddings_matrix = await asyncio.to_thread(
            self._train_sync,
            adj_idx,
            len(ordered_entity_ids),
            cfg,
            initial_in,
        )

        result = {}
        for i, eid in enumerate(ordered_entity_ids):
            result[eid] = embeddings_matrix[i].tolist()

        logger.info(
            "Node2Vec incremental: trained %d entity embeddings (dim=%d)",
            len(result), cfg.graph_embedding_node2vec_dimensions,
        )
        return result

    async def _build_adjacency(
        self, graph_store, group_id: str,
    ) -> tuple[dict[str, list[tuple[str, float]]], list[str]]:
        """Build weighted adjacency list from graph store.

        Returns (adjacency_dict, entity_id_list).
        """
        entities = await graph_store.find_entities(group_id=group_id, limit=50000)
        entity_ids = [e.id for e in entities]
        adj: dict[str, list[tuple[str, float]]] = {eid: [] for eid in entity_ids}
        entity_set = set(entity_ids)

        for eid in entity_ids:
            try:
                neighbors = await graph_store.get_active_neighbors_with_weights(
                    eid, group_id=group_id,
                )
                for nid, weight, _pred, *_rest in neighbors:
                    if nid in entity_set:
                        adj[eid].append((nid, max(weight, 0.01)))
            except Exception:
                logger.debug("Failed to get neighbors for %s, skipping", eid)
                continue

        return adj, entity_ids

    async def _build_adjacency_subset(
        self,
        graph_store,
        group_id: str,
        entity_ids: set[str],
    ) -> tuple[dict[str, list[tuple[str, float]]], list[str]]:
        """Build weighted adjacency for a scoped subgraph only."""
        ordered_ids = sorted(entity_ids)
        entity_set = set(ordered_ids)
        adj: dict[str, list[tuple[str, float]]] = {eid: [] for eid in ordered_ids}

        for eid in ordered_ids:
            try:
                neighbors = await graph_store.get_active_neighbors_with_weights(
                    eid, group_id=group_id,
                )
                for nid, weight, _pred, *_rest in neighbors:
                    if nid in entity_set:
                        adj[eid].append((nid, max(weight, 0.01)))
            except Exception:
                logger.debug("Failed to get neighbors for %s, skipping", eid)
                continue

        return adj, ordered_ids

    @staticmethod
    def _train_sync(
        adj_idx: dict[int, list[tuple[int, float]]],
        vocab_size: int,
        cfg: ActivationConfig,
        initial_in: np.ndarray | None = None,
    ) -> np.ndarray:
        """Synchronous training: generate walks + Skip-gram."""
        rng = np.random.RandomState(42)

        # Generate biased random walks
        walks = _generate_walks(
            adj_idx,
            vocab_size,
            num_walks=cfg.graph_embedding_node2vec_num_walks,
            walk_length=cfg.graph_embedding_node2vec_walk_length,
            p=cfg.graph_embedding_node2vec_p,
            q=cfg.graph_embedding_node2vec_q,
            rng=rng,
        )

        # Train Skip-gram
        sg = NumpySkipGram(
            vocab_size=vocab_size,
            dimensions=cfg.graph_embedding_node2vec_dimensions,
            window=cfg.graph_embedding_node2vec_window,
            negative_samples=5,
            lr=0.025,
            epochs=cfg.graph_embedding_node2vec_epochs,
            seed=42,
            initial_in=initial_in,
        )
        return sg.train(walks)


def _build_initial_matrix(
    entity_ids: list[str],
    dimensions: int,
    existing_embeddings: dict[str, list[float]] | None,
) -> np.ndarray | None:
    if not existing_embeddings:
        return None

    matrix = np.zeros((len(entity_ids), dimensions), dtype=np.float32)
    found = False
    for index, entity_id in enumerate(entity_ids):
        vector = existing_embeddings.get(entity_id)
        if vector is None or len(vector) != dimensions:
            continue
        matrix[index] = np.asarray(vector, dtype=np.float32)
        found = True
    return matrix if found else None


def _build_neighbor_sets(
    adj: dict[int, list[tuple[int, float]]],
) -> dict[int, set[int]]:
    """Build O(1) neighbor lookup sets from adjacency lists."""
    return {node: {n for n, _ in neighbors} for node, neighbors in adj.items()}


def _generate_walks(
    adj: dict[int, list[tuple[int, float]]],
    vocab_size: int,
    num_walks: int,
    walk_length: int,
    p: float,
    q: float,
    rng: np.random.RandomState,
) -> list[list[int]]:
    """Generate biased random walks for all nodes."""
    walks: list[list[int]] = []
    nodes = list(range(vocab_size))
    neighbor_sets = _build_neighbor_sets(adj)

    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = _random_walk(adj, neighbor_sets, start, walk_length, p, q, rng)
            if len(walk) > 1:
                walks.append(walk)

    return walks


def _random_walk(
    adj: dict[int, list[tuple[int, float]]],
    neighbor_sets: dict[int, set[int]],
    start: int,
    walk_length: int,
    p: float,
    q: float,
    rng: np.random.RandomState,
) -> list[int]:
    """Single biased random walk with Node2Vec p, q parameters.

    p: Return parameter (higher = less likely to revisit previous node).
    q: In-out parameter (higher = more local, lower = more exploratory).
    """
    walk = [start]
    prev: int | None = None

    for _ in range(walk_length - 1):
        curr = walk[-1]
        neighbors = adj.get(curr, [])
        if not neighbors:
            break

        if prev is None:
            # First step: weighted random choice
            weights = np.array([w for _, w in neighbors], dtype=np.float64)
            weights /= weights.sum()
            idx = rng.choice(len(neighbors), p=weights)
            prev = curr
            walk.append(neighbors[idx][0])
        else:
            # Biased walk: adjust weights based on p, q
            neighbor_ids = [n for n, _ in neighbors]
            raw_weights = np.array([w for _, w in neighbors], dtype=np.float64)
            prev_neighbors = neighbor_sets.get(prev, set())

            # Apply Node2Vec bias
            bias = np.ones(len(neighbors), dtype=np.float64)
            for i, (nid, _) in enumerate(neighbors):
                if nid == prev:
                    bias[i] = 1.0 / p  # Return to previous
                elif nid in prev_neighbors:
                    bias[i] = 1.0  # BFS-like (neighbor of previous)
                else:
                    bias[i] = 1.0 / q  # DFS-like (away from previous)

            weights = raw_weights * bias
            total = weights.sum()
            if total == 0:
                break
            weights /= total

            idx = rng.choice(len(neighbors), p=weights)
            prev = curr
            walk.append(neighbor_ids[idx])

    return walk
