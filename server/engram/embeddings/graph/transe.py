"""TransE knowledge graph embeddings — pure numpy implementation."""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from engram.config import ActivationConfig
from engram.embeddings.graph.base import GraphEmbeddingTrainer

logger = logging.getLogger(__name__)


class TransETrainer(GraphEmbeddingTrainer):
    """TransE: translational distance model for knowledge graphs.

    Learns entity vectors e and relation vectors r such that
    e_head + r ~= e_tail for valid triples (h, r, t).
    Uses margin-based ranking loss with negative sampling.
    """

    def __init__(self, cfg: ActivationConfig) -> None:
        self._cfg = cfg

    def method_name(self) -> str:
        return "transe"

    async def train(
        self,
        graph_store,
        group_id: str,
        existing_embeddings: dict[str, list[float]] | None = None,
    ) -> dict[str, list[float]]:
        """Extract triples, train TransE, return entity + relation embeddings."""
        cfg = self._cfg

        # 1. Extract all relationship triples
        triples, entity_ids, relation_types = await self._extract_triples(
            graph_store,
            group_id,
        )

        if len(triples) < cfg.graph_embedding_transe_min_triples:
            logger.info(
                "TransE: only %d triples (min %d), skipping",
                len(triples),
                cfg.graph_embedding_transe_min_triples,
            )
            return {}

        # 2. Build vocabularies
        ent_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
        rel_to_idx = {r: i for i, r in enumerate(relation_types)}

        # Convert triples to index form
        triple_indices = []
        for h, r, t in triples:
            if h in ent_to_idx and r in rel_to_idx and t in ent_to_idx:
                triple_indices.append((ent_to_idx[h], rel_to_idx[r], ent_to_idx[t]))

        if not triple_indices:
            return {}

        # 3. Train (CPU-bound)
        entity_emb, relation_emb = await asyncio.to_thread(
            self._train_sync,
            triple_indices,
            len(entity_ids),
            len(relation_types),
            cfg,
        )

        # 4. Map back to IDs
        result: dict[str, list[float]] = {}
        for i, eid in enumerate(entity_ids):
            result[eid] = entity_emb[i].tolist()

        # Also store relation embeddings with predicate name as ID
        for i, rel_type in enumerate(relation_types):
            result[rel_type] = relation_emb[i].tolist()

        logger.info(
            "TransE: trained %d entity + %d relation embeddings (dim=%d)",
            len(entity_ids),
            len(relation_types),
            cfg.graph_embedding_transe_dimensions,
        )
        return result

    async def _extract_triples(
        self,
        graph_store,
        group_id: str,
    ) -> tuple[list[tuple[str, str, str]], list[str], list[str]]:
        """Extract (head_id, predicate, tail_id) triples from the graph.

        Returns (triples, entity_ids, relation_types).
        """
        entities = await graph_store.find_entities(group_id=group_id, limit=50000)
        entity_ids = [e.id for e in entities]
        entity_set = set(entity_ids)

        triples: list[tuple[str, str, str]] = []
        relation_set: set[str] = set()

        for eid in entity_ids:
            try:
                rels = await graph_store.get_relationships(
                    eid,
                    direction="outgoing",
                    group_id=group_id,
                )
                for rel in rels:
                    if rel.target_id in entity_set:
                        triples.append((rel.source_id, rel.predicate, rel.target_id))
                        relation_set.add(rel.predicate)
            except Exception:
                logger.debug("Failed to get relationships for %s, skipping", eid)
                continue

        return triples, entity_ids, sorted(relation_set)

    @staticmethod
    def _train_sync(
        triples: list[tuple[int, int, int]],
        num_entities: int,
        num_relations: int,
        cfg: ActivationConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synchronous TransE training loop."""
        dim = cfg.graph_embedding_transe_dimensions
        margin = cfg.graph_embedding_transe_margin
        lr = cfg.graph_embedding_transe_lr
        epochs = cfg.graph_embedding_transe_epochs
        neg_samples = cfg.graph_embedding_transe_negative_samples
        batch_size = cfg.graph_embedding_transe_batch_size

        rng = np.random.RandomState(42)

        # Initialize embeddings uniformly and L2-normalize
        entity_emb = rng.uniform(-6.0 / dim, 6.0 / dim, (num_entities, dim)).astype(np.float32)
        relation_emb = rng.uniform(-6.0 / dim, 6.0 / dim, (num_relations, dim)).astype(np.float32)

        _l2_normalize(entity_emb)
        _l2_normalize(relation_emb)

        triple_arr = np.array(triples, dtype=np.int32)
        n_triples = len(triples)

        for epoch in range(epochs):
            # Shuffle triples
            perm = rng.permutation(n_triples)

            total_loss = 0.0
            for batch_start in range(0, n_triples, batch_size):
                batch_idx = perm[batch_start : batch_start + batch_size]
                batch = triple_arr[batch_idx]

                heads = batch[:, 0]
                rels = batch[:, 1]
                tails = batch[:, 2]

                h_emb = entity_emb[heads]  # (B, dim)
                r_emb = relation_emb[rels]
                t_emb = entity_emb[tails]

                # Positive distance: ||h + r - t||
                pos_dist = np.linalg.norm(h_emb + r_emb - t_emb, axis=1)

                # Negative sampling: corrupt head or tail
                for _ in range(neg_samples):
                    corrupt_head = rng.random(len(batch)) < 0.5
                    neg_entities = rng.randint(0, num_entities, size=len(batch))

                    neg_h = np.where(corrupt_head[:, None], entity_emb[neg_entities], h_emb)
                    neg_t = np.where(~corrupt_head[:, None], entity_emb[neg_entities], t_emb)

                    neg_dist = np.linalg.norm(neg_h + r_emb - neg_t, axis=1)

                    # Margin-based loss: max(0, margin + d_pos - d_neg)
                    loss = np.maximum(0.0, margin + pos_dist - neg_dist)
                    mask = loss > 0
                    total_loss += loss.sum()

                    if not mask.any():
                        continue

                    # Gradient updates for violating triples
                    # d/dh (||h+r-t||) = (h+r-t) / ||h+r-t||
                    pos_diff = h_emb + r_emb - t_emb
                    pos_norm = np.linalg.norm(pos_diff, axis=1, keepdims=True)
                    pos_norm = np.maximum(pos_norm, 1e-8)
                    pos_grad = pos_diff / pos_norm  # (B, dim)

                    neg_diff = neg_h + r_emb - neg_t
                    neg_norm = np.linalg.norm(neg_diff, axis=1, keepdims=True)
                    neg_norm = np.maximum(neg_norm, 1e-8)
                    neg_grad = neg_diff / neg_norm

                    # Apply gradients only where loss > 0
                    mask_f = mask.astype(np.float32)[:, None]

                    # Update positive triple entities: decrease d(h+r, t)
                    # grad w.r.t. h = +pos_grad → h -= lr * pos_grad
                    # grad w.r.t. t = -pos_grad → t += lr * pos_grad
                    # grad w.r.t. r = +pos_grad → r -= lr * pos_grad
                    np.subtract.at(entity_emb, heads, lr * mask_f * pos_grad)
                    np.add.at(entity_emb, tails, lr * mask_f * pos_grad)
                    np.subtract.at(relation_emb, rels, lr * mask_f * pos_grad)

                    # Update negative triple entities: increase d(neg_h+r, neg_t)
                    # For corrupt head: grad w.r.t. neg_h = +neg_grad → neg_h += lr * neg_grad
                    # For corrupt tail: grad w.r.t. neg_t = -neg_grad → neg_t -= lr * neg_grad
                    neg_update = lr * mask_f * neg_grad
                    corrupt_mask = corrupt_head[:, None].astype(np.float32)
                    # Corrupt head: push neg entity away (add gradient)
                    head_update = neg_update * corrupt_mask
                    # Corrupt tail: push neg entity away (subtract gradient)
                    tail_update = neg_update * (1.0 - corrupt_mask)
                    np.add.at(entity_emb, neg_entities, head_update)
                    np.subtract.at(entity_emb, neg_entities, tail_update)

                    # Update relation for negative triples (symmetric to positive)
                    np.add.at(relation_emb, rels, lr * mask_f * neg_grad)

                # Re-normalize after each batch
                _l2_normalize(entity_emb)

        return entity_emb, relation_emb


def _l2_normalize(matrix: np.ndarray) -> None:
    """In-place L2 normalization of each row."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    matrix /= norms
