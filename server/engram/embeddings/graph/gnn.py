"""GNN (GraphSAGE) graph embeddings — requires torch-geometric."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from engram.config import ActivationConfig
from engram.embeddings.graph.base import GraphEmbeddingTrainer
from engram.embeddings.graph.feature_builder import build_feature_matrix

logger = logging.getLogger(__name__)


class GNNTrainer(GraphEmbeddingTrainer):
    """2-layer GraphSAGE with contrastive loss. Requires torch.

    Trains structural embeddings using self-supervised contrastive learning.
    Node features are initialized from text embeddings. Inductive — can
    embed new nodes without full retraining.
    """

    def __init__(self, cfg: ActivationConfig) -> None:
        self._cfg = cfg
        self._last_weights: dict[str, np.ndarray] | None = None
        self._last_entity_ids: list[str] | None = None
        self._last_id_to_idx: dict[str, int] | None = None

    def method_name(self) -> str:
        return "gnn"

    async def train(
        self,
        graph_store,
        group_id: str,
        existing_embeddings: dict[str, list[float]] | None = None,
        search_index: Any = None,
    ) -> dict[str, list[float]]:
        """Build graph, train GraphSAGE, return embeddings."""
        try:
            import torch  # noqa: F401
        except ImportError:
            logger.warning("GNN training requires torch. Install with: pip install torch")
            return {}

        cfg = self._cfg

        # 1. Get entities and build adjacency
        entities = await graph_store.find_entities(group_id=group_id, limit=50000)
        entity_ids = [e.id for e in entities]

        if len(entity_ids) < cfg.graph_embedding_gnn_min_entities:
            logger.info(
                "GNN: only %d entities (min %d), skipping",
                len(entity_ids), cfg.graph_embedding_gnn_min_entities,
            )
            return {}

        id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

        # 2. Build edge index
        edge_src: list[int] = []
        edge_dst: list[int] = []
        for eid in entity_ids:
            try:
                neighbors = await graph_store.get_active_neighbors_with_weights(
                    eid, group_id=group_id,
                )
                src_idx = id_to_idx[eid]
                for nid, _w, _pred, *_rest in neighbors:
                    if nid in id_to_idx:
                        dst_idx = id_to_idx[nid]
                        edge_src.append(src_idx)
                        edge_dst.append(dst_idx)
            except Exception:
                continue

        if not edge_src:
            logger.info("GNN: no edges found, skipping")
            return {}

        # 3. Build feature matrix from text embeddings
        if search_index is not None:
            features = await build_feature_matrix(entity_ids, search_index, group_id)
        else:
            features = np.random.randn(len(entity_ids), 768).astype(np.float32)

        # 4. Train (CPU-bound)
        result_matrix, weights = await asyncio.to_thread(
            self._train_sync,
            features,
            edge_src,
            edge_dst,
            cfg,
        )

        # 5. Map back to entity IDs
        result = {}
        for i, eid in enumerate(entity_ids):
            result[eid] = result_matrix[i].tolist()

        # Store trained weights for numpy-only inference on new nodes
        self._last_weights = weights
        self._last_entity_ids = entity_ids
        self._last_id_to_idx = id_to_idx

        logger.info(
            "GNN: trained %d entity embeddings (dim=%d)",
            len(result), cfg.graph_embedding_gnn_output_dim,
        )
        return result

    @staticmethod
    def _train_sync(
        features: np.ndarray,
        edge_src: list[int],
        edge_dst: list[int],
        cfg: ActivationConfig,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Synchronous GraphSAGE training with contrastive loss."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional

        num_nodes = features.shape[0]
        input_dim = features.shape[1]
        hidden_dim = cfg.graph_embedding_gnn_hidden_dim
        output_dim = cfg.graph_embedding_gnn_output_dim
        num_layers = cfg.graph_embedding_gnn_layers
        lr = cfg.graph_embedding_gnn_lr
        epochs = cfg.graph_embedding_gnn_epochs

        # Build adjacency dict for mean aggregation
        adj: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
        for s, d in zip(edge_src, edge_dst):
            adj[s].append(d)

        # Simple GraphSAGE model
        class GraphSAGE(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
                for i in range(num_layers):
                    self.layers.append(nn.ModuleDict({
                        "self_lin": nn.Linear(dims[i], dims[i + 1], bias=False),
                        "neigh_lin": nn.Linear(dims[i], dims[i + 1], bias=False),
                    }))

            def forward(self, x: torch.Tensor, adj_dict: dict[int, list[int]]) -> torch.Tensor:
                h = x
                for layer in self.layers:
                    # Mean aggregation
                    h_neigh = torch.zeros_like(h)
                    for node in range(h.shape[0]):
                        neighbors = adj_dict.get(node, [])
                        if neighbors:
                            h_neigh[node] = h[neighbors].mean(dim=0)

                    h = functional.relu(layer["self_lin"](h) + layer["neigh_lin"](h_neigh))
                    h = functional.normalize(h, p=2, dim=-1)
                return h

        model = GraphSAGE()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        x = torch.tensor(features, dtype=torch.float32)

        # Edge tensors for contrastive loss
        edge_tensor = torch.tensor(list(zip(edge_src, edge_dst)), dtype=torch.long)

        for _epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            embeddings = model(x, adj)

            # BPR contrastive loss on edges
            if len(edge_tensor) > 0:
                pos_src = embeddings[edge_tensor[:, 0]]
                pos_dst = embeddings[edge_tensor[:, 1]]

                # Positive scores
                pos_scores = (pos_src * pos_dst).sum(dim=-1)

                # Negative samples: random nodes
                neg_idx = torch.randint(0, num_nodes, (len(edge_tensor),))
                neg_dst = embeddings[neg_idx]
                neg_scores = (pos_src * neg_dst).sum(dim=-1)

                # BPR-like loss
                loss = -functional.logsigmoid(pos_scores - neg_scores).mean()
            else:
                loss = torch.tensor(0.0)

            loss.backward()
            optimizer.step()

        # Extract final embeddings in inference mode
        model.requires_grad_(False)
        with torch.no_grad():
            final_emb = model(x, adj).numpy()

        # Extract weights for numpy inference
        saved_weights: dict[str, np.ndarray] = {}
        for i, layer in enumerate(model.layers):
            saved_weights[f"layer_{i}_self_weight"] = (
                layer["self_lin"].weight.detach().numpy().T
            )
            saved_weights[f"layer_{i}_neigh_weight"] = (
                layer["neigh_lin"].weight.detach().numpy().T
            )

        return final_emb, saved_weights
