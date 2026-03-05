"""Numpy-only GraphSAGE forward pass for inference without torch."""

from __future__ import annotations

import numpy as np


class GraphSAGEInference:
    """Numpy inference with saved weight matrices -- no torch needed.

    Performs mean-aggregation GraphSAGE forward pass using pre-trained weights.
    """

    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        """Initialize with saved weights from training.

        Expected keys per layer i:
            f"layer_{i}_self_weight", f"layer_{i}_neigh_weight", f"layer_{i}_bias"
        """
        self._weights = weights
        self._num_layers = 0
        while f"layer_{self._num_layers}_self_weight" in weights:
            self._num_layers += 1

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def forward(
        self,
        features: np.ndarray,
        adj: dict[int, list[int]],
    ) -> np.ndarray:
        """Forward pass: mean-aggregate neighbors, concatenate with self, project.

        Args:
            features: (N, input_dim) node feature matrix.
            adj: {node_idx: [neighbor_indices]}.

        Returns:
            (N, output_dim) embedding matrix.
        """
        h = features.astype(np.float32)
        n = h.shape[0]

        for layer_i in range(self._num_layers):
            w_self = self._weights[f"layer_{layer_i}_self_weight"]  # (in, out)
            w_neigh = self._weights[f"layer_{layer_i}_neigh_weight"]  # (in, out)
            bias = self._weights.get(f"layer_{layer_i}_bias")

            # Mean aggregation of neighbors
            h_neigh = np.zeros_like(h)
            for node in range(n):
                neighbors = adj.get(node, [])
                if neighbors:
                    h_neigh[node] = h[neighbors].mean(axis=0)
                # else: zero vector (self-loop effect via w_self)

            # h_new = ReLU(h @ w_self + h_neigh @ w_neigh + bias)
            out = h @ w_self + h_neigh @ w_neigh
            if bias is not None:
                out += bias
            h = np.maximum(out, 0.0)  # ReLU

            # L2-normalize after each layer
            norms = np.linalg.norm(h, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            h = h / norms

        return h
