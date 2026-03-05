"""Pure numpy Skip-gram with negative sampling for graph embedding training."""

from __future__ import annotations

import numpy as np


class NumpySkipGram:
    """Skip-gram word2vec implementation using numpy only.

    Trains embeddings from sequences of entity IDs (random walks).
    Uses negative sampling for efficient training.
    """

    def __init__(
        self,
        vocab_size: int,
        dimensions: int = 64,
        window: int = 5,
        negative_samples: int = 5,
        lr: float = 0.025,
        epochs: int = 5,
        seed: int | None = None,
    ) -> None:
        self._vocab_size = vocab_size
        self._dimensions = dimensions
        self._window = window
        self._negative_samples = negative_samples
        self._lr = lr
        self._epochs = epochs
        self._rng = np.random.RandomState(seed)

        # Initialize embedding matrices
        self._W_in = self._rng.uniform(
            -0.5 / dimensions, 0.5 / dimensions,
            (vocab_size, dimensions),
        ).astype(np.float32)
        self._W_out = np.zeros(
            (vocab_size, dimensions), dtype=np.float32,
        )

    def train(self, walks: list[list[int]]) -> np.ndarray:
        """Train skip-gram on random walk sequences.

        Args:
            walks: List of walks, each walk is a list of integer node indices.

        Returns:
            W_in embedding matrix of shape (vocab_size, dimensions).
        """
        # Build unigram distribution for negative sampling (^0.75 smoothing)
        counts = np.zeros(self._vocab_size, dtype=np.float64)
        for walk in walks:
            for idx in walk:
                counts[idx] += 1.0
        counts_smoothed = np.power(counts, 0.75)
        total = counts_smoothed.sum()
        if total == 0:
            return self._W_in.copy()
        neg_dist = counts_smoothed / total

        for epoch in range(self._epochs):
            # Decay learning rate linearly
            lr = self._lr * (1.0 - epoch / max(self._epochs, 1))
            lr = max(lr, self._lr * 0.0001)

            for walk in walks:
                walk_len = len(walk)
                for pos, center in enumerate(walk):
                    # Dynamic window
                    win = self._rng.randint(1, self._window + 1)
                    start = max(0, pos - win)
                    end = min(walk_len, pos + win + 1)

                    for ctx_pos in range(start, end):
                        if ctx_pos == pos:
                            continue
                        context = walk[ctx_pos]
                        self._train_pair(center, context, neg_dist, lr)

        return self._W_in.copy()

    def _train_pair(
        self,
        center: int,
        context: int,
        neg_dist: np.ndarray,
        lr: float,
    ) -> None:
        """Train a single (center, context) pair with negative sampling."""
        v_in = self._W_in[center]  # (dim,)

        # Positive sample
        v_out = self._W_out[context]  # (dim,)
        dot = np.dot(v_in, v_out)
        dot = np.clip(dot, -6.0, 6.0)
        sig = _sigmoid(dot)
        grad = lr * (1.0 - sig)
        grad_in = grad * v_out

        self._W_out[context] += grad * v_in

        # Negative samples (with collision replacement)
        neg_drawn = 0
        max_attempts = self._negative_samples * 3
        attempts = 0
        while neg_drawn < self._negative_samples and attempts < max_attempts:
            neg_idx = self._rng.choice(self._vocab_size, p=neg_dist)
            attempts += 1
            if neg_idx == context:
                continue
            neg_drawn += 1
            v_neg = self._W_out[neg_idx]
            dot_neg = np.dot(v_in, v_neg)
            dot_neg = np.clip(dot_neg, -6.0, 6.0)
            sig_neg = _sigmoid(dot_neg)
            grad_neg = lr * (-sig_neg)
            grad_in += grad_neg * v_neg
            self._W_out[neg_idx] += grad_neg * v_in

        self._W_in[center] += grad_in


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))
