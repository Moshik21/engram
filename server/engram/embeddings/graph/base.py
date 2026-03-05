"""Base class for graph embedding trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class GraphEmbeddingTrainer(ABC):
    """Abstract base for graph embedding methods.

    Each trainer takes a graph store and produces entity embeddings
    that capture structural position in the knowledge graph.
    """

    @abstractmethod
    async def train(
        self,
        graph_store,
        group_id: str,
        existing_embeddings: dict[str, list[float]] | None = None,
    ) -> dict[str, list[float]]:
        """Train graph embeddings and return {entity_id: vector}.

        Args:
            graph_store: GraphStore to read topology from.
            group_id: Group to train on.
            existing_embeddings: Previous embeddings for warm-start.

        Returns:
            Dict mapping entity_id to embedding vector.
        """

    @abstractmethod
    def method_name(self) -> str:
        """Return the method identifier (e.g. 'node2vec', 'transe', 'gnn')."""
