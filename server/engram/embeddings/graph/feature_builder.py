"""Build node feature matrices from stored text embeddings."""

from __future__ import annotations

import numpy as np


async def build_feature_matrix(
    entity_ids: list[str],
    search_index,
    group_id: str,
) -> np.ndarray:
    """Build node feature matrix from stored text embeddings.

    Args:
        entity_ids: Ordered list of entity IDs.
        search_index: SearchIndex with get_entity_embeddings().
        group_id: Group ID for embedding lookup.

    Returns:
        (N, D) float32 matrix. Entities without embeddings get zero vectors.
    """
    embeddings = await search_index.get_entity_embeddings(entity_ids, group_id=group_id)

    # Determine dimension from first non-empty embedding
    dim = 0
    for vec in embeddings.values():
        if vec:
            dim = len(vec)
            break
    if dim == 0:
        dim = 768  # fallback

    matrix = np.zeros((len(entity_ids), dim), dtype=np.float32)
    for i, eid in enumerate(entity_ids):
        if eid in embeddings and embeddings[eid]:
            vec = embeddings[eid]
            matrix[i, :len(vec)] = vec[:dim]

    return matrix
