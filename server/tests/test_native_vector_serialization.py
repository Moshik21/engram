"""Native HelixDB entity-vector serialization regression.

Closes the "native vector serialization gap": ``get_entity_embeddings`` used to
return empty float payloads on the native PyO3 transport because the generated
``find_entity_vectors_by_ids[_all]`` handlers loaded vector *metadata* without
the embedding floats (``v_from_type(label, false)``). Projecting the ``data``
field in ``schema.hx`` makes the HelixQL compiler emit ``v_from_type(label,
true)``, so the float payload is now serialized on plain-traversal id lookups.

Consequence of the gap (now fixed): MMR diversity, inhibitory spreading, and
state-dependent retrieval silently degraded to no-ops on native because they had
no entity embeddings to compute cosine similarity against.

These tests run in-process (no Docker / running server) and skip when the
``helix_native`` extension is not installed.
"""

from __future__ import annotations

import importlib.util
import logging

import pytest

from engram.config import EmbeddingConfig, HelixDBConfig
from engram.embeddings.provider import EmbeddingProvider
from engram.models import Entity
from engram.retrieval.mmr import apply_mmr
from engram.retrieval.scorer import ScoredResult
from engram.storage.helix.search import HelixSearchIndex

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("helix_native") is None,
    reason="helix_native PyO3 extension is not installed",
)


class _AxisProvider(EmbeddingProvider):
    """Deterministic provider mapping each text to a distinct unit axis.

    Vectors are mutually orthogonal so cosine similarity is exactly 0 between
    different entities and 1 against itself — keeps MMR / cosine assertions
    exact and embedding-content independent.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._axis: dict[str, int] = {}

    def _vector_for(self, text: str) -> list[float]:
        idx = self._axis.setdefault(text, len(self._axis) % self._dim)
        vec = [0.0] * self._dim
        vec[idx] = 1.0
        return vec

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)

    def dimension(self) -> int:
        return self._dim


def _make_native_index(tmp_path) -> HelixSearchIndex:
    return HelixSearchIndex(
        helix_config=HelixDBConfig(
            transport="native",
            data_dir=str(tmp_path / "native-vector-serialization"),
        ),
        provider=_AxisProvider(dim=8),
        embed_config=EmbeddingConfig(),
        embed_provider="test",
        embed_model="axis",
    )


def _entity(eid: str, name: str) -> Entity:
    return Entity(
        id=eid,
        name=name,
        entity_type="Concept",
        group_id="native_vec",
        summary=f"summary for {name}",
    )


@pytest.mark.asyncio
async def test_get_entity_embeddings_returns_full_float_payload(tmp_path, caplog):
    """get_entity_embeddings must return N/N non-empty float vectors on native.

    Asserts the loud "native vector serialization gap" warning is NOT emitted —
    that warning is the observable signal that floats came back empty.
    """
    index = _make_native_index(tmp_path)
    await index.initialize()
    try:
        entity_ids = [f"ent_{i}" for i in range(4)]
        for i, eid in enumerate(entity_ids):
            await index.index_entity(_entity(eid, f"concept-{i}"))

        with caplog.at_level(logging.WARNING, logger="engram.storage.helix.search"):
            # group-scoped endpoint (find_entity_vectors_by_ids)
            scoped = await index.get_entity_embeddings(entity_ids, group_id="native_vec")
            # group-less endpoint (find_entity_vectors_by_ids_all)
            unscoped = await index.get_entity_embeddings(entity_ids, group_id=None)

        for label, embeddings in (("scoped", scoped), ("unscoped", unscoped)):
            assert len(embeddings) == len(entity_ids), (
                f"{label}: expected {len(entity_ids)}/{len(entity_ids)} vectors, "
                f"got {len(embeddings)}"
            )
            for eid in entity_ids:
                vec = embeddings.get(eid)
                assert vec, f"{label}: entity {eid} returned no float payload"
                assert len(vec) == 8, f"{label}: {eid} wrong dim {len(vec)}"
                assert any(v != 0.0 for v in vec), f"{label}: {eid} all-zero vector"

        gap_warnings = [
            r.getMessage()
            for r in caplog.records
            if "native vector serialization gap" in r.getMessage()
        ]
        assert not gap_warnings, (
            "native vector serialization gap warning was emitted — floats are "
            f"still empty on the native transport: {gap_warnings}"
        )
    finally:
        await index.close()


@pytest.mark.asyncio
async def test_mmr_reorders_with_native_embeddings(tmp_path):
    """MMR must actually reorder (not no-op) using native-fetched embeddings.

    With orthogonal axis embeddings, two duplicate-axis entities are maximally
    similar; MMR should demote the second duplicate below a lower-relevance but
    diverse entity, changing the order vs pure relevance sort.
    """
    index = _make_native_index(tmp_path)
    await index.initialize()
    try:
        # a0 and a1 share the same name text => same embedding axis (cosine 1.0).
        # b uses a different axis (orthogonal, cosine 0.0).
        await index.index_entity(_entity("a0", "alpha"))
        await index.index_entity(_entity("a1_dup", "alpha"))
        await index.index_entity(_entity("b", "beta"))

        entity_ids = ["a0", "a1_dup", "b"]
        embeddings = await index.get_entity_embeddings(entity_ids, group_id="native_vec")
        assert len(embeddings) == 3, "precondition: native embeddings must be present"

        # Relevance order a0 > a1_dup > b. Pure relevance keeps that order.
        results = [
            ScoredResult("a0", 0.90, 0.90, 0.0, 0.0, 0.0),
            ScoredResult("a1_dup", 0.80, 0.80, 0.0, 0.0, 0.0),
            ScoredResult("b", 0.50, 0.50, 0.0, 0.0, 0.0),
        ]

        reranked = apply_mmr(
            results,
            entity_embeddings=embeddings,
            lambda_param=0.5,
            top_n=3,
        )
        order = [r.node_id for r in reranked]

        # MMR must demote the near-duplicate a1_dup below the diverse b.
        assert order != ["a0", "a1_dup", "b"], "MMR did not reorder (no-op)"
        assert order.index("b") < order.index("a1_dup"), (
            f"MMR did not diversify away from the duplicate axis: {order}"
        )
    finally:
        await index.close()
