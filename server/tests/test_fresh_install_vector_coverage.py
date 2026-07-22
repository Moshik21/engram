"""Fresh-install episode-vector coverage gate.

Episode vectors used to be written ONLY at projection, and shell installs
never project — every fresh install regrew the vector-less state that made
deep recall return 0/42. Capture-time indexing is the fix; these tests are
the release gate: coverage is asserted, not just that the stack boots.
"""

from __future__ import annotations

import asyncio

import pytest

from engram.config import ActivationConfig
from engram.ingestion.capture_service import EpisodeCaptureService
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.hybrid_search import HybridSearchIndex
from engram.storage.sqlite.search import FTS5SearchIndex
from engram.storage.sqlite.vectors import SQLiteVectorStore


async def _noop_materialize(*_args, **_kwargs) -> None:
    return None


def _service(graph, search, cfg: ActivationConfig) -> EpisodeCaptureService:
    return EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=cfg,
        publish_event=lambda *_args: None,
        materialize_decisions=_noop_materialize,
    )


class FakeGraphStore:
    def __init__(self) -> None:
        self.episodes = {}

    async def create_episode(self, episode):
        self.episodes[episode.id] = episode

    async def update_episode(self, episode_id, updates, group_id="default"):
        return None


class TopicEmbeddingProvider:
    """Deterministic bag-of-topics embedding: same topic => same direction."""

    _TOPICS = (
        ("pottery", "ceramics", "kiln", "glaze"),
        ("kubernetes", "cluster", "deployment", "helm"),
        ("sourdough", "baking", "starter", "flour"),
        ("astronomy", "telescope", "nebula", "stargazing"),
    )

    def dimension(self) -> int:
        return len(self._TOPICS) + 1

    async def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        vec = [0.0] * self.dimension()
        for slot, words in enumerate(self._TOPICS):
            if any(word in lowered for word in words):
                vec[slot] = 1.0
        if not any(vec):
            vec[-1] = 1.0
        return vec

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_query(text) for text in texts]


class RecordingSearchIndex:
    def __init__(self) -> None:
        self.indexed_episodes = []

    async def index_episode(self, episode):
        self.indexed_episodes.append(episode)


class CountingSlowEpisodeIndex:
    def __init__(self, delay_seconds: float = 0.05) -> None:
        self.delay_seconds = delay_seconds
        self.active = 0
        self.max_active = 0
        self.indexed = []

    async def index_episode(self, episode):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay_seconds)
            self.indexed.append(episode.id)
        finally:
            self.active -= 1


class RaisingEpisodeIndex:
    async def index_episode(self, episode):
        raise RuntimeError("embed backend unavailable")


class LatchedHelixLikeIndex:
    """Mirrors the Helix index contract: embed failures are swallowed into
    ``_embed_stats['episodes_failed']`` (the FastEmbed broken-model latch
    returns [] without raising)."""

    def __init__(self, broken: bool) -> None:
        self.broken = broken
        self._embed_stats = {"episodes_indexed": 0, "episodes_failed": 0}
        self.indexed = []

    async def index_episode(self, episode):
        if self.broken:
            self._embed_stats["episodes_failed"] += 1
            return
        self._embed_stats["episodes_indexed"] += 1
        self.indexed.append(episode.id)


@pytest.mark.asyncio
async def test_fresh_install_lite_capture_gives_full_episode_vector_coverage(tmp_path):
    """The release gate: N raw captures on a fresh lite install -> N episode
    vectors present, and a semantic (vector-only) recall finds the episode."""
    db_path = str(tmp_path / "fresh-install.db")
    graph = SQLiteGraphStore(db_path)
    await graph.initialize()
    fts = FTS5SearchIndex(db_path)
    vectors = SQLiteVectorStore(db_path)
    search = HybridSearchIndex(
        fts=fts,
        vector_store=vectors,
        provider=TopicEmbeddingProvider(),
        embed_provider="fake",
        embed_model="topic-5",
    )
    await search.initialize()

    service = _service(graph, search, ActivationConfig())
    contents = [
        "Melanie opened a pottery studio in Bend last spring.",
        "The team moved every service onto the new kubernetes cluster.",
        "Konner keeps a sourdough starter named Clint on the counter.",
        "Saturday night astronomy: the telescope finally resolved a nebula.",
    ]
    episode_ids = [
        await service.store_episode(content, group_id="default", source="test")
        for content in contents
    ]
    await service.drain_cue_indexing()

    # (a) Every captured episode has a vector — coverage, not just boots.
    rows = await (
        await vectors.db.execute("SELECT id FROM embeddings WHERE content_type = 'episode'")
    ).fetchall()
    assert {row["id"] for row in rows} == set(episode_ids)
    assert service.last_stage_timings()["episode_vector_enqueue"] >= 0

    # (b) Semantic recall: query shares zero tokens with the episode text, so
    # FTS5 finds nothing — only the capture-time vector can surface it.
    results = await search.search_episodes("ceramics glaze kiln hobby", group_id="default")
    assert results, "semantic episode recall returned nothing on a fresh install"
    assert results[0][0] == episode_ids[0]

    await search.close()
    await graph.close()


_MACHINERY_CONTENT = (
    "[user|Engram] <task-notification>\n"
    "<task-id>wixqg9mwq</task-id>\n"
    "<tool-use-id>toolu_01M4qmn3MLYm4yhGAfQrmKAb</tool-use-id>\n"
    "<output-file>/private/tmp/tasks/wixqg9mwq.output</output-file>\n"
    "</task-notification>"
)


@pytest.mark.asyncio
async def test_fresh_install_machinery_stored_but_vector_absent(tmp_path):
    """M1.2 salience gate on a fresh install: machinery-class noise is stored
    and BM25-reachable, but never enters vector space."""
    db_path = str(tmp_path / "fresh-install-salience.db")
    graph = SQLiteGraphStore(db_path)
    await graph.initialize()
    fts = FTS5SearchIndex(db_path)
    vectors = SQLiteVectorStore(db_path)
    search = HybridSearchIndex(
        fts=fts,
        vector_store=vectors,
        provider=TopicEmbeddingProvider(),
        embed_provider="fake",
        embed_model="topic-5",
    )
    await search.initialize()

    service = _service(graph, search, ActivationConfig())
    machinery_id = await service.store_episode(
        _MACHINERY_CONTENT,
        group_id="default",
        source="auto:prompt",
    )
    genuine_id = await service.store_episode(
        "Melanie opened a pottery studio in Bend last spring.",
        group_id="default",
        source="auto:prompt",
    )
    await service.drain_cue_indexing()

    # (a) Vector space contains ONLY the genuine episode.
    rows = await (
        await vectors.db.execute("SELECT id FROM embeddings WHERE content_type = 'episode'")
    ).fetchall()
    assert {row["id"] for row in rows} == {genuine_id}

    # (b) The noise episode is stored with its class persisted (round-trip).
    episodes = {ep.id: ep for ep in await graph.get_episodes(group_id="default")}
    assert episodes[machinery_id].salience_class == "machinery"
    assert episodes[genuine_id].salience_class == "substantive"

    # (c) BM25/grep reachability unchanged: lexical search still finds it
    # (the vector lane cannot — there is no vector).
    results = await search.search_episodes("wixqg9mwq task", group_id="default")
    assert any(res[0] == machinery_id for res in results)

    await search.close()
    await graph.close()


@pytest.mark.asyncio
async def test_salience_kill_switch_restores_uniform_indexing(tmp_path):
    """Flag off => byte-identical behavior: machinery is vector-indexed and
    no salience class is persisted."""
    db_path = str(tmp_path / "fresh-install-kill-switch.db")
    graph = SQLiteGraphStore(db_path)
    await graph.initialize()
    fts = FTS5SearchIndex(db_path)
    vectors = SQLiteVectorStore(db_path)
    search = HybridSearchIndex(
        fts=fts,
        vector_store=vectors,
        provider=TopicEmbeddingProvider(),
        embed_provider="fake",
        embed_model="topic-5",
    )
    await search.initialize()

    service = _service(
        graph,
        search,
        ActivationConfig(salience_gated_embedding_enabled=False),
    )
    machinery_id = await service.store_episode(
        _MACHINERY_CONTENT,
        group_id="default",
        source="auto:prompt",
    )
    await service.drain_cue_indexing()

    rows = await (
        await vectors.db.execute("SELECT id FROM embeddings WHERE content_type = 'episode'")
    ).fetchall()
    assert {row["id"] for row in rows} == {machinery_id}

    episodes = {ep.id: ep for ep in await graph.get_episodes(group_id="default")}
    assert episodes[machinery_id].salience_class == ""
    assert episodes[machinery_id].encoding_context is None

    await search.close()
    await graph.close()


@pytest.mark.asyncio
async def test_capture_ack_does_not_wait_for_episode_embedding():
    graph = FakeGraphStore()
    search = CountingSlowEpisodeIndex(delay_seconds=0.5)
    service = _service(graph, search, ActivationConfig())

    started = asyncio.get_running_loop().time()
    episode_id = await service.store_episode(
        "Capture acknowledgement must not pay the embedding latency.",
        group_id="default",
        source="test",
    )
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 0.1
    await service.drain_cue_indexing()
    assert search.indexed == [episode_id]


@pytest.mark.asyncio
async def test_capture_episode_vector_indexing_disabled_by_knob():
    graph = FakeGraphStore()
    search = RecordingSearchIndex()
    service = _service(
        graph,
        search,
        ActivationConfig(capture_episode_vector_index_enabled=False),
    )

    await service.store_episode("Knob off means no capture-time episode embed.")
    await service.drain_cue_indexing()

    assert search.indexed_episodes == []


@pytest.mark.asyncio
async def test_capture_serializes_background_episode_vector_indexing():
    graph = FakeGraphStore()
    search = CountingSlowEpisodeIndex(delay_seconds=0.03)
    service = _service(graph, search, ActivationConfig())

    await asyncio.gather(
        *[
            service.store_episode(
                f"Concurrent capture {index} must not fan out embed writes.",
                group_id="default",
                source="test",
            )
            for index in range(4)
        ],
    )
    await service.drain_cue_indexing()

    assert len(search.indexed) == 4
    assert search.max_active == 1


@pytest.mark.asyncio
async def test_capture_defers_episode_vector_indexing_until_quiet_period():
    graph = FakeGraphStore()
    search = RecordingSearchIndex()
    service = _service(
        graph,
        search,
        ActivationConfig(capture_episode_vector_index_quiet_period_ms=50),
    )

    await service.store_episode(
        "Episode vector indexing should wait out the live capture burst.",
        group_id="default",
        source="test",
    )
    assert search.indexed_episodes == []

    await service.drain_cue_indexing()

    assert len(search.indexed_episodes) == 1
    assert service.last_stage_timings()["episode_vector_quiet_wait"] >= 45


@pytest.mark.asyncio
async def test_episode_index_outbox_retries_after_provider_heals(tmp_path):
    """Broken provider at capture (raise path) -> durable row -> indexed after
    restart once the provider heals."""
    outbox_path = tmp_path / "cue-index-outbox.sqlite3"
    graph = FakeGraphStore()
    broken_service = _service(
        graph,
        RaisingEpisodeIndex(),
        ActivationConfig(cue_index_outbox_path=str(outbox_path)),
    )

    episode_id = await broken_service.store_episode(
        "Episode vectors must survive an embedding outage at capture time.",
        group_id="brain",
        source="test",
    )
    await broken_service.drain_cue_indexing()

    assert episode_id in graph.episodes
    assert broken_service.episode_index_outbox_pending_count() == 1

    healed = RecordingSearchIndex()
    replay_service = _service(
        graph,
        healed,
        ActivationConfig(cue_index_outbox_path=str(outbox_path)),
    )

    startup_replayed = await replay_service.drain_cue_index_outbox(
        limit=10,
        include_failed=False,
    )
    replayed = await replay_service.drain_cue_index_outbox(limit=10)

    assert startup_replayed == 0
    assert replayed == 1
    assert healed.indexed_episodes[0].id == episode_id
    assert healed.indexed_episodes[0].content.startswith("Episode vectors must survive")
    assert replay_service.episode_index_outbox_pending_count() == 0


@pytest.mark.asyncio
async def test_episode_index_outbox_catches_swallowed_broken_model_latch(tmp_path):
    """The FastEmbed broken-model latch returns [] without raising; the index
    swallows it into episodes_failed. The outbox row must stay retryable."""
    outbox_path = tmp_path / "cue-index-outbox.sqlite3"
    graph = FakeGraphStore()
    broken = LatchedHelixLikeIndex(broken=True)
    service = _service(
        graph,
        broken,
        ActivationConfig(cue_index_outbox_path=str(outbox_path)),
    )

    episode_id = await service.store_episode(
        "A latched embed model silently drops vectors; capture must notice.",
        group_id="brain",
        source="test",
    )
    await service.drain_cue_indexing()

    assert broken._embed_stats["episodes_failed"] == 1
    assert service.episode_index_outbox_pending_count() == 1

    healed = LatchedHelixLikeIndex(broken=False)
    replay_service = _service(
        graph,
        healed,
        ActivationConfig(cue_index_outbox_path=str(outbox_path)),
    )
    replayed = await replay_service.drain_cue_index_outbox(limit=10)

    assert replayed == 1
    assert healed.indexed == [episode_id]
    assert replay_service.episode_index_outbox_pending_count() == 0


@pytest.mark.asyncio
async def test_episode_outbox_cleared_after_successful_capture_index(tmp_path):
    outbox_path = tmp_path / "cue-index-outbox.sqlite3"
    graph = FakeGraphStore()
    search = RecordingSearchIndex()
    service = _service(
        graph,
        search,
        ActivationConfig(cue_index_outbox_path=str(outbox_path)),
    )

    await service.store_episode(
        "Successful capture-time indexing should leave no durable debt.",
        group_id="brain",
        source="test",
    )
    await service.drain_cue_indexing()

    assert len(search.indexed_episodes) == 1
    assert service.episode_index_outbox_pending_count() == 0
