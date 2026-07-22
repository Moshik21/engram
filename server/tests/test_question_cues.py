"""M2.1/M2.2 question-space observe (AGENT_EXPERIENCE_GOAL, D3).

Covers:
- CLI contract: `engram axi observe --question` is repeatable and rides the
  observe POST body; remember has no such flag.
- Question-cue creation: one cue per anticipated question, cue_text IS the
  question, route_reason='agent_question', indexed via the EXISTING cue vector
  lane. Gated on cue lane flags (flag-off inert).
- Storage: question-cue vectors never clobber the base cue vector; cue search
  collapses question-cue hits back to episode ids.
- Proposals on observe: agent-proposed entities ride the deferred-evidence
  pipeline (no projection, no new extraction path).
- End-to-end (lite brain): observe with --question, then recall the question
  verbatim AND paraphrased -> the episode surfaces rank-1 via cue_episode.
"""

from __future__ import annotations

import argparse
import io
from types import SimpleNamespace

import pytest
import pytest_asyncio

from engram.axi import cli as axi_cli
from engram.axi.client import AxiRestClient
from engram.axi.surfaces import AxiResult
from engram.config import ActivationConfig
from engram.graph_manager import GraphManager
from engram.ingestion.capture_surface import (
    AGENT_QUESTION_ROUTE_REASON,
    build_api_observe_write_surface,
    build_mcp_observe_write_surface,
    create_agent_question_cues,
    normalize_observe_questions,
)
from engram.models.episode import EpisodeProjectionState
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.hybrid_search import (
    HybridSearchIndex,
    _collapse_question_cue_hits,
)
from engram.storage.sqlite.search import FTS5SearchIndex
from engram.storage.sqlite.vectors import SQLiteVectorStore
from tests.conftest import MockExtractor

VERBATIM_QUESTION = "why are new memories missing vectors"
PARAPHRASED_QUESTION = "what is the reason recent observations have no embeddings"
ANSWER_CONTENT = (
    "FastEmbed outage root cause: the model.onnx download was interrupted, "
    "leaving a corrupted local model file that silently returned empty output."
)
DISTRACTOR_CONTENT = "Dashboard styling uses Tailwind tokens for the stats panel layout."


class _QuestionSpaceProvider:
    """Deterministic 3D embedding stub separating question-space from answer-space.

    The natural single-fact questions share NO tokens with the stored answer
    content (the answer-locality defect), so only the question-cue lane can
    bridge them.
    """

    def dimension(self) -> int:
        return 3

    async def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        if "vectors" in lowered or "embeddings" in lowered:
            return [1.0, 0.0, 0.0]  # question space
        if "fastembed" in lowered or "model.onnx" in lowered:
            return [0.0, 1.0, 0.0]  # answer content space
        return [0.0, 0.0, 1.0]  # unrelated space

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_query(text) for text in texts]


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="axi")
    axi_cli.configure_axi_parser(parser)
    return parser.parse_args(argv)


def _quiet_question_cue_config() -> ActivationConfig:
    cfg = ActivationConfig(
        cue_layer_enabled=True,
        cue_vector_index_enabled=True,
        cue_recall_enabled=True,
        episode_retrieval_enabled=True,
        # The battery answer-locality condition: episodes without usable
        # answer-space vectors (coarse backfill / FastEmbed outage corpus).
        # The natural single-fact question then shares no tokens with the
        # content either, so ONLY the question-cue lane can reach the episode.
        capture_episode_vector_index_enabled=False,
    )
    cfg.multi_pool_enabled = False
    cfg.graph_query_expansion_enabled = False
    cfg.template_reformulation_enabled = False
    cfg.query_decomposition_enabled = False
    cfg.recall_planner_enabled = False
    return cfg


@pytest_asyncio.fixture
async def lite_question_brain(tmp_path):
    cfg = _quiet_question_cue_config()
    db_path = str(tmp_path / "question_cues.db")
    graph_store = SQLiteGraphStore(db_path)
    await graph_store.initialize()
    search_index = HybridSearchIndex(
        FTS5SearchIndex(db_path),
        SQLiteVectorStore(db_path),
        _QuestionSpaceProvider(),
        cfg=cfg,
        storage_dim=0,
        embed_provider="stub",
        embed_model="question-space-3",
    )
    await search_index.initialize(db=graph_store._db)
    activation_store = MemoryActivationStore(cfg=cfg)
    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        MockExtractor(),
        cfg=cfg,
    )
    yield manager, graph_store, search_index
    await graph_store.close()


# ── CLI flag contract (D3) ───────────────────────────────────────────────────


class TestObserveQuestionFlag:
    def test_question_flag_is_repeatable(self):
        args = _parse(
            [
                "observe",
                "--stdin",
                "--question",
                "who owns engram",
                "--question",
                "what is the north star",
            ]
        )
        assert args.questions == ["who owns engram", "what is the north star"]

    def test_question_flag_defaults_to_none(self):
        args = _parse(["observe", "--stdin"])
        assert args.questions is None

    def test_remember_has_no_question_flag(self):
        with pytest.raises(SystemExit):
            _parse(["remember", "--stdin", "--question", "who owns engram"])

    def test_dispatch_passes_questions_to_write_payload(self, monkeypatch):
        captured: dict = {}

        def _fake_build_write_payload(client, **kwargs):
            captured.update(kwargs)
            return AxiResult(payload={"operation": kwargs["operation"]}, exit_code=0)

        monkeypatch.setattr(axi_cli, "build_write_payload", _fake_build_write_payload)
        monkeypatch.setattr("sys.stdin", io.StringIO("fact text"))
        args = _parse(["observe", "--stdin", "--question", "who owns engram"])
        result = axi_cli._dispatch(args, client=object())
        assert result.exit_code == 0
        assert captured["operation"] == "observe"
        assert captured["questions"] == ["who owns engram"]

    def test_client_observe_body_carries_questions(self, monkeypatch):
        client = AxiRestClient(server_url="http://localhost:1", timeout_seconds=1)
        captured: dict = {}

        def _fake_request_json(method, path, **kwargs):
            captured.update({"method": method, "path": path, **kwargs})
            return {"status": "observed"}

        monkeypatch.setattr(client, "request_json", _fake_request_json)
        client.observe(content="c", source="axi", questions=["q1", "q2"])
        assert captured["path"] == "/api/knowledge/observe"
        assert captured["body"]["questions"] == ["q1", "q2"]

    def test_client_observe_body_omits_questions_when_absent(self, monkeypatch):
        client = AxiRestClient(server_url="http://localhost:1", timeout_seconds=1)
        captured: dict = {}

        def _fake_request_json(method, path, **kwargs):
            captured.update(kwargs)
            return {"status": "observed"}

        monkeypatch.setattr(client, "request_json", _fake_request_json)
        client.observe(content="c", source="axi")
        assert "questions" not in captured["body"]


# ── Question normalization + creation gating ─────────────────────────────────


class TestQuestionCueCreation:
    def test_normalize_dedupes_strips_and_caps(self):
        questions = [
            "  what   broke   recall  ",
            "What broke recall",
            "",
            "   ",
            42,
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
        ]
        normalized = normalize_observe_questions(questions)
        assert normalized[0] == "what broke recall"
        assert len(normalized) == 5  # capped, deduped case-insensitively

    def test_normalize_rejects_non_list(self):
        assert normalize_observe_questions("not a list") == []
        assert normalize_observe_questions(None) == []

    @pytest.mark.asyncio
    async def test_flag_off_is_inert(self):
        calls: list = []

        async def _index(cue):
            calls.append(cue)

        manager = SimpleNamespace(
            _cfg=ActivationConfig(cue_layer_enabled=False, cue_vector_index_enabled=True),
            _search=SimpleNamespace(index_episode_cue=_index),
        )
        created = await create_agent_question_cues(
            manager,
            episode_id="ep_x",
            group_id="g",
            questions=["who owns engram"],
        )
        assert created == 0
        assert calls == []

    @pytest.mark.asyncio
    async def test_creates_one_cue_per_question(self):
        calls: list = []

        async def _index(cue):
            calls.append(cue)

        manager = SimpleNamespace(
            _cfg=ActivationConfig(cue_layer_enabled=True, cue_vector_index_enabled=True),
            _search=SimpleNamespace(index_episode_cue=_index),
        )
        created = await create_agent_question_cues(
            manager,
            episode_id="ep_x",
            group_id="g",
            questions=["who owns engram", "what is the north star"],
        )
        assert created == 2
        assert [c.cue_text for c in calls] == [
            "who owns engram",
            "what is the north star",
        ]
        for cue in calls:
            assert cue.episode_id == "ep_x"
            assert cue.route_reason == AGENT_QUESTION_ROUTE_REASON
            assert cue.projection_state == EpisodeProjectionState.CUE_ONLY

    @pytest.mark.asyncio
    async def test_index_failure_never_raises(self):
        async def _boom(cue):
            raise RuntimeError("embed outage")

        manager = SimpleNamespace(
            _cfg=ActivationConfig(cue_layer_enabled=True, cue_vector_index_enabled=True),
            _search=SimpleNamespace(index_episode_cue=_boom),
        )
        created = await create_agent_question_cues(
            manager,
            episode_id="ep_x",
            group_id="g",
            questions=["who owns engram"],
        )
        assert created == 0


# ── Storage: question-cue vectors coexist with the base cue vector ───────────


class TestQuestionCueVectorStorage:
    def test_collapse_keeps_best_score_per_episode_preserving_order(self):
        hits = [
            ("ep_a::q::abc123", 0.9),
            ("ep_b", 0.7),
            ("ep_a", 0.4),
            ("ep_a::q::def456", 0.2),
        ]
        assert _collapse_question_cue_hits(hits) == [("ep_a", 0.9), ("ep_b", 0.7)]

    def test_collapse_is_identity_without_question_cues(self):
        hits = [("ep_a", 0.9), ("ep_b", 0.7)]
        assert _collapse_question_cue_hits(hits) == hits

    @pytest.mark.asyncio
    async def test_question_vectors_do_not_clobber_base_cue_vector(
        self,
        lite_question_brain,
    ):
        from engram.models.episode_cue import EpisodeCue

        manager, graph_store, search_index = lite_question_brain
        base = EpisodeCue(
            episode_id="ep_vec",
            group_id="g_vec",
            cue_text="FastEmbed model.onnx interrupted download",
        )
        await search_index.index_episode_cue(base)
        for question in (VERBATIM_QUESTION, "why are embeddings absent"):
            await search_index.index_episode_cue(
                EpisodeCue(
                    episode_id="ep_vec",
                    group_id="g_vec",
                    cue_text=question,
                    route_reason=AGENT_QUESTION_ROUTE_REASON,
                )
            )

        cursor = await graph_store._db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE content_type = 'episode_cue' "
            "AND group_id = 'g_vec'"
        )
        row = await cursor.fetchone()
        assert row[0] == 3  # base + 2 question cues, no clobbering

        results = await search_index.search_episode_cues(
            VERBATIM_QUESTION,
            group_id="g_vec",
            limit=5,
        )
        assert results
        assert results[0][0] == "ep_vec"  # collapsed back to the episode id


# ── Proposals on observe ride the deferred-evidence pipeline ─────────────────


class TestObserveProposals:
    @pytest.mark.asyncio
    async def test_api_observe_proposals_persist_deferred(self, lite_question_brain):
        manager, graph_store, _search = lite_question_brain
        content = "Aurelia works at Nimbus Corp on the platform team."
        payload = await build_api_observe_write_surface(
            manager,
            content=content,
            group_id="g_prop",
            source="api",
            proposed_entities=[
                {"name": "Aurelia", "entity_type": "Person", "source_span": content},
            ],
        )
        assert payload["status"] == "observed"

        # No projection at observe time.
        nodes = await graph_store.find_entity_candidates("Aurelia", "g_prop")
        assert not [e for e in nodes if e.name == "Aurelia"]

        pending = await graph_store.get_pending_evidence(group_id="g_prop", limit=20)
        assert pending, "observe proposals must persist deferred evidence"
        assert {p["status"] for p in pending} == {"deferred"}
        names = {p["payload"].get("name") for p in pending if p["fact_class"] == "entity"}
        assert "Aurelia" in names


# ── End-to-end: observe with questions -> recall verbatim + paraphrased ──────


async def _observe_corpus(manager) -> str:
    payload = await build_api_observe_write_surface(
        manager,
        content=ANSWER_CONTENT,
        group_id="default",
        source="axi",
        questions=[VERBATIM_QUESTION],
    )
    assert payload["status"] == "observed"
    assert payload["questionCues"] == 1
    episode_id = payload["episodeId"]

    distractor = await build_api_observe_write_surface(
        manager,
        content=DISTRACTOR_CONTENT,
        group_id="default",
        source="axi",
    )
    assert "questionCues" not in distractor
    await manager._capture_service.drain_cue_indexing()
    return episode_id


async def _assert_rank1_cue_episode(manager, query: str, episode_id: str) -> None:
    results = await manager.recall(
        query,
        group_id="default",
        limit=5,
        record_access=False,
    )
    assert results, f"no recall results for {query!r}"
    top = results[0]
    assert top["result_type"] == "cue_episode", (
        f"expected cue_episode rank-1 for {query!r}, got {top['result_type']}"
    )
    assert top["episode"]["id"] == episode_id


@pytest.mark.asyncio
async def test_question_recall_verbatim_rank1_via_cue_episode(lite_question_brain):
    manager, _graph, _search = lite_question_brain
    episode_id = await _observe_corpus(manager)
    await _assert_rank1_cue_episode(manager, VERBATIM_QUESTION, episode_id)


@pytest.mark.asyncio
async def test_question_recall_paraphrased_rank1_via_cue_episode(lite_question_brain):
    manager, _graph, _search = lite_question_brain
    episode_id = await _observe_corpus(manager)
    await _assert_rank1_cue_episode(manager, PARAPHRASED_QUESTION, episode_id)


@pytest.mark.asyncio
async def test_mcp_observe_reports_question_cue_count(lite_question_brain):
    manager, _graph, _search = lite_question_brain

    async def _noop(*_args, **_kwargs):
        return None

    session = SimpleNamespace(session_id="s_q", episode_count=0, last_activity=None)
    response = await build_mcp_observe_write_surface(
        manager,
        content=ANSWER_CONTENT,
        group_id="default",
        session=session,
        source="mcp",
        questions=[VERBATIM_QUESTION, "why are embeddings absent"],
        ingest_live_turn=_noop,
        recall_middleware=_noop,
    )
    assert response["question_cues"] == 2
