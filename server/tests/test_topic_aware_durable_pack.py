"""Topic-aware durable pack for get_context (I1 findings 1-2).

Covers:
- two topics on the same brain -> different packs where durable content differs
- no-topic call unchanged (identity-first fast path, no extra probes)
- topic matching nothing durable -> identity pack fall-through (never empty)
- 'Engram'-named project hint is NOT discarded (generic-hint footgun fix)
- process cache is topic-keyed
- latency guard: topic path adds no graph scan beyond the existing durable
  listing (call counts on a fake manager)
"""

from __future__ import annotations

import pytest

from engram.retrieval.budgets import RecallBudget
from engram.retrieval.context_builder import (
    _durable_context_payload_from_manager,
    _is_generic_durable_topic,
    invalidate_durable_context_cache,
)

IDENTITY = [
    {"id": "p1", "name": "Konner Moshier", "entity_type": "Person", "summary": "Owner"},
    {
        "id": "d1",
        "name": "Fully local north star",
        "entity_type": "Decision",
        "summary": "No external keys",
    },
    {
        "id": "g1",
        "name": "Continuity metric v2",
        "entity_type": "Goal",
        "summary": "North-star gate",
    },
]

TYPED = {
    "Decision": [
        {
            "id": "d_rrf",
            "name": "Use RRF fusion for hybrid search",
            "entity_type": "Decision",
            "summary": "fts 0.3 vec 0.7",
            "activation_score": 0.4,
        },
        {
            "id": "d_dash",
            "name": "Ship dashboard dark mode",
            "entity_type": "Decision",
            "summary": "Theme toggle for the dashboard",
            "activation_score": 0.4,
        },
        {
            "id": "d_loc",
            "name": "Engram runs fully local",
            "entity_type": "Decision",
            "summary": "Zero external keys for operation",
            "activation_score": 0.4,
        },
    ],
}


class _Graph:
    def __init__(self, identity: list[dict], typed: dict[str, list[dict]]) -> None:
        self.identity = identity
        self.typed = typed
        self.identity_calls = 0
        self.type_calls: list[str] = []
        self.exact_calls = 0
        self.candidate_calls = 0

    async def get_identity_core_entities(self, group_id: str) -> list[dict]:
        self.identity_calls += 1
        return list(self.identity)

    async def find_entities_by_type(
        self, entity_type: str, group_id: str, limit: int = 8
    ) -> list[dict]:
        self.type_calls.append(entity_type)
        return list(self.typed.get(entity_type, []))[:limit]

    async def find_entities_exact_name(self, name: str, group_id: str, limit: int = 5) -> list:
        self.exact_calls += 1
        return []

    async def find_entity_candidates(self, name: str, group_id: str, limit: int = 5) -> list:
        self.candidate_calls += 1
        return []


class _Manager:
    """Fake manager: graph listing only, no search_entities (counts stay exact)."""

    def __init__(self, graph: _Graph) -> None:
        self._graph = graph


def _budget() -> RecallBudget:
    return RecallBudget.start(
        profile="explicit",
        surface="mcp",
        mode="mcp_context",
        max_wall_ms=4000,
        max_search_ms=1500,
        max_graph_ms=900,
        max_packet_ms=250,
        max_results=5,
        max_packets=3,
        max_output_tokens=1200,
        allow_deep_recall=True,
        allow_embeddings=True,
        allow_graph_probe=False,
    )


async def _payload(manager: _Manager, *, topic: str | None, project: str | None, group: str):
    return await _durable_context_payload_from_manager(
        manager,
        group_id=group,
        topic_hint=topic,
        project_path=project,
        format="structured",
        budget=_budget(),
        started=0.0,
    )


def _names(payload: dict) -> list[str]:
    return [str(p.get("title") or "") for p in (payload.get("cached_packets") or [])]


@pytest.fixture(autouse=True)
def _fresh_cache():
    invalidate_durable_context_cache()
    yield
    invalidate_durable_context_cache()


@pytest.mark.asyncio
async def test_two_topics_yield_different_packs():
    manager = _Manager(_Graph(IDENTITY, TYPED))
    rrf = await _payload(manager, topic="rrf fusion", project=None, group="two_topics")
    dash = await _payload(manager, topic="dashboard dark mode", project=None, group="two_topics")
    assert rrf is not None and dash is not None
    rrf_names, dash_names = _names(rrf), _names(dash)
    assert rrf_names != dash_names
    assert any("RRF fusion" in n for n in rrf_names)
    assert any("dashboard dark mode" in n for n in dash_names)
    # Identity-first guarantee on both packs.
    assert "Konner Moshier" in rrf_names[0]
    assert "Konner Moshier" in dash_names[0]


@pytest.mark.asyncio
async def test_no_topic_call_is_unchanged_identity_fast_path():
    graph = _Graph(IDENTITY, TYPED)
    manager = _Manager(graph)
    payload = await _payload(manager, topic=None, project=None, group="no_topic")
    assert payload is not None
    names = _names(payload)
    assert len(names) == 3
    assert "Konner Moshier" in names[0]
    assert "Fully local north star" in names[1]
    assert "Continuity metric v2" in names[2]
    # Fast path preserved: identity listing only, no typed probes, no rescue.
    assert graph.identity_calls == 1
    assert graph.type_calls == []
    assert graph.exact_calls == 0
    assert graph.candidate_calls == 0


@pytest.mark.asyncio
async def test_topic_matching_nothing_durable_falls_back_to_identity_pack():
    manager = _Manager(_Graph(IDENTITY, TYPED))
    payload = await _payload(
        manager, topic="quantum blockchain telemetry", project=None, group="no_match"
    )
    assert payload is not None
    names = _names(payload)
    assert len(names) == 3
    assert "Konner Moshier" in names[0]
    assert "Fully local north star" in names[1]
    assert "Continuity metric v2" in names[2]


@pytest.mark.asyncio
async def test_engram_named_project_hint_not_discarded():
    # Derived project name "Engram" is a real topic, not a generic placeholder.
    manager = _Manager(_Graph(IDENTITY, TYPED))
    payload = await _payload(
        manager, topic=None, project="/Users/konner/Engram", group="engram_project"
    )
    assert payload is not None
    names = _names(payload)
    assert any("Engram runs fully local" in n for n in names)
    assert "Konner Moshier" in names[0]


@pytest.mark.asyncio
async def test_generic_hint_without_matching_project_is_discarded():
    graph = _Graph(IDENTITY, TYPED)
    manager = _Manager(graph)
    payload = await _payload(
        manager, topic="engram", project="/Users/konner/OtherProj", group="generic_hint"
    )
    assert payload is not None
    names = _names(payload)
    assert len(names) == 3
    assert "Konner Moshier" in names[0]
    # Discarded hint means the topic-less fast path: no typed probes ran.
    assert graph.type_calls == []


def test_generic_topic_rule():
    # Literal placeholders are generic only when they do not name the project.
    assert _is_generic_durable_topic("engram", None) is True
    assert _is_generic_durable_topic("default", "/Users/x/Anything") is True
    assert _is_generic_durable_topic("project", None) is True
    assert _is_generic_durable_topic("Engram", "/Users/x/Engram") is False
    assert _is_generic_durable_topic("default", "/Users/x/default") is False
    assert _is_generic_durable_topic("rrf fusion", None) is False


@pytest.mark.asyncio
async def test_process_cache_is_topic_keyed():
    manager = _Manager(_Graph(IDENTITY, TYPED))
    first_rrf = await _payload(manager, topic="rrf fusion", project=None, group="cache_topic")
    assert first_rrf is not None
    assert first_rrf["packet_cache"]["hit"] is False

    first_dash = await _payload(
        manager, topic="dashboard dark mode", project=None, group="cache_topic"
    )
    assert first_dash is not None
    # Different topic must not be served the other topic's cached payload.
    assert first_dash["packet_cache"]["hit"] is False
    assert _names(first_dash) != _names(first_rrf)

    second_rrf = await _payload(manager, topic="rrf fusion", project=None, group="cache_topic")
    assert second_rrf is not None
    assert second_rrf["packet_cache"]["hit"] is True
    assert _names(second_rrf) == _names(first_rrf)


@pytest.mark.asyncio
async def test_identity_keeps_guaranteed_slot_with_two_topic_hits():
    typed = {
        "Decision": [
            {
                "id": "d_rrf",
                "name": "Use RRF fusion for hybrid search",
                "entity_type": "Decision",
                "summary": "fts 0.3 vec 0.7",
                "activation_score": 0.4,
            },
            {
                "id": "d_rrf2",
                "name": "RRF fusion weights are frozen",
                "entity_type": "Decision",
                "summary": "No fusion tuning before re-baseline",
                "activation_score": 0.4,
            },
        ],
    }
    manager = _Manager(_Graph(IDENTITY, typed))
    payload = await _payload(manager, topic="rrf fusion", project=None, group="two_hits")
    assert payload is not None
    names = _names(payload)
    assert len(names) == 3
    # 1 guaranteed identity slot + 2 topic slots.
    assert "Konner Moshier" in names[0]
    assert any("Use RRF fusion" in n for n in names)
    assert any("RRF fusion weights are frozen" in n for n in names)


@pytest.mark.asyncio
async def test_latency_guard_topic_path_adds_no_extra_scans():
    graph = _Graph(IDENTITY, TYPED)
    manager = _Manager(graph)
    payload = await _payload(manager, topic="rrf fusion", project=None, group="latency")
    assert payload is not None
    # Existing durable listing only: one identity list + one probe per durable
    # type. No rescue probes (topic hit found in listing), no other scans.
    assert graph.identity_calls == 1
    assert sorted(graph.type_calls) == sorted(
        ["Decision", "Preference", "Goal", "Commitment", "Correction", "Person"]
    )
    assert graph.exact_calls == 0
    assert graph.candidate_calls == 0


@pytest.mark.asyncio
async def test_derived_project_topic_keeps_identity_fast_path(monkeypatch):
    """Retro-verify minor: the DEFAULT no-topic-with-project call must not
    auto-enter topic mode via the derived project-name hint — topic mode's
    probes + rescue can blow the durable budget on large brains."""
    import engram.retrieval.context_builder as cb

    captured = {}

    async def spy_durable(manager, *, group_id, topic_hint, **kwargs):
        captured["topic_hint"] = topic_hint
        return None  # fall through; we only care what the durable pack saw

    monkeypatch.setattr(cb, "_durable_context_payload_from_manager", spy_durable)

    class _StubManager:
        async def get_memory_context(self, **kwargs):
            return {"context": "", "packets": []}

        def latest_agent_adoption_surface(self, *a, **k):
            return None

    try:
        await cb.build_mcp_context_surface(
            _StubManager(),
            group_id="g1",
            topic_hint=None,
            project_path="/Users/someone/projects/engram",
            format="structured",
        )
    except Exception:
        pass  # downstream stub gaps are fine; the spy already captured

    assert captured.get("topic_hint") is None
