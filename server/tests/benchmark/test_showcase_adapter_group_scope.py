"""Showcase adapter contract tests."""

from __future__ import annotations

import pytest

from engram.benchmark.showcase.adapters import EngramAdapter
from engram.benchmark.showcase.models import ExtractionSpec, ScenarioProbe, ScenarioTurn
from engram.config import ActivationConfig


class _RecordingManager:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def store_episode(
        self,
        content: str,
        *,
        group_id: str,
        source: str | None = None,
        session_id: str | None = None,
    ) -> str:
        self.calls.append(("store_episode", group_id, content, source, session_id))
        return "ep_observed"

    async def ingest_episode(
        self,
        content: str,
        *,
        group_id: str,
        source: str | None = None,
        session_id: str | None = None,
    ) -> str:
        self.calls.append(("ingest_episode", group_id, content, source, session_id))
        return "ep_remembered"

    async def project_episode(self, episode_id: str, *, group_id: str) -> None:
        self.calls.append(("project_episode", group_id, episode_id))

    async def create_intention(
        self,
        *,
        trigger_text: str,
        action_text: str,
        trigger_type: str,
        entity_names: list[str],
        threshold: float | None,
        priority: str,
        group_id: str,
        context: str | None,
        see_also: list[str],
    ) -> str:
        self.calls.append(
            (
                "create_intention",
                group_id,
                trigger_text,
                action_text,
                trigger_type,
                tuple(entity_names),
                threshold,
                priority,
                context,
                tuple(see_also),
            )
        )
        return "int_showcase"

    async def dismiss_intention(self, intention_id: str, *, group_id: str, hard: bool) -> None:
        self.calls.append(("dismiss_intention", group_id, intention_id, hard))

    async def get_context(
        self,
        *,
        group_id: str,
        max_tokens: int,
        topic_hint: str | None,
    ) -> dict:
        self.calls.append(("get_context", group_id, max_tokens, topic_hint))
        return {"context": "showcase context", "token_estimate": 2}

    async def recall(
        self,
        *,
        query: str,
        group_id: str,
        limit: int,
        interaction_type: str,
        interaction_source: str,
    ) -> list[dict]:
        self.calls.append(("recall", group_id, query, limit, interaction_type, interaction_source))
        return [
            {
                "result_type": "entity",
                "entity": {
                    "id": "ent_source",
                    "name": "Source",
                    "type": "TestMemory",
                    "summary": "Source summary",
                },
                "relationships": [
                    {
                        "source_id": "ent_source",
                        "target_id": "ent_target",
                        "predicate": "USES",
                        "polarity": "positive",
                    }
                ],
            }
        ]

    async def resolve_entity_name(self, entity_id: str, group_id: str) -> str:
        self.calls.append(("resolve_entity_name", group_id, entity_id))
        return entity_id


class _FakeClosable:
    def __init__(self, name: str, closed: list[str]) -> None:
        self.name = name
        self._closed = closed

    async def close(self) -> None:
        self._closed.append(self.name)


@pytest.mark.asyncio
async def test_engram_showcase_adapter_uses_explicit_non_default_group() -> None:
    manager = _RecordingManager()
    adapter = EngramAdapter(
        "engram full smoke",
        ActivationConfig(integration_profile="rework"),
        {"remembered": ExtractionSpec()},
    )
    adapter._manager = manager

    group_id = adapter._group_id
    assert group_id == "showcase_engram_full_smoke"

    await adapter.apply_turn(
        ScenarioTurn(id="observe", action="observe", content="observed", source="test")
    )
    await adapter.apply_turn(
        ScenarioTurn(id="remember", action="remember", content="remembered", source="test")
    )
    await adapter.apply_turn(ScenarioTurn(id="project", action="project", ref="observe"))
    await adapter.apply_turn(
        ScenarioTurn(
            id="intend",
            action="intend",
            trigger_text="when Source activates",
            action_text="surface Target",
            entity_names=["Source"],
        )
    )
    await adapter.apply_turn(
        ScenarioTurn(id="dismiss", action="dismiss_intention", ref="intend", hard_delete=True)
    )

    await adapter.retrieve_evidence(
        ScenarioProbe(
            id="context",
            after_turn_index=0,
            operation="get_context",
            topic_hint="Source",
        )
    )
    await adapter.retrieve_evidence(
        ScenarioProbe(
            id="recall",
            after_turn_index=0,
            operation="recall",
            query="Source",
        )
    )

    group_scoped_methods = {
        "store_episode",
        "ingest_episode",
        "project_episode",
        "create_intention",
        "dismiss_intention",
        "get_context",
        "recall",
        "resolve_entity_name",
    }
    scoped_calls = [call for call in manager.calls if call[0] in group_scoped_methods]
    assert scoped_calls
    assert all(call[1] == group_id for call in scoped_calls)


@pytest.mark.asyncio
async def test_engram_showcase_adapter_closes_search_before_graph(tmp_path) -> None:
    closed: list[str] = []
    adapter = EngramAdapter(
        "engram full smoke",
        ActivationConfig(integration_profile="rework"),
        {"remembered": ExtractionSpec()},
    )
    temp_dir = tmp_path / "showcase"
    temp_dir.mkdir()
    adapter._temp_dir = temp_dir
    adapter._search_index = _FakeClosable("search", closed)
    adapter._graph_store = _FakeClosable("graph", closed)

    await adapter.close()

    assert closed == ["search", "graph"]
    assert adapter._search_index is None
    assert adapter._graph_store is None
    assert adapter._manager is None
    assert adapter._temp_dir is None
    assert not temp_dir.exists()
