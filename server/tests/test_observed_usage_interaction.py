"""The `used` interaction must be reachable from a non-dashboard surface.

`retrieval/chat_feedback.py apply_chat_recall_feedback` was the only emitter of
the `used`/`dismissed` interaction vocabulary, and it hangs off POST
`/api/knowledge/chat` — the dashboard. Every MCP/axi consumer, i.e. the actual
product, could never emit it. `used_count` and `surfaced_to_used_ratio` were
therefore structurally `0` for the real user: not "memories are never used" but
"this surface cannot say".

The detector already existed. `record_observed_usage_events` runs on the observe
fast path, finds surfaced entities and cues genuinely relied on in the next
captured turn, and writes a `used`-TIER ACCESS EVENT for each — a signal the
product already trusts. It just never became a `used` INTERACTION, so the
recall-need controller (rolling counters + the adaptive-threshold learner that
reads `interaction_counts["used"]`) never saw it.

`dismissed` is deliberately NOT emitted here and that is a declaration, not an
omission: the capture surface has no bounded response to partition, so every
memory not echoed in one observe would read as rejected. An over-firing
`dismissed` is a plausible-but-wrong metric (STANDING_GOAL 2.1) and strictly
worse than an honest absence. `test_observed_echo_never_emits_dismissed` pins
that choice so a later change has to argue with it.

Regime: isolated, in-process, lite/SQLite store. No live measurement.
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.extraction.extractor import EntityExtractor, ExtractionResult
from engram.graph_manager import GraphManager
from engram.ingestion.capture_surface import store_observation
from engram.models import Entity
from engram.retrieval.feedback import get_usage_buffer
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex

_GROUP = "default"
_ENTITY_ID = "ent_nimbus_corp"
_ENTITY_NAME = "Nimbus Corp"
# What recall handed the agent.
_SURFACED_SNIPPET = "Nimbus Corp is the platform vendor for the ingest tier."
# What the agent captured next. The name is reused inside NOVEL tokens, so the
# echo guard sees reliance rather than a verbatim parrot of the payload.
_ECHOING_CAPTURE = (
    "Kicked off the Q3 contract renewal negotiation with Nimbus Corp this "
    "morning and agreed a two week extension."
)


class _EmptyExtractor(EntityExtractor):
    def __init__(self) -> None:
        self._result = ExtractionResult(entities=[], relationships=[])

    async def extract(self, text: str) -> ExtractionResult:
        return self._result


@pytest_asyncio.fixture
async def usage_manager():
    """Lite manager with usage feedback on, plus a clean surfaced-usage buffer."""
    get_usage_buffer().reset()
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "observed-usage.db")
    graph_store = SQLiteGraphStore(db_path)
    await graph_store.initialize()
    search_index = FTS5SearchIndex(db_path)
    await search_index.initialize(db=graph_store._db)
    cfg = ActivationConfig(recall_usage_feedback_enabled=True)
    activation_store = MemoryActivationStore(cfg=cfg)
    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        _EmptyExtractor(),
        cfg=cfg,
    )
    await graph_store.create_entity(
        Entity(
            id=_ENTITY_ID,
            name=_ENTITY_NAME,
            entity_type="Organization",
            group_id=_GROUP,
            summary=_SURFACED_SNIPPET,
        )
    )
    yield manager, activation_store
    get_usage_buffer().reset()
    await graph_store.close()


def _mark_surfaced() -> None:
    """Register the entity as surfaced, exactly as record_entity_access does.

    ``ts`` must be now: the buffer ages surfaced payloads out of eligibility on
    a TTL, so a fixed timestamp makes this fixture silently stop firing.
    """
    get_usage_buffer().note_surfaced(
        _GROUP,
        entity_id=_ENTITY_ID,
        name=_ENTITY_NAME,
        snippet=f"{_ENTITY_NAME} {_SURFACED_SNIPPET}",
        ts=time.time(),
    )


async def _observe(manager: GraphManager, content: str) -> None:
    await store_observation(
        manager,
        content=content,
        group_id=_GROUP,
        source="mcp_observe",
    )


def _usage_event_count(activation_store: MemoryActivationStore) -> int:
    state = activation_store._states.get(_ENTITY_ID)
    return len(state.usage_events) if state is not None else 0


@pytest.mark.asyncio
async def test_observed_echo_emits_a_used_interaction(usage_manager):
    """THE ASSERTION NOBODY WROTE. Before the fix the echo detector fired, wrote
    its used-tier access event, and `used_count` stayed 0 — the interaction had
    no emitter outside the dashboard."""
    manager, _activation = usage_manager
    _mark_surfaced()

    assert manager.get_recall_metrics(_GROUP)["used_count"] == 0

    await _observe(manager, _ECHOING_CAPTURE)

    metrics = manager.get_recall_metrics(_GROUP)
    assert metrics["used_count"] == 1, (
        "an entity the agent demonstrably relied on must reach the used "
        f"interaction vocabulary from the observe surface; got {metrics}"
    )


@pytest.mark.asyncio
async def test_observed_echo_reaches_per_memory_feedback(usage_manager):
    """The per-memory trust summary is a second consumer of the same signal."""
    manager, _activation = usage_manager
    _mark_surfaced()

    await _observe(manager, _ECHOING_CAPTURE)

    summary = manager.get_recall_feedback_summary(_GROUP, [_ENTITY_ID])
    assert summary.get(_ENTITY_ID, {}).get("used_count") == 1


@pytest.mark.asyncio
async def test_observed_echo_records_exactly_one_access_event(usage_manager):
    """The detector already wrote the used-tier access event. Emitting the
    interaction must not write a second one — a double access event is phantom
    reinforcement, which is the exact bug class this repo keeps paying for."""
    manager, activation = usage_manager
    _mark_surfaced()

    await _observe(manager, _ECHOING_CAPTURE)

    assert _usage_event_count(activation) == 1


@pytest.mark.asyncio
async def test_observed_echo_never_emits_dismissed(usage_manager):
    """DECLARED, not accidental: the capture surface cannot derive dismissal."""
    manager, _activation = usage_manager
    _mark_surfaced()

    await _observe(manager, _ECHOING_CAPTURE)

    assert manager.get_recall_metrics(_GROUP)["dismissed_count"] == 0


@pytest.mark.asyncio
async def test_capture_without_reliance_emits_nothing(usage_manager):
    """Control: an unrelated capture must not manufacture a use."""
    manager, activation = usage_manager
    _mark_surfaced()

    await _observe(manager, "Reviewed the quarterly infrastructure budget spreadsheet.")

    assert manager.get_recall_metrics(_GROUP)["used_count"] == 0
    assert _usage_event_count(activation) == 0


@pytest.mark.asyncio
async def test_verbatim_parrot_emits_nothing(usage_manager):
    """Control: echoing the surfaced payload back is not reliance."""
    manager, activation = usage_manager
    _mark_surfaced()

    await _observe(manager, _SURFACED_SNIPPET)

    assert manager.get_recall_metrics(_GROUP)["used_count"] == 0
    assert _usage_event_count(activation) == 0
