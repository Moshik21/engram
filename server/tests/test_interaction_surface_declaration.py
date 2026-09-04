"""A verb no surface can emit must read as UNMEASURABLE, never as a zero.

Ticket #37, half (b). `used` got a real non-dashboard emitter (see
tests/test_observed_usage_interaction.py). `dismissed` and `selected` did not,
because neither can be honestly derived from a capture or an MCP recall — and
`confirmed`/`corrected` are in the accepted vocabulary with no emitter on any
surface at all. Before this, every one of them reported `0` on an MCP install
and `false_recall_rate: 0.0` went out alongside, which reads as "no false
recalls measured" when the truth is "nothing here can say". That is the
plausible-but-wrong metric STANDING_GOAL 2.1 forbids: it does not merely fail
to inform, it ENDS the investigation with a false answer.

So the deadness is now declared (engram/retrieval/interaction_surfaces.py) and
published next to the counts, and the one derived RATE that depended on a verb
nothing can emit goes absent instead of confident.

Two properties hold this honest and both are pinned below:
  * the declaration can never label a verb dead that actually fired — the
    observed counts override the table, so a stale table under-reports deadness
    and can never bury a live signal;
  * the declaration cannot go stale silently — a verb declared emitter-less
    with an emitter in the tree, or declared emittable with none, fails.

Regime: isolated, in-process, lite/SQLite store. No live measurement.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.evaluation.brain_loop_report import _recall_summary
from engram.extraction.extractor import EntityExtractor, ExtractionResult
from engram.graph_manager import GraphManager
from engram.ingestion.capture_surface import store_observation
from engram.models import Entity
from engram.retrieval.control import RecallNeedController
from engram.retrieval.feedback import get_usage_buffer
from engram.retrieval.interaction_surfaces import (
    AGENT_CAPTURE,
    DASHBOARD_CHAT,
    INTERACTION_EMITTERS,
    surface_for_source,
)
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex

_GROUP = "default"
_ENTITY_ID = "ent_nimbus_corp"
_ENTITY_NAME = "Nimbus Corp"
_SURFACED_SNIPPET = "Nimbus Corp is the platform vendor for the ingest tier."
_ECHOING_CAPTURE = (
    "Kicked off the Q3 contract renewal negotiation with Nimbus Corp this "
    "morning and agreed a two week extension."
)

# The real `source` strings, taken from the call sites rather than invented
# (STANDING_GOAL 2.7): retrieval/chat_feedback.py posts `chat_response` after a
# dashboard reply; public_surface_policy hands the chat tool loop
# `chat_tool_select`; capture_surface emits `observed_echo`; recall_surface uses
# `mcp_recall`.
_DASHBOARD_SOURCE = "chat_response"
_MCP_RECALL_SOURCE = "mcp_recall"


class _EmptyExtractor(EntityExtractor):
    def __init__(self) -> None:
        self._result = ExtractionResult(entities=[], relationships=[])

    async def extract(self, text: str) -> ExtractionResult:
        return self._result


@pytest_asyncio.fixture
async def usage_manager():
    get_usage_buffer().reset()
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "interaction-surface.db")
    graph_store = SQLiteGraphStore(db_path)
    await graph_store.initialize()
    search_index = FTS5SearchIndex(db_path)
    await search_index.initialize(db=graph_store._db)
    cfg = ActivationConfig(recall_usage_feedback_enabled=True)
    manager = GraphManager(
        graph_store,
        MemoryActivationStore(cfg=cfg),
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
    yield manager
    get_usage_buffer().reset()
    await graph_store.close()


async def _observe_an_echo(manager: GraphManager) -> None:
    """Drive the real capture-surface echo so the only interaction is a capture one."""
    get_usage_buffer().note_surfaced(
        _GROUP,
        entity_id=_ENTITY_ID,
        name=_ENTITY_NAME,
        snippet=f"{_ENTITY_NAME} {_SURFACED_SNIPPET}",
        ts=time.time(),
    )
    await store_observation(
        manager,
        content=_ECHOING_CAPTURE,
        group_id=_GROUP,
        source="mcp_observe",
    )


@pytest.mark.asyncio
async def test_capture_only_window_cannot_measure_dismissal(usage_manager):
    """THE ASSERTION NOBODY WROTE: on the agent surface, 0 dismissals is not a
    measurement of zero dismissals."""
    await _observe_an_echo(usage_manager)

    metrics = usage_manager.get_recall_metrics(_GROUP)

    assert metrics["dismissed_count"] == 0
    assert "dismissed" in metrics["unmeasurable_interactions"], (
        "nothing on the capture/recall surfaces emits `dismissed`, so the 0 above "
        f"must be declared unmeasurable; got {metrics['unmeasurable_interactions']}"
    )
    assert "selected" in metrics["unmeasurable_interactions"]
    assert metrics["false_recall_rate"] is None, (
        "false_recall_rate is derived from a verb no observed surface can emit; "
        f"it must be absent, not {metrics['false_recall_rate']!r}"
    )


@pytest.mark.asyncio
async def test_capture_window_measures_used_and_says_which_surface(usage_manager):
    """The other half of the same window: `used` IS reachable here now, and the
    payload names the surface that reached it."""
    await _observe_an_echo(usage_manager)

    metrics = usage_manager.get_recall_metrics(_GROUP)

    assert metrics["used_count"] == 1
    assert "used" not in metrics["unmeasurable_interactions"]
    assert AGENT_CAPTURE in metrics["interaction_surfaces_observed"]


def test_dashboard_window_reports_a_real_zero():
    """Control: where the verb HAS an emitter in the tree, 0 keeps meaning zero.
    (Since 2026-09-04 the chat route answers 501, so this window cannot occur
    live; the declaration counts call sites, not reachability.)"""
    controller = RecallNeedController(ActivationConfig())
    controller.record_interaction(_GROUP, "used", source=_DASHBOARD_SOURCE)

    metrics = controller.snapshot(_GROUP)

    assert DASHBOARD_CHAT in metrics["interaction_surfaces_observed"]
    assert "dismissed" not in metrics["unmeasurable_interactions"]
    assert metrics["false_recall_rate"] == 0.0
    assert metrics["dismissed_count"] == 0


def test_a_verb_that_fired_is_never_declared_dead():
    """The self-correcting guard: an emitter the table does not know about still
    wins. A stale table may under-report deadness; it may never bury a signal."""
    controller = RecallNeedController(ActivationConfig())
    controller.record_interaction(_GROUP, "dismissed", source="some_future_surface")

    metrics = controller.snapshot(_GROUP)

    assert metrics["dismissed_count"] == 1
    assert "dismissed" not in metrics["unmeasurable_interactions"]
    assert metrics["false_recall_rate"] is not None


def test_mcp_recall_is_not_classified_as_the_dashboard():
    """The classifier is what makes the window-scoped claim true at all."""
    assert surface_for_source(_MCP_RECALL_SOURCE) != DASHBOARD_CHAT
    assert surface_for_source(_DASHBOARD_SOURCE) == DASHBOARD_CHAT
    assert surface_for_source("observed_echo") == AGENT_CAPTURE


def _production_sources() -> list[tuple[Path, str]]:
    root = Path(__file__).resolve().parents[1] / "engram"
    return [(path, path.read_text(encoding="utf-8")) for path in root.rglob("*.py")]


def _emitter_sites(verb: str) -> list[str]:
    pattern = re.compile(rf'interaction_type\s*=\s*"{verb}"')
    return [str(path) for path, text in _production_sources() if pattern.search(text)]


@pytest.mark.parametrize(
    "verb", sorted(v for v, emitters in INTERACTION_EMITTERS.items() if not emitters)
)
def test_verbs_declared_emitterless_have_no_emitter_in_the_tree(verb):
    """`confirmed`/`corrected` are accepted by the applier and produced by
    nothing. If that changes, this declaration must change with it."""
    sites = _emitter_sites(verb)
    assert not sites, (
        f"`{verb}` is declared to have no emitter, but one exists in {sites}. "
        "Update INTERACTION_EMITTERS — a stale declaration is the same class of "
        "lie as the count it exists to explain."
    )


@pytest.mark.parametrize(
    "verb", sorted(v for v, emitters in INTERACTION_EMITTERS.items() if emitters)
)
def test_verbs_declared_emittable_still_have_an_emitter(verb):
    """The inverse ratchet: a declaration that outlives its emitter would make a
    dead verb look measurable, which is worse than the zero it replaced."""
    assert _emitter_sites(verb), (
        f"`{verb}` is declared emittable by {sorted(INTERACTION_EMITTERS[verb])} "
        "but no call site sets it. Either it moved or it died; say which."
    )


@pytest.mark.asyncio
async def test_lifecycle_cue_block_declares_who_can_emit_selected():
    """The ticket's headline evidence: live `selectedCount: 0` next to
    `surfacedCount: 1208`. The operator instrument now says which surface could
    have moved that 0."""
    from engram.lifecycle_summary import build_lifecycle_summary

    class _StubManager:
        async def get_graph_state(self, **_kwargs):
            return {
                "stats": {
                    "cue_metrics": {
                        "cue_count": 4,
                        "cue_surfaced_count": 1208,
                        "cue_selected_count": 0,
                    }
                },
                "top_activated": [],
            }

    class _StubEngine:
        is_running = False

        async def get_recent_cycles(self, _group_id, limit=10):
            return []

    summary = await build_lifecycle_summary(
        group_id=_GROUP,
        manager=_StubManager(),
        consolidation_engine=_StubEngine(),
    )

    cue = summary["cue"]
    assert cue["selectedCount"] == 0
    assert cue["selectedCountEmitters"] == [DASHBOARD_CHAT], (
        "a 0 next to 1208 surfacings must come with the one surface that could "
        f"have moved it; got {cue.get('selectedCountEmitters')!r}"
    )


def test_brain_loop_report_does_not_fabricate_a_zero():
    """The gate's own reader. `_float()` turns None into 0.0, so the honesty had
    to be carried one level up or it would be undone in the report."""
    summary = _recall_summary(
        {
            "total_analyses": 5,
            "trigger_count": 3,
            "false_recall_rate": None,
            "unmeasurable_interactions": ["confirmed", "corrected", "dismissed", "selected"],
        },
        None,
        None,
    )

    assert summary["runtime_false_recall_rate"] is None
    assert "dismissed" in summary["runtime_unmeasurable_interactions"]
