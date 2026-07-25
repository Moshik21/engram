"""Ticket #7: a surfaced cue must record a use ACROSS A PROCESS BOUNDARY.

The producer of ``EpisodeCue.usage_used_count`` — the only field the RF flip
gate can read — requires a cue to be surfaced and then echoed back. Before this
module's fix the surfaced half lived only in a process-local ring buffer
(``SurfacedUsageBuffer``), so the loop could fire only when both halves landed
in one shell process lifetime. An MCP/axi agent session spans shell restarts,
so the counter was not merely unfired: it was architecturally incapable of
firing for the actual consumer.

The process boundary is simulated the only way that is honest here: the
"surfacing" process gets its own ``SurfacedUsageBuffer`` instance, that instance
is then DISCARDED, and a brand-new instance stands in for the restarted shell.
Nothing carries over except the durable sidecar on disk. Every test in
``TestCrossProcessUse`` fails against the pre-fix code, because a fresh buffer
had no way to learn what the dead one surfaced.

The conservative half is tested just as hard: ``TestPhantomUseGuards`` asserts
that the SAME boundary crossing does NOT manufacture a use for a verbatim
parrot, for a payload that has aged out, or for a cue that already fired.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.ingestion.capture_surface import store_observation
from engram.models.activation import ActivationState
from engram.models.episode_cue import EpisodeCue
from engram.retrieval.feedback import (
    SurfacedUsageBuffer,
    record_observed_usage_events,
)
from engram.retrieval.usage_surface_store import (
    SurfacedUsageStore,
    resolve_usage_surface_path,
)

# Seeded from the shape of a live cue row: `cue_text` is the deterministic
# "mentions: ... | spans: ..." rendering the cue builder emits, and the span is
# a lowercase declarative clause lifted from episode content.
CUE_TEXT = "mentions: Helix | spans: the native helix migration finished on tuesday afternoon"
CUE_SPAN = "the native helix migration finished on tuesday afternoon"
EPISODE_CONTENT = (
    "We finally cut over: the native helix migration finished on tuesday "
    "afternoon and HelixDB is now the only engine behind the vector lane."
)
ENTITY_ID = "ent_helixdb"
ENTITY_NAME = "HelixDB"
ENTITY_SNIPPET = "HelixDB. Technology. Native pyo3 graph and vector engine."

# The agent's own next turn. It reuses a phrase from the cue ("helix migration")
# in wording that is not in the surfaced payload — a citation, not an echo.
AGENT_REUSE = "Given the helix migration landed, drop the docker compose lane from the runbook."

# Must be near real wall clock: `is_empty()` is the capture fast path's
# short-circuit and takes no timestamp, so its TTL floor is `time.time()`.
# Everything downstream is relative to NOW, so the tests stay deterministic.
NOW = time.time()


def _cfg(**overrides: Any) -> ActivationConfig:
    return ActivationConfig(recall_usage_feedback_enabled=True, **overrides)


def _store(tmp_path) -> SurfacedUsageStore:
    return SurfacedUsageStore(tmp_path / "surfaced-usage.sqlite3")


def _surfacing_process(store: SurfacedUsageStore, *, ts: float = NOW) -> None:
    """Process A: a recall surfaces one cue-backed episode, then dies.

    Mirrors what production actually registers for one cue-backed result:
    ``RecallEntityAccessRecorder.record_entity_access`` notes the entity,
    ``RecallCueFeedbackRecorder._register_surfaced_cue`` notes the cue — the one
    place every surfaced cue passes through — and the episode content that
    travels with the result becomes a mask-only source.
    """
    buffer = SurfacedUsageBuffer(store=store)
    buffer.note_surfaced(
        "g-live",
        entity_id=ENTITY_ID,
        name=ENTITY_NAME,
        snippet=ENTITY_SNIPPET,
        ts=ts,
    )
    buffer.note_surfaced_cue(
        "g-live",
        episode_id="ep_helix",
        cue_text=CUE_TEXT,
        supporting_spans=[CUE_SPAN],
        ts=ts,
    )
    buffer.note_surfaced_text("g-live", EPISODE_CONTENT, ts)
    buffer.persist("g-live", ts)
    # Process A is gone. `buffer` is deliberately not returned.


def _restarted_process(store: SurfacedUsageStore) -> SurfacedUsageBuffer:
    """Process B: a fresh shell that never surfaced anything itself."""
    return SurfacedUsageBuffer(store=store)


class _FakeGraph:
    """Records cue reads/writes the way the capture path drives them."""

    def __init__(self) -> None:
        self.cue = EpisodeCue(episode_id="ep_helix", group_id="g-live", cue_text=CUE_TEXT)
        self.updates: list[dict] = []

    async def get_episode_cue(self, episode_id: str, group_id: str) -> EpisodeCue | None:
        return self.cue if episode_id == "ep_helix" else None

    async def update_episode_cue(self, episode_id: str, updates: dict, *, group_id: str) -> None:
        self.updates.append(dict(updates))
        if "usage_used_count" in updates:
            self.cue.usage_used_count = float(updates["usage_used_count"])
        if "usage_last_used_at" in updates:
            self.cue.usage_last_used_at = updates["usage_last_used_at"]


class SpyActivationStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, str | None, str]] = []

    async def record_access(
        self,
        entity_id: str,
        timestamp: float,
        group_id: str | None = None,
        tier: str = "surfaced",
    ) -> None:
        self.calls.append((entity_id, timestamp, group_id, tier))

    async def get_activation(self, entity_id: str) -> ActivationState | None:
        return None

    async def set_activation(self, entity_id: str, state: ActivationState) -> None:
        return None


class TestCrossProcessUse:
    """The probe. Every one of these is RED without the durable registry."""

    def test_fresh_process_sees_nothing_without_the_sidecar(self, tmp_path):
        """Control: this IS the pre-fix behaviour, kept visible on purpose."""
        _surfacing_process(_store(tmp_path))
        unbacked = SurfacedUsageBuffer()  # no store == the old ring buffer
        assert unbacked.is_empty("g-live")
        assert unbacked.scan_novel_cue_matches("g-live", AGENT_REUSE, now=NOW + 60.0) == []

    def test_restarted_process_still_knows_the_cue_was_surfaced(self, tmp_path):
        store = _store(tmp_path)
        _surfacing_process(store)
        buffer = _restarted_process(store)
        assert not buffer.is_empty("g-live")

    def test_restarted_process_fires_the_cue_scan(self, tmp_path):
        store = _store(tmp_path)
        _surfacing_process(store)
        buffer = _restarted_process(store)
        fired = buffer.scan_novel_cue_matches("g-live", AGENT_REUSE, now=NOW + 60.0)
        assert [entry.episode_id for entry in fired] == ["ep_helix"]

    @pytest.mark.asyncio
    async def test_counter_moves_across_the_boundary(self, tmp_path):
        """surface (process A) -> restart -> observe (process B) -> counter up."""
        store = _store(tmp_path)
        _surfacing_process(store)

        cfg = _cfg()
        graph = _FakeGraph()
        assert graph.cue.usage_used_count == 0.0

        fired = await record_observed_usage_events(
            activation_store=AsyncMock(),
            cfg=cfg,
            group_id="g-live",
            content=AGENT_REUSE,
            now=NOW + 60.0,
            usage_buffer=_restarted_process(store),
            graph_store=graph,
        )

        assert fired == ["cue::ep_helix"]
        assert graph.cue.usage_used_count == pytest.approx(cfg.usage_tier_weights["used"])
        assert graph.cue.usage_used_count > 0.0
        assert graph.cue.usage_last_used_at == datetime.fromtimestamp(NOW + 60.0, tz=timezone.utc)

    @pytest.mark.asyncio
    async def test_capture_fast_path_short_circuit_does_not_block_the_boundary(self, tmp_path):
        """`is_empty` is the capture path's short-circuit AND the hydrate hook.

        Without hydration inside `is_empty`, `_record_observed_usage_events`
        returns before the scan ever runs and the loop stays dead even with a
        populated sidecar.
        """
        store = _store(tmp_path)
        _surfacing_process(store)
        buffer = _restarted_process(store)
        assert buffer.is_empty("g-other-group")  # unrelated group unaffected
        assert not buffer.is_empty("g-live")

    def test_entity_lane_also_crosses_the_boundary(self, tmp_path):
        store = _store(tmp_path)
        _surfacing_process(store)

        fired = _restarted_process(store).scan_novel_mentions(
            "g-live",
            "Point the smoke gate at HelixDB so the vector engine is what we measure",
            now=NOW + 60.0,
        )
        assert [entry.entity_id for entry in fired] == [ENTITY_ID]


class TestPhantomUseGuards:
    """A phantom use is worse than a missed one — it would unblock a flip on
    fabricated evidence. These assert the fix did not buy durability with
    false positives."""

    def test_verbatim_parrot_across_the_boundary_does_not_fire(self, tmp_path):
        """THE reason the echo mask has to be persisted with the payload.

        The agent repeats the delivered episode content word for word. That is
        possession of the payload, not reliance on the memory. If the sidecar
        carried the cue and the entity but not the episode-content mask, the
        entity mention inside the parrot reads as a novel mention — the cue's
        own phrases self-mask, the entity's short snippet does not — and the
        gate is fed a manufactured number.
        """
        store = _store(tmp_path)
        _surfacing_process(store)
        buffer = _restarted_process(store)
        parrot = f"Sure — {EPISODE_CONTENT}"
        assert buffer.scan_novel_mentions("g-live", parrot, now=NOW + 60.0) == []
        assert buffer.scan_novel_cue_matches("g-live", parrot, now=NOW + 60.0) == []
        assert buffer.scan_novel_cue_matches("g-live", CUE_SPAN, now=NOW + 60.0) == []

    def test_mask_is_actually_present_after_hydration(self, tmp_path):
        """Direct read of the mechanism the previous test depends on."""
        store = _store(tmp_path)
        _surfacing_process(store)
        buffer = _restarted_process(store)
        buffer.hydrate("g-live", NOW + 60.0)
        sources = buffer._mask_sources("g-live")
        joined = {" ".join(tokens) for tokens in sources}
        assert any("only engine behind the vector lane" in text for text in joined), (
            "the surfaced episode content did not survive as an echo-mask source"
        )

    def test_aged_out_rows_are_not_loaded_from_the_sidecar(self, tmp_path):
        store = _store(tmp_path)
        _surfacing_process(store)
        buffer = _restarted_process(store)
        stale = NOW + 3600.0  # one hour later, TTL is 30 minutes
        assert buffer.scan_novel_cue_matches("g-live", AGENT_REUSE, now=stale) == []

    def test_aged_out_surfacing_expires_inside_one_process_too(self, tmp_path):
        """The load-time floor is not the whole guard.

        A shell that stays up for hours never re-loads, so without an in-ring
        eligibility check a cue surfaced this morning would still be armed
        tonight — which is the pre-fix semantics and the one place durability
        could quietly widen the false-positive window rather than narrow it.
        No restart in this test: one buffer, one lifetime.
        """
        buffer = SurfacedUsageBuffer(store=_store(tmp_path))
        buffer.note_surfaced_cue(
            "g-live",
            episode_id="ep_helix",
            cue_text=CUE_TEXT,
            supporting_spans=[CUE_SPAN],
            ts=NOW,
        )
        assert buffer.scan_novel_cue_matches("g-live", AGENT_REUSE, now=NOW + 60.0)
        buffer.note_surfaced(
            "g-live", entity_id=ENTITY_ID, name=ENTITY_NAME, snippet=ENTITY_SNIPPET, ts=NOW
        )
        stale = NOW + 3600.0
        assert buffer.scan_novel_cue_matches("g-live", AGENT_REUSE, now=stale) == []
        assert (
            buffer.scan_novel_mentions(
                "g-live",
                "Point the smoke gate at HelixDB so the vector engine is what we measure",
                now=stale,
            )
            == []
        )

    def test_dedup_survives_the_boundary(self, tmp_path):
        """A restart must not re-arm a cue that already fired."""
        store = _store(tmp_path)
        _surfacing_process(store)

        first = _restarted_process(store)
        assert first.scan_novel_cue_matches("g-live", AGENT_REUSE, now=NOW + 60.0)
        first.persist("g-live", NOW + 60.0)

        second = _restarted_process(store)  # another restart
        assert second.scan_novel_cue_matches("g-live", AGENT_REUSE, now=NOW + 120.0) == []

    def test_unrelated_content_does_not_fire_across_the_boundary(self, tmp_path):
        store = _store(tmp_path)
        _surfacing_process(store)
        buffer = _restarted_process(store)
        fired = buffer.scan_novel_cue_matches(
            "g-live",
            "Bump the ruff config and regenerate the pnpm lockfile",
            now=NOW + 60.0,
        )
        assert fired == []

    def test_a_different_group_never_sees_another_groups_surfacing(self, tmp_path):
        store = _store(tmp_path)
        _surfacing_process(store)
        buffer = _restarted_process(store)
        assert buffer.scan_novel_cue_matches("g-other", AGENT_REUSE, now=NOW + 60.0) == []


class TestBindingAndInertness:
    def test_unresolvable_config_leaves_behaviour_process_local(self):
        """A bare ActivationConfig must not start writing to the filesystem."""
        assert resolve_usage_surface_path(ActivationConfig()) is None

    def test_path_lands_next_to_the_other_recall_sidecars(self, tmp_path):
        cfg = ActivationConfig(recall_packet_cache_path=str(tmp_path / "packet-cache.sqlite3"))
        resolved = resolve_usage_surface_path(cfg)
        assert resolved is not None
        assert resolved.parent == tmp_path
        assert resolved.name == "surfaced-usage.sqlite3"

    def test_outbox_path_is_the_fallback_directory(self, tmp_path):
        cfg = ActivationConfig(cue_index_outbox_path=str(tmp_path / "cue-index-outbox.sqlite3"))
        resolved = resolve_usage_surface_path(cfg)
        assert resolved is not None
        assert resolved.parent == tmp_path

    def test_a_non_path_config_value_is_refused_not_stringified(self, tmp_path, monkeypatch):
        """Regression: a Mock cfg wrote `server/MagicMock/.../surfaced-usage.sqlite3`.

        Logged against this module by another lane. It is not Mock-specific —
        any cfg whose path field is not a real absolute path used to yield a
        garbage RELATIVE directory under the CWD, which the store then created.
        Refusing is the right answer: an unbound registry is the pre-fix
        behaviour, a sidecar in the repo root is a new mess.
        """
        from unittest.mock import MagicMock

        monkeypatch.chdir(tmp_path)
        assert resolve_usage_surface_path(MagicMock()) is None
        assert resolve_usage_surface_path(SimpleNamespace(recall_packet_cache_path=123)) is None
        assert (
            resolve_usage_surface_path(
                SimpleNamespace(
                    recall_packet_cache_path="relative/packet-cache.sqlite3",
                    cue_index_outbox_path="",
                )
            )
            is None
        )
        assert list(tmp_path.iterdir()) == [], "resolution must not create anything"

    @pytest.mark.asyncio
    async def test_capture_path_arms_the_registry_before_short_circuiting(self, tmp_path):
        """End-to-end through `store_observation`, the real observe entry point.

        Nothing is bound by hand: the process-wide buffer starts empty and
        unbound, exactly as a freshly restarted shell's does, and the ONLY thing
        that can arm it is the config that reaches the observe path. If the
        capture path stops self-arming, this goes red — which is the difference
        between a durable registry and a durable registry nobody opens.
        """
        from engram.retrieval import feedback as feedback_module

        # The surfacing process wrote to the path the config resolves to.
        _surfacing_process(_store(tmp_path), ts=NOW)

        feedback_module.get_usage_buffer().reset()
        assert feedback_module.usage_surface_store_path() is None
        try:
            graph = _FakeGraph()

            class _Manager:
                def __init__(self) -> None:
                    self._cfg = _cfg(
                        recall_packet_cache_path=str(tmp_path / "packet-cache.sqlite3"),
                    )
                    self._activation = SpyActivationStore()
                    self._graph = graph

                async def store_episode(self, **kwargs: Any) -> str:
                    return "ep_new"

            manager = _Manager()
            episode_id = await store_observation(
                manager,
                content=AGENT_REUSE,
                group_id="g-live",
            )
            assert episode_id == "ep_new"
            assert feedback_module.usage_surface_store_path() == tmp_path / (
                "surfaced-usage.sqlite3"
            )
            assert graph.cue.usage_used_count > 0.0
        finally:
            feedback_module.get_usage_buffer().reset()

    @pytest.mark.asyncio
    async def test_remember_path_records_a_use_too(self, tmp_path):
        """`remember` is where an agent puts the turn it just relied on.

        The citation scan ran on `observe` only, so the highest-signal capture
        in a session was the one that could never record a use.
        """
        from engram.ingestion.capture_surface import ingest_projecting_memory
        from engram.retrieval import feedback as feedback_module

        _surfacing_process(_store(tmp_path), ts=NOW)
        feedback_module.get_usage_buffer().reset()
        try:
            graph = _FakeGraph()

            class _Manager:
                def __init__(self) -> None:
                    self._cfg = _cfg(
                        recall_packet_cache_path=str(tmp_path / "packet-cache.sqlite3"),
                    )
                    self._activation = SpyActivationStore()
                    self._graph = graph

                async def ingest_episode(self, **kwargs: Any) -> str:
                    return "ep_remembered"

            episode_id = await ingest_projecting_memory(
                _Manager(),
                content=AGENT_REUSE,
                group_id="g-live",
            )
            assert episode_id == "ep_remembered"
            assert graph.cue.usage_used_count > 0.0
        finally:
            feedback_module.get_usage_buffer().reset()

    @pytest.mark.asyncio
    async def test_recall_surface_arms_the_registry_too(self, tmp_path):
        """The producing side must self-arm as well, or nothing is ever written.

        Drives the response-time mask hook the recall surfaces call, with an
        unbound process-wide buffer, and asserts a sidecar appeared on disk.
        """
        from engram.retrieval import feedback as feedback_module

        feedback_module.get_usage_buffer().reset()
        try:
            cfg = _cfg(recall_packet_cache_path=str(tmp_path / "packet-cache.sqlite3"))
            feedback_module.note_surfaced_texts_from_response(
                "g-live",
                {
                    "results": [
                        {
                            "result_type": "cue_episode",
                            "episode_id": "ep_helix",
                            "cue_text": CUE_TEXT,
                            "supporting_spans": [CUE_SPAN],
                            "text": EPISODE_CONTENT,
                        }
                    ],
                    "packets": [],
                },
                cfg,
                now=NOW,
            )
            assert (tmp_path / "surfaced-usage.sqlite3").exists()
            # And a different process can read what it wrote.
            reader = _restarted_process(SurfacedUsageStore(tmp_path / "surfaced-usage.sqlite3"))
            fired = reader.scan_novel_cue_matches("g-live", AGENT_REUSE, now=NOW + 60.0)
            assert [entry.episode_id for entry in fired] == ["ep_helix"]
        finally:
            feedback_module.get_usage_buffer().reset()

    def test_sidecar_failure_degrades_to_process_local(self, tmp_path):
        """A broken sidecar must never take recall or capture down."""
        bad = tmp_path / "nested"
        bad.write_text("not a directory")
        store = SurfacedUsageStore(bad / "surfaced-usage.sqlite3")
        assert store.failed is True
        buffer = SurfacedUsageBuffer(store=store)
        buffer.note_surfaced_cue(
            "g-live", episode_id="ep_helix", cue_text=CUE_TEXT, supporting_spans=[CUE_SPAN], ts=NOW
        )
        assert buffer.persist("g-live", NOW) is False
        # Still works in-process.
        assert buffer.scan_novel_cue_matches("g-live", AGENT_REUSE, now=NOW + 60.0)
