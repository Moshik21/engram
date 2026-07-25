"""Suppress secondary graph reads after recall preflight probe timeouts.

The suppression itself is deliberate (a probe timeout means the store is
already over budget, so secondary reads would only deepen the overrun). What
was NOT deliberate is that the suppression used to be **indistinguishable from
a miss**: every gated method returned ``None``/``[]`` in ~0.11 ms, so a bounded
caller asking "is there an episode with this id?" was told "no" when the honest
answer was "I stopped trying".

That is a wrong answer, not a slow one, and it is the common path —
``recall_stats_timeout`` fires routinely. Measured consequence: reranker
document coverage of 0.2-0.55 when episodes are read through the gate versus
1.0 read directly, which is one of the reasons the rerank stage has been
measured as useless three separate times.

Refusals therefore raise :class:`GraphGateTimeoutError`. This mirrors the storage
layer's ``NativeQueryError`` contract one layer up: a caller that can
legitimately proceed without the read must catch THAT type explicitly and
record the degradation, so the give-up stays visible instead of being metered
as an empty success.
"""

from __future__ import annotations

from typing import Any

from engram.config import ActivationConfig

GATED_GRAPH_METHODS: tuple[str, ...] = (
    "get_entity",
    "find_entities",
    "find_entity_candidates",
    "get_relationships",
    "get_episode_by_id",
    "get_active_neighbors_with_weights",
    "get_identity_core_entities",
)

# Preflight probes whose timeout arms the gate. Both are written before the
# gate is constructed (pipeline.py Step 0/0.1), so the gate's verdict is
# CONSTANT for the remainder of one recall — a caller that sees one refusal
# will see a refusal for every subsequent gated read in that request.
PROBE_TIMEOUT_KEYS: tuple[str, ...] = ("recall_stats_timeout", "graph_expand_timeout")

# Count of refusals served in one recall. A refusal is a give-up, so it must be
# countable next to the stage metrics rather than inferred from a suspiciously
# fast empty result.
GATE_REFUSAL_METRIC_KEY = "recall_graph_gate_refusals"


class GraphGateTimeoutError(RuntimeError):
    """A gated graph read was REFUSED because a recall probe already timed out.

    NOT a miss. ``None`` from a graph store means "no such row"; this means
    "the read was never attempted". Callers must be able to branch on the
    difference — see the module docstring for the measured cost of conflating
    them.

    Same shape as ``storage.helix.native_transport.NativeQueryError``: failures
    raise, and a caller that can tolerate one catches this type explicitly and
    marks the degradation.
    """

    def __init__(self, method: str, probe: str) -> None:
        self.method = method
        self.probe = probe
        super().__init__(
            f"graph read {method!r} refused: recall probe {probe!r} already timed out "
            "(this is a give-up, not a miss)"
        )


def graph_probe_timeout_reason(stage_timings_ms: dict[str, float] | None) -> str | None:
    """Which preflight probe armed the gate, or ``None`` when it is open."""
    if not stage_timings_ms:
        return None
    for key in PROBE_TIMEOUT_KEYS:
        if key in stage_timings_ms:
            return key
    return None


def graph_probe_timed_out(stage_timings_ms: dict[str, float] | None) -> bool:
    return graph_probe_timeout_reason(stage_timings_ms) is not None


def skip_secondary_graph_after_probe_timeout(
    cfg: ActivationConfig,
    stage_timings_ms: dict[str, float] | None,
) -> bool:
    return bool(
        cfg.retrieval_skip_secondary_graph_after_probe_timeout
        and graph_probe_timed_out(stage_timings_ms)
    )


class GatedGraphStore:
    """Proxy that refuses secondary graph reads after a probe timeout.

    Refusal raises :class:`GraphGateTimeoutError`; it never fabricates a miss.
    Methods outside :data:`GATED_GRAPH_METHODS` pass straight through.
    """

    def __init__(
        self,
        graph_store: Any,
        cfg: ActivationConfig,
        stage_timings_ms: dict[str, float] | None,
    ) -> None:
        self._graph_store = graph_store
        self._cfg = cfg
        self._stage_timings_ms = stage_timings_ms

    @property
    def underlying(self) -> Any:
        return self._graph_store

    def _blocked(self) -> bool:
        return skip_secondary_graph_after_probe_timeout(
            self._cfg,
            self._stage_timings_ms,
        )

    def _refuse(self, method: str) -> None:
        probe = graph_probe_timeout_reason(self._stage_timings_ms) or "unknown"
        if self._stage_timings_ms is not None:
            current = self._stage_timings_ms.get(GATE_REFUSAL_METRIC_KEY, 0.0)
            self._stage_timings_ms[GATE_REFUSAL_METRIC_KEY] = float(current) + 1.0
        raise GraphGateTimeoutError(method, probe)

    async def get_entity(self, *args: Any, **kwargs: Any) -> Any:
        if self._blocked():
            self._refuse("get_entity")
        return await self._graph_store.get_entity(*args, **kwargs)

    async def find_entities(self, *args: Any, **kwargs: Any) -> list[Any]:
        if self._blocked():
            self._refuse("find_entities")
        return await self._graph_store.find_entities(*args, **kwargs)

    async def find_entity_candidates(self, *args: Any, **kwargs: Any) -> list[Any]:
        if self._blocked():
            self._refuse("find_entity_candidates")
        return await self._graph_store.find_entity_candidates(*args, **kwargs)

    async def get_relationships(self, *args: Any, **kwargs: Any) -> list[Any]:
        if self._blocked():
            self._refuse("get_relationships")
        return await self._graph_store.get_relationships(*args, **kwargs)

    async def get_episode_by_id(self, *args: Any, **kwargs: Any) -> Any:
        if self._blocked():
            self._refuse("get_episode_by_id")
        return await self._graph_store.get_episode_by_id(*args, **kwargs)

    async def get_active_neighbors_with_weights(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        if self._blocked():
            self._refuse("get_active_neighbors_with_weights")
        return await self._graph_store.get_active_neighbors_with_weights(
            *args,
            **kwargs,
        )

    async def get_identity_core_entities(self, *args: Any, **kwargs: Any) -> list[Any]:
        if self._blocked():
            self._refuse("get_identity_core_entities")
        return await self._graph_store.get_identity_core_entities(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph_store, name)
