"""Spreading activation strategy protocol and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from engram.config import ActivationConfig

if TYPE_CHECKING:
    from engram.activation.context_gate import ContextGate


class SpreadingStrategy(Protocol):
    """Protocol for spreading activation algorithms."""

    async def spread(
        self,
        seed_nodes: list[tuple[str, float]],
        neighbor_provider,
        cfg: ActivationConfig,
        group_id: str | None = None,
        community_store=None,
        context_gate: ContextGate | None = None,
        seed_entity_types: dict[str, str] | None = None,
        max_reads: int | None = None,
        deadline: float | None = None,
        traversal_stats: dict[str, float | str] | None = None,
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Spread activation from seed nodes.

        ``max_reads`` (nodes whose adjacency may be read) and ``deadline`` (an
        absolute ``time.monotonic()`` instant) are the CALLER's traversal bound.
        Both default to unbounded; every strategy must honour them so a bounded
        caller does not have to know which algorithm is configured.

        ``traversal_stats`` is a caller-owned dict every strategy must fill in
        with its ``ReadBudget`` provenance (reads, stop reason, unspent budget).
        Without it a collapsed traversal and a healthy one are indistinguishable
        from the outside: both return a small frontier and report success.

        Returns (bonuses, hop_distances):
          - bonuses: {node_id: spreading_bonus}
          - hop_distances: {node_id: min_hops_from_seed}
        """
        ...


def create_strategy(name: str) -> SpreadingStrategy:
    """Factory for spreading strategies."""
    if name == "bfs":
        from engram.activation.bfs import BFSStrategy

        return BFSStrategy()
    elif name == "ppr":
        from engram.activation.ppr import PPRStrategy

        return PPRStrategy()
    elif name == "actr":
        from engram.activation.actr import ACTRStrategy

        return ACTRStrategy()
    else:
        raise ValueError(f"Unknown spreading strategy: {name!r}")
