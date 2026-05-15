"""Runtime capability validation for consolidation phases."""

from __future__ import annotations

from collections.abc import Sequence

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase


class ConsolidationCapabilityValidator:
    """Validate selected consolidation phases against runtime store capabilities."""

    def __init__(
        self,
        *,
        graph_store: object,
        activation_store: object,
        search_index: object,
    ) -> None:
        self._targets = (
            ("graph_store", graph_store),
            ("activation_store", activation_store),
            ("search_index", search_index),
        )

    def validate(
        self,
        phases: Sequence[ConsolidationPhase],
        *,
        cfg: ActivationConfig,
    ) -> None:
        """Raise a clear startup error when a selected phase cannot run."""
        for phase in phases:
            self._validate_phase(phase, cfg=cfg)

    def _validate_phase(self, phase: ConsolidationPhase, *, cfg: ActivationConfig) -> None:
        required_by_target = {
            "graph_store": phase.required_graph_store_methods(cfg),
            "activation_store": phase.required_activation_store_methods(cfg),
            "search_index": phase.required_search_index_methods(cfg),
        }
        for target_name, target in self._targets:
            self._validate_capability_group(
                phase.name,
                target_name,
                target,
                required_by_target[target_name],
            )

    @staticmethod
    def _validate_capability_group(
        phase_name: str,
        target_name: str,
        target: object,
        required_methods: set[str],
    ) -> None:
        if not required_methods:
            return
        missing = sorted(method for method in required_methods if not hasattr(target, method))
        if missing:
            raise RuntimeError(
                f"Phase '{phase_name}' requires {target_name} methods: {', '.join(missing)}"
            )
