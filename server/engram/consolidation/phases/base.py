"""Abstract base class for consolidation phases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from engram.config import ActivationConfig
from engram.models.consolidation import CycleContext, PhaseResult


class ConsolidationPhase(ABC):
    """Base class for a single consolidation phase."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable phase name."""
        ...

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        """Optional graph-store capabilities this phase requires."""
        return set()

    def required_activation_store_methods(self, cfg: ActivationConfig) -> set[str]:
        """Optional activation-store capabilities this phase requires."""
        return set()

    def required_search_index_methods(self, cfg: ActivationConfig) -> set[str]:
        """Optional search-index capabilities this phase requires."""
        return set()

    @abstractmethod
    async def execute(
        self,
        group_id: str,
        graph_store,
        activation_store,
        search_index,
        cfg: ActivationConfig,
        cycle_id: str,
        dry_run: bool = False,
        context: CycleContext | None = None,
    ) -> tuple[PhaseResult, list[Any]]:
        """Run this phase. Returns (result, audit_records)."""
        ...
