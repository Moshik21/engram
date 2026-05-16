"""Runtime dependencies used by the background episode worker."""

from __future__ import annotations

from dataclasses import dataclass

from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex


@dataclass(frozen=True)
class EpisodeWorkerRuntimeStores:
    """Stores the worker needs beside the manager projection facade."""

    graph: GraphStore
    activation: ActivationStore
    search: SearchIndex
