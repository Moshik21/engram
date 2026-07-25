"""Three-arm rig for the graph deciding experiment (GRAPH_THESIS.md §5).

Arms A (shipped default), B (A + entity->episode traversal from the candidate
pool, zero relationship triples in evidence), and C (the kill arm: no graph, one
second recall round seeded from A).

Build-only module. Running the experiment is a separate, gated decision.
"""

from engram.evaluation.graph_kill_rig.runner import FAULTS, RigOptions, purge_scratch, run_rig
from engram.evaluation.graph_kill_rig.thresholds import (
    PRE_REGISTRATION_SOURCE,
    ArmResult,
    Verdict,
    evaluate,
)

__all__ = [
    "FAULTS",
    "PRE_REGISTRATION_SOURCE",
    "ArmResult",
    "RigOptions",
    "Verdict",
    "evaluate",
    "purge_scratch",
    "run_rig",
]
