"""Depth-tier evaluation: strict fact-presence judge + paired stats."""

from engram.benchmark.depth.judge import (
    JudgeVerdict,
    aggregate_repeated_runs,
    delta_bootstrap_ci,
    judge_current_value,
    judge_multi_hop,
    judge_synthesis,
    mcnemar_p,
    pass_rate_bootstrap_ci,
    verdict_flip_count,
)

__all__ = [
    "JudgeVerdict",
    "aggregate_repeated_runs",
    "delta_bootstrap_ci",
    "judge_current_value",
    "judge_multi_hop",
    "judge_synthesis",
    "mcnemar_p",
    "pass_rate_bootstrap_ci",
    "verdict_flip_count",
]
