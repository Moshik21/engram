"""Depth-tier evaluation: strict fact-presence judge + paired stats."""

from engram.benchmark.depth.judge import (
    JudgeVerdict,
    judge_current_value,
    judge_multi_hop,
    judge_synthesis,
    mcnemar_p,
    pass_rate_bootstrap_ci,
)

__all__ = [
    "JudgeVerdict",
    "judge_current_value",
    "judge_multi_hop",
    "judge_synthesis",
    "mcnemar_p",
    "pass_rate_bootstrap_ci",
]
