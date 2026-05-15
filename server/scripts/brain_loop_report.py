"""Build a local Engram brain-loop evaluation report.

Examples:
    uv run python scripts/brain_loop_report.py
    uv run python scripts/brain_loop_report.py --format json
    uv run python scripts/brain_loop_report.py --from-json stats-export.json
"""

from __future__ import annotations

import argparse
import asyncio

from engram.evaluation.cli import configure_evaluate_parser, run_evaluate_command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report Capture -> Cue -> Project -> Recall -> Consolidate health.",
    )
    configure_evaluate_parser(parser)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_evaluate_command(_parse_args()))
