#!/usr/bin/env python3
"""CLI for the showcase benchmark suite."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.benchmark.showcase.runner import run_showcase_benchmark


def _parse_multi(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items or None


async def main(args: argparse.Namespace) -> None:
    result = await run_showcase_benchmark(
        mode=args.mode,
        track=args.track,
        seeds=args.seeds,
        output_dir=args.output_dir,
        scenario_ids=_parse_multi(args.scenario),
        baseline_names=_parse_multi(args.baseline),
        primary_baselines=_parse_multi(args.primary_baselines),
        appendix_baselines=_parse_multi(args.appendix_baselines),
        include_ablations=not args.no_ablations,
        engram_vector_provider=args.engram_vector_provider,
        answer_model=args.answer_model,
        answer_provider=args.answer_provider,
        strict_fairness=args.strict_fairness,
        emit_readme_snippet=args.emit_readme_snippet,
        locomo_dataset_path=args.locomo_dataset_path,
        website_export_path=args.website_export_path,
    )

    print("\n" + "=" * 60)
    print("Engram Showcase Benchmark")
    print("=" * 60)
    print(f"Track:           {result.track}")
    print(f"Mode:            {result.mode}")
    print(f"Seeds:           {', '.join(str(seed) for seed in result.seeds)}")
    print(f"Output dir:      {result.output_dir}")
    for label, path in sorted(result.artifact_paths.items()):
        print(f"{label.title():16} {path}")
    print("")

    for summary in result.baseline_summaries:
        availability = (
            "available" if summary.available else f"unavailable ({summary.availability_reason})"
        )
        print(
            f"- {summary.baseline_name}: "
            f"pass_rate={summary.scenario_pass_rate:.3f}, "
            f"false_recall={summary.false_recall_rate:.3f}, "
            f"latency_p50={summary.latency_p50_ms:.1f}ms, "
            f"{availability}"
        )

    for summary in result.answer_summaries:
        if summary.available:
            print(
                f"- {summary.baseline_name} [answer]: "
                f"pass_rate={summary.answer_pass_rate:.3f}, "
                f"score={summary.average_score:.3f}, "
                f"latency_p50={summary.latency_p50_ms:.1f}ms"
            )

    for summary in result.track_summaries:
        if summary.track == "showcase":
            continue
        status = (
            "available" if summary.available
            else f"unavailable ({summary.availability_reason})"
        )
        print(
            f"- track={summary.track}: executed={summary.executed}, "
            f"{status}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engram showcase benchmark")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "scale"],
        default="quick",
        help="Benchmark mode to run",
    )
    parser.add_argument(
        "--track",
        choices=["showcase", "answer", "external", "all"],
        default="showcase",
        help="Benchmark track to run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where benchmark artifacts should be written",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Run only the named scenario id(s); comma separated or repeatable",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=None,
        help="Run only the named baseline(s); comma separated or repeatable",
    )
    parser.add_argument(
        "--primary-baselines",
        action="append",
        default=None,
        help="Override the primary baseline set; comma separated or repeatable",
    )
    parser.add_argument(
        "--appendix-baselines",
        action="append",
        default=None,
        help="Override the appendix baseline set; comma separated or repeatable",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seed list to use",
    )
    parser.add_argument(
        "--no-ablations",
        action="store_true",
        help="Skip internal Engram ablation baselines",
    )
    parser.add_argument(
        "--engram-vector-provider",
        choices=["none", "local", "voyage", "auto"],
        default="none",
        help="Optional vector-backed Engram baseline provider",
    )
    parser.add_argument(
        "--answer-model",
        type=str,
        default=None,
        help="Shared answer model identifier for Track B; use deterministic for offline runs",
    )
    parser.add_argument(
        "--answer-provider",
        type=str,
        default=None,
        help="Shared answer provider for Track B; deterministic is currently supported",
    )
    parser.add_argument(
        "--strict-fairness",
        action="store_true",
        help="Fail the run if baseline contracts diverge from the fairness contract",
    )
    parser.add_argument(
        "--emit-readme-snippet",
        action="store_true",
        help="Write a compact README-facing summary block",
    )
    parser.add_argument(
        "--locomo-dataset-path",
        type=str,
        default=None,
        help="Optional LoCoMo dataset path for external-track status reporting",
    )
    parser.add_argument(
        "--website-export-path",
        type=str,
        default=None,
        help="Optional path for a website-friendly benchmark summary JSON export",
    )
    asyncio.run(main(parser.parse_args()))
