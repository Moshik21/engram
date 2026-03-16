#!/usr/bin/env python3
"""LongMemEval baseline: Claude + full context, no Engram.

Uses the real Claude Code CLI (`claude -p`) which bills to your
Max subscription — NOT API credits.

Usage:
    cd server
    uv run python scripts/benchmark_baseline.py run \
        --dataset data/longmemeval/longmemeval_oracle.json \
        --n-per-type 5 \
        --output results/longmemeval_baseline.json \
        --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Prevent nested session detection
os.environ.pop("CLAUDECODE", None)
os.environ.pop("CLAUDE_CODE_ENTRYPOINT", None)
# Ensure we use subscription, not API credits
os.environ.pop("ANTHROPIC_API_KEY", None)

logger = logging.getLogger(__name__)

BASELINE_SYSTEM_PROMPT = (
    "You are answering questions about a user's past conversations. "
    "You are given the full conversation history as context. "
    "Answer based ONLY on the provided conversations. "
    "If the conversations don't contain enough information, say exactly: "
    "\"I don't have enough information to answer this question.\" "
    "Be concise - answer in 1-2 sentences maximum."
)

CLAUDE_CLI = os.environ.get(
    "CLAUDE_CLI_PATH",
    str(Path.home() / ".local" / "bin" / "claude"),
)


def _query_claude_cli(prompt: str, model: str = "sonnet") -> str:
    """Call the real Claude Code CLI (uses Max subscription).

    Pipes prompt via stdin to handle 40K+ char prompts.
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    env.pop("ANTHROPIC_API_KEY", None)

    result = subprocess.run(
        [
            CLAUDE_CLI,
            "-p", "Answer the question in the provided context below.",
            "--model", model,
            "--output-format", "text",
            "--append-system-prompt", BASELINE_SYSTEM_PROMPT,
            "--no-session-persistence",
        ],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    if result.returncode != 0:
        logger.error("Claude CLI stderr: %s", result.stderr[:500])
        return "[ERROR]"

    return result.stdout.strip()


_SEMAPHORE: asyncio.Semaphore | None = None


async def _query_claude_async(prompt: str, model: str = "sonnet") -> str:
    """Rate-limited async wrapper around the CLI call."""
    assert _SEMAPHORE is not None
    async with _SEMAPHORE:
        return await asyncio.to_thread(_query_claude_cli, prompt, model)


async def cmd_run(args: argparse.Namespace) -> None:
    """Run baseline benchmark — Claude + full context, no Engram."""
    from engram.benchmark.longmemeval.dataset import load_dataset
    from engram.benchmark.longmemeval.evaluator import (
        compute_containment_score,
        judge_by_containment,
    )

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Load .env for GEMINI_API_KEY (embedding judge only — not Anthropic)
    from dotenv import load_dotenv
    load_dotenv(Path.home() / ".engram" / ".env", override=False)
    load_dotenv()
    # Re-clear ANTHROPIC_API_KEY in case .env set it
    os.environ.pop("ANTHROPIC_API_KEY", None)

    # Embedding provider for containment scoring
    embed_fn = None
    try:
        from engram.embeddings.provider import GeminiProvider
        provider = GeminiProvider()
        embed_fn = provider.embed
        logger.info("Embedding provider ready for containment scoring")
    except Exception:
        logger.warning("No embedding provider - using token overlap only")

    start = time.perf_counter()

    dataset = load_dataset(
        args.dataset, max_instances=args.max_instances, variant="auto"
    )
    if args.types:
        dataset = dataset.filter_types(args.types.split(","))
    if args.n_per_type:
        dataset = dataset.stratified_subset(args.n_per_type)
    elif args.max_instances and len(dataset.instances) > args.max_instances:
        dataset = dataset.subset(args.max_instances)

    logger.info(
        "Baseline benchmark (no Engram): %d instances, model=%s, cli=%s",
        len(dataset.instances),
        args.model,
        CLAUDE_CLI,
    )

    if not Path(CLAUDE_CLI).exists():
        logger.error("Claude CLI not found at %s", CLAUDE_CLI)
        sys.exit(1)

    global _SEMAPHORE
    _SEMAPHORE = asyncio.Semaphore(args.parallel)

    # Load checkpoint
    completed_ids: set[str] = set()
    results: list[dict] = []
    if args.checkpoint and Path(args.checkpoint).exists():
        with open(args.checkpoint) as f:
            cp = json.load(f)
        results = cp.get("instances", [])
        completed_ids = {r["question_id"] for r in results}

    for i, instance in enumerate(dataset.instances):
        if instance.question_id in completed_ids:
            continue

        logger.info(
            "[%d/%d] %s (%s)",
            i + 1,
            len(dataset.instances),
            instance.question_id,
            instance.question_type,
        )

        # Build full-context prompt with all session transcripts
        session_text = ""
        for session in instance.sessions:
            header = "\n--- Session"
            if session.date:
                header += f" ({session.date})"
            header += " ---\n"
            session_text += header + session.text + "\n"

        prompt = (
            f"Conversations:\n{session_text}\n"
            f"Question (asked on {instance.question_date}): "
            f"{instance.question}"
        )

        hypothesis = await _query_claude_async(prompt, args.model)

        # Compute embedding containment
        containment_score = 0.0
        if embed_fn and hypothesis not in ("[NO ANSWER]", "[ERROR]"):
            containment_score = await compute_containment_score(
                gold_answer=instance.answer,
                evidence_texts=[hypothesis],
                embed_fn=embed_fn,
            )

        verdict = judge_by_containment(
            question_id=instance.question_id,
            question_type=instance.question_type,
            containment_score=containment_score,
            is_abstention=instance.is_abstention,
            hypothesis=hypothesis,
            gold_answer=instance.answer,
        )

        result = {
            "question_id": instance.question_id,
            "question_type": instance.question_type,
            "question": instance.question,
            "gold_answer": instance.answer,
            "hypothesis": hypothesis[:500],
            "correct": verdict.correct,
            "judge_raw": verdict.judge_raw,
            "containment_score": round(containment_score, 4),
        }
        results.append(result)

        status = "CORRECT" if verdict.correct else "WRONG"
        correct_so_far = sum(1 for r in results if r["correct"])
        logger.info(
            "  -> %s (contain=%.4f) | Running: %d/%d = %.1f%%",
            status,
            containment_score,
            correct_so_far,
            len(results),
            100 * correct_so_far / len(results),
        )

        if args.checkpoint:
            _save_checkpoint(args.checkpoint, results)

    elapsed = time.perf_counter() - start

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total else 0.0

    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r)

    print(f"\n{'='*60}")
    print("LongMemEval BASELINE (No Engram)")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"CLI: {CLAUDE_CLI} (Max subscription)")
    print(f"Instances: {total}")
    print(f"Accuracy: {correct}/{total} = {accuracy*100:.1f}%")
    print(f"Elapsed: {elapsed:.0f}s")
    print()
    for t, items in sorted(by_type.items()):
        c = sum(1 for i in items if i["correct"])
        print(f"  {t}: {c}/{len(items)} ({100*c/len(items):.0f}%)")

    if args.output:
        output = {
            "variant": dataset.variant,
            "model": args.model,
            "auth": "max_subscription",
            "assessment_method": "baseline_no_engram",
            "total_instances": total,
            "total_correct": correct,
            "overall_accuracy": round(accuracy, 4),
            "elapsed_seconds": round(elapsed, 1),
            "instances": results,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def _save_checkpoint(path: str, results: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"instances": results}, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LongMemEval baseline (Claude CLI, Max subscription)",
    )
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="Run baseline")
    run.add_argument("--dataset", required=True)
    run.add_argument("--model", default="sonnet")
    run.add_argument("--max-instances", type=int, default=None)
    run.add_argument("--n-per-type", type=int, default=None)
    run.add_argument("--types", default=None)
    run.add_argument("--output", default=None)
    run.add_argument("--checkpoint", default=None)
    run.add_argument("--parallel", type=int, default=3, help="Concurrent queries (default: 3)")
    run.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    asyncio.run(cmd_run(args))
