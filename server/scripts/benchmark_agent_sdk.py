#!/usr/bin/env python3
"""LongMemEval benchmark via Claude Code CLI.

Runs the full benchmark as Engram would be deployed in production:
Claude Code connects to Engram's MCP server, uses recall to retrieve
evidence, and reasons over it to answer questions.

Uses `claude -p` which bills to your Max subscription — NOT API credits.

Usage:
    cd server

    # Quick calibration (5 per type)
    uv run python scripts/benchmark_agent_sdk.py run \
        --dataset data/longmemeval/longmemeval_oracle.json \
        --n-per-type 5

    # Full run
    uv run python scripts/benchmark_agent_sdk.py run \
        --dataset data/longmemeval/longmemeval_oracle.json \
        --output results/longmemeval_agent_sdk.json
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

logger = logging.getLogger(__name__)

READER_SYSTEM_PROMPT = (
    "You are answering questions about a user's past conversations. "
    "You have access to Engram's memory tools. "
    "For each question: "
    "1. Use the recall tool to retrieve relevant memories. "
    "2. Read the retrieved evidence carefully. "
    "3. Answer the question based ONLY on the evidence. "
    "4. If the evidence doesn't contain enough information, say "
    "\"I don't have enough information to answer this question.\" "
    "For temporal questions, pay attention to dates and chronological order. "
    "For \"how many\" questions, prefer the most recent evidence. "
    "Be concise - answer in 1-2 sentences maximum."
)

CLAUDE_CLI = os.environ.get(
    "CLAUDE_CLI_PATH",
    str(Path.home() / ".local" / "bin" / "claude"),
)


def _query_claude_cli(
    prompt: str,
    model: str = "claude-sonnet-4-6",
    mcp_config: dict | None = None,
) -> tuple[str, bool]:
    """Call Claude Code CLI with optional MCP server config.

    Returns (answer, used_recall).
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)

    cmd = [
        CLAUDE_CLI,
        "-p", prompt,
        "--model", model,
        "--output-format", "json",
    ]

    # Add MCP server config if provided
    if mcp_config:
        cmd.extend(["--mcp-config", json.dumps(mcp_config)])
        # Allow MCP tools
        cmd.extend(["--allowedTools", "mcp__engram__recall,mcp__engram__search_facts"])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )

    if result.returncode != 0:
        logger.error("Claude CLI error: %s", result.stderr[:500])
        return "[ERROR]", False

    # Parse JSON output for answer text and tool usage
    answer = ""
    used_recall = False
    try:
        output = json.loads(result.stdout)
        if isinstance(output, dict):
            answer = output.get("result", "")
            # Check messages for tool use
            for msg in output.get("messages", []):
                if msg.get("type") == "tool_use":
                    if "recall" in msg.get("name", ""):
                        used_recall = True
        elif isinstance(output, str):
            answer = output
    except json.JSONDecodeError:
        answer = result.stdout.strip()

    return answer.strip() or "[NO ANSWER]", used_recall


async def cmd_run(args: argparse.Namespace) -> None:
    """Run the benchmark using the real Claude Code CLI."""
    from engram.benchmark.longmemeval.adapter import EngramLongMemEvalAdapter
    from engram.benchmark.longmemeval.dataset import load_dataset
    from engram.benchmark.longmemeval.evaluator import (
        compute_containment_score,
        compute_retrieval_metrics,
        judge_by_containment,
    )
    from engram.config import ActivationConfig

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

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
        "Engram benchmark: %d instances, model=%s, cli=%s",
        len(dataset.instances),
        args.model,
        CLAUDE_CLI,
    )

    # Verify CLI exists
    if not Path(CLAUDE_CLI).exists():
        logger.error("Claude CLI not found at %s", CLAUDE_CLI)
        sys.exit(1)

    # Create Engram adapter for ingestion
    cfg = ActivationConfig()
    adapter = EngramLongMemEvalAdapter(
        cfg=cfg,
        extraction_mode=args.extraction,
        consolidation=False,
        top_k=10,
    )

    # MCP server config for Engram
    server_cwd = str(Path(__file__).resolve().parent.parent)

    # Load checkpoint
    completed_ids: set[str] = set()
    results: list[dict] = []
    if args.checkpoint and Path(args.checkpoint).exists():
        with open(args.checkpoint) as f:
            cp = json.load(f)
        results = cp.get("instances", [])
        completed_ids = {r["question_id"] for r in results}
        logger.info("Loaded %d checkpointed results", len(completed_ids))

    for i, instance in enumerate(dataset.instances):
        if instance.question_id in completed_ids:
            continue

        logger.info(
            "[%d/%d] %s (%s) -- %d sessions",
            i + 1,
            len(dataset.instances),
            instance.question_id,
            instance.question_type,
            instance.num_sessions,
        )

        # 1. Ingest sessions into Engram
        try:
            await adapter.ingest_instance(instance)
        except Exception:
            logger.error(
                "Ingestion failed for %s", instance.question_id, exc_info=True
            )
            results.append(_error_result(instance))
            if args.checkpoint:
                _save_checkpoint(args.checkpoint, results)
            continue

        # 2. Build MCP config with correct group_id
        group_id = adapter._current_group_id or "default"
        mcp_config = {
            "mcpServers": {
                "engram": {
                    "command": "uv",
                    "args": ["run", "engram", "mcp"],
                    "cwd": server_cwd,
                    "env": {
                        "ENGRAM_GROUP_ID": group_id,
                        "ENGRAM_MODE": "helix",
                        "ENGRAM_HELIX__TRANSPORT": "native",
                    },
                }
            }
        }

        prompt = (
            f"{READER_SYSTEM_PROMPT}\n\n"
            f"Question (asked on {instance.question_date}): "
            f"{instance.question}"
        )

        # 3. Call Claude Code CLI (uses subscription)
        hypothesis, used_recall = await asyncio.to_thread(
            _query_claude_cli, prompt, args.model, mcp_config
        )

        if used_recall:
            logger.info("  Tools used: recall")
        else:
            logger.info("  WARNING: recall not used")

        # 4. Evaluate with embedding containment
        embed_fn = adapter.get_embed_fn()
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
            threshold=args.containment_threshold,
            hypothesis=hypothesis,
            gold_answer=instance.answer,
        )

        retrieval_metrics = compute_retrieval_metrics(
            retrieved_session_ids=[],
            answer_session_ids=instance.answer_session_ids,
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
            "used_recall": used_recall,
            "retrieval_metrics": retrieval_metrics,
            "num_sessions": instance.num_sessions,
        }
        results.append(result)

        correct_so_far = sum(1 for r in results if r["correct"])
        logger.info(
            "  -> %s (contain=%.4f) | Running: %d/%d = %.1f%%",
            "CORRECT" if verdict.correct else "WRONG",
            containment_score,
            correct_so_far,
            len(results),
            100 * correct_so_far / len(results),
        )

        if args.checkpoint:
            _save_checkpoint(args.checkpoint, results)

    await adapter.close()
    elapsed = time.perf_counter() - start

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total else 0.0
    avg_contain = (
        sum(r.get("containment_score", 0) for r in results) / total if total else 0.0
    )
    used_count = sum(1 for r in results if r.get("used_recall"))

    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r)

    print(f"\n{'='*60}")
    print("LongMemEval Engram Benchmark")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"CLI: {CLAUDE_CLI} (Max subscription)")
    print(f"Instances: {total}")
    print(f"Accuracy: {correct}/{total} = {accuracy*100:.1f}%")
    print(f"Avg containment: {avg_contain:.4f}")
    print(f"Recall tool used: {used_count}/{total} ({100*used_count/total:.0f}%)")
    print(f"Elapsed: {elapsed:.0f}s")
    print()
    for t, items in sorted(by_type.items()):
        c = sum(1 for i in items if i["correct"])
        u = sum(1 for i in items if i.get("used_recall"))
        avg_s = sum(i.get("containment_score", 0) for i in items) / len(items)
        print(f"  {t}: {c}/{len(items)} ({100*c/len(items):.0f}%) contain={avg_s:.4f} recall={u}")

    if args.output:
        output = {
            "variant": dataset.variant,
            "model": args.model,
            "auth": "max_subscription",
            "extraction_mode": args.extraction,
            "assessment_method": "claude_cli_embedding_containment",
            "total_instances": total,
            "total_correct": correct,
            "overall_accuracy": round(accuracy, 4),
            "avg_containment": round(avg_contain, 4),
            "elapsed_seconds": round(elapsed, 1),
            "instances": results,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def _error_result(instance) -> dict:
    return {
        "question_id": instance.question_id,
        "question_type": instance.question_type,
        "question": instance.question,
        "gold_answer": instance.answer,
        "hypothesis": "[ERROR]",
        "correct": False,
        "judge_raw": "error",
        "containment_score": 0.0,
        "used_recall": False,
        "retrieval_metrics": {},
        "num_sessions": instance.num_sessions,
    }


def _save_checkpoint(path: str, results: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"instances": results}, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LongMemEval via Claude Code CLI (Max subscription)",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run the benchmark")
    run.add_argument("--dataset", required=True, help="Dataset JSON path")
    run.add_argument("--model", default="claude-sonnet-4-6")
    run.add_argument(
        "--extraction",
        choices=["none", "narrow", "auto"],
        default="narrow",
    )
    run.add_argument("--containment-threshold", type=float, default=0.65)
    run.add_argument("--max-instances", type=int, default=None)
    run.add_argument("--n-per-type", type=int, default=None)
    run.add_argument("--types", default=None)
    run.add_argument("--output", default=None)
    run.add_argument("--checkpoint", default=None)
    run.add_argument("--verbose", action="store_true")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    asyncio.run(cmd_run(args))
