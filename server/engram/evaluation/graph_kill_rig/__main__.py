"""CLI for the graph deciding-experiment rig.

    uv run python -m engram.evaluation.graph_kill_rig \
        --scratch /path/to/scratch/graph_kill --n 60 --out result.json

The rig always writes a throwaway lite brain under ``--scratch``. It never
opens ``~/.helix``: CLAUDE.md forbids multi-opening the native graph, and a
deciding experiment must not be able to mutate the corpus it is deciding about.

``--fault`` breaks one precondition on purpose so the VOID path can be
demonstrated rather than asserted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from engram.evaluation.graph_kill_rig.arms import SCORERS
from engram.evaluation.graph_kill_rig.runner import FAULTS, RigOptions, run_rig

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="graph_kill_rig", description=__doc__)
    parser.add_argument("--scratch", required=True, type=Path, help="throwaway brain dir")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--n", type=int, default=60, help="bridge questions to build")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=10, help="recall limit per query")
    parser.add_argument(
        "--producer",
        default="proposals",
        choices=["proposals", "narrow"],
        help="who builds the graph: 'proposals' replays M3.1's planted control, "
        "everything else builds it organically from the same text",
    )
    parser.add_argument(
        "--fault",
        default="none",
        choices=list(FAULTS),
        help="deliberately break one precondition to demonstrate the VOID path",
    )
    parser.add_argument("--reuse", action="store_true", help="reuse an existing scratch brain")
    parser.add_argument("--distractors", type=int, default=1)
    parser.add_argument("--filler", type=int, default=30)
    parser.add_argument(
        "--scorer",
        default="gold_episode",
        choices=sorted(SCORERS),
        help="gold_episode resolves the gold by id; multi_source_cover applies "
        "lane 1's union rule (engram/evaluation/meter.py) to the same rows",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the envelope here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    options = RigOptions(
        scratch_dir=args.scratch,
        repo_root=args.repo_root,
        n=args.n,
        seed=args.seed,
        limit=args.limit,
        producer=args.producer,
        fault=args.fault,
        reuse=args.reuse,
        distractors_per_person=args.distractors,
        filler=args.filler,
    )
    envelope = asyncio.run(run_rig(options, SCORERS[args.scorer]()))
    text = json.dumps(envelope, indent=1, sort_keys=True, default=str)
    if args.out:
        args.out.write_text(text)
    print(text)

    if envelope["status"] == "VOID":
        print("\nVOID — no result produced. Failed pre-flight:", file=sys.stderr)
        for reason in envelope["refusal_reasons"]:
            print(f"  - {reason}", file=sys.stderr)
        return 2
    verdict = envelope["verdict"]["verdict"]
    print(f"\nVERDICT: {verdict}", file=sys.stderr)
    for reason in envelope["verdict"]["kill_reasons"]:
        print(f"  KILL {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
