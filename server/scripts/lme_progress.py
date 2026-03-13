#!/usr/bin/env python3
"""Live progress monitor for LongMemEval benchmark.

Usage: python scripts/lme_progress.py [checkpoint_path]
"""

import json
import sys
from pathlib import Path

def show_progress(path: str = "results/longmemeval_oracle_checkpoint.json"):
    p = Path(path)
    if not p.exists():
        print("No checkpoint file found.")
        return

    d = json.load(open(p))
    instances = d["instances"]
    total = len(instances)
    correct = sum(1 for i in instances if i["correct"])

    print(f"\n{'='*60}")
    print(f"  LongMemEval Progress: {total}/500")
    print(f"  Overall: {correct}/{total} = {100*correct/total:.1f}%")
    print(f"{'='*60}\n")

    by_type: dict = {}
    for i in instances:
        qid = i["question_id"]
        t = "abstention" if qid.endswith("_abs") else i["question_type"]
        by_type.setdefault(t, {"n": 0, "c": 0, "errors": []})
        by_type[t]["n"] += 1
        if i["correct"]:
            by_type[t]["c"] += 1
        else:
            by_type[t]["errors"].append(i)

    print(f"  {'Type':<30} {'Score':>10} {'Acc':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*8}")
    for t in sorted(by_type.keys()):
        v = by_type[t]
        acc = 100 * v["c"] / v["n"] if v["n"] else 0
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"  {t:<30} {v['c']:>3}/{v['n']:<3}    {acc:>5.1f}%  {bar}")

    # Category accuracy (official metric)
    non_abs = {k: v for k, v in by_type.items() if k != "abstention"}
    if non_abs:
        cat_acc = sum(
            v["c"] / v["n"] for v in non_abs.values() if v["n"]
        ) / len(non_abs)
        print(f"\n  Category accuracy (official): {100*cat_acc:.1f}%")

    # Recent results
    print(f"\n  Last 5 results:")
    for i in instances[-5:]:
        mark = "✓" if i["correct"] else "✗"
        print(f"    {mark} {i['question_id'][:12]:<14} {i['question_type']}")

    print()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/longmemeval_oracle_checkpoint.json"
    show_progress(path)
