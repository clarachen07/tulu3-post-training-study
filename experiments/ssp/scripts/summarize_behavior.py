#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math

from ssp_tulu.io import atomic_json, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def pass_at_k(correct: int, total: int, k: int) -> float:
    if total - correct < k:
        return 1.0
    return 1.0 - math.comb(total - correct, k) / math.comb(total, k)


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.input))
    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 MATH-500 rows, got {len(rows)}")
    sample_counts = {len(row["score"]) for row in rows}
    if len(sample_counts) != 1:
        raise RuntimeError(f"Inconsistent sample counts: {sample_counts}")
    total = sample_counts.pop()
    correct_generations = sum(sum(bool(value) for value in row["score"]) for row in rows)
    summary = {
        "condition": args.condition,
        "mode": args.mode,
        "problems": len(rows),
        "samples_per_problem": total,
        "correct_generations": correct_generations,
        "accuracy_over_generations": correct_generations / (len(rows) * total),
        "observed_pass": {
            str(k): sum(pass_at_k(sum(row["score"]), total, k) for row in rows) / len(rows)
            for k in range(1, total + 1)
        },
        "finish_reasons": {
            reason: sum(row["finish_reason"].count(reason) for row in rows)
            for reason in sorted({reason for row in rows for reason in row["finish_reason"]})
        },
    }
    atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

