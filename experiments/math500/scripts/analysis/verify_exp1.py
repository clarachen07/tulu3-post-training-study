#!/usr/bin/env python3
"""Validate Experiment 1 JSONL files and independently recompute pass@k."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path


EXPECTED_SEEDS = {
    "base": (1, 2, 3),
    "sft": (1, 2, 3, 4),
    "dpo": (1, 2, 3, 4),
    "rlvr": (1, 2, 3, 4),
}
KEY_ARRAYS = ("code", "pred", "score", "finish_reason")
FILENAME_RE = re.compile(r"seed(?P<seed>[1-4])_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pass_at_k(n: int, c: int, k: int) -> float:
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def exact_two_sided_binomial(successes: int, trials: int) -> float:
    """Two-sided exact binomial test for p=0.5."""
    observed = math.comb(trials, successes) / (2**trials)
    return min(
        1.0,
        sum(
            math.comb(trials, value) / (2**trials)
            for value in range(trials + 1)
            if math.comb(trials, value) / (2**trials) <= observed + 1e-15
        ),
    )


def paired_bootstrap(
    differences: list[int],
    *,
    repetitions: int = 100_000,
    seed: int = 20_260_729,
) -> dict[str, object]:
    """Question-level paired bootstrap of percentage-point differences."""
    rng = random.Random(seed)
    count = len(differences)
    draws = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count * 100
        for _ in range(repetitions)
    )
    lower = draws[int(0.025 * repetitions)]
    upper = draws[int(0.975 * repetitions)]
    return {
        "repetitions": repetitions,
        "seed": seed,
        "point_pp": sum(differences) / count * 100,
        "ci95_pp": [lower, upper],
        "probability_rlvr_gt_dpo": sum(value > 0 for value in draws) / repetitions,
    }


def discover(root: Path) -> dict[tuple[str, int], Path]:
    found: dict[tuple[str, int], Path] = {}
    for path in root.rglob("test_plain-boxed_-1_seed*_t0.6_s0_e-1.jsonl"):
        match = FILENAME_RE.search(path.name)
        if not match:
            continue
        seed = int(match.group("seed"))
        stage = next((part for part in path.parts if part in EXPECTED_SEEDS), None)
        if stage is None:
            continue
        key = (stage, seed)
        if key in found:
            raise ValueError(f"duplicate run for {stage} seed {seed}: {path}")
        found[key] = path
    expected = {(stage, seed) for stage, seeds in EXPECTED_SEEDS.items() for seed in seeds}
    missing = sorted(expected - set(found))
    extra = sorted(set(found) - expected)
    if missing or extra:
        raise ValueError(f"run matrix mismatch; missing={missing}, extra={extra}")
    return found


def audit_run(stage: str, seed: int, path: Path) -> tuple[dict, dict[int, list[bool]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    indices = [row.get("idx") for row in rows]
    if len(rows) != 500 or indices != list(range(500)):
        raise ValueError(f"{stage} seed {seed}: expected ordered indices 0..499")

    none_counts = Counter()
    finish_reasons = Counter()
    by_question: dict[int, list[bool]] = {}
    empty_generations = 0
    empty_parses = 0

    for row in rows:
        for key in KEY_ARRAYS:
            value = row.get(key)
            if not isinstance(value, list) or len(value) != 4:
                raise ValueError(f"{stage} seed {seed} idx {row['idx']}: bad {key}")
            none_counts[key] += sum(item is None for item in value)
        if not all(isinstance(item, bool) for item in row["score"]):
            raise ValueError(f"{stage} seed {seed} idx {row['idx']}: non-boolean score")
        by_question[row["idx"]] = list(row["score"])
        finish_reasons.update(row["finish_reason"])
        empty_generations += sum(not str(item or "").strip() for item in row["code"])
        empty_parses += sum(not str(item or "").strip() for item in row["pred"])

    if any(none_counts.values()):
        raise ValueError(f"{stage} seed {seed}: None values found: {dict(none_counts)}")

    correct = sum(sum(scores) for scores in by_question.values())
    pass_values = {}
    for k in (1, 2, 4):
        pass_values[str(k)] = (
            sum(pass_at_k(4, sum(scores), k) for scores in by_question.values()) / 500 * 100
        )

    return (
        {
            "stage": stage,
            "seed": seed,
            "path": str(path),
            "sha256": sha256(path),
            "rows": len(rows),
            "total_generations": 2000,
            "correct_generations": correct,
            "pass_at": pass_values,
            "any_correct_questions": sum(any(scores) for scores in by_question.values()),
            "empty_generations": empty_generations,
            "empty_parsed_answers": empty_parses,
            "finish_reasons": dict(finish_reasons),
            "indices_exact_0_499": True,
            "array_lengths_all_4": True,
            "none_counts": dict(none_counts),
        },
        by_question,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--index-output", type=Path)
    args = parser.parse_args()

    paths = discover(args.raw_root)
    runs = {}
    scores: dict[tuple[str, int], dict[int, list[bool]]] = {}
    for stage, seeds in EXPECTED_SEEDS.items():
        for seed in seeds:
            audit, per_question = audit_run(stage, seed, paths[(stage, seed)])
            audit["path"] = str(paths[(stage, seed)].relative_to(args.raw_root))
            runs[f"{stage}-seed{seed}"] = audit
            scores[(stage, seed)] = per_question

    pooled = {}
    for stage, seeds in EXPECTED_SEEDS.items():
        n = len(seeds) * 4
        per_question = {
            idx: [score for seed in seeds for score in scores[(stage, seed)][idx]]
            for idx in range(500)
        }
        pooled[stage] = {
            "samples_per_question": n,
            "correct_generations": sum(sum(items) for items in per_question.values()),
            "pass_at": {
                str(k): sum(pass_at_k(n, sum(items), k) for items in per_question.values())
                / 500
                * 100
                for k in (1, 2, 4, 8, 12, 16)
                if k <= n
            },
            "maximum_k_solved_questions": sum(any(items) for items in per_question.values()),
        }

    dpo = {
        idx: [score for seed in EXPECTED_SEEDS["dpo"] for score in scores[("dpo", seed)][idx]]
        for idx in range(500)
    }
    rlvr = {
        idx: [score for seed in EXPECTED_SEEDS["rlvr"] for score in scores[("rlvr", seed)][idx]]
        for idx in range(500)
    }
    paired = Counter(
        (
            "both"
            if any(dpo[idx]) and any(rlvr[idx])
            else "dpo_only"
            if any(dpo[idx])
            else "rlvr_only"
            if any(rlvr[idx])
            else "neither"
        )
        for idx in range(500)
    )
    paired_differences = [
        int(any(rlvr[idx])) - int(any(dpo[idx])) for idx in range(500)
    ]
    bootstrap = paired_bootstrap(paired_differences)
    exact_p = exact_two_sided_binomial(
        paired["rlvr_only"],
        paired["rlvr_only"] + paired["dpo_only"],
    )

    checks = {
        "base_pass12_is_27_6": abs(pooled["base"]["pass_at"]["12"] - 27.6) < 1e-9,
        "sft_pass16_is_73_2": abs(pooled["sft"]["pass_at"]["16"] - 73.2) < 1e-9,
        "dpo_pass16_is_79_2": abs(pooled["dpo"]["pass_at"]["16"] - 79.2) < 1e-9,
        "rlvr_pass16_is_80_6": abs(pooled["rlvr"]["pass_at"]["16"] - 80.6) < 1e-9,
        "paired_counts_match": dict(paired)
        == {"both": 372, "rlvr_only": 31, "dpo_only": 24, "neither": 73},
        "paired_ci_matches": bootstrap["ci95_pp"] == [-1.4000000000000001, 4.3999999999999995],
        "exact_p_matches": abs(exact_p - 0.41875418826901556) < 1e-12,
    }
    if not all(checks.values()):
        raise ValueError(f"frozen-result mismatch: {checks}")

    result = {
        "definition": "Independent standard-library recomputation from actual archived samples.",
        "run_count": len(runs),
        "runs": runs,
        "pooled": pooled,
        "dpo_rlvr_pass_at_16": {
            **dict(paired),
            "difference_pp": pooled["rlvr"]["pass_at"]["16"]
            - pooled["dpo"]["pass_at"]["16"],
            "bootstrap": bootstrap,
            "exact_two_sided_p": exact_p,
        },
        "frozen_checks": checks,
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    if args.index_output:
        args.index_output.parent.mkdir(parents=True, exist_ok=True)
        with args.index_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "run_id",
                    "protocol",
                    "stage",
                    "seed",
                    "status",
                    "questions",
                    "samples_per_question",
                    "sha256",
                    "release_asset",
                    "relative_raw_path",
                ]
            )
            for run_id, run in runs.items():
                path = paths[(run["stage"], run["seed"])]
                writer.writerow(
                    [
                        run_id,
                        "shared-tulu-sft-template",
                        run["stage"],
                        run["seed"],
                        "complete",
                        500,
                        4,
                        run["sha256"],
                        "exp1-shared-chat-raw-jsonl.tar.zst",
                        path.relative_to(args.raw_root),
                    ]
                )


if __name__ == "__main__":
    main()
