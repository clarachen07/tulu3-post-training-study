#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


STAGES = ("base", "sft", "dpo", "rlvr")
KS = (1, 2, 4)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def pass_value(correct, k, n=4):
    return 1.0 - (math.comb(n - correct, k) / math.comb(n, k) if n - correct >= k else 0.0)


def load_run(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

    indices = [row.get("idx") for row in rows]
    expected = list(range(500))
    array_fields = ("code", "pred", "score", "finish_reason")
    lengths_ok = all(
        isinstance(row.get(field), list) and len(row[field]) == 4
        for row in rows
        for field in array_fields
    )
    none_counts = {
        field: sum(item is None for row in rows for item in row.get(field, []))
        for field in array_fields
    }
    score_types_ok = all(type(x) is bool for row in rows for x in row.get("score", []))

    codes = [x for row in rows for x in row["code"]]
    preds = [x for row in rows for x in row["pred"]]
    scores = [x for row in rows for x in row["score"]]
    finishes = [x for row in rows for x in row["finish_reason"]]
    correct_per_question = np.asarray([sum(row["score"]) for row in rows], dtype=np.int8)
    position_correct = [sum(bool(row["score"][j]) for row in rows) for j in range(4)]
    per_question = {
        str(k): np.asarray([pass_value(int(c), k) for c in correct_per_question], dtype=np.float64)
        for k in KS
    }
    char_lengths = [len(x) if isinstance(x, str) else 0 for x in codes]
    box_counts = [x.count("\\boxed{") if isinstance(x, str) else 0 for x in codes]
    empty_code = [not isinstance(x, str) or not x.strip() for x in codes]
    empty_pred = [not isinstance(x, str) or not x.strip() for x in preds]

    result = {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(rows),
        "indices_exact_0_499": indices == expected,
        "unique_indices": len(set(indices)),
        "array_lengths_all_4": lengths_ok,
        "none_counts": none_counts,
        "scores_all_boolean": score_types_ok,
        "total_generations": len(scores),
        "correct_generations": int(sum(scores)),
        "correct_by_position": position_correct,
        "pass_at": {str(k): float(per_question[str(k)].mean() * 100) for k in KS},
        "any_correct_questions": int((correct_per_question > 0).sum()),
        "empty_generations": int(sum(empty_code)),
        "empty_parsed_answers": int(sum(empty_pred)),
        "questions_with_any_empty_parse": int(
            sum(any(not isinstance(x, str) or not x.strip() for x in row["pred"]) for row in rows)
        ),
        "questions_with_all_empty_parse": int(
            sum(all(not isinstance(x, str) or not x.strip() for x in row["pred"]) for row in rows)
        ),
        "finish_reasons": dict(Counter(str(x) for x in finishes)),
        "characters": {
            "mean": float(np.mean(char_lengths)),
            "median": percentile(char_lengths, 50),
            "p90": percentile(char_lengths, 90),
            "p95": percentile(char_lengths, 95),
            "p99": percentile(char_lengths, 99),
            "max": int(max(char_lengths)),
        },
        "boxed": {
            "at_least_one": int(sum(x >= 1 for x in box_counts)),
            "multiple": int(sum(x >= 2 for x in box_counts)),
            "total_occurrences": int(sum(box_counts)),
        },
    }
    hidden = {
        "rows": rows,
        "correct_per_question": correct_per_question,
        "per_question": per_question,
    }
    return result, hidden


def paired_bootstrap(diff, reps=100000, seed=20260727):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=np.float64)
    n = len(diff)
    values = np.empty(reps, dtype=np.float64)
    chunk = 2000
    for start in range(0, reps, chunk):
        size = min(chunk, reps - start)
        idx = rng.integers(0, n, size=(size, n))
        values[start : start + size] = diff[idx].mean(axis=1) * 100
    return {
        "point_pp": float(diff.mean() * 100),
        "ci95_pp": [float(x) for x in np.percentile(values, [2.5, 97.5])],
        "probability_rlvr_gt_dpo": float((values > 0).mean()),
    }


def exact_two_sided_binomial(x, n):
    if n == 0:
        return 1.0
    lo = min(x, n - x)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2**n))


def paired_any_correct(dpo_hidden, rlvr_hidden):
    d = dpo_hidden["correct_per_question"] > 0
    r = rlvr_hidden["correct_per_question"] > 0
    both = int((d & r).sum())
    r_only = int((~d & r).sum())
    d_only = int((d & ~r).sum())
    neither = int((~d & ~r).sum())
    return {
        "both": both,
        "rlvr_only": r_only,
        "dpo_only": d_only,
        "neither": neither,
        "discordant": r_only + d_only,
        "exact_two_sided_p": exact_two_sided_binomial(r_only, r_only + d_only),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="append", nargs=3, metavar=("STAGE", "SEED", "JSONL"), required=True)
    args = parser.parse_args()

    public = {}
    hidden = {}
    for stage, seed, filename in args.run:
        if stage not in STAGES:
            raise ValueError(stage)
        key = f"{stage}-seed{seed}"
        public[key], hidden[key] = load_run(Path(filename))

    identity_fields = ("idx", "question", "gt", "level", "answer")
    ref_key = sorted(hidden)[0]
    ref = hidden[ref_key]["rows"]
    identity = {}
    for key, payload in hidden.items():
        identity[key] = all(
            all(row.get(field) == ref[i].get(field) for field in identity_fields)
            for i, row in enumerate(payload["rows"])
        )

    available_seeds = sorted(
        {key.rsplit("seed", 1)[1] for key in hidden if "-seed" in key},
        key=int,
    )
    comparisons = {}
    for comparator_index, comparator in enumerate(("base", "sft", "dpo"), 1):
        comparison = {}
        comparator_seeds = [
            seed
            for seed in available_seeds
            if f"{comparator}-seed{seed}" in hidden and f"rlvr-seed{seed}" in hidden
        ]
        for seed in comparator_seeds:
            ckey, rkey = f"{comparator}-seed{seed}", f"rlvr-seed{seed}"
            comparison[f"seed{seed}"] = {
                "paired_any_correct": paired_any_correct(hidden[ckey], hidden[rkey]),
                "bootstrap": {
                    str(k): paired_bootstrap(
                        hidden[rkey]["per_question"][str(k)]
                        - hidden[ckey]["per_question"][str(k)],
                        seed=20260727 + comparator_index * 10000 + int(seed) * 100 + k,
                    )
                    for k in KS
                },
            }

        if comparator_seeds:
            pooled = {}
            for k in KS:
                diff = np.mean(
                    [
                        hidden[f"rlvr-seed{seed}"]["per_question"][str(k)]
                        - hidden[f"{comparator}-seed{seed}"]["per_question"][str(k)]
                        for seed in comparator_seeds
                    ],
                    axis=0,
                )
                pooled[str(k)] = paired_bootstrap(
                    diff, seed=20260727 + comparator_index * 10000 + 1000 + k
                )
            comparison[f"{len(comparator_seeds)}seed_question_clustered"] = {
                "seeds": comparator_seeds,
                "bootstrap": pooled,
            }
        comparisons[f"rlvr_minus_{comparator}"] = comparison

    output = {
        "runs": public,
        "identity_matches_reference": identity,
        "comparisons": comparisons,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
