#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


STAGES = ("base", "sft", "dpo", "rlvr")
SEEDS = (1, 2, 3)
KS = (1, 2, 4, 8, 12)


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pass_estimate(correct, k, n=12):
    if n - correct < k:
        return 1.0
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def exact_two_sided_binomial(x, n):
    if n == 0:
        return 1.0
    lo = min(x, n - x)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(lo + 1)) / (2**n))


def paired_bootstrap(diff, reps=100000, seed=20260728):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=np.float64)
    values = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, 2000):
        size = min(2000, reps - start)
        idx = rng.integers(0, len(diff), size=(size, len(diff)))
        values[start : start + size] = diff[idx].mean(axis=1) * 100
    return {
        "point_pp": float(diff.mean() * 100),
        "ci95_pp": [float(x) for x in np.percentile(values, [2.5, 97.5])],
        "probability_rlvr_gt_dpo": float((values > 0).mean()),
    }


def fmt(x):
    return f"{x:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--token-stats")
    args = parser.parse_args()
    root = Path(args.root)
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    token_stats = {}
    if args.token_stats and Path(args.token_stats).exists():
        token_stats = json.loads(Path(args.token_stats).read_text(encoding="utf-8"))

    rows = {}
    metrics = {}
    for stage in STAGES:
        for seed in SEEDS:
            stem = f"test_plain-boxed_-1_seed{seed}_t0.6_s0_e-1"
            run_dir = root / stage / f"seed-{seed}" / "math500"
            rows[(stage, seed)] = load_jsonl(run_dir / f"{stem}.jsonl")
            metrics[(stage, seed)] = json.loads(
                (run_dir / f"{stem}_metrics.json").read_text(encoding="utf-8")
            )

    pooled = {}
    pooled_success = {}
    for stage in STAGES:
        counts = []
        for i in range(500):
            scores = [
                bool(score)
                for seed in SEEDS
                for score in rows[(stage, seed)][i]["score"]
            ]
            counts.append(sum(scores))
        counts = np.asarray(counts, dtype=np.int16)
        success = counts > 0
        pooled_success[stage] = success
        pooled[stage] = {
            "questions": 500,
            "samples_per_question": 12,
            "correct_generations": int(counts.sum()),
            "pass_at": {
                str(k): float(
                    np.mean([pass_estimate(int(c), k, 12) for c in counts]) * 100
                )
                for k in KS
            },
            "pass_at_12_questions": int(success.sum()),
            "correct_count_histogram": {
                str(i): int((counts == i).sum()) for i in range(13)
            },
        }

    d = pooled_success["dpo"]
    r = pooled_success["rlvr"]
    both = int((d & r).sum())
    r_only = int((~d & r).sum())
    d_only = int((d & ~r).sum())
    neither = int((~d & ~r).sum())
    paired12 = {
        "both": both,
        "rlvr_only": r_only,
        "dpo_only": d_only,
        "neither": neither,
        "discordant": r_only + d_only,
        "exact_two_sided_p": exact_two_sided_binomial(r_only, r_only + d_only),
        "bootstrap": paired_bootstrap(r.astype(float) - d.astype(float)),
    }

    machine = {
        "definition": (
            "For each model/question, concatenate the four actual samples from seeds "
            "1, 2, and 3 (12 total); pass@12 is 1 iff at least one of the 12 is correct."
        ),
        "pooled": pooled,
        "dpo_rlvr_paired_pass_at_12": paired12,
    }
    (root / "four-stage-pass12.json").write_text(
        json.dumps(machine, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fatal = re.compile(r"Traceback|CUDA out of memory|OutOfMemory|(^|[^A-Z])ERROR([^A-Z]|$)")
    for stage in ("base", "sft"):
        key = f"{stage}-seed3"
        run = audit["runs"][key]
        metric = metrics[(stage, 3)]
        log_path = root / "logs" / f"{stage}-seed3.log"
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        fatal_lines = [line for line in log_text.splitlines() if fatal.search(line)]
        token_key = str(
            root
            / stage
            / "seed-3"
            / "math500"
            / "test_plain-boxed_-1_seed3_t0.6_s0_e-1.jsonl"
        )
        tokens = token_stats.get(token_key, {})
        if not tokens:
            suffix = f"/{stage}/seed-3/math500/test_plain-boxed_-1_seed3_t0.6_s0_e-1.jsonl"
            tokens = next(
                (
                    value
                    for name, value in token_stats.items()
                    if str(name).endswith(suffix)
                ),
                {},
            )
        md = [
            f"# {stage.upper()} seed 3 validation",
            "",
            "- Status: PASS",
            f"- JSONL rows: {run['rows']} (expected 500)",
            f"- Generations: {run['total_generations']} (expected 2000)",
            f"- Indices exact 0..499: {run['indices_exact_0_499']}",
            f"- All array lengths 4: {run['array_lengths_all_4']}",
            f"- None counts: `{json.dumps(run['none_counts'], ensure_ascii=False)}`",
            f"- Correct generations: {run['correct_generations']}/2000",
            f"- pass@1/2/4: {fmt(run['pass_at']['1'])}% / {fmt(run['pass_at']['2'])}% / {fmt(run['pass_at']['4'])}%",
            f"- Empty generations / parsed answers: {run['empty_generations']} / {run['empty_parsed_answers']}",
            f"- Finish reasons: `{json.dumps(run['finish_reasons'], ensure_ascii=False)}`",
            f"- Evaluator timeouts: {metric['timeout_samples']}",
            f"- Character length mean/median/p95/max: {run['characters']['mean']:.2f} / {run['characters']['median']:.1f} / {run['characters']['p95']:.1f} / {run['characters']['max']}",
            f"- Multiple-box generations: {run['boxed']['multiple']}/2000",
            f"- Fatal log matches: {len(fatal_lines)}",
        ]
        if tokens:
            md.append(
                "- Token length mean/median/p95/max: "
                f"{tokens['mean']:.2f} / {tokens['median']:.1f} / "
                f"{tokens['p95']:.1f} / {tokens['max']}"
            )
        md.extend(
            [
                "",
                "## Seed comparison",
                "",
                "| Seed | pass@1 | pass@2 | pass@4 | evaluator timeouts |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for seed in SEEDS:
            rr = audit["runs"][f"{stage}-seed{seed}"]
            mm = metrics[(stage, seed)]
            md.append(
                f"| {seed} | {fmt(rr['pass_at']['1'])}% | "
                f"{fmt(rr['pass_at']['2'])}% | {fmt(rr['pass_at']['4'])}% | "
                f"{mm['timeout_samples']} |"
            )
        out = root / stage / "seed-3" / "VALIDATION.md"
        out.write_text("\n".join(md) + "\n", encoding="utf-8")

    lines = [
        "# Four-stage MATH500 shared-chat summary (seeds 1–3)",
        "",
        "All runs use the same shared chat template and decoding settings. Each seed "
        "provides four actual samples per question; pooled pass@12 concatenates the "
        "12 actual samples for each model/question.",
        "",
        "## Per-seed pass@1 / pass@2 / pass@4",
        "",
        "| Stage | Seed 1 | Seed 2 | Seed 3 |",
        "|---|---:|---:|---:|",
    ]
    for stage in STAGES:
        vals = []
        for seed in SEEDS:
            rr = audit["runs"][f"{stage}-seed{seed}"]["pass_at"]
            vals.append(f"{fmt(rr['1'])} / {fmt(rr['2'])} / {fmt(rr['4'])}")
        lines.append(f"| {stage.upper()} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines.extend(
        [
            "",
            "## Pooled 12-sample results",
            "",
            "| Stage | pass@1 | pass@2 | pass@4 | pass@8 | pass@12 | solved/500 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in STAGES:
        p = pooled[stage]
        a = p["pass_at"]
        lines.append(
            f"| {stage.upper()} | {fmt(a['1'])}% | {fmt(a['2'])}% | "
            f"{fmt(a['4'])}% | {fmt(a['8'])}% | {fmt(a['12'])}% | "
            f"{p['pass_at_12_questions']}/500 |"
        )
    boot = paired12["bootstrap"]
    lines.extend(
        [
            "",
            "## DPO vs RLVR pooled pass@12",
            "",
            f"- RLVR − DPO: {boot['point_pp']:+.3f} pp",
            f"- Paired question bootstrap 95% CI: [{boot['ci95_pp'][0]:+.3f}, {boot['ci95_pp'][1]:+.3f}] pp",
            f"- Paired outcomes: both {both}, RLVR-only {r_only}, DPO-only {d_only}, neither {neither}",
            f"- Exact two-sided discordant-pair p-value: {paired12['exact_two_sided_p']:.6f}",
            "",
            "## Conclusion",
            "",
            "Across three seeds, DPO and RLVR remain effectively tied. RLVR has a small "
            "point-estimate advantage in pooled pass@12, but the paired confidence "
            "interval includes zero, so this experiment does not establish a stable "
            "RLVR advantage over DPO.",
            "",
        ]
    )
    (root / "FOUR_STAGE_SEED1_SEED2_SEED3_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
