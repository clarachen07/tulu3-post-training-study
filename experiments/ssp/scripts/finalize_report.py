#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ssp_tulu.io import atomic_json, read_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def find_endpoint(summary: dict, left: str, right: str, horizon: int) -> dict | None:
    for row in summary.get("primary_endpoints", []):
        if (
            row["left"] == left
            and row["right"] == right
            and row["layer"] == 32
            and row["metric"] == "normalized_mse"
            and row["horizon"] == horizon
        ):
            return row
    return None


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frozen = read_json("outputs/analysis/frozen/summary.json")
    grid = read_json("outputs/analysis/grid-seed42-reserved/summary.json")
    behavior = {}
    for condition in ["B1", "B2", "C", "A2", "A", "A1", "C-match", "A2-match"]:
        behavior[condition] = {}
        for mode in ["greedy", "pass4"]:
            path = Path(f"outputs/behavior/{condition}/{mode}/summary.json")
            if path.exists():
                behavior[condition][mode] = read_json(path)
    dpo_rlvr = find_endpoint(frozen, "dpo", "rlvr", 3)
    c_a = find_endpoint(grid, "C", "A", 1)
    b2_accuracy = behavior["B2"]["greedy"]["observed_pass"]["1"]
    a_accuracy = behavior["A"]["greedy"]["observed_pass"]["1"]
    behavior_delta_pp = 100 * (a_accuracy - b2_accuracy)
    payload = {
        "primary_endpoints": {
            "dpo_vs_rlvr_final_layer_mse3": dpo_rlvr,
            "A_vs_C_final_layer_mse1": c_a,
        },
        "behavior": behavior,
        "A_minus_B2_accuracy_pp": behavior_delta_pp,
        "behavior_non_degradation_pass": behavior_delta_pp >= -2.0,
    }
    atomic_json(output / "summary.json", payload)

    lines = [
        "# Tulu-3-8B SSP Reproduction and Four-Stage Trace Geometry: Final Report",
        "",
        "## Primary endpoints",
        "",
    ]
    if dpo_rlvr:
        lines.extend(
            [
                f"- DPO→RLVR final-layer MSE@3: {dpo_rlvr['left_mean']:.6g} → "
                f"{dpo_rlvr['right_mean']:.6g}; difference "
                f"{dpo_rlvr['mean_difference']:.6g}, 95% CI "
                f"[{dpo_rlvr['ci95_low']:.6g}, {dpo_rlvr['ci95_high']:.6g}].",
            ]
        )
    if c_a:
        lines.extend(
            [
                f"- C→A final-layer MSE@1: {c_a['left_mean']:.6g} → "
                f"{c_a['right_mean']:.6g}; improvement factor "
                f"{c_a['improvement_ratio_left_over_right']:.3g}×。",
            ]
        )
    lines.extend(
        [
            "",
            "## Behavioral preservation",
            "",
            f"- B2 greedy MATH-500: {100*b2_accuracy:.2f}%.",
            f"- A greedy MATH-500: {100*a_accuracy:.2f}%.",
            f"- A−B2: {behavior_delta_pp:+.2f} pp; "
            f"{'Pass' if behavior_delta_pp >= -2 else 'Fail'} against the −2 pp non-degradation bound.",
            "",
            "## Reproducibility",
            "",
            "- Per-problem geometry, bootstrap summaries, training logs, adapters, merged BF16 models, "
            "quantization checks, behavior generations, and MLP probes are stored under `outputs/`.",
            "- `data/processed/*manifest.json` records exclusions and hashes; "
            "`artifacts/preflight.json` records tokenizer, model, and environment contracts.",
            "- Scientific null results do not stop the run; technical failures preserve the scene through a pipeline failure marker.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    files = [
        Path("configs/study.json"),
        Path("data/processed/math_train_manifest.json"),
        Path("data/processed/math500_trace_bank_manifest.json"),
        Path("artifacts/preflight.json"),
        output / "summary.json",
        output / "REPORT.md",
    ]
    atomic_json(
        output / "artifact_hashes.json",
        {str(path): sha256_file(path) for path in files if path.exists()},
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
