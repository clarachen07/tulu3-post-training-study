#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ssp_tulu.io import atomic_json, read_json, write_jsonl
from ssp_tulu.metrics import (
    aggregate_problem_rows,
    bootstrap_mean_ci,
    layer_smoothness,
    trajectory_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--geometry-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(
    problem_rows: list[dict],
    left: str,
    right: str,
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    lookup = {}
    for row in problem_rows:
        key = (
            row["model"],
            row["boundary_mode"],
            row["problem_id"],
            row["layer"],
            row["metric"],
            row["horizon"],
        )
        lookup[key] = row["value"]
    structural_keys = sorted(
        {
            (row["boundary_mode"], row["problem_id"], row["layer"], row["metric"], row["horizon"])
            for row in problem_rows
            if row["model"] in {left, right}
        }
    )
    grouped: dict[tuple, list[float]] = defaultdict(list)
    paired_values: dict[tuple, tuple[list[float], list[float]]] = {}
    for boundary, problem, layer, metric, horizon in structural_keys:
        left_key = (left, boundary, problem, layer, metric, horizon)
        right_key = (right, boundary, problem, layer, metric, horizon)
        if left_key not in lookup or right_key not in lookup:
            continue
        group = (boundary, layer, metric, horizon)
        grouped[group].append(lookup[right_key] - lookup[left_key])
        left_list, right_list = paired_values.setdefault(group, ([], []))
        left_list.append(lookup[left_key])
        right_list.append(lookup[right_key])
    output = []
    for group, differences in sorted(grouped.items()):
        mean_diff, low, high = bootstrap_mean_ci(
            np.asarray(differences), samples=bootstrap_samples, seed=seed
        )
        left_values, right_values = paired_values[group]
        left_mean = float(np.mean(left_values))
        right_mean = float(np.mean(right_values))
        output.append(
            {
                "comparison": f"{right}-{left}",
                "left": left,
                "right": right,
                "boundary_mode": group[0],
                "layer": group[1],
                "metric": group[2],
                "horizon": group[3],
                "n_problems": len(differences),
                "left_mean": left_mean,
                "right_mean": right_mean,
                "mean_difference": mean_diff,
                "ci95_low": low,
                "ci95_high": high,
                "improvement_ratio_left_over_right": (
                    left_mean / right_mean if right_mean != 0 else float("inf")
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    horizons = list(map(int, config["geometry"]["horizons"]))
    point_rows: list[dict] = []
    manifests = []
    for directory_string in args.geometry_dirs:
        directory = Path(directory_string)
        manifest = read_json(directory / "manifest.json")
        manifests.append(manifest)
        for shard_info in manifest["shards"]:
            records = torch.load(shard_info["path"], map_location="cpu", weights_only=False)
            for record in records:
                for layer, states in record["layer_states"].items():
                    identity = {
                        "model": record["model"],
                        "boundary_mode": record["boundary_mode"],
                        "pair_id": record["pair_id"],
                        "problem_id": record["problem_id"],
                        "source": record["source"],
                        "correct": record["correct"],
                        "layer": int(layer),
                    }
                    for metric in trajectory_metrics(states, horizons):
                        point_rows.append({**identity, **metric})
                point_rows.append(
                    {
                        "model": record["model"],
                        "boundary_mode": record["boundary_mode"],
                        "pair_id": record["pair_id"],
                        "problem_id": record["problem_id"],
                        "source": record["source"],
                        "correct": record["correct"],
                        "layer": -1,
                        "metric": "layer_smoothness",
                        "horizon": 0,
                        "position": -1,
                        "value": layer_smoothness(record["layer_states"]),
                    }
                )
    write_jsonl(output / "point_metrics.jsonl", point_rows)
    problem_rows = aggregate_problem_rows(point_rows)
    write_jsonl(output / "problem_metrics.jsonl", problem_rows)

    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in problem_rows:
        key = (
            row["model"],
            row["boundary_mode"],
            row["layer"],
            row["metric"],
            row["horizon"],
        )
        grouped[key].append(row["value"])
    aggregate_rows = []
    for key, values in sorted(grouped.items()):
        mean, low, high = bootstrap_mean_ci(
            np.asarray(values),
            samples=int(config["geometry"]["bootstrap_samples"]),
            seed=int(config["geometry"]["bootstrap_seed"]),
        )
        aggregate_rows.append(
            {
                "model": key[0],
                "boundary_mode": key[1],
                "layer": key[2],
                "metric": key[3],
                "horizon": key[4],
                "n_problems": len(values),
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    models = {row["model"] for row in problem_rows}
    comparisons = []
    for left, right in [("dpo", "rlvr"), ("C", "A")]:
        if {left, right}.issubset(models):
            comparisons.extend(
                comparison_rows(
                    problem_rows,
                    left=left,
                    right=right,
                    bootstrap_samples=int(config["geometry"]["bootstrap_samples"]),
                    seed=int(config["geometry"]["bootstrap_seed"]),
                )
            )
    write_csv(output / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output / "paired_comparisons.csv", comparisons)
    primary = [
        row
        for row in comparisons
        if (
            row["left"] == "dpo"
            and row["right"] == "rlvr"
            and row["layer"] == 32
            and row["metric"] == "normalized_mse"
            and row["horizon"] == 3
        )
        or (
            row["left"] == "C"
            and row["right"] == "A"
            and row["layer"] == 32
            and row["metric"] == "normalized_mse"
            and row["horizon"] == 1
        )
    ]
    atomic_json(
        output / "summary.json",
        {
            "geometry_manifests": manifests,
            "aggregate_rows": len(aggregate_rows),
            "problem_rows": len(problem_rows),
            "point_rows": len(point_rows),
            "primary_endpoints": primary,
        },
    )
    print(json.dumps({"primary_endpoints": primary, "models": sorted(models)}, indent=2))


if __name__ == "__main__":
    main()

