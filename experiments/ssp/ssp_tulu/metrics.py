from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def trajectory_metrics(
    states: torch.Tensor,
    horizons: Iterable[int],
    eps: float = 1e-8,
) -> list[dict[str, float | int]]:
    z = states.float()
    rows: list[dict[str, float | int]] = []
    for horizon in horizons:
        for k in range(1, len(z) - horizon):
            prediction = z[k] + horizon * (z[k] - z[k - 1])
            target = z[k + horizon]
            normalized_mse = torch.sum((prediction - target) ** 2) / torch.clamp(
                torch.sum(target**2), min=eps
            )
            rows.append(
                {
                    "metric": "normalized_mse",
                    "horizon": int(horizon),
                    "position": int(k),
                    "value": float(normalized_mse),
                }
            )

    for k in range(1, len(z) - 1):
        before = z[k] - z[k - 1]
        after = z[k + 1] - z[k]
        cos_score = 1.0 - F.cosine_similarity(before[None], after[None], eps=eps)[0]
        secant = z[k + 1] - z[k - 1]
        unit = secant / torch.clamp(torch.linalg.vector_norm(secant), min=eps)
        perpendicular = before - torch.dot(before, unit) * unit
        perp_score = torch.linalg.vector_norm(perpendicular) / torch.clamp(
            torch.linalg.vector_norm(before), min=eps
        )
        rows.extend(
            [
                {
                    "metric": "cos_score",
                    "horizon": 1,
                    "position": int(k),
                    "value": float(cos_score),
                },
                {
                    "metric": "perp_score",
                    "horizon": 1,
                    "position": int(k),
                    "value": float(perp_score),
                },
            ]
        )
    return rows


def layer_smoothness(layer_states: dict[int, torch.Tensor], eps: float = 1e-8) -> float:
    layers = sorted(layer_states)
    if len(layers) < 3:
        return float("nan")
    minimum_steps = min(layer_states[layer].shape[0] for layer in layers)
    scores: list[float] = []
    for position in range(minimum_steps):
        path = torch.stack([layer_states[layer][position].float() for layer in layers])
        before = path[1:-1] - path[:-2]
        after = path[2:] - path[1:-1]
        score = 1.0 - F.cosine_similarity(before, after, dim=-1, eps=eps)
        scores.extend(score.cpu().tolist())
    return float(np.mean(scores)) if scores else float("nan")


def aggregate_problem_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    identity: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row["model"],
            row["boundary_mode"],
            row["problem_id"],
            row["layer"],
            row["metric"],
            row["horizon"],
        )
        grouped[key].append(float(row["value"]))
        identity[key] = {field: row[field] for field in (
            "model", "boundary_mode", "problem_id", "layer", "metric", "horizon"
        )}
    output = []
    for key, values in sorted(grouped.items()):
        output.append({**identity[key], "value": float(np.mean(values)), "n_points": len(values)})
    return output


def bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))

