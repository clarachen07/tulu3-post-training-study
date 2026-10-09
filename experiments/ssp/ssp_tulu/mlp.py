from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ResidualPredictor(nn.Module):
    def __init__(self, dimension: int, hidden_width: int = 2048):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * dimension, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, dimension),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([current, previous], dim=-1))


@dataclass
class ProbeExamples:
    previous: torch.Tensor
    current: torch.Tensor
    target: torch.Tensor
    baseline: torch.Tensor
    problem_ids: list[str]
    pair_ids: list[str]

    def subset(self, indices: list[int]) -> "ProbeExamples":
        tensor_indices = torch.tensor(indices, dtype=torch.long)
        return ProbeExamples(
            previous=self.previous.index_select(0, tensor_indices),
            current=self.current.index_select(0, tensor_indices),
            target=self.target.index_select(0, tensor_indices),
            baseline=self.baseline.index_select(0, tensor_indices),
            problem_ids=[self.problem_ids[index] for index in indices],
            pair_ids=[self.pair_ids[index] for index in indices],
        )


def normalized_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum((prediction.float() - target.float()) ** 2, dim=-1)
    denominator = torch.clamp(torch.sum(target.float() ** 2, dim=-1), min=1e-8)
    return numerator / denominator

