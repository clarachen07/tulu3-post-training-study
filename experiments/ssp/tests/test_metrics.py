import torch

from ssp_tulu.metrics import layer_smoothness, trajectory_metrics


def test_linear_trajectory_has_zero_prediction_error():
    direction = torch.tensor([1.0, 2.0, -1.0])
    states = torch.stack([index * direction for index in range(8)])
    rows = trajectory_metrics(states, [1, 2, 3])
    mse = [row["value"] for row in rows if row["metric"] == "normalized_mse"]
    cosine = [row["value"] for row in rows if row["metric"] == "cos_score"]
    assert max(mse) < 1e-8
    assert max(cosine) < 1e-6


def test_linear_layer_path_is_smooth():
    layers = {index: torch.ones(3, 4) * index for index in [4, 8, 12, 16]}
    assert layer_smoothness(layers) < 1e-8
