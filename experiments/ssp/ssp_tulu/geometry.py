from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidate = model
    if hasattr(candidate, "get_base_model"):
        candidate = candidate.get_base_model()
    if hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
        return candidate.model.layers
    if (
        hasattr(candidate, "model")
        and hasattr(candidate.model, "model")
        and hasattr(candidate.model.model, "layers")
    ):
        return candidate.model.model.layers
    raise TypeError(f"Cannot locate decoder layers on {type(model)}")


class BoundaryHookCapture:
    """Capture decoder-layer outputs before the model's final RMSNorm."""

    def __init__(self, model: torch.nn.Module, layers: list[int]):
        modules = decoder_layers(model)
        if max(layers) > len(modules):
            raise ValueError(f"Requested layer {max(layers)} but model has {len(modules)}")
        self.positions: torch.Tensor | None = None
        self.captured: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer_number in layers:
            module = modules[layer_number - 1]
            self.handles.append(module.register_forward_hook(self._hook(layer_number)))

    def _hook(self, layer_number: int) -> Callable:
        def capture(_module: torch.nn.Module, _inputs: tuple, output: Any) -> None:
            if self.positions is None:
                raise RuntimeError("Boundary positions were not set before forward")
            hidden = output[0] if isinstance(output, tuple) else output
            selected = hidden[0].index_select(0, self.positions.to(hidden.device))
            self.captured[layer_number] = selected.detach().to(
                device="cpu", dtype=torch.float16
            )

        return capture

    def begin(self, positions: list[int]) -> None:
        self.positions = torch.tensor(positions, dtype=torch.long)
        self.captured = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self) -> "BoundaryHookCapture":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

