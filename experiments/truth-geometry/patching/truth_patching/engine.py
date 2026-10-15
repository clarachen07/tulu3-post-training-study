"""Single-site post-block residual replacement, before final RMSNorm."""

from __future__ import annotations

import torch


def extract_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"Unsupported block output {type(output)!r}")


def replace_hidden(output, position, vector):
    hidden = extract_hidden(output)
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError(
            "Replacement requires batch size one and [batch,sequence,hidden]"
        )
    if not 0 <= position < hidden.shape[1] or tuple(vector.shape) != (hidden.shape[2],):
        raise ValueError("Invalid position or donor vector shape")
    replaced = hidden.clone()
    replaced[0, position, :] = vector.to(device=hidden.device, dtype=hidden.dtype)
    return (replaced,) + output[1:] if isinstance(output, tuple) else replaced


def _inputs(model, input_ids):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Expected a single unpadded input")
    ids = input_ids.to(next(model.parameters()).device)
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)


@torch.inference_mode()
def capture(model, input_ids):
    """Capture raw outputs of every decoder block, not output_hidden_states[-1]."""
    model.eval()
    activations, handles = {}, []

    def callback(index):
        def hook(module, args, output):
            activations[index] = extract_hidden(output).detach().to("cpu").clone()

        return hook

    try:
        for index, block in enumerate(model.model.layers):
            handles.append(block.register_forward_hook(callback(index)))
        logits = model(**_inputs(model, input_ids)).logits.detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return logits, activations


@torch.inference_mode()
def forward(model, input_ids, patch=None, observe=False):
    model.eval()
    handles = []
    if patch is not None:
        layer, position, vector = patch
        if not 0 <= layer < len(model.model.layers):
            raise ValueError("Invalid block index")
    try:
        if observe:
            for block in model.model.layers:
                handles.append(
                    block.register_forward_hook(lambda module, args, output: None)
                )
        if patch is not None:
            handles.append(
                model.model.layers[layer].register_forward_hook(
                    lambda module, args, output: replace_hidden(
                        output, position, vector
                    )
                )
            )
        return model(**_inputs(model, input_ids)).logits.detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
