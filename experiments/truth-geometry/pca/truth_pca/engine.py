"""Read only raw decoder-block outputs; do not pool chat suffixes or final norm."""

import torch


def hidden(output):
    return output[0] if isinstance(output, tuple) else output


@torch.inference_mode()
def capture(model, batch, positions, return_final=False):
    acts, handles = {}, []
    row_indices = torch.arange(len(positions), device=batch["input_ids"].device)
    token_indices = torch.tensor(positions, device=row_indices.device)

    def hook(layer):
        def observe(module, args, output):
            value = hidden(output)[row_indices, token_indices]
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Nonfinite residual at block {layer}")
            acts[layer] = value.detach().cpu().clone()

        return observe

    try:
        for layer, block in enumerate(model.model.layers):
            handles.append(block.register_forward_hook(hook(layer)))
        output = model.model(**batch, return_dict=True)
        final = (
            output.last_hidden_state[row_indices, token_indices].detach().cpu()
            if return_final
            else None
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(acts) != set(range(len(model.model.layers))):
        raise RuntimeError("Missing block captures")
    return torch.stack([acts[i] for i in range(len(acts))]), final
