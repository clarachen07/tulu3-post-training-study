"""GPU-resident BF16 trajectories; selected activations stored losslessly in FP32."""
from __future__ import annotations

import os
import time

import numpy as np

from .common import effective_output, positions


def configure(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def load_model(path, config):
    import torch
    from transformers import AutoModelForCausalLM
    configure(config["seed"])
    if config["dtype"] != "bfloat16" or config["device"] != "cuda:0":
        raise ValueError("This protocol requires BF16 on cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen inference requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native BF16 support is required")
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager", low_cpu_mem_usage=True,
        device_map={"": 0})
    model.eval().requires_grad_(False)
    if any(p.dtype != torch.bfloat16 or p.device.type != "cuda" or p.device.index != 0 for p in model.parameters()):
        raise RuntimeError("All BF16 parameters must reside on cuda:0; CPU/disk offload is forbidden")
    if model.config.num_hidden_layers != 32 or model.config.hidden_size != 4096:
        raise RuntimeError("Unexpected model architecture")
    return model


def model_device(model):
    import torch
    if hasattr(model, "hf_device_map"):
        return torch.device("cuda:0")
    return next(model.parameters()).device


def greedy(model, tokenizer, prompt_ids, config):
    """Use only last-position logits; no inherited sampling defaults/processors."""
    import torch
    device = model_device(model)
    eos = int(model.config.eos_token_id)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cache, generated = None, []
    reason = "length"
    started = time.monotonic()
    with torch.inference_mode():
        for step in range(config["max_new_tokens"]):
            total = len(prompt_ids) + step
            out = model.model(input_ids=ids, attention_mask=torch.ones((1, total), dtype=torch.long, device=device),
                              past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            logits = model.lm_head(out.last_hidden_state[:, -1:, :])
            if logits.dtype != torch.bfloat16 or out.last_hidden_state.dtype != torch.bfloat16 or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid or non-BF16 forward/logits")
            token = int(logits[0, -1].argmax())
            generated.append(token)
            if token == eos:
                reason = "eos"
                break
            text = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if config["stop"] in text:
                reason = "stop_string"
                break
            ids = torch.tensor([[token]], dtype=torch.long, device=device)
    text, effective, stopped = effective_output(tokenizer, generated, eos, config["stop"])
    return {"output_token_ids": generated, "effective_token_ids": effective, "continuation": text,
            "finish_reason": reason, "truncated": reason == "length", "output_tokens": len(generated),
            "effective_output_tokens": len(effective), "stop_string_observed": stopped,
            "generation_seconds": time.monotonic() - started}


class Capture:
    def __init__(self, backbone, indices):
        self.backbone = backbone
        self.indices = indices
        self.values = {}
        self.handles = [layer.register_forward_hook(self.hook(i)) for i, layer in enumerate(backbone.layers)]

    def hook(self, i):
        def callback(module, args, output):
            import torch
            x = output[0] if isinstance(output, tuple) else output
            if x.dtype != torch.bfloat16:
                raise RuntimeError("Non-BF16 decoder output")
            self.values[i] = x[0, self.indices, :].detach().to(device="cpu", dtype=torch.float32).clone()
        return callback

    def close(self):
        for handle in self.handles:
            handle.remove()


def extract(model, prompt_ids, effective_ids, *, check_norm=False):
    import torch
    all_positions = positions(len(prompt_ids), len(effective_ids))
    anchors = [i for i, p in enumerate(all_positions) if p is not None]
    indices = [all_positions[i] for i in anchors]
    capture = Capture(model.model, indices)
    device = model_device(model)
    ids = torch.tensor([list(prompt_ids) + list(effective_ids)], dtype=torch.long, device=device)
    error = None
    try:
        with torch.inference_mode():
            out = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
            if sorted(capture.values) != list(range(len(model.model.layers))):
                raise RuntimeError("Missing layer output")
            hidden = torch.stack([capture.values[i] for i in sorted(capture.values)], dim=1)
            if not torch.isfinite(hidden).all():
                raise RuntimeError("Nonfinite activation")
            if check_norm:
                actual = model.model.norm(hidden[:, -1, :][None, :, :].to(device=device, dtype=torch.bfloat16)).detach().float().cpu()[0]
                expected = out.last_hidden_state[0, indices, :].detach().float().cpu()
                torch.testing.assert_close(actual, expected, rtol=2 * torch.finfo(torch.bfloat16).eps, atol=1e-6)
                error = float((actual - expected).abs().max())
        return hidden.numpy(), anchors, all_positions, error
    finally:
        capture.close()


def memory_check(model, tokenizer, sources, config):
    """Worst planned context; synthetic suffix is resource QA, never experimental data."""
    import torch
    longest = max(sources, key=lambda row: (len(row["prompt_token_ids"]), row["id"]))
    filler = tokenizer.encode("0", add_special_tokens=False)
    if not filler or filler[0] == model.config.eos_token_id:
        raise RuntimeError("Invalid resource-check filler token")
    suffix = [int(filler[0])] * config["max_new_tokens"]
    sequence = longest["prompt_token_ids"] + suffix
    if len(sequence) > config["max_context"]:
        raise RuntimeError("Memory-check context exceeds protocol")
    torch.cuda.reset_peak_memory_stats()
    ids = torch.tensor([sequence], dtype=torch.long, device=model_device(model))
    with torch.inference_mode():
        out = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True)
        if out.last_hidden_state.dtype != torch.bfloat16:
            raise RuntimeError("Non-BF16 resource-check output")
        cache = out.past_key_values
        legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
        if any(value.dtype != torch.bfloat16 or value.device.type != "cuda" for pair in legacy for value in pair):
            raise RuntimeError("Non-BF16 or offloaded KV cache")
        torch.cuda.synchronize()
    del out, cache, legacy, ids
    hidden, anchors, pos, norm_error = extract(model, longest["prompt_token_ids"], suffix, check_norm=True)
    if hidden.dtype != np.float32 or anchors != list(range(5)):
        raise RuntimeError("Invalid resource-check activation storage")
    result = {"status": "passed", "source_id": longest["id"], "prompt_tokens": len(longest["prompt_token_ids"]),
              "synthetic_suffix_tokens": len(suffix), "total_tokens": len(sequence), "anchors": anchors,
              "positions": pos, "norm_max_abs": norm_error, "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
              "max_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "experimental_sample": False,
              "meaning": "Resource check only; no synthetic answer or activation enters experiment outputs"}
    del hidden
    torch.cuda.empty_cache()
    return result
