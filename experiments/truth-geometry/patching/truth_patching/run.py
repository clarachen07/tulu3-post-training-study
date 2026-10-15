"""Run the complete, fixed city-pair grid on one frozen checkpoint at a time."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import sys

from .store import PointStore, atomic_json, fingerprint


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_utils.runtime import gpu_admission, protect_published

STAGES = ("base", "sft", "dpo", "rlvr")


def now():
    return datetime.now(timezone.utc).isoformat()


def source_inventory(root):
    items = {}
    for folder in (
        "truth_patching",
        "tests",
        "configs",
        "scripts",
        "third_party",
        "experiment_utils",
    ):
        for path in sorted((root / folder).rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ):
                items[str(path.relative_to(root))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    if not items:
        raise RuntimeError("Missing source files")
    return items


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            key: importlib.metadata.version(key)
            for key in (
                "torch",
                "transformers",
                "tokenizers",
                "accelerate",
                "safetensors",
                "numpy",
            )
        },
        "forward_dtype": "bfloat16",
        "score_dtype": "float32",
        "attention": "sdpa",
        "batch_size": 1,
        "use_cache": False,
        "tf32": False,
        "seed": 0,
    }


def score(logits, prompt, tokenizer, top=False):
    import torch

    vector = logits[0, -1].float()
    if not torch.isfinite(vector).all():
        raise ValueError("Nonfinite logits")
    lp = vector.log_softmax(dim=-1)
    result = {}
    for name, candidates in prompt["candidate_ids"].items():
        t, f = candidates["true"], candidates["false"]
        difference = float(vector[t] - vector[f])
        result[name] = dict(
            difference=difference,
            true_logit=float(vector[t]),
            false_logit=float(vector[f]),
            true_logprob=float(lp[t]),
            false_logprob=float(lp[f]),
            true_prob=float(lp[t].exp()),
            false_prob=float(lp[f].exp()),
            binary_true_prob=float(torch.sigmoid(vector[t] - vector[f])),
        )
    if top:
        values, indices = vector.topk(10)
        result["top_tokens"] = [
            {
                "id": int(index),
                "text": tokenizer.decode([int(index)]),
                "logit": float(value),
                "prob": float(lp[index].exp()),
            }
            for index, value in zip(indices, values)
        ]
    return result


def validate_controls(
    model, false_ids, true_ids, false_logits, true_logits, donor, prompt
):
    import torch
    from .engine import capture, forward

    checks = []
    tolerance = 1e-5

    def check(name, actual, expected):
        error = float((actual - expected).abs().max())
        checks.append(
            {
                "name": name,
                "max_abs_error": error,
                "atol": tolerance,
                "rtol": 0.0,
                "passed": bool(
                    torch.allclose(actual, expected, atol=tolerance, rtol=0)
                ),
            }
        )

    captured, own = capture(model, false_ids)
    check("capture_hooks_preserve_false_logits", captured, false_logits)
    check(
        "observe_hooks_preserve_false_logits",
        forward(model, false_ids, observe=True),
        false_logits,
    )
    # Check representative early/middle/final blocks and all four semantic anchors.
    for layer in (0, 15, 31):
        for label, position in prompt["anchors"].items():
            check(
                f"self_layer_{layer}_{label}",
                forward(model, false_ids, (layer, position, own[layer][0, position])),
                false_logits,
            )
        position = prompt["positions"][0] - 1
        check(
            f"identical_prefix_layer_{layer}",
            forward(model, false_ids, (layer, position, donor[layer][0, position])),
            false_logits,
        )
    for position in prompt["positions"][:-1]:
        check(
            f"last_block_nonreadout_{position}",
            forward(model, false_ids, (31, position, donor[31][0, position]))[:, -1],
            false_logits[:, -1],
        )
    position = prompt["anchors"]["readout"]
    check(
        "last_block_readout_recovers_donor",
        forward(model, false_ids, (31, position, donor[31][0, position]))[:, -1],
        true_logits[:, -1],
    )
    del own
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "comparison": "full vocabulary logits; all positions except explicitly final-position controls",
        "time": now(),
    }


def finalize_existing_grid(folder, manifest, expected):
    """Recover a crash after the last atomic point but before status completion."""
    from .report import _validate_score

    baseline = json.loads((folder / "baseline.json").read_text())
    controls = json.loads((folder / "controls.json").read_text())
    for name, document in (("baseline", baseline), ("controls", controls)):
        if document.get("signature") != manifest["signature"]:
            raise ValueError(f"Complete grid has mismatched {name} identity")
    for label in ("false", "true"):
        _validate_score(baseline[label], f"baseline.{label}")
    checks = controls.get("checks", [])
    if (
        not controls.get("passed")
        or not checks
        or not all(c.get("passed") is True for c in checks)
    ):
        raise ValueError("Complete grid lacks passed controls")
    for layer, position in expected:
        record = json.loads(
            (folder / "points" / PointStore.filename(layer, position)).read_text()
        )
        _validate_score(record["scores"], "saved point")
    status_path = folder / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get("signature", manifest["signature"]) != manifest["signature"]:
        raise ValueError("Complete grid has mismatched status identity")
    if not status.get("complete"):
        status.update(
            complete=True,
            completed=len(expected),
            expected=len(expected),
            state="complete",
            signature=manifest["signature"],
            recovered_at=now(),
            recovered_without_recompute=True,
        )
        atomic_json(status_path, status)


def run_stage(root, config, stage, audit, resume):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import capture, forward
    from .preflight import validate_stage_assets
    from .prompts import build_prompt

    spec = config["models"][stage]
    validate_stage_assets(stage, spec, audit)
    if (
        audit.get("config_sha256")
        != hashlib.sha256((root / "configs/experiment.json").read_bytes()).hexdigest()
    ):
        raise RuntimeError("Config differs from the audited config")
    tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True)
    prompt = build_prompt(tokenizer, stage)
    if stage != "base":
        for previous in ("sft", "dpo", "rlvr"):
            path = root / "results" / previous / "manifest.json"
            if path.exists():
                other = json.loads(path.read_text())["prompt"]
                for key in ("false_ids", "true_ids", "chat_template"):
                    if prompt[key] != other[key]:
                        raise ValueError(
                            "Post-training stages do not share identical native inputs"
                        )
    identity = {
        "stage": stage,
        "model": spec,
        "prompt": prompt,
        "environment": environment(),
        "sources": source_inventory(root),
        "asset_hashes": {
            name: rec["sha256"] for name, rec in audit["models"][stage]["files"].items()
        },
    }
    manifest = dict(identity, signature=fingerprint(identity), schema_version=1)
    folder = root / "results" / stage
    if (folder / "manifest.json").exists() and not resume:
        raise ValueError("Existing run requires --resume")
    store = PointStore(folder, manifest)
    expected = {
        (layer, position) for layer in range(32) for position in prompt["positions"]
    }
    if not store.completed <= expected:
        raise ValueError("Saved point outside the planned grid")
    if store.completed == expected:
        finalize_existing_grid(folder, manifest, expected)
        print(f"{stage}: already complete, identity verified", flush=True)
        return
    stage_start = time.monotonic()
    atomic_json(
        folder / "status.json",
        {
            "complete": False,
            "completed": len(store.completed),
            "expected": len(expected),
            "state": "loading",
            "time": now(),
        },
    )
    print(f"{stage}: loading frozen BF16 checkpoint", flush=True)
    model, loading = AutoModelForCausalLM.from_pretrained(
        spec["path"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
        output_loading_info=True,
    )
    atomic_json(
        folder / "loading.json", {"signature": manifest["signature"], **loading}
    )
    if (
        loading.get("missing_keys")
        or loading.get("mismatched_keys")
        or loading.get("error_msgs")
    ):
        raise RuntimeError(f"{stage}: incomplete or mismatched checkpoint loading")
    if any(
        not key.endswith(".rotary_emb.inv_freq")
        for key in loading.get("unexpected_keys", [])
    ):
        raise RuntimeError(f"{stage}: unexpected checkpoint tensors require review")
    model.eval().requires_grad_(False)
    if len(model.model.layers) != 32 or model.config.hidden_size != 4096:
        raise ValueError("Unexpected architecture")
    if str(next(model.parameters()).dtype) != "torch.bfloat16":
        raise ValueError("Model is not BF16")
    false_ids = torch.tensor([prompt["false_ids"]], dtype=torch.long)
    true_ids = torch.tensor([prompt["true_ids"]], dtype=torch.long)
    false_logits = forward(model, false_ids)
    true_logits, donor = capture(model, true_ids)
    baseline = {
        "signature": manifest["signature"],
        "false": score(false_logits, prompt, tokenizer, top=True),
        "true": score(true_logits, prompt, tokenizer, top=True),
    }
    baseline_path = folder / "baseline.json"
    if baseline_path.exists() and json.loads(baseline_path.read_text()) != baseline:
        raise ValueError("Resume baseline changed; refusing to mix observations")
    atomic_json(baseline_path, baseline)
    controls = validate_controls(
        model, false_ids, true_ids, false_logits, true_logits, donor, prompt
    )
    controls["signature"] = manifest["signature"]
    atomic_json(folder / "controls.json", controls)
    if not controls["passed"]:
        raise RuntimeError(f"{stage}: failed controls; refusing formal scan")
    print(
        f"{stage}: {len(controls['checks'])} controls passed; starting {len(expected)} cells",
        flush=True,
    )
    for layer in range(32):
        for position in prompt["positions"]:
            if (layer, position) in store.completed:
                continue
            started = time.monotonic()
            logits = forward(
                model, false_ids, (layer, position, donor[layer][0, position])
            )
            result = score(logits, prompt, tokenizer)
            store.write(
                {
                    "signature": manifest["signature"],
                    "stage": stage,
                    "layer": layer,
                    "position": position,
                    "scores": result,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
        atomic_json(
            folder / "status.json",
            {
                "complete": False,
                "completed": len(store.completed),
                "expected": len(expected),
                "state": "scanning",
                "last_layer": layer,
                "elapsed_seconds": time.monotonic() - stage_start,
                "time": now(),
            },
        )
        print(f"{stage}: {len(store.completed)}/{len(expected)} cells", flush=True)
    validate_stage_assets(stage, spec, audit)
    atomic_json(
        folder / "status.json",
        {
            "complete": True,
            "completed": len(store.completed),
            "expected": len(expected),
            "state": "complete",
            "signature": manifest["signature"],
            "elapsed_seconds": time.monotonic() - stage_start,
            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
            "time": now(),
        },
    )
    del model, donor, false_logits, true_logits, logits
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{stage}: complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    protect_published(root)
    config = json.loads((root / "configs/experiment.json").read_text())
    audit = json.loads((root / "artifacts/preflight.json").read_text())
    if not audit.get("passed"):
        raise RuntimeError("Successful full preflight required")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stages = STAGES if args.stage == "all" else (args.stage,)
    started = time.monotonic()
    try:
        with gpu_admission(config):
            for stage in stages:
                torch.cuda.reset_peak_memory_stats()
                run_stage(root, config, stage, audit, args.resume)
        atomic_json(
            root / "artifacts/status.json",
            {
                "state": "complete",
                "stages": stages,
                "elapsed_seconds": time.monotonic() - started,
                "time": now(),
            },
        )
    except BaseException as error:
        atomic_json(
            root / "artifacts/status.json",
            {
                "state": "failed",
                "error": repr(error),
                "elapsed_seconds": time.monotonic() - started,
                "time": now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
