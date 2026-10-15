"""Extract and analyze all frozen stages under one shared GPU lock."""

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import sys

from .common import STAGES, atomic_json, immutable_manifest, inventory, now, rows, sha


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_utils.runtime import gpu_admission, protect_published


def controls(model, records, pad_id, device):
    import torch
    from .engine import capture
    from .inputs import padded_batch

    selected = [
        records[0],
        min(records, key=lambda r: r["token_count"]),
        max(records, key=lambda r: r["token_count"]),
    ]
    used = {r["row_id"] for r in selected}
    selected += [r for r in records if r["row_id"] not in used][:22]
    batch, positions = padded_batch(selected, pad_id, device)
    packed, final = capture(model, batch, positions, return_final=True)
    checks = []
    # Shape-dependent BF16 kernels are not elementwise identities. Eight BF16 epsilons
    # bound each block's relative vector error; a same-shape control below is strict.
    relative_limit = 8 * torch.finfo(torch.bfloat16).eps
    for i, record in enumerate(selected[:3]):
        single, anchor = padded_batch([record], pad_id, device)
        value, _ = capture(model, single, anchor)
        delta = packed[:, i].float() - value[:, 0].float()
        relative = delta.norm(dim=-1) / value[:, 0].float().norm(dim=-1).clamp_min(
            1e-12
        )
        passed = bool((relative <= relative_limit).all())
        checks.append(
            {
                "name": f"batch_vs_single_{i}",
                "row_id": record["row_id"],
                "max_abs_error": float(delta.abs().max()),
                "rms_error": float(delta.square().mean().sqrt()),
                "relative_l2_by_block": relative.tolist(),
                "relative_l2_limit": relative_limit,
                "initial_coordinate_close": bool(
                    torch.allclose(
                        packed[:, i].float(), value[:, 0].float(), atol=0.125, rtol=0.03
                    )
                ),
                "passed": passed,
            }
        )
        repeated = {
            k: (
                v[i : i + 1].repeat(len(selected), 1)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }
        repeated_positions = [positions[i]] * len(selected)
        # Preserve the presence of padding so SDPA takes the same mask/kernel path,
        # including when the selected statement is the longest unpadded example.
        has_padding = bool((batch["attention_mask"] == 0).any())
        if has_padding and bool((repeated["attention_mask"] == 1).all()):
            donor = next(
                j
                for j in range(len(selected))
                if bool((batch["attention_mask"][j] == 0).any())
            )
            dummy = (i + 1) % len(selected)
            for key in ("input_ids", "attention_mask", "position_ids"):
                repeated[key][dummy] = batch[key][donor]
            repeated_positions[dummy] = positions[donor]
        repeated_value, _ = capture(model, repeated, repeated_positions)
        checks.append(
            {
                "name": f"same_shape_example_independence_{i}",
                "row_id": record["row_id"],
                "max_abs_error": float(
                    (packed[:, i].float() - repeated_value[:, i].float()).abs().max()
                ),
                "passed": bool(torch.equal(packed[:, i], repeated_value[:, i])),
            }
        )
    wrong, _ = capture(model, batch, [p - 1 for p in positions])
    relative_wrong = (wrong[0, :3].float() - packed[0, :3].float()).norm(
        dim=-1
    ) / packed[0, :3].float().norm(dim=-1)
    checks.append(
        {
            "name": "negative_control_previous_token_is_detected",
            "relative_l2_first_block": relative_wrong.tolist(),
            "passed": bool((relative_wrong > relative_limit).all()),
        }
    )
    with torch.inference_mode():
        normalized = model.model.norm(packed[-1].to(device)).float().cpu()
    checks.append(
        {
            "name": "last_capture_before_final_rmsnorm",
            "raw_differs_from_final": not torch.equal(packed[-1], final),
            "max_abs_error": float((normalized - final.float()).abs().max()),
            "passed": bool(
                torch.allclose(normalized, final.float(), atol=0.01, rtol=0.01)
            )
            and not torch.equal(packed[-1], final),
        }
    )
    return {
        "passed": all(c["passed"] for c in checks),
        "batch_size": len(selected),
        "checks": checks,
    }


def run_stage(root, config, stage, audit):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from .engine import capture
    from .inputs import padded_batch, persist_inputs
    from .pca import analyze_stage
    from .preflight import validate_stage_assets
    from .raw import save_chunk, validate_chunk

    spec = config["models"][stage]
    validate_stage_assets(stage, spec, audit)
    tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Offset-preserving fast tokenizer required")
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    folder = root / "results" / stage
    folder.mkdir(parents=True, exist_ok=True)
    encoded, inputs = {}, {}
    for name, count in config["datasets"].items():
        encoded[name], inputs[name] = persist_inputs(
            folder / "inputs" / f"{name}.jsonl.gz",
            tokenizer,
            stage,
            rows(root, name, count),
        )
    if stage != "base":
        for previous in ("sft", "dpo", "rlvr"):
            path = root / "results" / previous / "manifest.json"
            if path.exists():
                other = json.loads(path.read_text())
                if (
                    other["inputs"] != inputs
                    or other["chat_template"] != tokenizer.chat_template
                ):
                    raise RuntimeError("Post-training native input sequences differ")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            n: importlib.metadata.version(n)
            for n in config["runtime"]["required_package_versions"]
        },
        "forward_dtype": "bfloat16",
        "pca_dtype": "float32",
        "attention": "sdpa",
        "batch_size": 25,
        "padding": "left",
        "padding_token_id": pad_id,
        "position_ids": "attention_mask.cumsum-1; padding positions zero",
        "tf32": False,
        "use_cache": False,
        "seed": 0,
    }
    identity = {
        "schema_version": 1,
        "stage": stage,
        "model": spec,
        "protocol": config["protocol"],
        "inputs": inputs,
        "chat_template": tokenizer.chat_template,
        "environment": environment,
        "sources": inventory(root),
        "asset_hashes": {
            n: r["sha256"] for n, r in audit["models"][stage]["files"].items()
        },
    }
    manifest = immutable_manifest(folder / "manifest.json", identity)
    signature = manifest["signature"]
    raw_path = folder / "raw_manifest.json"
    raw_complete = False
    if raw_path.exists():
        old = json.loads(raw_path.read_text())
        if old["signature"] != signature:
            raise RuntimeError("Raw manifest identity differs")
        raw_complete = old.get("complete", False)
    records_by_dataset = {}
    # Validate all persisted chunks before resuming or trusting an already-complete raw manifest.
    for name, count in config["datasets"].items():
        raw_folder = root / "raw_activations" / stage / name
        receipts = []
        for start in range(0, count, 25):
            item = validate_chunk(raw_folder, start, min(start + 25, count), signature)
            if item is not None:
                receipts.append(item)
        expected_names = {f"rows_{start:06d}.json" for start in range(0, count, 25)}
        if (
            raw_folder.exists()
            and not {p.name for p in raw_folder.glob("*.json")} <= expected_names
        ):
            raise RuntimeError("Unexpected raw chunk receipt")
        if raw_complete and len(receipts) != (count + 24) // 25:
            raise RuntimeError("Completed raw manifest lacks all chunks")
        records_by_dataset[name] = receipts
    stage_started = time.monotonic()
    if not raw_complete:
        atomic_json(folder / "status.json", {"state": "loading", "time": now()})
        model, loading = AutoModelForCausalLM.from_pretrained(
            spec["path"],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": config["runtime"]["device"]},
            output_loading_info=True,
        )
        atomic_json(folder / "loading.json", {"signature": signature, **loading})
        if any(
            loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")
        ):
            raise RuntimeError("Incomplete checkpoint loading")
        if any(
            not k.endswith(".rotary_emb.inv_freq")
            for k in loading.get("unexpected_keys", [])
        ):
            raise RuntimeError("Unexpected checkpoint tensors")
        model.eval().requires_grad_(False)
        if len(model.model.layers) != 32 or model.config.hidden_size != 4096:
            raise RuntimeError("Frozen architecture differs")
        validation = {"signature": signature, "datasets": {}}
        for name in ("cities", "sp_en_trans", "common_claim_true_false"):
            validation["datasets"][name] = controls(
                model, encoded[name], pad_id, config["runtime"]["device"]
            )
        validation["passed"] = all(v["passed"] for v in validation["datasets"].values())
        atomic_json(folder / "controls.json", validation)
        if not validation["passed"]:
            raise RuntimeError("Real-model extraction controls failed")
        print(f"{stage}: extraction controls passed", flush=True)
        for name, count in config["datasets"].items():
            raw_folder = root / "raw_activations" / stage / name
            raw_folder.mkdir(parents=True, exist_ok=True)
            complete_starts = {r["start"] for r in records_by_dataset[name]}
            for start in range(0, count, 25):
                if start in complete_starts:
                    continue
                batch, positions = padded_batch(
                    encoded[name][start : start + 25],
                    pad_id,
                    config["runtime"]["device"],
                )
                acts, _ = capture(model, batch, positions)
                item = save_chunk(raw_folder, start, acts, signature)
                records_by_dataset[name].append(item)
                del acts, batch
                if start % 500 == 0 or start + 25 >= count:
                    atomic_json(
                        folder / "status.json",
                        {
                            "state": "extracting",
                            "dataset": name,
                            "completed_rows": min(start + 25, count),
                            "dataset_rows": count,
                            "time": now(),
                        },
                    )
            records_by_dataset[name].sort(key=lambda r: r["start"])
            print(f"{stage}: {name} {count} rows captured, all 32 blocks", flush=True)
        validate_stage_assets(stage, spec, audit)
        datasets = {
            name: {"rows": count, "chunks": records_by_dataset[name]}
            for name, count in config["datasets"].items()
        }
        raw_document = {
            "signature": signature,
            "complete": True,
            "stage": stage,
            "datasets": datasets,
            "total_rows": sum(config["datasets"].values()),
            "layers": 32,
            "raw_bytes": sum(
                c["bytes"] for d in datasets.values() for c in d["chunks"]
            ),
            "extraction_seconds": time.monotonic() - stage_started,
            "completed_at": now(),
        }
        atomic_json(raw_path, raw_document)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        validation = json.loads((folder / "controls.json").read_text())
        if validation["signature"] != signature or not validation["passed"]:
            raise RuntimeError("Completed raw data lacks passed extraction controls")
    del encoded
    analyze_stage(root, config, stage, signature)
    validate_stage_assets(stage, spec, audit)
    atomic_json(
        folder / "status.json",
        {
            "state": "complete",
            "complete": True,
            "signature": signature,
            "elapsed_seconds": time.monotonic() - stage_started,
            "time": now(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    args = parser.parse_args()
    root = args.root.resolve()
    protect_published(root)
    config = json.loads((root / "configs/experiment.json").read_text())
    audit = json.loads((root / "artifacts/preflight.json").read_text())
    if not audit.get("passed") or audit["config_sha256"] != sha(
        root / "configs/experiment.json"
    ):
        raise RuntimeError("Successful matching preflight is required")
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stages = STAGES if args.stage == "all" else (args.stage,)
    started = time.monotonic()
    try:
        with gpu_admission(config):
            for stage in stages:
                print(f"{stage}: begin", flush=True)
                run_stage(root, config, stage, audit)
        atomic_json(
            root / "artifacts/status.json",
            {
                "state": "complete",
                "complete": True,
                "elapsed_seconds": time.monotonic() - started,
                "time": now(),
            },
        )
    except BaseException as error:
        atomic_json(
            root / "artifacts/status.json",
            {"state": "failed", "error": repr(error), "time": now()},
        )
        raise


if __name__ == "__main__":
    main()
