#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from ssp_tulu.io import atomic_json, read_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-trajectories", type=int, default=200)
    return parser.parse_args()


def final_norm_and_head(model):
    if not hasattr(model, "model") or not hasattr(model.model, "norm"):
        raise TypeError("Expected a Llama-style model.model.norm")
    return model.model.norm, model.lm_head


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    manifest = read_json(Path(args.geometry_dir) / "manifest.json")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    norm, head = final_norm_and_head(model)
    horizons = list(map(int, config["geometry"]["horizons"]))
    final_layer = max(map(int, config["geometry"]["layers"]))
    rows = []
    trajectory_count = 0
    with torch.inference_mode():
        for shard_info in manifest["shards"]:
            records = torch.load(shard_info["path"], map_location="cpu", weights_only=False)
            for record in records:
                if trajectory_count >= args.max_trajectories:
                    break
                z = record["layer_states"][final_layer].to(model.device, dtype=torch.bfloat16)
                for horizon in horizons:
                    if len(z) <= horizon + 1:
                        continue
                    k = 1
                    target_index = k + horizon
                    prediction = z[k] + horizon * (z[k] - z[k - 1])
                    target = z[target_index]
                    prediction_logits = head(norm(prediction[None]))[0].float()
                    target_logits = head(norm(target[None]))[0].float()
                    prediction_logp = F.log_softmax(prediction_logits, dim=-1)
                    target_logp = F.log_softmax(target_logits, dim=-1)
                    target_p = target_logp.exp()
                    kl = torch.sum(target_p * (target_logp - prediction_logp))
                    candidates = [
                        index for index in range(len(z)) if index not in {k - 1, k}
                    ]
                    distances = torch.stack(
                        [torch.sum((prediction - z[index]) ** 2) for index in candidates]
                    )
                    nearest = candidates[int(torch.argmin(distances))]
                    rows.append(
                        {
                            "model": record["model"],
                            "boundary_mode": record["boundary_mode"],
                            "pair_id": record["pair_id"],
                            "problem_id": record["problem_id"],
                            "source": record["source"],
                            "horizon": horizon,
                            "top1_agreement": int(
                                torch.argmax(prediction_logits) == torch.argmax(target_logits)
                            ),
                            "kl_target_to_prediction": float(kl),
                            "step_retrieval_correct": int(nearest == target_index),
                        }
                    )
                trajectory_count += 1
            if trajectory_count >= args.max_trajectories:
                break
    write_jsonl(output / "decode_rows.jsonl", rows)
    summary = {}
    for horizon in horizons:
        selected = [row for row in rows if row["horizon"] == horizon]
        if not selected:
            continue
        summary[str(horizon)] = {
            "n": len(selected),
            "top1_agreement": sum(row["top1_agreement"] for row in selected) / len(selected),
            "mean_kl": sum(row["kl_target_to_prediction"] for row in selected) / len(selected),
            "step_retrieval_accuracy": sum(
                row["step_retrieval_correct"] for row in selected
            )
            / len(selected),
        }
    atomic_json(
        output / "summary.json",
        {
            "model": manifest["model"],
            "boundary_mode": manifest["boundary_mode"],
            "trajectories": trajectory_count,
            "metrics": summary,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

