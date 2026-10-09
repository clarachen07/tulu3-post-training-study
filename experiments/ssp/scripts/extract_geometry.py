#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ssp_tulu.data import encode_geometry_trajectory
from ssp_tulu.geometry import BoundaryHookCapture
from ssp_tulu.io import atomic_json, read_json, read_jsonl, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-bank", default="data/processed/math500_trace_bank.jsonl")
    parser.add_argument("--boundary-mode", choices=["natural", "reserved", "literal"], required=True)
    parser.add_argument("--boundary-label")
    parser.add_argument("--step-field", default="steps", choices=["steps", "steps_clean"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-pairs", type=int)
    return parser.parse_args()


def save_shard(output: Path, shard_index: int, records: list[dict]) -> dict:
    path = output / f"shard-{shard_index:05d}.pt"
    torch.save(records, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "records": len(records),
        "problem_ids": sorted({record["problem_id"] for record in records}),
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty geometry output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    layers = list(map(int, config["geometry"]["layers"]))
    boundary_label = args.boundary_label or args.boundary_mode
    shard_size = int(config["geometry"]["shard_size"])
    max_pairs = args.max_pairs
    allowed_pairs: set[str] | None = None
    if max_pairs:
        ordered_pairs = []
        for row in read_jsonl(args.trace_bank):
            if row["pair_id"] not in ordered_pairs:
                ordered_pairs.append(row["pair_id"])
        allowed_pairs = set(ordered_pairs[:max_pairs])

    shard: list[dict] = []
    shards: list[dict] = []
    excluded: dict[str, int] = {}
    start = time.time()
    processed = 0
    with BoundaryHookCapture(model, layers) as capture, torch.inference_mode():
        for row in read_jsonl(args.trace_bank):
            if allowed_pairs is not None and row["pair_id"] not in allowed_pairs:
                continue
            encoded = encode_geometry_trajectory(
                tokenizer,
                question=row["question"],
                steps=row[args.step_field],
                max_length=int(config["data"]["max_length"]),
                boundary_mode=args.boundary_mode,
                step_token_id=int(config["data"]["step_token_id"]),
                literal_marker=config["data"]["literal_step_marker"],
            )
            if encoded is None:
                excluded["fewer_than_three_complete_boundaries"] = (
                    excluded.get("fewer_than_three_complete_boundaries", 0) + 1
                )
                continue
            input_ids, boundaries = encoded
            capture.begin(boundaries)
            tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
            attention = torch.ones_like(tensor)
            model(input_ids=tensor, attention_mask=attention, use_cache=False, return_dict=True)
            if set(capture.captured) != set(layers):
                raise RuntimeError(f"Missing captured layers: {set(layers) - set(capture.captured)}")
            shard.append(
                {
                    "model": args.model_name,
                    "boundary_mode": boundary_label,
                    "pair_id": row["pair_id"],
                    "pairing_mode": row["pairing_mode"],
                    "problem_id": row["problem_id"],
                    "source": row["source"],
                    "source_seed": row["source_seed"],
                    "sample_id": row["sample_id"],
                    "correct": row["correct"],
                    "segmentation_mode": row["segmentation_mode"],
                    "input_token_count": len(input_ids),
                    "boundary_positions": boundaries,
                    "layer_states": dict(capture.captured),
                }
            )
            processed += 1
            if len(shard) >= shard_size:
                shards.append(save_shard(output, len(shards), shard))
                shard = []
            if processed % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "geometry_progress",
                            "model": args.model_name,
                            "boundary_mode": boundary_label,
                            "processed": processed,
                            "elapsed_seconds": time.time() - start,
                        }
                    ),
                    flush=True,
                )
    if shard:
        shards.append(save_shard(output, len(shards), shard))
    manifest = {
        "model": args.model_name,
        "model_path": args.model_path,
        "boundary_mode": boundary_label,
        "encoding_boundary_mode": args.boundary_mode,
        "step_field": args.step_field,
        "layers": layers,
        "trace_bank": str(Path(args.trace_bank).resolve()),
        "trace_bank_sha256": sha256_file(args.trace_bank),
        "records": processed,
        "excluded": excluded,
        "elapsed_seconds": time.time() - start,
        "pre_final_rmsnorm": True,
        "shards": shards,
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "complete.json", {"ok": True, "records": processed})
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
