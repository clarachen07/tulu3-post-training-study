#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ssp_tulu.io import atomic_json, read_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty merge output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    base_path = config["paths"]["models"]["base"]
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False)
    merged = merged.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True, use_fast=True)
    tokenizer.save_pretrained(output)
    index_candidates = list(output.glob("*.index.json"))
    atomic_json(
        output / "merge_manifest.json",
        {
            "base_model": base_path,
            "adapter_dir": str(Path(args.adapter_dir).resolve()),
            "adapter_config_sha256": sha256_file(Path(args.adapter_dir) / "adapter_config.json"),
            "weight_index": str(index_candidates[0]) if index_candidates else None,
            "dtype": "bfloat16",
            "safe_merge": True,
        },
    )
    print(f"Merged model written to {output}")


if __name__ == "__main__":
    main()

