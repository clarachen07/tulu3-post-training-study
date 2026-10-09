#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

from ssp_tulu.io import atomic_json, read_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--output", default="artifacts/preflight.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    model_audits = {}
    reference_ids = None
    step_id = int(config["data"]["step_token_id"])
    step_token = config["data"]["step_token"]
    sample = "A fixed tokenizer identity check: $x^2 + y^2 = 1$."
    for stage, path_string in config["paths"]["models"].items():
        path = Path(path_string).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        token_id = tokenizer.convert_tokens_to_ids(step_token)
        if token_id != step_id:
            raise RuntimeError(f"{stage} step token mismatch: {token_id} != {step_id}")
        ids = tokenizer.encode(sample, add_special_tokens=False)
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise RuntimeError(f"Ordinary tokenizer identity mismatch at {stage}")
        model_config = AutoConfig.from_pretrained(path, local_files_only=True)
        model_audits[stage] = {
            "path": str(path),
            "step_token_id": token_id,
            "tokenizer_length": len(tokenizer),
            "vocab_size": model_config.vocab_size,
            "hidden_size": model_config.hidden_size,
            "layers": model_config.num_hidden_layers,
        }
    required = [
        Path(config["paths"]["math_data_root"]) / "math/train.jsonl",
        Path(config["paths"]["math_data_root"]) / "math/test.jsonl",
        Path(config["paths"]["math_data_root"]) / "math500/test.jsonl",
    ]
    for source in config["data"]["trace_sources"]:
        for seed in config["data"]["trace_seeds"]:
            matches = list(
                (
                    Path(config["paths"]["generation_root"])
                    / source
                    / f"seed-{seed}"
                    / "math500"
                ).glob("*.jsonl")
            )
            if len(matches) != 1:
                raise RuntimeError(f"Expected one trace file for {source}/seed-{seed}: {matches}")
            required.extend(matches)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    disk = shutil.disk_usage(Path(config["paths"]["remote_root"]).parent)
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip() or None
    git_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    manifest = {
        "ok": True,
        "models": model_audits,
        "ordinary_token_ids_identical": True,
        "step_token": step_token,
        "step_token_id": step_id,
        "sources": {str(path): sha256_file(path) for path in required},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": gpu_query,
            "disk_free_bytes": disk.free,
            "cwd": os.getcwd(),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
    }
    atomic_json(args.output, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
