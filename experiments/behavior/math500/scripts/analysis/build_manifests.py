#!/usr/bin/env python3
"""Create the frozen per-run manifests from independent verification output."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path


MODELS = {
    "base": "meta-llama/Meta-Llama-3.1-8B",
    "sft": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "dpo": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "rlvr": "allenai/Llama-3.1-Tulu-3-8B",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def parse_pipeline_times(root: Path) -> dict[tuple[str, int], tuple[str, str, str]]:
    sources = {
        2: root / "runs/shared-chat/seed-2-status/pipeline.log",
        3: root / "runs/shared-chat/seed-3-base-sft-status/pipeline.log",
        4: root / "runs/shared-chat/seed-4-sft-dpo-rlvr-status/pipeline.log",
    }
    seed3_dpo_rlvr = root / "runs/shared-chat/seed-3-status/pipeline.log"
    result: dict[tuple[str, int], tuple[str, str, str]] = {}

    def consume(path: Path, seed: int) -> None:
        starts: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 3 or parts[1] not in {"STAGE_START", "STAGE_COMPLETE"}:
                continue
            stamp, event, stage_token, *_ = parts
            stage = stage_token.split("=", 1)[1]
            if event == "STAGE_START":
                starts[stage] = stamp
            elif event == "STAGE_COMPLETE":
                result[(stage, seed)] = (
                    starts[stage],
                    stamp,
                    f"exact pipeline markers in {path.relative_to(root)}",
                )

    for seed, path in sources.items():
        consume(path, seed)
    consume(seed3_dpo_rlvr, 3)
    return result


def parse_seed1_time(root: Path, stage: str) -> tuple[str, str, str]:
    """Recover seed-1 timing from the preserved log when no pipeline marker exists."""
    log_path = root / "runs/shared-chat/logs" / f"{stage}-seed1.log"
    text = log_path.read_text(encoding="utf-8")
    start_match = re.search(
        r"(?:WARNING|INFO) (?P<month>\d{2})-(?P<day>\d{2}) "
        r"(?P<time>\d{2}:\d{2}:\d{2})",
        text,
    )
    duration_matches = re.findall(
        r"Processed prompts: 100%.*?\[(?P<duration>(?:\d+:)?\d{2}:\d{2})<",
        text,
    )
    if not start_match or not duration_matches:
        raise ValueError(f"could not recover seed-1 timing from {log_path}")
    started = datetime.fromisoformat(
        f"2026-{start_match.group('month')}-{start_match.group('day')}"
        f"T{start_match.group('time')}+08:00"
    )
    parts = [int(part) for part in duration_matches[-1].split(":")]
    if len(parts) == 2:
        elapsed = timedelta(minutes=parts[0], seconds=parts[1])
    else:
        elapsed = timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])
    finished_estimate = started + elapsed
    return (
        started.isoformat(),
        finished_estimate.isoformat(),
        "start marker plus final generation-progress duration; excludes short load/evaluation overhead",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    verification = json.loads((root / "results/independent-verification.json").read_text())
    evaluator_hash = sha256(root / "provenance/math_eval.py")
    utils_hash = sha256(root / "provenance/utils.py")
    patch_hash = sha256(root / "provenance/behavior-eval.patch")
    environment_hash = sha256(root / "provenance/environment.txt")
    dataset_hash = sha256(root / "configs/math500-test.jsonl")
    pipeline_times = parse_pipeline_times(root)

    for run_id, run in verification["runs"].items():
        stage = run["stage"]
        seed = run["seed"]
        started_at, completed_at, time_source = (
            parse_seed1_time(root, stage)
            if seed == 1
            else pipeline_times[(stage, seed)]
        )
        model_config_root = root / "provenance/model-configs" / stage
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "protocol": "shared-tulu-sft-template",
            "stage": stage,
            "model_id": MODELS[stage],
            "seed": seed,
            "status": "complete",
            "started_at": started_at,
            "completed_at": completed_at,
            "time_source": time_source,
            "dataset": {
                "name": "MATH500",
                "split": "test",
                "questions": 500,
                "sha256": dataset_hash,
            },
            "sampling": {
                "samples_per_question": 4,
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens_per_call": 16000,
                "max_model_len": 16384,
                "gpu_memory_utilization": 0.85,
                "dtype": "float16",
            },
            "prompt": {
                "type": "plain-boxed",
                "apply_chat_template": True,
                "chat_template_source": "allenai/Llama-3.1-Tulu-3-8B-SFT",
            },
            "code": {
                "upstream_commit": "48f3e35cf9bc5c37ffa54fe2035d0b1e2cd648ad",
                "portable_patch_sha256": patch_hash,
                "portable_math_eval_sha256": evaluator_hash,
                "utils_sha256": utils_hash,
                "environment_sha256": environment_hash,
            },
            "model_files": {
                "config_sha256": sha256(model_config_root / "config.json"),
                "tokenizer_config_sha256": sha256(
                    model_config_root / "tokenizer_config.json"
                ),
                "weights_archived": False,
            },
            "output": {
                "release_asset": "exp1-shared-chat-raw-jsonl.tar.zst",
                "relative_path": run["path"],
                "sha256": run["sha256"],
                "rows": 500,
                "generations": 2000,
                "structurally_valid": True,
            },
            "metrics": {
                "correct_generations": run["correct_generations"],
                "pass_at": run["pass_at"],
                "any_correct_questions": run["any_correct_questions"],
                "empty_generations": run["empty_generations"],
                "empty_parsed_answers": run["empty_parsed_answers"],
                "finish_reasons": run["finish_reasons"],
            },
        }
        destination = root / "runs/shared-chat" / stage / f"seed-{seed}" / "manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
