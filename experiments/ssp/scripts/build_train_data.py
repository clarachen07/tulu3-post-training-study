#!/usr/bin/env python
from __future__ import annotations

import argparse
import collections
import math
from pathlib import Path

from transformers import AutoTokenizer

from ssp_tulu.data import encode_training_trajectory, generate_triples, segment_record_solution
from ssp_tulu.io import (
    atomic_json,
    normalized_text_hash,
    read_json,
    read_jsonl,
    sha256_file,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(config["paths"]["math_data_root"])
    train_path = data_root / "math/train.jsonl"
    test_paths = [data_root / "math/test.jsonl", data_root / "math500/test.jsonl"]
    model_path = config["paths"]["models"]["base"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)

    test_hashes: set[str] = set()
    for path in test_paths:
        test_hashes.update(normalized_text_hash(row["problem"]) for row in read_jsonl(path))

    conditions = list(config["conditions"])
    max_length = int(config["data"]["max_length"])
    step_id = int(config["data"]["step_token_id"])
    records = list(read_jsonl(train_path))
    if args.limit:
        records = records[: args.limit]

    accepted = []
    excluded = collections.Counter()
    segmentation_modes = collections.Counter()
    lengths: list[int] = []
    steps_kept: list[int] = []
    seen_hashes: set[str] = set()
    for index, row in enumerate(records):
        question = row["problem"]
        problem_hash = normalized_text_hash(question)
        if problem_hash in test_hashes:
            excluded["test_overlap"] += 1
            continue
        if problem_hash in seen_hashes:
            excluded["duplicate_train_question"] += 1
            continue
        seen_hashes.add(problem_hash)

        steps, segmentation_mode = segment_record_solution(row)
        encoded = encode_training_trajectory(
            tokenizer,
            question=question,
            steps=steps,
            step_token_id=step_id,
            max_length=max_length,
        )
        if encoded is None:
            excluded["insufficient_complete_steps_or_length"] += 1
            continue
        record_id = str(row.get("unique_id") or f"math-train-{index}")
        triples = {
            condition: generate_triples(
                condition,
                token_count=len(encoded.input_ids),
                boundary_positions=encoded.boundary_positions,
                record_id=record_id,
                seed=int(config["conditions"][condition]["seed"]),
            )
            for condition in conditions
        }
        accepted.append(
            {
                "record_id": record_id,
                "problem_hash": problem_hash,
                "question": question,
                "solution": row["solution"],
                "answer": row.get("answer"),
                "subject": row.get("subject"),
                "level": row.get("level"),
                "segmentation_mode": segmentation_mode,
                "steps_total": len(steps),
                "steps_kept": encoded.steps_kept,
                "input_ids": encoded.input_ids,
                "labels": encoded.labels,
                "boundary_positions": encoded.boundary_positions,
                "triples": triples,
            }
        )
        segmentation_modes[segmentation_mode] += 1
        lengths.append(len(encoded.input_ids))
        steps_kept.append(encoded.steps_kept)

    train_output = output / "math_train_ssp.jsonl"
    write_jsonl(train_output, accepted)
    effective_batch = int(config["training"]["micro_batch_size"]) * int(
        config["training"]["gradient_accumulation_steps"]
    )
    manifest = {
        "source": str(train_path),
        "source_sha256": sha256_file(train_path),
        "test_sources": {str(path): sha256_file(path) for path in test_paths},
        "model_tokenizer": model_path,
        "step_token": config["data"]["step_token"],
        "step_token_id": step_id,
        "max_length": max_length,
        "records_seen": len(records),
        "records_accepted": len(accepted),
        "records_excluded": dict(excluded),
        "segmentation_modes": dict(segmentation_modes),
        "token_length": {
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "mean": sum(lengths) / len(lengths) if lengths else None,
        },
        "steps_kept": {
            "min": min(steps_kept) if steps_kept else None,
            "max": max(steps_kept) if steps_kept else None,
            "mean": sum(steps_kept) / len(steps_kept) if steps_kept else None,
        },
        "effective_batch_size": effective_batch,
        "projected_optimizer_steps": math.ceil(len(accepted) / effective_batch)
        * int(config["training"]["epochs"]),
        "output": str(train_output),
    }
    atomic_json(output / "math_train_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()

