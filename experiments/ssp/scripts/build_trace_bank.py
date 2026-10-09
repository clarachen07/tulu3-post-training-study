#!/usr/bin/env python
from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ssp_tulu.io import atomic_json, read_json, read_jsonl, sha256_file, write_jsonl
from ssp_tulu.segmentation import raw_paragraph_steps, semantic_clean_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--limit-problems", type=int)
    return parser.parse_args()


def flatten_source(config: dict, source: str, tokenizer: Any) -> dict[int, list[dict]]:
    root = Path(config["paths"]["generation_root"])
    max_length = int(config["data"]["max_length"])
    by_problem: dict[int, list[dict]] = collections.defaultdict(list)
    for seed in config["data"]["trace_seeds"]:
        pattern = root / source / f"seed-{seed}/math500"
        paths = sorted(pattern.glob("*.jsonl"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected exactly one JSONL in {pattern}, found {paths}")
        path = paths[0]
        for row in read_jsonl(path):
            arrays = [row.get("code"), row.get("score"), row.get("finish_reason")]
            if not all(isinstance(value, list) for value in arrays):
                raise ValueError(f"Malformed generation arrays in {path}, idx={row.get('idx')}")
            for sample_id, (response, score, finish_reason) in enumerate(zip(*arrays)):
                segmentation = raw_paragraph_steps(response)
                clean_segmentation = semantic_clean_steps(response, minimum_steps=3)
                token_count = len(tokenizer.encode(response, add_special_tokens=False))
                eligible = (
                    bool(str(response).strip())
                    and finish_reason == "stop"
                    and token_count <= max_length
                    and len(segmentation.steps) >= 3
                )
                by_problem[int(row["idx"])].append(
                    {
                        "source": source,
                        "problem_id": int(row["idx"]),
                        "source_seed": int(seed),
                        "sample_id": int(sample_id),
                        "question": row["question"],
                        "response": response,
                        "correct": bool(score),
                        "finish_reason": finish_reason,
                        "token_count_response": token_count,
                        "steps": segmentation.steps,
                        "steps_clean": clean_segmentation.steps,
                        "segmentation_mode": segmentation.mode,
                        "clean_segmentation_mode": clean_segmentation.mode,
                        "eligible": eligible,
                        "source_file": str(path),
                    }
                )
    return by_problem


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config["paths"]["models"]["base"], local_files_only=True, use_fast=True
    )
    sources = config["data"]["trace_sources"]
    banks = {source: flatten_source(config, source, tokenizer) for source in sources}
    common_problems = sorted(set.intersection(*(set(bank) for bank in banks.values())))
    if args.limit_problems:
        common_problems = common_problems[: args.limit_problems]

    paired_records: list[dict] = []
    pairing_modes = collections.Counter()
    excluded = collections.Counter()
    for problem_id in common_problems:
        eligible = {
            source: {
                (row["source_seed"], row["sample_id"]): row
                for row in banks[source][problem_id]
                if row["eligible"]
            }
            for source in sources
        }
        if any(not eligible[source] for source in sources):
            excluded["missing_eligible_source"] += 1
            continue
        common_keys = sorted(set.intersection(*(set(eligible[source]) for source in sources)))
        pairs: list[tuple[dict, dict, str]] = []
        if common_keys:
            key = common_keys[0]
            pairs.append((eligible[sources[0]][key], eligible[sources[1]][key], "exact_seed_sample"))
        else:
            left = sorted(eligible[sources[0]].values(), key=lambda row: (row["source_seed"], row["sample_id"]))
            right = sorted(eligible[sources[1]].values(), key=lambda row: (row["source_seed"], row["sample_id"]))
            pairs.append((left[0], right[0], "rank_fallback"))

        for pair_index, (left, right, mode) in enumerate(pairs):
            pair_id = f"math500-{problem_id:04d}-pair-{pair_index:02d}"
            pairing_modes[mode] += 1
            for row in (left, right):
                paired_records.append(
                    {
                        **{key: value for key, value in row.items() if key != "eligible"},
                        "pair_id": pair_id,
                        "pairing_mode": mode,
                    }
                )

    trace_output = output / "math500_trace_bank.jsonl"
    write_jsonl(trace_output, paired_records)
    source_counts = collections.Counter(row["source"] for row in paired_records)
    problem_count = len({row["problem_id"] for row in paired_records})
    pair_count = len({row["pair_id"] for row in paired_records})
    source_files = sorted({row["source_file"] for row in paired_records})
    manifest = {
        "sources": sources,
        "source_files": {path: sha256_file(path) for path in source_files},
        "problem_count": problem_count,
        "pair_count": pair_count,
        "record_count": len(paired_records),
        "source_counts": dict(source_counts),
        "pairing_modes": dict(pairing_modes),
        "excluded": dict(excluded),
        "eligibility": {
            "finish_reason": "stop",
            "nonempty": True,
            "max_response_tokens": config["data"]["max_length"],
            "minimum_paragraph_steps": 3
        },
        "output": str(trace_output),
    }
    if len(set(source_counts.values())) != 1:
        raise RuntimeError(f"Trace bank is not source-balanced: {source_counts}")
    atomic_json(output / "math500_trace_bank_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
