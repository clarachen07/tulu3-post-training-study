#!/usr/bin/env python3
"""Download and inspect the HuggingFaceH4/MATH-500 test split."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_NAME = "HuggingFaceH4/MATH-500"
SPLIT = "test"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "math500_full.jsonl"
EXPORT_FIELDS = (
    "idx",
    "problem_id",
    "problem",
    "solution",
    "answer",
    "subject",
    "level",
)


def normalized_value(value: Any, default: str = "N/A") -> Any:
    """Return a printable/exportable value while making missing values explicit."""
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return value


def get_problem_id(example: dict[str, Any], idx: int) -> str:
    problem_id = example.get("problem_id") or example.get("unique_id")
    if problem_id is None or (isinstance(problem_id, str) and not problem_id.strip()):
        return f"math500_{idx:03d}"
    return str(problem_id)


def export_record(example: dict[str, Any], idx: int) -> dict[str, Any]:
    record = {
        "idx": idx,
        "problem_id": get_problem_id(example, idx),
        "problem": normalized_value(example.get("problem"), ""),
        "solution": normalized_value(example.get("solution"), ""),
        "answer": normalized_value(example.get("answer"), ""),
        "subject": normalized_value(example.get("subject")),
        "level": normalized_value(example.get("level")),
    }
    return {field: record[field] for field in EXPORT_FIELDS}


def print_distribution(title: str, counter: Counter[Any]) -> None:
    print(f"\n{title}:")
    for value, count in sorted(counter.items(), key=lambda item: str(item[0])):
        print(f"  {value}: {count}")


def main() -> int:
    try:
        from datasets import load_dataset

        dataset = load_dataset(DATASET_NAME, split=SPLIT)
    except Exception as exc:
        print(f"Failed to load dataset {DATASET_NAME!r} split {SPLIT!r}: {exc}", file=sys.stderr)
        return 1

    try:
        print(f"Total samples: {len(dataset)}")
        print(f"Fields: {dataset.column_names}")

        subjects = Counter(normalized_value(item.get("subject")) for item in dataset)
        levels = Counter(normalized_value(item.get("level")) for item in dataset)

        print_distribution("Subject distribution", subjects)
        print_distribution("Level distribution", levels)

        print("\nFirst 3 samples:")
        for idx, item in enumerate(dataset.select(range(min(3, len(dataset))))):
            print(f"\nSample {idx}:")
            print(f"problem: {normalized_value(item.get('problem'), '')}")
            print(f"answer: {normalized_value(item.get('answer'), '')}")
            print(f"subject: {normalized_value(item.get('subject'))}")
            print(f"level: {normalized_value(item.get('level'))}")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
            for idx, item in enumerate(dataset):
                output_file.write(json.dumps(export_record(item, idx), ensure_ascii=False) + "\n")

        print(f"\nSaved full dataset to: {OUTPUT_PATH}")
    except Exception as exc:
        print(f"Failed while inspecting or saving dataset: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
