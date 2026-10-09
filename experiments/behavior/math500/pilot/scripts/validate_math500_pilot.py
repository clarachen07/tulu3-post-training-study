#!/usr/bin/env python3
"""Validate the prepared MATH-500 pilot prompt file."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = PROJECT_ROOT / "data" / "math500_pilot_20_prompts.jsonl"
EXPECTED_COUNT = 20
REQUIRED_NONEMPTY_FIELDS = ("problem", "answer", "subject", "level", "prompt")
UNIQUE_FIELDS = ("pilot_id", "problem_id")
BOXED_MARKER = "\\boxed{}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}. Run scripts/build_math500_prompts.py first.")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def require_count(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} rows, found {len(rows)}.")


def require_unique(rows: list[dict[str, Any]], field: str) -> None:
    values = [row.get(field) for row in rows]
    missing_positions = [idx for idx, value in enumerate(values) if is_empty(value)]
    if missing_positions:
        raise ValueError(f"Missing {field} at row indices: {missing_positions}")

    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate {field} values: {duplicates}")


def require_nonempty_fields(rows: list[dict[str, Any]]) -> None:
    failures: list[str] = []
    for idx, row in enumerate(rows):
        for field in REQUIRED_NONEMPTY_FIELDS:
            if is_empty(row.get(field)):
                failures.append(f"row {idx}: {field}")
    if failures:
        raise ValueError("Empty required fields found: " + ", ".join(failures))


def require_prompt_marker(rows: list[dict[str, Any]]) -> None:
    failures = [
        row.get("pilot_id", f"row_{idx}")
        for idx, row in enumerate(rows)
        if BOXED_MARKER not in str(row.get("prompt", ""))
    ]
    if failures:
        raise ValueError(f"Prompts missing {BOXED_MARKER!r}: {failures}")


def print_distribution(rows: list[dict[str, Any]], field: str) -> None:
    print(f"{field} distribution:")
    for value, count in sorted(Counter(row[field] for row in rows).items(), key=lambda item: str(item[0])):
        print(f"  {value}: {count}")


def validate(rows: list[dict[str, Any]]) -> None:
    require_count(rows)
    for field in UNIQUE_FIELDS:
        require_unique(rows, field)
    require_nonempty_fields(rows)
    require_prompt_marker(rows)


def main() -> int:
    rows = load_jsonl(PROMPTS_PATH)
    validate(rows)
    print_distribution(rows, "subject")
    print_distribution(rows, "level")
    print("Validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
