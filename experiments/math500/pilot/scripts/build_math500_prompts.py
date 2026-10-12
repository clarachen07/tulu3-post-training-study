#!/usr/bin/env python3
"""Build chain-of-thought prompts for the MATH-500 pilot subset."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "math500_pilot_20.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "math500_pilot_20_prompts.jsonl"
PROMPT_PATH = PROJECT_ROOT / "prompts" / "math_cot_prompt.txt"
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. Put the final answer in \\boxed{}.\n\n"
    "Problem:\n"
    "{problem}"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}. Run scripts/select_math500_pilot.py first.")

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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(problem: str) -> str:
    return PROMPT_TEMPLATE.replace("{problem}", problem)


def build_prompt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt_rows: list[dict[str, Any]] = []
    for row in rows:
        problem = str(row.get("problem", "")).strip()
        if not problem:
            raise ValueError(f"Empty problem for pilot_id={row.get('pilot_id')!r}")
        prompt_row = dict(row)
        prompt_row["prompt"] = build_prompt(problem)
        prompt_rows.append(prompt_row)
    return prompt_rows


def write_prompt_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PROMPT_TEMPLATE, encoding="utf-8")


def main() -> int:
    try:
        rows = load_jsonl(INPUT_PATH)
        prompt_rows = build_prompt_rows(rows)
        write_jsonl(OUTPUT_PATH, prompt_rows)
        write_prompt_template(PROMPT_PATH)

        print(f"Built {len(prompt_rows)} prompts")
        print(f"Wrote {OUTPUT_PATH}")
        print(f"Wrote {PROMPT_PATH}")
        return 0
    except Exception as exc:
        print(f"Failed to build MATH-500 pilot prompts: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
