#!/usr/bin/env python3
"""Select a fixed 20-problem pilot subset from MATH-500."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SEED = 20260514
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "math500_full.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "math500_pilot_20.jsonl"
SUMMARY_PATH = PROJECT_ROOT / "data" / "math500_pilot_20_summary.csv"
README_PATH = PROJECT_ROOT / "data" / "math500_pilot_20_readme.md"

LEVEL_TARGETS = {3: 6, 4: 8, 5: 6}
SUBJECT_TARGETS = {
    "Algebra": 4,
    "Intermediate Algebra": 3,
    "Number Theory": 3,
    "Geometry": 3,
    "Counting & Probability": 3,
    "Precalculus": 2,
    "Prealgebra": 2,
}
OUTPUT_FIELDS = (
    "pilot_id",
    "source_idx",
    "problem_id",
    "problem",
    "solution",
    "answer",
    "subject",
    "level",
    "selection_note",
)
SUMMARY_FIELDS = (
    "pilot_id",
    "source_idx",
    "problem_id",
    "subject",
    "level",
    "answer",
    "selection_note",
    "problem_preview",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}. Run scripts/inspect_math500.py first.")

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


def normalize_level(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Cannot normalize level value: {value!r}")


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["idx"] = int(item["idx"])
        item["level"] = normalize_level(item["level"])
        item["subject"] = str(item["subject"])
        normalized.append(item)
    return normalized


def group_by_subject_level(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["subject"], row["level"])].append(row)
    return grouped


def generate_subject_allocations(
    subject: str,
    subject_quota: int,
    remaining_levels: dict[int, int],
    grouped: dict[tuple[str, int], list[dict[str, Any]]],
    rng: random.Random,
) -> list[dict[int, int]]:
    levels = list(LEVEL_TARGETS)
    options: list[dict[int, int]] = []

    def backtrack(level_index: int, remaining_subject: int, current: dict[int, int]) -> None:
        if level_index == len(levels):
            if remaining_subject == 0:
                options.append(dict(current))
            return

        level = levels[level_index]
        capacity = len(grouped.get((subject, level), []))
        max_count = min(remaining_subject, remaining_levels[level], capacity)
        counts = list(range(max_count, -1, -1))
        rng.shuffle(counts)

        for count in counts:
            current[level] = count
            backtrack(level_index + 1, remaining_subject - count, current)
            current.pop(level, None)

    backtrack(0, subject_quota, {})
    rng.shuffle(options)
    return options


def find_exact_allocation(
    grouped: dict[tuple[str, int], list[dict[str, Any]]],
    rng: random.Random,
) -> dict[str, dict[int, int]] | None:
    subjects = list(SUBJECT_TARGETS)
    rng.shuffle(subjects)
    initial_remaining_levels = dict(LEVEL_TARGETS)

    def backtrack(
        subject_index: int,
        remaining_levels: dict[int, int],
        allocation: dict[str, dict[int, int]],
    ) -> dict[str, dict[int, int]] | None:
        if subject_index == len(subjects):
            if all(count == 0 for count in remaining_levels.values()):
                return allocation
            return None

        subject = subjects[subject_index]
        subject_quota = SUBJECT_TARGETS[subject]
        options = generate_subject_allocations(subject, subject_quota, remaining_levels, grouped, rng)

        for option in options:
            next_remaining_levels = dict(remaining_levels)
            for level, count in option.items():
                next_remaining_levels[level] -= count
            allocation[subject] = option
            result = backtrack(subject_index + 1, next_remaining_levels, allocation)
            if result is not None:
                return result
            allocation.pop(subject, None)

        return None

    return backtrack(0, initial_remaining_levels, {})


def sample_exact_quota(
    rows: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    grouped = group_by_subject_level(rows)
    allocation = find_exact_allocation(grouped, rng)
    if allocation is None:
        return [], ["Exact level and subject quotas were infeasible; fallback sampling was used."], True

    selected: list[dict[str, Any]] = []
    for subject, level_counts in allocation.items():
        for level, count in level_counts.items():
            if count == 0:
                continue
            pool = list(grouped[(subject, level)])
            rng.shuffle(pool)
            for row in pool[:count]:
                item = dict(row)
                item["selection_note"] = "exact quota match: requested subject and level targets"
                selected.append(item)

    rng.shuffle(selected)
    return selected, ["Exact level and subject quotas were satisfied."], False


def sample_fallback(rows: list[dict[str, Any]], rng: random.Random) -> tuple[list[dict[str, Any]], list[str]]:
    total = sum(LEVEL_TARGETS.values())
    remaining_levels = Counter(LEVEL_TARGETS)
    remaining_subjects = Counter(SUBJECT_TARGETS)
    candidates = [
        row
        for row in rows
        if row["level"] in LEVEL_TARGETS and row["subject"] in SUBJECT_TARGETS
    ]
    rng.shuffle(candidates)

    selected: list[dict[str, Any]] = []
    used_problem_ids: set[str] = set()

    for row in candidates:
        if remaining_levels[row["level"]] <= 0 or remaining_subjects[row["subject"]] <= 0:
            continue
        item = dict(row)
        item["selection_note"] = "fallback: preserved both subject and level quota where possible"
        selected.append(item)
        used_problem_ids.add(str(row["problem_id"]))
        remaining_levels[row["level"]] -= 1
        remaining_subjects[row["subject"]] -= 1
        if len(selected) == total:
            return selected, ["Fallback selected 20 rows while preserving all quotas."]

    fill_pool = [row for row in rows if str(row["problem_id"]) not in used_problem_ids]
    rng.shuffle(fill_pool)
    fill_pool.sort(
        key=lambda row: (
            int(remaining_levels[row["level"]] > 0) + int(remaining_subjects[row["subject"]] > 0),
            int(row["level"] in LEVEL_TARGETS),
            int(row["subject"] in SUBJECT_TARGETS),
        ),
        reverse=True,
    )

    for row in fill_pool:
        item = dict(row)
        item["selection_note"] = "fallback: filled remaining pilot slot after quota constraints"
        selected.append(item)
        used_problem_ids.add(str(row["problem_id"]))
        if row["level"] in remaining_levels and remaining_levels[row["level"]] > 0:
            remaining_levels[row["level"]] -= 1
        if row["subject"] in remaining_subjects and remaining_subjects[row["subject"]] > 0:
            remaining_subjects[row["subject"]] -= 1
        if len(selected) == total:
            break

    if len(selected) != total:
        raise ValueError(f"Fallback sampling selected {len(selected)} rows, expected {total}.")

    notes = [
        "Fallback sampling was used because exact quotas could not be met.",
        f"Unmet level quota after fallback: {dict(+remaining_levels)}",
        f"Unmet subject quota after fallback: {dict(+remaining_subjects)}",
    ]
    return selected, notes


def attach_pilot_ids(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for pilot_index, row in enumerate(selected):
        output_row = {
            "pilot_id": f"math500_pilot_{pilot_index:03d}",
            "source_idx": row["idx"],
            "problem_id": str(row["problem_id"]),
            "problem": row.get("problem", ""),
            "solution": row.get("solution", ""),
            "answer": row.get("answer", ""),
            "subject": row["subject"],
            "level": row["level"],
            "selection_note": row["selection_note"],
        }
        output_rows.append({field: output_row[field] for field in OUTPUT_FIELDS})
    return output_rows


def problem_preview(problem: str, max_length: int = 120) -> str:
    compact = " ".join(problem.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for row in rows:
        summary_rows.append(
            {
                "pilot_id": row["pilot_id"],
                "source_idx": row["source_idx"],
                "problem_id": row["problem_id"],
                "subject": row["subject"],
                "level": row["level"],
                "answer": row["answer"],
                "selection_note": row["selection_note"],
                "problem_preview": problem_preview(row["problem"]),
            }
        )
    pd.DataFrame(summary_rows, columns=SUMMARY_FIELDS).to_csv(path, index=False)


def count_values(rows: list[dict[str, Any]], field: str) -> Counter[Any]:
    return Counter(row[field] for row in rows)


def format_counter(counter: Counter[Any], order: list[Any]) -> str:
    lines = []
    for key in order:
        lines.append(f"- {key}: {counter.get(key, 0)}")
    extras = sorted(key for key in counter if key not in set(order))
    for key in extras:
        lines.append(f"- {key}: {counter[key]}")
    return "\n".join(lines)


def write_readme(path: Path, rows: list[dict[str, Any]], notes: list[str], used_fallback: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subject_counts = count_values(rows, "subject")
    level_counts = count_values(rows, "level")
    fallback_status = "Used fallback sampling." if used_fallback else "No fallback was needed."

    text = f"""# MATH-500 Pilot Subset

这个 20 题子集用于答辩前 pilot：验证 Tülu-3-8B 四阶段模型的行为复现、
hidden states 提取、linear correctness probe 和 ∆logp 分析 pipeline 是否能
端到端跑通。它只用于验证实验流程可行性，不用于报告最终结论。

## Source

- Dataset: HuggingFaceH4/MATH-500
- Split: test
- Source file: `{INPUT_PATH.relative_to(PROJECT_ROOT)}`
- Output file: `{OUTPUT_PATH.relative_to(PROJECT_ROOT)}`
- Summary file: `{SUMMARY_PATH.relative_to(PROJECT_ROOT)}`
- Random seed: `{SEED}`

## Requested Level Quota

{format_counter(Counter(LEVEL_TARGETS), list(LEVEL_TARGETS))}

## Actual Level Distribution

{format_counter(level_counts, list(LEVEL_TARGETS))}

## Requested Subject Quota

{format_counter(Counter(SUBJECT_TARGETS), list(SUBJECT_TARGETS))}

## Actual Subject Distribution

{format_counter(subject_counts, list(SUBJECT_TARGETS))}

## Fallback Notes

- {fallback_status}
""" + "".join(f"- {note}\n" for note in notes)

    path.write_text(text, encoding="utf-8")


def main() -> int:
    try:
        rows = normalize_rows(load_jsonl(INPUT_PATH))
        rng = random.Random(SEED)
        selected, notes, used_fallback = sample_exact_quota(rows, rng)
        if used_fallback:
            selected, fallback_notes = sample_fallback(rows, rng)
            notes.extend(fallback_notes)

        output_rows = attach_pilot_ids(selected)
        write_jsonl(OUTPUT_PATH, output_rows)
        write_summary(SUMMARY_PATH, output_rows)
        write_readme(README_PATH, output_rows, notes, used_fallback)

        print(f"Selected {len(output_rows)} pilot examples")
        print(f"Wrote {OUTPUT_PATH}")
        print(f"Wrote {SUMMARY_PATH}")
        print(f"Wrote {README_PATH}")
        print("Subject distribution:")
        for subject, count in count_values(output_rows, "subject").most_common():
            print(f"  {subject}: {count}")
        print("Level distribution:")
        for level, count in sorted(count_values(output_rows, "level").items()):
            print(f"  {level}: {count}")
        return 0
    except Exception as exc:
        print(f"Failed to select MATH-500 pilot subset: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
