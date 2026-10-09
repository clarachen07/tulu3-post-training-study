import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def expand_input_paths(patterns: list[str]) -> list[str]:
    paths = []
    for p in patterns:
        matched = glob.glob(p)
        if matched:
            paths.extend(matched)
        else:
            paths.append(p)
    return sorted(set(paths))


def group_by(records: list[dict], keys: list[str]):
    groups = defaultdict(list)
    for r in records:
        group_key = tuple(r.get(k, "") for k in keys)
        groups[group_key].append(r)
    return groups


def compute_metrics(records: list[dict]) -> dict:
    if not records:
        return {}

    problem_ids = sorted({r.get("problem_id") for r in records})
    num_problems = len(problem_ids)
    num_generations = len(records)

    correct_flags = [bool(r.get("is_correct_auto", False)) for r in records]
    num_correct = sum(correct_flags)

    generated_tokens = []
    for r in records:
        try:
            generated_tokens.append(int(r.get("num_generated_tokens", 0)))
        except Exception:
            pass

    manual_check_count = sum(bool(r.get("needs_manual_check", False)) for r in records)

    # pass@1 approximation: sample_id == 0 for each problem.
    by_problem = defaultdict(list)
    for r in records:
        by_problem[r.get("problem_id")].append(r)

    pass1_flags = []
    passk_flags = []

    for pid, rs in by_problem.items():
        rs_sorted = sorted(rs, key=lambda x: int(x.get("sample_id", 0)))
        first = rs_sorted[0]
        pass1_flags.append(bool(first.get("is_correct_auto", False)))
        passk_flags.append(any(bool(x.get("is_correct_auto", False)) for x in rs_sorted))

    return {
        "num_problems": num_problems,
        "num_generations": num_generations,
        "num_correct_auto": num_correct,
        "accuracy_over_generations": num_correct / num_generations if num_generations else 0.0,
        "pass_at_1_approx": sum(pass1_flags) / len(pass1_flags) if pass1_flags else 0.0,
        "pass_at_k_observed": sum(passk_flags) / len(passk_flags) if passk_flags else 0.0,
        "avg_generated_tokens": statistics.mean(generated_tokens) if generated_tokens else 0.0,
        "median_generated_tokens": statistics.median(generated_tokens) if generated_tokens else 0.0,
        "manual_check_count": manual_check_count,
    }


def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_markdown_table(rows: list[dict], columns: list[str]):
    if not rows:
        print("No rows to display.")
        return

    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")

    for row in rows:
        vals = []
        for c in columns:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        print("| " + " | ".join(vals) + " |")


def summarize(records: list[dict], keys: list[str]) -> list[dict]:
    rows = []
    groups = group_by(records, keys)
    for key_tuple, rs in sorted(groups.items()):
        row = {k: v for k, v in zip(keys, key_tuple)}
        row.update(compute_metrics(rs))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-paths", nargs="+", required=True)
    parser.add_argument("--output-csv", default="outputs/summaries/math500_behavior_summary.csv")
    parser.add_argument("--by-level-csv", default="outputs/summaries/math500_behavior_by_level.csv")
    parser.add_argument("--by-subject-csv", default="outputs/summaries/math500_behavior_by_subject.csv")
    args = parser.parse_args()

    paths = expand_input_paths(args.input_paths)
    records = []

    for path in paths:
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"WARNING: input file not found: {path}")
            continue
        file_records = list(read_jsonl(path))
        print(f"Loaded {len(file_records)} records from {path}")
        records.extend(file_records)

    if not records:
        raise RuntimeError("No evaluation records loaded.")

    required = ["model_stage", "problem_id", "sample_id", "is_correct_auto", "needs_manual_check"]
    for field in required:
        if any(field not in r for r in records):
            print(f"WARNING: some records are missing field: {field}")

    summary_rows = summarize(records, ["model_stage"])
    by_level_rows = summarize(records, ["model_stage", "level"])
    by_subject_rows = summarize(records, ["model_stage", "subject"])

    metric_fields = [
        "num_problems",
        "num_generations",
        "num_correct_auto",
        "accuracy_over_generations",
        "pass_at_1_approx",
        "pass_at_k_observed",
        "avg_generated_tokens",
        "median_generated_tokens",
        "manual_check_count",
    ]

    write_csv(args.output_csv, summary_rows, ["model_stage"] + metric_fields)
    write_csv(args.by_level_csv, by_level_rows, ["model_stage", "level"] + metric_fields)
    write_csv(args.by_subject_csv, by_subject_rows, ["model_stage", "subject"] + metric_fields)

    print(f"\nWrote summary CSV to {args.output_csv}")
    print(f"Wrote by-level CSV to {args.by_level_csv}")
    print(f"Wrote by-subject CSV to {args.by_subject_csv}")

    print("\nBehavior summary:")
    print_markdown_table(summary_rows, ["model_stage"] + metric_fields)


if __name__ == "__main__":
    main()
