"""Recompute descriptive and paired statistics from committed raw outputs."""
import csv
import json
from pathlib import Path
import numpy as np
from format_policy import answer_only
from common import ROOT, digest_file, load_json, read_jsonl, score, utc, write_json

STAGES = ["base", "sft", "dpo", "rlvr"]


def load_results():
    expected = {r["id"]: r for r in read_jsonl(ROOT / "data/prepared/formal.jsonl")}
    results = {}
    sources = {}
    for stage in STAGES:
        panel = {}
        for file in sorted((ROOT / f"outputs/formal/{stage}").glob("batch_*.jsonl")):
            sources[str(file.relative_to(ROOT))] = digest_file(file)
            for row in read_jsonl(file):
                assert row["id"] not in panel
                assert row["stage"] == stage and row["phase"] == "formal"
                assert row["gold"] == expected[row["id"]]["gold"]
                answer, correct = score(row["continuation"], row["gold"])
                assert answer == row["extracted_answer"] and correct == row["correct"]
                panel[row["id"]] = row
        assert set(panel) == set(expected), (stage, len(panel), len(expected))
        results[stage] = panel
    return expected, results, sources


def bootstrap(matrix, count, seed):
    rng = np.random.default_rng(seed)
    estimates = []
    for offset in range(0, count, 500):
        k = min(500, count - offset)
        indices = rng.integers(0, len(matrix), size=(k, len(matrix)))
        estimates.append(matrix[indices].mean(axis=1))
    estimates = np.concatenate(estimates)
    return np.quantile(estimates, [0.025, 0.975], axis=0), np.quantile(np.diff(estimates, axis=1), [0.025, 0.975], axis=0)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = load_json(ROOT / "experiment.json")
    expected, results, sources = load_results()
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    summaries, pairs, examples = [], [], []
    for si, split in enumerate(["train", "test"]):
        ids = sorted(i for i, r in expected.items() if r["split"] == split)
        matrix = np.array([[results[s][i]["correct"] for s in STAGES] for i in ids], dtype=np.int8)
        ci, delta_ci = bootstrap(matrix, cfg["bootstrap_replicates"], cfg["bootstrap_seed"] + si)
        for j, stage in enumerate(STAGES):
            rows = [results[stage][i] for i in ids]
            summaries.append({"split": split, "stage": stage, "n": len(ids),
                "correct": int(matrix[:, j].sum()), "accuracy": float(matrix[:, j].mean()),
                "ci_low": float(ci[0, j]), "ci_high": float(ci[1, j]),
                "answer_only": sum(answer_only(r["continuation"]) for r in rows),
                "answer_only_rate": sum(answer_only(r["continuation"]) for r in rows) / len(rows),
                "no_answer": sum(r["extracted_answer"] is None for r in rows),
                "no_answer_rate": sum(r["extracted_answer"] is None for r in rows) / len(rows),
                "truncated": sum(r["truncated"] for r in rows),
                "truncation_rate": sum(r["truncated"] for r in rows) / len(rows),
                "mean_output_tokens": float(np.mean([r["output_tokens"] for r in rows])),
                "median_output_tokens": float(np.median([r["output_tokens"] for r in rows]))})
        for j in range(3):
            a, b = matrix[:, j], matrix[:, j + 1]
            pair = {"split": split, "from_stage": STAGES[j], "to_stage": STAGES[j + 1], "n": len(ids),
                    "wrong_to_correct": int(((a == 0) & (b == 1)).sum()),
                    "correct_to_wrong": int(((a == 1) & (b == 0)).sum()),
                    "both_correct": int(((a == 1) & (b == 1)).sum()),
                    "both_wrong": int(((a == 0) & (b == 0)).sum()),
                    "delta": float((b - a).mean()), "ci_low": float(delta_ci[0, j]), "ci_high": float(delta_ci[1, j])}
            assert sum(pair[k] for k in ["wrong_to_correct", "correct_to_wrong", "both_correct", "both_wrong"]) == len(ids)
            pairs.append(pair)
            for direction, mask in [("wrong_to_correct", (a == 0) & (b == 1)), ("correct_to_wrong", (a == 1) & (b == 0))]:
                for idx in np.flatnonzero(mask)[:2]:
                    ident = ids[int(idx)]
                    examples.append({"id": ident, "split": split, "pair": STAGES[j] + "→" + STAGES[j + 1],
                                     "direction": direction, "question": expected[ident]["question"],
                                     "gold": expected[ident]["gold"],
                                     "responses": {s: results[s][ident] for s in STAGES}})
    timing = []
    for stage in STAGES:
        jobs = [load_json(p) for p in (ROOT / "runs").glob("*.json")]
        jobs = [j for j in jobs if j["phase"] == "formal" and j["stage"] == stage]
        batch_ids = {r["batch_id"] for r in results[stage].values()}
        metas = [load_json(ROOT / f"outputs/formal/{stage}/batch_{bid}.meta.json") for bid in batch_ids]
        timing.append({"stage": stage, "allocated_process_seconds": sum(j["elapsed_seconds"] for j in jobs),
                       "committed_generation_seconds": sum(m["generation_seconds"] for m in metas),
                       "output_tokens": sum(m["output_tokens"] for m in metas),
                       "failed_attempts": sum(j["status"] == "failed" for j in jobs)})
    write_csv(output / "stage_summary.csv", summaries)
    write_csv(output / "paired_changes.csv", pairs)
    write_csv(output / "timing.csv", timing)
    write_json(output / "examples.json", examples)
    write_json(output / "statistics.json", {"created_utc": utc(), "formal_count": sum(len(x) for x in results.values()),
               "summary": summaries, "pairs": pairs, "timing": timing,
               "bootstrap": {"replicates": cfg["bootstrap_replicates"], "seed": cfg["bootstrap_seed"],
                             "unit": "paired question within each split", "method": "percentile 95% intervals; descriptive, no familywise significance claim"},
               "sources": sources, "numpy_version": np.__version__})
    write_csv(output / "per_question.csv", [
        {"id": i, "split": expected[i]["split"], "rlvr_row": expected[i]["rlvr_row"], "gold": expected[i]["gold"],
         **{s + "_correct": results[s][i]["correct"] for s in STAGES},
         **{s + "_answer": results[s][i]["extracted_answer"] for s in STAGES},
         **{s + "_answer_only": answer_only(results[s][i]["continuation"]) for s in STAGES}}
        for i in sorted(expected)])
    print(json.dumps({"formal_count": 9372, "summary": summaries, "pairs": pairs}), flush=True)


if __name__ == "__main__":
    main()
