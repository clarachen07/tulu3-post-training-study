import argparse
import csv
import json

from .common import ROOT, config, digest, read_jsonl, write_json

STAGES = ["base", "sft", "dpo", "final"]


def select_prompt():
    means = []
    for index, candidate in enumerate(["P0", "P1", "P2", "P3"]):
        metrics = [json.loads((ROOT / f"reports/dev128/{candidate}/{s}.json").read_text())["with_punct"]["metrics"] for s in STAGES]
        means.append({"candidate": candidate, "task1_upos": sum(x["task1_upos"] for x in metrics) / 4, "task3_las": sum(x["task3_las"] for x in metrics) / 4, "index": index})
    winner = max(means, key=lambda x: (x["task1_upos"], x["task3_las"], -x["index"]))
    spec = json.loads((ROOT / "manifests/prompts.json").read_text())[winner["candidate"]]
    result = {"selected": winner["candidate"], "criterion": "mean strict Task1 UPOS, then mean strict LAS, then P0/P1/P2/P3 order", "split": "dev128", "candidates": means, "prompt_spec_sha256": digest(spec)}
    write_json(ROOT / "manifests/selection.json", result)
    return result


def paired_ci(candidate):
    import numpy as np
    rows = {s: read_jsonl(ROOT / f"reports/test/{candidate}/{s}.sentences.jsonl") for s in STAGES}
    ids = [r["sent_id"] for r in rows["base"]]
    assert all([r["sent_id"] for r in rows[s]] == ids for s in STAGES)
    counts = {s: {k: np.array([r["counts"].get(k, 0) for r in rows[s]], dtype=np.float64) for k in ["gold", "task1_upos", "task3_uas", "task3_las"]} for s in STAGES}
    denom = counts["base"]["gold"]
    rng = np.random.default_rng(42)
    samples = {(a, b, k): [] for a, b in zip(STAGES[:-1], STAGES[1:]) for k in ["task1_upos", "task3_uas", "task3_las"]}
    # Chunking avoids allocating 10000 x 2077 all at once.
    for start in range(0, 10000, 100):
        chosen = rng.integers(0, len(ids), size=(100, len(ids)))
        d = denom[chosen].sum(axis=1)
        for (a, b, k), values in samples.items():
            values.extend(((counts[b][k] - counts[a][k])[chosen].sum(axis=1) / d * 100).tolist())
    result = []
    for (a, b, k), values in samples.items():
        result.append({"from": a, "to": b, "metric": k, "delta_percentage_points": (counts[b][k].sum() - counts[a][k].sum()) / denom.sum() * 100, "ci95": np.quantile(values, [0.025, 0.975]).tolist()})
    write_json(ROOT / f"reports/test/{candidate}/paired-bootstrap.json", {"seed": 42, "replicates": 10000, "comparisons": result})
    return result


def report(candidate, phase):
    base = ROOT / "reports/test" / candidate
    summaries = {s: json.loads((base / f"{s}.json").read_text()) for s in STAGES}
    fields = ["stage", "task1_UPOS_strict", "task1_UPOS_compat_extension", "task3_UPOS_strict", "UPOS_author_F1", "UAS_strict", "LAS_strict", "LAS_subtypes_strict", "UAS_author_F1", "LAS_author_F1", "task1_complete_rate", "task3_complete_rate", "truncated_rate", "valid_tree_rate"]
    table = []
    for s, summary in summaries.items():
        m = summary["with_punct"]["metrics"]
        a = summary["author"]["with_punct"]["metrics"]
        d = summary["with_punct"]["diagnostic_counts"]
        n = d["sentences"]
        table.append(dict(zip(fields, [s, m["task1_upos"], summary["with_punct"]["task1_author_extension"]["f1"], m["task3_upos"], a["UPOS"], m["task3_uas"], m["task3_las"], m["task3_las_subtypes"], a["UAS"], a["LAS"], d.get("task1_table_complete", 0) / n, d.get("task3_table_complete", 0) / n, d.get("truncated", 0) / n, d.get("valid_tree", 0) / n])))
    with (base / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    title = "Original-prompt main experiment" if candidate == "P0" else f"Development-selected supplementary experiment {candidate}"
    text = [f"# EWT r2.15: {title} — {'UPOS' if phase == 'upos' else 'Dependency parsing'}", "", "Each model evaluates 2,077 sentences and 25,094 words with frozen weights and BF16, generating all three task tables in a single response.", "", "Values are percentages. Strict scoring uses fixed gold denominators; author-compatible scoring includes the authors' recovery rules.", ""]
    if phase == "upos":
        names = ["stage", "task1_UPOS_strict", "task1_UPOS_compat_extension", "task3_UPOS_strict", "UPOS_author_F1", "task1_complete_rate", "truncated_rate"]
    else:
        names = ["stage", "UAS_strict", "LAS_strict", "LAS_subtypes_strict", "UAS_author_F1", "LAS_author_F1", "task3_complete_rate", "valid_tree_rate"]
    text += ["| " + " | ".join(names) + " |", "| " + " | ".join(["---"] * len(names)) + " |"]
    for row in table:
        text.append("| " + " | ".join(row[k] if k == "stage" else f"{row[k] * 100:.2f}" for k in names) + " |")
    if phase == "dependency":
        ci = paired_ci(candidate)
        text += ["", "## Adjacent-stage differences (percentage points; paired sentence bootstrap 95% CI)", "", "| Comparison | Metric | Difference | 95% CI |", "| --- | --- | --- | --- |"]
        for x in ci:
            lo, hi = x["ci95"]
            text.append(f"| {x['from']} → {x['to']} | {x['metric']} | {x['delta_percentage_points']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    text += ["", "## Interpretation limits", "", "Base→SFT changes both weights and input wrappers, so it cannot isolate a pure SFT effect. The three Tülu stages use the same template; behavioral scores do not directly establish changes in internal syntactic representations. Frozen inference does not establish that the models have never encountered EWT, and this experiment makes no claim of freedom from data contamination.", "", "The original-prompt main experiment and the development-optimized supplementary experiment are reported separately. P3 includes a training example and uses few-shot prompting. All truncations and format failures are retained; unresolved infrastructure failures cannot be marked as complete.", "", "Per-model JSON files include auxiliary scores excluding punctuation, counts of author-compatible recovery operations, and evaluator adaptation exceptions. sentences.jsonl files contain per-sentence strict counts and error diagnostics."]
    (base / f"{phase}.md").write_text("\n".join(text) + "\n")
    print(f"Report ready: {base / (phase + '.md')}", flush=True)
    return table


def comparison(candidate):
    if candidate == "P0":
        write_json(ROOT / "reports/optimized-comparison.json", {"selected": "P0", "reused_original": True, "delta": 0})
        return
    rows = []
    for stage in STAGES:
        p0 = json.loads((ROOT / f"reports/test/P0/{stage}.json").read_text())["with_punct"]["metrics"]
        opt = json.loads((ROOT / f"reports/test/{candidate}/{stage}.json").read_text())["with_punct"]["metrics"]
        rows.append({"stage": stage, **{k + "_delta_pp": (opt[k] - p0[k]) * 100 for k in ["task1_upos", "task3_uas", "task3_las"]}})
    write_json(ROOT / "reports/optimized-comparison.json", {"selected": candidate, "comparisons": rows})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--select", action="store_true")
    p.add_argument("--candidate", default="P0")
    p.add_argument("--phase", choices=["upos", "dependency"], default="upos")
    args = p.parse_args()
    if args.select:
        print(json.dumps(select_prompt(), indent=2))
    else:
        report(args.candidate, args.phase)


if __name__ == "__main__":
    main()
