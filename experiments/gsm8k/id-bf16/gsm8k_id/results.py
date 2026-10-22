"""CPU checks of retained results; never claim to audit absent raw evidence."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from decimal import Decimal
from itertools import product
from pathlib import Path

import numpy as np

from .common import ANCHORS, ROOT, STAGES, digest, read, sample_schedule, sha

METHODS = ("twonn", "correlation", "covariance")
KEY = ("panel", "stage", "anchor", "layer", "method")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(np.isfinite(float(actual)) and np.isclose(float(actual), float(expected), rtol=1e-11, atol=1e-12), message)


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return tuple(int(row[k]) if k in ("anchor", "layer") else row[k] for k in KEY)


def check_hashes(root, files):
    root = Path(root).resolve()
    for rel, expected in files.items():
        path = (root / rel).resolve()
        require(path.is_relative_to(root), f"Invalid manifest path: {rel}")
        require(path.is_file() and sha(path) == expected, f"Published file changed: {rel}")


def verify_results(root):
    root = Path(root)
    reports = root / "reports"
    stats = read(reports / "statistics.json")
    analysis = read(reports / "analysis_manifest.json")
    sources = read(reports / "analysis_sources.json")
    require(digest({k: v for k, v in sources.items() if k != "identity"}) == sources["identity"], "Source inventory identity differs")
    require(stats["source_identity"] == analysis["source_identity"] == sources["identity"], "Analysis identities differ")
    check_hashes(root, analysis["reports"])
    # These are historical artifact checks, not new raw-answer or activation audits.
    publication = reports / "publication.json"
    if publication.exists():
        published = read(publication)
        require(published["source_identity"] == sources["identity"], "Figure identity differs")
        check_hashes(root, published["files"])
    frozen_path = root / "provenance/historical_freeze.json"
    if frozen_path.exists():
        frozen = read(frozen_path)
        require(digest({k: v for k, v in frozen.items() if k != "identity"}) == frozen["identity"] == sources["frozen_identity"], "Historical freeze identity differs")
    audit = read(reports / "independent_audit.json")
    require(audit["status"] == "passed" and audit["source_identity"] == sources["identity"] and audit["checked_records"] == 5276, "Historical audit receipt differs")
    cfg = read(root / "experiment.json")
    require(cfg["inference"]["dtype"] == "bfloat16" and cfg["storage_dtype"] == "float32" and cfg["analysis_dtype"] == "float64", "Precision protocol differs")
    per_question = csv_rows(reports / "per_question.csv")
    grouped = {stage: {} for stage in STAGES}
    for row in per_question:
        require(row["stage"] in grouped, "Unexpected stage")
        stage = grouped[row["stage"]]
        require(row["id"] not in stage, "Duplicate question")
        stage[row["id"]] = row
        require(row["correct"] in ("True", "False") and row["truncated"] in ("True", "False"), "Invalid boolean")
        # Validate retained numeric answers, not the absent answer text.
        correct = bool(row["extracted_answer"]) and Decimal(row["extracted_answer"]) == Decimal(row["gold"].replace(",", ""))
        require(correct == (row["correct"] == "True"), "Retained answer score differs")
        length, effective = int(row["output_tokens"]), int(row["effective_output_tokens"])
        require(0 <= effective <= length <= cfg["inference"]["max_new_tokens"], "Invalid output length")
        require(row["finish_reason"] in ("length", "eos", "stop_string"), "Unexpected finish reason")
        require((row["finish_reason"] == "length") == (row["truncated"] == "True"), "Truncation flag differs")
        if row["truncated"] == "True":
            require(length == cfg["inference"]["max_new_tokens"], "Truncated length differs")
        position = int(row["position_0"])
        for anchor in ANCHORS[1:]:
            expected = position + (anchor * effective + 99) // 100 if effective else None
            actual = int(row[f"position_{anchor}"]) if row[f"position_{anchor}"] else None
            require(actual == expected, "Progress position differs")
    ids = [f"gsm8k-test-{i:05d}" for i in range(cfg["test_size"])]
    require(cfg["test_size"] == 1319 and len(per_question) == 5276, "Question coverage differs")
    for stage, rows in grouped.items():
        require(sorted(rows) == ids, f"Incomplete question panel: {stage}")
        behavior = next(r for r in stats["behavior"] if r["stage"] == stage)
        for metric, field in (("correct", "correct"), ("truncated", "truncated")):
            count = sum(r[field] == "True" for r in rows.values())
            require(count == behavior[metric] == audit["coverage"][stage][metric], f"Behavior count differs: {stage}/{metric}")
        require(behavior["n"] == 1319, "Behavior denominator differs")
        empty = sum(int(r["effective_output_tokens"]) == 0 for r in rows.values())
        require(empty == behavior["empty_output"] == audit["coverage"][stage]["empty_output"], "Empty-output count differs")
        lengths = [int(r["output_tokens"]) for r in rows.values()]
        close(behavior["accuracy"], behavior["correct"] / 1319, "Accuracy differs")
        close(behavior["mean_output_tokens"], np.mean(lengths), "Mean output length differs")
        close(behavior["median_output_tokens"], np.median(lengths), "Median output length differs")
    require(len(stats["behavior"]) == 4, "Behavior matrix differs")
    for ident in ids:
        require(len({grouped[s][ident]["gold"] for s in STAGES}) == 1, "Gold differs across stages")
    nonempty = [i for i in ids if all(int(grouped[s][i]["effective_output_tokens"]) for s in STAGES)]
    complete = [i for i in nonempty if all(grouped[s][i]["truncated"] == "False" for s in STAGES)]
    panels = {"all_input": ids, "all_trajectory": nonempty, "common_complete": complete}
    require([len(panels[p]) for p in panels] == [1319, 1319, 1299], "Frozen panel sizes differ")
    require(sources["input_ids"] == ids and sources["trajectory_ids"] == nonempty and sources["complete_ids"] == complete, "Panel membership differs")
    require({p["panel"] for p in stats["panels"]} == set(panels), "Panel summary differs")
    for panel, selected in panels.items():
        schedule = sample_schedule(len(selected), cfg["statistics"])
        np.testing.assert_array_equal(np.load(reports / f"schedules/{panel}.npy", allow_pickle=False), schedule)
        saved = next(p for p in stats["panels"] if p["panel"] == panel)
        require(saved["n"] == len(selected) and saved["subsample_size"] == schedule.shape[1] and saved["status"] == "ok", "Panel sampling differs")
    summaries = {key(r): r for r in stats["summary"]}
    expected_keys = {(p, s, a, l, m) for p in panels for s, a, l, m in product(STAGES, (0,) if p == "all_input" else ANCHORS, range(1, 33), METHODS)}
    require(set(summaries) == expected_keys and len(summaries) == len(stats["summary"]), "Geometry matrix differs")
    estimates = defaultdict(dict)
    for row in csv_rows(reports / "estimates.csv"):
        k, rep = key(row), int(row["replicate"])
        require(k in summaries and rep not in estimates[k], "Unexpected or duplicate estimate")
        require(row["status"] == "ok", "Nonfinite or failed published estimate")
        require(int(row["n"]) == (summaries[k]["n"] if rep == -1 else summaries[k]["subsample_n"]), "Estimate denominator differs")
        estimates[k][rep] = float(row["value"])
    for k, summary in summaries.items():
        require(set(estimates[k]) == {-1, *range(cfg["statistics"]["repeats"])}, "Replicate coverage differs")
        values = [estimates[k][i] for i in range(cfg["statistics"]["repeats"])]
        require(summary["valid_repeats"] == 20 and summary["n"] == len(panels[k[0]]) and summary["subsample_n"] == int(len(panels[k[0]]) * .9), "Summary denominator differs")
        for field, expected in (("full", estimates[k][-1]), ("mean", np.mean(values)), ("sd", np.std(values, ddof=0))):
            close(summary[field], expected, f"Geometry summary differs: {k}/{field}")
    for filename, originals, fields in (("summary.csv", summaries, KEY), ("behavior.csv", {r["stage"]: r for r in stats["behavior"]}, ("stage",))):
        exported = csv_rows(reports / filename)
        require(len(exported) == len(originals), "CSV coverage differs")
        seen = set()
        for row in exported:
            k = key(row) if fields == KEY else row["stage"]
            require(k not in seen, "Duplicate CSV row")
            seen.add(k)
            for field, actual in row.items():
                expected = originals[k][field]
                if field in fields:
                    require(str(actual) == str(expected), "CSV coordinate differs")
                else:
                    close(actual, expected, "CSV value differs")
    pair_count = 0
    for filename, kind in (("paired_stage_deltas.csv", "stage"), ("paired_progress_deltas.csv", "progress")):
        expected = {}
        for k in expected_keys:
            panel, stage, anchor, layer, method = k
            if kind == "stage" and stage != STAGES[-1]:
                other = STAGES[STAGES.index(stage) + 1]
                target = (panel, other, anchor, layer, method)
                coordinate = (panel, anchor, layer, method, stage, other)
            elif kind == "progress" and panel != "all_input" and anchor != 0:
                target, k = k, (panel, stage, 0, layer, method)
                coordinate = (panel, stage, anchor, layer, method)
            else:
                continue
            expected[coordinate] = np.array([estimates[target][i] - estimates[k][i] for i in range(20)])
        rows = csv_rows(reports / filename)
        require(len(rows) == len(expected), "Paired comparison coverage differs")
        seen = set()
        for row in rows:
            coordinate = (row["panel"], int(row["anchor"]), int(row["layer"]), row["method"], row["from_stage"], row["to_stage"]) if kind == "stage" else (row["panel"], row["stage"], int(row["anchor"]), int(row["layer"]), row["method"])
            require(coordinate in expected and coordinate not in seen and int(row["valid_repeats"]) == 20, "Paired comparison coordinate differs")
            seen.add(coordinate)
            values = expected[coordinate]
            close(row["mean_delta" if kind == "stage" else "mean_delta_from_input"], np.mean(values), "Paired mean differs")
            close(row["sd_delta"], np.std(values, ddof=0), "Paired SD differs")
        pair_count += len(rows)
    shapes = csv_rows(reports / "shapes.csv")
    require(len(shapes) == 120, "Curve shape coverage differs")
    seen = set()
    for row in shapes:
        condition, panel, stage, anchor, method = row["condition"], row["panel"], row["stage"], int(row["anchor"]), row["method"]
        coordinate = (condition, stage, anchor, method)
        require(coordinate not in seen and condition in ("primary", "common_complete"), "Duplicate curve shape")
        seen.add(coordinate)
        require(panel == ("common_complete" if condition == "common_complete" else "all_input" if anchor == 0 else "all_trajectory"), "Curve panel differs")
        means = np.array([summaries[(panel, stage, anchor, l, method)]["mean"] for l in range(1, 33)])
        peak = int(np.argmax(means))
        require(int(row["peak_layer"]) == peak + 1 and int(row["n"]) == len(panels[panel]), "Curve peak differs")
        for field, expected in (("first", means[0]), ("peak", means[peak]), ("last", means[-1]), ("peak_to_last_drop", means[peak] - means[-1]), ("relative_peak_to_last_drop", 1 - means[-1] / means[peak])):
            close(row[field], expected, "Curve shape value differs")
        middle = 9 <= peak + 1 <= 24 and means[peak] > means[0] and means[peak] > means[-1]
        require(row["middle_peak_L9_L24"] == str(bool(middle)), "Middle-peak label differs")
    release_path = root / "provenance/publication_manifest.json"
    if release_path.exists():
        check_hashes(root, read(release_path)["files"])
    return {"status": "passed", "question_records": len(per_question), "geometry_summary_cells": len(summaries), "replicate_estimates": sum(len(v) for v in estimates.values()), "paired_comparisons": pair_count, "common_complete_questions": len(complete), "raw_answers_rescored": False, "activation_geometry_recomputed": False}


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify_results(parser.parse_args().root), indent=2))
