"""Fixed-denominator scores for one full parse and balanced choice diagnostics."""
import argparse
from collections import Counter, defaultdict
import json
import re

from .common import ROOT, config, latest_results, read_jsonl, write_json, write_jsonl
from .protocol import END_PARSE, STOP_STRINGS

UPOS = set("ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split())
DEPREL = set("acl advcl advmod amod appos aux case cc ccomp clf compound conj cop csubj dep det discourse dislocated expl fixed flat goeswith iobj list mark nmod nsubj nummod obj obl orphan parataxis punct reparandum root vocative xcomp".split())
METRICS = ("upos", "uas", "las")


def first_answer(output):
    """Use the earliest declared boundary; never inspect whether an answer is right."""
    matches = [(output.find(stop), stop) for stop in STOP_STRINGS if stop and stop in output]
    if not matches:
        return output, None
    position, stop = min(matches, key=lambda item: item[0])
    return output[:position], stop


def parse_table(output):
    """Blank lines do not split a table; distinct tables are not selected among."""
    tables, current = [], []
    header = ["ID", "FORM", "UPOS", "HEAD", "DEPREL"]
    headers = 0
    for line in output.splitlines():
        if not line.strip():
            continue
        if "\t" in line:
            fields = [field.strip() for field in line.split("\t")]
            if [field.upper() for field in fields] == header:
                headers += 1
                if current:
                    tables.append(current)
                    current = []
                continue
            current.append(fields)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    failure = "missing_table" if not tables else "ambiguous_table" if len(tables) != 1 else None
    return (tables[0] if failure is None else []), failure, len(tables), headers


def tree_diagnostics(predictions, n):
    heads = {int(i): p["head"] for i, p in predictions.items() if p["head"] is not None}
    in_range = all(0 <= head <= n for head in heads.values())
    complete = len(heads) == n and in_range
    roots = sum(head == 0 for head in heads.values())
    cycle = False
    for start in heads:
        seen, current = set(), start
        while current in heads and current != 0:
            if current in seen:
                cycle = True
                break
            seen.add(current)
            current = heads[current]
    return {
        "head_table_complete": complete,
        "roots": roots,
        "head_out_of_range": sum(head < 0 or head > n for head in heads.values()),
        "self_loop_tokens": sum(i == head for i, head in heads.items()),
        "cycle": cycle,
        "multiple_roots": roots > 1,
        "zero_roots": complete and roots == 0,
        "valid_tree": complete and roots == 1 and not cycle,
    }


def score_sentence(gold, output, finish_reason=None, stop_reason=None, ignore_punct=False):
    answer, boundary = first_answer(output)
    rows, failure, table_count, headers = parse_table(answer)
    gold_by_id = {g["id"]: g for g in gold}
    frequencies, indexed, issues = Counter(), {}, Counter()
    for row in rows:
        if len(row) != 5:
            issues["wrong_field_count"] += 1
        if not row or not re.fullmatch(r"[1-9][0-9]*", row[0]):
            issues["invalid_id"] += 1
            continue
        idx = int(row[0])
        frequencies[idx] += 1
        if idx not in gold_by_id:
            issues["out_of_range_id"] += 1
            continue
        if len(row) != 5:
            continue
        if row[1] != gold_by_id[idx]["form"]:
            issues["form_mismatch"] += 1
            continue
        head = int(row[3]) if re.fullmatch(r"-?[0-9]+", row[3]) else None
        valid_head = head is not None and 0 <= head <= len(gold) and head != idx
        coarse = row[4].split(":", 1)[0]
        g = gold_by_id[idx]
        indexed[idx] = {
            "id": idx, "form": row[1], "upos": row[2], "head": head,
            "head_text": row[3], "deprel": row[4], "coarse_deprel": coarse,
            "valid_upos": row[2] in UPOS, "valid_head": valid_head,
            "valid_deprel": coarse in DEPREL,
            "upos_correct": row[2] in UPOS and row[2] == g["upos"],
            "uas_correct": valid_head and head == g["head"],
            "las_correct": valid_head and head == g["head"] and coarse in DEPREL and coarse == g["deprel"].split(":", 1)[0],
        }
    for idx, frequency in frequencies.items():
        if frequency > 1:
            indexed.pop(idx, None)
            issues["duplicate_id"] += frequency
    counts = {key: 0 for key in ("gold", "aligned", "upos", "uas", "las", "valid_upos", "valid_head", "valid_deprel")}
    for g in gold:
        if ignore_punct and g["upos"] == "PUNCT":
            continue
        counts["gold"] += 1
        pred = indexed.get(g["id"])
        if pred is None:
            continue
        counts["aligned"] += 1
        for key in ("valid_upos", "valid_head", "valid_deprel"):
            counts[key] += pred[key]
        for key in METRICS:
            counts[key] += pred[key + "_correct"]
    predictions = {str(idx): pred for idx, pred in indexed.items()}
    diagnostics = {
        "table_failure": failure, "table_count": table_count, "headers": headers,
        "has_header": headers > 0,
        "table_complete": failure is None and not issues and len(indexed) == len(gold),
        "table_issues": dict(issues), "truncated": finish_reason == "length" and boundary is None,
        "finish_reason": finish_reason, "stop_reason": stop_reason,
        "boundary_trimmed": boundary is not None, "boundary": boundary,
        "early_boundary": boundary is not None and boundary != END_PARSE and len(indexed) < len(gold),
        "tree": tree_diagnostics(predictions, len(gold)),
    }
    return {"counts": counts, "diagnostics": diagnostics, "predictions": predictions}


def summarize(rows):
    totals, diagnostics, failures, issues = Counter(), Counter(), Counter(), Counter()
    finish_reasons, stop_reasons, boundaries = Counter(), Counter(), Counter()
    for row in rows:
        totals.update(row["counts"])
        d = row["diagnostics"]
        diagnostics["sentences"] += 1
        for key in ("table_complete", "truncated", "boundary_trimmed", "early_boundary", "has_header"):
            diagnostics[key] += d[key]
        finish_reasons[str(d["finish_reason"])] += 1
        stop_reasons[str(d["stop_reason"])] += 1
        boundaries[str(d["boundary"])] += 1
        for key, value in d["tree"].items():
            diagnostics[key] += value
        if d["table_failure"]:
            failures[d["table_failure"]] += 1
        issues.update(d["table_issues"])
    denom = totals["gold"]
    metrics = {key: totals[key] / denom if denom else None for key in ("upos", "uas", "las", "aligned", "valid_upos", "valid_head", "valid_deprel")}
    for metric in METRICS:
        metrics[metric + "_conditional_on_aligned"] = totals[metric] / totals["aligned"] if totals["aligned"] else None
    return {"counts": dict(totals), "metrics": metrics, "diagnostic_counts": dict(diagnostics), "failure_counts": dict(failures), "issue_counts": dict(issues), "finish_reason_counts": dict(finish_reasons), "stop_reason_counts": dict(stop_reasons), "boundary_counts": dict(boundaries)}


def completed_outputs(path, expected_ids):
    outputs = latest_results(path)
    expected = set(expected_ids)
    if len(expected_ids) != len(expected):
        raise ValueError("duplicate input request IDs")
    missing, extra = expected - outputs.keys(), outputs.keys() - expected
    failed = [key for key in expected & outputs.keys() if outputs[key].get("status") != "ok"]
    if missing or extra or failed:
        raise ValueError(f"Incomplete output {path}: missing={len(missing)}, extra={len(extra)}, failed={len(failed)}")
    return outputs


def score_full(split, stage):
    sentences = read_jsonl(ROOT / f"data/{split}.jsonl")
    outputs = completed_outputs(ROOT / f"outputs/{split}/full/{stage}.jsonl", [s["sent_id"] for s in sentences])
    rows, no_punct_rows = [], []
    for sentence in sentences:
        raw = outputs[sentence["sent_id"]]
        args = (sentence["gold"], raw["output"], raw.get("finish_reason"), raw.get("stop_reason"))
        result = score_sentence(*args)
        no_punct = score_sentence(*args, ignore_punct=True)
        rows.append({"sent_id": sentence["sent_id"], **result, "no_punct_counts": no_punct["counts"]})
        no_punct_rows.append(no_punct)
    summary = {"protocol": "ewt-fair-v2", "stage": stage, "split": split, "with_punct": summarize(rows), "no_punct": summarize(no_punct_rows)}
    dest = ROOT / f"reports/{split}/full"
    write_jsonl(dest / f"{stage}.sentences.jsonl", rows)
    write_json(dest / f"{stage}.json", summary)
    return summary


def cluster_summary(records, replicates=10000, seed=42):
    """Resample whole sentences, retaining all targets and both answer orders."""
    import numpy as np
    if not records:
        return {"requests": 0, "sentences": 0, "micro_accuracy": None, "macro_sentence_accuracy": None, "chance": None, "micro_ci95": None, "macro_sentence_ci95": None}
    groups = defaultdict(list)
    for r in records:
        groups[r["sent_id"]].append(r)
    counts = np.array([[sum(r["correct"] for r in group), len(group), sum(r["chance"] for r in group)] for group in groups.values()], dtype=float)
    macro = counts[:, 0] / counts[:, 1]
    rng = np.random.default_rng(seed)
    micros, macros = [], []
    for start in range(0, replicates, 100):
        chosen = rng.integers(0, len(counts), size=(min(100, replicates - start), len(counts)))
        selected = counts[chosen].sum(axis=1)
        micros.extend((selected[:, 0] / selected[:, 1]).tolist())
        macros.extend(macro[chosen].mean(axis=1).tolist())
    return {
        "requests": len(records), "sentences": len(counts), "correct": sum(r["correct"] for r in records),
        "micro_accuracy": float(counts[:, 0].sum() / counts[:, 1].sum()),
        "macro_sentence_accuracy": float(macro.mean()), "chance": float(counts[:, 2].sum() / counts[:, 1].sum()),
        "micro_ci95": np.quantile(micros, [.025, .975]).tolist(),
        "macro_sentence_ci95": np.quantile(macros, [.025, .975]).tolist(),
    }


def evaluate_choices(sentences, requests, outputs, full_rows, replicates=10000, seed=42, mapping_threshold=.95):
    gold = {s["sent_id"]: {g["id"]: g for g in s["gold"]} for s in sentences}
    full = {r["sent_id"]: r for r in full_rows}
    details = []
    pairs = defaultdict(list)
    for req in requests:
        if req["task"] not in ("upos", "head") or req["kind"] not in ("grammar", "mapping"):
            raise ValueError("unknown diagnostic task or kind")
        candidates = req["candidates"]
        by_code = {str(c["code"]): c for c in candidates}
        if len(by_code) != len(candidates) or not by_code:
            raise ValueError("empty or duplicate candidate codes")
        raw = outputs[req["request_id"]]
        code = raw["output"].strip()
        selected = by_code.get(code)
        if raw.get("valid_code") is False or ("output_ids" in raw and len(raw["output_ids"]) != 1):
            selected = None
        if "output_ids" in raw and "candidate_token_ids" in raw and raw["output_ids"] != [raw["candidate_token_ids"].get(code)]:
            selected = None
        token = gold[req["sent_id"]][req["target_id"]]
        target = req["mapping_target"] if req["kind"] == "mapping" else token[req["task"]]
        prediction = selected["value"] if selected is not None else None
        if req["task"] == "head":
            target = int(target)
            prediction = int(prediction) if prediction is not None else None
        if target not in [int(c["value"]) if req["task"] == "head" else c["value"] for c in candidates]:
            raise ValueError("gold target absent from candidates")
        row = {
            "request_id": req["request_id"], "sent_id": req["sent_id"], "target_id": req["target_id"],
            "task": req["task"], "kind": req["kind"], "order": req["order"], "code": code,
            "prediction": prediction, "target": target, "valid_choice": selected is not None,
            "correct": selected is not None and prediction == target, "chance": 1 / len(candidates),
        }
        details.append(row)
        pairs[(req["sent_id"], req["target_id"], req["task"], req["kind"])].append(row)
    consistency = defaultdict(list)
    for (sent_id, target_id, task, kind), group in pairs.items():
        if len(group) != 2 or len({str(r["order"]) for r in group}) != 2:
            raise ValueError("each diagnostic target/task/kind must have exactly two distinct orders")
        if len({str(r["target"]) for r in group}) != 1:
            raise ValueError("answer mapping changed semantic target across orders")
        consistency[(task, kind)].append({"sent_id": sent_id, "correct": all(r["valid_choice"] for r in group) and group[0]["prediction"] == group[1]["prediction"], "chance": group[0]["chance"], "target_id": target_id})
    summary = {"bootstrap": {"replicates": replicates, "seed": seed, "unit": "sentence"}, "tasks": {}, "mapping_alerts": [], "invalid_choices": sum(not r["valid_choice"] for r in details), "requests": len(details)}
    for task in ("upos", "head"):
        metric = "upos" if task == "upos" else "uas"
        task_summary = {}
        for kind in ("grammar", "mapping"):
            records = [r for r in details if r["task"] == task and r["kind"] == kind]
            item = cluster_summary(records, replicates, seed)
            item["by_order"] = {}
            for order in sorted({str(r["order"]) for r in records}):
                order_records = [r for r in records if str(r["order"]) == order]
                order_summary = cluster_summary(order_records, replicates, seed)
                item["by_order"][order] = order_summary
                if kind == "mapping" and order_summary["micro_accuracy"] < mapping_threshold:
                    summary["mapping_alerts"].append({"task": task, "order": order, "accuracy": order_summary["micro_accuracy"], "threshold": mapping_threshold})
            item["order_consistency"] = cluster_summary(consistency[(task, kind)], replicates, seed)
            task_summary[kind] = item
        full_records, comparison_rows = [], []
        for sentence in sentences:
            sent_id = sentence["sent_id"]
            if sent_id not in full:
                raise ValueError("full-table prediction missing for diagnostic sentence")
            selected = [r for r in details if r["sent_id"] == sent_id and r["task"] == task and r["kind"] == "grammar"]
            observed_ids = {r["target_id"] for r in selected}
            if observed_ids != set(sentence["target_ids"]):
                raise ValueError("diagnostic target IDs do not match frozen sentence targets")
            full_correct = 0
            for target_id in sentence["target_ids"]:
                prediction = full[sent_id]["predictions"].get(str(target_id))
                correct = bool(prediction and prediction[metric + "_correct"])
                full_correct += correct
                full_records.append({"sent_id": sent_id, "correct": correct, "chance": 0})
            comparison_rows.append({"sent_id": sent_id, "gold": len(sentence["target_ids"]), "full_correct": full_correct, "choice_correct": sum(r["correct"] for r in selected) / 2})
        task_summary["full_table_same_targets"] = cluster_summary(full_records, replicates, seed)
        # Random guessing is undefined for unconstrained free-generation tables.
        task_summary["full_table_same_targets"]["chance"] = None
        task_summary["choice_minus_full"] = paired_values(comparison_rows, "full_correct", "choice_correct", replicates, seed)
        task_summary["choice_minus_full"].pop("ci98_333_bonferroni")
        summary["tasks"][task] = task_summary
    return summary, details


def paired_values(rows, before_key, after_key, replicates=10000, seed=42):
    import numpy as np
    if not rows:
        raise ValueError("empty paired comparison")
    values = np.array([[r["gold"], r[after_key] - r[before_key]] for r in rows], dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, replicates, 100):
        indices = rng.integers(0, len(rows), size=(min(100, replicates - start), len(rows)))
        sums = values[indices].sum(axis=1)
        draws.extend((100 * sums[:, 1] / sums[:, 0]).tolist())
    return {"delta_percentage_points": float(100 * values[:, 1].sum() / values[:, 0].sum()), "ci95": np.quantile(draws, [.025, .975]).tolist(), "ci98_333_bonferroni": np.quantile(draws, [.05 / 6, 1 - .05 / 6]).tolist()}


def score_diagnostics(stage):
    cfg = config()
    sentences = read_jsonl(ROOT / "data/diagnostic128.jsonl")
    requests = read_jsonl(ROOT / "data/diagnostic-requests.jsonl")
    outputs = completed_outputs(ROOT / f"outputs/diagnostic/choice/{stage}.jsonl", [r["request_id"] for r in requests])
    full_rows = read_jsonl(ROOT / f"reports/test/full/{stage}.sentences.jsonl")
    summary, details = evaluate_choices(sentences, requests, outputs, full_rows, cfg.get("bootstrap_replicates", 10000), cfg.get("seed", 42), cfg.get("mapping_alert_threshold", .95))
    summary.update({"protocol": "ewt-fair-v2", "stage": stage})
    dest = ROOT / "reports/diagnostic/choice"
    write_json(dest / f"{stage}.json", summary)
    write_jsonl(dest / f"{stage}.requests.jsonl", details)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("base", "sft", "dpo", "final"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    result = score_diagnostics(args.stage) if args.diagnostics else score_full(args.split, args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
