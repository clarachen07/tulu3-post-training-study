"""Select common output budgets using completeness, never answer accuracy."""
from __future__ import annotations

from collections import defaultdict
import html
import json
from pathlib import Path

import numpy as np

from .common import (ROOT, STAGES, CONDITIONS, atomic_json, fingerprint, load_config,
                     physical_wrappers, read_jsonl, sha256_file, utc_now)
from .freeze import validate_preflight
from .run import output_directory, input_path, batch_groups, batch_paths, verify_batch


def budget_decision(rows, budgets, candidates):
    """A missing/incomplete final answer at a hard limit triggers the next shared tier."""
    next_budgets = dict(budgets)
    exhausted, increased = [], []
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        if not subset:
            raise ValueError(f"Missing pilot condition: {condition}")
        if any(row["budget"] != budgets[condition] for row in subset):
            raise ValueError("Pilot panel contains mixed budgets")
        # This decision deliberately cannot inspect answer_correct or proof_success.
        if any(row["budget_incomplete"] for row in subset):
            index = candidates[condition].index(budgets[condition])
            if index + 1 == len(candidates[condition]):
                exhausted.append(condition)
            else:
                next_budgets[condition] = candidates[condition][index + 1]
                increased.append(condition)
    state = "needs_diagnosis" if exhausted else ("increase_budget" if increased else "passed")
    return {"status": state, "next_budgets": next_budgets,
            "exhausted_conditions": exhausted, "increased_conditions": increased}


def collect_panel(root=ROOT, batch_size=4, budgets=None):
    root = Path(root)
    config = load_config(root)
    pre = validate_preflight(root)
    records, evidence, seconds = [], {}, defaultdict(float)
    for stage in STAGES:
        for condition in CONDITIONS:
            for wrapper in physical_wrappers(stage):
                sources = read_jsonl(input_path(root, "pilot", stage, condition, wrapper))
                identity = dict(phase="pilot", stage=stage, condition=condition, wrapper=wrapper,
                                budget=budgets[condition], batch_size=batch_size, run_digest=pre["digest"])
                directory = output_directory(root, "pilot", stage, condition, wrapper,
                                             budgets[condition], batch_size, pre["digest"])
                for start, group in batch_groups(sources, config["inference"]["write_batch_size"]):
                    file, meta_path = batch_paths(directory, start)
                    rows = verify_batch(file, meta_path, group, identity)
                    if rows is None:
                        raise RuntimeError(f"Incomplete pilot panel: {file}")
                    records.extend(rows)
                    for path in (file, meta_path):
                        evidence[str(path.relative_to(root))] = sha256_file(path)
                    meta = json.loads(meta_path.read_text())
                    seconds[(stage, condition, wrapper)] += meta["generation_seconds"]
    expected = config["expected"]["physical_generations_pilot_initial"]
    if len(records) != expected or len({r["generation_id"] for r in records}) != expected:
        raise RuntimeError("Pilot panel has missing or duplicate generations")
    return records, evidence, seconds


def summarize(root=ROOT, batch_size=4, budgets=None):
    root = Path(root)
    config = load_config(root)
    pre = validate_preflight(root)
    records, evidence, seconds = collect_panel(root, batch_size, budgets)
    decision = budget_decision(records, budgets, config["inference"]["budget_candidates"])
    groups = defaultdict(list)
    for row in records:
        groups[(row["stage"], row["condition"], row["wrapper"], row["hop"])].append(row)
    cells = []
    for (stage, condition, wrapper, hop), rows in sorted(groups.items()):
        lengths = [r["output_tokens"] for r in rows]
        cells.append(dict(stage=stage, condition=condition, wrapper=wrapper, hop=hop, n=len(rows),
                          budget=budgets[condition],
                          truncated=sum(r["truncated"] for r in rows),
                          budget_incomplete=sum(r["budget_incomplete"] for r in rows),
                          missing_answer=sum(r["scoring"]["extracted_answer"] is None for r in rows),
                          repeated=sum(r["repetition"]["repetition_detected"] for r in rows),
                          format_compliant=sum(r["scoring"]["format_compliant"] for r in rows),
                          output_tokens_mean=float(np.mean(lengths)),
                          output_tokens_max=max(lengths), output_tokens_p95=float(np.percentile(lengths, 95)),
                          proof_parseable=(sum(bool(r["scoring"]["proof_parseable"]) for r in rows)
                                           if condition == "cot" else None)))
    # Estimate only; the pilot's length distribution and the queue can differ.
    ratio = config["expected"]["formal_items"] / config["expected"]["pilot_items"]
    estimated_hours = sum(seconds.values()) * ratio / 3600
    failure_examples = [dict(sample_id=r["sample_id"], hop=r["hop"], stage=r["stage"],
                             condition=r["condition"], wrapper=r["wrapper"], budget=r["budget"],
                             continuation=r["continuation"], output_tokens=r["output_tokens"],
                             scoring=r["scoring"], repetition=r["repetition"],
                             generation_id=r["generation_id"])
                        for r in records if r["budget_incomplete"]]
    summary = dict(preflight_digest=pre["digest"], created_utc=utc_now(), batch_size=batch_size,
                   budgets=budgets, physical_generations=len(records), decision=decision,
                   cells=cells, evidence_files=evidence, budget_failure_examples=failure_examples,
                   selected_panel_generation_seconds=sum(seconds.values()),
                   estimated_formal_gpu_hours=estimated_hours,
                   estimate_caveat="Pilot-based estimate excludes loading, queuing, retries, audits, and reporting.",
                   selection_policy="Completeness only; never select using answer accuracy or proof correctness.")
    panel_id = fingerprint([pre["digest"], batch_size, budgets])[:20]
    path = root / f"reports/pilot/{panel_id}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["evidence_files"] != evidence:
            raise RuntimeError("Pilot report evidence changed")
        summary = existing
    else:
        atomic_json(path, summary)
    atomic_json(root / "reports/pilot/latest.json", summary)
    render_pilot(root, summary)
    return summary


def render_pilot(root, summary):
    headers = ["Stage", "Condition", "Template", "Hop", "N", "Budget", "Truncated",
               "Incomplete at cap", "Missing answer", "Repeated", "Mean output", "Max output"]
    rows = []
    for c in summary["cells"]:
        values = [c["stage"], c["condition"], c["wrapper"], c["hop"], c["n"], c["budget"],
                  c["truncated"], c["budget_incomplete"], c["missing_answer"], c["repeated"],
                  f"{c['output_tokens_mean']:.1f}", c["output_tokens_max"]]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>")
    failures = []
    for r in summary["budget_failure_examples"]:
        title = f"{r['stage']} / {r['condition']} / {r['wrapper']} / {r['sample_id']}"
        failures.append(f"<details><summary>{html.escape(title)}</summary><pre>{html.escape(r['continuation'])}</pre></details>")
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>PrOntoQA independent pilot</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #d5dce0}}pre{{white-space:pre-wrap}}</style>
<h1>PrOntoQA four-stage independent pilot</h1>
<p>Status: <strong>{summary['decision']['status']}</strong>. {summary['physical_generations']} physical generations, batch {summary['batch_size']}; reasoning budget {summary['budgets']['cot']}, direct budget {summary['budgets']['direct']} tokens.</p>
<p>Base template views share one output. Budgets are selected using completeness only, never accuracy.</p>
<p>Pilot throughput suggests {summary['estimated_formal_gpu_hours']:.1f} formal GPU hours, excluding loading, retries, audits and reporting.</p>
<table><thead><tr>{''.join('<th>'+v+'</th>' for v in headers)}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Incomplete outputs at the budget cap</h2>{''.join(failures) or '<p>No incomplete output at the cap was observed.</p>'}
<p>This small independent pilot does not guarantee that no formal output will truncate. All formal failures are retained.</p></html>'''
    (Path(root) / "reports/pilot/report.html").write_text(body, encoding="utf-8")


def save_selection(root, summary):
    if summary["decision"]["status"] != "passed":
        raise RuntimeError("Cannot freeze a pilot with unresolved budget failures")
    result = dict(status="passed", preflight_digest=summary["preflight_digest"], selected_utc=utc_now(),
                  generation={"batch_size": summary["batch_size"], "budgets": summary["budgets"]},
                  evidence_files=summary["evidence_files"],
                  completeness_criterion="No pilot response reaches its cap without a complete final answer.",
                  estimated_formal_gpu_hours=summary["estimated_formal_gpu_hours"])
    path = Path(root) / "manifests/pilot_selection.json"
    if path.exists():
        old = json.loads(path.read_text())
        for key in ("status", "preflight_digest", "generation", "evidence_files"):
            if old[key] != result[key]:
                raise RuntimeError("An existing pilot selection has different evidence")
        return old
    atomic_json(path, result)
    return result


def main():
    import argparse
    from .common import add_workspace_arguments, workspace
    parser = argparse.ArgumentParser(description=__doc__)
    add_workspace_arguments(parser)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cot-budget", type=int, default=512)
    parser.add_argument("--direct-budget", type=int, default=32)
    parser.add_argument("--select", action="store_true", help="Save selection only when completeness passes")
    args = parser.parse_args()
    root = workspace(args)
    result = summarize(root, args.batch_size, {"cot": args.cot_budget, "direct": args.direct_budget})
    if args.select:
        save_selection(root, result)
    print(json.dumps(result["decision"]))


if __name__ == "__main__":
    main()
