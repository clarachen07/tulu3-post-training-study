"""Paired behavioral statistics and a self-contained, offline English report.

Only complete physical generation matrices may enter formal analysis. Base's
single plain-text generation is referenced by both logical template views;
it is never counted as two independent observations.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .common import (
    ROOT, atomic_json, atomic_jsonl, fingerprint, load_config, read_jsonl,
    sha256_file, utc_now,
)
from .freeze import validate_frozen

STAGES = ("base", "sft", "dpo", "rlvr")
CONDITIONS = ("cot", "direct")
TEMPLATES = ("native", "plain")
HOPS = (1, 3, 5)
METRICS = ("answer_correct", "proof_success", "joint_success")
METRIC_NAMES = {"answer_correct": "Answer accuracy", "proof_success": "Strict proof success",
                "joint_success": "Joint answer and proof success"}
LIMITATIONS = [
    "Results apply to these four fixed checkpoints and this evaluation protocol; stage differences are not causal effects of training methods.",
    "Hop strata were generated independently and cannot be paired by position; difficulty curves do not isolate reasoning depth.",
    "Intervals quantify item-sampling uncertainty, excluding demonstration selection and training randomness.",
    "Base native and plain views reference the same generation, not independent observations.",
    "All success rates use every item in the respective condition as their denominator; direct answers have no proof metrics.",
]


def holm_adjust(p_values):
    """Holm step-down family-wise correction, preserving input order."""
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite probabilities")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    out = np.empty_like(values)
    out[order] = adjusted
    return out.tolist()


def exact_mcnemar(before, after):
    """Two-sided exact binomial test of discordant paired answers."""
    before, after = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    if before.shape != after.shape or before.ndim != 1:
        raise ValueError("McNemar inputs must be aligned one-dimensional vectors")
    gain = int((~before & after).sum())
    loss = int((before & ~after).sum())
    return {
        "both_correct": int((before & after).sum()), "gained_correct": gain,
        "lost_correct": loss, "both_wrong": int((~before & ~after).sum()),
        "n": len(before), "discordant": gain + loss,
        "p_value": float(binomtest(gain, gain + loss, .5).pvalue) if gain + loss else 1.,
    }


def _label(value):
    if value is True or value == "True":
        return "True"
    if value is False or value == "False":
        return "False"
    raise ValueError(f"Invalid ground-truth label: {value!r}")


def _validated_matrix(items, records):
    if not items:
        raise ValueError("No formal items; cannot create a formal report")
    item_by_id = {}
    for item in items:
        sid = item.get("sample_id")
        if not sid or sid in item_by_id:
            raise ValueError(f"Missing or duplicate sample_id: {sid}")
        if item.get("hop") not in HOPS:
            raise ValueError(f"Unexpected hop for {sid}: {item.get('hop')}")
        _label(item.get("answer"))
        item_by_id[sid] = item
    missing_hops = set(HOPS) - {item["hop"] for item in items}
    if missing_hops:
        raise ValueError(f"Missing hop strata: {sorted(missing_hops)}")
    physical, generation_ids = {}, set()
    for record in records:
        sid, stage = record.get("sample_id"), record.get("stage")
        condition, wrapper = record.get("condition"), record.get("wrapper")
        key = (sid, stage, condition, wrapper)
        if sid not in item_by_id:
            raise ValueError(f"Unknown sample_id in generation: {sid}")
        if stage not in STAGES or condition not in CONDITIONS or wrapper not in TEMPLATES:
            raise ValueError(f"Unexpected generation condition: {key}")
        if stage == "base" and wrapper != "plain":
            raise ValueError("Base must have one physical plain generation, never a native duplicate")
        if record.get("phase") != "formal":
            raise ValueError(f"Non-formal generation mixed into analysis: {key}")
        if record.get("hop") != item_by_id[sid]["hop"]:
            raise ValueError(f"Generation hop mismatch: {key}")
        if key in physical:
            raise ValueError(f"Duplicate physical generation cell: {key}")
        gid = record.get("generation_id")
        if not gid or gid in generation_ids:
            raise ValueError(f"Missing or duplicate physical generation_id: {gid}")
        score = record.get("scoring", {})
        required = ("answer_correct", "format_compliant")
        if condition == "cot":
            required += ("proof_success", "joint_success", "proof_parseable", "has_skip", "has_logic_error")
        for name in required:
            if type(score.get(name)) is not bool:
                raise ValueError(f"Missing boolean scoring.{name} for {key}")
        if not isinstance(score.get("answer_status"), str):
            raise ValueError(f"Missing answer_status for {key}")
        extracted = score.get("extracted_answer")
        if extracted is not None:
            extracted = _label(extracted)
        if score["answer_correct"] != (extracted == _label(item_by_id[sid]["answer"])):
            raise ValueError(f"Answer score disagrees with extracted answer for {key}")
        if condition == "cot" and score["joint_success"] != (score["answer_correct"] and score["proof_success"]):
            raise ValueError(f"Joint success inconsistent with answer and proof for {key}")
        if condition == "direct" and any(score.get(name) is not None for name in METRICS[1:]):
            raise ValueError(f"Direct-answer proof metrics must be null: {key}")
        for name in ("input_tokens", "output_tokens"):
            if type(record.get(name)) is not int or record[name] < 0:
                raise ValueError(f"Invalid {name} for {key}")
        if type(record.get("truncated")) is not bool:
            raise ValueError(f"Missing truncated flag for {key}")
        if not isinstance(record.get("continuation"), str) or not isinstance(record.get("finish_reason"), str):
            raise ValueError(f"Missing raw generation or stop reason for {key}")
        generation_ids.add(gid)
        physical[key] = record
    expected = {
        (sid, stage, condition, wrapper)
        for sid in item_by_id for stage in STAGES for condition in CONDITIONS
        for wrapper in (("plain",) if stage == "base" else TEMPLATES)
    }
    missing = expected - physical.keys()
    if missing:
        raise ValueError(f"Incomplete formal matrix: {len(missing)} missing physical cells; first={sorted(missing)[0]}")
    return item_by_id, physical


def _ci(values):
    return [float(v) for v in np.quantile(values, [.025, .975])]


def _check_generation_identity(records, frozen):
    generation = frozen["identity"]["generation"]
    digest = frozen["protocol_digest"]
    for row in records:
        if (row.get("run_digest") != digest
                or row.get("budget") != generation["budgets"][row["condition"]]
                or row.get("batch_size") != generation["batch_size"]):
            raise ValueError(f"Generation differs from frozen protocol: {row.get('generation_id')}")
    return generation


def _summarize_tokens(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "max": int(values.max())}


def _logical_rows(items, physical):
    rows = []
    for item in items:
        for stage in STAGES:
            for condition in CONDITIONS:
                for template in TEMPLATES:
                    wrapper = "plain" if stage == "base" else template
                    rec = physical[item["sample_id"], stage, condition, wrapper]
                    score = rec["scoring"]
                    rows.append({
                        "sample_id": item["sample_id"], "hop": item["hop"], "label": _label(item["answer"]),
                        "stage": stage, "condition": condition, "template": template,
                        "physical_wrapper": wrapper, "generation_id": rec["generation_id"],
                        "shared_base_generation": stage == "base", "extracted_answer": score.get("extracted_answer"),
                        **{name: score.get(name) for name in (*METRICS, "answer_status", "format_compliant",
                                                             "proof_parseable", "has_skip", "has_logic_error", "step_count")},
                        **{name: rec[name] for name in ("input_tokens", "output_tokens", "finish_reason", "truncated")},
                    })
    return rows


def analyze_records(items, records, bootstrap_replicates=10000, seed=20261007):
    """Compute statistics without I/O; require all 14 physical cells per item.

    Each hop is sampled independently with replacement. Every model, template,
    condition and metric uses the same sampled indices within that hop and
    bootstrap replicate. Aggregate estimates are equal-weight means of hops.
    """
    items, records = list(items), list(records)
    item_by_id, physical = _validated_matrix(items, records)
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    items = sorted(items, key=lambda item: (item["hop"], item["sample_id"]))
    logical = _logical_rows(items, physical)
    columns = [
        (stage, condition, template, metric)
        for stage in STAGES for condition in CONDITIONS for template in TEMPLATES
        for metric in (METRICS if condition == "cot" else METRICS[:1])
    ]
    col_index = {key: index for index, key in enumerate(columns)}
    arrays, points, bootstrap, hop_ids = {}, {}, {}, {}
    rng = np.random.default_rng(seed)
    for hop in HOPS:
        hop_ids[hop] = [item["sample_id"] for item in items if item["hop"] == hop]
        arrays[hop] = np.asarray([
            [physical[sid, stage, condition, "plain" if stage == "base" else template]["scoring"][metric]
             for stage, condition, template, metric in columns]
            for sid in hop_ids[hop]
        ], dtype=float)
        points[hop] = arrays[hop].mean(axis=0)
        bootstrap[hop] = np.empty((bootstrap_replicates, len(columns)))
        n = len(hop_ids[hop])
        for start in range(0, bootstrap_replicates, 64):
            end = min(bootstrap_replicates, start + 64)
            indices = rng.integers(0, n, size=(end - start, n))
            bootstrap[hop][start:end] = arrays[hop][indices].mean(axis=1)
    points["all"] = np.mean([points[hop] for hop in HOPS], axis=0)
    bootstrap["all"] = np.mean([bootstrap[hop] for hop in HOPS], axis=0)
    summaries = []
    for index, (stage, condition, template, metric) in enumerate(columns):
        for hop in ("all", *HOPS):
            summaries.append({
                "stage": stage, "condition": condition, "template": template, "metric": metric, "hop": hop,
                "n": len(items) if hop == "all" else len(hop_ids[hop]),
                "estimate": float(points[hop][index]), "ci_low": _ci(bootstrap[hop][:, index])[0],
                "ci_high": _ci(bootstrap[hop][:, index])[1],
                "aggregation": "equal_weight_hops" if hop == "all" else "within_hop",
            })
    paired_differences = []

    def difference(kind, before, after, hop):
        a, b = col_index[before], col_index[after]
        interval = _ci(bootstrap[hop][:, b] - bootstrap[hop][:, a])
        return {
            "kind": kind, "before_stage": before[0], "after_stage": after[0],
            "before_condition": before[1], "after_condition": after[1],
            "before_template": before[2], "after_template": after[2], "metric": before[3], "hop": hop,
            "delta_pp": float((points[hop][b] - points[hop][a]) * 100),
            "ci_low_pp": interval[0] * 100, "ci_high_pp": interval[1] * 100,
        }

    for condition in CONDITIONS:
        for template in TEMPLATES:
            for before_stage, after_stage in zip(STAGES, STAGES[1:]):
                for metric in (METRICS if condition == "cot" else METRICS[:1]):
                    for hop in ("all", *HOPS):
                        paired_differences.append(difference(
                            "adjacent_stage", (before_stage, condition, template, metric),
                            (after_stage, condition, template, metric), hop))
    for stage in STAGES:
        for condition in CONDITIONS:
            for metric in (METRICS if condition == "cot" else METRICS[:1]):
                for hop in ("all", *HOPS):
                    paired_differences.append(difference(
                        "template_native_minus_plain", (stage, condition, "plain", metric),
                        (stage, condition, "native", metric), hop))
        for template in TEMPLATES:
            for hop in ("all", *HOPS):
                paired_differences.append(difference(
                    "cot_minus_direct", (stage, "direct", template, "answer_correct"),
                    (stage, "cot", template, "answer_correct"), hop))
    transitions = []
    primary_tests = []
    for condition in CONDITIONS:
        for template in TEMPLATES:
            for before_stage, after_stage in zip(STAGES, STAGES[1:]):
                a = col_index[before_stage, condition, template, "answer_correct"]
                b = col_index[after_stage, condition, template, "answer_correct"]
                for hop in ("all", *HOPS):
                    arr = np.concatenate([arrays[h] for h in HOPS]) if hop == "all" else arrays[hop]
                    test = exact_mcnemar(arr[:, a], arr[:, b])
                    transition = {"before_stage": before_stage, "after_stage": after_stage,
                                  "condition": condition, "template": template, "hop": hop,
                                  **{key: value for key, value in test.items() if key not in ("p_value", "discordant")}}
                    transitions.append(transition)
                    if condition == "cot" and template == "native" and hop == "all":
                        primary_tests.append({
                            **difference("adjacent_stage", columns[a], columns[b], hop),
                            **test, "test": "exact_two_sided_mcnemar", "family_size": 3,
                        })
    for test, adjusted in zip(primary_tests, holm_adjust([test["p_value"] for test in primary_tests])):
        test["p_holm"] = adjusted
        test["reject_holm_0_05"] = adjusted < .05
    grouped = defaultdict(list)
    for row in logical:
        for hop in ("all", row["hop"]):
            grouped[row["stage"], row["condition"], row["template"], hop].append(row)
    diagnostics, label_summaries = [], []
    for (stage, condition, template, hop), group in grouped.items():
        n = len(group)
        # Diagnostic "all" rows are descriptive pooled rates; main estimates
        # above are explicitly macro-averaged. Formal strata have equal size.
        diagnostics.append({
            "stage": stage, "condition": condition, "template": template, "hop": hop, "n": n,
            "aggregation": "pooled_descriptive", "answer_parse_failure_rate": sum(r["extracted_answer"] is None for r in group) / n,
            "answer_status_counts": dict(Counter(r["answer_status"] for r in group)),
            "format_compliance_rate": sum(r["format_compliant"] for r in group) / n,
            "proof_parse_failure_rate": sum(not r["proof_parseable"] for r in group) / n if condition == "cot" else None,
            "skip_rate": sum(r["has_skip"] for r in group) / n if condition == "cot" else None,
            "logic_error_rate": sum(r["has_logic_error"] for r in group) / n if condition == "cot" else None,
            "truncation_rate": sum(r["truncated"] for r in group) / n,
            "finish_reason_counts": dict(Counter(r["finish_reason"] for r in group)),
            "input_tokens": _summarize_tokens([r["input_tokens"] for r in group]),
            "output_tokens": _summarize_tokens([r["output_tokens"] for r in group]),
        })
        for label in ("True", "False"):
            subgroup = [r for r in group if r["label"] == label]
            for metric in (METRICS if condition == "cot" else METRICS[:1]):
                label_summaries.append({
                    "stage": stage, "condition": condition, "template": template, "hop": hop,
                    "label": label, "metric": metric, "n": len(subgroup),
                    "estimate": float(np.mean([r[metric] for r in subgroup])) if subgroup else None,
                    "aggregation": "pooled_within_label_descriptive",
                })
    dataset_summary = []
    for hop in HOPS:
        group = [item for item in items if item["hop"] == hop]
        # Fictional preorder MP data have one premise fact and universal rules.
        # Count universal statements lexically without depending on a scoring
        # parser that might normalize/drop a statement.
        rule_counts = []
        for item in group:
            sentences = [s.strip() for s in item.get("question", "").split(".") if s.strip()]
            rule_counts.append(sum(s.startswith(("Every ", "Each ", "All ")) or " are " in s for s in sentences))
        dataset_summary.append({
            "hop": hop, "n": len(group), "labels": dict(Counter(_label(item["answer"]) for item in group)),
            "question_characters": _summarize_tokens([len(item.get("text", item.get("question", "") + " " + item.get("query", ""))) for item in group]),
            "rule_count": _summarize_tokens(rule_counts), "rule_count_method": "universal_statement_count",
        })
    return {
        "schema_version": 1, "status": "complete_matrix_analyzed", "n_items": len(items),
        "physical_generations": len(physical), "logical_records": len(logical),
        "statistics": {"bootstrap_replicates": bootstrap_replicates, "seed": seed,
                       "confidence_level": .95, "bootstrap": "percentile_stratified_paired_shared_indices",
                       "primary": "native/cot/answer_correct/equal_weight_hops", "multiplicity": "Holm_family_3"},
        "dataset_summary": dataset_summary, "summaries": summaries, "paired_differences": paired_differences,
        "primary_tests": primary_tests, "transitions": transitions, "diagnostics": diagnostics,
        "label_summaries": label_summaries, "logical_results": logical, "limitations": LIMITATIONS,
    }


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refuse to write an empty summary: {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(stream.getvalue(), encoding="utf-8-sig")
    temporary.replace(path)


def _audit_cases(items, records, seed, per_stratum=2):
    by_id = {item["sample_id"]: item for item in items}
    pools = defaultdict(list)
    for rec in records:
        score = rec["scoring"]
        categories = []
        if score.get("extracted_answer") is None:
            categories.append("answer_parse_failure")
        if rec["condition"] == "cot":
            if not score["proof_parseable"]:
                categories.append("proof_parse_failure")
            if score["has_skip"]:
                categories.append("skipped_step")
            if score["has_logic_error"]:
                categories.append("logic_error")
            if score["answer_correct"] != score["proof_success"]:
                categories.append("answer_proof_disagreement")
        if rec["truncated"]:
            categories.append("truncated")
        if not categories:
            categories.append("other_correct" if score["answer_correct"] else "other_incorrect")
        for category in categories:
            pools[rec["stage"], rec["hop"], rec["condition"], rec["wrapper"], category].append(rec)
    rng = np.random.default_rng(seed)
    sampled = []
    for key, pool in sorted(pools.items()):
        pool = sorted(pool, key=lambda record: record["generation_id"])
        for index in rng.choice(len(pool), size=min(per_stratum, len(pool)), replace=False):
            rec = pool[int(index)]
            sampled.append({
                "category": key[-1], "stratum_population": len(pool),
                "item": by_id[rec["sample_id"]], "record": rec,
            })
    return {"seed": seed, "per_stratum": per_stratum,
            "stratification": ["stage", "hop", "condition", "physical_wrapper", "error_category"],
            "case_review": {"reviewer": "Codex", "status_source": "manifests/case_review.json"},
            "note": "Fixed stratified evidence for case review; see the separate attestation. A generation can appear in multiple error categories.",
            "cases": sampled}


def _case_review_status(root, protocol_digest, audit):
    """Display completion only for an attestation bound to this exact sample.

    The sample itself never embeds changing review state. That keeps its hash
    stable when a report is rendered again after the Codex case review.
    """
    root = Path(root)
    path = root / "manifests/case_review.json"
    pending = {"status": "pending", "reviewer": "Codex", "required_for_pipeline_completion": True}
    if not path.is_file():
        return {**pending, "reason": "case_review_manifest_missing"}
    try:
        review = json.loads(path.read_text(encoding="utf-8"))
        expected = {case["record"]["generation_id"] for case in audit["cases"]}
        reviewed = review.get("reviewed_generation_ids")
        if not (isinstance(reviewed, list) and all(isinstance(gid, str) for gid in reviewed)):
            return {**pending, "reason": "invalid_reviewed_generation_ids"}
        checks = (
            (review.get("status") == "passed", "review_not_passed"),
            (review.get("protocol_digest") == protocol_digest, "review_protocol_mismatch"),
            (review.get("audit_sample_sha256") == sha256_file(root / "reports/error_audit_sample.json"), "review_sample_mismatch"),
            (review.get("reviewer") == "Codex", "reviewer_mismatch"),
            (expected <= set(reviewed), "review_coverage_incomplete"),
            (isinstance(review.get("findings"), list), "review_findings_missing"),
        )
        for valid, reason in checks:
            if not valid:
                return {**pending, "reason": reason}
        return {"status": "passed", "reviewer": "Codex", "required_for_pipeline_completion": True,
                "reviewed_generation_count": len(expected), "findings": review["findings"],
                "manifest_sha256": sha256_file(path), "manifest": "manifests/case_review.json"}
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return {**pending, "reason": "invalid_case_review_manifest"}


def _figures(summary, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    lookup = {(r["stage"], r["condition"], r["template"], r["hop"], r["metric"]): r for r in summary["summaries"]}
    colors = ("#255f85", "#ce773b", "#397c68")
    plotted = {"stage_comparison": [], "difficulty_curves": []}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for offset, (metric, name, color) in enumerate(zip(METRICS, ("Answer", "Strict proof", "Joint"), colors)):
            rows = [lookup[stage, "cot", "native", "all", metric] for stage in STAGES]
            y = np.array([r["estimate"] * 100 for r in rows])
            bounds = np.array([[r["ci_low"] * 100, r["ci_high"] * 100] for r in rows])
            x = np.arange(4) + (offset - 1) * .13
            artist = ax.errorbar(x, y, yerr=np.maximum(0, np.array([y - bounds[:, 0], bounds[:, 1] - y])),
                                 marker="o", capsize=3, label=name, color=color)
            artist.lines[0].set_gid("stage_" + metric)
            artist.lines[2][0].set_gid("ci_stage_" + metric)
            plotted["stage_comparison"].append({"metric": metric, "x": artist.lines[0].get_xdata().tolist(),
                                                "y_percent": artist.lines[0].get_ydata().tolist(),
                                                "ci_percent": bounds.tolist()})
        ax.set(xticks=np.arange(4), xticklabels=[s.upper() for s in STAGES], ylim=(-2, 102),
               ylabel="Success rate (%)", title="Native template + reasoning | equal-weight hops")
        ax.legend(frameon=False, ncol=3)
        ax.grid(axis="y", alpha=.2)
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"stage_comparison.{suffix}", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
        for ax, condition in zip(axes, CONDITIONS):
            for stage in STAGES:
                rows = [lookup[stage, condition, "native", hop, "answer_correct"] for hop in HOPS]
                y = np.array([r["estimate"] * 100 for r in rows])
                bounds = np.array([[r["ci_low"] * 100, r["ci_high"] * 100] for r in rows])
                artist = ax.errorbar(HOPS, y, yerr=np.maximum(0, np.array([y - bounds[:, 0], bounds[:, 1] - y])),
                                     marker="o", capsize=3, label=stage.upper())
                artist.lines[0].set_gid("difficulty_" + condition + "_" + stage)
                artist.lines[2][0].set_gid("ci_difficulty_" + condition + "_" + stage)
                plotted["difficulty_curves"].append({"condition": condition, "stage": stage,
                                                     "x": artist.lines[0].get_xdata().tolist(),
                                                     "y_percent": artist.lines[0].get_ydata().tolist(),
                                                     "ci_percent": bounds.tolist()})
            ax.set(xticks=HOPS, ylim=(-2, 102), xlabel="Hop", title="Reasoning" if condition == "cot" else "Direct answer")
            ax.grid(axis="y", alpha=.2)
        axes[0].set_ylabel("Answer accuracy (%)")
        axes[1].legend(frameon=False)
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"difficulty_curves.{suffix}", dpi=180)
        plt.close(fig)
    atomic_json(directory / "values.json", plotted)


def _table(headers, rows):
    return ("<div class='table-wrap'><table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row)
                                                + "</tr>" for row in rows) + "</tbody></table></div>")


def _pct(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def _report(summary, config, audit, *, published=False):
    lookup = {(r["stage"], r["condition"], r["template"], r["hop"], r["metric"]): r for r in summary["summaries"]}
    primary = []
    for stage in STAGES:
        row = [stage.upper()]
        for metric in METRICS:
            r = lookup[stage, "cot", "native", "all", metric]
            row.append(f"{_pct(r['estimate'])} [{_pct(r['ci_low'])}, {_pct(r['ci_high'])}]")
        primary.append(row)
    tests = [[r["before_stage"].upper() + " → " + r["after_stage"].upper(),
              f"{r['delta_pp']:+.2f} [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]",
              f"{r['gained_correct']} / {r['lost_correct']}", f"{r['p_value']:.6g}", f"{r['p_holm']:.6g}"]
             for r in summary["primary_tests"]]
    metadata = summary["provenance"]
    lead = ", ".join(f"{stage.upper()} {_pct(lookup[stage, 'cot', 'native', 'all', 'answer_correct']['estimate'])}" for stage in STAGES)
    diagnostic = [[r["stage"].upper(), r["condition"], r["template"], _pct(r["answer_parse_failure_rate"]),
                   _pct(r["proof_parse_failure_rate"]), _pct(r["skip_rate"]), _pct(r["logic_error_rate"]),
                   _pct(r["truncation_rate"]), f"{r['output_tokens']['mean']:.1f}"]
                  for r in summary["diagnostics"] if r["hop"] == "all"]
    secondary = [[r["stage"].upper(), r["condition"], r["template"], r["hop"],
                   f"{_pct(r['estimate'])} [{_pct(r['ci_low'])}, {_pct(r['ci_high'])}]"]
                  for r in summary["summaries"] if r["metric"] == "answer_correct" and r["hop"] != "all"]
    data = [[r["hop"], r["n"], r["labels"].get("True", 0), r["labels"].get("False", 0),
             f"{r['question_characters']['mean']:.1f}", f"{r['rule_count']['mean']:.1f}"] for r in summary["dataset_summary"]]
    generation = metadata["generation"]
    review = summary["case_review"]
    if published:
        evidence = ("Publication contains aggregate evidence only. The original Codex case review covered "
                    f"{review['reviewed_generation_count']} unique generations. "
                    '<a href="../provenance/history.json">Historical acceptance summary</a>. '
                    '<a href="../provenance/reanalysis.json">Full local rescoring and statistical comparison</a>. '
                    '<a href="../provenance/publication.json">Aggregate-only publication validation</a>. '
                    "The published counts permit McNemar/Holm verification, but do not permit rescoring or paired-bootstrap recomputation.")
        downloads = '<a href="../data/formal.jsonl">Fixed formal corpus</a>'
    else:
        evidence = (f"Case review: passed. Reviewer: Codex; {review['reviewed_generation_count']} unique generations. "
                    '<a href="../manifests/case_review.json">Attestation</a>.' if review["status"] == "passed" else
                    "Case review: pending. Reviewer: Codex; completion is not yet attested.")
        downloads = ('<a href="logical_results.csv">Per-item logical views</a> · '
                     '<a href="logical_results.jsonl">Per-item JSONL</a> · '
                     '<a href="error_audit_sample.json">Stratified raw audit sample</a>')
    tables = _table(["Stage", *[METRIC_NAMES[m] + " [95% CI]" for m in METRICS]], primary)
    tests_table = _table(["Comparison", "Change, pp [95% CI]", "Wrong→right / right→wrong", "Exact McNemar p", "Holm p"], tests)
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tulu 3 × PrOntoQA behavioral evaluation</title><style>
body{font-family:system-ui,sans-serif;color:#20313e;background:#f4f5f2;margin:0;line-height:1.65}main{max-width:1120px;margin:auto;padding:38px 28px 70px}h1{font-size:32px;line-height:1.3}h2{margin-top:42px;color:#255f85}p{max-width:95ch}.lead{font-size:19px;background:white;padding:20px;border-left:4px solid #255f85}.note{padding:15px 20px;background:#fff2db;border:1px solid #e7d3a9}.table-wrap{overflow-x:auto;background:white}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #dae1e4;white-space:nowrap}th{background:#e5edf0}figure{margin:24px 0;background:white;padding:12px}img{max-width:100%;height:auto}figcaption{font-size:13px;color:#5c6a6e}a{color:#255f85}code{overflow-wrap:anywhere}footer{margin-top:45px;font-size:12px;color:#5c6a6e}
</style><main>""" + f"""
<h1>Tulu 3 four-stage behavior on PrOntoQA</h1>
<p class="lead">Primary endpoint: native template, reasoning then answer, equal-weight 1 / 3 / 5-hop answer accuracy: {lead}.</p>
<p>{summary['n_items']:,} formal questions; {summary['physical_generations']:,} physical generations; {summary['logical_records']:,} logical views. Base's two template views share one physical output. Pilot observations are excluded.</p>
<div class="note">{evidence}</div>
<h2>Primary results</h2><p>Brackets are 95% percentile stratified paired bootstrap intervals: {summary['statistics']['bootstrap_replicates']:,} replicates, seed {summary['statistics']['seed']}. Indices are shared across every checkpoint, template, condition and metric within each hop.</p>
{tables}
<figure><img src="figures/stage_comparison.png" alt="Answer, strict proof and joint success with confidence intervals"><figcaption>Native template + reasoning; equal-weight hops. <a href="figures/stage_comparison.svg">SVG</a></figcaption></figure>
{tests_table}
<p>Two-sided exact McNemar tests apply only to these three primary adjacent-stage comparisons, with Holm correction in one family of three. Changes are later minus earlier stage. Other comparisons are descriptive secondary results.</p>
<h2>Difficulty, response condition and template</h2>
<figure><img src="figures/difficulty_curves.png" alt="Native-template accuracy by hop and response condition"><figcaption>Independent hop pools; these curves do not isolate reasoning depth. <a href="figures/difficulty_curves.svg">SVG</a></figcaption></figure>
<details><summary>Accuracy by hop, condition and template</summary>{_table(['Stage', 'Condition', 'Template', 'Hop', 'Accuracy [95% CI]'], secondary)}</details>
<p><a href="paired_differences.csv">Paired differences</a> include adjacent stages, native minus plain, and reasoning minus direct. <a href="label_summary.csv">Label summaries</a> and <a href="transitions.csv">Stage transitions</a> retain every group.</p>
<h2>Failure modes and output length</h2>
{_table(['Stage', 'Condition', 'Template', 'Answer parse failure', 'Proof parse failure', 'Skip', 'Logic error', 'Truncation', 'Mean output tokens'], diagnostic)}
<p>Failure flags overlap and must not be summed. A proof parse failure can cause downstream skip flags; those flags alone do not establish a true omission. Direct answers have no proof metrics. <a href="diagnostics.csv">Diagnostics</a> retain format compliance, stop counts, lengths and hop-specific rates.</p>
<h2>Data and protocol</h2>
{_table(['Hop', 'Questions', 'True', 'False', 'Mean text characters', 'Mean rules'], data)}
{_table(['Stage', 'Checkpoint', 'Fixed revision'], [[m['stage'].upper(), m['id'], m['revision']] for m in config['models']])}
<p>Greedy BF16 inference without quantization or TF32; one response per physical condition. Four independent 3-hop demonstrations are shared, ordered True, False, True, False. Native prompts contain one user message and no extra system instruction. Plain prompts use the same body. Only the pilot's completeness criterion selects budgets, never correctness.</p>
<p>Reasoning {generation['budgets']['cot']} tokens, direct answer {generation['budgets']['direct']} tokens; batch size {generation['batch_size']}. Historical generation identity: <code>{html.escape(metadata['protocol_digest'])}</code>.</p>
<p>Answers are parsed independently from proofs. A strict proof must parse every assertion, derive each new fact from premises or one valid Modus Ponens step, and reach the correct proposition. False items require the explicit negation of the query. Valid repetition and alternate paths are allowed; missing necessary steps and extra invalid assertions fail.</p>
<p>Downloads: <a href="summary.csv">Success summaries</a> · <a href="primary_tests.csv">Primary tests</a> · <a href="dataset_summary.csv">Dataset summaries</a> · <a href="analysis.json">Aggregate statistics</a> · <a href="provenance.json">Provenance</a> · {downloads}</p>
<h2>Interpretation limits</h2><ul>{''.join('<li>' + html.escape(t) + '</li>' for t in LIMITATIONS)}</ul>
<footer>Generated at {html.escape(metadata['generated_at_utc'])} UTC · {html.escape(config['protocol'])}</footer></main></html>
"""


def analyze(root=ROOT, *, destination=None):
    """Read frozen formal outputs and write reports only after completeness checks."""
    root = Path(root)
    destination = Path(destination) if destination is not None else root
    config = load_config(root)
    items_path = root / "data/formal.jsonl"
    items = read_jsonl(items_path)
    counts = Counter(item["hop"] for item in items)
    expected_counts = {hop: config["data"]["count_per_hop"] for hop in HOPS}
    if counts != expected_counts:
        raise ValueError(f"Formal data counts mismatch: expected {expected_counts}, got {dict(counts)}")
    files = sorted((root / "outputs/formal").glob("*/*/*/batch_*.jsonl"))
    if not files:
        raise ValueError("No formal generations; report will not be created")
    records = [rec for path in files for rec in read_jsonl(path)]
    # Reject incomplete panels before requesting the formal freeze; validate
    # protocol identity before spending time on the bootstrap or writing files.
    _validated_matrix(items, records)
    frozen = validate_frozen(root, check_environment=False)
    generation = _check_generation_identity(records, frozen)
    summary = analyze_records(items, records, config["statistics"]["bootstrap_replicates"], config["statistics"]["seed"])
    expected = config.get("expected", {}).get("physical_generations_formal")
    if expected is not None and summary["physical_generations"] != expected:
        raise ValueError(f"Formal generation count mismatch: expected {expected}, got {summary['physical_generations']}")
    provenance = {
        "generated_at_utc": utc_now(), "protocol": config["protocol"],
        "config_sha256": sha256_file(root / "experiment.json"), "dataset_sha256": sha256_file(items_path),
        "raw_batches": {str(path.relative_to(root)): sha256_file(path) for path in files},
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "scoring_code_sha256": sha256_file(Path(__file__).parent / "scoring.py"),
        "models": config["models"], "statistics": summary["statistics"],
        "protocol_digest": frozen["protocol_digest"], "generation": generation,
        "frozen_protocol_sha256": sha256_file(root / "manifests/frozen_protocol.json"),
    }
    provenance["source_fingerprint"] = fingerprint({key: value for key, value in provenance.items() if key != "generated_at_utc"})
    summary["provenance"] = provenance
    audit = _audit_cases(items, records, config["statistics"]["seed"])
    reports = destination / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    logical = summary.pop("logical_results")
    atomic_jsonl(reports / "logical_results.jsonl", logical)
    _write_csv(reports / "logical_results.csv", logical)
    for key, name in (("summaries", "summary"), ("primary_tests", "primary_tests"),
                      ("paired_differences", "paired_differences"), ("transitions", "transitions"),
                      ("diagnostics", "diagnostics"), ("label_summaries", "label_summary"),
                      ("dataset_summary", "dataset_summary")):
        _write_csv(reports / f"{name}.csv", summary[key])
    atomic_json(reports / "error_audit_sample.json", audit)
    summary["case_review"] = _case_review_status(destination, frozen["protocol_digest"], audit)
    atomic_json(reports / "analysis.json", summary)
    atomic_json(reports / "provenance.json", provenance)
    _figures(summary, reports / "figures")
    temporary = reports / "report.html.tmp"
    temporary.write_text(_report(summary, config, audit), encoding="utf-8")
    temporary.replace(reports / "report.html")
    produced = [reports / name for name in (
        "logical_results.jsonl", "logical_results.csv", "summary.csv", "primary_tests.csv",
        "paired_differences.csv", "transitions.csv", "diagnostics.csv", "label_summary.csv",
        "dataset_summary.csv", "analysis.json", "provenance.json", "error_audit_sample.json", "report.html",
        "figures/stage_comparison.png", "figures/stage_comparison.svg",
        "figures/difficulty_curves.png", "figures/difficulty_curves.svg",
        "figures/values.json",
    )]
    atomic_json(destination / "manifests/analysis.json", {
        "schema_version": 1, "generated_at_utc": utc_now(),
        "protocol_digest": frozen["protocol_digest"], "config_sha256": provenance["config_sha256"],
        "physical_generations": summary["physical_generations"], "logical_records": summary["logical_records"],
        "files": {str(path.relative_to(destination)): sha256_file(path) for path in produced},
    })
    return summary


def main():
    from .common import add_workspace_arguments, workspace
    parser = argparse.ArgumentParser(description="Analyze complete PrOntoQA behavioral generations")
    add_workspace_arguments(parser, analysis=True)
    args = parser.parse_args()
    if args.source_dir is not None:
        from .reanalysis import reanalyze
        result = reanalyze(args.source_dir, args.output_dir)
        print(json.dumps({"status": result["status"], "rescored_generations": result["rescored_generations"]}))
    else:
        root = workspace(args)
        result = analyze(root)
        print(json.dumps({"status": result["status"], "physical_generations": result["physical_generations"],
                          "report": str(root / "reports/report.html")}))


if __name__ == "__main__":
    main()
