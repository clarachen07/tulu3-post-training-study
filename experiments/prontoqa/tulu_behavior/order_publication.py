"""Aggregate-only publication and semantic checks for the order experiment."""

from __future__ import annotations

import copy
import itertools
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest

from .analyze import _figures, holm_adjust
from .common import (
    REPO,
    STAGES,
    atomic_json,
    fingerprint,
    publication_code_hashes,
    sha256_file,
    utc_now,
    verify_files,
)
from .compare import ORDER_TABLES
from .publication import (
    TABLE_NAMES,
    _close,
    _require,
    _svg_series,
    check_links,
    check_statistics,
    published_corpus,
)
from .verify import _verify_csv

ROOT = REPO / "behavior-order-sensitivity"
METRICS = ("answer_correct", "proof_success", "joint_success")
RESULT_TABLES = {
    **TABLE_NAMES,
    "primary_tests": "order_primary_tests",
    "secondary_stage_tests": "secondary_stage_tests",
}
ALLOWED = {
    "README.md",
    "experiment.json",
    "reports/analysis.json",
    "reports/order_comparison.json",
    "reports/provenance.json",
    "provenance/history.json",
    "provenance/reanalysis.json",
    "provenance/validation.json",
    "reports/figures/values.json",
    *[f"reports/{n}.csv" for n in RESULT_TABLES.values()],
    *[f"reports/{n}.csv" for n in ORDER_TABLES.values()],
    *[
        f"reports/figures/{n}.{ext}"
        for n in ("order_effect", "difficulty_curves")
        for ext in ("png", "svg")
    ],
}


def stage_view(summary):
    result = copy.deepcopy(summary)
    result["primary_tests"] = result.pop("secondary_stage_tests")
    result["statistics"] = result.pop("secondary_statistics")
    return result


def check_order_statistics(comparison, summary, baseline):
    _require(
        comparison["n_items"] == 4320
        and comparison["physical_generations_per_version"] == 60480
        and comparison["logical_pairs"] == 69120,
        "Order experiment totals differ",
    )
    _require(
        comparison["statistics"]
        == {
            "bootstrap_replicates": 10000,
            "seed": 20261007,
            "confidence_level": 0.95,
            "bootstrap": "stratified_paired_shared_indices_both_versions",
            "primary": "native/cot/answer_correct/equal_weight_hops",
            "multiplicity": "Holm_family_4",
            "direction": "shuffled_minus_baseline",
        },
        "Order statistical protocol differs",
    )
    keys = ("stage", "condition", "template", "hop", "metric")
    lookup = {tuple(r[k] for k in keys): r for r in comparison["summaries"]}
    old = {tuple(r[k] for k in keys): r for r in baseline["summaries"]}
    new = {tuple(r[k] for k in keys): r for r in summary["summaries"]}
    _require(
        len(lookup) == len(comparison["summaries"])
        and lookup.keys() == old.keys() == new.keys(),
        "Order summary matrix is incomplete or duplicated",
    )
    for key, row in lookup.items():
        stage, condition, template, hop, metric = key
        n = 4320 if hop == "all" else 1440
        parts = [
            row[k]
            for k in ("both_correct", "gained_correct", "lost_correct", "both_wrong")
        ]
        _require(
            all(type(v) is int and v >= 0 for v in parts)
            and sum(parts) == row["n"] == n,
            "Invalid order transition partition",
        )
        _close(
            row["baseline_estimate"],
            old[key]["estimate"],
            "Baseline order marginal differs",
        )
        _close(
            row["shuffled_estimate"],
            new[key]["estimate"],
            "Shuffled order marginal differs",
        )
        _close(
            (parts[0] + parts[2]) / n,
            row["baseline_estimate"],
            "Baseline transition marginal differs",
        )
        _close(
            (parts[0] + parts[1]) / n,
            row["shuffled_estimate"],
            "Shuffled transition marginal differs",
        )
        _close(
            row["delta_pp"],
            100 * (row["shuffled_estimate"] - row["baseline_estimate"]),
            "Order difference differs",
        )
        _require(
            row["aggregation"]
            == ("equal_weight_hops" if hop == "all" else "within_hop"),
            "Order weighting differs",
        )
        _require(
            row["discordant"] == parts[1] + parts[2]
            and -100 <= row["ci_low_pp"] <= row["ci_high_pp"] <= 100,
            "Invalid order discordance or saved interval",
        )
        if hop == "all":
            for field in (
                "n",
                "both_correct",
                "gained_correct",
                "lost_correct",
                "both_wrong",
            ):
                _require(
                    row[field]
                    == sum(
                        lookup[stage, condition, template, h, metric][field]
                        for h in (1, 3, 5)
                    ),
                    "Order pooled transition differs from hop partitions",
                )
        if stage == "base":
            _require(
                row
                == {
                    **lookup[stage, condition, "plain", hop, metric],
                    "template": template,
                },
                "Order Base aliases differ",
            )
    tests = comparison["primary_tests"]
    _require(
        [r["stage"] for r in tests] == list(STAGES), "Order primary family differs"
    )
    probabilities = []
    for row in tests:
        expected = lookup[row["stage"], "cot", "native", "all", "answer_correct"]
        _require(
            all(row[k] == v for k, v in expected.items()),
            "Order primary test differs from summary",
        )
        gain, loss = row["gained_correct"], row["lost_correct"]
        p = float(binomtest(gain, gain + loss, 0.5).pvalue) if gain + loss else 1.0
        _close(row["p_value"], p, "Order exact McNemar differs", absolute=0)
        _require(
            row["family_size"] == 4 and row["test"] == "exact_two_sided_mcnemar",
            "Order test family differs",
        )
        probabilities.append(p)
    for row, adjusted in zip(tests, holm_adjust(probabilities)):
        _close(row["p_holm"], adjusted, "Order Holm differs", absolute=0)
        _require(
            row["reject_holm_0_05"] == (adjusted < 0.05), "Order Holm decision differs"
        )
    _require(
        summary["primary_tests"] == tests
        and summary["statistics"] == comparison["statistics"],
        "Main order analysis primary endpoint differs",
    )
    interactions = comparison["stage_gain_changes"]
    seen = set()
    for row in interactions:
        a, b, c, t, h, m = (
            row[k]
            for k in (
                "before_stage",
                "after_stage",
                "condition",
                "template",
                "hop",
                "metric",
            )
        )
        key = a, b, c, t, h, m
        _require(key not in seen, "Duplicate stage-gain change")
        seen.add(key)
        _close(
            row["stage_gain_change_pp"],
            lookup[b, c, t, h, m]["delta_pp"] - lookup[a, c, t, h, m]["delta_pp"],
            "Stage-gain interaction differs",
        )
        _require(
            -200 <= row["ci_low_pp"] <= row["ci_high_pp"] <= 200,
            "Invalid saved interaction interval",
        )
    expected = {
        (a, b, c, t, h, m)
        for a, b in itertools.pairwise(STAGES)
        for c in ("cot", "direct")
        for t in ("native", "plain")
        for h in ("all", 1, 3, 5)
        for m in (METRICS if c == "cot" else METRICS[:1])
    }
    _require(seen == expected, "Stage-gain change matrix is incomplete")
    ordering = comparison["ordering_summary"]
    _require(
        set(ordering) == {"formal", "pilot", "demonstrations"},
        "Ordering split diagnostics are incomplete",
    )
    for split, groups in ordering.items():
        counts = {
            "formal": {str(h): 1440 for h in (1, 3, 5)},
            "pilot": {str(h): 16 for h in (1, 3, 5)},
            "demonstrations": {"3": 4},
        }[split]
        _require(
            {h: r["n"] for h, r in groups.items()} == counts,
            "Ordering diagnostic strata differ",
        )
        for hop, row in groups.items():
            n = row["n"]
            _require(
                0 <= row["unchanged_questions"] <= n,
                "Invalid unchanged permutation count",
            )
            _close(
                row["unchanged_fraction"],
                row["unchanged_questions"] / n,
                "Unchanged permutation rate differs",
            )
            _require(
                sum(row["fact_position_counts_1based"].values()) == n,
                "Fact-position counts differ",
            )
            if hop != "1":
                _require(
                    0 <= row["full_reverse_rule_chain"] <= n,
                    "Invalid reverse-rule count",
                )
                _close(
                    row["full_reverse_rule_chain_fraction"],
                    row["full_reverse_rule_chain"] / n,
                    "Reverse-rule rate differs",
                )
    return len(lookup)


def effect_limits(tests):
    return (
        2 * math.floor(min(r["ci_low_pp"] for r in tests) / 2) - 2,
        2 * math.ceil(max(r["ci_high_pp"] for r in tests) / 2) + 2,
    )


def figures(summary, comparison, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        _figures(summary, temp)
        for extension in ("png", "svg"):
            shutil.copyfile(
                temp / f"difficulty_curves.{extension}",
                directory / f"difficulty_curves.{extension}",
            )
        difficulty = json.loads((temp / "values.json").read_text())["difficulty_curves"]
    rows = comparison["primary_tests"]
    y = [r["delta_pp"] for r in rows]
    bounds = [[r["ci_low_pp"], r["ci_high_pp"]] for r in rows]
    limits = effect_limits(rows)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
        artist = ax.errorbar(
            range(4),
            y,
            yerr=[
                [v - b[0] for v, b in zip(y, bounds)],
                [b[1] - v for v, b in zip(y, bounds)],
            ],
            fmt="o",
            color="#255f85",
            capsize=5,
        )
        artist.lines[0].set_gid("order_effect")
        artist.lines[2][0].set_gid("ci_order_effect")
        ax.axhline(0, color="#888888", linewidth=0.9, linestyle="--")
        for i, value in enumerate(y):
            ax.annotate(
                f"{value:+.2f}",
                (i, value),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        ax.set(
            xticks=range(4),
            xticklabels=[s.upper() for s in STAGES],
            ylim=limits,
            xlim=(-0.5, 3.65),
            ylabel="Shuffled minus preorder (percentage points)",
            title="Sentence-order effect | native template + reasoning",
        )
        for extension in ("png", "svg"):
            fig.savefig(directory / f"order_effect.{extension}", dpi=180)
        plt.close(fig)
    atomic_json(
        directory / "values.json",
        {
            "order_effect": {"y_pp": y, "ci_pp": bounds, "ylim": list(limits)},
            "difficulty_curves": difficulty,
        },
    )


def check_order_figures(summary, comparison, root=ROOT):
    directory = Path(root) / "reports/figures"
    plotted = json.loads((directory / "values.json").read_text())
    tests = comparison["primary_tests"]
    y = [r["delta_pp"] for r in tests]
    bounds = [[r["ci_low_pp"], r["ci_high_pp"]] for r in tests]
    limits = effect_limits(tests)
    _require(
        plotted["order_effect"] == {"y_pp": y, "ci_pp": bounds, "ylim": list(limits)},
        "Order plotted values differ",
    )
    _svg_series(
        directory / "order_effect.svg", "order_effect", y, limits=limits, markers=True
    )
    _svg_series(
        directory / "order_effect.svg",
        "ci_order_effect",
        bounds,
        intervals=True,
        limits=limits,
    )
    lookup = {
        (r["stage"], r["condition"], r["hop"]): r
        for r in summary["summaries"]
        if r["metric"] == "answer_correct" and r["template"] == "native"
    }
    series = plotted["difficulty_curves"]
    _require(
        Counter((r["stage"], r["condition"]) for r in series)
        == Counter((s, c) for s in STAGES for c in ("cot", "direct")),
        "Shuffled difficulty plot series differ",
    )
    for row in series:
        rows = [lookup[row["stage"], row["condition"], h] for h in (1, 3, 5)]
        expected = [r["estimate"] * 100 for r in rows]
        intervals = [[r["ci_low"] * 100, r["ci_high"] * 100] for r in rows]
        _require(
            row["y_percent"] == expected and row["ci_percent"] == intervals,
            "Shuffled difficulty values differ",
        )
        gid = "difficulty_" + row["condition"] + "_" + row["stage"]
        _svg_series(directory / "difficulty_curves.svg", gid, expected)
        _svg_series(
            directory / "difficulty_curves.svg", "ci_" + gid, intervals, intervals=True
        )


def reference_files(root=ROOT):
    root = Path(root)
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or any(
            p in ("__pycache__", ".pytest_cache") for p in path.parts
        ):
            continue
        name = str(path.relative_to(root))
        _require(
            name in ALLOWED
            or (name.startswith("tests/test_") and path.suffix == ".py"),
            f"File is outside the order publication allowlist: {name}",
        )
        if name != "provenance/validation.json":
            files.append(path)
    return files


def verify_order_publication(root=ROOT, *, check_hashes=True):
    root = Path(root)
    summary = json.loads((root / "reports/analysis.json").read_text())
    comparison = json.loads((root / "reports/order_comparison.json").read_text())
    baseline = json.loads((REPO / "behavior/reports/analysis.json").read_text())
    check_statistics(stage_view(summary), published_corpus(REPO / "behavior"))
    cells = check_order_statistics(comparison, summary, baseline)
    for key, name in RESULT_TABLES.items():
        _verify_csv(root / f"reports/{name}.csv", summary[key])
    for key, name in ORDER_TABLES.items():
        _verify_csv(root / f"reports/{name}.csv", comparison[key])
    _require(
        summary["provenance"]
        == json.loads((root / "reports/provenance.json").read_text()),
        "Embedded provenance differs",
    )
    history = json.loads((root / "provenance/history.json").read_text())
    reanalysis = json.loads((root / "provenance/reanalysis.json").read_text())
    _require(
        history["case_review"]["status"] == "passed"
        and history["case_review"]["reviewer"] == "Codex",
        "Historical review differs",
    )
    _require(
        reanalysis["status"] == "passed"
        and reanalysis["comparison"]["status"] == "passed"
        and reanalysis["baseline_rescored_generations"]
        == reanalysis["shuffled_rescored_generations"]
        == 60480
        and reanalysis["saved_scoring_disagreements"] == 0,
        "Full reanalysis record is incomplete",
    )
    _require(
        comparison["provenance"]["shuffled_protocol_digest"]
        == history["protocol_digest"]
        and comparison["provenance"] == reanalysis["provenance"],
        "Historical/reanalysis identities differ",
    )
    check_order_figures(summary, comparison, root)
    links = check_links(
        [root / "README.md", REPO / "README.md", REPO / "reports/report.zh-CN.md"]
    )
    reference_files(root)
    if check_hashes:
        ledger = json.loads((root / "provenance/validation.json").read_text())
        _require(
            ledger["status"] == "passed"
            and ledger["scope"] == "published_aggregate_evidence_only",
            "Order publication scope differs",
        )
        _require(
            fingerprint({k: v for k, v in ledger.items() if k != "identity"})
            == ledger["identity"],
            "Order ledger identity differs",
        )
        verify_files(REPO, ledger["files"])
        _require(
            ledger["code_files"] == publication_code_hashes(),
            "Order publication code identity changed",
        )
    return {
        "status": "passed",
        "scope": "published_aggregate_evidence_only",
        "order_summary_cells": cells,
        "primary_tests": 4,
        "secondary_stage_tests": 3,
        "svg_series_and_intervals_checked": 9,
        "local_links_checked": links,
        "rescoring_performed": False,
        "bootstrap_recomputed": False,
    }


def save_validation():
    if not (ROOT / "provenance/validation.json").exists():
        atomic_json(ROOT / "provenance/validation.json", {"status": "pending"})
    checks = verify_order_publication(check_hashes=False)
    paths = reference_files() + list((REPO / "tulu_behavior").glob("*.py"))
    paths += [
        REPO / "README.md",
        REPO / "reports/report.zh-CN.md",
        REPO / "behavior/reports/analysis.json",
    ]
    record = {
        **checks,
        "schema_version": 1,
        "verified_utc": utc_now(),
        "code_files": publication_code_hashes(),
        "files": {str(p.relative_to(REPO)): sha256_file(p) for p in sorted(set(paths))},
        "limitations": [
            "Published counts verify McNemar/Holm and point estimates; raw rescoring and paired bootstrap need external evidence."
        ],
    }
    record["identity"] = fingerprint(record)
    atomic_json(ROOT / "provenance/validation.json", record)
    return verify_order_publication()
