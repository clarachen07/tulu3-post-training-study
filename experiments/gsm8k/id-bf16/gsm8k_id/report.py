"""Render only independently verified experiment results; never invent curves."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import ANCHORS, ROOT, STAGES, atomic_bytes, read, sha, utc, write
from .analyze import export_csv

COLORS = {"base": "#4b5563", "sft": "#167d9a", "dpo": "#bd663b", "rlvr": "#7958a5"}
LABELS = {"twonn": "TwoNN ID", "correlation": "Correlation PCA: PC-ID90", "covariance": "Covariance PCA: PC-ID90"}


def report(root, output_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    from .results import verify_results
    if output_dir is not None:
        verify_results(root)
    else:
        # A new complete-evidence workflow renders only after its independent audit.
        audit = read(root / "reports/independent_audit.json")
        manifest = read(root / "reports/analysis_manifest.json")
        stats = read(root / "reports/statistics.json")
        if audit["status"] != "passed" or audit["source_identity"] != stats["source_identity"]:
            raise ValueError("Complete independent verification required before publishing")
        from .results import check_hashes
        check_hashes(root, manifest["reports"])
    stats = read(root / "reports/statistics.json")
    if output_dir is not None:
        import shutil
        destination = Path(output_dir).resolve()
        if destination == root.resolve():
            raise ValueError("Use a separate output directory")
        (destination / "reports").mkdir(parents=True, exist_ok=True)
        for name in ("statistics.json", "estimates.csv", "summary.csv", "behavior.csv", "per_question.csv"):
            shutil.copy2(root / "reports" / name, destination / "reports" / name)
        root = destination
    figdir = root / "reports/figures"
    figdir.mkdir(exist_ok=True)
    figures, shapes = [], []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "savefig.dpi": 180})
    for condition in ("primary", "common_complete"):
        for method in LABELS:
            fig, axes = plt.subplots(1, 5, figsize=(20, 4.4), sharey=True, constrained_layout=True)
            for ax, anchor in zip(axes, ANCHORS):
                panel = "common_complete" if condition == "common_complete" else ("all_input" if anchor == 0 else "all_trajectory")
                for stage in STAGES:
                    data = sorted((r for r in stats["summary"] if (r["panel"], r["stage"], r["anchor"], r["method"]) == (panel, stage, anchor, method)), key=lambda r: r["layer"])
                    if not data:
                        continue
                    layers = np.array([r["layer"] for r in data])
                    means = np.array([np.nan if r["mean"] is None else r["mean"] for r in data])
                    sd = np.array([np.nan if r["sd"] is None else r["sd"] for r in data])
                    ax.plot(layers, means, color=COLORS[stage], label=stage.upper(), lw=1.8)
                    ax.fill_between(layers, means - sd, means + sd, color=COLORS[stage], alpha=.12)
                    if np.isfinite(means).all():
                        peak = int(np.argmax(means))
                        shapes.append({"condition": condition, "panel": panel, "method": method, "stage": stage, "anchor": anchor,
                                       "n": data[0]["n"], "first": means[0], "peak_layer": int(layers[peak]), "peak": means[peak],
                                       "last": means[-1], "peak_to_last_drop": means[peak] - means[-1],
                                       "relative_peak_to_last_drop": float(1 - means[-1] / means[peak]) if means[peak] else None,
                                       "middle_peak_L9_L24": bool(9 <= layers[peak] <= 24 and means[peak] > means[0] and means[peak] > means[-1])})
                ax.set_title("Before output" if anchor == 0 else f"{anchor}% of output")
                ax.set_xlabel("Decoder block")
                ax.set_xticks([1, 8, 16, 24, 32])
                ax.grid(alpha=.18)
                if not ax.lines:
                    ax.text(.5, .5, "Insufficient samples", transform=ax.transAxes, ha="center")
            axes[0].set_ylabel(LABELS[method])
            axes[-1].legend(frameon=False)
            fig.suptitle(f"GSM8K · BF16 · 8-shot native wrappers · {condition.replace('_', ' ')}\n20 paired 90% subsamples; shading = population SD")
            name = f"{condition}_{method}"
            for ext in ("png", "svg"):
                path = figdir / f"{name}.{ext}"
                fig.savefig(path)
                figures.append(path)
            plt.close(fig)
    export_csv(root / "reports/shapes.csv", shapes)
    # Paired changes use the same replicate, rather than differences of error bars.
    import csv
    estimates = list(csv.DictReader((root / "reports/estimates.csv").open()))
    paired, progress = [], []
    lookup = {(r["panel"], r["stage"], r["anchor"], r["layer"], r["method"], r["replicate"]): r for r in estimates}
    for panel in ("all_input", "all_trajectory", "common_complete"):
        for anchor in ANCHORS:
            for layer in range(1, 33):
                for method in LABELS:
                    for a, b in zip(STAGES[:-1], STAGES[1:]):
                        ds = []
                        for rep in range(20):
                            first = lookup.get((panel, a, str(anchor), str(layer), method, str(rep)))
                            second = lookup.get((panel, b, str(anchor), str(layer), method, str(rep)))
                            if first and second and first["status"] == second["status"] == "ok":
                                ds.append(float(second["value"]) - float(first["value"]))
                        if ds:
                            paired.append({"panel": panel, "anchor": anchor, "layer": layer, "method": method,
                                           "from_stage": a, "to_stage": b, "valid_repeats": len(ds), "mean_delta": float(np.mean(ds)), "sd_delta": float(np.std(ds))})
    export_csv(root / "reports/paired_stage_deltas.csv", paired)
    for panel in ("all_trajectory", "common_complete"):
        for stage in STAGES:
            for anchor in ANCHORS[1:]:
                for layer in range(1, 33):
                    for method in LABELS:
                        ds = []
                        for rep in range(20):
                            baseline = lookup.get((panel, stage, "0", str(layer), method, str(rep)))
                            current = lookup.get((panel, stage, str(anchor), str(layer), method, str(rep)))
                            if baseline and current and baseline["status"] == current["status"] == "ok":
                                ds.append(float(current["value"]) - float(baseline["value"]))
                        if ds:
                            progress.append({"panel": panel, "stage": stage, "anchor": anchor, "layer": layer,
                                             "method": method, "valid_repeats": len(ds), "mean_delta_from_input": float(np.mean(ds)), "sd_delta": float(np.std(ds))})
    export_csv(root / "reports/paired_progress_deltas.csv", progress)
    lines = ["# GSM8K four-stage BF16 representation geometry", "",
             "This report describes the frozen experiment. Retained-result checks do not rescore external raw answers or recompute activation geometry.", "",
             "## Protocol", "",
             "All 1,319 test questions; historical eight reasoning demonstrations; Base plain wrapper and tuned native chat templates. BF16 greedy generation with 2,048 new tokens and an 8,192-token context. Stops: EOS or the next-question marker.", "",
             "Raw decoder block outputs are read before output and at 25%, 50%, 75% and 100% of the effective continuation, using ceil(progress * output length). EOS and stop material are excluded. Final RMSNorm is excluded. BF16 values are stored as float32; TwoNN distances and both PCA analyses use float64. PC-ID90 retains at least 90% variance.", "",
             "Curves show the mean of 20 paired 90% subsamples without replacement; shading is population SD, not a confidence interval. The all-input panel contains every question; the main trajectory panel retains truncations. The common-complete panel requires a nonempty, nontruncated output from every stage. A truncated output at 100% is its budget endpoint, not a completed solution.", "",
             "## Behavior", "", "| Stage | Correct / total | Accuracy | Truncated | Empty | Mean output tokens |", "|---|---:|---:|---:|---:|---:|"]
    for row in stats["behavior"]:
        lines.append(f"| {row['stage'].upper()} | {row['correct']}/{row['n']} | {row['accuracy']:.2%} | {row['truncated']} | {row['empty_output']} | {row['mean_output_tokens']:.1f} |")
    lines.extend(["", "## Geometry", ""])
    for condition in ("primary", "common_complete"):
        lines.extend([f"### {condition.replace('_', ' ').title()}", ""])
        for method, label in LABELS.items():
            lines.extend([f"![{label}](figures/{condition}_{method}.png)", "", f"[Vector figure](figures/{condition}_{method}.svg)", ""])
    lines.extend(["## Interpretation", "",
                  "Stage comparisons combine checkpoint weights with native wrappers and readout contexts. Output-progress alignment does not align semantic reasoning steps. TwoNN and PCA quantify different geometric properties; ID is not a direct measure of knowledge, reasoning ability or information. These fixed-checkpoint comparisons do not identify causal effects of training algorithms.", "",
                  "[Layerwise summary](summary.csv) · [Replicate estimates](estimates.csv) · [Paired stage changes](paired_stage_deltas.csv) · [Paired progress changes](paired_progress_deltas.csv) · [Per-question metrics](per_question.csv)"])
    path = root / "reports/report.md"
    atomic_bytes(path, ("\n".join(lines) + "\n").encode())
    write(root / "reports/publication.json", {"created_utc": utc(), "source_identity": stats["source_identity"],
          "files": {str(p.relative_to(root)): sha(p) for p in [path, *figures, root / "reports/shapes.csv", root / "reports/paired_stage_deltas.csv", root / "reports/paired_progress_deltas.csv"]}})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, required=True, help="Separate destination for regenerated figures and English report")
    args = p.parse_args()
    report(args.root, args.output_dir)
