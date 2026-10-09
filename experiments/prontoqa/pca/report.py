"""Deterministic PCA tables and descriptive figures from frozen result caches.

This module does not fit PCA or resample data.  Every table is derived from
cache records, and every comparison uses the matching frozen TwoNN replicate.
"""
from __future__ import annotations

import csv
import html
import io
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import atomic_json, atomic_text, sha256_file, utc_now


RECORD_FIELDS = [
    "stage", "layer", "preprocessing", "sample_set", "replicate", "k", "block",
    "n_samples", "n_features", "n_dropped", "n_constant_features", "rank_cap",
    "numerical_rank", "pc_id90", "legacy_index_gt90", "total_variance",
    "trace_relative_error", "negative_eigenvalues_clipped", "eigenvalue_tolerance",
    "min_raw_eigenvalue",
]
SUMMARY_FIELDS = [
    "stage", "layer", "preprocessing", "n_estimates", "n_samples", "n_features",
    "rank_cap", "pc_id90_mean", "pc_id90_sd", "legacy_index_gt90_mean",
    "legacy_index_gt90_sd", "numerical_rank_min", "numerical_rank_max",
    "n_constant_features_min", "n_constant_features_max", "total_variance_mean",
    "total_variance_sd",
]
COMPARISON_FIELDS = [
    "stage", "layer", "preprocessing", "replicate", "n_samples", "rank_cap",
    "pc_id90", "legacy_index_gt90", "twonn_id", "pc_minus_twonn", "pc_over_twonn",
]
COMPARISON_SUMMARY_FIELDS = [
    "stage", "layer", "preprocessing", "n_estimates", "n_samples", "rank_cap",
    "pc_id90_mean", "pc_id90_sd", "twonn_id_mean", "twonn_id_sd",
    "pc_minus_twonn_mean", "pc_minus_twonn_sd", "pc_over_twonn_mean", "pc_over_twonn_sd",
]
SCALE_SUMMARY_FIELDS = [
    "stage", "layer", "preprocessing", "k", "n_samples", "n_features", "n_dropped",
    "n_estimates", "rank_cap", "pc_id90_mean", "pc_id90_sd", "legacy_index_gt90_mean",
    "legacy_index_gt90_sd", "numerical_rank_min", "numerical_rank_max",
    "twonn_id_mean", "twonn_id_sd",
]
CURVE_FIELDS = [
    "stage", "preprocessing", "first_layer", "peak_layer", "last_layer",
    "first_mean", "first_sd", "peak_mean", "peak_sd", "last_mean", "last_sd",
    "peak_minus_first_mean", "peak_minus_first_sd", "peak_minus_last_mean",
    "peak_minus_last_sd", "shape",
]
TABLE_FIELDS = {
    "full.csv": RECORD_FIELDS,
    "replicates.csv": RECORD_FIELDS,
    "summary.csv": SUMMARY_FIELDS,
    "comparison_replicates.csv": COMPARISON_FIELDS,
    "comparison_summary.csv": COMPARISON_SUMMARY_FIELDS,
    "scale_blocks.csv": RECORD_FIELDS,
    "scale_summary.csv": SCALE_SUMMARY_FIELDS,
    "curve_summary.csv": CURVE_FIELDS,
}
STAGE_NAMES = {"base": "Base", "sft": "SFT", "dpo": "DPO", "rlvr": "RLVR"}
STAGE_COLORS = {"base": "#526D82", "sft": "#20877C", "dpo": "#BC6B26", "rlvr": "#8251A5"}
MODE_NAMES = {"correlation": "Correlation PCA (standardized)", "covariance": "Covariance PCA (raw)"}
MODE_LABELS = {"correlation": "Correlation PCA (standardized)", "covariance": "Covariance PCA (raw scale)"}


def _csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _unique_index(rows: list[dict], keys: tuple[str, ...], label: str) -> dict:
    result = {}
    for row in rows:
        key = tuple(int(row[k]) if k in {"layer", "replicate", "k", "block"} else row[k] for k in keys)
        if key in result:
            raise ValueError(f"Duplicate {label} coordinate: {key}")
        result[key] = row
    return result


def _consistent(rows: list[dict], field: str):
    values = {row[field] for row in rows}
    if len(values) != 1:
        raise ValueError(f"Inconsistent {field} within report group: {values}")
    return next(iter(values))


def _mean_sd(values) -> tuple[float, float]:
    vector = np.asarray(list(values), dtype=np.float64)
    if not len(vector) or not np.all(np.isfinite(vector)):
        raise ValueError("Report statistics require nonempty, finite observations")
    return float(np.mean(vector)), float(np.std(vector, ddof=0))


def _statistics(rows: list[dict]) -> dict:
    result = {key: _consistent(rows, key) for key in ("stage", "layer", "preprocessing", "n_samples", "n_features")}
    result["rank_cap"] = min(row["rank_cap"] for row in rows)
    result["n_estimates"] = len(rows)
    for key in ("pc_id90", "legacy_index_gt90", "total_variance"):
        result[key + "_mean"], result[key + "_sd"] = _mean_sd(row[key] for row in rows)
    for key in ("numerical_rank", "n_constant_features"):
        result[key + "_min"] = min(row[key] for row in rows)
        result[key + "_max"] = max(row[key] for row in rows)
    return result


def _twonn_value(row: dict, field: str = "id") -> float:
    value = float(row[field])
    if row.get("status") != "ok" or not np.isfinite(value) or value <= 0:
        raise ValueError("PCA comparison requires valid positive TwoNN estimates")
    return value


def _tables(root: Path, config: dict, manifest: dict, caches: list[dict]) -> tuple[dict, dict]:
    """Build tables, checking coordinate completeness before writing anything."""
    stages, layers, modes = config["stages"], sorted(config["layers"]), config["preprocessings"]
    if config.get("std_ddof", 0) != 0:
        raise ValueError("This report uses population SD (std_ddof=0)")
    cache_index, records = {}, []
    for cache in caches:
        metadata = cache["metadata"]
        key = (metadata["stage"], int(metadata["layer"]))
        if key in cache_index:
            raise ValueError(f"Duplicate PCA cache: {key}")
        if metadata["identity"] != manifest["identity"]:
            raise ValueError("PCA cache identity does not match report identity")
        cache_index[key] = cache
        for record in metadata["records"]:
            if (record["stage"], int(record["layer"])) != key:
                raise ValueError("PCA record belongs to a different cache coordinate")
            if set(RECORD_FIELDS) - set(record):
                raise ValueError("PCA cache record is missing report fields")
            records.append(dict(record))
    if set(cache_index) != {(stage, layer) for stage in stages for layer in layers}:
        raise ValueError("PCA cache coordinates do not match configured stages/layers")
    coordinates = ("stage", "layer", "preprocessing", "sample_set", "replicate", "k", "block")
    by_coordinate = _unique_index(records, coordinates, "PCA record")
    full_keys = [(s, l, m, "full", -1, 0, -1) for s in stages for l in layers for m in modes]
    rep_keys = [(s, l, m, "subsample", r, 0, -1) for s in stages for l in layers for m in modes for r in range(config["subsample_count"])]
    scale_keys = [(s, l, m, "scale", -1, k, b) for s in stages for l in sorted(config["scale_layers"]) for m in modes for k in sorted(config["scale_k"]) for b in range(k)]
    if set(by_coordinate) != set(full_keys + rep_keys + scale_keys):
        raise ValueError("PCA record coordinates are incomplete or contain unexpected records")
    full = [by_coordinate[key] for key in full_keys]
    replicates = [by_coordinate[key] for key in rep_keys]
    scale_blocks = [by_coordinate[key] for key in scale_keys]
    groups = defaultdict(list)
    for row in replicates:
        groups[(row["stage"], row["layer"], row["preprocessing"])].append(row)
    summary = [_statistics(rows) for rows in groups.values()]

    twonn_reps = _unique_index(_read_csv(root / "reports/id_replicates.csv"), ("stage", "layer", "replicate"), "TwoNN replicate")
    twonn_summary = _unique_index(_read_csv(root / "reports/id_summary.csv"), ("stage", "layer"), "TwoNN summary")
    twonn_scale = _unique_index(_read_csv(root / "reports/scale_checks.csv"), ("stage", "layer", "k"), "TwoNN scale")
    comparison_replicates, comparison_summary = [], []
    for key, rows in groups.items():
        paired = []
        for row in rows:
            match = twonn_reps[(row["stage"], row["layer"], row["replicate"])]
            if int(match["n_total"]) != row["n_samples"]:
                raise ValueError("PCA and TwoNN replicate sample counts differ")
            twonn_id = _twonn_value(match)
            pair = {field: row[field] for field in COMPARISON_FIELDS if field in row}
            pair.update(twonn_id=twonn_id, pc_minus_twonn=float(row["pc_id90"]) - twonn_id, pc_over_twonn=float(row["pc_id90"]) / twonn_id)
            paired.append(pair)
        comparison_replicates.extend(paired)
        aggregate = {field: _consistent(paired, field) for field in ("stage", "layer", "preprocessing", "n_samples")}
        aggregate["rank_cap"] = min(row["rank_cap"] for row in paired)
        aggregate["n_estimates"] = len(paired)
        for field in ("pc_id90", "twonn_id", "pc_minus_twonn", "pc_over_twonn"):
            aggregate[field + "_mean"], aggregate[field + "_sd"] = _mean_sd(row[field] for row in paired)
        reference = twonn_summary[(key[0], key[1])]
        _twonn_value(reference, "id_mean")
        if not np.allclose([aggregate["twonn_id_mean"], aggregate["twonn_id_sd"]], [float(reference["id_mean"]), float(reference["id_sd"])], rtol=1e-11, atol=1e-11):
            raise ValueError("TwoNN summary disagrees with paired replicate estimates")
        comparison_summary.append(aggregate)

    scale_groups = defaultdict(list)
    for row in scale_blocks:
        scale_groups[(row["stage"], row["layer"], row["preprocessing"], row["k"])].append(row)
    scale_summary = []
    for (stage, layer, mode, k), rows in scale_groups.items():
        aggregate = _statistics(rows)
        aggregate.update(k=k, n_dropped=_consistent(rows, "n_dropped"))
        reference = twonn_scale[(stage, layer, k)]
        if int(reference["n_block"]) != aggregate["n_samples"] or int(reference["n_dropped"]) != aggregate["n_dropped"]:
            raise ValueError("PCA and TwoNN scale partitions have different sizes")
        aggregate["twonn_id_mean"] = _twonn_value(reference, "id_mean")
        aggregate["twonn_id_sd"] = float(reference["id_sd"])
        if not np.isfinite(aggregate["twonn_id_sd"]) or aggregate["twonn_id_sd"] < 0:
            raise ValueError("Invalid TwoNN scale SD")
        scale_summary.append(aggregate)

    curve_summary = []
    for stage in stages:
        for mode in modes:
            values = np.array([[row["pc_id90"] for row in sorted(groups[(stage, layer, mode)], key=lambda row: row["replicate"])] for layer in layers], dtype=float)
            means, sds = values.mean(axis=1), values.std(axis=1, ddof=0)
            peak = int(np.argmax(means))
            shape = "flat" if np.all(means == means[0]) else ("start_peak" if peak == 0 else "end_peak" if peak == len(layers) - 1 else "interior_peak")
            row = {"stage": stage, "preprocessing": mode, "first_layer": layers[0], "peak_layer": layers[peak], "last_layer": layers[-1], "shape": shape}
            for name, position in (("first", 0), ("peak", peak), ("last", -1)):
                row[name + "_mean"], row[name + "_sd"] = float(means[position]), float(sds[position])
            for name, position in (("first", 0), ("last", -1)):
                row["peak_minus_" + name + "_mean"], row["peak_minus_" + name + "_sd"] = _mean_sd(values[peak] - values[position])
            curve_summary.append(row)
    tables = {
        "full.csv": full, "replicates.csv": replicates, "summary.csv": summary,
        "comparison_replicates.csv": comparison_replicates, "comparison_summary.csv": comparison_summary,
        "scale_blocks.csv": scale_blocks, "scale_summary.csv": scale_summary, "curve_summary.csv": curve_summary,
    }
    return tables, cache_index


def _plots(directory: Path, config: dict, tables: dict, caches: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "savefig.dpi": 170})
    stages, layers = config["stages"], sorted(config["layers"])
    scale_layers = sorted(config["scale_layers"])
    summary = _unique_index(tables["summary.csv"], ("stage", "layer", "preprocessing"), "summary")
    comparison = _unique_index(tables["comparison_summary.csv"], ("stage", "layer", "preprocessing"), "comparison")
    scale = _unique_index(tables["scale_summary.csv"], ("stage", "layer", "preprocessing", "k"), "scale")
    paths = []

    def save(fig, name):
        for suffix in ("png", "svg"):
            path = directory / f"{name}.{suffix}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            paths.append(path)
        plt.close(fig)

    def line(ax, x, mean, sd, color, label, style="-"):
        mean, sd = np.asarray(mean, float), np.asarray(sd, float)
        ax.plot(x, mean, color=color, linestyle=style, linewidth=1.9, label=label)
        ax.fill_between(x, np.maximum(mean - sd, 1e-12), mean + sd, color=color, alpha=.13, linewidth=0)

    for mode in config["preprocessings"]:
        fig, ax = plt.subplots(figsize=(9.3, 5.4), layout="constrained")
        for stage in stages:
            rows = [summary[(stage, layer, mode)] for layer in layers]
            line(ax, layers, [row["pc_id90_mean"] for row in rows], [row["pc_id90_sd"] for row in rows], STAGE_COLORS.get(stage, "#555555"), STAGE_NAMES.get(stage, stage))
        cap = summary[(stages[0], layers[0], mode)]["rank_cap"]
        ax.text(.99, .98, f"Subsample rank cap: {cap}", transform=ax.transAxes,
                horizontalalignment="right", verticalalignment="top", color="#637382", fontsize=9)
        ax.set(xlabel="Transformer layer", ylabel="PC-ID90 (component count)", title=MODE_NAMES[mode], ylim=(0, None))
        ax.grid(alpha=.16)
        ax.legend(ncol=min(3, len(stages) + 1), frameon=False)
        save(fig, mode + "_overlay")

    columns = min(2, len(stages))
    fig, axes = plt.subplots((len(stages) + columns - 1) // columns, columns, figsize=(11, 7.5), squeeze=False, layout="constrained")
    for ax, stage in zip(axes.flat, stages):
        for mode, color, label in (("correlation", "#20877C", "PC-ID90 standardized"), ("covariance", "#BC6B26", "PC-ID90 raw")):
            rows = [comparison[(stage, layer, mode)] for layer in layers]
            line(ax, layers, [row["pc_id90_mean"] for row in rows], [row["pc_id90_sd"] for row in rows], color, label)
        rows = [comparison[(stage, layer, config["preprocessings"][0])] for layer in layers]
        line(ax, layers, [row["twonn_id_mean"] for row in rows], [row["twonn_id_sd"] for row in rows], "#526D82", "TwoNN")
        ax.set(xlabel="Transformer layer", ylabel="Dimension estimate (log scale)", title=STAGE_NAMES.get(stage, stage), yscale="log")
        ax.grid(alpha=.16)
        ax.legend(frameon=False, fontsize=8)
    for ax in list(axes.flat)[len(stages):]:
        ax.set_visible(False)
    fig.suptitle("Different estimators on paired subsamples: mean ± population SD")
    save(fig, "method_comparison_four_panel")

    for mode in config["preprocessings"]:
        fig, axes = plt.subplots(2, len(scale_layers), figsize=(4 * len(scale_layers), 7), squeeze=False, layout="constrained")
        for column, layer in enumerate(scale_layers):
            top, bottom = axes[:, column]
            for stage in stages:
                cache = caches[(stage, layer)]
                with np.load(cache["npz_path"], allow_pickle=False) as spectra:
                    eigenvalues = np.asarray(spectra[f"{mode}_full_eigenvalues"], dtype=float)
                row = next(row for row in cache["metadata"]["records"] if row["preprocessing"] == mode and row["sample_set"] == "full")
                evr = eigenvalues / eigenvalues.sum()
                limit = min(int(row["rank_cap"]), len(evr))
                x = np.arange(1, limit + 1)
                positive = evr[:limit] > 0
                color = STAGE_COLORS.get(stage, "#555555")
                top.plot(x[positive], evr[:limit][positive], color=color, label=STAGE_NAMES.get(stage, stage), linewidth=1.3)
                cumulative = np.cumsum(evr)
                bottom.plot(x, cumulative[:limit], color=color, linewidth=1.5)
                count = int(row["pc_id90"])
                bottom.scatter([count], [cumulative[count - 1]], color=color, s=18)
            top.set(title=f"Layer {layer}", yscale="log", ylabel="Explained variance ratio", xlabel="Component index")
            top.legend(frameon=False, fontsize=8)
            bottom.axhline(config["threshold"], color="#858D97", linestyle="--", linewidth=1)
            bottom.set(ylim=(0, 1.025), ylabel="Cumulative explained variance", xlabel="Component count")
            top.grid(alpha=.15)
            bottom.grid(alpha=.15)
        fig.suptitle(MODE_NAMES[mode] + ": full-sample spectra")
        save(fig, "spectra_" + mode)

        fig, axes = plt.subplots(1, len(scale_layers), figsize=(4 * len(scale_layers), 4.7), squeeze=False, layout="constrained")
        for ax, layer in zip(axes.flat, scale_layers):
            for stage in stages:
                rows = sorted([scale[(stage, layer, mode, k)] for k in config["scale_k"]], key=lambda row: row["n_samples"])
                line(ax, [row["n_samples"] for row in rows], [row["pc_id90_mean"] for row in rows], [row["pc_id90_sd"] for row in rows], STAGE_COLORS.get(stage, "#555555"), STAGE_NAMES.get(stage, stage))
            ax.plot([row["n_samples"] for row in rows], [row["rank_cap"] for row in rows], color="#858D97", linestyle=":", linewidth=1, label="Sample rank cap")
            ax.set(title=f"Layer {layer}", xlabel="Samples per disjoint block (log scale)",
                   ylabel="PC-ID90 (log scale)", xscale="log", yscale="log")
            ax.grid(alpha=.15)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(MODE_NAMES[mode] + ": sample-scale check (mean ± population SD)")
        save(fig, "scale_" + mode)
    return paths


def _html_report(config: dict, manifest: dict, tables: dict) -> str:
    esc = html.escape
    def table(rows):
        fields = list(rows[0])
        return "<div class='scroll'><table><tr>" + "".join(f"<th>{esc(x)}</th>" for x in fields) + "</tr>" + "".join("<tr>" + "".join(f"<td>{esc(str(row[x]))}</td>" for x in fields) + "</tr>" for row in rows) + "</table></div>"
    figure_names = ["correlation_overlay", "covariance_overlay", "method_comparison_four_panel",
                    "spectra_correlation", "spectra_covariance", "scale_correlation", "scale_covariance"]
    figures = "".join(f"<figure><img src='{name}.png' alt='{name.replace('_', ' ')}'><figcaption>{name.replace('_', ' ')} · <a href='{name}.svg'>SVG</a></figcaption></figure>" for name in figure_names)
    links = "".join(f"<li><a href='{name}'>{name}</a>: {len(rows)} rows</li>" for name, rows in tables.items())
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tulu3 PrOntoQA PCA and TwoNN</title><style>body{{max-width:1280px;margin:40px auto;padding:0 28px;font:16px/1.7 system-ui;color:#172432}}img{{width:100%}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:8px;border-bottom:1px solid #ddd;white-space:nowrap}}code{{overflow-wrap:anywhere}}a{{color:#146c74}}</style>
<h1>Tulu3 PrOntoQA: PCA and TwoNN</h1><p>Analysis identity: <code>{esc(manifest['identity'])}</code>. Activation protocol: <code>{esc(manifest.get('parent_protocol_digest', 'recorded in input manifest'))}</code>.</p>
<h2>Data and methods</h2><p>{config['n_samples']} shared questions; {config['n_features']} coordinates; four stages and 32 raw decoder blocks. Each layer uses the same {config['subsample_count']} saved subsets of {config['subsample_size']} items. PC-ID90 is the 1-based count required to reach or exceed 90% cumulative explained variance. Correlation PCA centers and standardizes each coordinate within the current sample set using population SD (ddof=0), then computes covariance with denominator n−1. Its nonconstant diagonal is n/(n−1); this common scale factor does not affect variance ratios or counts. Covariance PCA only centers and preserves coordinate scales. Constant coordinates contribute zero variance.</p>
<p>The exact sample Gram matrix is diagonalized in float64, and every saved spectrum contains all {config['n_features']} feature-space eigenvalues, including zero padding. Negative eigenvalues within the documented numerical tolerance are clipped; trace agreement and centered-rank bounds are independently checked. The additional legacy_index_gt90 is the zero-based index at strictly greater than 90%, which need not equal pc_id90−1 at an exact threshold hit.</p>
<h2>Curves and paired comparisons</h2><p>Means and population SD describe the saved overlapping subsamples, not independent experiments or confidence intervals. Peaks use unsmoothed means and earliest ties; paired peak-minus-endpoint differences use the chosen common layer. This data-driven selection is descriptive. Differences and ratios against TwoNN are calculated within each matched stage, layer and replicate before aggregation. Mean ratios are not ratios of means. PCA measures global linear variance concentration while TwoNN estimates local neighbor-ratio dimension; their difference is not an estimator error.</p>
{table(tables['curve_summary.csv'])}
{figures}
<h2>Sample size and interpretation</h2><p>Only prespecified layers {config['scale_layers']} use k=1…20 disjoint equal blocks from the saved permutations. Each block fits its own moments. Remainders are recorded, and k=1 has SD=0. The centered rank is bounded by min(n−1,d), with constants reducing d; full-sample cap 1439, main-subset cap 1295, minimum-block cap 71. Sample-size sensitivity and these limits prevent direct extrapolation to a population dimension.</p>
<p>Interior peaks do not imply strictly monotonic sides. Neither curve shape, stage ordering nor PCA–TwoNN separation proves curvature, compression mechanisms or causal training effects. Native input formatting and final-token readout differ by stage. No generation accuracy or performance is measured.</p>
<h2>Artifacts</h2><ul>{links}<li><a href="spectra_index.json">Complete spectra index and SHA-256</a></li><li><a href="manifest.json">Report file hashes</a></li><li><a href="../manifests/run.json">Current analysis identity and environment</a></li><li><a href="../../reports/report.html">TwoNN report</a></li></ul>
<p><a href="https://arxiv.org/abs/1905.12784v2">Paper v2</a> · <a href="https://github.com/ansuini/IntrinsicDimDeep/tree/2c5140009eb3efccca1d774dc66cb043c04b26d8">Pinned author implementation</a>. Shared PCA subsets and blockwise PCA scale checks are explicit extensions.</p></html>"""


def write_reports(root: Path, config: dict, manifest: dict, caches: list[dict]) -> dict:
    """Write the complete report and return its manifest (no formal PCA fitting)."""
    root = Path(root).resolve()
    directory = root / "pca/reports"
    directory.mkdir(parents=True, exist_ok=True)
    tables, cache_index = _tables(root, config, manifest, caches)
    paths = []
    spectrum_index = []
    for cache in caches:
        path = Path(cache["npz_path"]).resolve()
        with np.load(path, allow_pickle=False) as archive:
            keys = {name: list(archive[name].shape) for name in archive.files}
        spectrum_index.append({"stage": cache["metadata"]["stage"], "layer": cache["metadata"]["layer"],
                               "project_relative_path": str(path.relative_to(root)),
                               "sha256": sha256_file(path), "array_shapes": keys})
    index_path = directory / "spectra_index.json"
    atomic_json(index_path, {"identity": manifest["identity"], "spectra": spectrum_index,
                            "dtype": "float64", "last_axis": "all feature-space eigenvalues, descending"})
    paths.append(index_path)
    for name, fields in TABLE_FIELDS.items():
        path = directory / name
        _csv(path, fields, tables[name])
        paths.append(path)
    paths.extend(_plots(directory, config, tables, cache_index))
    path = directory / "report.html"
    atomic_text(path, _html_report(config, manifest, tables))
    paths.append(path)
    output_manifest = {
        "identity": manifest["identity"], "parent_protocol_digest": manifest.get("parent_protocol_digest"),
        "created_utc": utc_now(), "row_counts": {name: len(rows) for name, rows in tables.items()},
        "outputs": {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(paths)},
    }
    atomic_json(directory / "manifest.json", output_manifest)
    return output_manifest
