"""Audit frozen activation batches, estimate TwoNN, and make offline reports.

This module never generates model responses or substitutes synthetic results
for missing experiment data. Per-layer caches are bound to all source batches,
their provenance sidecars, the frozen protocol, and the shared sample indices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from . import common
from .twonn import block_estimates, estimate_distances, euclidean_distances


CACHE_VERSION = 1
STAGE_LABELS = {"base": "Base", "sft": "SFT", "dpo": "DPO", "rlvr": "RLVR"}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, out.getvalue())


def load_inputs(root: Path, config: dict) -> dict:
    frozen = common.validate_frozen(root)
    items = common.read_jsonl(root / "data/items.jsonl")
    n = config["data"]["sample_count"]
    if len(items) != n or n < 3:
        raise ValueError(f"Expected {n} real samples; found {len(items)}")
    sample_ids = [item["sample_id"] for item in items]
    if len(set(sample_ids)) != n:
        raise ValueError("Duplicate sample IDs")
    if len({item["text"] for item in items}) != n:
        raise ValueError("Duplicate question text")
    if any(item["hop"] != config["data"]["hop"] for item in items):
        raise ValueError("Unexpected hop count in the frozen dataset")
    stats = config["statistics"]
    reps = np.load(root / "data/subsamples.npy", allow_pickle=False)
    if reps.shape != (stats["subsample_repeats"], stats["subsample_size"]):
        raise ValueError("Shared subsample array has the wrong shape")
    if not np.issubdtype(reps.dtype, np.integer):
        raise ValueError("Subsample indices must be integers")
    for row in reps:
        if np.any(row < 0) or np.any(row >= n) or len(np.unique(row)) != len(row):
            raise ValueError("Subsamples must contain valid, distinct sample indices")
    perms = np.load(root / "data/scale_permutations.npy", allow_pickle=False)
    if perms.shape != (len(stats["scale_k"]), n) or not np.issubdtype(perms.dtype, np.integer):
        raise ValueError("Scale permutations have the wrong shape or dtype")
    if any(not np.array_equal(np.sort(row), np.arange(n)) for row in perms):
        raise ValueError("Every scale row must permute the full dataset")
    if stats["std_ddof"] != 0:
        raise ValueError("The fixed protocol requires population SD (ddof=0)")
    files = ["data/items.jsonl", "data/subsamples.npy", "data/scale_permutations.npy"]
    return {
        "sample_ids": sample_ids,
        "subsamples": reps,
        "scale_permutations": perms,
        "protocol_digest": frozen["protocol_digest"],
        "input_hashes": {p: common.sha256_file(root / p) for p in files},
    }


def audit_stage(root: Path, stage: str, config: dict, inputs: dict) -> dict:
    files = sorted((root / "activations" / stage).glob("batch_*.npz"))
    if not files:
        raise FileNotFoundError(f"No activation batches for {stage}; no results were fabricated")
    layers = config["inference"]["layers"]
    width = config["inference"]["hidden_size"]
    max_batch = config["inference"]["write_batch_size"]
    batch_records = []
    observed_ids = []
    for path in files:
        side_path = path.with_suffix(".json")
        side = json.loads(side_path.read_text())
        if side.get("protocol_digest") != inputs["protocol_digest"]:
            raise ValueError(f"Protocol mismatch in {side_path}")
        validation = side.get("validation")
        if not isinstance(validation, dict):
            raise ValueError(f"Missing forward validation evidence: {side_path}")
        if validation.get("stage") != stage or validation.get("mode") != "formal":
            raise ValueError(f"Forward validation stage/mode mismatch: {side_path}")
        if validation.get("protocol_digest") != inputs["protocol_digest"]:
            raise ValueError(f"Forward validation protocol mismatch: {side_path}")
        if validation.get("fp32") is not True or validation.get("tf32") is not False:
            raise ValueError(f"Forward validation precision mismatch: {side_path}")
        batch_sha = common.sha256_file(path)
        if side.get("sha256") != batch_sha:
            raise ValueError(f"Activation checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as z:
            ids = z["sample_ids"]
            hidden = z["hidden"]
            if ids.ndim != 1 or ids.dtype.kind != "U":
                raise ValueError(f"sample_ids must be a Unicode vector: {path}")
            expected_shape = (len(ids), layers, width)
            if not 0 < len(ids) <= max_batch or hidden.shape != expected_shape:
                raise ValueError(f"Activation shape mismatch: {path}")
            if hidden.dtype != np.dtype("float32"):
                raise ValueError(f"Activations must be stored in float32: {path}")
            if not np.isfinite(hidden).all():
                raise ValueError(f"Nonfinite activations: {path}")
            if side.get("shape") != list(hidden.shape) or side.get("dtype") != "float32":
                raise ValueError(f"Activation metadata mismatch: {side_path}")
            ids_list = ids.tolist()
            if side.get("sample_ids") != ids_list:
                raise ValueError(f"Sidecar sample order mismatch: {side_path}")
        observed_ids.extend(ids_list)
        batch_records.append({
            "path": str(path.relative_to(root)),
            "sha256": batch_sha,
            "sidecar_sha256": common.sha256_file(side_path),
        })
    if observed_ids != inputs["sample_ids"]:
        raise ValueError(f"{stage}: batches must contain all frozen samples in exact order")
    provenance = {
        "cache_version": CACHE_VERSION,
        "stage": stage,
        "protocol_digest": inputs["protocol_digest"],
        "input_hashes": inputs["input_hashes"],
        "batches": batch_records,
        "statistics": config["statistics"],
        "inference": config["inference"],
        "implementation": {
            name: common.sha256_file(Path(__file__).parent / name)
            for name in ["twonn.py", "analyze.py", "common.py"]
        },
    }
    return {"stage": stage, "files": files, "provenance": provenance, "digest": _digest(provenance)}


def _load_layer(audit: dict, layer: int) -> np.ndarray:
    chunks = []
    for path in audit["files"]:
        with np.load(path, allow_pickle=False) as z:
            chunks.append(z["hidden"][:, layer - 1, :].copy())
    return np.concatenate(chunks, axis=0)


def _cache_path(root: Path, stage: str, layer: int) -> Path:
    return root / "reports/cache" / stage / f"layer_{layer:02d}.json"


def _valid_cache(root: Path, audit: dict, layer: int) -> dict | None:
    path = _cache_path(root, audit["stage"], layer)
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if cached.get("source_digest") != audit["digest"] or cached.get("layer") != layer:
        return None
    if cached.get("payload_sha256") != _digest({
        k: v for k, v in cached.items() if k != "payload_sha256"
    }):
        return None
    return cached


def _summary(estimates: list[dict]) -> dict:
    good = [e["id"] for e in estimates if e["status"] == "ok"]
    complete = len(good) == len(estimates)
    # Do not silently drop failed replicates from the scientific curve.
    return {
        "status": "ok" if complete else "invalid",
        "id_mean": float(np.mean(good)) if complete else None,
        "id_sd": float(np.std(good, ddof=0)) if complete else None,
        "n_estimates": len(estimates),
        "n_valid_estimates": len(good),
        "n_valid_min": min(e["n_valid"] for e in estimates),
        "n_valid_max": max(e["n_valid"] for e in estimates),
        "n_zero_max": max(e["n_zero"] for e in estimates),
        "n_tied_max": max(e["n_tied"] for e in estimates),
    }


def analyze_layer(root: Path, audit: dict, layer: int, config: dict, inputs: dict) -> dict:
    cached = _valid_cache(root, audit, layer)
    if cached is not None:
        print(f"{audit['stage']} layer {layer}: validated cache", flush=True)
        return cached
    print(f"{audit['stage']} layer {layer}: exact float64 Euclidean distances", flush=True)
    distances = euclidean_distances(_load_layer(audit, layer))
    stats = config["statistics"]
    estimates = []
    for rep, indices in enumerate(inputs["subsamples"]):
        est = estimate_distances(distances[np.ix_(indices, indices)], stats["fit_fraction"])
        est.update(replicate=rep, stage=audit["stage"], layer=layer)
        estimates.append(est)
    scales = []
    if layer in stats["scale_layers"]:
        for row, k in enumerate(stats["scale_k"]):
            blocks = block_estimates(distances, inputs["scale_permutations"][row], k, stats["fit_fraction"])
            scales.append({
                "k": k,
                "n_block": len(distances) // k,
                "n_dropped": len(distances) % k,
                "summary": _summary(blocks),
                "blocks": blocks,
            })
    result = {
        "source_digest": audit["digest"],
        "protocol_digest": inputs["protocol_digest"],
        "stage": audit["stage"],
        "layer": layer,
        "replicates": estimates,
        "summary": _summary(estimates),
        "scales": scales,
    }
    result["payload_sha256"] = _digest(result)
    common.atomic_json(_cache_path(root, audit["stage"], layer), result)
    return result


def _observed_shape(rows: list[dict]) -> dict:
    if any(row["status"] != "ok" for row in rows):
        return {"status": "incomplete", "description": "Some layers are invalid; the complete curve cannot be interpreted."}
    y = np.array([r["id_mean"] for r in rows])
    peak = int(np.argmax(y))
    peak_layer = rows[peak]["layer"]
    interior = 0 < peak < len(y) - 1
    strict_rise = bool(np.all(np.diff(y[: peak + 1]) > 0)) if peak > 0 else False
    strict_fall = bool(np.all(np.diff(y[peak:]) < 0)) if peak < len(y) - 1 else False
    if interior and y[peak] > y[0] and y[peak] > y[-1]:
        desc = f"The mean curve peaks at layer {peak_layer}, above both endpoints."
        desc += "Both sides are strictly monotonic." if strict_rise and strict_fall else "Both sides are not strictly monotonic; the interior peak is descriptive."
    else:
        desc = "The mean curve has no interior maximum above both endpoints."
    return {
        "status": "ok", "peak_layer": peak_layer, "first_id": float(y[0]),
        "peak_id": float(y[peak]), "last_id": float(y[-1]),
        "peak_minus_first": float(y[peak] - y[0]), "peak_minus_last": float(y[peak] - y[-1]),
        "interior_peak": bool(interior), "strict_rise": strict_rise, "strict_fall": strict_fall,
        "description": desc,
    }


def _plots(root: Path, rows: list[dict], stages: list[str], config: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#174A7E", "#CB7725", "#33846B", "#A14966"]
    figures = root / "reports/figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 5.7), layout="constrained")
    fig2, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, layout="constrained")
    n_layers = config["inference"]["layers"]
    ticks = list(range(1, n_layers + 1)) if n_layers <= 8 else [1] + list(range(4, n_layers + 1, 4))
    for i, stage in enumerate(stages):
        selected = [r for r in rows if r["stage"] == stage]
        x = [r["layer"] for r in selected]
        y = [r["id_mean"] if r["status"] == "ok" else np.nan for r in selected]
        sd = [r["id_sd"] if r["status"] == "ok" else np.nan for r in selected]
        for axis in [ax, axes.flat[i]]:
            axis.errorbar(x, y, yerr=sd, color=colors[i], marker="o", markersize=3.5,
                          linewidth=1.6, elinewidth=0.8, capsize=1.7, label=STAGE_LABELS.get(stage, stage))
            axis.grid(axis="y", alpha=0.2)
            axis.set_xlabel("Raw decoder block")
            axis.set_ylabel("TwoNN intrinsic dimension")
            axis.set_xticks(ticks)
        axes.flat[i].set_title(STAGE_LABELS.get(stage, stage), loc="left", weight="bold")
    hop = config["data"]["hop"]
    n = config["data"]["sample_count"]
    repeats = config["statistics"]["subsample_repeats"]
    title = f"PrOntoQA {hop}-hop · N={n} · prompt-only · last input token"
    ax.set_title(title + f"\nMean ± population SD across {repeats} overlapping subsamples", loc="left", pad=15, fontsize=12)
    ax.legend(frameon=False)
    fig2.suptitle(title + f"\nMean ± population SD across {repeats} overlapping subsamples", fontsize=13)
    for figure, name in [(fig, "id_overlay"), (fig2, "id_four_panel")]:
        for ext in ["png", "svg"]:
            target = figures / f"{name}.{ext}"
            temporary = figures / f".{name}.{ext}.tmp"
            figure.savefig(temporary, format=ext, dpi=180, facecolor="white")
            os.replace(temporary, target)
        plt.close(figure)


def write_report(root: Path, audits: list[dict], config: dict, inputs: dict) -> dict:
    rows, reps, scales, blocks, provenance = [], [], [], [], {}
    stages = [model["stage"] for model in config["models"]]
    for audit in audits:
        stage = audit["stage"]
        provenance[stage] = audit["provenance"]
        for layer in range(1, config["inference"]["layers"] + 1):
            cached = _valid_cache(root, audit, layer)
            if cached is None:
                raise RuntimeError(f"Missing or stale analysis for {stage} layer {layer}")
            reps.extend(cached["replicates"])
            rows.append({"stage": stage, "layer": layer, **cached["summary"]})
            for scale in cached["scales"]:
                blocks.extend({"stage": stage, "layer": layer, "k": scale["k"],
                               "block": index, "n_block": scale["n_block"],
                               "n_dropped": scale["n_dropped"], **estimate}
                              for index, estimate in enumerate(scale["blocks"]))
                scales.append({"stage": stage, "layer": layer, "k": scale["k"],
                               "n_block": scale["n_block"], "n_dropped": scale["n_dropped"], **scale["summary"]})
    summary_fields = ["status", "id_mean", "id_sd", "n_estimates", "n_valid_estimates", "n_valid_min", "n_valid_max", "n_zero_max", "n_tied_max"]
    _write_csv(root / "reports/id_replicates.csv", reps,
               ["stage", "layer", "replicate", "status", "reason", "id", "n_total", "n_valid", "n_zero", "n_tied", "n_fit", "fit_fraction", "pearson_r", "pearson_p"])
    _write_csv(root / "reports/id_summary.csv", rows, ["stage", "layer"] + summary_fields)
    _write_csv(root / "reports/scale_checks.csv", scales, ["stage", "layer", "k", "n_block", "n_dropped"] + summary_fields)
    _write_csv(root / "reports/scale_blocks.csv", blocks,
               ["stage", "layer", "k", "block", "n_block", "n_dropped", "status", "reason", "id", "n_total", "n_valid", "n_zero", "n_tied", "n_fit", "fit_fraction", "pearson_r", "pearson_p"])
    _plots(root, rows, stages, config)
    shapes = {stage: _observed_shape([r for r in rows if r["stage"] == stage]) for stage in stages}
    common.atomic_json(root / "reports/shape_descriptions.json", shapes)
    source = {"protocol_digest": inputs["protocol_digest"], "stages": provenance}
    common.atomic_json(root / "reports/analysis_provenance.json", source)
    descriptions = "".join(f"<li><strong>{html.escape(STAGE_LABELS.get(s, s))}</strong>: {html.escape(shapes[s]['description'])}</li>" for s in stages)
    invalid_count = sum(r["status"] != "ok" for r in rows)
    diagnostic_rows = "".join(
        f"<tr><td>{html.escape(s)}</td><td>{max(r['n_zero_max'] for r in rows if r['stage']==s)}</td>"
        f"<td>{max(r['n_tied_max'] for r in rows if r['stage']==s)}</td>"
        f"<td>{min(r['n_valid_min'] for r in rows if r['stage']==s)}</td></tr>"
        for s in stages
    )
    scale_rows = "".join(
        f"<tr><td>{html.escape(r['stage'])}</td><td>{r['layer']}</td><td>{r['k']}</td><td>{r['n_block']}</td>"
        f"<td>{'invalid' if r['status'] != 'ok' else format(r['id_mean'], '.4f')}</td>"
        f"<td>{'—' if r['status'] != 'ok' else format(r['id_sd'], '.4f')}</td></tr>"
        for r in scales
    )
    data, stat, inf = config["data"], config["statistics"], config["inference"]
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tulu3 PrOntoQA TwoNN</title><style>body{{max-width:1120px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#172b3a}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}code{{overflow-wrap:anywhere}}a{{color:#174a7e}}</style>
<h1>Tulu3 PrOntoQA representation geometry: TwoNN</h1>
<p>{data['sample_count']} fixed {data['hop']}-hop questions; Base, SFT, DPO and RLVR; prompt only, last nonpadding input token, raw decoder blocks 1–{inf['layers']}.</p>
<figure><img src="figures/id_overlay.svg" alt="Four-stage TwoNN curves"><figcaption>Mean ± population SD across {stat['subsample_repeats']} overlapping subsamples; these are not confidence intervals.</figcaption></figure>
<figure><img src="figures/id_four_panel.svg" alt="TwoNN curves by stage"></figure>
<h2>Observed curves</h2><ul>{descriptions}</ul><p>Peak selection uses unsmoothed means and the earliest maximum. It is descriptive, without a significance test. Invalid layers: {invalid_count}.</p>
<h2>Method</h2><p>The four stages use the same items and saved sampling indices. Each layer uses {stat['subsample_repeats']} subsets of {stat['subsample_size']} items sampled without replacement. Two nearest neighbors are recomputed inside each subset. Raw coordinates are neither standardized nor L2-normalized. Distances are direct float64 Euclidean distances. The independent estimator follows the pinned author's zero-distance and tied-neighbor filtering, empirical CDF, regression through the origin, lowest {stat['fit_fraction']:.0%} of ratios and terminal two-point trimming.</p>
<p>Forward and storage: float32; TF32 disabled. Embeddings and final RMSNorm are excluded. Overlapping subsamples are not independent experiments. No missing estimate is replaced with zero.</p>
<h2>Nearest-neighbor diagnostics</h2><table><tr><th>Stage</th><th>Maximum zero neighbors</th><th>Maximum tied neighbors</th><th>Minimum valid points</th></tr>{diagnostic_rows}</table><p>Zero and tied counts may overlap.</p>
<h2>Sample-size checks</h2><p>Only prespecified layers {stat['scale_layers']} are checked. Each k uses its saved permutation, disjoint equal blocks and a discarded remainder. Block SD is distinct from overlapping-subsample SD. Sample-size drift limits interpretation to the observed scale.</p><details><summary>All k=1…20 results</summary><table><tr><th>Stage</th><th>Layer</th><th>k</th><th>Items/block</th><th>Mean</th><th>Population SD</th></tr>{scale_rows}</table></details>
<h2>Interpretation</h2><p>Network depth is not generation time or reasoning-step count. This experiment does not measure accuracy or performance. Base uses plain input while the three tuned checkpoints use native chat templates; weight, input-format and final-token differences jointly affect stage comparisons. Curves do not establish causal training effects, manifold curvature or an information-compression mechanism. Discrete template-generated text does not automatically satisfy continuous-manifold assumptions. Finite-sample high-dimensional estimates are not proven exact lower bounds.</p>
<h2>Artifacts</h2><p><a href="id_replicates.csv">Replicates</a> · <a href="id_summary.csv">Layer summaries</a> · <a href="scale_checks.csv">Scale summaries</a> · <a href="scale_blocks.csv">Scale blocks</a> · <a href="analysis_provenance.json">Input and implementation identity</a> · <a href="shape_descriptions.json">Curve descriptions</a></p>
<p>Activation protocol: <code>{html.escape(inputs['protocol_digest'])}</code>. <a href="https://arxiv.org/abs/1905.12784v2">Method paper v2</a> · <a href="https://github.com/ansuini/IntrinsicDimDeep/blob/2c5140009eb3efccca1d774dc66cb043c04b26d8/IDNN/intrinsic_dimension.py">Pinned author estimator</a>.</p></html>"""
    _atomic_text(root / "reports/report.html", document)
    outputs = sorted((root / "reports").glob("*.csv")) + [root / "reports/report.html"]
    outputs += sorted((root / "reports/figures").glob("*"))
    outputs += [root / "reports/analysis_provenance.json", root / "reports/shape_descriptions.json"]
    manifest = {
        "protocol_digest": inputs["protocol_digest"],
        "source_digests": {audit["stage"]: audit["digest"] for audit in audits},
        "invalid_layer_count": invalid_count,
        "outputs": {str(p.relative_to(root)): common.sha256_file(p) for p in outputs},
    }
    common.atomic_json(root / "reports/analysis_manifest.json", manifest)
    return manifest


def run(root: Path, stage: str | None = None, layer: int | None = None) -> dict | None:
    root = Path(root).resolve()
    config = common.load_config(root)
    inputs = load_inputs(root, config)
    stages = [m["stage"] for m in config["models"]]
    if len(stages) != 4 or len(set(stages)) != 4:
        raise ValueError("This report requires exactly four distinct frozen stages")
    if stage is not None and stage not in stages:
        raise ValueError(f"Unknown stage: {stage}")
    n_layers = config["inference"]["layers"]
    if layer is not None and not 1 <= layer <= n_layers:
        raise ValueError(f"layer must be between 1 and {n_layers}")
    targets = stages if stage is None else [stage]
    audits = {s: audit_stage(root, s, config, inputs) for s in targets}
    for s in targets:
        for l in ([layer] if layer is not None else range(1, n_layers + 1)):
            analyze_layer(root, audits[s], l, config, inputs)
    # Partial runs only prepare resumable caches, never a misleading complete report.
    for s in stages:
        if s not in audits:
            try:
                audits[s] = audit_stage(root, s, config, inputs)
            except FileNotFoundError:
                print("Requested caches saved; full report awaits all four stages.", flush=True)
                return None
        for l in range(1, n_layers + 1):
            if _valid_cache(root, audits[s], l) is None:
                print("Requested caches saved; full report awaits all stage/layer analyses.", flush=True)
                return None
    result = write_report(root, [audits[s] for s in stages], config, inputs)
    print(f"Completed report: {root / 'reports/report.html'}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_workspace_arguments(parser, analysis=True)
    parser.add_argument("--stage", help="Calculate only this stage; other existing valid caches may be reused")
    parser.add_argument("--layer", type=int, help="Calculate only this one-based raw decoder block")
    args = parser.parse_args()
    try:
        args.root = common.workspace(args, analysis=True)
        result = run(args.root, args.stage, args.layer)
        if result is not None:
            from .verify import final_check
            final_check(args.root)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f"Analysis stopped: {exc}\n")


if __name__ == "__main__":
    main()
