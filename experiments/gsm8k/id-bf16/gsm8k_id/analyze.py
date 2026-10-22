"""Full spectra and paired subsamples, bound to exact raw activation receipts."""
from __future__ import annotations

import argparse
import csv
import io
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .common import ANCHORS, ROOT, STAGES, atomic_bytes, digest, read, rows, sample_schedule, sha, utc, validate, write
from .pca_core import fit_pca
from .twonn import euclidean_distances, estimate_distances
from .run import committed, generation


def export_csv(path, records):
    if not records:
        return
    fields = list(dict.fromkeys(k for row in records for k in row))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    w.writerows(records)
    atomic_bytes(path, buf.getvalue().encode())


def audit_inputs(root):
    frozen = validate(root)
    records, evidence = {}, []
    for stage in STAGES:
        panel = rows(root / f"data/tokenized/{stage}/formal.jsonl")
        records[stage] = {}
        expected_ids = {r["id"] for r in panel}
        observed = {p.stem for p in (root / f"outputs/formal/{stage}").glob("*.json")}
        activations = {p.stem for p in (root / f"activations/formal/{stage}").glob("*.json")}
        if observed != expected_ids or activations != expected_ids:
            raise ValueError(f"Incomplete/unexpected formal panel: {stage}")
        for source in panel:
            r = generation(root, "formal", stage, source, frozen["identity"])
            if not committed(root, "formal", stage, source, frozen["identity"], r):
                raise ValueError(f"Missing activation {stage}/{source['id']}")
            records[stage][source["id"]] = r
            stem = source["id"]
            for rel in (f"outputs/formal/{stage}/{stem}.json", f"activations/formal/{stage}/{stem}.json"):
                evidence.append({"path": rel, "sha256": sha(root / rel)})
    ids = sorted(records[STAGES[0]])
    assert all(sorted(records[s]) == ids for s in STAGES)
    nonempty = [i for i in ids if all(records[s][i]["effective_output_tokens"] > 0 for s in STAGES)]
    complete = [i for i in nonempty if all(not records[s][i]["truncated"] for s in STAGES)]
    manifest = {"frozen_identity": frozen["identity"], "source_receipts": evidence,
                "input_ids": ids, "trajectory_ids": nonempty, "complete_ids": complete}
    manifest["identity"] = digest(manifest)
    write(root / "reports/analysis_sources.json", manifest)
    return records, manifest


def compute_cell(x, schedule):
    records, spectra = [], {}
    distances = euclidean_distances(x)
    selections = [(-1, np.arange(len(x)))] + list(enumerate(schedule))
    for rep, indices in selections:
        twonn = estimate_distances(distances[np.ix_(indices, indices)])
        records.append({"method": "twonn", "replicate": rep, "value": twonn["id"], "n": len(indices),
                        "status": twonn["status"], "n_valid": twonn["n_valid"], "n_zero": twonn["n_zero"],
                        "n_tied": twonn["n_tied"], "n_fit": twonn["n_fit"], "pearson_r": twonn["pearson_r"]})
        for method in ("correlation", "covariance"):
            result = fit_pca(x[indices], method)
            spectra[f"{method}_{rep}"] = result["eigenvalues"]
            records.append({"method": method, "replicate": rep, "value": result["pc_id90"], "n": len(indices),
                            "status": "ok", "trace_relative_error": result["trace_relative_error"],
                            "n_constant_features": result["n_constant_features"], "numerical_rank": result["numerical_rank"]})
    return records, spectra


def run(root):
    root = Path(root)
    cfg = read(root / "experiment.json")
    records, source = audit_inputs(root)
    summary, estimates, cache_receipts, panel_records = [], [], [], []
    # One schedule per exact ordered question panel, shared across all comparisons.
    panels = {"all_input": source["input_ids"], "all_trajectory": source["trajectory_ids"], "common_complete": source["complete_ids"]}
    for panel, ids in panels.items():
        schedule = sample_schedule(len(ids), cfg["statistics"])
        panel_records.append({"panel": panel, "n": len(ids), "subsample_size": None if schedule is None else schedule.shape[1],
                              "status": "insufficient_samples" if schedule is None else "ok"})
        if schedule is not None:
            buf = io.BytesIO()
            np.save(buf, schedule, allow_pickle=False)
            atomic_bytes(root / f"reports/schedules/{panel}.npy", buf.getvalue())
    for stage in STAGES:
        # ~3.22 GiB/stage; selected positions only, never all-token activations.
        store = {}
        for ident in source["input_ids"]:
            with np.load(root / f"activations/formal/{stage}/{ident}.npz", allow_pickle=False) as z:
                store[ident] = {int(a): x.copy() for a, x in zip(z["anchors"], z["hidden"])}
        for panel, ids in panels.items():
            schedule = sample_schedule(len(ids), cfg["statistics"])
            if schedule is None:
                continue
            anchors = (0,) if panel == "all_input" else range(5)
            for a in anchors:
                x = np.stack([store[i][a] for i in ids])
                for layer in range(1, 33):
                    # Identical point panels share numerical caches, including completed-case aliases.
                    key = digest({"source_identity": source["identity"], "ids": ids, "stage": stage, "anchor": a,
                                  "layer": layer, "statistics": cfg["statistics"]})
                    path = root / f"cache/{key}.json"
                    spectrum_path = path.with_suffix(".npz")
                    if path.exists():
                        cached = read(path)
                        if cached["key"] != key or digest(cached["records"]) != cached["records_hash"] or sha(spectrum_path) != cached["spectrum_sha256"]:
                            raise ValueError(f"Changed geometry cache: {path}")
                    else:
                        started = time.monotonic()
                        values, spectra = compute_cell(x[:, layer - 1, :], schedule)
                        buf = io.BytesIO()
                        np.savez_compressed(buf, **spectra)
                        atomic_bytes(spectrum_path, buf.getvalue())
                        cached = {"key": key, "records": values, "records_hash": digest(values), "source_identity": source["identity"],
                                  "spectrum_sha256": sha(spectrum_path), "seconds": time.monotonic() - started}
                        write(path, cached)
                    cache_receipts.append({"panel": panel, "stage": stage, "anchor": ANCHORS[a], "layer": layer,
                                           "path": str(path.relative_to(root)), "sha256": sha(path), "spectrum_sha256": cached["spectrum_sha256"]})
                    context = {"panel": panel, "stage": stage, "anchor": ANCHORS[a], "layer": layer}
                    estimates.extend({**context, **r} for r in cached["records"])
                    for method in ("twonn", "correlation", "covariance"):
                        selected = [r for r in cached["records"] if r["method"] == method]
                        values = [r["value"] for r in selected if r["replicate"] >= 0 and r["status"] == "ok"]
                        full = next(r["value"] for r in selected if r["replicate"] == -1)
                        summary.append({**context, "method": method, "n": len(ids), "subsample_n": schedule.shape[1],
                                        "valid_repeats": len(values), "full": full,
                                        "mean": float(np.mean(values)) if values else None,
                                        "sd": float(np.std(values, ddof=0)) if values else None})
                    print(f"geometry {panel}/{stage}/p{ANCHORS[a]:03d}/L{layer:02d}", flush=True)
                del x
        del store
    per_question, behavior = [], []
    for stage in STAGES:
        rs = list(records[stage].values())
        behavior.append({"stage": stage, "n": len(rs), "correct": sum(r["correct"] for r in rs),
                         "accuracy": float(np.mean([r["correct"] for r in rs])), "truncated": sum(r["truncated"] for r in rs),
                         "empty_output": sum(r["effective_output_tokens"] == 0 for r in rs),
                         "mean_output_tokens": float(np.mean([r["output_tokens"] for r in rs])),
                         "median_output_tokens": float(np.median([r["output_tokens"] for r in rs]))})
        for r in rs:
            side = read(root / f"activations/formal/{stage}/{r['id']}.json")
            per_question.append({k: r[k] for k in ("id", "stage", "gold", "correct", "extracted_answer", "finish_reason", "truncated", "output_tokens", "effective_output_tokens")})
            for a, pos in zip(ANCHORS, side["positions"]):
                per_question[-1][f"position_{a}"] = pos
                per_question[-1][f"token_id_{a}"] = None if pos is None else (r["prompt_token_ids"] + r["effective_token_ids"])[pos]
    for name, values in (("summary", summary), ("estimates", estimates), ("behavior", behavior), ("per_question", per_question)):
        export_csv(root / f"reports/{name}.csv", values)
    write(root / "reports/statistics.json", {"source_identity": source["identity"], "panels": panel_records, "behavior": behavior, "summary": summary})
    write(root / "reports/analysis_manifest.json", {"source_identity": source["identity"], "caches": cache_receipts,
          "reports": {f"reports/{name}": sha(root / f"reports/{name}") for name in ("summary.csv", "estimates.csv", "behavior.csv", "per_question.csv", "statistics.json")}, "created_utc": utc()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    with threadpool_limits(limits=4):
        run(p.parse_args().root)
