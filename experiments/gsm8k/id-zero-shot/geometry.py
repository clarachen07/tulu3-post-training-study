"""Estimate and audit layerwise ID on the exact archived question panels."""

import argparse
import fcntl
import io
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from math import ceil
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from archive import TEST_N, PUBLISHED_STATS, archive_inputs, archive_outputs, published_statistics, require
from gsm8k_id.analyze import compute_cell, export_csv
from gsm8k_id.common import ANCHORS, STAGES, atomic_bytes, digest, read, sample_schedule, sha, utc, write
from gsm8k_id.pca_core import fit_pca
from gsm8k_id.twonn import euclidean_distances, estimate_distances
from gsm8k_id.verify import independent_twonn
from replay import capture_path, load_capture, pack, prepare_replays, unpack

METHODS = ("twonn", "correlation", "covariance")
SCALE_LAYERS = (1, 16, 32)
STATISTICS = {"fraction": .9, "repeats": 20, "seed": 20261007, "scale_seed": 20261008}


def read_evidence(root, manifest):
    archive = Path(manifest["archive"])
    cfg, prompts, _, _ = archive_inputs(archive)
    archive_ok = manifest["trajectory_issue"] is None
    outputs = archive_outputs(archive, cfg, prompts) if archive_ok else None
    replays = prepare_replays(prompts, outputs, cfg)
    resume = read(root / "manifests/resume.json")
    require(resume["status"] == "passed", "Completed-capture restart check is missing")
    verified_paths = set()
    evidence = {stage: {} for stage in STAGES}
    for phase, group in replays.items():
        for stage in STAGES:
            for ident, (prompt, continuation, response) in group[stage].items():
                path = capture_path(root, phase, stage, ident)
                side_path = path.with_suffix(".json")
                side = read(side_path)
                expected = [len(prompt) - 1] + [(len(prompt) + ceil(a * len(continuation) / 100) - 1)
                                                  if continuation else None for a in ANCHORS[1:]]
                sequence = prompt + continuation
                tokens = [sequence[i] if i is not None else None for i in expected]
                anchors = [a for a, i in enumerate(expected) if i is not None]
                require(side["phase"] == phase and side["stage"] == stage and side["id"] == ident
                        and side["positions"] == expected and side["token_ids"] == tokens and side["anchors"] == anchors
                        and side["prompt_token_hash"] == digest(prompt) and side["prompt_tokens"] == len(prompt),
                        f"Independent input/readout validation failed: {phase}/{stage}/{ident}")
                if response:
                    require(side["effective_tokens"] == len(continuation) and
                            side["archived_record_hash"] == response["record_hash"] and
                            all(side[k] == response[k] for k in ("correct", "truncated", "finish_reason", "output_tokens")),
                            f"Independent response audit failed: {phase}/{stage}/{ident}")
                else:
                    require(side["archived_record_hash"] is None and side["effective_tokens"] is None,
                            f"Unexpected trajectory evidence: {phase}/{stage}/{ident}")
                if phase == "pilot":
                    require(side["pilot_repeat_equal"] and all(e is not None for e in side["norm_errors"]),
                            f"Missing pilot verification: {stage}/{ident}")
                else:
                    evidence[stage][ident] = side
                for local in (path, side_path):
                    rel = str(local.relative_to(root))
                    require(sha(local) == resume["files"][rel], f"Changed committed file: {local}")
                    verified_paths.add(rel)
    require(verified_paths == resume["files"].keys(), "Unexpected capture files in resume receipt")
    return evidence


def choose_panels(evidence, archive_ok):
    ids = sorted(evidence["base"])
    require(len(ids) == TEST_N and all(sorted(evidence[s]) == ids for s in STAGES), "Wrong formal coverage")
    panels = {"all_input": (ids, (0,))}
    if archive_ok:
        trajectory = [ident for ident in ids if all(evidence[s][ident]["effective_tokens"] > 0 for s in STAGES)]
        complete = [ident for ident in trajectory if all(not evidence[s][ident]["truncated"] for s in STAGES)]
        panels.update(all_trajectory=(trajectory, ANCHORS[1:]), common_complete=(complete, ANCHORS))
    return panels


def save_indices(path, data):
    with io.BytesIO() as buffer:
        np.save(buffer, data, allow_pickle=False)
        atomic_bytes(path, buffer.getvalue())


def load_stage(root, identity, stage, records):
    # Store the formal panel in the archived bit representation until each
    # numerical worker needs one matrix. Reuse the single prefill capture.
    anchors = max(len(row["anchors"]) for row in records.values())
    store = np.empty((len(records), anchors, 32, 4096), dtype="<u2")
    for i, (ident, side) in enumerate(records.items()):
        hidden = load_capture(capture_path(root, "formal", stage, ident), side, identity)
        store[i, side["anchors"]] = pack(hidden)
    return store


def cache_cell(root, identity, store, stage, task):
    panel, ids, indices, schedule, anchor, layer = task
    key = digest({"identity": identity, "stage": stage, "ids": ids, "anchor": anchor, "layer": layer})
    path = root / f"cache/{key}.json"
    spectrum_path = path.with_suffix(".npz")
    if path.exists():
        result = read(path)
        require(result["key"] == key and result["records_sha256"] == digest(result["records"])
                and result["spectra_sha256"] == sha(spectrum_path), f"Changed ID cache: {path}")
    else:
        matrix = unpack(store[indices, ANCHORS.index(anchor), layer - 1]).astype(np.float64)
        records, spectra = compute_cell(matrix, schedule)
        # Feature-space entries beyond the sample count are structural zeros.
        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **{name: values[:len(indices)] for name, values in spectra.items()})
            atomic_bytes(spectrum_path, buffer.getvalue())
        result = {"key": key, "records": records, "records_sha256": digest(records),
                  "spectra_sha256": sha(spectrum_path), "spectrum_width": 4096}
        write(path, result)
    cell = {"panel": panel, "stage": stage, "anchor": anchor, "layer": layer}
    receipt = {**cell, "path": str(path.relative_to(root)), "sha256": sha(path), "spectra_sha256": result["spectra_sha256"]}
    estimates = [{**cell, **row} for row in result["records"]]
    summary = []
    for method in METHODS:
        values = [row["value"] for row in estimates if row["method"] == method and row["replicate"] >= 0
                  and row["status"] == "ok"]
        full = next(row["value"] for row in estimates if row["method"] == method and row["replicate"] < 0)
        summary.append({**cell, "method": method, "n": len(ids), "subsample_n": schedule.shape[1],
                        "valid_repeats": len(values), "full": full,
                        "mean": float(np.mean(values)) if values else None,
                        "sd": float(np.std(values)) if values else None})
    return summary, estimates, receipt


def scale_cell(store, stage, layer, permutations):
    matrix = unpack(store[:, 0, layer - 1]).astype(np.float64)
    distances = euclidean_distances(matrix)
    blocks, summary = [], []
    n = len(matrix)
    for k, permutation in enumerate(permutations, 1):
        block_n = n // k
        for block, ix in enumerate(permutation[:block_n * k].reshape(k, block_n)):
            estimate = estimate_distances(distances[np.ix_(ix, ix)])
            common = {"stage": stage, "layer": layer, "k": k, "block": block, "n": block_n, "dropped": n % k}
            blocks.append({**common, "method": "twonn", "value": estimate["id"], "status": estimate["status"],
                           "n_valid": estimate["n_valid"], "n_zero": estimate["n_zero"], "n_tied": estimate["n_tied"]})
            for method in METHODS[1:]:
                pca = fit_pca(matrix[ix], method)
                blocks.append({**common, "method": method, "value": pca["pc_id90"], "status": "ok"})
        for method in METHODS:
            values = [r["value"] for r in blocks[-3 * k:] if r["method"] == method and r["status"] == "ok"]
            summary.append({"stage": stage, "layer": layer, "k": k, "n": block_n, "method": method,
                            "valid_blocks": len(values), "mean": float(np.mean(values)) if values else None,
                            "sd": float(np.std(values)) if values else None})
    return blocks, summary


def independent_cell(matrix, selected):
    results = {row["method"]: row["full"] for row in selected}
    two_nn = independent_twonn(matrix)
    if two_nn is None:
        require(results["twonn"] is None, "Independent TwoNN degeneracy differs")
    else:
        np.testing.assert_allclose(two_nn, results["twonn"], rtol=1e-10, atol=1e-10)
    centered = matrix - matrix.mean(axis=0)
    const = np.all(matrix == matrix[0], axis=0)
    centered[:, const] = 0
    for method in ("covariance", "correlation"):
        if method == "correlation":
            centered[:, ~const] /= np.std(matrix[:, ~const], axis=0)
        squared = np.linalg.svd(centered, compute_uv=False, full_matrices=False) ** 2
        pc_id = int(np.searchsorted(np.cumsum(squared / squared.sum()), .9)) + 1
        require(pc_id == results[method], f"Independent {method} PC-ID90 differs: {pc_id} != {results[method]}")
    return results


def question_table(evidence, panels):
    members = {panel: set(ids) for panel, (ids, _) in panels.items()}
    table = []
    for stage, records in evidence.items():
        for ident, side in records.items():
            row = {key: side[key] for key in ("id", "stage", "prompt_tokens", "prompt_token_hash",
                   "archived_record_hash", "correct", "truncated", "finish_reason", "output_tokens", "effective_tokens")}
            for panel, ids in members.items():
                row[panel] = ident in ids
            for anchor, index, token in zip(ANCHORS, side["positions"], side["token_ids"]):
                row[f"position_{anchor}"] = index
                row[f"token_id_{anchor}"] = token
            table.append(row)
    return table


def analyze(root, workers):
    manifest = read(root / "manifests/replay.json")
    require(manifest["identity"] == digest({k: v for k, v in manifest.items() if k != "identity"}),
            "Invalid replay identity")
    evidence = read_evidence(root, manifest)
    panels = choose_panels(evidence, manifest["trajectory_issue"] is None)
    identity = digest({"capture_identity": manifest["identity"], "statistics": STATISTICS,
                       "code": sha(Path(__file__)), "methods": [sha(Path(compute_cell.__code__.co_filename)),
                       sha(Path(fit_pca.__code__.co_filename)), sha(Path(estimate_distances.__code__.co_filename))]})
    report_root = root / "reports"
    n = len(panels["all_input"][0])
    rng = np.random.RandomState(STATISTICS["scale_seed"])
    permutations = np.stack([rng.permutation(n) for _ in range(20)])
    save_indices(report_root / "schedules/scale_permutations.npy", permutations)
    index = {ident: i for i, ident in enumerate(panels["all_input"][0])}
    tasks = []
    panel_data = {}
    for panel, (ids, anchors) in panels.items():
        indices = np.array([index[ident] for ident in ids], dtype=np.intp)
        schedule = sample_schedule(len(ids), STATISTICS)
        panel_data[panel] = {"ids": ids, "anchors": anchors, "subsample_n": None if schedule is None else schedule.shape[1]}
        if schedule is None:
            continue
        save_indices(report_root / f"schedules/{panel}.npy", schedule)
        tasks.extend((panel, ids, indices, schedule, anchor, layer)
                     for anchor in anchors for layer in range(1, 33))
    write(report_root / "panels.json", panel_data)
    summary, estimates, receipts, checks, scale_blocks, scale_summary = [], [], [], [], [], []
    audit_cells = {"base": (0, 1), "sft": (0, 16), "dpo": (50, 16), "rlvr": (100, 32)}
    for stage in STAGES:
        print(f"loading verified activations {stage}", flush=True)
        store = load_stage(root, manifest["identity"], stage, evidence[stage])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            calculate = partial(cache_cell, root, identity, store, stage)
            for count, (summaries, points, receipt) in enumerate(pool.map(calculate, tasks), 1):
                summary.extend(summaries)
                estimates.extend(points)
                receipts.append(receipt)
                if count % 16 == 0 or count == len(tasks):
                    print(f"geometry {stage}: {count}/{len(tasks)}", flush=True)
            scale = pool.map(partial(scale_cell, store, stage, permutations=permutations), SCALE_LAYERS)
            for blocks, summaries in scale:
                scale_blocks.extend(blocks)
                scale_summary.extend(summaries)
        anchor, layer = audit_cells[stage]
        panel = "all_trajectory" if anchor and panel_data.get("all_trajectory", {}).get("subsample_n") else "all_input"
        anchor = anchor if panel == "all_trajectory" else 0
        ids = panels[panel][0]
        matrix = unpack(store[[index[i] for i in ids], ANCHORS.index(anchor), layer - 1]).astype(np.float64)
        selected = [row for row in summary if (row["stage"], row["panel"], row["anchor"], row["layer"]) ==
                    (stage, panel, anchor, layer)]
        checks.append({"stage": stage, "panel": panel, "anchor": anchor, "layer": layer,
                       "estimates": independent_cell(matrix, selected)})
        del store, calculate, matrix
        print(f"independent TwoNN and SVD audit passed: {stage}/{panel}/L{layer}", flush=True)
    for name, records in (("summary", summary), ("estimates", estimates), ("scale_blocks", scale_blocks),
                          ("scale_summary", scale_summary), ("per_question", question_table(evidence, panels))):
        export_csv(report_root / (name + ".csv"), records)
    published = published_statistics(Path(manifest["archive"]))
    behavior = [row for row in published["summary"] if row["split"] == "test"]
    export_csv(report_root / "behavior.csv", behavior)
    write(report_root / "analysis_manifest.json", {"identity": identity, "replay_identity": manifest["identity"],
          "statistics": STATISTICS, "cells": receipts, "analysis_code_sha256": sha(Path(__file__))})
    artifacts = {str(p.relative_to(report_root)): sha(p) for p in sorted(report_root.rglob("*")) if p.is_file()
                 and p.name != "independent_audit.json"}
    write(report_root / "independent_audit.json", {"status": "passed" if manifest["trajectory_issue"] is None else "prefill_only",
          "replay_identity": manifest["identity"], "analysis_identity": identity, "archive_statistics_sha256": PUBLISHED_STATS,
          "checked_records": sum(map(len, evidence.values())), "trajectory_issue": manifest["trajectory_issue"],
          "resume_sha256": sha(root / "manifests/resume.json"), "independent_numerical_cells": checks,
          "artifacts": artifacts, "checked_utc": utc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / ".geometry.lock").open("a+") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / "status/workflow.json", {"phase": "geometry", "status": "running", "utc": utc()})
        try:
            analyze(root, args.workers)
        except BaseException as exc:
            write(root / "status/workflow.json", {"phase": "geometry", "status": "failed", "error": repr(exc), "utc": utc()})
            raise
        write(root / "status/workflow.json", {"phase": "geometry", "status": "complete", "utc": utc()})


if __name__ == "__main__":
    main()
